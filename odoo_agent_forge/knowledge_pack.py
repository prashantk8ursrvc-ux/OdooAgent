"""
Schema Knowledge Pack
=====================

Teaches the model the facts it cannot look up at run time.

Why this is separate from the behavioural families
--------------------------------------------------
The agent's MCP interface exposes six generic primitives —
``odoo_execute_method(model, method, res_ids, kwargs)`` and friends. Nothing in
that schema says which of Odoo's 35,482 methods are valid on which model, what
state each expects, or which fields a ``domain`` may reference. Unlike a
per-tool function-calling setup, there is no run-time source for any of it, so
it has to live in the weights.

That makes this a *recall* problem, not a *policy* problem, and the two want
opposite things from a dataset:

  * policy learning wants breadth — many distinct situations, little repetition
  * recall wants the same fact seen from several angles, repeatedly

Hence a separate pack, deliberately repetitive per fact, and weighted so
capacity goes where it is used.

What went wrong with the first attempt
--------------------------------------
``odoo_schema_knowledge_base.jsonl`` holds 4,531 rows: exactly one
``schema_reference`` and one ``model_lookup`` for each of 2,266 models. Even
weighting means ``decimal.precision`` — which no agent will ever touch — got the
same capacity as ``sale.order``, and only **3%** of the file concerns a model in
the agent surface. Coverage was total and usefulness was near zero.

This module weights by tier instead: heavy on the 20 curated models an agent
drives daily, light on the 51 discovered ones, nothing on the other 2,195.

Fact types
----------
``method_inventory``   which operations exist on a model, and what each does
``method_selection``   intent -> the correct method, and why not the near-miss
``field_reference``    the fields a domain or a create call may use
``relation``           how two models join, and through which field
``state_machine``      the lifecycle, and which method moves between states
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from odoo_agent_forge.agent_surface import MethodSpec, ModelSpec

logger = logging.getLogger(__name__)

#: How many knowledge samples each tier gets per model. Tier A models are what
#: an agent touches every day, so they get the capacity.
TIER_A_SAMPLES = 34
TIER_B_SAMPLES = 6

SYSTEM_PROMPT = (
    "You are an Odoo 19 operations agent. You know the Odoo 19 data model — its "
    "models, fields, relations, business methods, and record lifecycles — and you "
    "answer questions about it precisely, from memory, without needing to look "
    "anything up."
    # These live in the system role, not the user turn. Appended to the user
    # message they read as part of the task, and the teacher rehearsed them in
    # its reasoning ("Use exact names. Never invent...") — which the gate then
    # correctly rejected as instruction_echo. Same lesson as the house rules in
    # prompts.py: standing instructions belong in the system prompt.
    "\n\n"
    "How you answer, always:\n"
    "Use the exact technical names, and never invent a method, field, or model. "
    "If a question implies something you were not given, say what you do know and "
    "stop rather than filling the gap. When the question is about invoking "
    "something, give the concrete call shape. Explain the business meaning in one "
    "line — what the operation is for, not merely that it exists. Answer directly: "
    "open with the answer itself and close when it is given. A short paragraph, or a "
    "tight list when there genuinely are several items. This is Odoo 19."
)


@dataclass(frozen=True)
class Fact:
    """One grounded schema fact, ready to be phrased as a Q&A pair."""

    kind: str
    model: str
    payload: Dict[str, Any]

    def key(self) -> str:
        return f"{self.kind}|{self.model}|{self.payload.get('anchor', '')}"


# ──────────────────────────────────────────────────────────────────────────────
# Fact extraction — everything here comes from the knowledge graph
# ──────────────────────────────────────────────────────────────────────────────

def load_relations(db_path: str, models: Iterable[str]) -> Dict[str, List[Dict[str, str]]]:
    """Relations between models, from the extracted edge table.

    ``INHERITS`` self-edges are dropped: 5,647 of the 13,012 rows are a model
    inheriting itself via ``_inherit``, which teaches nothing.
    """
    wanted = set(models)
    out: Dict[str, List[Dict[str, str]]] = {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT source_model, target_model, field_name, rel_type "
            "FROM relationship_edges "
            "WHERE rel_type IN ('Many2one', 'One2many', 'Many2many')"
        ).fetchall()
    for source, target, field, rel in rows:
        if source not in wanted or not field or source == target:
            continue
        out.setdefault(source, []).append(
            {"target": target, "field": field, "type": rel})
    return out


def build_facts(
    surface: Dict[str, ModelSpec],
    core: Dict[str, ModelSpec],
    kg,
    db_path: str,
    rng: random.Random,
) -> List[Fact]:
    """Turns the verified surface into a weighted list of teachable facts."""
    relations = load_relations(db_path, surface)
    facts: List[Fact] = []

    for model, spec in surface.items():
        is_core = model in core
        budget = TIER_A_SAMPLES if is_core else TIER_B_SAMPLES

        try:
            fields = kg.get_model_fields(model)
        except Exception:                                             # pragma: no cover
            fields = []
        required = [f for f in fields if f.get("required")]
        relational = [f for f in fields if f.get("comodel_name")]

        pool: List[Fact] = []

        # -- what can I do with this model? ------------------------------------
        pool.append(Fact("method_inventory", model, {
            "anchor": "inventory",
            "label": spec.label,
            "methods": [
                {"name": m.name, "intent": m.intent,
                 "from": m.from_state, "to": m.to_state,
                 "wizard": m.returns_action}
                for m in spec.methods
            ],
        }))

        # -- which method for this intent? -------------------------------------
        for meth in spec.methods:
            pool.append(Fact("method_selection", model, {
                "anchor": meth.name,
                "label": spec.label,
                "method": meth.name,
                "intent": meth.intent,
                "from": meth.from_state,
                "to": meth.to_state,
                "wizard": meth.returns_action,
                "kwargs": meth.kwargs,
                "siblings": [m.name for m in spec.methods if m.name != meth.name][:6],
            }))

        # -- which fields may I filter or set? ---------------------------------
        if fields:
            pool.append(Fact("field_reference", model, {
                "anchor": "fields",
                "label": spec.label,
                "searchable": list(spec.search_fields),
                "required": [{"name": f["name"], "type": f["field_type"],
                              "comodel": f.get("comodel_name")} for f in required[:8]],
                "create": list(spec.create_fields),
            }))

        # -- how does it join to other models? ---------------------------------
        # A join is only worth teaching if the agent might traverse it, so
        # relations between two surface models come first. Odoo attaches a long
        # tail of niche links to the core documents (sale.order -> event.booth
        # among them); taking them in table order buried the useful ones.
        rels = relations.get(model, [])
        rels.sort(key=lambda r: (
            r["target"] not in surface,                       # surface targets first
            r["field"] not in ("partner_id", "product_id", "company_id",
                               "user_id", "order_id", "move_id", "picking_id",
                               "invoice_line_ids", "order_line", "move_ids"),
            len(r["field"]),
        ))
        for rel in rels[:6]:
            if rel["target"] not in surface and not is_core:
                continue
            pool.append(Fact("relation", model, {
                "anchor": rel["field"],
                "label": spec.label,
                "field": rel["field"],
                "target": rel["target"],
                "target_label": surface[rel["target"]].label
                if rel["target"] in surface else rel["target"],
                "type": rel["type"],
            }))

        # -- what is the lifecycle? --------------------------------------------
        transitions = [(m.from_state, m.to_state, m.name) for m in spec.methods
                       if m.from_state and m.to_state]
        if transitions:
            pool.append(Fact("state_machine", model, {
                "anchor": "lifecycle",
                "label": spec.label,
                "transitions": transitions,
            }))

        rng.shuffle(pool)
        # Cycle rather than truncate: a model with few facts should still fill
        # its budget, because repetition is the point for recall.
        if pool:
            facts.extend(pool[i % len(pool)] for i in range(budget))

    return facts


# ──────────────────────────────────────────────────────────────────────────────
# Question phrasing
# ──────────────────────────────────────────────────────────────────────────────

#: Several ways to ask for the same fact. Recall improves when the question
#: varies while the answer stays fixed, so these are cycled per repetition.
_QUESTION_FORMS: Dict[str, Tuple[str, ...]] = {
    "method_inventory": (
        "What can I actually do to a {label} in Odoo 19 — which methods can I call?",
        "List the business operations available on {model}.",
        "I'm wiring up an agent against {model}. What methods should it know about?",
        "Which RPC-callable actions does a {label} support?",
    ),
    "method_selection": (
        "Which method do I call to {intent}?",
        "How do I {intent} on {model} over RPC?",
        "What's the right method on {model} when I need to {intent}?",
        "I want to {intent}. Is there a method for that, or do I write the field?",
    ),
    "field_reference": (
        "What fields can I filter on when searching {model}?",
        "Which fields are required to create a {label} in Odoo 19?",
        "I need to build a domain against {model} — what are the field names?",
        "What does a minimal create call for {model} need?",
    ),
    "relation": (
        "How does {model} link to {target}?",
        "Which field joins a {label} to its {target_label}?",
        "If I have a {label}, how do I get to the related {target_label}?",
        "What's the relation between {model} and {target}?",
    ),
    "state_machine": (
        "What's the lifecycle of a {label} in Odoo 19?",
        "Which states does {model} move through, and what triggers each transition?",
        "Walk me through the states on {model}.",
        "How does a {label} get from draft to done?",
    ),
}


def question_for(fact: Fact, variant: int) -> str:
    forms = _QUESTION_FORMS[fact.kind]
    template = forms[variant % len(forms)]
    p = fact.payload
    return template.format(
        model=fact.model,
        label=p.get("label", fact.model),
        intent=p.get("intent", ""),
        target=p.get("target", ""),
        target_label=p.get("target_label", ""),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Grounding block handed to the teacher
# ──────────────────────────────────────────────────────────────────────────────

def grounding_for(fact: Fact) -> str:
    """The verified facts the answer must be built from, and nothing else."""
    p = fact.payload
    lines = [f"Model: {fact.model}  (a {p.get('label', fact.model)})"]

    if fact.kind == "method_inventory":
        lines.append("Verified callable methods:")
        for m in p["methods"]:
            bits = [f"  {m['name']}()"]
            if m["from"] and m["to"]:
                bits.append(f"[{m['from']} -> {m['to']}]")
            elif m["wizard"]:
                bits.append("[returns a wizard action]")
            bits.append(f"— {m['intent']}")
            lines.append(" ".join(bits))

    elif fact.kind == "method_selection":
        lines.append(f"Correct method: {p['method']}()")
        lines.append(f"Purpose: {p['intent']}")
        if p["from"] and p["to"]:
            lines.append(f"Moves the record from '{p['from']}' to '{p['to']}'.")
        if p["wizard"]:
            lines.append("Returns an ir.actions.act_window (a wizard), "
                         "it does not mutate the record directly.")
        if p["kwargs"]:
            lines.append(f"Takes kwargs: {json.dumps(p['kwargs'])}")
        lines.append(f"Call shape: odoo_execute_method(model='{fact.model}', "
                     f"method='{p['method']}', res_ids=[<id>], kwargs={{}})")
        if p["siblings"]:
            lines.append(f"Other methods on this model (do NOT confuse them with "
                         f"the answer): {', '.join(p['siblings'])}")

    elif fact.kind == "field_reference":
        lines.append(f"Fields safe to read and filter on: {', '.join(p['searchable'])}")
        if p["required"]:
            req = ", ".join(
                f"{f['name']} ({f['type']}"
                + (f" -> {f['comodel']}" if f["comodel"] else "") + ")"
                for f in p["required"])
            lines.append(f"Required on create: {req}")
        if p["create"]:
            lines.append(f"A realistic create call supplies: {', '.join(p['create'])}")

    elif fact.kind == "relation":
        lines.append(f"Relation: {fact.model}.{p['field']} is a {p['type']} "
                     f"to {p['target']}.")
        if p["type"] == "Many2one":
            lines.append(f"Reading it returns [id, display_name]. To filter across "
                         f"it, use a dotted path such as "
                         f"[['{p['field']}.name', 'ilike', '...']].")
        else:
            lines.append("It holds a list of ids; filter with 'in' or a dotted path.")

    elif fact.kind == "state_machine":
        lines.append("Verified transitions:")
        for frm, to, name in p["transitions"]:
            lines.append(f"  {frm} --{name}()--> {to}")

    return "\n".join(lines)


def build_teacher_prompt(fact: Fact, question: str) -> Tuple[str, str]:
    """The user turn carries only the question and the facts — no directives."""
    return SYSTEM_PROMPT, (
        f"Question:\n  {question}\n\n"
        f"=== Verified facts, extracted from the Odoo 19 source tree ===\n"
        f"{grounding_for(fact)}\n\n"
        f"Answer the question using only these facts."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Filtering the first-attempt knowledge base
# ──────────────────────────────────────────────────────────────────────────────

def filter_legacy_kb(path: str, surface: Iterable[str]) -> List[Dict[str, Any]]:
    """Keeps only the rows of the old KB that concern a model in the surface.

    The file covers all 2,266 models evenly, so ~97% of it is about models an
    agent will never touch. Those rows are not wrong, they are just capacity
    spent on ``decimal.precision``.
    """
    wanted = set(surface)
    kept: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("_model") in wanted:
                    kept.append(row)
    except FileNotFoundError:
        logger.info("No legacy knowledge base at %s; skipping.", path)
    return kept
