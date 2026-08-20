"""
Teacher Prompts
===============

Two phases, and the split is the point.

**Phase 1 — write the request.**  The teacher is put in the shoes of a named
persona reacting to a concrete business situation, and asked for the message that
person would type.  It is explicitly forbidden from naming the Odoo model, the
method, or the record id, because inferring those is the skill being trained.
This replaces the f-string templates that made 8,470 of the previous rows
unusable.

**Phase 2 — write the agent's turns.**  The plan has already been executed
against the simulator, so the teacher is shown the calls *and the results they
returned*, and writes the reasoning and the closing answer conditioned on both.
Previously the teacher saw only the user prompt and its ``<think>`` was glued
onto a tool sequence it had never seen, so the reasoning described a plan the
transcript did not follow.

Every prompt here carries the same set of hard constraints, because each one
corresponds to a defect measured in the legacy cache:

  * Odoo **19** only — 798 legacy rows cited Odoo 14-18.
  * No sign-offs — 597 rows ended with "Happy Odoo-building! 🚀".
  * No narrating the prompt back — 2,547 rows leaked generation artifacts.
  * Finish the answer — 4,327 rows stopped mid-sentence.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from odoo_agent_forge.agent_surface import TODAY, ModelSpec

# ──────────────────────────────────────────────────────────────────────────────
# The system prompt the trained model will actually ship with
# ──────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = (
    "You are an Odoo 19 operations agent connected to a live company database "
    "through an MCP server. You have no internet access; everything you know "
    "about this company comes from the tools.\n\n"
    "Tools:\n"
    "  odoo_search_read(model, domain, fields, limit)\n"
    "  odoo_read_group(model, domain, groupby, aggregates)\n"
    "  odoo_create(model, values)\n"
    "  odoo_write(model, res_ids, values)\n"
    "  odoo_execute_method(model, method, res_ids, kwargs)\n"
    "  odoo_unlink(model, res_ids)\n\n"
    "Work the way a competent colleague would: resolve what the user means to "
    "concrete records, prefer the business method over writing a state field by "
    "hand, read a result before relying on it, and confirm before doing anything "
    "irreversible. If a request is ambiguous, ask rather than guess."
)

# House rules live in the *system* prompt, never in the user turn.
#
# When they were appended to the user message the teacher treated them as part of
# the task and rehearsed them in its reasoning — producing think blocks like
# "Must be concrete, 2-6 sentences, no sign-offs, no fluff. Must speak to the
# user as the agent." That text then became training data, teaching the model to
# spend its reasoning budget on style compliance. Stated once as standing
# instructions in the system role, they are followed without being restated.
_HOUSE_RULES = """

How you write, always:
This is Odoo 19; you have never worked with an earlier version and never mention
one. You speak directly to the user as their agent — you do not narrate the task
back, discuss how you were asked, or refer to instructions. You are concrete:
real record references, real field names, real figures from the tool results,
and never a record the tools did not return. You finish your sentences. You say
what is useful and then stop, without a closing pleasantry.

You are brief. A colleague reading over your shoulder gets the answer in a few
sentences — rarely more than a short paragraph, and never a document with
headings and sections. Length is not thoroughness; a reply that runs past a
screen has stopped being an answer and become a report nobody asked for.

You only ever report what the tools actually returned. When a call comes back
with just a success flag and an id, that is what you say — the operation
succeeded and what it means for the business. You do not name a resulting status
or quote a field value you have not read; if the user would want that confirmed,
you can offer to read the record back, though you do not offer it every time.

Your reasoning is for working out the answer. Think about the records, the
states, and the business consequences — not about how to format your reply.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — the user's message
# ──────────────────────────────────────────────────────────────────────────────

_CHANNEL_HINT = {
    "chat": "You are typing into a chat box, so it is short and informal.",
    "email": "You are writing a short internal email, so it is a little more "
             "structured but still brief.",
    "voice-transcript": "This is a transcription of you speaking, so it is loose "
                        "and has verbal fillers.",
    "ticket": "You are writing in an internal ticket, so you may paste a fragment "
              "of what the customer said.",
}

_VERBOSITY_HINT = {
    "terse": "Keep it to one short sentence or even a fragment. Skip pleasantries.",
    "normal": "One to three sentences.",
    "rambling": "Two to four sentences, with a bit of context or worry that is not "
                "strictly necessary.",
}

#: Occasionally the request should be scruffy, because real ones are.
_ROUGHNESS = (
    (0.14, "Include one small typo or a missing apostrophe, the kind you make "
           "typing quickly. Do not make it hard to read."),
    (0.10, "Write it entirely in lowercase with minimal punctuation."),
    (0.08, "Start mid-thought, as if continuing an earlier conversation."),
    (0.07, "Include a short irrelevant aside about your day or the customer."),
)


def build_user_request_prompt(ctx: Dict[str, Any],
                              rng: random.Random) -> Tuple[str, str]:
    """Asks the teacher to write the message a real person would send."""
    spec: ModelSpec = ctx["spec"]
    persona = ctx["persona"]
    situation = ctx["situation"]
    shape = ctx.get("shape", "single_call")

    sys_prompt = (
        f"You are role-playing a {persona.role} at a mid-sized European company "
        f"that runs Odoo. {persona.style}\n\n"
        f"You are messaging an AI assistant that is connected to your Odoo system "
        f"and can act on it for you. You are not a developer and you are not "
        f"writing a specification — you are asking a colleague to get something done."
    )

    lines: List[str] = []
    if situation:
        lines.append(f"What is going on right now: {situation.text}")
    lines.append(f"The document involved is a {spec.label}"
                 + (f", reference {ctx['doc_ref']}" if spec.doc_prefix else "")
                 + f". The customer/partner is {ctx['partner_name']}.")
    lines.append(f"Today is {TODAY.strftime('%A %d %B %Y')}.")
    lines.append("")

    # For the question-shaped families the method is only there to anchor the
    # subject matter. Telling the persona to "confirm the order" and then to
    # "ask a question about the data" produces a self-contradictory request.
    if shape == "analysis":
        # The question must be answerable from the query that will actually run.
        # Generating them independently left 63.8% of report_analysis samples
        # concluding "this data does not answer your question" — technically
        # correct behaviour, but it made the family teach refusal instead of
        # analysis. ctx["data_hint"] describes the query built beforehand.
        hint = ctx.get("data_hint")
        if hint:
            lines.append(
                f"You want to know something your system can answer from this: "
                f"{hint}\n"
                f"Ask the business question a person would ask when that is the "
                f"figure they need — about the totals, the biggest or smallest, "
                f"or which group needs attention. Do not ask for anything outside "
                f"it (no other filters, no other fields, no other time period).\n"
                f"Say the time window the way a person speaks — 'since the start "
                f"of the year', 'over the last couple of months', 'this quarter' — "
                f"not as an ISO date. Do not read the description back; ask the "
                f"question it implies.")
        else:
            lines.append(f"You want to know something about your {spec.label} "
                         f"records — a total, a count, a shortlist, or which ones "
                         f"need attention.")
    elif shape == "explain":
        lines.append(f"You want to understand how {spec.label}s work in Odoo — why the "
                     f"system behaves a certain way, what a setting controls, or what "
                     f"the consequences of something are. Ask about that, not for a change.")
    elif shape == "clarify":
        lines.append(f"Roughly, you want to {ctx['method'].intent} — but see the "
                     f"vagueness instruction below.")
    elif shape == "record_update":
        hint = ctx.get("data_hint")
        lines.append(
            f"Something on an existing {spec.label.lower()} is wrong and you want it "
            f"corrected. This is a clerical fix, not a workflow step — you are not "
            f"asking to confirm, cancel, validate or post anything.")
        if hint:
            # Ask for exactly what the write will do. Otherwise the request and the
            # call describe different changes and the sample teaches the model to
            # edit fields the user never mentioned.
            lines.append(
                f"The correction you want is: {hint}. "
                f"Say it the way a person would — the new value in plain words, not "
                f"as a field name. Ask for those changes and no others.")
    elif shape == "state_write_refusal":
        lines.append(
            f"You want the {spec.label.lower()} to end up in a different state, and "
            f"you think of that as just changing a field — so ask for it that way, "
            f"as though the status were a value to be set. You do not know that "
            f"Odoo runs anything behind it.")
    elif shape == "record_creation":
        # The method on ctx is only there to anchor the subject matter. Falling
        # through to "what you want to happen: {method.intent}" produced requests
        # like "close this manufacturing order out as produced" answered by a
        # create call — data that teaches the model to create a record when asked
        # to cancel one.
        lines.append(
            f"You want a NEW {spec.label.lower()} set up — one that does not exist "
            f"yet. Ask for it the way someone does when they have half the details "
            f"to hand: give one or two concrete particulars and leave the rest "
            f"unsaid, because you assume the system knows or it does not matter to "
            f"you. Do not ask to change, confirm, cancel or close anything that "
            f"already exists.")
    else:
        lines.append(f"What you want to happen: {ctx['method'].intent}.")
    lines.append("")

    lines.append(_CHANNEL_HINT.get(persona.channel, ""))
    lines.append(_VERBOSITY_HINT.get(persona.verbosity, "One to three sentences."))

    for probability, instruction in _ROUGHNESS:
        if rng.random() < probability:
            lines.append(instruction)
            break

    lines.append("")
    lines.append("Rules for what you write:")
    lines.append(f"- Do NOT name the Odoo model ('{spec.model}'), the method "
                 f"('{ctx['method'].name}'), any field name, or any record id. "
                 f"You do not know or care about those.")
    lines.append("- Do NOT describe steps for the assistant to follow. Say what you "
                 "want, not how to do it.")
    lines.append("- Do NOT use phrases like 'execute', 'perform operation', "
                 "'initialize a record', or 'business workflow'. Nobody talks that way.")

    if shape == "clarify":
        lines.append(f"- Leave something out on purpose: {ctx.get('gap_desc', '')} "
                     f"Be vague in exactly that way — it should read as a normal "
                     f"hurried message, not as a deliberately broken one.")
    elif shape == "refuse_or_warn":
        lines.append("- Ask for it flatly and casually, as if it were routine. "
                     "You are not aware it is consequential.")
    elif shape in ("analysis",):
        lines.append("- Ask a question about the data rather than requesting a change. "
                     "You want a number or a shortlist, not a report.")
    elif shape == "explain":
        lines.append("- Ask how something works or why Odoo behaves a certain way. "
                     "You want to understand it, not change anything.")
    elif shape == "multi_step":
        lines.append("- Describe the outcome you want, which happens to take several "
                     "steps. Do not enumerate the steps yourself.")
        lines.append(
            f"You want the {spec.label.lower()} to end up in a different state, and "
            f"you think of that as just changing a field — so ask for it that way, "
            f"as though the status were a value to be set. You do not know that "
            f"Odoo runs anything behind it.")
    elif shape == "record_update":
        lines.append("- Ask for a correction to something that already exists. "
                     "Name the record.")
    elif shape == "state_write_refusal":
        lines.append("- Phrase it as setting or changing the status directly, the "
                     "way someone thinks of a spreadsheet column.")
    elif shape == "record_creation":
        lines.append("- Ask for something new to be set up. Mention one or two "
                     "details and no more — the point is that you have not given "
                     "the full picture.")

    lines.append("")
    lines.append("Output the message text and nothing else. No quotation marks, "
                 "no 'User:' label, no explanation.")

    return sys_prompt, "\n".join(l for l in lines if l is not None)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — the agent's turns
# ──────────────────────────────────────────────────────────────────────────────

def _render_transcript(calls: Sequence[Dict[str, Any]],
                       results: Sequence[Dict[str, Any]]) -> str:
    """Shows the teacher exactly what ran and exactly what came back."""
    out: List[str] = []
    for i, (call, result) in enumerate(zip(calls, results), start=1):
        args = json.dumps(call["arguments"], ensure_ascii=False)
        payload = json.dumps(result["payload"], ensure_ascii=False)
        if len(payload) > 1400:
            payload = payload[:1400] + " …(truncated for brevity)"
        status = "OK" if result["ok"] else "RAISED"
        out.append(f"Step {i} [{status}]  {call['name']}({args})")
        out.append(f"         returned: {payload}")
    return "\n".join(out)


def _results_carry_state(results: Sequence[Dict[str, Any]]) -> bool:
    """True when at least one tool result actually reports a record's state."""
    for r in results:
        payload = r.get("payload")
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if isinstance(row, dict) and any(
                k in row for k in ("state", "status", "payment_state", "invoice_status")
            ):
                return True
    return False


def build_agent_turn_prompt(
    ctx: Dict[str, Any],
    request: str,
    calls: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    emphasis: str = "",
) -> Tuple[str, str]:
    """Asks the teacher for the agent's reasoning and closing answer.

    The calls have already been executed, so the teacher writes *about what
    happened* rather than about a plan. This is what makes the ``<think>`` block
    consistent with the tool sequence in the finished sample.
    """
    spec: ModelSpec = ctx["spec"]
    failed = [r for r in results if not r["ok"]]

    sys_prompt = (
        AGENT_SYSTEM_PROMPT
        + "\n\nYou are producing the assistant side of one exchange. Your reasoning "
          "will be captured separately from your answer, so think freely and then "
          "give the user a clean, final reply."
        + _HOUSE_RULES
    )

    lines = [
        f"The user wrote:\n  {request}",
        "",
        f"You resolved this to the Odoo 19 model `{spec.model}` "
        f"({spec.label}), record {ctx['doc_ref']} (id {ctx['res_id']}).",
        "",
        "These calls have already been made on your behalf, with the results shown:",
        _render_transcript(calls, results),
        "",
    ]

    if failed:
        lines.append(
            "The last call raised an exception. Your reply must explain in plain "
            "business language what went wrong, why Odoo refused, and what happens "
            "next — either the corrective call you would make, or exactly what the "
            "user has to do first. Do not pretend it succeeded.")
    else:
        lines.append(
            "Write the reply the user gets after this worked. Report what changed, "
            "using the actual references and values above. If something is now "
            "possible or required as a consequence, say so in one line.")

        # A one-line factual note, not a directive.
        #
        # This used to be a paragraph of IMPORTANT/do-not instructions here in
        # the user turn. It worked — state fabrication stopped — but the teacher
        # then recited it: think blocks reading "Must not mention state, status,
        # or any field values not returned. Must not quote... Must not
        # fabricate." The rule itself moved to the system prompt; what is left
        # here is simply what the call returned, which is context, not a brief.
        if not _results_carry_state(results):
            lines += ["", "Note: that call returned only a success flag and the "
                          "record id — no state and no field values."]

    if emphasis:
        lines += ["", emphasis]

    return sys_prompt, "\n".join(lines)


def build_recovery_prompt(
    ctx: Dict[str, Any],
    request: str,
    call: Dict[str, Any],
    failure: Dict[str, Any],
) -> Tuple[str, str]:
    """The failure family, where diagnosis quality is the whole point."""
    spec: ModelSpec = ctx["spec"]
    sys_prompt = AGENT_SYSTEM_PROMPT + _HOUSE_RULES

    lines = [
        f"The user wrote:\n  {request}",
        "",
        f"You resolved this to `{spec.model}` record {ctx['doc_ref']} "
        f"(id {ctx['res_id']}) and called `{ctx['method'].name}`.",
        "",
        f"Odoo raised:\n  {failure['error']}",
        "",
        "Write your reply. It must:",
        "1. Say what failed, in the user's language, without the stack-trace tone.",
        "2. Explain the actual reason Odoo enforces this rule — the business "
        "   constraint behind it, not a restatement of the message.",
        "3. Give the concrete next step. If you can fix it yourself with another "
        "   call, say which one and what it would do. If it needs a human "
        "   (a permission, a missing configuration, a decision), say precisely who "
        "   has to do what.",
        "",
        "Do not apologise more than once. Do not offer a numbered list of five "
        "generic troubleshooting ideas — give the one that applies.",
    ]
    return sys_prompt, "\n".join(lines)


def build_clarification_prompt(
    ctx: Dict[str, Any],
    request: str,
    required_fields: Sequence[str],
) -> Tuple[str, str]:
    """The ask-don't-guess family."""
    spec: ModelSpec = ctx["spec"]
    sys_prompt = AGENT_SYSTEM_PROMPT + _HOUSE_RULES

    lines = [
        f"The user wrote:\n  {request}",
        "",
        f"This is about `{spec.model}` ({spec.label}), but the request is "
        f"under-specified in this way: {ctx.get('gap_desc', 'something needed is missing')}",
        "",
        f"Fields Odoo marks as required on this model: "
        f"{', '.join(required_fields) if required_fields else '(none extracted)'}",
        "",
        "Do NOT make a tool call. Write the reply that asks for what you need.",
        "It must:",
        "- Show you understood the intent, in one short line.",
        "- Ask only for what genuinely blocks you. One or two questions, not a form.",
        "- Where you can, offer the likely answer so the user can just confirm "
        "  ('I'm assuming the Rotterdam warehouse — say if not').",
        "",
        "Do not lecture the user about Odoo's data model. Do not list every "
        "required field. A good colleague asks the one question that unblocks them.",
        "",
        "You have not looked anything up yet, so you have no reference numbers, "
        "dates, or amounts. Do not invent one to 'offer as a likely answer' — "
        "offering a plausible-looking invoice number the user then confirms is "
        "worse than asking. Where you narrow things down, do it from what they "
        "actually told you, or offer to search.",
    ]
    return sys_prompt, "\n".join(lines)


def build_analysis_prompt(
    ctx: Dict[str, Any],
    request: str,
    call: Dict[str, Any],
    result: Dict[str, Any],
) -> Tuple[str, str]:
    """The aggregate-and-interpret family."""
    spec: ModelSpec = ctx["spec"]
    sys_prompt = AGENT_SYSTEM_PROMPT + _HOUSE_RULES

    lines = [
        f"The user asked:\n  {request}",
        "",
        f"You ran this aggregation on `{spec.model}`:",
        _render_transcript([call], [result]),
        "",
        "Answer the question from these figures. Requirements:",
        "- Lead with the answer, not with method.",
        "- Quote the actual numbers and group labels returned above. Do not round "
        "  away the detail and do not invent groups that are not there.",
        "- Add one observation that is genuinely useful — the outlier, the "
        "  concentration, the thing that would change what they do next.",
        "- If the data does not actually answer what they asked, say so and say "
        "  what you would need to query instead.",
        "",
        "A short table is fine when there are more than three groups. Prose "
        "otherwise. No executive-summary padding.",
    ]
    return sys_prompt, "\n".join(lines)


def build_explain_prompt(ctx: Dict[str, Any], request: str, kg) -> Tuple[str, str]:
    """The knowledge family, grounded in the extracted schema."""
    spec: ModelSpec = ctx["spec"]
    sys_prompt = (
        "You are an Odoo 19 functional and technical consultant answering a "
        "colleague's question. You know the codebase and you answer from it."
        + _HOUSE_RULES
    )

    lines = [
        f"The question:\n  {request}",
        "",
        f"It concerns the Odoo 19 model `{spec.model}` ({spec.label}).",
        "",
        _schema_block(spec, kg),
        "",
        "Answer the question that was actually asked. Requirements:",
        "- Ground every claim in the schema above. Use the real field and method "
        "  names; do not invent any.",
        "- Explain the business reason, not just the mechanics.",
        "- If the honest answer is 'it depends on configuration', say what it "
        "  depends on and where that setting lives.",
        "- No headings unless the answer genuinely has parts. No bulleted list of "
        "  every field on the model.",
    ]
    return sys_prompt, "\n".join(lines)


def _schema_block(spec: ModelSpec, kg, max_fields: int = 22) -> str:
    """Renders the verified schema for the teacher, from the AST extraction."""
    out = [f"=== Verified schema for {spec.model} ==="]

    try:
        fields = kg.get_model_fields(spec.model)
    except Exception:
        fields = []

    required = [f for f in fields if f.get("required")]
    relational = [f for f in fields if f.get("comodel_name")]
    others = [f for f in fields if f not in required and f not in relational]

    def render(f: Dict[str, Any]) -> str:
        t = f.get("field_type", "?")
        if f.get("comodel_name"):
            t += f" -> {f['comodel_name']}"
        tag = " [required]" if f.get("required") else ""
        sel = f" [options: {f['selection']}]" if f.get("selection") else ""
        return f"  {f['name']} ({t}){tag}{sel}: {f.get('string') or ''}"

    if required:
        out.append("Required fields:")
        out += [render(f) for f in required[:10]]
    if relational:
        out.append("Relations:")
        out += [render(f) for f in relational[:10]]
    if others:
        out.append("Other fields:")
        out += [render(f) for f in others[:max_fields]]

    out.append("Callable business methods (verified against the source tree):")
    for m in spec.methods:
        transition = (f" [{m.from_state} -> {m.to_state}]"
                      if m.from_state and m.to_state else
                      " [opens a wizard]" if m.returns_action else "")
        out.append(f"  {m.name}(){transition}: {m.intent}")

    return "\n".join(out)
