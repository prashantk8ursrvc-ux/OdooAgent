"""
Production Quality Gate
=======================

One gate, two callers:

  * the generator, which rejects a freshly-produced sample before it is ever
    written to the cache, and
  * the salvage tool, which runs the same checks over the legacy cache so the
    surviving rows meet exactly the standard new rows must meet.

Every rejection carries a machine-readable reason so the run produces a report
you can act on rather than a single "n samples dropped" number.

Checks, and why each one exists
-------------------------------
``truncated``       4,717 of 14,285 legacy rows end mid-sentence.  ``max_tokens``
                    and ``reasoning_budget`` were both 16,384, so reasoning ate
                    the budget and the answer was cut — then cached anyway
                    because nothing looked at ``finish_reason``.
``version_drift``   798 of 1,498 planning rows cite Odoo 14–18.  A model trained
                    to answer "Odoo 19" questions with Odoo 15 behaviour is worse
                    than useless.
``teacher_voice``   The teacher was handed a meta-prompt ("The user asked: ...")
                    and echoed it.  1,303 rows contain reasoning that talks
                    *about* a user instead of *to* one.
``prompt_leak``     Schema blocks handed to the teacher appear in the output.
``chatty``          "Let me know if you'd like...", "Happy Odoo-building! 🚀".
                    An MCP agent driving a live ERP does not sign off like that.
``robotic_prompt``  The user turn is machine-generated template text.
``bad_tool_call``   The tool call targets a method that does not exist on that
                    model, or ships placeholder values.
``duplicate``       Exact and near-exact repeats.
``thin``            Assistant turn too short to teach anything.
``broken_think``    Unbalanced ``<think>`` tags.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Patterns
# ──────────────────────────────────────────────────────────────────────────────

# Any Odoo version at or below 18 is wrong for this dataset.
VERSION_DRIFT = re.compile(
    r"\bOdoo\s*(?:v\.?\s*)?(?:[89]|1[0-8])(?:\.\d)?\b"
    r"|\bversion\s+(?:[89]|1[0-8])(?:\.\d)?\s+of\s+Odoo\b"
    r"|\bOdoo\s+(?:[89]|1[0-8])\.\d\b",
    re.IGNORECASE,
)

# Artifacts of how the sample was manufactured.  These are always a defect,
# including inside <think>, because they describe the generation harness rather
# than the business task.
PROMPT_ARTIFACT = re.compile(
    r"\bScenario\s*#\d+"
    r"|\bThey labeled this as\b"
    r"|\bAs an AI\b|\bas an AI language model\b"
    r"|\bI don'?t have (?:the )?context for (?:previous|prior|earlier)\b"
    r"|\bthe teacher (?:prompt|model|said)\b"
    r"|\bmy system prompt\b"
    r"|\bWe need to respond as\b"
    r"|\bthe user'?s (?:teacher|meta) prompt\b",
    re.IGNORECASE,
)

# The teacher reasoning about the *formatting constraints* it was given rather
# than about the business problem: "Must be 2-6 sentences, no sign-offs, no
# emojis." Harmless-looking, but it trains the model to spend its reasoning
# budget rehearsing style rules. Checked inside <think> as well as outside.
# Every alternative must be a *style* directive. Do not add bare modal patterns
# like `\bmust be\b`: an earlier version did, and it rejected ordinary Odoo
# reasoning ("the order must be confirmed before the delivery can be created"),
# taking a whole family's yield to zero. When adding a rule here, check it
# against tests/test_quality_gate.py::LEGITIMATE_REASONING first.
INSTRUCTION_ECHO = re.compile(
    # Formatting constraints restated verbatim
    r"\bno sign-?offs?\b"
    r"|\bno emojis?\b"
    r"|\b\d\s*(?:-|–|to)\s*\d\s+sentences\b"
    r"|\bmust be (?:concrete|brief|short|useful and no more)\b"
    r"|\bno (?:fluff|padding|preamble)\b"
    r'|\bno "?let me know"?\b'
    r"|\bmust speak to the user\b"
    r"|\b(?:without mentioning|not mention|never mention) Odoo\b"
    r"|\bas (?:instructed|per the instructions)\b"
    r"|\bthe (?:hard )?constraints? (?:say|require|state)\b"
    r"|\bfollow(?:ing)? the (?:house )?rules\b"
    # Planning the *response* rather than the problem
    r"|\bWe need to (?:respond|reply|produce|write|answer|provide|explain|present)\b"
    r"|\bWe also need to (?:present|provide|include|mention|show)\b"
    r"|\b(?:should|do) not make (?:a )?tool call\b"
    r"|\bprovide (?:one or two|1-2) questions?\b"
    r"|\boffer (?:the )?likely answer\b"
    r"|\bprovide (?:a )?step(?:-| )?by(?:-| )?step (?:plan|recovery)\b"
    r"|\bprovide stepwise\b"
    r"|\bend with (?:a )?step(?:-| )?by(?:-| )?step\b"
    r"|\buse clear language\b"
    r"|\bpolitely,? then\b"
    r"|\bSo (?:reply|answer|something like)\s*:"
    r"|\bLet'?s craft\b"
    # Directives about the *reply* rather than the business. Deliberately
    # narrow: a bare `must (be|not)` also matches "the order must be confirmed
    # before the delivery can be created", which is what correct Odoo reasoning
    # sounds like. These verbs only ever govern the response.
    r"|\b[Mm]ust not (?:mention|quote|say|state|fabricate|invent|include|use)\b"
    r"|\b[Mm]ust (?:report|say|mention|state|include|avoid|keep it|write)\b"
    r"|\b[Ww]rite (?:a )?concise reply\b"
    r"|\bone sentence,? maybe two\b"
    r"|\bwe (?:cannot|can'?t) state its status\b"
    r"|\bonly (?:have|got) (?:a )?success flag\b"
    r"|\bnot returned by the (?:call|tool)\b"
    # Standalone imperative style notes on their own line
    r"|^\s*(?:Keep (?:it )?concise|Be concise|Keep it short|No fluff)\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# The teacher writing a tool invocation as prose in the final answer, e.g. a bare
# line reading ``odoo_search_read('account.move', [['id','=',33392]], [...])``.
# Tool calls belong in the ``tool_calls`` field; as visible text they teach the
# model to narrate calls it never actually issues.
TOOL_CALL_AS_TEXT = re.compile(
    r"^\s*`?odoo_(?:search_read|read_group|create|write|execute_method|unlink)\s*\(",
    re.MULTILINE,
)

# A document reference carrying a year the dataset never generates. All synthetic
# references are stamped with the dataset year, so any other year in a reference
# is a value the teacher invented rather than read from a tool result.
_DATASET_YEAR = 2026
FABRICATED_REF = re.compile(
    r"\b(?:INV|BILL|RINV|RBILL|SO|PO|POS|BNK\d?|MO|RO)[\-/](?:20[0-2]\d)[\-/]\d"
    r"|\b(?:INV|BILL|SO|PO)[\-/]?(?:20[0-2]\d)\b",
    re.IGNORECASE,
)

# Third-person narration about "the user".  Fine inside <think> — that is how a
# reasoning trace reads — but a defect in the answer the user actually sees,
# where the assistant should be addressing them directly.
TEACHER_VOICE = re.compile(
    r"\bthe user (?:asked|wants|is asking|said|requested|has requested)\b"
    r"|\bWe need to (?:produce|provide|answer)\b",
    re.IGNORECASE,
)

# Fragments of the grounding block leaking into the answer.
PROMPT_LEAK = re.compile(
    r"===\s*ODOO 19 MODEL TECHNICAL DEFINITION\s*==="
    r"|\[MANDATORY REQUIRED FIELD\]"
    r"|\[SELECTION OPTIONS:"
    r"|^Model Technical Name:"
    r"|Extracted Fields & Constraints"
    r"|Relational Edges:"
    r"|\bUser request: '"
    r"|\bTeacher prompt\b",
    re.IGNORECASE | re.MULTILINE,
)

# Assistant sign-offs that do not belong in an agent transcript.
CHATTY = re.compile(
    r"Happy (?:implementing|Odoo|building|configuring)"
    r"|Let me know if you(?:'d| would) like"
    r"|Would you like me to (?:elaborate|expand|continue|dive)"
    r"|feel free to (?:ask|reach out)"
    r"|I hope this helps"
    r"|Good luck (?:with|on)"
    r"|🚀|🎉|💡\s*$",
    re.IGNORECASE,
)

# The old Python templates.  A user turn matching any of these is machine text.
ROBOTIC_PROMPT = re.compile(
    r"^Execute standard ERP database operation on"
    r"|^Execute full multi-step business workflow"
    r"|^Invoke MCP tool `"
    r"|^Search and retrieve (?:active )?records from Odoo model"
    r"|^Update security policy and access rights configuration for"
    r"|^Execute manufacturing workflow operation for"
    r"|^Manage inventory transfer and stock valuation for"
    r"|^Process human resources record for"
    r"|^Configure project management stage and workflow settings for"
    r"|^Update inventory catalog definition and pricing rules for"
    r"|^Create and initialize a new .* record for partner"
    r"|^Verify database mutation and lock status for"
    r"|^Confirm whether database record"
    r"|^Check if .* record .* has completed its state transition"
    r"|^Call business method '"
    r"|^Transition Odoo .* from '.*' state to '.*' state"
    r"|^Validate that Odoo .* has been properly processed with state"
    r"|^Verify whether Odoo .* is .* in the database"
    r"|^Provide a complete execution plan for an Odoo business workflow from"
    r"|^Perform full end-to-end management of Odoo model"
    r"|^I want to create a new .* record\.$"
    r"|^Update Odoo .* record ID \d+ with"
    r"|^Create a new .* \(.*\) record in Odoo with"
    # The error-recovery family's hand-written templates. Less obviously
    # machine-generated than the rest, but still one fixed sentence per
    # scenario with the reference substituted in — eleven distinct user turns
    # across 1,476 samples.
    r"|^Post customer invoice \S+ for .+\.$"
    r"|^Confirm sales order \S+ for .+\.$"
    r"|^Confirm purchase order \S+ from vendor .+\.$"
    r"|^Validate warehouse delivery transfer \S+\.$"
    r"|^Create partner record for '.*' with internal reference"
    r"|^Reverse posted customer invoice \S+\.$"
    r"|^Register payment for invoice \S+\.$"
    r"|^Send invoice \S+ via email and download PDF\.$"
    r"|^Approve employee leave request \S+\.$"
    r"|^Confirm employee payslip \S+\.$"
    r"|^Scrap damaged components for production order \S+\.$",
    re.IGNORECASE,
)

# Placeholder values the old _build_exact_tool_args emitted.
PLACEHOLDER_VALUE = re.compile(r"^Sample .*#\d+$|^Sample Value$|^option1$|^Updated #\d+$")

# Sentence-final characters that indicate the answer actually finished.
_TERMINALS = ".!?)]`:\"'*…—✓✅"

# A line of an unbulleted list: an identifier, optionally a call or a field path,
# then a separator and a description. e.g. "action_confirm() – confirms a quote"
_LIST_ITEM = re.compile(
    r"^`?[\w.]+(?:\(\))?`?\s*[–—:-]\s+\S",
)


def _is_truncated(text: str) -> bool:
    """True when the text stops mid-thought rather than ending."""
    t = (text or "").rstrip()
    if not t:
        return True
    last = t[-1]
    if last in _TERMINALS:
        return False
    # Emoji or other symbol as the final character counts as a deliberate ending.
    if unicodedata.category(last) in ("So", "Sk"):
        return False
    # A markdown table row or list bullet that ends without punctuation is fine.
    tail = t.rsplit("\n", 1)[-1].strip()
    if tail.startswith(("|", "-", "*", "#")) and len(tail) > 3:
        return False

    # An unbulleted list is just as complete. A method inventory reads
    #   action_confirm() – confirms a quotation into a sales order
    #   action_cancel()  – cancels the sales order
    # with no terminal punctuation and no bullet, and was being read as cut off.
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if len(lines) >= 3 and _LIST_ITEM.match(tail):
        matching = sum(1 for ln in lines if _LIST_ITEM.match(ln))
        if matching >= 3:
            return False
    return True


# "its state is now 'scrapped'", "the status is now **posted**", "is now in
# state `done`". Captures the claimed value so it can be checked against what
# the tools actually returned.
STATE_ASSERTION = re.compile(
    r"\b(?:state|status)\s+(?:is|was|has been (?:set|changed|updated))\s+"
    r"(?:now\s+)?(?:to\s+|set to\s+)?[`'\"*]{0,2}(?P<state>[a-z_][a-z_ ]{1,24})[`'\"*]{0,2}"
    r"|\bis now in (?:the\s+)?(?:state|status)\s+[`'\"*]{0,2}(?P<state2>[a-z_]{2,24})",
    re.IGNORECASE,
)

# Phase 1 asks the teacher to role-play a persona and emit only their message.
# Smaller pool models sometimes emit their working-out as ordinary content, and
# it lands in the user turn: "We need to start mid-thought... Output only that
# text.", 'the rule says "do not describe steps"', "\\boxed{}".
#
# This is the worst possible place for a defect, because the user turn is the
# only span the trained model conditions on. Every alternative below must be
# about the *writing task*; nothing here may match a real person's message.
# Note the absence of a bare "^actually" — a user legitimately opens with it.
USER_TURN_META = re.compile(
    # Referring to the phase-1 instructions
    r"\bthe rules? (?:say|says|said|only forbids)\b"
    r"|\bdo not (?:name the|describe steps|use phrases)\b"
    r"|\bmust not (?:use|name|describe)\b"
    r"|\bper the instruction\b|\bthe instruction says\b"
    r"|\bas the persona\b|\brole-?play\b"
    r"|\bthe odoo model\b|\btechnical name\b"
    # Talking about the act of writing the message
    r"|\boutput (?:only|just) (?:that|the message|the text)\b"
    r"|\bno (?:quotation marks|label|preamble|extra formatting)\b"
    r"|\bwe need to (?:write|start mid-thought|finish the sentence)\b"
    r"|\bwe need to (?:keep|avoid) (?:the reference|it to|naming|the word)\b"
    r"|\bmissing apostrophe\b|\binclude (?:a|one) small typo\b"
    r"|\bthat'?s (?:one|two|three) (?:long )?sentences?\b"
    r"|\bcould add a third\b|\bcomma splice\b"
    r"|\bLet'?s (?:count|do two sentences|write|draft)\b"
    r"|\bso (?:maybe|we can|to be safe) \""
    r"|\bensure no extra\b"
    r"|\\boxed\{",
    re.IGNORECASE,
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Returns only the part of an assistant turn the user actually sees."""
    return _THINK_BLOCK.sub("", text or "")


def _normalise(text: str) -> str:
    """Lowercased, digit- and whitespace-collapsed form used for near-dup detection."""
    t = re.sub(r"\d+", "#", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    """Outcome of gating one sample."""

    ok: bool
    reasons: List[str] = field(default_factory=list)

    def fail(self, reason: str) -> "Verdict":
        self.ok = False
        self.reasons.append(reason)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# The gate
# ──────────────────────────────────────────────────────────────────────────────

class QualityGate:
    """Stateful across a run so it can detect duplicates and near-duplicates.

    Parameters
    ----------
    methods_by_model
        ``{model: {method, ...}}`` from the knowledge graph.  Tool calls naming a
        method outside this set are rejected.  Pass ``None`` to skip the check
        (only sensible when no KG is available).
    max_chars
        Sequence-length ceiling, in characters.  Compared against the whole
        serialised conversation, matching the trainer's ``max_seq_length``.
    allow_robotic
        Escape hatch for regression tests.  Never enable for a production run.
    """

    MIN_ASSISTANT_CHARS = 120
    MIN_USER_CHARS = 12

    def __init__(
        self,
        methods_by_model: Optional[Dict[str, Set[str]]] = None,
        max_chars: int = 32768,
        allow_robotic: bool = False,
    ) -> None:
        self.methods_by_model = methods_by_model
        self.max_chars = max_chars
        self.allow_robotic = allow_robotic
        self._exact: Set[str] = set()
        self._near: Set[str] = set()
        self.counts: Dict[str, int] = {}

    # -- helpers ---------------------------------------------------------------
    def _tally(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1

    @staticmethod
    def _assistant_turns(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [m for m in messages if m.get("role") == "assistant"]

    # -- the checks ------------------------------------------------------------
    def check(self, sample: Dict[str, Any]) -> Verdict:
        v = Verdict(ok=True)
        messages = sample.get("messages")

        if not isinstance(messages, list) or len(messages) < 2:
            return self._reject(v, "malformed")

        roles = [m.get("role") for m in messages]
        if any(r not in ("system", "user", "assistant", "tool") for r in roles):
            return self._reject(v, "malformed")
        if "assistant" not in roles or "user" not in roles:
            return self._reject(v, "malformed")

        # -- structural: every tool result must answer a real tool call --------
        issued: Set[str] = set()
        for m in messages:
            for tc in (m.get("tool_calls") or []):
                if tc.get("id"):
                    issued.add(tc["id"])
        for m in messages:
            if m.get("role") == "tool":
                tcid = m.get("tool_call_id")
                if not tcid or tcid not in issued:
                    self._reject(v, "orphan_tool_result")
                    break

        # -- the user turn -----------------------------------------------------
        user_turns = [m.get("content") or "" for m in messages if m.get("role") == "user"]
        first_user = user_turns[0] if user_turns else ""
        if len(first_user.strip()) < self.MIN_USER_CHARS:
            self._reject(v, "thin_user_turn")
        if not self.allow_robotic:
            for u in user_turns:
                if ROBOTIC_PROMPT.search(u.strip()):
                    self._reject(v, "robotic_prompt")
                    break
        for u in user_turns:
            if USER_TURN_META.search(u):
                self._reject(v, "user_turn_meta")
                break

        # -- the assistant turns ----------------------------------------------
        assistants = self._assistant_turns(messages)
        if not assistants:
            return self._reject(v, "malformed")

        joined = "\n".join(a.get("content") or "" for a in assistants)

        if joined.count("<think>") != joined.count("</think>"):
            self._reject(v, "broken_think")

        final = ""
        for a in reversed(assistants):
            if a.get("content"):
                final = a["content"]
                break

        # Only text-final turns need a length floor; a turn that ends in a tool
        # call is legitimately short.
        last_is_tool_call = bool(assistants[-1].get("tool_calls"))
        if not last_is_tool_call and len(final.strip()) < self.MIN_ASSISTANT_CHARS:
            self._reject(v, "thin_assistant_turn")

        if not last_is_tool_call and _is_truncated(final):
            self._reject(v, "truncated")

        if VERSION_DRIFT.search(joined):
            self._reject(v, "version_drift")
        if PROMPT_ARTIFACT.search(joined):
            self._reject(v, "prompt_artifact")
        if INSTRUCTION_ECHO.search(joined):
            self._reject(v, "instruction_echo")
        if PROMPT_LEAK.search(joined):
            self._reject(v, "prompt_leak")
        if CHATTY.search(final):
            self._reject(v, "chatty")
        # Writing a call as prose is a defect in a transcript, where the call
        # belongs in tool_calls. In an explanation with no tool calls at all,
        # showing the call shape is the answer — "use action_cancel() like this:
        # odoo_execute_method(...)" is exactly what was asked for.
        has_tool_calls = any(m.get("tool_calls") for m in messages)
        if has_tool_calls and TOOL_CALL_AS_TEXT.search(strip_think(joined)):
            self._reject(v, "tool_call_as_text")
        if self._claims_unreported_state(messages, strip_think(final)):
            self._reject(v, "unsupported_state_claim")
        for match in FABRICATED_REF.finditer(strip_think(joined)):
            year = re.search(r"20[0-2]\d", match.group())
            if year and int(year.group()) != _DATASET_YEAR:
                self._reject(v, "fabricated_reference")
                break
        # Third-person narration is only a defect outside the reasoning trace.
        if TEACHER_VOICE.search(strip_think(joined)):
            self._reject(v, "teacher_voice")

        # -- tool calls --------------------------------------------------------
        for m in messages:
            for tc in (m.get("tool_calls") or []):
                if not self._check_tool_call(tc, v):
                    break

        # -- budget ------------------------------------------------------------
        serialised = json.dumps(messages, ensure_ascii=False)
        if len(serialised) > self.max_chars:
            self._reject(v, "over_length")

        # -- duplication -------------------------------------------------------
        exact = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
        if exact in self._exact:
            self._reject(v, "duplicate")
        else:
            self._exact.add(exact)
            near = hashlib.sha256(
                (_normalise(first_user) + "||" + _normalise(final[:600])).encode("utf-8")
            ).hexdigest()
            if near in self._near:
                self._reject(v, "near_duplicate")
            else:
                self._near.add(near)

        return v

    def _check_tool_call(self, tc: Dict[str, Any], v: Verdict) -> bool:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        if not isinstance(raw, str):
            self._reject(v, "bad_tool_args")
            return False
        try:
            args = json.loads(raw)
        except (ValueError, TypeError):
            self._reject(v, "bad_tool_args")
            return False
        if not isinstance(args, dict):
            self._reject(v, "bad_tool_args")
            return False

        model = args.get("model")

        if fn.get("name") == "odoo_execute_method" and self.methods_by_model is not None:
            method = args.get("method")
            known = self.methods_by_model.get(model)
            if known is None:
                self._reject(v, "unknown_model")
                return False
            if method not in known:
                self._reject(v, "hallucinated_method")
                return False

        values = args.get("values")
        if isinstance(values, dict):
            for val in values.values():
                if isinstance(val, str) and PLACEHOLDER_VALUE.match(val):
                    self._reject(v, "placeholder_values")
                    return False
        return True

    @staticmethod
    def _claims_unreported_state(messages: Sequence[Dict[str, Any]], final: str) -> bool:
        """True when the answer asserts a record state no tool result returned.

        The wide surface deliberately does not declare state transitions for
        models whose state machine was not hand-verified, so those calls return
        only ``{"result": true, "id": N}``. Left unchecked the teacher fills the
        gap: a live sample reported "its state is now 'scrapped' and
        quantity_done is set to 0" from exactly that payload. Asserting a status
        the tools never reported is a hallucination, and training on it teaches
        the model to invent confirmations.
        """
        if not STATE_ASSERTION.search(final):
            return False

        # Collect every state-ish value the tools actually returned.
        reported: Set[str] = set()
        for m in messages:
            if m.get("role") != "tool":
                continue
            try:
                payload = json.loads(m.get("content") or "null")
            except (ValueError, TypeError):
                continue
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in ("state", "status", "payment_state", "invoice_status"):
                    val = row.get(key)
                    if isinstance(val, str):
                        reported.add(val.lower())

        # An assertion is fine only if the value it names was actually returned.
        for match in STATE_ASSERTION.finditer(final):
            claimed = ((match.group("state") or match.group("state2") or "")
                       .strip(" '\"`*.").lower())
            if claimed and claimed not in reported:
                return True
        return False

    def _reject(self, v: Verdict, reason: str) -> Verdict:
        self._tally(reason)
        return v.fail(reason)

    # -- reporting -------------------------------------------------------------
    def report(self) -> str:
        if not self.counts:
            return "No rejections."
        width = max(len(k) for k in self.counts)
        lines = [f"  {k:<{width}}  {n:>7,}" for k, n in
                 sorted(self.counts.items(), key=lambda kv: -kv[1])]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────────

def load_methods_index(kg) -> Dict[str, Set[str]]:
    """Builds ``{model: {method, ...}}`` from the knowledge graph.

    Includes the ORM primitives an MCP server legitimately exposes on every
    model, so ``create`` / ``write`` / ``unlink`` / ``read`` are not flagged.
    """
    import sqlite3

    universal = {"create", "write", "unlink", "read", "search", "search_read",
                 "search_count", "copy", "name_search", "read_group",
                 "fields_get", "default_get", "message_post", "toggle_active"}

    index: Dict[str, Set[str]] = {}
    with sqlite3.connect(kg.db_path) as conn:
        for model, method in conn.execute("SELECT model_name, method_name FROM methods"):
            index.setdefault(model, set()).add(method)
    for model in list(index):
        index[model] |= universal
    return index


#: Sentence splitter that keeps the terminator, so rejoining preserves prose.
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")


def clean_reasoning(text: str, min_keep: int = 150) -> str:
    """Strips instruction rehearsal out of a ``<think>`` block, keeping the rest.

    Smaller teacher models narrate the brief before they reason — "We need to
    provide a reply to the user…" — and the gate rejects that as
    ``instruction_echo``. On a mixed pool this is expensive: a throughput test
    found every sample from the smaller endpoints rejected for it, so twelve of
    fifteen endpoints burned quota and produced nothing, while all nine accepted
    samples came from the single largest model.

    The rehearsal is almost always confined to a sentence or two at the start of
    the trace; the reasoning after it is sound and the answer is untouched. So
    drop those sentences rather than the sample. If too little survives to be
    worth training on, the caller drops the block entirely — a sample with a good
    answer and no trace still teaches the behaviour.
    """
    cleaned, _ = clean_reasoning_counted(text, min_keep)
    return cleaned


def clean_reasoning_counted(text: str, min_keep: int = 150) -> Tuple[str, int]:
    """As :func:`clean_reasoning`, but also reports how many sentences it removed.

    The count has to be explicit. Comparing the rebuilt string against the
    original does not work, because ``_SENTENCE.findall`` is lossy — it cannot
    match a fragment with no non-terminator characters — so the rejoined text
    differs from the input even when nothing was dropped. Inferring the count
    that way reported 3,559 scrubbed traces where only 389 had anything removed.
    """
    if not text:
        return text, 0

    sentences = _SENTENCE.findall(text)
    kept = [s for s in sentences if not INSTRUCTION_ECHO.search(s)]
    removed = len(sentences) - len(kept)
    if not removed:
        return text, 0

    cleaned = "".join(kept).strip()
    return (cleaned if len(cleaned) >= min_keep else ""), removed


def scrub_think_blocks(messages: Sequence[Dict[str, Any]]) -> bool:
    """Cleans rehearsal from every reasoning trace in a conversation, in place.

    Returns True if anything changed. A trace reduced to nothing is removed
    along with its tags, leaving the answer as the whole turn.
    """
    changed = False
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if "<think>" not in content:
            continue

        # Tracked separately from `new != content`, because rebuilding the block
        # also normalises whitespace. Counting that as a scrub reported 3,558
        # cleaned traces where only 389 had anything removed — a number alarming
        # enough to look like a much larger problem than it was.
        removed_something = False

        def _replace(match: "re.Match") -> str:
            nonlocal removed_something
            inner = match.group(0)[len("<think>"):-len("</think>")]
            cleaned, removed = clean_reasoning_counted(inner)
            if removed:
                removed_something = True
            return f"<think>\n{cleaned}\n</think>" if cleaned else ""

        new = _THINK_BLOCK.sub(_replace, content).lstrip("\n")
        if new != content:
            msg["content"] = new
        changed = changed or removed_something
    return changed


def clean_assistant_text(text: str) -> str:
    """Strips the sign-offs and meta-framing that survive prompt engineering.

    Applied before gating, so a sample is only rejected for ``chatty`` when the
    problem is structural rather than a single trailing sentence.
    """
    if not text:
        return text
    lines = text.rstrip().split("\n")
    while lines and CHATTY.search(lines[-1]):
        lines.pop()
    out = "\n".join(lines).rstrip()
    # A trailing chatty clause on the final line, e.g. "... done. Let me know if…"
    out = re.sub(r"\s*(?:Let me know if [^.\n]*\.?|I hope this helps\.?|"
                 r"Would you like me to [^?\n]*\?)\s*$", "", out).rstrip()
    return out
