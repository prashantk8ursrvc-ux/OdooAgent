"""
Stage 3: Grounded Agent Dataset Generator
=========================================

What changed, and why
---------------------
The previous version of this module produced 14,285 samples of which the quality
gate accepts 1,010.  The defects were structural, not incidental:

1. **The user turn was a Python f-string.**  8,470 rows opened with phrasing like
   ``"Execute standard ERP database operation on Delivery Zip Prefix
   (delivery.zip.prefix)."``  The user turn is the only span the trained model
   conditions on, so templating it teaches the model to expect a machine to name
   the model and method for it — the exact inference it is supposed to perform.
   *Now:* the teacher LLM writes the request, in a persona's voice, about a
   concrete business situation, with the technical detail deliberately withheld.

2. **Model × method pairing was index arithmetic.**  ``models[i % len(models)]``
   crossed with ``methods[i % len(methods)]`` yielded ``action_post()`` on
   ``decimal.precision``.  *Now:* models come from a curated allowlist verified
   against the AST scan (:mod:`odoo_agent_forge.agent_surface`), methods come
   from that model's own verified list, and both are drawn with a seeded RNG so
   sample *i* is an independent draw rather than a lockstep cycle.

3. **Reasoning was decoupled from the tool calls.**  The teacher saw only the
   user prompt; Python hardcoded the tool sequence afterwards and glued the
   ``<think>`` onto step 0.  The reasoning described a plan the transcript did
   not follow, and the closing summary was written without ever seeing a tool
   result.  *Now:* generation is two-phase — the plan is executed against a
   simulator first, and the teacher writes the agent's turns *given the actual
   results*.

4. **Every tool call succeeded.**  ``{"status": "success"}`` for everything.
   *Now:* results are synthesised from the real schema, and a configurable share
   of samples fail with an exception the method genuinely raises.

5. **Cache identity was the user prompt, and family identity was a substring of
   the system prompt.**  Three families searched for a string their own system
   message did not contain, so they never found their cache and re-generated
   from scratch on every run — 2,481 duplicate rows in one family alone.  *Now:*
   every sample carries a ``_meta`` block with its family and a content hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from odoo_agent_forge import agent_surface as surface
from odoo_agent_forge import knowledge_pack as kpack
from odoo_agent_forge import scenarios as scen
from odoo_agent_forge.agent_surface import MethodSpec, ModelSpec, ValueFactory
from odoo_agent_forge.knowledge_graph import OdooKnowledgeGraph
from odoo_agent_forge.llm_client import LocalLLMClient, NvidiaLLMClient
from odoo_agent_forge.quality import (
    USER_TURN_META,
    QualityGate,
    clean_assistant_text,
    load_methods_index,
    scrub_think_blocks,
)
from odoo_agent_forge.simulator import OdooSimulator
from odoo_agent_forge.prompts import (
    AGENT_SYSTEM_PROMPT,
    build_agent_turn_prompt,
    build_analysis_prompt,
    build_clarification_prompt,
    build_explain_prompt,
    build_recovery_prompt,
    build_user_request_prompt,
)

logger = logging.getLogger(__name__)

#: Bumped whenever a change to prompts or structure invalidates cached samples.
GENERATOR_VERSION = "2.0"

#: Methods whose effect is hard to undo, or which reach backwards through a
#: document chain. These are what the refusal family trains the agent to pause on.
_DESTRUCTIVE_TOKENS = ("cancel", "draft", "unreserve", "unreconcil",
                       "refuse", "unlink", "unbuild", "reset", "reverse", "scrap")


def _is_destructive(method: MethodSpec) -> bool:
    if method.to_state in ("cancel", "draft"):
        return True
    return any(tok in method.name for tok in _DESTRUCTIVE_TOKENS)



#: Tools that change the database. Everything else only reads.
_MUTATING_TOOLS = frozenset({
    "odoo_create", "odoo_write", "odoo_unlink", "odoo_execute_method",
})

#: Past-tense claims that a change was made. Only the completed forms — "I will
#: confirm" and "this would post the entry" are exactly what a refusal turn should
#: say, so the pattern must not touch them.
#: Written without backslash escapes on purpose. An earlier version used the
#: usual word-boundary and whitespace classes and was authored through a shell
#: heredoc, which ate one level of escaping and wrote a literal backspace byte
#: (0x08) into the source. The pattern still compiled, and silently matched
#: nothing -- so every sample passed the check. A guard that always says yes is
#: worse than no guard, because it is believed. Plain character classes cannot
#: be corrupted that way.
_PAST_TENSE_WRITE = re.compile(
    "(?:^|[^a-zA-Z])I (?:have )?"
    "(?:confirmed|created|posted|updated|cancelled|canceled|deleted|validated|archived|changed|moved|assigned)"
    "|(?:has|have|had) been "
    "(?:confirmed|created|posted|updated|cancelled|canceled|deleted|validated|archived|moved)"
    "|successfully "
    "(?:confirmed|created|posted|updated|cancelled|deleted|validated)",
    re.IGNORECASE,
)


#: Words that turn a claim into its opposite. "No journal entry has been posted"
#: is the model correctly reporting that nothing happened — the very behaviour the
#: refusal family teaches — and matching it as a false claim is exactly backwards.
#: Checked in the ~40 characters before the match, which covers the clause without
#: reaching back into an unrelated sentence.
_NEGATION = re.compile(
    r"\b(?:no|not|never|nothing|without|cannot|can't|won't|isn't|aren't|hasn't|"
    r"haven't|nor|neither|yet to be|before)\b[^.!?]{0,40}$",
    re.IGNORECASE,
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _claims_an_unmade_write(messages) -> bool:
    """True when the final answer reports a change no tool call actually made.

    Two exclusions, both found by auditing the existing 21,983 samples against an
    earlier version of this check. It flagged 108 samples and every one inspected
    was a false positive:

    * **Negated statements.** "No journal entries have been posted yet because it
      is still in draft" is the model being scrupulous, not lying. Rejecting those
      would have cut 2% from refusal_and_confirmation, the family whose whole job
      is teaching the agent to explain rather than act.
    * **Reasoning blocks.** "Whether invoices have been created from the sales
      order" inside <think> is the model considering a possibility. Only what it
      says to the user is a claim.
    """
    all_calls = [
        (tc.get("function", tc) or {}).get("name")
        for m in messages
        for tc in (m.get("tool_calls") or [])
    ]

    # A turn that called nothing is answering a question, not reporting an action.
    # consultant_knowledge explains Odoo's behaviour in the abstract — "only 'done'
    # means the physical transfer has been posted", "if it has been confirmed but
    # not yet approved" — and every one of the 35 samples this caught before the
    # exclusion was exposition of that kind. The failure being guarded against is
    # an agent that *acted* and misreported it, which requires it to have acted.
    if not all_calls:
        return False

    if any(name in _MUTATING_TOOLS for name in all_calls):
        return False

    final = next(
        (m.get("content") or "" for m in reversed(messages)
         if m.get("role") == "assistant" and m.get("content")),
        "",
    )
    spoken = _THINK_BLOCK.sub(" ", final)

    for match in _PAST_TENSE_WRITE.finditer(spoken):
        if not _NEGATION.search(spoken[:match.start()]):
            return True
    return False


class DatasetGeneratorFactory:
    """Produces grounded, schema-verified agent training samples."""

    def __init__(
        self,
        kg: OdooKnowledgeGraph,
        use_nvidia_llm: bool = True,
        cache_path: str = "./forge_outputs/generation_cache_v2.jsonl",
        api_key: Optional[str] = None,
        use_local_llm: bool = False,
        local_llm_model: str = "qwen2.5:7b",
        local_llm_base_url: Optional[str] = None,
        seed: int = 20260317,
        failure_rate: float = 0.22,
        strict_surface: bool = False,
        max_workers: int = 3,
        wide_surface: bool = True,
    ) -> None:
        self.kg = kg
        self.use_nvidia_llm = use_nvidia_llm
        self.llm_client = NvidiaLLMClient(api_key=api_key) if use_nvidia_llm else None
        self.use_local_llm = use_local_llm
        self.local_llm_client = (
            LocalLLMClient(model=local_llm_model, base_url=local_llm_base_url)
            if use_local_llm else None
        )
        self.seed = seed
        self.failure_rate = failure_rate
        # Concurrency is the main driver of HTTP 429 from the teacher pool, so it
        # is configurable rather than hardcoded at 4 as it previously was.
        self.max_workers = max(1, max_workers)
        self.lock = threading.Lock()
        # Thread-local: workers run concurrently, so a shared attribute would
        # attribute one worker's sample to another worker's endpoint.
        self._last_endpoint = threading.local()

        # -- grounding ---------------------------------------------------------
        # Tier A: hand-written state machines and real failure messages.
        self.core, warnings = surface.verify_against_kg(kg, strict=strict_surface)
        for w in warnings:
            logger.info("[agent surface] %s", w)
        if not self.core:
            raise RuntimeError(
                "No model in the curated agent surface could be verified against the "
                "knowledge graph. Run with --rebuild-db to populate forge_knowledge.db."
            )

        # Tier B: the wider Community + Enterprise surface, methods and fields
        # read from the AST scan. Carries no invented states or failures.
        self.surface = dict(self.core)
        if wide_surface:
            extra, notes = surface.discover_tier_b(kg, exclude=self.core)
            for n in notes:
                logger.debug("[tier B] %s", n)
            self.surface.update(extra)
            logger.info(
                "Agent surface: %d curated models (%d methods, %d with failure "
                "modes) + %d discovered models (%d methods) = %d total.",
                len(self.core), sum(len(s.methods) for s in self.core.values()),
                sum(1 for s in self.core.values() for m in s.methods if m.failures),
                len(extra), sum(len(s.methods) for s in extra.values()),
                len(self.surface),
            )
        else:
            logger.info("Agent surface: %d curated models, %d methods (Tier B off).",
                        len(self.core), sum(len(s.methods) for s in self.core.values()))

        self.pool = surface.build_sampling_pool(self.surface)
        # Families that need a real state machine or a real exception draw from
        # Tier A only; inventing either for a discovered model is what produced
        # v1's schema-invalid samples.
        self.core_pool = surface.build_sampling_pool(self.core)
        self.simulator = OdooSimulator(self.surface)

        # -- quality -----------------------------------------------------------
        self.gate = QualityGate(methods_by_model=load_methods_index(kg))

        # -- cache -------------------------------------------------------------
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.by_family: Dict[str, List[Dict[str, Any]]] = {}
        #: Highest sample index seen per family, so a resumed run draws fresh
        #: scenarios instead of re-deriving ones already on disk.
        self._max_index: Dict[str, int] = {}
        self._load_cache()

        self.stats: Dict[str, int] = {"generated": 0, "rejected": 0,
                                      "cache_hits": 0, "salvaged": 0}

    # ──────────────────────────────────────────────────────────────────────
    # Cache
    # ──────────────────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        """Loads the cache, indexed by the sample key stored in ``_meta``.

        The old cache keyed on the user prompt and identified a sample's family
        by searching for a substring of the system message.  Three families
        searched for a string their own system message did not contain, so their
        cache lookups always missed and every run appended a fresh copy.  Keying
        on explicit metadata makes a cache hit deterministic.
        """
        if not self.cache_path.exists():
            logger.info("No cache at %s; starting fresh.", self.cache_path)
            return

        skipped = 0
        regate_failed = 0
        regate_scrubbed = 0
        with open(self.cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except ValueError:
                    skipped += 1
                    continue
                meta = sample.get("_meta") or {}
                key, family = meta.get("key"), meta.get("family")
                if not key or not family:
                    skipped += 1
                    continue
                if meta.get("generator_version") != GENERATOR_VERSION:
                    skipped += 1
                    continue

                # Re-gate on load. The gate gains rules as audits find new defect
                # classes, and a sample cached before a rule existed would
                # otherwise survive forever. Dropping it here means the slot is
                # simply regenerated, with no surgery on a file a live run may
                # still be appending to.
                #
                # Scrub first, exactly as the generation path does. When a rule
                # tightens around instruction rehearsal, most affected rows have
                # a sound answer and a polluted trace — an audit found 388 of 389
                # recoverable — and regenerating those would be pure waste.
                if scrub_think_blocks(sample["messages"]):
                    regate_scrubbed += 1
                if not self.gate.check(sample).ok:
                    regate_failed += 1
                    continue

                self.cache[key] = sample
                self.by_family.setdefault(family, []).append(sample)
                idx = meta.get("index")
                if isinstance(idx, int):
                    self._max_index[family] = max(self._max_index.get(family, -1), idx)

        logger.info(
            "Loaded %d cached samples across %d families from %s "
            "(%d skipped as stale or unlabelled).",
            len(self.cache), len(self.by_family), self.cache_path, skipped,
        )
        if regate_scrubbed:
            logger.info(
                "Scrubbed instruction rehearsal from %d cached reasoning traces; "
                "their answers were sound, so they are kept rather than "
                "regenerated.", regate_scrubbed)
        if regate_failed:
            logger.warning(
                "%d cached samples no longer pass the quality gate and were "
                "dropped; they will be regenerated. Reasons:\n%s",
                regate_failed, self.gate.report(),
            )
            # Those rejections are historical, not a verdict on this run's output.
            self.gate.counts.clear()

    def _write_to_cache(self, sample: Dict[str, Any]) -> bool:
        """Appends the sample. Returns False if this key was already cached.

        The return value matters: a collision means the work was wasted, and the
        caller must not count it. Reporting success on a silent no-op is how a
        run can log "5/12 accepted" while the file on disk never grows.
        """
        meta = sample["_meta"]
        with self.lock:
            if meta["key"] in self.cache:
                return False
            self.cache[meta["key"]] = sample
            self.by_family.setdefault(meta["family"], []).append(sample)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return True

    @staticmethod
    def _sample_key(family: str, index: int, model: str, method: str, shape: str) -> str:
        raw = f"{GENERATOR_VERSION}|{family}|{index}|{model}|{method}|{shape}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    # ──────────────────────────────────────────────────────────────────────
    # LLM access
    # ──────────────────────────────────────────────────────────────────────

    def _ask(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 6144,
        reasoning_budget: int = 4096,
        temperature: float = 0.7,
    ) -> Tuple[Optional[str], Optional[str]]:
        """One teacher call. Returns ``(reasoning, answer)``; either may be None."""
        if not self.use_nvidia_llm or not self.llm_client:
            raise RuntimeError(
                "LLM generation is disabled or NVIDIA_API_KEY is missing. Template "
                "fallbacks are deliberately not implemented: they are what produced "
                "the 8,470 robotic prompts in the previous dataset. Run with "
                "--use-nvidia-llm and a valid key."
            )
        reasoning, answer, endpoint = self.llm_client.generate_with_thinking(
            sys_prompt, user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_budget=reasoning_budget,
            require_complete=True,
            return_endpoint=True,
        )
        # Which teacher wrote a sample is worth keeping. Mixing providers widens
        # throughput but also widens quality variance, and the gate only catches
        # structural defects — not "this model's reasoning is shallower". Tagging
        # each sample means a weak endpoint can be audited or filtered out later
        # without regenerating the rest.
        if endpoint:
            self._last_endpoint.endpoint = endpoint
        return reasoning, answer

    def _write_user_request(self, ctx: Dict[str, Any], rng: random.Random) -> Optional[str]:
        """Phase 1 — the teacher writes what the human would actually type.

        This is the single most important call in the pipeline.  It replaces the
        f-string templates that made 8,470 of the previous rows unusable.

        Phase 1 accounts for roughly 45% of all API calls while producing one
        short line, so on a rate-limited account it is where most of the wall
        clock goes. When a local model is configured it handles this phase
        instead: writing "what would a warehouse operator type" is well within a
        7B model's range, and it removes those calls from the throttled pool
        entirely. The graded reasoning in phase 2 always stays on the teacher.
        """
        sys_prompt, user_prompt = build_user_request_prompt(ctx, rng)

        if self.local_llm_client is not None:
            try:
                _, text = self.local_llm_client.generate_with_thinking(
                    sys_prompt, user_prompt, temperature=0.95, max_tokens=400)
            except Exception as exc:
                logger.debug("Local model failed on phase 1 (%s); using the teacher.", exc)
                text = None
            if text:
                return self._tidy_request(text)

        _, text = self._ask(
            sys_prompt, user_prompt,
            max_tokens=1024, reasoning_budget=512, temperature=0.95,
        )
        if not text:
            return None
        return self._tidy_request(text)

    def _tidy_request(self, text: str) -> Optional[str]:
        """Extracts the persona's message from whatever the teacher returned.

        Phase 1 asks for the message and nothing else, but smaller pool models
        emit their working-out as ordinary content rather than as reasoning. An
        audit of 4,271 samples found 89 (2.1%, and 9.2% of one family) whose user
        turn was teacher scratch-work — things like
        ``We need to start mid-thought... Output only that text.`` Taking the
        first paragraph, as this used to, passes that straight through into the
        one span the trained model conditions on.

        So: if the candidate reads as meta-commentary, look for the real message
        quoted inside it, and give up rather than emit scratch-work. Giving up
        just costs a retry.
        """
        request = (text or "").strip().strip('"').strip()
        for prefix in ("User:", "Message:", "Request:", "user:", "Here is", "Here's"):
            if request.startswith(prefix):
                request = request.split(":", 1)[-1].strip().strip('"')

        for candidate in self._request_candidates(request):
            if 12 <= len(candidate) <= 900 and not USER_TURN_META.search(candidate):
                return candidate
        return None

    @staticmethod
    def _request_candidates(text: str) -> List[str]:
        """Ordered guesses at the actual message inside a teacher response.

        When the teacher reasons out loud it usually quotes the line it settled
        on — ``So maybe "Hey, can you bring that variant back for Hollis & Vane?"
        That's fine.`` — so quoted spans are tried before raw paragraphs.
        """
        out: List[str] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        clean = [p for p in paragraphs if not USER_TURN_META.search(p)]
        if len(clean) == 1:
            out.append(clean[0])

        # Quoted spans, longest first: the settled-on message is usually the
        # longest thing in quotes.
        for quoted in sorted(re.findall(r'"([^"]{12,600})"', text),
                             key=len, reverse=True):
            out.append(quoted.strip())

        out.extend(clean)
        out.append(text)

        seen, ordered = set(), []
        for c in out:
            c = c.strip().strip('"').strip()
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    # ──────────────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────────────

    def _draw_context(
        self,
        index: int,
        family: str,
        mutating_only: bool = True,
        model_filter: Optional[Callable[[ModelSpec], bool]] = None,
        method_filter: Optional[Callable[[MethodSpec], bool]] = None,
        core_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Builds one fully grounded, independently drawn scenario context.

        ``method_filter`` matters: filtering only the *model* and then drawing a
        random method wastes draws, because a model that merely contains one
        destructive method usually yields a harmless one. Filtering both keeps
        the family on-topic.
        """
        rng = random.Random(f"{self.seed}|{family}|{index}")
        # A method_filter always means the family depends on curated metadata
        # (a real from/to state, or a real exception), which only Tier A carries.
        base = self.core_pool if (method_filter or core_only) else self.pool
        candidates = base
        if model_filter:
            candidates = [s for s in base if model_filter(s)]
        if not candidates:
            return None

        spec: ModelSpec = rng.choice(candidates)
        if method_filter:
            eligible = [m for m in spec.methods if method_filter(m)]
            method = rng.choice(eligible) if eligible else None
        else:
            method = surface.pick_method(spec, rng, mutating_only=mutating_only)
            if method is None:
                method = surface.pick_method(spec, rng, mutating_only=False)
        if method is None:
            return None

        vf = ValueFactory(rng.randrange(1 << 30))
        seq = rng.randint(1, 9999)
        persona = scen.pick_persona(rng, spec.personas)
        situation = scen.pick_situation(rng, spec.model, spec.domain)

        doc_ref = vf.doc_ref(spec, seq)
        partner_name = vf.company_name()
        product_name = vf.product_name()

        # On some models the document reference *is* the partner or the product.
        # Drawing them independently produced samples where the user asked about
        # "Delacroix Patisserie SARL" and the agent answered about "Vandenberg
        # Horticulture BV" — the single most jarring kind of incoherence, because
        # the two names sit in the same exchange.
        if spec.model == "res.partner":
            partner_name = doc_ref
        elif spec.model == "product.template":
            product_name = doc_ref

        return {
            "index": index,
            "family": family,
            "rng": rng,
            "spec": spec,
            "method": method,
            "vf": vf,
            "seq": seq,
            "res_id": rng.randint(101, 98999),
            "doc_ref": doc_ref,
            "partner_name": partner_name,
            "vendor_name": vf.vendor_name(),
            "product_name": product_name,
            "persona": persona,
            "situation": situation,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Parallel driver
    # ──────────────────────────────────────────────────────────────────────

    def _run_family(
        self,
        name: str,
        n: int,
        builder: Callable[[int], Optional[Dict[str, Any]]],
        family_num: int,
        total_families: int,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generates up to *n* accepted samples for one family.

        Overshoots the index range because the quality gate rejects some
        candidates; without headroom a family with a 15% rejection rate would
        silently return 15% short.
        """
        max_workers = max_workers or self.max_workers
        existing = list(self.by_family.get(name, []))
        if len(existing) >= n:
            logger.info(
                "[%d/%d %s] %d cached >= %d requested; nothing to generate.",
                family_num, total_families, name, len(existing), n,
            )
            self.stats["cache_hits"] += n
            return existing[:n]

        needed = n - len(existing)
        # Resume past every index this family has ever used, not merely past the
        # number of samples it kept. Attempts that the gate rejected still
        # consumed indices, so `len(existing)` lands back inside territory the
        # previous run already covered: the seeded draw is identical, the sample
        # key is identical, and the write is silently skipped as a duplicate.
        # A run in that state logs steady progress while the file never grows.
        if name in self._max_index:
            start = max(self._max_index[name] + 1, len(existing))
        else:
            # Samples cached before the index was recorded. Their consumed range
            # is unknown, but bounded: the attempt budget is 1.45x plus a small
            # constant, so twice the kept count clears it with room to spare.
            # Overshooting costs nothing — indices are just RNG seeds — whereas
            # undershooting silently re-derives draws already on disk.
            start = len(existing) * 2 + 64
            logger.info("[%s] no recorded indices (pre-upgrade cache); resuming "
                        "from %d to avoid re-deriving cached draws.", name, start)
        budget = int(needed * 1.45) + 8
        logger.info(
            "[%d/%d %s] %d cached, generating %d more (up to %d attempts, %d threads).",
            family_num, total_families, name, len(existing), needed, budget, max_workers,
        )

        accepted: List[Dict[str, Any]] = list(existing)
        produced = 0
        attempted = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(builder, start + i) for i in range(budget)]
            try:
                for future in as_completed(futures):
                    attempted += 1
                    try:
                        sample = future.result()
                    except Exception as exc:
                        logger.warning("[%s] sample failed: %s\n%s", name, exc,
                                       traceback.format_exc())
                        continue
                    if not sample:
                        continue
                    accepted.append(sample)
                    produced += 1
                    # Log early and often. At every-25 a family needing 29
                    # samples looked frozen for its whole first pass, which is
                    # indistinguishable from a stall.
                    if produced <= 3 or produced % 5 == 0 or produced == needed:
                        logger.info("   [%s] %d/%d accepted (%d attempts).",
                                    name, produced, needed, attempted)
                    if produced >= needed:
                        break
            finally:
                # Cancel before the pool's context manager blocks on shutdown.
                # Cancelling afterwards is too late: __exit__ waits for every
                # running future, so in-flight samples were completed and cached
                # but never counted, making the reported total look short.
                for f in futures:
                    f.cancel()

        # Anything an in-flight future cached after the break is still on disk and
        # will be picked up on the next run, so count what the cache actually holds.
        with self.lock:
            cached_now = list(self.by_family.get(name, []))
        if len(cached_now) > len(accepted):
            accepted = cached_now

        if self.gate.counts:
            logger.info("   [%s] gate rejections so far: %s", name,
                        ", ".join(f"{k}={v}" for k, v in
                                  sorted(self.gate.counts.items(), key=lambda kv: -kv[1])[:4]))
        if produced < needed:
            logger.warning(
                "[%d/%d %s] produced %d of %d requested from %d attempts. The "
                "shortfall is usually the teacher pool rate limiting; re-run to "
                "resume from cache.",
                family_num, total_families, name, produced, needed, attempted)

        logger.info("[%d/%d %s] done: %d samples.", family_num, total_families,
                    name, len(accepted))
        return accepted[:n]

    def _finalise(
        self,
        messages: List[Dict[str, Any]],
        ctx: Dict[str, Any],
        shape: str,
    ) -> Optional[Dict[str, Any]]:
        """Cleans, gates, tags, caches. Returns None when the sample is rejected."""
        for m in messages:
            if m.get("role") == "assistant" and m.get("content"):
                m["content"] = clean_assistant_text(m["content"])

        if _claims_an_unmade_write(messages):
            # Structural, so it does not depend on the teacher having obeyed an
            # instruction. Caught a state_write_refusal sample whose final turn read
            # "I confirmed purchase order P01992 using the button_confirm action"
            # when the only call in the trajectory was a search_read. Training on
            # that teaches the model to report work it never did — the failure the
            # whole confirmation design exists to prevent.
            self.gate.counts["claimed_unmade_write"] = (
                self.gate.counts.get("claimed_unmade_write", 0) + 1)
            self.stats["rejected"] += 1
            return None

        sample = {
            "messages": messages,
            "_meta": {
                "key": self._sample_key(ctx["family"], ctx["index"], ctx["spec"].model,
                                        ctx["method"].name, shape),
                "family": ctx["family"],
                "shape": shape,
                "model": ctx["spec"].model,
                "method": ctx["method"].name,
                "domain": ctx["spec"].domain,
                "persona": ctx["persona"].role,
                "index": ctx["index"],
                "generator_version": GENERATOR_VERSION,
                "teacher": getattr(self._last_endpoint, "endpoint", None),
            },
        }

        # Scrub instruction rehearsal out of the reasoning *before* gating.
        #
        # Doing it afterwards as a retry cannot work: the gate is stateful, so the
        # first check registers the sample's near-duplicate key, and re-checking
        # the scrubbed version — whose user turn and answer are unchanged — hits
        # that same key and reports near_duplicate against itself. Scrubbing up
        # front is also simpler, and a no-op when there is nothing to remove.
        if scrub_think_blocks(messages):
            self.stats["salvaged"] = self.stats.get("salvaged", 0) + 1

        with self.lock:
            verdict = self.gate.check(sample)

        if not verdict.ok:
            self.stats["rejected"] += 1
            logger.debug("[%s] rejected: %s", ctx["family"], ", ".join(verdict.reasons))
            return None

        if not self._write_to_cache(sample):
            # Same key already on disk: this index was consumed by an earlier
            # run. Returning it would inflate the counter without growing the
            # dataset, which is exactly how a run appears to stall.
            self.stats["duplicate_index"] = self.stats.get("duplicate_index", 0) + 1
            return None

        self.stats["generated"] += 1
        return sample

    # ──────────────────────────────────────────────────────────────────────
    # Families
    # ──────────────────────────────────────────────────────────────────────

    #: Declaration order. Scheduling order is decided at run time — see below.
    FAMILY_ORDER: Tuple[str, ...] = (
        "tool_calling", "lookup_then_act", "workflow_execution", "agent_trajectories",
        "error_recovery", "clarification_dialogues", "verification",
        "multi_turn_memory", "business_data_retrieval", "report_analysis",
        "consultant_knowledge", "mcp_agent_protocol", "refusal_and_confirmation",
        "schema_knowledge", "record_creation", "record_update",
        "method_discovery", "answer_and_stop",
    )

    def _family_map(self) -> Dict[str, Callable[[int], List[Dict[str, Any]]]]:
        return {
            "tool_calling": self.gen_tool_calling,
            "lookup_then_act": self.gen_lookup_then_act,
            "workflow_execution": self.gen_workflow_execution,
            "agent_trajectories": self.gen_agent_trajectories,
            "error_recovery": self.gen_error_recovery,
            "clarification_dialogues": self.gen_clarification,
            "verification": self.gen_verification,
            "multi_turn_memory": self.gen_multi_turn_memory,
            "business_data_retrieval": self.gen_data_retrieval,
            "report_analysis": self.gen_report_analysis,
            "consultant_knowledge": self.gen_consultant_qa,
            "mcp_agent_protocol": self.gen_mcp_agent_protocol,
            "refusal_and_confirmation": self.gen_refusal_and_confirmation,
            "schema_knowledge": self.gen_schema_knowledge,
            "record_creation": self.gen_record_creation,
            "record_update": self.gen_record_update,
            "method_discovery": self.gen_method_discovery,
            "answer_and_stop": self.gen_answer_and_stop,
        }

    def generate_all_families(
        self,
        count_per_family: int = 100,
        families: Optional[Sequence[str]] = None,
        chunk: int = 100,
        family_targets: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generates across families in balanced rounds rather than one at a time.

        The first version ran each family to completion before starting the next.
        At a realistic teacher throughput of ~100 samples/hour and a target of
        1,500 per family, that meant a week-long run produced four finished
        families and nine empty ones — a dataset with no coverage of verification,
        analysis, or refusal behaviour at all.

        Now each round tops every family up to the next multiple of ``chunk``, so
        an interrupted run yields thirteen partial families instead. That is
        strictly better for training, and because the family generators are
        cache-aware, stopping and resuming costs nothing.
        """
        fmap = self._family_map()
        names = list(families) if families else list(self.FAMILY_ORDER)

        unknown = [n for n in names if n not in fmap]
        if unknown:
            raise ValueError(
                f"Unknown famil{'y' if len(unknown) == 1 else 'ies'}: {unknown}. "
                f"Valid names: {', '.join(self.FAMILY_ORDER)}")

        # Per-family ceilings. Two families cannot reach an arbitrary target no
        # matter how long the run goes: `schema_knowledge` is bounded by the number
        # of facts in the knowledge base (986), and `error_recovery` by the number
        # of distinct real failure modes. Without a ceiling they are re-attempted
        # every round for the rest of the run, burning teacher calls on draws the
        # gate will reject as duplicates — the throughput just quietly halves.
        targets = {n: count_per_family for n in names}
        for name, value in (family_targets or {}).items():
            if name not in fmap:
                raise ValueError(
                    f"Unknown family in --family-target: {name}. "
                    f"Valid names: {', '.join(self.FAMILY_ORDER)}")
            if name in targets:
                targets[name] = int(value)

        chunk = max(1, min(chunk, count_per_family))
        rounds = -(-max(targets.values()) // chunk)      # ceiling division
        datasets: Dict[str, List[Dict[str, Any]]] = {n: [] for n in names}

        # Coverage up front. Knowing a family is 840 short before the run starts is
        # worth more than discovering it 13 hours later.
        have = {n: len(self.by_family.get(n, ())) for n in names}
        shortfall = sum(max(0, targets[n] - have[n]) for n in names)
        logger.info("Coverage before this run (cached / target):")
        for name in names:
            gap = targets[name] - have[name]
            flag = "" if gap > 0 else "  [already met]"
            logger.info("    %-26s %5d / %5d   %+d%s",
                        name, have[name], targets[name], gap, flag)
        logger.info("  Approximately %d new samples to generate.", shortfall)

        logger.info(
            "Scheduling %d famil%s to %d samples each, in %d round(s) of %d "
            "(round-robin, so an interrupted run stays balanced).",
            len(names), "y" if len(names) == 1 else "ies",
            count_per_family, rounds, chunk,
        )

        for r in range(1, rounds + 1):
            logger.info("\n%s\n= Round %d/%d — topping families up to %d\n%s",
                        "=" * 62, r, rounds, chunk * r, "=" * 62)
            for i, name in enumerate(names, start=1):
                target = min(targets[name], chunk * r)
                if len(self.by_family.get(name, ())) >= targets[name]:
                    # At its ceiling. Skipping is not just a speed win: re-running
                    # a saturated family re-derives draws already on disk, and the
                    # duplicate rejections make the gate's report misleading.
                    logger.info("> [round %d] %s: at target (%d), skipping.",
                                r, name, targets[name])
                    datasets[name] = list(self.by_family.get(name, ()))
                    continue
                logger.info("\n%s\n> [round %d] Family %d/%d: %s (-> %d)\n%s",
                            "-" * 62, r, i, len(names), name, target, "-" * 62)
                self._family_position = (i, len(names))
                datasets[name] = fmap[name](target)

            done = sum(len(v) for v in datasets.values())
            logger.info("= Round %d complete: %d samples across %d families.",
                        r, done, len(names))

        total = sum(len(v) for v in datasets.values())
        logger.info(
            "\nFinished: %d accepted across %d families, %d rejected by the gate.",
            total, len(names), self.stats["rejected"],
        )
        if self.gate.counts:
            logger.info("Rejection breakdown:\n%s", self.gate.report())
        return datasets

    def _pos(self) -> Tuple[int, int]:
        return getattr(self, "_family_position", (1, 13))

    # -- 1. single grounded tool call ------------------------------------------
    def gen_tool_calling(self, n: int) -> List[Dict[str, Any]]:
        """One natural request resolving to exactly one correct tool call."""
        family = "tool_calling"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=True)
            if not ctx:
                return None
            ctx["shape"] = "single_call"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            call = self.simulator.build_method_call(ctx)
            result = self.simulator.execute(call, ctx, force_success=True)

            sys_p, user_p = build_agent_turn_prompt(ctx, request, [call], [result])
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) + f"Calling `{call['name']}` on {ctx['spec'].model}.",
                 "tool_calls": [self._tool_call(call, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "single_call")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 2. resolve a human reference before acting ----------------------------
    def gen_lookup_then_act(self, n: int) -> List[Dict[str, Any]]:
        """The user names a document in human terms; the agent must find its id."""
        family = "lookup_then_act"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=True)
            if not ctx:
                return None
            ctx["shape"] = "lookup_then_act"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            search = self.simulator.build_search_call(ctx, by_reference=True)
            search_result = self.simulator.execute(search, ctx, force_success=True)
            act = self.simulator.build_method_call(ctx)
            act_result = self.simulator.execute(act, ctx, force_success=True)

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [search, act], [search_result, act_result])
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) +
                            f"Looking up {ctx['doc_ref']} to get its record id.",
                 "tool_calls": [self._tool_call(search, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(search_result["payload"], ensure_ascii=False)},
                {"role": "assistant",
                 "content": f"Found {ctx['doc_ref']} (id {ctx['res_id']}). "
                            f"Applying `{ctx['method'].name}`.",
                 "tool_calls": [self._tool_call(act, f"c{index}_1")]},
                {"role": "tool", "tool_call_id": f"c{index}_1",
                 "content": json.dumps(act_result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "lookup_then_act")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 17. discover the method, then call it ---------------------------------
    def gen_method_discovery(self, n: int) -> List[Dict[str, Any]]:
        """The agent lists the model's methods and picks one, instead of guessing.

        Every other family shows an agent that already knows the method, because
        the teacher was handed it from the knowledge graph. The consequence shows
        up only at inference: `odoo_methods_get` sits in the tool list but occurs
        zero times as a call in 21,902 samples, against 11,122 odoo_execute_method
        and 9,863 odoo_search_read. Never having seen it used, the model does not
        treat looking up as an option. Measured on the held-out split it emits a
        plausible *real* method (invented_method 1.5%) that is the *wrong* one
        (right_method 3.5%) - action_archive for "add products from the catalog" -
        or falls back to odoo_search_read and loops there.

        This family supplies the missing move. It also changes what method
        selection *is*: recall over a large surface becomes reading a returned
        list, which is the only version that generalises to methods this dataset
        never covers.

        Not filtered to hard-to-guess methods on purpose. A method_filter would
        confine the draw to the curated Tier A pool (see _draw_context), and the
        habit being taught - look before acting - is worth training on the whole
        surface rather than on a narrow slice of it.
        """
        family = "method_discovery"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=True)
            if not ctx:
                return None
            ctx["shape"] = "discover_then_act"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            probe = self.simulator.build_methods_get_call(ctx)
            probe_result = self.simulator.methods_get_result(ctx)
            act = self.simulator.build_method_call(ctx)
            act_result = self.simulator.execute(act, ctx, force_success=True)

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [probe, act], [probe_result, act_result],
                emphasis="The method was chosen from the list the tool returned, not "
                         "recalled from memory. State briefly which entry matched the "
                         "request and why, then report what the call did.")
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072,
                                          reasoning_budget=2048)
            if not answer:
                return None

            meth: MethodSpec = ctx["method"]
            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) +
                            f"Rather than guess a method name on {ctx['spec'].model}, "
                            f"I will list what it actually exposes.",
                 "tool_calls": [self._tool_call(probe, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(probe_result["payload"], ensure_ascii=False)},
                {"role": "assistant",
                 "content": f"`{meth.name}` is the entry that {meth.intent}. "
                            f"Calling it on {ctx['doc_ref']}.",
                 "tool_calls": [self._tool_call(act, f"c{index}_1")]},
                {"role": "tool", "tool_call_id": f"c{index}_1",
                 "content": json.dumps(act_result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "discover_then_act")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 18. answer from what you have, and stop -------------------------------
    def gen_answer_and_stop(self, n: int) -> List[Dict[str, Any]]:
        """Conclude from an imperfect result instead of querying again.

        The dataset already holds 11,199 read-then-answer trajectories, and they did
        not teach termination, because every one of them hands the agent a clean
        result that fully answers the question. Concluding is trivial there and the
        model learns nothing about *deciding* to conclude.

        The failure happens on the other cases. Measured on the finished 4B, asked
        "how many sales orders i have?", it issued a correct read_group on
        sale.order by state, got counts back, and then re-grouped by date_order,
        partner_id and invoice_status in turn — no answer, until the call budget ran
        out. Nothing was wrong with the first result. It simply never learned that a
        result which could be explored further is still a result worth answering.

        So this family is deliberately built on the awkward outcomes:

        * ``truncated``  - the read hit its limit. Answer with what came back and
          say it is capped, rather than re-running with a wider one.
        * ``empty``      - nothing matched. Say so. Do not retry with a different
          filter in the hope of better news.
        * ``sufficient`` - the result answers the question while other angles
          obviously exist. Answer anyway.

        In every shape the final turn is prose with no tool call, and the reply says
        plainly what is and is not covered. That last part matters: a model that
        stops but overstates what it found trades a loop for a lie.
        """
        family = "answer_and_stop"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=False)
            if not ctx:
                return None
            rng = ctx["rng"]
            shape = rng.choice(("truncated", "truncated", "empty", "sufficient"))
            ctx["shape"] = "answer_and_stop:%s" % shape

            read = self.simulator.build_search_call(ctx, by_reference=False)
            result = self.simulator.execute(read, ctx, force_success=True)
            ctx["data_hint"] = self.simulator.describe_query(read, ctx)

            request = self._write_user_request(ctx, rng)
            if not request:
                return None

            rows = result["payload"]
            if not isinstance(rows, list):
                return None

            # The tool content is shaped exactly as the runtime tool emits it - a
            # bare JSON array, then any note as plain text after it. Training on a
            # different shape is what made the model treat a real result as
            # unfamiliar and look again.
            if shape == "empty":
                content = "[]"
                emphasis = (
                    "Nothing matched. Say so directly, name the filter that was "
                    "applied so the user can see why, and offer to widen it. Do not "
                    "run another query and do not speculate about what might exist.")
            elif shape == "truncated":
                shown = rows[:5] or rows
                total = len(shown) + rng.randint(20, 400)
                content = (json.dumps(shown, ensure_ascii=False)
                           + "\nShowing %d of %d matches. Narrow the domain or use "
                             "offset to see more." % (len(shown), total))
                emphasis = (
                    "The read was capped. Answer from the rows that came back and "
                    "state plainly that it is the first %d of %d, so the user knows "
                    "the figure is partial. Do not re-run the query with a bigger "
                    "limit - report what you have." % (len(shown), total))
            else:
                content = json.dumps(rows, ensure_ascii=False)
                emphasis = (
                    "This answers the question. Give the figures and names from the "
                    "rows. Other groupings and filters are obviously possible - "
                    "mention at most one as an offer, do not go and run it.")

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [read], [result],
                emphasis=emphasis + " Close the turn: this reply is the answer, not "
                                    "a step towards one.")
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072,
                                          reasoning_budget=1792)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) +
                            f"Reading {ctx['spec'].model} to answer that.",
                 "tool_calls": [self._tool_call(read, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0", "content": content},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, ctx["shape"])

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 3. state machine transition -------------------------------------------
    def gen_workflow_execution(self, n: int) -> List[Dict[str, Any]]:
        """A lifecycle transition, grounded in the method's real from/to states."""
        family = "workflow_execution"

        def build(index: int) -> Optional[Dict[str, Any]]:
            # The model filter and the method filter must agree. Filtering models
            # that merely *contain* a from->to method, then drawing any mutating
            # method, discarded most draws before a single API call was made.
            def full_transition(m: MethodSpec) -> bool:
                return bool(m.from_state and m.to_state) and not m.returns_action

            ctx = self._draw_context(
                index, family,
                model_filter=lambda s: any(full_transition(m) for m in s.methods),
                method_filter=full_transition,
            )
            if not ctx:
                return None
            ctx["shape"] = "single_call"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            read = self.simulator.build_read_call(ctx, state=ctx["method"].from_state)
            read_result = self.simulator.execute(read, ctx, force_success=True)
            act = self.simulator.build_method_call(ctx)
            act_result = self.simulator.execute(act, ctx, force_success=True)

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [read, act], [read_result, act_result],
                emphasis="Explain the lifecycle transition in business terms: what state the "
                         "record was in, what the method changes, and what becomes possible next.")
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) +
                            "Checking the current state before I change anything.",
                 "tool_calls": [self._tool_call(read, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(read_result["payload"], ensure_ascii=False)},
                {"role": "assistant",
                 "content": f"It is in `{ctx['method'].from_state}`, so "
                            f"`{ctx['method'].name}` applies.",
                 "tool_calls": [self._tool_call(act, f"c{index}_1")]},
                {"role": "tool", "tool_call_id": f"c{index}_1",
                 "content": json.dumps(act_result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "workflow")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 4. full trajectory, some of which fail --------------------------------
    def gen_agent_trajectories(self, n: int) -> List[Dict[str, Any]]:
        """End-to-end work: search, create, act, verify — with real failure odds."""
        family = "agent_trajectories"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=True,
                                     model_filter=lambda s: bool(s.create_fields))
            if not ctx:
                return None
            ctx["shape"] = "multi_step"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            plan, results = self.simulator.build_trajectory(
                ctx, failure_rate=self.failure_rate)
            if not plan:
                return None

            sys_p, user_p = build_agent_turn_prompt(ctx, request, plan, results)
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=4096, reasoning_budget=2560)
            if not answer:
                return None

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ]
            for step, (call, result) in enumerate(zip(plan, results)):
                tc_id = f"c{index}_{step}"
                prefix = self._think(reasoning) if step == 0 else ""
                messages.append({
                    "role": "assistant",
                    "content": prefix + result["narration"],
                    "tool_calls": [self._tool_call(call, tc_id)],
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc_id,
                    "content": json.dumps(result["payload"], ensure_ascii=False),
                })
            messages.append({"role": "assistant", "content": answer})
            return self._finalise(messages, ctx, "multi_step")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 5. failure diagnosis and recovery -------------------------------------
    def gen_error_recovery(self, n: int) -> List[Dict[str, Any]]:
        """The call fails with an exception the method genuinely raises."""
        family = "error_recovery"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(
                index, family, mutating_only=True,
                model_filter=lambda s: any(m.failures for m in s.methods),
            )
            if not ctx or not ctx["method"].failures:
                return None
            ctx["shape"] = "recover"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            call = self.simulator.build_method_call(ctx)
            failure = self.simulator.execute(call, ctx, force_failure=True)

            sys_p, user_p = build_recovery_prompt(ctx, request, call, failure)
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=4608, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) + f"Running `{ctx['method'].name}`.",
                 "tool_calls": [self._tool_call(call, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(failure["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "recover")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 6. ask instead of guessing --------------------------------------------
    def gen_clarification(self, n: int) -> List[Dict[str, Any]]:
        """Under-specified request; the correct behaviour is a question, no call."""
        family = "clarification_dialogues"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=False)
            if not ctx:
                return None
            rng = ctx["rng"]
            gap, gap_desc = rng.choice(scen.UNDER_SPECIFICATIONS)
            ctx["shape"] = "clarify"
            ctx["gap"] = gap
            ctx["gap_desc"] = gap_desc

            request = self._write_user_request(ctx, rng)
            if not request:
                return None

            fields = self.kg.get_model_fields(ctx["spec"].model)
            required = [f["name"] for f in fields if f.get("required")][:10]

            sys_p, user_p = build_clarification_prompt(ctx, request, required)
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=2560, reasoning_budget=1536)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant", "content": self._think(reasoning) + answer},
            ]
            return self._finalise(messages, ctx, "clarify")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 7. confirm the mutation actually landed -------------------------------
    def gen_verification(self, n: int) -> List[Dict[str, Any]]:
        """Mutate, then re-read to prove the change committed before reporting."""
        family = "verification"

        def build(index: int) -> Optional[Dict[str, Any]]:
            # Verification is only meaningful when there is a state change to
            # verify, so restrict to methods that actually move the record.
            ctx = self._draw_context(
                index, family,
                model_filter=lambda s: any(m.to_state and not m.returns_action
                                           for m in s.methods),
                method_filter=lambda m: bool(m.to_state) and not m.returns_action,
            )
            if not ctx:
                return None
            ctx["shape"] = "verify"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            act = self.simulator.build_method_call(ctx)
            act_result = self.simulator.execute(act, ctx, force_success=True)
            check = self.simulator.build_read_call(
                ctx, state=ctx["method"].to_state or "posted")
            check_result = self.simulator.execute(check, ctx, force_success=True)

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [act, check], [act_result, check_result],
                emphasis="Close by stating what you verified and the field values that "
                         "prove it, not merely that the call returned true.")
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) + f"Applying `{ctx['method'].name}`.",
                 "tool_calls": [self._tool_call(act, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(act_result["payload"], ensure_ascii=False)},
                {"role": "assistant",
                 "content": "Re-reading the record to confirm the change committed.",
                 "tool_calls": [self._tool_call(check, f"c{index}_1")]},
                {"role": "tool", "tool_call_id": f"c{index}_1",
                 "content": json.dumps(check_result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "verify")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 8. coreference across turns -------------------------------------------
    def gen_multi_turn_memory(self, n: int) -> List[Dict[str, Any]]:
        """A follow-up turn refers to the earlier record as 'it' / 'that one'."""
        family = "multi_turn_memory"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=True,
                                     model_filter=lambda s: bool(s.create_fields))
            if not ctx:
                return None
            rng = ctx["rng"]
            ctx["shape"] = "coreference"

            first = self._write_user_request(ctx, rng)
            if not first:
                return None

            create = self.simulator.build_create_call(ctx)
            create_result = self.simulator.execute(create, ctx, force_success=True)

            follow_up = self._write_follow_up(ctx, first, rng)
            if not follow_up:
                return None

            act = self.simulator.build_method_call(ctx)
            act_result = self.simulator.execute(act, ctx, force_success=True)

            sys_p, user_p = build_agent_turn_prompt(
                ctx, follow_up, [act], [act_result],
                emphasis=f"This is a follow-up turn. '{follow_up}' refers to the "
                         f"{ctx['spec'].label} you created a moment ago, id "
                         f"{ctx['res_id']}. Make the resolution of that reference "
                         f"explicit and natural, not mechanical.")
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": first},
                {"role": "assistant",
                 "content": f"Creating the {ctx['spec'].label} now.",
                 "tool_calls": [self._tool_call(create, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(create_result["payload"], ensure_ascii=False)},
                {"role": "assistant",
                 "content": f"Done — {ctx['doc_ref']} is created (id {ctx['res_id']})."},
                {"role": "user", "content": follow_up},
                {"role": "assistant",
                 "content": self._think(reasoning) +
                            f"That refers to {ctx['doc_ref']}, id {ctx['res_id']}.",
                 "tool_calls": [self._tool_call(act, f"c{index}_1")]},
                {"role": "tool", "tool_call_id": f"c{index}_1",
                 "content": json.dumps(act_result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "coreference")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    def _write_follow_up(self, ctx: Dict[str, Any], first: str,
                         rng: random.Random) -> Optional[str]:
        """Asks the teacher for a natural second turn that uses a pronoun."""
        sys_p = (
            f"You are a {ctx['persona'].role} using a chat assistant connected to your "
            f"company's Odoo system. {ctx['persona'].style}"
        )
        user_p = (
            f"You just asked: \"{first}\"\n"
            f"The assistant created the {ctx['spec'].label} and confirmed it.\n\n"
            f"Write your next message. You now want to "
            f"{ctx['method'].intent}. Refer back to the record you just created "
            f"using a pronoun or a short phrase like 'it', 'that one', or "
            f"'the one you just made' — do not repeat its reference number, and "
            f"do not name any Odoo model or method.\n\n"
            f"Reply with the message text only. No quotes, no preamble."
        )
        _, text = self._ask(sys_p, user_p, max_tokens=512, reasoning_budget=256,
                            temperature=0.95)
        if not text:
            return None
        line = text.strip().strip('"').split("\n\n")[0].strip()
        return line if 6 <= len(line) <= 400 else None

    # -- 9. querying and filtering ---------------------------------------------
    def gen_data_retrieval(self, n: int) -> List[Dict[str, Any]]:
        """A business question that has to become a correct domain filter."""
        family = "business_data_retrieval"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=False)
            if not ctx:
                return None
            ctx["shape"] = "analysis"

            # Query first, then a question it can actually answer — same reason
            # as report_analysis above.
            search = self.simulator.build_search_call(ctx, by_reference=False)
            search_result = self.simulator.execute(search, ctx, force_success=True)
            ctx["data_hint"] = self.simulator.describe_query(search, ctx)

            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [search], [search_result],
                emphasis="Answer the business question from the rows returned. Give "
                         "figures and names, not a description of the query you ran. "
                         "Do not invent rows that are not in the result.")
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072, reasoning_budget=1792)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) + "Querying the records that match.",
                 "tool_calls": [self._tool_call(search, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(search_result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "analysis")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 10. aggregate and interpret -------------------------------------------
    def gen_report_analysis(self, n: int) -> List[Dict[str, Any]]:
        """read_group aggregation turned into an answer a manager can act on."""
        family = "report_analysis"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=False)
            if not ctx:
                return None
            ctx["shape"] = "analysis"

            # Build the query first, then ask for a question it can answer.
            # Drawing them independently made 63.8% of this family conclude
            # "this data does not answer your question".
            group = self.simulator.build_read_group_call(ctx)
            if not group:
                return None
            group_result = self.simulator.execute(group, ctx, force_success=True)
            ctx["data_hint"] = self.simulator.describe_query(group, ctx)

            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            sys_p, user_p = build_analysis_prompt(ctx, request, group, group_result)
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=4608, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) + "Aggregating the figures.",
                 "tool_calls": [self._tool_call(group, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(group_result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "analysis")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 11. explain the system, no tool call ----------------------------------
    def gen_consultant_qa(self, n: int) -> List[Dict[str, Any]]:
        """A real Odoo 19 question answered from the extracted schema."""
        family = "consultant_knowledge"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=False)
            if not ctx:
                return None
            ctx["shape"] = "explain"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            sys_p, user_p = build_explain_prompt(ctx, request, self.kg)
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=4096, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant", "content": self._think(reasoning) + answer},
            ]
            return self._finalise(messages, ctx, "explain")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 12. tool selection among alternatives ---------------------------------
    def gen_mcp_agent_protocol(self, n: int) -> List[Dict[str, Any]]:
        """Choosing the right primitive when several would superficially fit.

        Note the name: the previous ``generate_all_families`` called
        ``gen_family_12_mcp_agent`` while the method was defined as
        ``gen_family_12_mcp_agent_protocol``. The resulting AttributeError killed
        every run at family 12, which is why families 10-14 have zero rows in the
        legacy cache.
        """
        family = "mcp_agent_protocol"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family, mutating_only=True)
            if not ctx:
                return None
            ctx["shape"] = "single_call"
            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            call = self.simulator.build_method_call(ctx)
            result = self.simulator.execute(call, ctx, force_success=True)
            distractors = self.simulator.plausible_wrong_tools(ctx)

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [call], [result],
                emphasis="In your reasoning, say briefly why the primitive you chose is "
                         f"right and why these are not: {distractors}. Keep it to one or "
                         "two sentences; do not lecture.")
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=3072, reasoning_budget=2048)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) + f"Using `{call['name']}`.",
                 "tool_calls": [self._tool_call(call, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "tool_selection")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 13. stop and confirm before something irreversible --------------------
    def gen_refusal_and_confirmation(self, n: int) -> List[Dict[str, Any]]:
        """Consequential request: explain the impact and confirm, do not execute.

        Absent from the previous pipeline entirely. Without it, a model trained on
        this data will happily cancel a posted invoice or unreserve a picked
        delivery on a one-line instruction.
        """
        family = "refusal_and_confirmation"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(
                index, family,
                model_filter=lambda s: any(_is_destructive(m) for m in s.methods),
                method_filter=_is_destructive,
            )
            if not ctx:
                return None
            meth = ctx["method"]
            ctx["shape"] = "refuse_or_warn"

            request = self._write_user_request(ctx, ctx["rng"])
            if not request:
                return None

            read = self.simulator.build_read_call(ctx, state=meth.from_state or "posted")
            read_result = self.simulator.execute(read, ctx, force_success=True)

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [read], [read_result],
                emphasis=(
                    f"`{meth.name}` on this record is consequential and hard to undo. "
                    f"You have read the record but you must NOT call it yet. State "
                    f"plainly what would change, what downstream documents it affects, "
                    f"and ask for explicit confirmation. Be brief and concrete — one "
                    f"short paragraph plus the question. Do not moralise or list "
                    f"generic warnings."),
            )
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=2560, reasoning_budget=1536)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) +
                            "Reading the record before I touch anything.",
                 "tool_calls": [self._tool_call(read, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(read_result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "refuse_or_warn")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 15. creating a record from an incomplete request ----------------------
    def gen_record_creation(self, n: int) -> List[Dict[str, Any]]:
        """Create from a half-specified request: use defaults, ask only for what is
        genuinely required, and never invent a value.

        Added after watching an earlier build fail this exact task. Asked to
        "create a product for a 100 ml bottle" it produced a complete product it had
        memorised — ``CNC Milling Insert TNMG160408``, type service, price 3295.70,
        ``categ_id: 2`` — none of it from the user. Told to ask first, it swung the
        other way and demanded a warehouse (not a field on the model) and a unit of
        measure (which has a default), never creating anything.

        Both failures share a cause the other families do not cover.
        ``clarification_dialogues`` teaches asking when a *request* is ambiguous;
        this teaches the narrower and harder judgement inside a *create*: which
        fields the user must supply, which Odoo fills in by itself, and which are
        simply optional. The serving side computes that distinction and returns it
        (``missing_required``), so the behaviour to train is reading it and acting
        on it rather than guessing in either direction.

        Two shapes, drawn evenly:

        ``complete``  - everything required is present. Create it, mention what
                        Odoo defaulted, do not interrogate the user.
        ``incomplete`` - a required field is genuinely missing. Ask for exactly
                        that field, by name, and nothing else.
        """
        family = "record_creation"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family)
            if not ctx:
                return None
            rng = ctx["rng"]
            spec = ctx["spec"]
            ctx["shape"] = "record_creation"

            create = self.simulator.build_create_call(ctx)
            values = dict(create["arguments"].get("values") or {})
            if not values:
                return None

            # Split the fields the way the tool does at serving time: what the user
            # named, versus what Odoo would supply. Holding one back is what makes
            # the "ask for exactly this" case real rather than hypothetical.
            keys = list(values)
            incomplete = len(keys) > 1 and rng.random() < 0.5
            withheld = keys[-1] if incomplete else None
            if incomplete:
                create["arguments"]["values"] = {
                    k: v for k, v in values.items() if k != withheld
                }

            request = self._write_user_request(ctx, rng)
            if not request:
                return None

            if incomplete:
                result = {
                    "ok": True,
                    "payload": {
                        "created": [],
                        "confirmed": False,
                        "preview": [create["arguments"]["values"]],
                        "missing_required": [withheld],
                        "note": (
                            f"NOTHING WAS CREATED. This is a preview of the 1 "
                            f"{spec.model} record(s) you are proposing. Still "
                            f"required and not set: ['{withheld}']. Ask the user for "
                            f"these specific fields — nothing else."),
                    },
                }
                emphasis = (
                    f"The create returned a preview, not a record — nothing exists "
                    f"yet, so do not say it was created. One required field is "
                    f"missing: `{withheld}`. Ask the user for that field and only "
                    f"that field. Do not ask about optional fields, and do not "
                    f"invent a value for it. Two sentences.")
            else:
                result = {
                    "ok": True,
                    "payload": {
                        "created": [],
                        "confirmed": False,
                        "preview": [create["arguments"]["values"]],
                        "missing_required": [],
                        "note": (
                            f"NOTHING WAS CREATED. This is a preview of the 1 "
                            f"{spec.model} record(s) you are proposing. Nothing else "
                            f"is required. Odoo fills the remaining fields with its "
                            f"own defaults. Show these values to the user, flag any "
                            f"you guessed rather than were told, and when they "
                            f"approve call odoo_create again with the same values "
                            f"plus \"confirm\": true."),
                    },
                }
                emphasis = (
                    "The create returned a preview — nothing has been created yet, "
                    "so do not report it as done. Show the user the values you are "
                    "proposing and ask them to confirm. "
                    "Attribution must be exact: the ONLY things the user gave you "
                    "are what appears in their message above. Everything else you "
                    "filled in yourself — say so plainly for those, and do not "
                    "write 'you supplied' next to a value they never mentioned. "
                    "That mislabelling is the failure this is teaching against. "
                    "Nothing else is required, so do not ask about optional fields. "
                    "Brief.")

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, [create], [result], emphasis=emphasis)
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=2560,
                                          reasoning_budget=1536)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) +
                            f"Drafting the {spec.label.lower()} from what you gave me.",
                 "tool_calls": [self._tool_call(create, f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(result["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, "record_creation")

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 16. editing field data, and knowing when not to -----------------------
    def gen_record_update(self, n: int) -> List[Dict[str, Any]]:
        """Plain data edits with ``odoo_write`` — and refusing to use it for workflow.

        ``odoo_write`` had **zero** samples in the first two datasets. Not few:
        none, across 20,485 rows. ``build_write_call`` was written in the simulator
        and never called by any family, so one of the six primitives the agent is
        trained to expose was never once demonstrated.

        That leaves two holes. The model has no way to learn an ordinary correction
        — a changed address, a slipped date, a fixed reference — and, more subtly,
        it never sees the line it must not cross. Every other family teaches "call
        the business method"; without a counter-example the rule degenerates into
        "never write", and a request to fix a typo gets answered with a workflow
        method or a refusal.

        So the family draws both sides:

        ``edit``   - a field with no workflow attached to it. Write it.
        ``refuse`` - the user asks to set a state field directly. Explain that
                     writing it skips the logic Odoo runs, and call the method.
        """
        family = "record_update"

        def build(index: int) -> Optional[Dict[str, Any]]:
            ctx = self._draw_context(index, family)
            if not ctx:
                return None
            rng = ctx["rng"]
            spec = ctx["spec"]
            meth = ctx["method"]

            # The refuse half needs a state field to be asked about, and a method
            # that moves it; without both there is nothing to contrast.
            wants_refusal = bool(spec.state_field) and rng.random() < 0.4
            ctx["shape"] = "state_write_refusal" if wants_refusal else "record_update"

            # For the edit half the write is built *before* the request, and the
            # request is written from it. Generating the two independently produced
            # "the date should be 11 March" answered by a write of employee_id and
            # a different date entirely — a sample teaching the model to change
            # fields nobody asked about, which is the exact habit this family is
            # meant to remove. Same fix as report_analysis, where independent
            # generation left 63.8% of samples unanswerable.
            pending_write = None
            if not wants_refusal:
                pending_write = self.simulator.build_write_call(ctx)
                values = pending_write["arguments"].get("values") or {}
                if spec.state_field:
                    values.pop(spec.state_field, None)
                if not values:
                    return None
                pending_write["arguments"]["values"] = values
                # Describe the change without handing the raw values over.
                #
                # Passing them verbatim leaked database ids into the user's own
                # message — "the employee is changed to employee 10", "partner
                # should be the one with id 7875" — in about 5% of samples. Nobody
                # talks that way, and it quietly contradicts the rule the same
                # prompt states two paragraphs later: never mention a record id.
                # Worse, it teaches that ids are something a user supplies, when the
                # whole point of the relational guard is that the agent looks them
                # up.
                #
                # A many2one is therefore described by its role, not its number. The
                # teacher writes "assign it to a different supplier" and the model
                # still learns the write that follows; it just no longer learns that
                # people speak in primary keys.
                described = []
                for field_name, value in values.items():
                    if isinstance(value, int) and field_name.endswith("_id"):
                        described.append(
                            f"{field_name.removesuffix('_id').replace('_', ' ')} "
                            f"should be a different one (do not name a number)")
                    else:
                        described.append(f"{field_name} becomes {value!r}")
                ctx["data_hint"] = ", ".join(described)

            request = self._write_user_request(ctx, rng)
            if not request:
                return None

            if wants_refusal:
                # The model reads the record, then explains why a direct write is
                # wrong and calls the method instead. Two calls, one lesson.
                read = self.simulator.build_read_call(ctx, state=meth.from_state)
                read_result = self.simulator.execute(read, ctx, force_success=True)
                calls, results = [read], [read_result]
                emphasis = (
                    f"The user is asking you to set `{spec.state_field}` directly. "
                    f"Do not do that with odoo_write: writing a state field skips "
                    f"everything Odoo runs behind the transition — in this case "
                    f"`{meth.name}`. Say briefly why, in business terms rather than "
                    f"technical ones. Do not lecture; two or three sentences.\n"
                    f"TENSE MATTERS. The only call you have made is the read. You "
                    f"have NOT run `{meth.name}` — say what you are ABOUT to do, "
                    f"never what you have done. 'I will confirm it using "
                    f"`{meth.name}`' is right; 'I confirmed it using `{meth.name}`' "
                    f"is a false report of an action that never happened, and is the "
                    f"single worst thing an agent with database access can write.")
            else:
                write = pending_write
                values = write["arguments"]["values"]
                write_result = self.simulator.execute(write, ctx, force_success=True)
                calls, results = [write], [write_result]
                emphasis = (
                    f"This is an ordinary data correction — {', '.join(values)} — "
                    f"with no workflow attached, so odoo_write is exactly right. "
                    f"Confirm what you changed and on which record. Do not add "
                    f"caveats about business methods; they do not apply here.")

            sys_p, user_p = build_agent_turn_prompt(
                ctx, request, calls, results, emphasis=emphasis)
            reasoning, answer = self._ask(sys_p, user_p, max_tokens=2560,
                                          reasoning_budget=1536)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant",
                 "content": self._think(reasoning) +
                            ("Checking the record first."
                             if wants_refusal else
                             f"Updating {ctx['doc_ref']}."),
                 "tool_calls": [self._tool_call(calls[0], f"c{index}_0")]},
                {"role": "tool", "tool_call_id": f"c{index}_0",
                 "content": json.dumps(results[0]["payload"], ensure_ascii=False)},
                {"role": "assistant", "content": answer},
            ]
            return self._finalise(messages, ctx, ctx["shape"])

        i, t = self._pos()
        return self._run_family(family, n, build, i, t)

    # -- 14. schema and method recall ------------------------------------------
    def gen_schema_knowledge(self, n: int) -> List[Dict[str, Any]]:
        """Facts the agent cannot look up at run time, so must carry in weights.

        The MCP interface exposes generic primitives, not per-model tool schemas,
        so nothing at inference says which methods exist on ``account.move`` or
        which fields a domain may reference. This family installs that.

        Unlike the behavioural families it is deliberately repetitive per fact —
        recall wants the same fact from several angles, where policy learning
        wants breadth. Facts are weighted by tier so capacity lands on the twenty
        models an agent actually drives.
        """
        family = "schema_knowledge"
        rng = random.Random(f"{self.seed}|facts")
        facts = kpack.build_facts(self.surface, self.core, self.kg,
                                  self.kg.db_path, rng)
        if not facts:
            logger.warning("[%s] no facts could be extracted.", family)
            return []

        seen_counts: Dict[str, int] = {}
        plan: List[Tuple[kpack.Fact, int]] = []
        for fact in facts:
            k = fact.key()
            plan.append((fact, seen_counts.get(k, 0)))
            seen_counts[k] = seen_counts.get(k, 0) + 1
        rng.shuffle(plan)
        logger.info("[%s] %d fact instances from %d distinct facts across %d models.",
                    family, len(plan), len(seen_counts),
                    len({f.model for f, _ in plan}))

        def build(index: int) -> Optional[Dict[str, Any]]:
            if index >= len(plan):
                return None
            fact, variant = plan[index]
            question = kpack.question_for(fact, variant)
            sys_p, user_p = kpack.build_teacher_prompt(fact, question)

            # No <think> block on recall samples, deliberately.
            #
            # Two reasons. Pedagogically, you want the model to *know* that
            # sale.order has action_confirm, not to deliberate its way there;
            # training a reasoning trace onto a lookup teaches it to spend tokens
            # on facts it should have. Practically, the teacher rehearses whatever
            # style instruction it is given ("no preamble", "use exact names") in
            # its reasoning, which the gate then rejects as instruction_echo — a
            # loop no amount of prompt rewording escapes. Dropping the trace ends
            # it, and halves the tokens per sample.
            # A method_inventory answer lists every callable on a model, so it
            # is legitimately long — one was measured at 4,743 chars against a
            # 1,536-token cap and discarded as truncated despite being complete.
            # Budget for the longest fact type rather than the median.
            _, answer = self._ask(sys_p, user_p,
                                  max_tokens=3072, reasoning_budget=768)
            if not answer:
                return None

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            ctx = {
                "family": family,
                "index": index,
                "spec": self.surface[fact.model],
                "method": MethodSpec(
                    name=fact.payload.get("method", fact.kind),
                    intent=fact.kind, from_state=None, to_state=None),
                "persona": scen.Persona("ERP consultant", "", "normal", "chat"),
            }
            return self._finalise(messages, ctx, f"knowledge:{fact.kind}")

        i, t = self._pos()
        return self._run_family(family, min(n, len(plan)), build, i, t)

    # ──────────────────────────────────────────────────────────────────────
    # Message helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _think(reasoning: Optional[str]) -> str:
        return f"<think>\n{reasoning.strip()}\n</think>\n\n" if reasoning else ""

    @staticmethod
    def _tool_call(call: Dict[str, Any], tc_id: str) -> Dict[str, Any]:
        return {
            "id": tc_id,
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
            },
        }
