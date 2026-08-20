"""
Multi-Provider LLM Pool
=======================

Streams reasoning and answer tokens from every configured provider, key and
model, treating each combination as an independently schedulable endpoint.

Rate limits are enforced per account, so the way to go faster is more keys, not
more threads. Three NVIDIA keys over four models plus three OpenRouter keys is a
pool of eighteen-odd endpoints; a 429 on any one parks it briefly and routes the
next call elsewhere instead of sleeping.

Providers are identified by key prefix (``nvapi-`` / ``sk-or-``) rather than by
environment-variable name — a real .env was found with an NVIDIA key stored as
``OPENROUTER_API_KEY``, which would otherwise have failed every call with an
auth error that reads like a rate limit.
"""

import hashlib
import logging
import os
import random
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Force UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

#: Backoff schedule for HTTP 429. Doubles per attempt, jittered, capped.
_RATE_LIMIT_BASE_DELAY = 2.0
_MAX_BACKOFF = 45.0

#: Seconds to wait on a single completion before abandoning it.
#:
#: The openai SDK defaults to a 600-second read timeout. With a pool of shared
#: and free-tier endpoints, some requests simply hang — and at that default one
#: hung stream parks a worker for ten minutes. A run with six workers stalls
#: almost completely while the log still shows it "querying".
#:
#: A grounded answer that has not started arriving within this window is not
#: coming; abandoning it and trying another endpoint is strictly faster than
#: waiting. Override with LLM_TIMEOUT_SECONDS if an endpoint is legitimately slow.
_REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "75"))

#: Ceiling on the escalating cooldown applied to an endpoint that keeps timing
#: out. An hour effectively retires it for the run without hard-coding a
#: blocklist — if it recovers, it rejoins on its own.
_MAX_TIMEOUT_BACKOFF = 3600.0

# Handle jiter native binary extension load issues (e.g. Windows Application Control policy blocking jiter.pyd)
try:
    import jiter
    jiter.from_json("{}")
except Exception as _jiter_err:
    import json
    import types
    logger.info(f"Native jiter DLL load failed ({_jiter_err}); using pure-Python json fallback for jiter.")
    _jiter_fallback = types.ModuleType("jiter")
    def _from_json(s_or_b, **kw):
        if isinstance(s_or_b, (bytes, bytearray)):
            s_or_b = s_or_b.decode("utf-8")
        return json.loads(s_or_b)
    _jiter_fallback.from_json = _from_json
    sys.modules["jiter"] = _jiter_fallback

try:
    from openai import OpenAI
    # Pre-import streaming completions module to warm up streaming imports early
    import openai.lib.streaming.chat._completions
except ImportError:
    OpenAI = None


@dataclass
class Endpoint:
    """One (provider, key, model) triple the pool can send a request to.

    Rate limits are enforced per account, so two keys on the same provider are
    two independent budgets. Treating the endpoint rather than the model as the
    unit of scheduling is what lets a saturated key hand off to a fresh one.
    """

    provider: str          # "nvidia" | "openrouter"
    model: str
    key_name: str          # env var it came from, for logs — never the key itself
    client: Any            # openai.OpenAI bound to this provider and key

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}@{self.key_name}"


def _detect_provider(key: str) -> Optional[str]:
    """Identifies a provider from the key itself, not from its variable name.

    An audit of a real .env found ``OPENROUTER_API_KEY`` holding an ``nvapi-``
    key — a copy-paste of the NVIDIA one. Trusting the variable name would have
    sent it to OpenRouter and failed every call with an auth error that looks
    like a rate limit.
    """
    if key.startswith("nvapi-"):
        return "nvidia"
    if key.startswith("sk-or-"):
        return "openrouter"
    return None


def discover_keys(env: Optional[Dict[str, str]] = None) -> List[Tuple[str, str, str]]:
    """Finds every API key in the environment as ``(provider, key_name, key)``.

    Matches any variable containing ``API_KEY``, so ``NVIDIA_API_KEY``,
    ``NVIDIA_API_KEY2``, ``OPENROUTER_API_KEY1`` and so on are all picked up
    without needing to be enumerated. Duplicates are dropped: the same key under
    two names is one rate-limit budget, not two.
    """
    env = env if env is not None else dict(os.environ)
    found: List[Tuple[str, str, str]] = []
    seen: set = set()
    for name, value in sorted(env.items()):
        if "API_KEY" not in name.upper():
            continue
        value = (value or "").strip()
        if not value:
            continue
        provider = _detect_provider(value)
        if provider is None:
            continue
        digest = hashlib.sha256(value.encode()).hexdigest()
        if digest in seen:
            logger.info("%s duplicates another key; ignoring (same rate-limit "
                        "budget, no extra capacity).", name)
            continue
        seen.add(digest)
        found.append((provider, name, value))
    return found


class LLMPool:
    """Round-robins generation across every configured provider, key and model.

    Was ``NvidiaLLMClient``, a single key over four models. The name is kept as
    an alias because the generator imports it, but the pool now spans providers:
    three NVIDIA keys and three OpenRouter keys is eighteen-plus endpoints, and
    a 429 on any one of them routes to the next rather than sleeping.
    """

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    NVIDIA_MODELS = [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        # Dated snapshot, deliberately. The floating alias "deepseek-v4-flash"
        # reached end of life on 2026-08-07 and now returns 410 Gone, which took
        # down a run mid-flight. Pinned ids get retired too, but on a published
        # date rather than silently under you.
        "deepseek-ai/deepseek-v4-flash-0731",
        "nvidia/nemotron-3-nano-30b-a3b",
    ]

    #: OpenRouter free-tier models. Verified present on the account's model list
    #: at time of writing; override with OPENROUTER_MODEL_POOL if that changes.
    #: Free-tier keys can only reach ``:free`` variants, and those carry their
    #: own daily caps — so these widen the pool but do not replace NVIDIA.
    OPENROUTER_MODELS = [
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ]

    BASE_URLS = {"nvidia": NVIDIA_BASE_URL, "openrouter": OPENROUTER_BASE_URL}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra_keys: bool = True,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required. Run `pip install openai`.")

        pool_override = {
            "nvidia": _split_env("NVIDIA_MODEL_POOL"),
            "openrouter": _split_env("OPENROUTER_MODEL_POOL"),
        }
        models_for = {
            "nvidia": pool_override["nvidia"] or self.NVIDIA_MODELS,
            "openrouter": pool_override["openrouter"] or self.OPENROUTER_MODELS,
        }

        keys: List[Tuple[str, str, str]] = []
        if api_key:
            provider = _detect_provider(api_key) or "nvidia"
            keys.append((provider, "explicit", api_key))
        if extra_keys:
            for provider, name, value in discover_keys():
                if any(v == value for _, _, v in keys):
                    continue
                keys.append((provider, name, value))

        if not keys:
            raise ValueError(
                "No API key found. Add NVIDIA_API_KEY (nvapi-…) or "
                "OPENROUTER_API_KEY (sk-or-…) to odoo_agent_forge/.env. "
                "Any variable whose name contains API_KEY is picked up, so "
                "NVIDIA_API_KEY2, OPENROUTER_API_KEY1 and so on all work."
            )

        self.endpoints: List[Endpoint] = []
        clients: Dict[str, Any] = {}
        for provider, key_name, key in keys:
            url = base_url if (base_url and key_name == "explicit") \
                else self.BASE_URLS[provider]
            if key_name not in clients:
                clients[key_name] = OpenAI(base_url=url, api_key=key,
                                           timeout=_REQUEST_TIMEOUT,
                                           max_retries=0)
            for model in models_for[provider]:
                self.endpoints.append(
                    Endpoint(provider, model, key_name, clients[key_name]))

        by_provider: Dict[str, int] = {}
        for e in self.endpoints:
            by_provider[e.provider] = by_provider.get(e.provider, 0) + 1
        logger.info(
            "LLM pool: %d endpoints from %d keys (%s).",
            len(self.endpoints), len(keys),
            ", ".join(f"{k}: {v}" for k, v in sorted(by_provider.items())))

        self.pool_index = 0
        # An endpoint that just returned 429 will do so again a second later.
        # Parking it and moving on is what keeps workers busy: before this, a
        # pilot logged 428 rate-limit errors and finished 3 of 13 families.
        self._cooldown_until: Dict[str, float] = {}
        #: Consecutive timeouts per endpoint, so chronic offenders back off hard.
        #: Endpoints the provider has permanently retired (HTTP 410).
        self._retired: Set[str] = set()
        self._timeouts: Dict[str, int] = {}
        self._pool_lock = threading.Lock()
        #: label -> successful completions, so teacher mix can be audited.
        self.usage: Dict[str, int] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Endpoint scheduling
    # ──────────────────────────────────────────────────────────────────────

    def _available(self) -> List[Endpoint]:
        now = time.monotonic()
        with self._pool_lock:
            # Retired endpoints are gone for good (HTTP 410), so they are excluded
            # before the cooldown check — including from the "closest to expiry"
            # fallback, which would otherwise keep handing back a dead model when
            # everything else is busy.
            alive = [e for e in self.endpoints if e.label not in self._retired]
            if not alive:
                raise RuntimeError(
                    "Every endpoint in the pool has been retired by its provider. "
                    "Update NVIDIA_MODELS in llm_client.py, or set NVIDIA_MODEL_POOL "
                    "to a comma-separated list of live model ids.")
            free = [e for e in alive
                    if self._cooldown_until.get(e.label, 0.0) <= now]
            if free:
                return free
            # Everything is cooling down: return the one closest to expiry.
            return [min(alive,
                        key=lambda e: self._cooldown_until.get(e.label, 0.0))]

    def _mark_rate_limited(self, endpoint: Endpoint, seconds: float) -> None:
        with self._pool_lock:
            self._cooldown_until[endpoint.label] = time.monotonic() + seconds

    def _next_endpoint(self) -> Endpoint:
        free = self._available()
        with self._pool_lock:
            endpoint = free[self.pool_index % len(free)]
            self.pool_index += 1
        return endpoint

    def _extra_body(self, endpoint: Endpoint, reasoning_budget: int) -> Dict[str, Any]:
        """Provider-specific knobs.

        ``reasoning_budget`` and ``chat_template_kwargs`` are NVIDIA NIM
        extensions; OpenRouter rejects unknown fields, so it gets its own shape.
        """
        if endpoint.provider == "openrouter":
            return {"reasoning": {"max_tokens": reasoning_budget}}

        model = endpoint.model
        if "deepseek-v4-flash" in model:
            return {"chat_template_kwargs": {"thinking": True,
                                             "reasoning_effort": "high"}}
        if "120b" in model or "550b" in model:
            return {"chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": reasoning_budget}
        if "nano-30b" in model or "nemotron" in model or "deepseek" in model:
            return {"reasoning_budget": reasoning_budget}
        return {}

    # ──────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_with_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 6144,
        reasoning_budget: int = 4096,
        require_complete: bool = True,
        return_endpoint: bool = False,
    ):
        """Generates across the pool, failing over on rate limits and truncation.

        Returns ``(reasoning, answer)``, or ``(reasoning, answer, label)`` when
        ``return_endpoint`` is set — the label records which teacher produced a
        sample, so a weak provider can be audited or filtered out later without
        regenerating everything.

        Budget note
        -----------
        On these endpoints the reasoning trace and the answer share one token
        allowance. Setting both to the same number let a long think block consume
        it all and cut the answer mid-sentence — 4,327 cached samples ended that
        way before ``finish_reason`` was checked. Keep ``reasoning_budget``
        strictly below ``max_tokens``.
        """
        if reasoning_budget >= max_tokens:
            raise ValueError(
                f"reasoning_budget ({reasoning_budget}) must be below max_tokens "
                f"({max_tokens}); otherwise the answer has no budget left and "
                f"will be truncated.")

        time.sleep(0.15)
        # Cooldowns already steer traffic to live endpoints, so a long attempt
        # chain only means one doomed sample monopolises a worker. Cap it and
        # let the family loop retry with a fresh draw instead.
        attempts = min(len(self.endpoints), 8)
        last_truncated = None
        truncations = 0

        for attempt in range(attempts):
            endpoint = self._next_endpoint()
            if model and attempt == 0:
                endpoint = Endpoint(endpoint.provider, model,
                                    endpoint.key_name, endpoint.client)
            try:
                logger.info("Querying %s", endpoint.label)
                reasoning, content, finish_reason = self._execute_streaming_call(
                    endpoint, system_prompt, user_prompt, temperature, top_p,
                    max_tokens, reasoning_budget)

                if require_complete and finish_reason == "length":
                    truncations += 1
                    logger.warning(
                        "%s hit the %d-token ceiling (reasoning=%d chars, "
                        "answer=%d chars).", endpoint.label, max_tokens,
                        len(reasoning or ""), len(content or ""))
                    last_truncated = (reasoning, content)
                    # Truncation is a property of the prompt, not the endpoint:
                    # if one model runs past the ceiling on this input, the next
                    # almost certainly will too. Walking all eight endpoints to
                    # confirm that costs minutes per sample and starves the run —
                    # a family sat ten minutes without a single write this way.
                    # Two confirmations are enough; drop it and draw again.
                    if truncations >= 2:
                        logger.warning(
                            "Prompt truncated on %d endpoints; abandoning this "
                            "sample rather than polling the rest.", truncations)
                        break
                    continue

                with self._pool_lock:
                    self.usage[endpoint.label] = self.usage.get(endpoint.label, 0) + 1
                    # A success clears the timeout history: the endpoint is
                    # healthy again and should not carry an old penalty.
                    self._timeouts.pop(endpoint.label, None)
                if return_endpoint:
                    return reasoning, content, endpoint.label
                return reasoning, content

            except Exception as exc:
                # Two saturation signals: HTTP 429, and NVIDIA's
                # "ResourceExhausted: Worker local total request limit reached
                # (32/32)", which arrives as a plain APIError. Both mean this
                # endpoint has nothing right now, so both park it.
                msg = str(exc)

                # A model that has been retired is never coming back. Cooling it
                # down and retrying just cycles every worker through the same dead
                # endpoint until the run dies — which is what happened when
                # deepseek-v4-flash reached end of life mid-run and every worker
                # crashed on "410 Gone" rather than falling through to the models
                # that were still serving. Retire it for the process and carry on.
                retired = ("410" in msg and "Gone" in msg) or "end of life" in msg.lower()
                if retired:
                    self._retired.add(endpoint.label)
                    survivors = [e for e in self._available()
                                 if e.label not in self._retired]
                    logger.warning(
                        "%s retired by the provider (%s). Dropped from the pool; "
                        "%d endpoint(s) still serving.%s",
                        endpoint.label, msg.strip()[:120], len(survivors),
                        "" if survivors else
                        " No endpoints left — update the model pool in "
                        "llm_client.py or set NVIDIA_MODEL_POOL.")
                    if not survivors:
                        raise
                    continue

                saturated = ("429" in msg or "Too Many Requests" in msg
                             or "ResourceExhausted" in msg
                             or "request limit reached" in msg)
                timed_out = ("Timeout" in type(exc).__name__
                             or "timed out" in msg.lower())
                if saturated:
                    cooldown = min(_RATE_LIMIT_BASE_DELAY * (2 ** min(attempt, 5)),
                                   _MAX_BACKOFF)
                    cooldown += random.uniform(0.0, cooldown * 0.4)
                    self._mark_rate_limited(endpoint, cooldown)
                    remaining = [e for e in self._available() if e.label != endpoint.label]
                    delay = 0.0 if remaining else cooldown
                    logger.info("%s saturated; parked %.0fs. %s", endpoint.label,
                                cooldown,
                                f"{len(remaining)} endpoint(s) still serving."
                                if remaining else "Whole pool busy, waiting.")
                elif timed_out:
                    # Some models time out reliably rather than occasionally. A
                    # flat cooldown makes that expensive: the endpoint returns
                    # from a 30s nap, is chosen again, and burns another full
                    # timeout. Observed on deepseek-v4-flash — six timeouts in
                    # ninety seconds of log, ~450 worker-seconds for nothing.
                    #
                    # So escalate per consecutive timeout and effectively retire
                    # an endpoint that never succeeds.
                    with self._pool_lock:
                        strikes = self._timeouts.get(endpoint.label, 0) + 1
                        self._timeouts[endpoint.label] = strikes
                    cooldown = min(30.0 * (2 ** (strikes - 1)), _MAX_TIMEOUT_BACKOFF)
                    self._mark_rate_limited(endpoint, cooldown)
                    delay = 0.0
                    logger.info(
                        "%s timed out after %.0fs (strike %d); parked %.0fs.",
                        endpoint.label, _REQUEST_TIMEOUT, strikes, cooldown)
                elif "401" in msg or "invalid_api_key" in msg or "No auth" in msg:
                    # A bad key never recovers; park it for the whole run rather
                    # than retrying it on every call.
                    self._mark_rate_limited(endpoint, 86400)
                    delay = 0.0
                    logger.error("%s rejected the key (%s). Disabled for this run.",
                                 endpoint.label, msg[:120])
                else:
                    delay = 0.5 + random.uniform(0.0, 0.5)
                    logger.warning("Notice on %s: %s. Failing over.\n%s",
                                   endpoint.label, exc, traceback.format_exc())
                if delay:
                    time.sleep(delay)

        if last_truncated is not None:
            logger.error("Every endpoint truncated this prompt; dropping the "
                         "sample rather than caching a cut-off answer.")
        else:
            logger.error("No endpoint in the pool was available.")
        return (None, None, None) if return_endpoint else (None, None)

    def _execute_streaming_call(
        self,
        endpoint: Endpoint,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        reasoning_budget: int,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Returns ``(reasoning, content, finish_reason)``.

        ``finish_reason`` distinguishes an answer that finished from one the
        token ceiling cut off.
        """
        kwargs: Dict[str, Any] = {
            "model": endpoint.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        extra_body = self._extra_body(endpoint, reasoning_budget)
        if extra_body:
            kwargs["extra_body"] = extra_body

        completion = endpoint.client.chat.completions.create(**kwargs)

        full_reasoning: List[str] = []
        full_content: List[str] = []
        finish_reason = None

        for chunk in completion:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            # NVIDIA streams reasoning as `reasoning_content`; OpenRouter as
            # `reasoning`. Checking both keeps one code path for all providers.
            reasoning = (getattr(delta, "reasoning_content", None)
                         or getattr(delta, "reasoning", None))
            if reasoning:
                full_reasoning.append(reasoning)

            content = getattr(delta, "content", None)
            if content:
                full_content.append(content)

        reasoning_str = "".join(full_reasoning).strip() if full_reasoning else None
        content_str = "".join(full_content).strip() if full_content else None

        # Some endpoints omit finish_reason on the terminal chunk. An empty
        # answer beside a full reasoning trace is the same failure as "length".
        if finish_reason is None and reasoning_str and not content_str:
            finish_reason = "length"

        return reasoning_str, content_str, finish_reason

    def usage_report(self) -> str:
        """Completions per endpoint, so the teacher mix is visible after a run."""
        if not self.usage:
            return "  (no completions yet)"
        width = max(len(k) for k in self.usage)
        total = sum(self.usage.values())
        return "\n".join(
            f"  {k:<{width}}  {n:>6,}  ({100 * n / total:4.1f}%)"
            for k, n in sorted(self.usage.items(), key=lambda kv: -kv[1]))


def _split_env(name: str) -> List[str]:
    raw = (os.getenv(name) or "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


#: The generator imports this name; the pool replaces the single-key client.
NvidiaLLMClient = LLMPool


class LocalLLMClient:
    """
    Client for generating dataset samples using a local OpenAI-compatible LLM server.
    Compatible with: Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1),
    llama.cpp server (http://localhost:8080/v1), and any other OpenAI-API-compatible endpoint.

    Usage:
        # Ollama: ollama serve  (then pull a model: ollama pull qwen2.5:7b)
        client = LocalLLMClient(model="qwen2.5:7b")
        # LM Studio:
        client = LocalLLMClient(base_url="http://localhost:1234/v1", model="local-model")
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama default

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: Optional[str] = None,
        api_key: str = "local",  # Ollama/LM Studio don't need a real key
    ):
        if OpenAI is None:
            raise ImportError("openai package is required. Run `pip install openai`.")

        resolved_url = base_url or os.getenv("LOCAL_LLM_BASE_URL", self.DEFAULT_BASE_URL)
        resolved_model = model or os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")

        self.model = resolved_model
        self.client = OpenAI(base_url=resolved_url, api_key=api_key)
        logger.info(f"LocalLLMClient initialized: model={self.model} @ {resolved_url}")

    def generate_with_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates a response from the local LLM. Since local models don't have separate
        reasoning_content streams, we prompt them to produce <think>...</think> blocks
        inline, then parse them out to match the same (reasoning, content) tuple format
        as NvidiaLLMClient.
        """
        # Ask the local model to reason explicitly inline
        augmented_system = (
            system_prompt + "\n\n"
            "IMPORTANT: Before answering, write your step-by-step reasoning inside "
            "<think>...</think> XML tags. Then provide your final answer after the closing tag."
        )

        try:
            logger.info(f"Querying local LLM: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = response.choices[0].message.content or ""

            # Parse <think>...</think> from the response
            reasoning = None
            content = raw
            if "<think>" in raw and "</think>" in raw:
                think_start = raw.index("<think>") + len("<think>")
                think_end = raw.index("</think>")
                reasoning = raw[think_start:think_end].strip()
                content = raw[think_end + len("</think>"):].strip()

            # If no think tags, treat first 30% of response as "reasoning"
            if not reasoning and len(raw) > 100:
                split = len(raw) // 3
                reasoning = raw[:split].strip()
                content = raw[split:].strip()

            return reasoning or None, content or None

        except Exception as e:
            logger.warning(f"Local LLM call failed: {e}\n" + traceback.format_exc())
            return None, None
