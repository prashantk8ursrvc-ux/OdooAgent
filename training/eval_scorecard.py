#!/usr/bin/env python3
"""Score a packaged model on the held-out split. Six counters, no human judgement.

Why this exists
---------------
Every quality question on this project has so far been answered with an anecdote —
"it invented a product name", "it asked which database" — and anecdotes cannot tell
you whether a change helped. Worse, the one number that *was* available, eval loss,
went on improving through a run that produced a model unable to emit a tool call at
all. Loss measured the thing that was working and said nothing about the thing that
was broken.

The held-out samples already carry their correct answer, and the agent surface
knows which models, fields and methods exist, so the scoring needs no judge:

  right_method       the gold turn called a method and the model called that one
  invented_method    it called a method that does not exist on the model
  invented_field     it referenced a field the model does not have
  asked_when_unsure  the gold turn asked a question and so did the model
  held_destructive   it did not fire an irreversible call unprompted
  recovered_error    after a tool error, it tried something different

Two of these are inverted on purpose. `invented_method` and `invented_field` are
failures, so lower is better; the rest are successes. Mixing directions in one
table invites misreading, so the report labels each.

Usage
-----
    ollama serve
    python training/eval_scorecard.py --model odoo19-agent-4b
    python training/eval_scorecard.py --model odoo19-agent-4b-topup --limit 200
    python training/eval_scorecard.py --model a --compare b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.tools_schema import get_tools  # noqa: E402

#: Calls that cannot be undone. Firing one without being asked is the single
#: failure that costs a user real data, so it is counted separately from
#: correctness rather than averaged into it.
DESTRUCTIVE = {"odoo_unlink"}
DESTRUCTIVE_METHODS = re.compile(
    r"unlink|cancel|delete|remove|archive|reset", re.IGNORECASE)


def load_surface():
    """Real model/field/method names, for checking what the model invents."""
    try:
        from odoo_agent_forge.agent_surface import verify_against_kg
        from odoo_agent_forge.knowledge_graph import OdooKnowledgeGraph
        kg = OdooKnowledgeGraph(str(ROOT / "forge_knowledge.db"))
        surface, __ = verify_against_kg(kg)
    except Exception as exc:  # noqa: BLE001 - scoring must work without the KG
        print(f"note: agent surface unavailable ({exc}); "
              f"invented_* counters will be skipped.")
        return None
    return {
        spec.model: (
            {m.name for m in spec.methods},
            set(spec.search_fields) | {f for f in getattr(spec, "fields", ())},
        )
        for spec in surface.values()
    }


def first_exchange(messages):
    """Split a sample into (prompt messages, the gold assistant turn).

    Scoring stops at the model's *first* action. Later turns depend on tool
    results that only exist because the gold trajectory took the gold path, so a
    model that legitimately chose a different first step would be marked wrong for
    every turn after it — measuring divergence from one transcript rather than
    correctness.
    """
    prompt = []
    for message in messages:
        if message.get("role") == "assistant":
            return prompt, message
        prompt.append(message)
    return prompt, None


def ask(host, model, messages, tools, timeout):
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer scorecard"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"]


def recover_calls(message, known_names):
    """Structured tool calls, falling back to JSON the model wrote as text.

    A model whose lm_head was never trained cannot emit `<tool_call>`, so its
    calls arrive as plain text. Scoring only the structured field would report 0%
    for a model whose arguments are in fact correct, and would credit the fix to
    training when it belonged to the serving layer.
    """
    calls = []
    for call in (message.get("tool_calls") or []):
        function = call.get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
        calls.append((function.get("name"), arguments or {}))
    if calls:
        return calls, True

    content = message.get("content") or ""
    for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)".*?\}\s*\}', content, re.S):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if parsed.get("name") in known_names:
            calls.append((parsed["name"], parsed.get("arguments") or {}))
    return calls, False


def score_one(gold_turn, reply, surface, known_names, counters):
    gold_calls, __ = recover_calls(gold_turn, known_names)
    got_calls, structured = recover_calls(reply, known_names)

    counters["samples"] += 1
    if structured:
        counters["emitted_structured_tool_call"] += 1
    elif got_calls:
        counters["tool_call_recovered_from_text"] += 1

    gold_wants_call = bool(gold_calls)

    if not gold_wants_call:
        # The gold turn answered or asked rather than acting. Matching that is the
        # behaviour, not a missing call.
        if not got_calls:
            counters["asked_when_unsure"] += 1
        return

    if not got_calls:
        counters["no_call_when_one_was_needed"] += 1
        return

    gold_name, gold_args = gold_calls[0]
    got_name, got_args = got_calls[0]

    if gold_name == "odoo_execute_method" and got_name == "odoo_execute_method":
        if gold_args.get("method") == got_args.get("method"):
            counters["right_method"] += 1
    elif gold_name == got_name:
        counters["right_tool"] += 1

    model_name = got_args.get("model")
    if surface and model_name in surface:
        methods, fields = surface[model_name]
        method = got_args.get("method")
        if method and method not in methods:
            counters["invented_method"] += 1
        referenced = set(got_args.get("fields") or [])
        referenced |= set((got_args.get("values") or {}).keys())
        unknown = {f for f in referenced if f not in fields}
        if unknown:
            counters["invented_field"] += 1

    if got_name in DESTRUCTIVE or (
            got_name == "odoo_execute_method"
            and DESTRUCTIVE_METHODS.search(str(got_args.get("method") or ""))):
        # Only a failure when the gold turn did not do it too.
        if gold_name not in DESTRUCTIVE and not (
                gold_name == "odoo_execute_method"
                and DESTRUCTIVE_METHODS.search(str(gold_args.get("method") or ""))):
            counters["fired_destructive_unprompted"] += 1
        else:
            counters["held_destructive"] += 1


def report(name, counters):
    total = max(counters["samples"], 1)

    def pct(key):
        return f"{counters[key]:>5} ({counters[key] / total * 100:5.1f}%)"

    print(f"\n{'=' * 62}\n{name}   {counters['samples']} samples scored\n{'=' * 62}")
    print("  higher is better")
    for key in ("right_method", "right_tool", "asked_when_unsure",
                "held_destructive", "emitted_structured_tool_call"):
        print(f"    {key:<32} {pct(key)}")
    print("  lower is better")
    for key in ("invented_method", "invented_field", "no_call_when_one_was_needed",
                "fired_destructive_unprompted", "tool_call_recovered_from_text"):
        print(f"    {key:<32} {pct(key)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Mechanical eval over the held-out split")
    ap.add_argument("--model", required=True)
    ap.add_argument("--compare", default=None, help="Second model, scored alongside.")
    ap.add_argument("--eval", default="forge_outputs/odoo19_agent_eval.jsonl")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--introspection", action="store_true")
    args = ap.parse_args()

    path = ROOT / args.eval
    if not path.exists():
        print(f"No eval split at {path}. Run the forge export first.")
        return 1

    tools = [{"type": "function", "function": t.get("function", t)}
             for t in get_tools(with_introspection=args.introspection)]
    known_names = {t["function"]["name"] for t in tools}
    surface = load_surface()

    samples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            messages = json.loads(line).get("messages") or []
            prompt, gold = first_exchange(messages)
            if gold and prompt:
                samples.append((prompt, gold))
            if len(samples) >= args.limit:
                break
    print(f"scoring {len(samples)} held-out samples")

    for model in filter(None, (args.model, args.compare)):
        counters = Counter()
        for index, (prompt, gold) in enumerate(samples, start=1):
            try:
                reply = ask(args.host, model, prompt, tools, args.timeout)
            except urllib.error.URLError as exc:
                print(f"cannot reach Ollama at {args.host}: {exc}")
                return 1
            except Exception as exc:  # noqa: BLE001 - one bad sample must not stop the run
                counters["errors"] += 1
                if counters["errors"] <= 3:
                    print(f"  sample {index} failed: {exc}")
                continue
            score_one(gold, reply, surface, known_names, counters)
            if index % 25 == 0:
                print(f"  {model}: {index}/{len(samples)}")
        report(model, counters)

    print("\nThese are counts, not a grade. The two that decide whether the model is "
          "usable are\nright_method and invented_method; the rest explain why.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
