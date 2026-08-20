#!/usr/bin/env python3
"""
Turn the generation cache into train/eval files for a specific base model.

Why this is model-specific
--------------------------
Tool calls have to be rendered into text the way the base model expects. Qwen
wraps them in ``<tool_call>`` blocks; Llama uses a different convention. Getting
this wrong is the classic silent failure of a tool-calling fine-tune: training
looks fine, loss goes down, and at inference the model emits a format the server
cannot parse.

So rather than invent a format, this renders with the *base model's own chat
template* (``tokenizer.apply_chat_template``). Whatever Ollama or vLLM applies at
serving time is then the same transformation, because it comes from the same
tokenizer config.

The tool definitions are passed in too, so the model is trained with the tool
block in context exactly as the MCP server will present it.

Usage
-----
    python training/prepare_dataset.py --config training/configs/qwen3_4b.json
    python training/prepare_dataset.py --config training/configs/qwen3_4b_topup.json

Both configs read the same cache; they differ in whether reasoning traces are
kept and how long a sample may be.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from odoo_agent_forge.quality import (  # noqa: E402
    QualityGate,
    load_methods_index,
    scrub_think_blocks,
    strip_think,
)
from odoo_agent_forge.knowledge_graph import OdooKnowledgeGraph  # noqa: E402
from training.tools_schema import get_tools  # noqa: E402


def load_cache(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue          # a live run may be mid-write on the last line
    return rows


def to_chat_messages(sample: dict, keep_think: bool) -> list:
    """Normalises a cached sample into plain chat messages.

    Tool calls are passed through in the OpenAI shape the chat template expects:
    ``{"type": "function", "function": {"name": ..., "arguments": {...}}}`` with
    arguments as a dict rather than a JSON string, which is what
    ``apply_chat_template`` wants.
    """
    out = []
    for m in sample["messages"]:
        role = m["role"]
        content = m.get("content") or ""
        if not keep_think:
            content = strip_think(content).strip()

        msg = {"role": role, "content": content}

        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc["function"]
                args = fn["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass
                calls.append({"type": "function",
                              "function": {"name": fn["name"], "arguments": args}})
            msg["tool_calls"] = calls
            # An assistant turn that is purely a tool call may have empty text;
            # the template handles that, but None breaks it.
            msg["content"] = content or ""
        out.append(msg)
    return out


def group_key(sample: dict) -> str:
    """Splits are grouped by model+method so eval cannot leak.

    Two samples about ``sale.order.action_confirm`` differ mainly in their
    document reference. Splitting them across train and eval lets the model score
    well on eval by having memorised a pattern rather than learned the skill.
    """
    meta = sample.get("_meta") or {}
    return f"{meta.get('model','?')}::{meta.get('method','?')}"


def _focus_mix(rows, focus, anchor, seed):
    """Everything from the named families, plus an even slice of the others.

    A top-up trained only on what is new fits only what is new: the adapter has no
    way to know the earlier behaviour still matters, so it drifts off it. The anchor
    rows teach nothing themselves — they keep the gradient pointing at the Odoo
    behaviour 21,902 samples already paid for while the new families teach the new
    one.

    Drawn evenly per family rather than at random across the pool, so the anchor
    keeps the shape of the original dataset instead of over-weighting whichever
    family happens to be largest.
    """
    wanted = {name.strip() for name in focus.split(",") if name.strip()}
    by_family = collections.defaultdict(list)
    for row in rows:
        by_family[(row.get("_meta") or {}).get("family", "?")].append(row)

    missing = wanted - set(by_family)
    if missing:
        raise SystemExit(
            "--focus: nothing cached for %s. Generate it first:\n"
            "  python run_forge_ultra.py --families %s"
            % (sorted(missing), ",".join(sorted(missing))))

    focused = [r for name in wanted for r in by_family[name]]
    others = sorted(set(by_family) - wanted)
    per_family = max(1, anchor // max(len(others), 1)) if others else 0

    rng = random.Random(seed)
    kept = []
    for name in others:
        pool = list(by_family[name])
        rng.shuffle(pool)
        kept.extend(pool[:per_family])

    print("focus families  : %s -> %d samples" % (sorted(wanted), len(focused)))
    print("anchor          : %d from %d other families (~%d each)"
          % (len(kept), len(others), per_family))
    mixed = focused + kept
    rng.shuffle(mixed)
    return mixed



def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare SFT files for a base model")
    ap.add_argument("--config", required=True, help="training/configs/*.json")
    ap.add_argument("--cache", default=str(ROOT / "forge_outputs" / "generation_cache_v2.jsonl"))
    ap.add_argument("--db", default=str(ROOT / "forge_knowledge.db"))
    ap.add_argument("--focus", default=None, metavar="FAM,FAM",
                    help="Top-up mix: take every sample from these families, plus "
                         "--anchor drawn evenly from the rest. Without it the whole "
                         "cache is used, as before.")
    ap.add_argument("--anchor", type=int, default=3000,
                    help="How many older samples to carry into a --focus mix. They "
                         "teach nothing new; they stop the adapter drifting off the "
                         "behaviour it already has while it learns the new families.")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["data_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_think = bool(cfg.get("keep_reasoning", True))
    max_chars = int(cfg.get("max_sample_chars", 24000))
    tools = get_tools(with_introspection=bool(cfg.get("introspection_tools", False)))

    print(f"config          : {args.config}")
    print(f"base model      : {cfg['base_model']}")
    print(f"keep reasoning  : {keep_think}")
    print(f"tools in prompt : {len(tools)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tok.chat_template is None:
        print("\nERROR: this tokenizer has no chat template, so tool calls cannot be "
              "rendered consistently with serving. Pick an instruct/chat base model.")
        return 1

    rows = load_cache(Path(args.cache))
    print(f"\ncache rows      : {len(rows)}")

    if args.focus:
        rows = _focus_mix(rows, args.focus, args.anchor,
                          int(cfg.get("seed", 20260317)))

    # Same gate the generator uses, so training data meets the standard the
    # pipeline enforces. Scrub first: it repairs rehearsal in reasoning traces
    # rather than discarding an otherwise sound sample.
    kg = OdooKnowledgeGraph(db_path=args.db)
    gate = QualityGate(methods_by_model=load_methods_index(kg))

    kept, dropped = [], collections.Counter()
    for r in rows:
        scrub_think_blocks(r["messages"])
        verdict = gate.check({"messages": r["messages"]})
        if not verdict.ok:
            for reason in verdict.reasons:
                dropped[reason] += 1
            continue
        kept.append(r)
    print(f"passed the gate : {len(kept)}")
    if dropped:
        print(f"dropped         : {dict(dropped.most_common(5))}")

    # -- render ---------------------------------------------------------------
    rendered, too_long, render_fail = [], 0, 0
    for r in kept:
        messages = to_chat_messages(r, keep_think)
        try:
            text = tok.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False)
        except Exception:
            render_fail += 1
            continue
        if len(text) > max_chars:
            too_long += 1
            continue
        rendered.append({"text": text, "_meta": r.get("_meta", {})})

    print(f"rendered        : {len(rendered)}  "
          f"(dropped {too_long} over {max_chars} chars, {render_fail} template errors)")
    if not rendered:
        print("\nNothing to write. Check the base model name and the cache path.")
        return 1

    # -- leak-free split ------------------------------------------------------
    groups = collections.defaultdict(list)
    for r in rendered:
        groups[group_key(r)].append(r)
    keys = sorted(groups)
    random.Random(cfg.get("seed", 20260317)).shuffle(keys)

    target = int(len(rendered) * float(cfg.get("eval_fraction", 0.04)))
    eval_rows, eval_keys = [], set()
    for k in keys:
        if len(eval_rows) >= target:
            break
        eval_rows.extend(groups[k])
        eval_keys.add(k)
    train_rows = [r for k in keys if k not in eval_keys for r in groups[k]]

    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    for path, rowset in ((train_path, train_rows), (eval_path, eval_rows)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rowset:
                fh.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")

    # Sidecar keeps provenance out of the token stream but available for eval.
    with open(out_dir / "eval_meta.jsonl", "w", encoding="utf-8") as fh:
        for r in eval_rows:
            fh.write(json.dumps(r["_meta"], ensure_ascii=False) + "\n")

    lengths = sorted(len(tok(r["text"]).input_ids) for r in rendered)
    p50 = lengths[len(lengths) // 2]
    p95 = lengths[int(0.95 * len(lengths))]
    print(f"\ntrain           : {len(train_rows)} -> {train_path}")
    print(f"eval            : {len(eval_rows)} -> {eval_path}")
    print(f"token length    : p50={p50}  p95={p95}  max={lengths[-1]}")

    seq = int(cfg["max_seq_length"])
    over = sum(1 for x in lengths if x > seq)
    print(f"max_seq_length  : {seq}  ({over} samples, {100*over/len(lengths):.1f}%, "
          f"will be truncated)")
    if over / len(lengths) > 0.05:
        print("  WARNING: truncating more than 5% of samples cuts the tail off "
              "multi-step transcripts, which is where the trajectory families "
              "teach their lesson. Consider raising max_seq_length, or setting "
              "keep_reasoning=false to halve the token count.")

    print("\n--- one rendered sample (first 900 chars) ---")
    print(train_rows[0]["text"][:900])
    return 0


if __name__ == "__main__":
    sys.exit(main())
