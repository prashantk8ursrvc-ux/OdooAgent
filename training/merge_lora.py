#!/usr/bin/env python3
"""
Merge a LoRA adapter into its base model one shard at a time.

Why this exists
---------------
``FastLanguageModel.from_pretrained(..., load_in_4bit=False)`` followed by
``save_pretrained_merged`` materialises the *entire* model in 16-bit before
writing anything. For this 4B that is ~8.3 GB, which fits on a 16 GB machine
with little to spare and does not fit on a smaller one - and it fails at the
very last step, after the training has already been paid for.

This merges tensor by tensor instead. Peak memory is one shard (~4 GB), so the
size of the model stops mattering.

It also merges into the **full-precision** base rather than the 4-bit one the
adapter was trained against. Merging into 4-bit weights would bake the
quantisation error permanently into the exported model; the adapter is trained
on the quantised base but applied to the clean one, which is the standard and
higher-fidelity path.

Usage
-----
    python training/merge_lora.py --config training/configs/qwen3_4b_topup.json
    python training/merge_lora.py --adapter path/to/adapter --out path/to/merged
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


#: Files that describe the model but are not weights. Copied verbatim from the
#: base so the merged directory is self-contained and loadable.
CONFIG_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
    "merges.txt", "added_tokens.json", "chat_template.jinja",
)


def full_precision_base(name: str) -> str:
    """Map a bnb-4bit repo id to its full-precision twin.

    Unsloth's 4-bit repos are named ``<model>-bnb-4bit``. The adapter records
    the quantised repo it trained against, but we want to merge into the clean
    weights.
    """
    return name[: -len("-bnb-4bit")] if name.endswith("-bnb-4bit") else name


def lora_key_to_base(key: str) -> str | None:
    """Translate a PEFT parameter name into the base model's tensor name.

    PEFT stores ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
    for a base tensor called ``model.layers.0.self_attn.q_proj.weight``.
    """
    if ".lora_A." not in key and ".lora_B." not in key:
        return None
    stem = key.split(".lora_")[0]
    for prefix in ("base_model.model.", "base_model."):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return f"{stem}.weight"


def collect_pairs(adapter_dir: Path) -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    """Load the adapter and pair up its A/B matrices per target tensor.

    Returns the factors, not the products. Expanding every B@A up front costs
    the full size of all targeted weights in fp32 - about 15 GB for the 4B, on a
    machine with 15.6 GB total. The factors are tiny by comparison (r=32), so
    they all fit and each delta is expanded only while its shard is open.
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]

    weights_path = adapter_dir / "adapter_model.safetensors"
    if weights_path.exists():
        state = load_file(str(weights_path))
    else:
        state = torch.load(adapter_dir / "adapter_model.bin", map_location="cpu")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        base_key = lora_key_to_base(key)
        if base_key is None:
            continue
        side = "A" if ".lora_A." in key else "B"
        pairs.setdefault(base_key, {})[side] = tensor

    for base_key, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            # A half-pair means the checkpoint is damaged. Better to say so than
            # to silently merge a partial adapter.
            raise SystemExit(f"adapter is missing the other half of {base_key}")
    return pairs, scale


def delta_for(ab: dict[str, torch.Tensor], scale: float) -> torch.Tensor:
    """Expand one LoRA pair into its dense weight delta.

    lora_A is (r, in), lora_B is (out, r) -> delta is (out, in), matching W.
    Computed in fp32: r is small but the product accumulates, and the base is
    bf16 whose 8-bit mantissa loses real precision here.
    """
    return (ab["B"].float() @ ab["A"].float()) * scale


def delta_numel(ab: dict[str, torch.Tensor]) -> int:
    """Element count of the delta this pair would expand to, without expanding it."""
    return int(ab["B"].shape[0]) * int(ab["A"].shape[1])


def merge_adapter(adapter_dir: Path, out_dir: Path, base: str | None = None) -> Path:
    """Merge ``adapter_dir`` into its base, writing 16-bit weights to ``out_dir``.

    Peak memory is one shard, so this is safe on a machine where
    the load-everything-then-save path is not.
    """
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_repo = base or full_precision_base(adapter_cfg["base_model_name_or_path"])

    pairs, scale = collect_pairs(adapter_dir)
    print(f"adapter   : {adapter_dir}")
    print(f"base      : {base_repo}")
    print(f"targets   : {len(pairs)} tensors, scale {scale:g}")

    from huggingface_hub import snapshot_download
    print("fetching base weights (cached after the first run)...")
    base_dir = Path(snapshot_download(base_repo, allow_patterns=[
        "*.safetensors", "*.safetensors.index.json", "*.json", "*.txt", "*.jinja",
    ]))

    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No safetensors in {base_dir}")

    # Qwen3-4B sets tie_word_embeddings=True, so the checkpoint has no
    # lm_head.weight at all - the output projection reuses embed_tokens.weight.
    # But target_modules includes lm_head, and embed_tokens is NOT targeted, so
    # during training the output side carried the LoRA delta while the input
    # lookup did not. The two were already untied in effect.
    #
    # Reproducing that means materialising lm_head.weight = embed_tokens + BA and
    # leaving embed_tokens alone. Adding the delta to the shared tensor instead
    # would corrupt the input embeddings, and dropping it would silently discard
    # the lm_head training that makes token 151657 (<tool_call>) emittable, which
    # is the whole reason lm_head is in target_modules.
    base_cfg_path = base_dir / "config.json"
    base_tied = False
    if base_cfg_path.exists():
        base_tied = bool(json.loads(
            base_cfg_path.read_text(encoding="utf-8")).get("tie_word_embeddings"))
    untie_lm_head = base_tied and "lm_head.weight" in pairs
    untied_into: str | None = None
    if untie_lm_head:
        print("  base ties lm_head to embed_tokens; untying so the "
              "lm_head delta survives the merge")

    out_dir.mkdir(parents=True, exist_ok=True)
    applied: set[str] = set()

    for shard in shards:
        tensors = load_file(str(shard))
        hits = 0
        for name in list(tensors):
            ab = pairs.get(name)
            if ab is None:
                continue
            tensor = tensors[name]
            delta = delta_for(ab, scale)
            if delta.shape != tensor.shape:
                raise SystemExit(
                    f"shape mismatch on {name}: base {tuple(tensor.shape)} "
                    f"vs delta {tuple(delta.shape)} — wrong base model?")
            tensors[name] = (tensor.float() + delta).to(tensor.dtype)
            del delta
            applied.add(name)
            hits += 1

        if untie_lm_head and "model.embed_tokens.weight" in tensors:
            embed = tensors["model.embed_tokens.weight"]
            delta = delta_for(pairs["lm_head.weight"], scale)
            if delta.shape != embed.shape:
                raise SystemExit(
                    f"shape mismatch untying lm_head: embed {tuple(embed.shape)} "
                    f"vs delta {tuple(delta.shape)} — wrong base model?")
            tensors["lm_head.weight"] = (embed.float() + delta).to(embed.dtype)
            del delta
            applied.add("lm_head.weight")
            untied_into = shard.name
            hits += 1

        # metadata={"format": "pt"} is required; transformers refuses to load a
        # safetensors file without it.
        save_file(tensors, str(out_dir / shard.name), metadata={"format": "pt"})
        print(f"  {shard.name}: {hits} merged, {len(tensors)} written")
        del tensors

    missed = set(pairs) - applied
    if missed:
        # Silence here would mean shipping a model with part of the fine-tune
        # quietly dropped, which looks like a bad training run rather than a bug.
        raise SystemExit(
            f"{len(missed)} adapter tensors matched no base weight, e.g. "
            f"{sorted(missed)[:3]} — the adapter and base do not correspond.")

    for name in CONFIG_FILES:
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    index = base_dir / "model.safetensors.index.json"
    if index.exists():
        index_data = json.loads(index.read_text(encoding="utf-8"))
        if untied_into:
            # The loader reads weight_map to find each tensor. A materialised
            # lm_head.weight that is not listed here is simply never loaded, and
            # the model silently falls back to the tied embedding.
            weight_map = index_data.setdefault("weight_map", {})
            weight_map["lm_head.weight"] = untied_into
            meta = index_data.setdefault("metadata", {})
            if "total_size" in meta:
                added = delta_numel(pairs["lm_head.weight"]) * 2  # bf16/fp16
                meta["total_size"] = int(meta["total_size"]) + added
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index_data, indent=2), encoding="utf-8")

    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        dirty = False
        # The 4-bit base carries a quantization_config; the merged output is 16-bit
        # and would be re-quantised on load if we kept it.
        if model_cfg.pop("quantization_config", None) is not None:
            print("  dropped quantization_config from config.json")
            dirty = True
        if untied_into and model_cfg.get("tie_word_embeddings"):
            # Leaving this True makes transformers and convert_hf_to_gguf.py
            # discard the lm_head tensor we just wrote and re-tie to embed_tokens,
            # throwing away the merge.
            model_cfg["tie_word_embeddings"] = False
            print("  set tie_word_embeddings=false (lm_head is now explicit)")
            dirty = True
        if dirty:
            cfg_path.write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")

    total = sum(p.stat().st_size for p in out_dir.glob("*.safetensors")) / 1024 ** 3
    print(f"merged    : {out_dir}  ({total:.2f} GB)")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Low-memory LoRA merge")
    ap.add_argument("--config", help="Training config; supplies adapter and output paths.")
    ap.add_argument("--adapter", help="Adapter directory (overrides --config).")
    ap.add_argument("--out", help="Output directory (overrides --config).")
    ap.add_argument("--base", help="Base repo id (default: from adapter_config.json).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        adapter_dir = Path(args.adapter or root / cfg["output_dir"] / "adapter")
        out_dir = Path(args.out or root / cfg["output_dir"] / "merged-16bit")
    else:
        if not (args.adapter and args.out):
            return ap.error("give --config, or both --adapter and --out")
        adapter_dir, out_dir = Path(args.adapter), Path(args.out)

    if not (adapter_dir / "adapter_config.json").exists():
        print(f"No adapter at {adapter_dir}. Train first.")
        return 1

    out = merge_adapter(adapter_dir, out_dir, args.base)
    print(f"\nConvert it with:\n"
          f"  python llama_cpp_tools/convert_hf_to_gguf.py {out} "
          f"--outfile model-f16.gguf --outtype f16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
