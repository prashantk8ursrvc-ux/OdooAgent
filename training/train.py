#!/usr/bin/env python3
"""
LoRA / QLoRA fine-tune, sized for a single 8 GB card.

Two things here matter more than the hyperparameters.

**Loss is computed on assistant turns only.** Without that, the model spends its
capacity learning to predict the user's message and the tool results — text it
will never be asked to produce. The trainer masks everything up to each
assistant turn so gradients only flow through what the model must actually
generate. This is the single largest quality difference in an agent fine-tune,
and it is easy to omit because training "works" either way.

**VRAM is the binding constraint, and sequence length dominates it.** Attention
memory grows with the square of the sequence, so halving ``max_seq_length`` frees
far more than halving LoRA rank. If you hit OOM, cut the sequence first.

Usage
-----
    python training/train.py --config training/configs/qwen3_4b.json
    python training/train.py --config training/configs/qwen3_4b_topup.json
    python training/train.py --config ... --resume        # after a crash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _preimport_env() -> None:
    """Set Unsloth's env knobs before it is imported.

    ``unsloth_zoo.fused_losses.cross_entropy_loss`` reads
    ``UNSLOTH_CE_LOSS_TARGET_GB`` into a module-level constant at *import* time,
    so setting it inside main() has no effect at all — the run still aborts with
    "No or negligible GPU memory available for fused cross entropy". Hence this
    ugly early peek at --config before any Unsloth import happens.
    """
    cfg_path = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
        elif a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
    if not cfg_path or not Path(cfg_path).exists():
        return
    try:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except ValueError:
        return

    # Budget the fused cross-entropy explicitly instead of letting it guess from
    # free VRAM. Unsloth takes 50% of whatever mem_get_info reports free when the
    # loss runs; after the forward pass that is effectively nothing. Smaller target =
    # more chunks = slower but far lower peak. The loss is exact either way.
    if cfg.get("ce_loss_target_gb"):
        os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = str(cfg["ce_loss_target_gb"])
    # Reduces fragmentation, which matters a great deal at 8 GB.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_preimport_env()

# Unsloth patches transformers/peft and must be imported before them.
import unsloth  # noqa: F401,E402  isort:skip
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402


#: Where the assistant turn begins and ends, used to mask the loss.
#:
#: Deliberately WITHOUT a trailing newline. Qwen uses byte-level BPE, so the "\n"
#: after "assistant" merges with the first character of the reply into a single
#: token — and Unsloth's default token-level matching then never finds the
#: marker. The result is silent and total: every sample gets fully masked,
#: Unsloth drops all of them ("no response found"), and the trainer dies with
#: "num_samples=0" long after you have gone to bed.
#:
#: Omitting the newline means the newline itself is trained on, which is correct
#: anyway — the model should learn to emit it.
RESPONSE_MARKERS = {
    "chatml": ("<|im_start|>assistant", "<|im_start|>user"),
    "llama3": ("<|start_header_id|>assistant<|end_header_id|>",
               "<|start_header_id|>user<|end_header_id|>"),
}


def preflight(cfg: dict) -> None:
    """Fails early and legibly rather than 40 minutes into a run."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device. This venv had torch+cu128 with sm_120 support — if "
            "that changed, reinstall torch for your card before training.")

    props = torch.cuda.get_device_properties(0)
    cap = f"sm_{props.major}{props.minor}"
    vram = props.total_memory / 1024 ** 3
    print(f"gpu             : {props.name}  {cap}  {vram:.1f} GB")

    if cap not in torch.cuda.get_arch_list():
        raise SystemExit(
            f"This torch build has no kernels for {cap} (built for "
            f"{torch.cuda.get_arch_list()}). On Blackwell you need torch built "
            f"with CUDA 12.8 or later, otherwise every kernel launch fails with "
            f"'no kernel image is available'.")

    data_dir = ROOT / cfg["data_dir"]
    if not (data_dir / "train.jsonl").exists():
        raise SystemExit(
            f"No training data at {data_dir}. Run first:\n"
            f"  python training/prepare_dataset.py --config <your config>")

    if vram < 7.0:
        # The shipped config is measured on 8 GB with max_seq_length 2048. Below
        # that the sequence length is the first thing to cut, because attention
        # memory grows with its square - see training/README.md.
        print("  NOTE: under 7 GB is tight for this configuration. If it OOMs, "
              "drop max_seq_length to 1536 before changing anything else.")


def _check_output_head_trainable(cfg, tokenizer, dataset, sample_size=200):
    """Refuse to train if the model would be unable to emit its own special tokens.

    This exists because of a bug that cost a full 21-hour run.

    A LoRA over ``q/k/v/o/gate/up/down`` never touches ``lm_head``. For an ordinary
    fine-tune that is fine — every token the model needs already has a well-trained
    output row. For a *tool-calling* fine-tune it is not: ``<tool_call>`` is a rare
    added token whose row in a base model is close to its initialisation, norm 0.39
    against a 0.67 vocabulary mean, sitting in a cluster of near-identical low-norm
    rows.

    Loss still falls, the eval curve looks healthy, and the model learns to produce
    perfect tool-call JSON. But it cannot emit the tag that marks the JSON *as* a
    call, because the row it must match is degenerate. The argmax lands on an
    arbitrary neighbour — junk in a different language each time — and downstream
    nothing parses a single tool call. Nothing in the training logs hints at it;
    the failure only appears at inference.

    So: if the responses being trained on contain added tokens, ``lm_head`` must be
    in ``target_modules``. A rank-r adapter on it is cheap (about 10M parameters at
    a higher rank) next to discovering this after the run.
    """
    target_modules = cfg.get("target_modules") or []
    if "lm_head" in target_modules or "embed_tokens" in target_modules:
        return

    added = set(getattr(tokenizer, "added_tokens_decoder", {}) or {})
    if not added:
        return

    # Only tokens inside the *response* region matter — those are the ones the
    # model has to generate. An added token appearing solely in the prompt is read,
    # never produced, so its output row is irrelevant.
    needed = set()
    for row in range(min(sample_size, len(dataset))):
        item = dataset[row]
        for token, label in zip(item["input_ids"], item["labels"]):
            if label != -100 and token in added:
                needed.add(token)

    # The turn delimiter is unavoidable and every instruct model can already emit
    # it; flagging it would make this fire on every run and train people to ignore it.
    eos = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    needed -= {t for t in eos if t is not None}
    if not needed:
        return

    names = sorted(tokenizer.convert_ids_to_tokens(list(needed))[:8])
    raise SystemExit(
        f"\nRefusing to train: the assistant turns contain {len(needed)} added "
        f"token(s) that this LoRA cannot learn to emit.\n\n"
        f"  tokens        : {', '.join(names)}\n"
        f"  target_modules: {target_modules}\n\n"
        f"'lm_head' is missing, so the output rows for those tokens stay at their "
        f"base-model values.\nThe run will look fine — loss falls, eval improves — "
        f"and the finished model will be\nunable to emit them, which for "
        f"<tool_call> means zero parseable tool calls.\n\n"
        f"Fix: add \"lm_head\" to target_modules in your config.\n"
        f"Override only if you know these tokens do not need to be generated:\n"
        f"  --allow-frozen-head\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the Odoo 19 agent model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-frozen-head", action="store_true",
                    help="Skip the check that lm_head can emit the added tokens "
                         "appearing in assistant turns. Only correct if those "
                         "tokens never need to be generated.")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from the last checkpoint in output_dir.")
    ap.add_argument("--init-from", dest="init_from", default=None, metavar="ADAPTER",
                    help="Start from a finished adapter instead of the base model, "
                         "for a top-up run on new families. Use a lower "
                         "learning_rate than the original run.")
    ap.add_argument("--template", default="chatml", choices=list(RESPONSE_MARKERS),
                    help="Which markers delimit an assistant turn. Qwen is chatml.")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Cap steps for a smoke test, e.g. --max-steps 20.")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    preflight(cfg)

    if cfg.get("ce_loss_target_gb"):
        print(f"fused CE budget : {cfg['ce_loss_target_gb']} GB "
              f"(set before import; chunked, exact)")

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / cfg["data_dir"]

    print(f"base model      : {cfg['base_model']}")
    print(f"max_seq_length  : {cfg['max_seq_length']}")
    print(f"lora r/alpha    : {cfg['lora_r']} / {cfg['lora_alpha']}")
    eff = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    print(f"effective batch : {eff}")

    # --init-from continues a *finished* adapter on new data, which is not what
    # --resume does: that reloads a checkpoint of the same run, with the same
    # dataset and the same LR schedule, and refuses a changed mix.
    #
    # Loading the adapter directory instead of the base gives Unsloth the trained
    # weights as the starting point, so a top-up teaches the new families without
    # relearning the 21,902 samples already paid for. Pair it with a lower
    # learning_rate (2e-5 against the 1e-4 used from scratch) — this is a nudge, and
    # a from-scratch rate on a trained adapter will overwrite what it knows.
    #
    # Keep some of the original families in the mix. Training only on what is new
    # moves the adapter to fit only what is new; the old samples are the anchor.
    init_from = (args.init_from or "").strip()
    if init_from:
        adapter_dir = Path(init_from)
        if not (adapter_dir / "adapter_config.json").exists():
            raise SystemExit(f"--init-from: no adapter at {adapter_dir}")
        print(f"init from       : {adapter_dir} (continuing a trained adapter)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir) if init_from else cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,                       # let Unsloth pick bf16/fp16 for the card
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
    )

    # get_peft_model on an already-adapted model would wrap a second adapter around
    # the first. The loaded one is already trainable, so it is used as it is.
    if init_from:
        FastLanguageModel.for_training(model)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.0),
            target_modules=cfg["target_modules"],
            bias="none",
            # "unsloth" checkpointing trades a little speed for a large VRAM saving.
            # On 8 GB that trade is not optional.
            use_gradient_checkpointing="unsloth",
            random_state=cfg.get("seed", 20260317),
            use_rslora=False,
        )

    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"),
                            split="train")
    eval_path = data_dir / "eval.jsonl"
    eval_ds = (load_dataset("json", data_files=str(eval_path), split="train")
               if eval_path.exists() else None)

    # Eval runs at batch 1 and costs about seven minutes over the full 600-sample
    # split. At one eval per 250 steps across a 3-epoch run that is over an hour
    # of pure overhead, for a loss number whose only job is to show the curve is
    # still descending — a subsample tracks that just as well.
    #
    # The full split stays on disk untouched; it is what the real scorecard
    # scores, and that measures behaviour rather than loss.
    eval_cap = int(cfg.get("eval_max_samples", 200))
    full_eval = len(eval_ds) if eval_ds else 0
    if eval_ds is not None and len(eval_ds) > eval_cap:
        eval_ds = eval_ds.select(range(eval_cap))
    print(f"train / eval    : {len(train_ds)} / "
          f"{len(eval_ds) if eval_ds else 0}"
          f"{f' (subsampled from {full_eval})' if full_eval > eval_cap else ''}")

    # TRL 0.24 moved dataset_text_field / max_length / packing onto SFTConfig and
    # renamed `tokenizer` to `processing_class`. SFTTrainer swallows unknown
    # kwargs, so passing them the old way is accepted silently and ignored —
    # leaving SFTConfig.max_length at its default of **1024**.
    #
    # That default truncates every sample partway through the ~900-token tools
    # block, so no assistant turn survives, response-masking finds nothing, and
    # all 14,416 samples are dropped with the opaque message "no response found
    # after truncation". Set it explicitly, on the config.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=cfg["max_seq_length"],
            packing=False,                # packing blurs turn boundaries and
                                          # breaks the response-only masking
            output_dir=str(out_dir),
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["num_train_epochs"],
            max_steps=args.max_steps,
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg.get("warmup_steps", 20),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
            weight_decay=cfg.get("weight_decay", 0.01),
            optim=cfg.get("optim", "adamw_8bit"),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=cfg.get("logging_steps", 10),
            save_steps=cfg.get("save_steps", 250),
            save_total_limit=2,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=cfg.get("eval_steps", 250),
            per_device_eval_batch_size=1,
            report_to="none",
            seed=cfg.get("seed", 20260317),
        ),
    )

    # Mask everything that is not an assistant turn. Without this the model
    # spends capacity predicting user messages and tool results, which it will
    # never be asked to generate.
    start, end = RESPONSE_MARKERS[args.template]
    trainer = train_on_responses_only(trainer,
                                      instruction_part=end,
                                      response_part=start)

    # Verify the masking actually found assistant turns.
    #
    # When the markers do not match, Unsloth masks every token, silently drops
    # every sample, and the failure surfaces much later as an opaque
    # "num_samples=0". Checking here costs a second and turns a wasted night into
    # an immediate, explicable error.
    n_train = len(trainer.train_dataset)
    if n_train == 0:
        raise SystemExit(
            f"Response masking removed every sample.\n\n"
            f"The markers for --template {args.template} did not match the "
            f"rendered text:\n"
            f"  response_part    = {start!r}\n"
            f"  instruction_part = {end!r}\n\n"
            f"Inspect what the template actually produced:\n"
            f"  python -c \"import json;print(json.loads(open("
            f"r'{data_dir / 'train.jsonl'}',encoding='utf-8').readline())['text'][:600])\"\n"
            f"and set RESPONSE_MARKERS to the exact assistant/user delimiters it "
            f"uses. Avoid a trailing newline in the marker: byte-level BPE merges "
            f"it with the following character, so it will never match.")
    if n_train < 0.9 * len(train_ds):
        print(f"  WARNING: masking kept only {n_train} of {len(train_ds)} samples. "
              f"The rest had no assistant turn inside max_seq_length — raise it, "
              f"or the model trains mostly on truncated prompts.")
    unmasked = sum(1 for x in trainer.train_dataset[0]["labels"] if x != -100)
    total = len(trainer.train_dataset[0]["labels"])
    print(f"loss masking    : {n_train} samples kept; "
          f"{unmasked}/{total} tokens trained on in sample 0 "
          f"({100*unmasked/max(1,total):.0f}%)")

    if not args.allow_frozen_head:
        _check_output_head_trainable(cfg, tokenizer, trainer.train_dataset)

    if torch.cuda.is_available():
        # Hand PyTorch's reserved-but-unused blocks back to the driver before
        # training starts.
        #
        # Unsloth's fused cross entropy sizes its chunks from
        # torch.cuda.mem_get_info(), which reports what the *driver* sees free.
        # PyTorch's caching allocator holds on to everything it has ever
        # allocated, so after loading the model that reads as ~0 free even when
        # gigabytes are reclaimable — and Unsloth aborts at step 0 with "No or
        # negligible GPU memory available for fused cross entropy", which sounds
        # like a sequence-length problem and is not.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        print(f"free VRAM       : {free/1024**3:.2f} GB of {total/1024**3:.2f} GB "
              f"before training")
        if free / 1024 ** 3 < 0.6:
            print("  WARNING: under 0.6 GB free. The fused loss needs headroom "
                  "proportional to vocab size (152k here), so this will likely "
                  "abort. Reduce max_seq_length, or use a smaller base.")
        torch.cuda.reset_peak_memory_stats()

    trainer.train(resume_from_checkpoint=args.resume or None)

    if torch.cuda.is_available():
        print(f"\npeak VRAM       : "
              f"{torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"adapter saved   : {adapter_dir}")
    print("\nNext:\n"
          f"  python training/export_gguf.py --config {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
