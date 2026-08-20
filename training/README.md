# Fine-tuning the Odoo 19 agent — Qwen3-4B on an RTX 5050, 8 GB

| | |
|---|---|
| Base | `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit` (Apache 2.0) |
| Method | QLoRA, rank 32, over a 4-bit base |
| `max_seq_length` | 2048 |
| Reasoning traces | stripped |
| Introspection tools | advertised |
| Epochs | 2 |
| Steps | 2,610 — 20,875 samples at 16 per step |
| Train time (measured) | **~38 h**, at ~53 s/step |
| Quantised size | 2.7 GB at `q4_k_m` |

Then a second, short run — the top-up — continues that adapter for 1 epoch on a
focused 5,409-sample mix, about 5 h.

`torch 2.10.0+cu128` with `sm_120` in the arch list is what Blackwell needs.

---

## Two settings that look like preferences and are not

Both were forced by measurement. `qwen3_4b.json` carries the same reasoning in
its `_comment` block; this is the short version.

### `max_seq_length` is 2048, not 4096

4096 was tried. It **survives two steps and dies on the third**:

```
torch.AcceleratorError: CUDA error: out of memory
  in unsloth_zoo/gradient_checkpointing.py, resizing the offload buffer
```

The budget that predicted 4096 would fit counted the 4-bit weights, the adapter
and the `lm_head` logits tensor — and missed the gradient-checkpoint offload
buffers, which also scale with sequence length. Those consumed the margin.
Arithmetic is not a substitute for a smoke test, and **two steps is not a smoke
test**. Five is the minimum that catches this.

4096 was also slow where it ran: 240–268 s/step against 53, because gradient
offloading was shuttling to CPU RAM every step. That is ~180 h for two epochs.

Measured truncation at 2048 with reasoning stripped: **5.0%**.

### Reasoning traces are stripped

They are 42% of the tokens in this dataset. Dropping them roughly halves tokens
per sample, which cuts the work per step *and* the activation memory — speed and
headroom from one change. It is also what makes 2048 viable at 5% truncation
rather than the ~30% it would be with traces kept, and truncation removes the
**end** of a transcript, which is exactly where the multi-step families teach the
agent to verify its own work.

Flip `keep_reasoning` if you want to test the opposite, but raise the sequence
length in the same edit or you will silently truncate a third of the corpus.

---

## Three decisions that matter more than the hyperparameters

### Loss is masked to assistant turns only

`train.py` calls `train_on_responses_only`. Without it the model spends capacity
learning to predict *user messages and tool results* — text it will never be
asked to produce. Training completes and the loss curve looks fine either way,
which is why this is easy to omit and expensive to omit.

### The dataset is rendered with the base model's own chat template

`prepare_dataset.py` calls `tokenizer.apply_chat_template(..., tools=ODOO_TOOLS)`
rather than formatting tool calls by hand. Whatever Ollama applies at serving
time is then the same transformation, because it comes from the same tokenizer
config. Hand-rolling this is the classic silent failure of a tool-calling
fine-tune: the model learns one format and the server expects another.

### The Modelfile must render tool calls

A minimal Ollama template that renders only `role` and `content`:

```
TEMPLATE "{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>{{ end }}"
```

**discards `ToolCalls` entirely** and has no `.Tools` block. A model trained to
emit tool calls emits them and Ollama drops them, while the model never sees the
tool definitions it was trained with. The symptom is a model that seems dumber
than it was in training, for reasons no amount of retraining fixes.

`export_gguf.py` writes a template that renders `.Tools`, `.ToolCalls` and the
`tool` role in Qwen's native format. `smoke_test.py` exists to prove it: if no
case produces a tool call, the template is the first thing to check.

---

## Introspection tools

`introspection_tools` is on, which advertises `odoo_fields_get` and
`odoo_methods_get` to the model. 190 model+method pairs appear fewer than ten
times in the training data — too rare to be retained reliably at 4B — and a model
that cannot recall a method name will invent one. These let it look up instead of
guess.

**Only leave this on if your server actually implements those two tools.**
Training a model to call something that does not exist is worse than not having
it at all.

---

## When it runs out of memory

Cut in this order:

1. **`max_seq_length`** — attention memory grows with the *square* of sequence
   length, so this frees far more than anything else. 1536 is the next step down
   from 2048.
2. **`ce_loss_target_gb`** — 0.4 here. Lowering it chunks the cross-entropy over
   the 152k vocabulary into smaller pieces, trading speed for the peak that the
   `lm_head` logits tensor causes.
3. `gradient_accumulation_steps` up, `per_device_train_batch_size` stays at 1 —
   accumulation is free in memory terms, batch size is not.
4. **LoRA rank last.** It barely moves VRAM and directly costs quality, which
   makes it the most tempting and least useful knob.

**Never drop `lm_head` from `target_modules` to make it fit** — see below.

Close anything else using the GPU. On a laptop the display alone can hold several
hundred MB, which matters at this margin.

---

## The `lm_head` bug (found the expensive way)

A run that trains `q/k/v/o/gate/up/down` and **not** `lm_head` completes normally.
The loss falls, eval improves, the masking is correct, and the model produces
perfect tool-call JSON — with the opening `<tool_call>` tag replaced by a random
junk token. Parsed tool calls: **zero**.

The cause is that token 151657 sits at roughly the **0.2 percentile by norm** in
every Qwen base checked, among near-identical low-norm rows. The model's hidden
state says "emit the tag"; the frozen output row cannot express it, and the argmax
falls on a random neighbour — a different one each time.

Nothing during training reveals this. It is invisible until inference, and by then
the run is already paid for.

Two consequences for this configuration:

- `lm_head` is in `target_modules`, and `train.py` refuses to start if the
  assistant turns contain added tokens the LoRA cannot learn to emit
  (`_check_output_head_trainable`, override with `--allow-frozen-head`).
- Qwen3-4B ships with `tie_word_embeddings: true`, so `lm_head` shares its tensor
  with the input embeddings. **The merge must untie them** — `merge_lora.py` does,
  writing `lm_head.weight` as the embedding plus the trained delta. A merge that
  ignores the tie silently discards everything the run taught the output head, and
  you get the same zero-tool-call model at the very last step.

Repairing the row afterwards by averaging hidden states does **not** work: the
mean direction is generic, so scaling it enough to win at tool-call positions
makes that token the argmax at 29% of *all* positions. The discriminative version
of that repair is simply training the head, which is what the config does.

---

## Run it

```bash
# 1. Render the dataset for this base model
python training/prepare_dataset.py --config training/configs/qwen3_4b.json

# 2. Smoke-test the loop before committing ~38 h to it. Five steps, not two:
#    at a longer sequence length this configuration dies on the third.
python training/train.py --config training/configs/qwen3_4b.json --max-steps 5

# 3. Train for real
python training/train.py --config training/configs/qwen3_4b.json

# 4. Merge, quantise, write the Ollama Modelfile.
#    merge_lora.py is the low-memory alternative if this merge exhausts RAM.
python training/export_gguf.py --config training/configs/qwen3_4b.json

# 5. Install and check that tool calls survive
ollama create odoo19-agent-4b -f training/runs/odoo19-agent-4b/Modelfile
python training/smoke_test.py --model odoo19-agent-4b --introspection
```

If a run dies, `--resume` continues from the last checkpoint (every 250 steps).

### The top-up

```bash
python training/prepare_dataset.py --config training/configs/qwen3_4b_topup.json \
    --focus method_discovery,answer_and_stop --anchor 3000
python training/train.py --config training/configs/qwen3_4b_topup.json \
    --init-from training/runs/odoo19-agent-4b/adapter
```

`--init-from` is not optional: without it this trains a fresh adapter and the 38 h
already spent are gone. The learning rate drops to 2e-5 from 1e-4 for the same
reason — a from-scratch rate applied to a trained adapter overwrites what it
already knows. `--anchor` mixes in samples from the original distribution so the
focused run does not forget everything else.

---

## Where the tokens go

About **800 tokens of every sample is the tool-schema block** — the same constant
text repeated 20,875 times. Trimming the `description` strings in
`tools_schema.py` is the one lever that shortens every sample at once without
dropping any data, and it is worth reaching for before cutting epochs.

---

## Judging the result

Don't guess at whether it worked. `eval_scorecard.py` runs the model over the
held-out split and counts the things that actually break: right tool, right
model, invented method, invented field, malformed domain. Every held-out sample
already carries its correct answer, so no human judgement is involved.

```bash
ollama pull qwen3:4b        # the untrained baseline
python training/eval_scorecard.py --model odoo19-agent-4b --compare qwen3:4b
```

The measured comparison for this build, and where it loses to the untrained base,
is in the [root README](../README.md).
