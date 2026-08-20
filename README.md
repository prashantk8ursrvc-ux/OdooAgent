# Odoo 19 Agent — fine-tuning a 4B model for tool calling

How to take a stock 4B instruct model and teach it to operate an Odoo 19 database
through tool calls: generating the dataset, training on a single 8 GB consumer
GPU, and packaging the result for Ollama.

Everything needed to reproduce it is here — the synthetic dataset generator, the
training configuration, the scripts, and the datasets themselves. The end result
runs entirely locally: no API keys at inference time and no per-token cost.

---

## Does the fine-tuning actually help?

Measured, not asserted. 11 questions × 3 runs against each model, identical system
prompt and identical tool schemas, scoring the **raw model output**:

| model | called a tool | right tool | right Odoo model | valid domain | grounded | s/call |
|---|---|---|---|---|---|---|
| `qwen3:4b` (thinking) | 55% | 55% | 55% | 100% | 100% | 13.6 |
| `qwen3:4b` (`/no_think`) | 45% | 45% | 42% | 100% | 100% | 13.4 |
| `odoo19-agent-4b` | 100% | 82% | 100% | 100% | 67% | 1.8 |
| **`odoo19-agent-4b-topup`** | **100%** | **91%** | **100%** | **100%** | 76% | **1.9** |

Read that in both directions.

**What the training bought.** Stock Qwen3-4B does not act — in about half the cases
it answers in prose and calls no tool at all. It also picks the wrong Odoo model
most of the time, and choosing `sale.order` over `res.partner` for an order
question is the single hardest thing in this task. The fine-tune calls a tool every
time, picks the right model in 100% of cases, and is **7× faster** because it does
not need a reasoning pass.

**What the training cost.** Grounding fell from 100% to 76%. Asked to
`create a product for haier microwave at cost of 50$`, stock Qwen3 answers
`{"name": "haier microwave", "cost": 50, "sales_price": 80}` correctly every time;
the fine-tune substitutes a product name it memorised from the synthetic
catalogue. See [Known weaknesses](#known-weaknesses) — that is a fixable
data-generation mistake, not something inherent to fine-tuning.

Caveat: 33 samples per model, one question set. 100% vs 45% is far beyond noise;
82% vs 91% is three cases and should not be leaned on.

---

## Hardware this was built on

| | |
|---|---|
| GPU | NVIDIA RTX 5050, 8 GB VRAM |
| OS | Windows 11 |
| CUDA | 12.8 |
| Training time | ~40 h for the base run, ~5 h for the top-up |

8 GB is the binding constraint throughout, and most of the non-obvious choices in
`training/configs/*.json` exist because of it. Those files carry long `_comment`
blocks explaining each one — read them before changing anything, because several
settings that look like preferences are not.

The most important: **`lm_head` must stay in `target_modules`.** Token 151657
(`<tool_call>`) sits at roughly the 0.2 percentile by norm in every Qwen base
checked. Without training the output head, the model learns the JSON perfectly and
then cannot emit the opening tag — 0% parseable tool calls, measured. It is the
difference between a working run and forty wasted hours.

---

## Layout

```
odoo_agent_forge/        dataset generation: scenarios, simulator, prompts, quality gates
run_forge_ultra.py       entry point for generation
forge_outputs/           the generated corpus (Git LFS)
training/
  configs/*.json         qwen3_4b.json (base run) and qwen3_4b_topup.json,
                         heavily commented - read them before changing anything
  prepare_dataset.py     cache -> train/eval jsonl for a given base model
  train.py               QLoRA fine-tune via unsloth
  merge_lora.py          adapter + base -> fp16 merged model
  export_gguf.py         merged -> GGUF -> Ollama Modelfile
  eval_scorecard.py      mechanical scoring over the held-out split
  smoke_test.py          is the packaged model sane?
  data/                  prepared datasets (Git LFS)
  runs/*/Modelfile       what turns a GGUF into an Ollama model
docs/PIPELINE.md         design notes from the generation phase
```

### Dataset sizes

Only the three files the pipeline actually reads are committed. The generator
also writes an SFT view, a sample index and a rejected-sample log; those are
outputs, nothing consumes them, and they are left out rather than carried around.

| file | rows | read by |
|---|---|---|
| `forge_outputs/generation_cache_v2.jsonl` | 24,566 | `prepare_dataset.py` — the source of truth |
| `forge_outputs/odoo_schema_knowledge_base.jsonl` | 4,531 | `run_forge_ultra.py --legacy-kb`, merged into generation |
| `forge_outputs/odoo19_agent_eval.jsonl` | — | `eval_scorecard.py` |
| `training/data/qwen3_4b/train.jsonl` | 20,875 | prepared for the base run |
| `training/data/qwen3_4b/eval.jsonl` | 912 | held out |
| `training/data/qwen3_4b_topup/train.jsonl` | 5,409 | focused mix for the top-up |
| `training/data/qwen3_4b_topup/eval.jsonl` | 230 | held out |

`train.jsonl` is **derived from the cache** by `prepare_dataset.py`, not the other
way round. The split is seeded (`seed` in the config), so re-running prepare
reproduces it exactly. If you change how samples are rendered, re-run prepare; do
not hand-edit the prepared files.

---

## The pipeline at a glance

Seven steps, in order. Step 6 leaves you with a working model and step 7 makes it
the better one; nothing before that can be skipped or reordered.

```
  generation_cache_v2.jsonl        committed, 24,566 samples
            │
            │  step 3   prepare_dataset.py
            ▼
  train.jsonl / eval.jsonl         20,875 / 912
            │
            │  step 4   train.py                      ~38 h
            ▼
  adapter/                         LoRA weights, ~1 GB
            │
            │  step 5   export_gguf.py                ~20 min
            ▼
  merged-16bit/ → gguf/ → Modelfile
            │
            │  step 5   ollama create
            ▼
  odoo19-agent-4b                  2.7 GB, runnable
            │
            │  step 7   the top-up                    ~5 h
            ▼
  odoo19-agent-4b-topup            the recommended build
```

---

## Step 1 — Install the prerequisites

| | |
|---|---|
| Python | 3.11 |
| GPU | NVIDIA with 8 GB VRAM or more, and a CUDA install matching your torch build |
| [Ollama](https://ollama.com/download) | runs the packaged model |
| [Git LFS](https://git-lfs.com) | the datasets are stored with it |
| Disk | ~30 GB free: ~10 GB base model, ~1 GB adapter, ~8 GB merged, ~9 GB GGUF |

Training runs on the GPU; everything else is CPU only.

```bash
pip install -r requirements-training.txt
```

Install that into **its own virtual environment**. Unsloth pins torch closely and
will fight anything else sharing the venv. `torch` must match your CUDA — see
[pytorch.org](https://pytorch.org) for the right index URL.

---

## Step 2 — Get the code and the data

```bash
git lfs install
git clone <this repo>
cd OdooAgent
```

`git lfs install` first, not after. The datasets are LFS objects; clone without it
and you get small text pointer files instead, and step 3 fails in a way that does
not mention LFS.

**Check it worked** — this should print a number in the tens of thousands, not 3:

```bash
wc -l forge_outputs/generation_cache_v2.jsonl     # expect 24566
```

---

## Step 3 — Prepare the training files

```bash
python training/prepare_dataset.py --config training/configs/qwen3_4b.json
```

Reads the generation cache, applies the config's `keep_reasoning`,
`max_sample_chars` and `introspection_tools` settings, renders each sample with
the base model's own chat template, and splits off `eval_fraction`.

**Produces** `training/data/qwen3_4b/train.jsonl` (20,875 rows) and `eval.jsonl`
(912). The split is seeded, so this is reproducible byte-for-byte.

Takes a couple of minutes. Those files are already committed, so this step
mainly matters when you change what goes into them.

---

## Step 4 — Train

Smoke-test first. **Five steps, not two** — at a longer sequence length this
configuration survives two steps and dies on the third, when a batch of longer
samples arrives:

```bash
python training/train.py --config training/configs/qwen3_4b.json --max-steps 5
```

If those five pass, commit to the run:

```bash
python training/train.py --config training/configs/qwen3_4b.json
```

**~38 hours** on an RTX 5050 — 2,610 steps at ~53 s/step, 2 epochs at 16 samples
per step. Checkpoints land every 250 steps, and `--resume` continues from the
newest one after a crash or a power cut.

**Produces** `training/runs/odoo19-agent-4b/adapter/`, about 1 GB.

---

## Step 5 — Package it for Ollama

```bash
python training/export_gguf.py --config training/configs/qwen3_4b.json
ollama create odoo19-agent-4b -f training/runs/odoo19-agent-4b/Modelfile
```

One command does the merge: `export_gguf.py` merges the adapter into the
**full-precision** base, converts to GGUF, and writes the Modelfile. Merging into
the 4-bit base instead would bake the quantisation error permanently into the
exported model.

That merge holds the whole 16-bit model in RAM at once (~8.3 GB here). If it dies
— a larger model, or under ~16 GB of RAM — `merge_lora.py` does the same merge one
shard at a time, peaking at about 4 GB:

```bash
python training/merge_lora.py --config training/configs/qwen3_4b.json
```

It is an alternative to the merge inside `export_gguf.py`, **not a step before
it**. It also unties `lm_head`: Qwen3 ships with `tie_word_embeddings: true`, and
a merge that ignores that silently discards everything the run taught the output
head — the one thing that must not be lost.

**Produces** a 2.7 GB `odoo19-agent-4b` in Ollama at `q4_k_m`.

---

## Step 6 — Check that it actually works

```bash
python training/smoke_test.py --model odoo19-agent-4b --introspection
```

The thing to look for is **parsed tool calls**. A fine-tune can score well on
every loss curve and still emit zero of them — see the `lm_head` section in
[`training/README.md`](training/README.md). If no case produces a tool call, the
Modelfile template is the first thing to check, not the training.

Then score it against the untrained base over the held-out split:

```bash
ollama pull qwen3:4b        # the baseline to compare against
python training/eval_scorecard.py --model odoo19-agent-4b --compare qwen3:4b
```

You now have a working model. Stop here if that is all you wanted.

---

## Step 7 — The top-up (recommended)

**This is the build the benchmark at the top of this file recommends.** A second,
short run — about 5 hours — that continues the first adapter on a focused mix
rather than starting over. It moved right-tool from 82% to 91% and grounding from
67% to 76%.

```bash
python training/prepare_dataset.py --config training/configs/qwen3_4b_topup.json \
    --focus method_discovery,answer_and_stop --anchor 3000

python training/train.py --config training/configs/qwen3_4b_topup.json \
    --init-from training/runs/odoo19-agent-4b/adapter

python training/export_gguf.py --config training/configs/qwen3_4b_topup.json
ollama create odoo19-agent-4b-topup -f training/runs/odoo19-agent-4b-topup/Modelfile
```

`--init-from` is **not optional**. Without it you train a fresh adapter and the 38
hours from step 4 are gone. The learning rate drops to 2e-5 from 1e-4 for the same
reason: a from-scratch rate applied to a trained adapter overwrites what it
already knows.

`--anchor` mixes samples from the original distribution back in, so the focused
run does not forget everything else.

---

## Optional — regenerate the dataset

Only needed if you want to change *what* the model learns. The corpus is
committed, so steps 1–7 never require this.

The generator reads an Odoo source tree to extract the real model and field
surface, and calls a hosted LLM to write the natural-language halves — so unlike
every other step, this one needs an API key:

```bash
pip install -r requirements.txt
cp odoo_agent_forge/.env.example odoo_agent_forge/.env   # then add your key
export ODOO_SOURCE=/path/to/odoo            # or pass --codebase
python run_forge_ultra.py --samples-per-family 200 --use-nvidia-llm --workers 3
```

Output appends to `forge_outputs/generation_cache_v2.jsonl`. Generation is
resumable — it skips what it already has, so an interrupted run costs nothing.

`--families` restricts to specific generators, `--family-target NAME=N` sets
per-family counts, and `--failure-rate` controls how often a scenario contains a
tool error the model must recover from (0.22 by default — a model that has only
ever seen happy paths falls apart on the first failed call).

Then go back to **step 3**, which re-renders the prepared files from the new cache.

---

## Using the model

The packaged model is an ordinary Ollama model exposing an OpenAI-compatible
endpoint, so any tool-calling client can drive it:

```bash
ollama run odoo19-agent-4b-topup
# or POST to http://localhost:11434/v1/chat/completions with a `tools` array
```

The tool names it was trained to emit are fixed, and your server must expose the
same ones for the training to transfer:

| tool | purpose |
|---|---|
| `odoo_search_read` | find and read records |
| `odoo_read_group` | aggregate |
| `odoo_create` / `odoo_write` | create and update |
| `odoo_execute_method` | call a business method |
| `odoo_unlink` | delete |
| `odoo_fields_get` / `odoo_methods_get` | introspection |

`training/tools_schema.py` holds the exact schemas used during training.

**Two things to expect when wiring it up**, both cheap to handle and expensive to
discover:

- If your tool schema declares a domain as an array whose `items` are strings —
  which some validators force — a grammar-constrained runtime cannot emit a
  `[field, operator, value]` triplet through it. The model looks broken and is not.
  Send `items: {"type": "array"}` on the wire.
- A 4B model will happily fill an argument with a plausible value nobody asked
  for. Take names, prices and references from the user's own message rather than
  trusting the model's; see [Known weaknesses](#known-weaknesses).

---

## Where to read more

This README is the route through. Two documents go deeper, and both are worth
reading before you change anything:

| | |
|---|---|
| [`training/README.md`](training/README.md) | why the training is set up the way it is — loss masking, the chat template, what to do when it runs out of memory, and the `lm_head` bug found the expensive way |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | how the dataset is generated — the scenario families, the simulator, the quality gates, and how to run generation for a subset |

---

## Known weaknesses

Stated plainly, because they are the most useful part of this repository.

**The model invents product names.** Grounded 76% against stock Qwen3's 100%. It
substitutes names memorised from the synthetic catalogue — "Acoustic Wall Panel
60x60" in answer to a request about a microwave. The real fix is in generation:
**every target value must appear verbatim in the user turn of the same sample**,
and the names used in create scenarios should come from a pool disjoint from the
catalogue used elsewhere. As written, the model learns "product names come from
the catalogue I memorised" rather than "from the request".

**Named operations get a spurious lookup.** The top-up taught the model to call
`odoo_methods_get` before acting, and it fires even on operations that already
know their method — a confirm becomes a methods lookup with an invented
`res_ids`. Exclude single-purpose operations from the `method_discovery` family.

**Ids are hallucinated when absent.** Never let an id appear in a target that did
not appear in a prior tool result within the same sample.

All three are the same underlying mistake: the training data permits the model to
produce values it was never given. Fixing the generator would remove the need for
a good deal of defensive validation downstream.

---

## What is not in this repository

| | why | how to get it |
|---|---|---|
| LoRA adapters (~1 GB each) | past the free Git LFS tier; `lm_head` being a target module is what makes them large | retrain — configs and data are here |
| Merged fp16 model (8.3 GB) | reproducible | `merge_lora.py` |
| GGUF (8.8 GB) | reproducible | `export_gguf.py` |
| Base model (~10 GB) | someone else's to distribute | downloaded from HuggingFace on first run |
| `odoo_agent_forge/.env` | API keys | create your own from `.env.example` |

---

## Licence

Qwen3-4B-Instruct-2507 is Apache 2.0, so a fine-tune of it can be redistributed
without a licence problem — which is part of why it was chosen. The Odoo source
tree read during generation is never redistributed here; only the extracted model
and field names appear in the dataset.
