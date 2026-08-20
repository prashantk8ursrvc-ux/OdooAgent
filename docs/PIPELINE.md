# Odoo 19 Agent Dataset Forge — v2

Generates SFT data for an MCP agent that drives a live Odoo 19 database.

---

## Why v2 exists

The v1 pipeline produced 14,285 cached samples. Run through the production
quality gate, **5 of them pass**. The failures were structural, not incidental.

### 1. The user turn was a Python f-string

The user turn is the only span the trained model ever conditions on. v1 generated
it with templates:

```python
f"Execute standard ERP database operation on {title} ({model})."
f"Execute full multi-step business workflow: {base} Search existing records, "
f"create new entry, execute '{method}()', and verify state."
```

**8,470 of 14,285** rows opened that way. Nobody types that. Worse, the template
hands the model the technical model name and method — the exact inference the
agent is supposed to perform. Training on it teaches the model to wait for
information a real user never supplies.

### 2. Model × method pairing was index arithmetic

`models[i % len(models)]` walked all 2,266 extracted Odoo models — including
`ir.*` plumbing, EDI XML serializers, report handlers, and abstract mixins — and
crossed them with `methods[i % len(methods)]`. When a model had no action
methods, the code substituted `"action_post"`.

The result: `action_post()` on `decimal.precision`, `action_register_payment()`
on `crm.tag`, `action_post()` on `account.edi.xml.ubl_sg`. **1,417** tool calls
invoked a method that does not exist on the target model.

### 3. The reasoning was decoupled from the tool calls

In the trajectory families the teacher LLM saw only the user prompt. Python then
hardcoded the tool sequence and glued the teacher's `<think>` onto step 0. The
reasoning described a plan the transcript did not follow, and the closing summary
was written without ever having seen a tool result.

### 4. Every tool call succeeded

19,168 tool calls, all returning `{"status": "success"}`. Combined with
placeholder arguments (`"Sample Name #1"`, `partner_id: 1`) in **5,196** calls,
this teaches two false things: that ERP operations never fail, and that a tool
result is not worth reading.

### 5. Cache identity was broken, so one family looped

`_run_family` decided whether a sample was already cached by searching for a
substring of the system prompt. Family 7 searched for `"multi-step workflows"`;
its system message says `"multi-step long-horizon business processes"`. The
lookup never matched, so every run regenerated the whole family and appended it.
That family holds **4,792 rows, 2,481 of them duplicates**. Families 12 and 13
had the same mismatch.

### 6. The run crashed at family 12

`generate_all_families` called `self.gen_family_12_mcp_agent(...)`; the method
was defined as `gen_family_12_mcp_agent_protocol`. `AttributeError` killed every
run at that point, which is why families 10–14 have **zero** rows.

### 7. Truncation was never detected

`max_tokens=16384` and `reasoning_budget=16384`. On these endpoints reasoning and
answer share one allowance, so a long think block consumed it and the answer was
cut. Nothing inspected `finish_reason`, so **4,327** truncated answers were
cached as if complete.

### 8. Version drift went unchecked

**798 of 1,498** planning samples cite Odoo 14–18. This is meant to be an Odoo 19
dataset.

---

## How v2 works

```
Odoo 19 source ──► AST extraction ──► knowledge graph (SQLite)
                                            │
                        curated allowlist ──┼──► verified agent surface
                                            │      20 models, 97 methods
                                            ▼
   persona × situation × verified model/method ──► seeded independent draw
                                            │
                     ┌──────────────────────┴───────────────────────┐
              PHASE 1: teacher writes            PHASE 2: plan executed against
              the user's message, in             the simulator, then teacher writes
              persona, forbidden from            the agent's reasoning and reply
              naming model/method/id             *given the actual results*
                     └──────────────────────┬───────────────────────┘
                                            ▼
                                     quality gate  ──► rejected (with reasons)
                                            │
                                            ▼
                                   cache (resumable) ──► export + held-out split
```

### The agent surface (`agent_surface.py`)

**71 models, 380 verified callable methods**, in two tiers. Everything in both
tiers is **intersected with the AST extraction** at load time — anything the scan
cannot confirm is dropped and logged. This is what makes `action_post()` on
`decimal.precision` structurally impossible rather than merely unlikely.

**Tier A — 20 models, 97 methods, hand-curated.** Each carries the state a method
expects and produces, and the exceptions it really raises. Cannot be derived:
the `state` field's selection values are absent from the extraction for most
models, and guessing which method moves a record between which states is exactly
the invention that broke v1.

**Tier B — 51 models, 283 methods, discovered from the Community + Enterprise
tree.** Methods and fields are read from the AST scan, so they are verified the
same way Tier A is. What is *not* derived:

| | Source | Why |
|---|---|---|
| methods, fields | auto | verified against the scan |
| state machine | **omitted** | cannot be inferred safely |
| failure modes | **omitted** | a fabricated Odoo exception is worse than none |
| label, domain, personas | hand | `models.name` is just the technical name (`mrp.workorder` → `mrp.workorder`), and feeding that to the persona makes them say it |

Methods are **ranked before truncation**, not taken alphabetically — otherwise
`hr.payslip` surfaced `action_absence_swiss_employee_from_payslip` while
`action_payslip_done` was cut off. Localisation-specific methods score negative
and are dropped entirely.

Families needing a real state machine or a real exception (`workflow_execution`,
`error_recovery`, `verification`, `refusal_and_confirmation`) draw from **Tier A
only**. The rest use both.

Because Tier B calls return only `{"result": true, "id": N}`, the teacher is told
explicitly not to assert a resulting status, and `unsupported_state_claim` in the
gate rejects any answer that names a state no tool result returned. Without both,
a live sample reported *"its state is now 'scrapped' and quantity_done is 0"*
from exactly that payload — both values invented.

```bash
python run_forge_ultra.py --audit-surface        # both tiers, no API calls
python run_forge_ultra.py --audit-surface --no-wide-surface   # Tier A only
```

### Situations and personas (`scenarios.py`)

**89 concrete business circumstances** ("the truck is at the gate and the delivery
paperwork still has not been validated") and **20 personas** with distinct
communication styles. The situation is background for the teacher — it is never
shown to the trained model and never copied verbatim.

A share of requests are deliberately scruffy: lowercase, a typo, starting
mid-thought. Real messages are.

**These two lists are the main lever on dataset size.** A distinct sample is
bounded by `model × method × persona × situation`, so widening either list
multiplies the ceiling directly. Measured distinct groundings at 1,500 per family:

| Family | Tier | Models | Distinct scenarios | Reuse |
|---|---|---:|---:|---:|
| `tool_calling` | A+B | 71 | 1,239 | 1.2× |
| `agent_trajectories` | A+B | 67 | 1,220 | 1.2× |
| `consultant_knowledge` | A+B | 71 | 1,261 | 1.2× |
| `error_recovery` | A | 15 | 638 | 2.4× |
| `refusal_and_confirmation` | A | 11 | 604 | 2.5× |

Before Tier B and the second wave of situations, `error_recovery` had 370
scenarios at the same volume — 4× reuse, which the near-duplicate check (it
normalises digits away) then rejects. If you push past ~1,500 per family, widen
`SITUATIONS` and `TIER_B_MODELS` first; adding failure modes to Tier A specs is
what lifts the two constrained families.

### Two-phase generation (`prompts.py`, `dataset_generators.py`)

**Phase 1** puts the teacher in a persona reacting to a situation and asks for
the message that person would type. It is explicitly forbidden from naming the
model, the method, or the record id.

**Phase 2** executes the plan against the simulator first, then shows the teacher
the calls *and their results* and asks for the reasoning and reply. The `<think>`
block therefore describes what actually happened.

House rules (Odoo 19 only, no sign-offs, no narration) live in the **system**
prompt. When they were in the user message the teacher rehearsed them in its
reasoning — producing think blocks like *"Must be concrete, 2-6 sentences, no
sign-offs"* — which then became training data.

### The simulator (`simulator.py`)

Builds each tool call's realistic response from the model's own verified fields.
Values for the record a sample is *about* are memoised, so re-reading returns the
same data; only `state` changes, because that is what the method call changed.
Date fields respect their meaning — `date_done` is always past, `date_deadline`
always future.

A configurable share of calls (`--failure-rate`, default 0.22) raise an exception
the method genuinely raises, and the trajectory **stops there**, as a real agent
loop would.

### The quality gate (`quality.py`)

Every sample is gated before it is cached. Rejection reasons:

| Reason | Catches |
|---|---|
| `robotic_prompt` | Template-generated user turns |
| `placeholder_values` | `Sample X #1` in tool arguments |
| `truncated` | Answers that stop mid-sentence |
| `hallucinated_method` | Method absent from the model in the KG |
| `version_drift` | Any mention of Odoo 8–18 |
| `prompt_artifact` | Generation-harness text in the output |
| `instruction_echo` | Reasoning that rehearses formatting rules |
| `tool_call_as_text` | A tool invocation written as prose |
| `fabricated_reference` | Document references the tools never returned |
| `chatty` | Sign-offs an agent would not write |
| `teacher_voice` | Third-person narration in the visible answer |
| `duplicate` / `near_duplicate` | Exact and digit-normalised repeats |
| `orphan_tool_result` | A tool result with no matching call |
| `broken_think` | Unbalanced `<think>` tags |

The same gate can be run over an older cache before merging it, so
surviving old rows meet exactly the standard new rows must meet.

### The 18 families

| Family | Skill trained |
|---|---|
| `tool_calling` | One request → one correct call |
| `lookup_then_act` | Resolve a human reference to an id, then act |
| `workflow_execution` | Lifecycle transition, grounded in real from/to states |
| `agent_trajectories` | Multi-step work, with real failure odds |
| `error_recovery` | Diagnose a genuine Odoo exception and recover |
| `clarification_dialogues` | Ask instead of guessing |
| `verification` | Re-read to prove the mutation committed |
| `multi_turn_memory` | Resolve "it" / "that one" across turns |
| `business_data_retrieval` | Business question → correct domain filter |
| `report_analysis` | `read_group` → an answer a manager can act on |
| `consultant_knowledge` | Explain Odoo 19 from the extracted schema |
| `mcp_agent_protocol` | Choose the right primitive among plausible wrong ones |
| `refusal_and_confirmation` | Pause and confirm before something irreversible |
| `schema_knowledge` | Which methods and fields exist, and how models join |
| `record_creation` | Build a record from what the user actually said |
| `record_update` | Change only the fields named, on the right record |
| `method_discovery` | List a model's methods, pick one, then act |
| `answer_and_stop` | Stop reading once the result answers the question |

### Why `schema_knowledge` exists (`knowledge_pack.py`)

The MCP interface exposes six **generic primitives** —
`odoo_execute_method(model, method, res_ids, kwargs)` and friends. That schema
declares the *primitive*, not which of Odoo's 35,482 methods are valid on which
model, what state each expects, or which fields a `domain` may reference. Unlike
a per-tool function-calling setup, there is no run-time source for any of it.

**So that knowledge has to be in the weights.** This family installs it.

It behaves differently from the behavioural families on purpose. Policy learning
wants breadth — many situations, little repetition. *Recall* wants the same fact
seen from several angles, repeatedly. So this pack is deliberately repetitive per
fact, and weighted by tier: 34 samples per Tier A model, 6 per Tier B, none for
the other 2,195.

Five fact types, all extracted from the knowledge graph:

| Type | Teaches |
|---|---|
| `method_inventory` | which operations exist on a model |
| `method_selection` | intent → the right method, and why not the near-miss |
| `field_reference` | the fields a domain or create call may use |
| `relation` | how two models join, and through which field |
| `state_machine` | the lifecycle, and which method moves between states |

Relations are ranked so both ends are models the agent actually drives — 99% now
point at a surface model. Ranking by table order buried `sale.order.partner_id`
under `sale.order.event_booth_ids`.

**The earlier attempt is reused, not discarded.** `odoo_schema_knowledge_base.jsonl`
holds 4,531 rows — exactly one `schema_reference` and one `model_lookup` per
model, for all 2,266 of them. Even weighting meant `decimal.precision` received
the same capacity as `sale.order`, and only **3%** of the file concerns a model
in the agent surface. `--legacy-kb` merges the relevant slice and drops the rest.

`refusal_and_confirmation` is new. Without it, a model trained on this data will
cancel a posted invoice on a one-line instruction.

---

## Running it

The teacher key lives in `odoo_agent_forge/.env` as `NVIDIA_API_KEY`. It is read
from there regardless of what directory you launch from.

```bash
# Inspect the grounding. No API calls.
python run_forge_ultra.py --audit-surface

# Inspect what would be generated. No API calls.
python run_forge_ultra.py --dry-run

# Generate. Resumable — cached samples are never regenerated.
python run_forge_ultra.py --use-nvidia-llm --samples-per-family 200 --workers 3
```

Start small (`--samples-per-family 5`) to confirm the key has quota before
committing to a long run. Because generation is resumable, scaling up afterwards
costs nothing: the second run picks up where the first stopped.

### Scheduling: rounds, not one family at a time

Families are generated **round-robin**. Each round tops every family up to the
next multiple of `--chunk` (default 100), so an interrupted run leaves thirteen
partial families rather than a few finished ones and the rest empty.

This matters more than it sounds. A measured run produced 1,746 samples in 18
hours (~97/hour). Under the original sequential scheduler that meant family 1
alone consumed the first ~15 hours, and **six of thirteen families still had zero
samples** — no verification, analysis, or refusal behaviour in the dataset at all.
Round-robin gives balanced coverage at every point you might stop.

```bash
# only the families that have nothing yet
python run_forge_ultra.py --use-nvidia-llm --samples-per-family 1500 --families empty

# a specific set, in the order you want
python run_forge_ultra.py --use-nvidia-llm --families refusal_and_confirmation,report_analysis

# declared order reversed
python run_forge_ultra.py --use-nvidia-llm --families reverse
```

### Throughput: move phase 1 off the throttled pool

Phase 1 writes one short line — the user's message — but accounts for roughly
**45% of all API calls**. On a rate-limited account that is where most of the wall
clock goes. Point it at a local model and the teacher is reserved for phase 2,
where the graded reasoning actually happens:

```bash
python run_forge_ultra.py --use-nvidia-llm --samples-per-family 1500 \
    --local-phase1 qwen2.5:7b
```

Writing "what would a warehouse operator type" is well within a 7B model's range,
and it is the one part of the pipeline where a weaker model costs nothing —
phase 1 output still passes the same `robotic_prompt` gate. If the local model is
unreachable or returns nothing, the call silently falls back to the teacher.

Concurrency is the main driver of HTTP 429 from the teacher pool. `--workers 3`
is a reasonable default.

**On rate limits.** A model that just returned 429 will return 429 again a second
later, so the client parks it for an exponentially-growing cooldown and routes
the next call to whichever endpoint is still serving — it only sleeps when
everything is saturated. Without this, workers spend most of their wall-clock
retrying known-bad endpoints: a pilot logged 428 rate-limit errors and completed
3 of 13 families.

### More keys, not more threads

Rate limits are enforced **per account**, so raising `--workers` past a handful
just produces 429s faster. Extra keys are the only real throughput lever, and the
pool treats every *(provider, key, model)* combination as an independently
schedulable endpoint.

```bash
python run_forge_ultra.py --show-endpoints    # what the pool sees, no keys printed
```

Any environment variable whose name contains `API_KEY` is picked up, so
`NVIDIA_API_KEY2`, `OPENROUTER_API_KEY1` and so on need no code change. Three
NVIDIA keys plus three OpenRouter keys gives 15 endpoints against the original 4.

Two things the discovery does deliberately:

**Provider is detected from the key prefix** (`nvapi-` / `sk-or-`), never from the
variable name. A real `.env` here held an NVIDIA key under `OPENROUTER_API_KEY`;
trusting the name would have sent it to OpenRouter and failed every call with an
auth error that reads exactly like a rate limit.

**Duplicate keys are dropped.** The same key under two names is one rate-limit
budget, not two — counting it twice would just mean two endpoints throttling in
lockstep.

A key that returns 401 is parked for the whole run rather than retried on every
call, so one bad key costs one failed request instead of thousands.

**Calibrate your expectations by tier.** Extra NVIDIA keys multiply capacity
directly. OpenRouter *free-tier* keys reach only `:free` models, which carry
their own daily caps — useful headroom, but not a substitute. Buying a small
amount of credit on an OpenRouter account raises its daily allowance
substantially; check `--show-endpoints` and your OpenRouter dashboard before
assuming a key is pulling its weight.

**Every sample records which teacher produced it** in `_meta.teacher`. Mixing
providers widens throughput and also widens quality variance, and the gate only
catches structural defects — not "this model's reasoning is shallower". The tag
means a weak endpoint can be audited or filtered out afterwards without
regenerating the rest:

```bash
python -c "import json,collections;print(collections.Counter(json.loads(l)['_meta'].get('teacher') for l in open('forge_outputs/generation_cache_v2.jsonl',encoding='utf-8') if l.strip()).most_common())"
```

Generation is resumable, so a long run can be interrupted and restarted freely.

Output lands in `forge_outputs/`:

| File | Contents | Committed | Read by |
|---|---|---|---|
| `generation_cache_v2.jsonl` | Resumable cache — the source of truth every later stage reads | yes | `prepare_dataset.py` |
| `odoo_schema_knowledge_base.jsonl` | One `schema_reference` and one `model_lookup` per model, from the earlier pass | yes | `run_forge_ultra.py --legacy-kb` |
| `odoo19_agent_eval.jsonl` | Held-out split, grouped by model+method so it cannot leak | yes | `eval_scorecard.py` |
| `odoo19_agent_sft.jsonl` | Training split, messages only | no | — |
| `odoo19_mcp_tool_calling.jsonl` | Tool-calling families | no | — |
| `odoo19_agent_trajectories.jsonl` | Multi-step families | no | — |
| `odoo19_sample_index.jsonl` | Provenance sidecar (family, model, persona, split) | no | — |
| `dataset_manifest.json` | Counts by family, model, persona, shape | no | — |

The rule is simply whether something later opens the file. The three that are
read are committed; the rest are views over the same cache, rewritten on every
run, and carrying them would mean shipping the same samples several times over.

Note that `odoo_schema_knowledge_base.jsonl` is an *input* to generation rather
than an output of it — see "The earlier attempt is reused, not discarded" above.
It sits here because it lives in the same folder and is easy to mistake for a
by-product.

`_meta` is stripped from training files and kept in the sidecar, so generator
bookkeeping never enters the token stream.

**The eval split is grouped by `model::method`.** Two samples about
`sale.order.action_confirm` differ mainly in their document reference; splitting
them across train and eval would let the model score well by memorising a
pattern.
