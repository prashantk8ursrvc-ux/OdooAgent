#!/usr/bin/env python3
"""
Merge the adapter, export GGUF, and write an Ollama Modelfile that keeps tool
calls intact.

The Modelfile is the part worth reading carefully. The template shipped with the
shipped by default is:

    TEMPLATE "{{- range .Messages }}<|im_start|>{{ .Role }}
    {{ .Content }}<|im_end|>{{ end }}"

That renders ``role`` and ``content`` only. It silently discards ``ToolCalls``
and has no ``.Tools`` block, so a model fine-tuned to emit tool calls would emit
them and Ollama would drop them on the floor — and separately, the model would
never see the tool definitions it was trained with. The symptom is a model that
"seems dumber than in training" for reasons no amount of retraining fixes.

The template written here renders the tool list, assistant tool calls, and tool
responses in Qwen's native format, matching what ``prepare_dataset.py`` produced
via the tokenizer's own chat template.

Usage
-----
    python training/export_gguf.py --config training/configs/qwen3_4b.json
    python training/export_gguf.py --config ... --quant q5_k_m
    python training/export_gguf.py --config ... --skip-gguf   # merge 16-bit only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


#: Ollama template for Qwen-style ChatML with tools.
#:
#: Three things the old template lacked, each of which breaks tool calling on its
#: own: the .Tools block so the model sees what it may call; .ToolCalls so its
#: calls survive; and the tool role so results come back in the shape it was
#: trained on.
#: Written with real Go template syntax and filled by plain substitution, not
#: str.format. Escaping a Go template through .format means writing {{{{ for
#: every {{, and the literal JSON example below then comes out as {{"name": ...}}
#: — which Go parses as an action and rejects with
#: "template error: template: :21: expected :=". Single braces are literal to Go;
#: only {{ is special. Substitution keeps the two brace languages separate.
MODELFILE_TEMPLATE = '''FROM __GGUF_PATH__

TEMPLATE """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{{ .Function }}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

SYSTEM """__SYSTEM__"""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx __NUM_CTX__
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
'''


def build_modelfile(cfg: dict, gguf_path: Path) -> str:
    from odoo_agent_forge.prompts import AGENT_SYSTEM_PROMPT

    # Only the operating instructions belong here. The house rules in
    # prompts.py steered the *teacher* while generating data; the student
    # learned that behaviour from the data itself and does not need to be told
    # again at inference.
    system = AGENT_SYSTEM_PROMPT.split("\n\nHow you write, always:")[0].strip()
    # Low temperature: this model emits JSON tool calls, where creativity is
    # only ever a source of malformed arguments.
    # Absolute, always. Ollama resolves FROM against its own working directory,
    # not the Modelfile's, so a relative path silently becomes a *model name*
    # lookup and fails with "400 Bad Request: invalid model name" — an error that
    # says nothing about paths. Forward slashes work on Windows too.
    return (MODELFILE_TEMPLATE
            .replace("__GGUF_PATH__", str(gguf_path.resolve()).replace("\\", "/"))
            .replace("__SYSTEM__", system.replace('"""', "'''"))
            .replace("__NUM_CTX__", str(_inference_ctx(cfg))))


def _inference_ctx(cfg):
    """Context window to serve with — not the training sequence length.

    These are different numbers and conflating them caused a real failure. Training
    saw 2,048-token samples, so this used to emit max(4096, 2048*2) = 4096. But an
    agent turn has to hold the system prompt, the tool schemas, the whole replayed
    chat history *and* room for the reply. Measured in production: prompt tokens
    climbed 1,620 -> 2,594 over ten messages, and with 1,024 reserved for output
    that is 88% of 4,096 by the fourth question.

    Past the window llama.cpp truncates from the front, which is exactly where the
    system prompt and tool descriptions live — so the model loses its instructions
    mid-conversation and starts calling the wrong tools. That is the "it gets weird
    after a few messages" symptom.

    8,192 doubles the headroom and still fits: q4_K_M is ~2.7 GB and the KV cache
    at this length is ~1.2 GB, comfortable inside 8 GB. The base model itself
    supports 262,144, so this is a memory decision, not a model limit.
    """
    return max(8192, int(cfg.get("max_seq_length", 2048)) * 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge, quantise and package for Ollama")
    ap.add_argument("--config", required=True)
    ap.add_argument("--quant", default=None,
                    help="Override gguf_quant, e.g. q4_k_m, q5_k_m, q8_0.")
    ap.add_argument("--skip-gguf", action="store_true",
                    help="Merge to 16-bit only; skip GGUF conversion.")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output_dir"]
    adapter_dir = out_dir / "adapter"
    if not adapter_dir.exists():
        print(f"No adapter at {adapter_dir}. Train first:\n"
              f"  python training/train.py --config {args.config}")
        return 1

    quant = args.quant or cfg.get("gguf_quant", "q4_k_m")

    import unsloth  # noqa: F401  patches before transformers
    from unsloth import FastLanguageModel

    print(f"loading adapter : {adapter_dir}")
    # Load in 16-bit for merging. A merge from the 4-bit weights would bake the
    # quantisation error into the exported model permanently.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=False,
    )

    merged_dir = out_dir / "merged-16bit"
    print(f"merging         : {merged_dir}")
    # This holds the whole model in 16-bit before writing - ~8.3 GB for this 4B - so
    # close other applications first. If it ever dies partway — leaving a
    # model.safetensors.index.json with no shard files beside it — then
    # training/merge_lora.py does the same merge one shard at a time, peaking at
    # ~4 GB, and produces a bit-identical result tensor for tensor.
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    if args.skip_gguf:
        print("skipped GGUF conversion (--skip-gguf)")
        return 0

    # Convert with the local llama.cpp checkout, not Unsloth's save_pretrained_gguf.
    #
    # save_pretrained_gguf git-clones llama.cpp and builds it with cmake/make to
    # get the `llama-quantize` binary. On this machine `make` and `gcc` are not on
    # PATH, so that build fails — which is why this path needs a manual
    # converter both times.
    #
    # convert_hf_to_gguf.py is pure Python and needs no compiler, so it produces
    # an F16 GGUF reliably. Quantisation is then handed to Ollama, which ships its
    # own quantiser: `ollama create -q q4_K_M`. No build step anywhere.
    gguf_dir = out_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / f"{cfg.get('ollama_model_name', 'model')}-f16.gguf"

    converter = ROOT / "llama_cpp_tools" / "convert_hf_to_gguf.py"
    if not converter.exists():
        print(f"No converter at {converter}.\n"
              f"Clone llama.cpp there, or point --converter at your copy:\n"
              f"  git clone --depth 1 https://github.com/ggml-org/llama.cpp llama_cpp_tools")
        return 1

    print(f"converting f16  : {gguf_path.name}")
    cmd = [sys.executable, str(converter), str(merged_dir),
           "--outfile", str(gguf_path), "--outtype", "f16"]
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0 or not gguf_path.exists():
        print(f"\nConversion failed. The 16-bit merge at {merged_dir} is intact, "
              f"so nothing is lost — retry with:\n  {' '.join(cmd)}")
        return 1

    size_gb = gguf_path.stat().st_size / 1024 ** 3

    modelfile = out_dir / "Modelfile"
    modelfile.write_text(build_modelfile(cfg, gguf_path), encoding="utf-8")

    name = cfg.get("ollama_model_name", "odoo19-agent")
    print(f"\ngguf (f16)      : {gguf_path}  ({size_gb:.2f} GB)")
    print(f"modelfile       : {modelfile}")
    print(f"\nInstall it — Ollama quantises on create, so no llama-quantize binary")
    print(f"and no compiler is needed:\n")
    print(f'  ollama create {name} -q {quant} -f "{modelfile}"\n')
    print(f"That writes a ~{size_gb/3.5:.1f} GB {quant} model. Drop -q to keep f16 "
          f"({size_gb:.2f} GB) if you have the VRAM and want maximum fidelity.")
    print(f"\nThen check tool calls survive the round trip:\n"
          f"  python training/smoke_test.py --model {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
