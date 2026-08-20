#!/usr/bin/env python3
"""
Odoo 19 Agent Dataset Forge
===========================
Scans the Odoo 19 source tree, builds a knowledge graph, and generates grounded
MCP agent training data across 13 behavioural families.

Pipeline:
  1. AST extraction from the Odoo codebase        (skipped if the DB is populated)
  2. Knowledge graph construction                 (SQLite)
  3. Agent-surface verification                   (curated allowlist x extracted AST)
  4. Two-phase grounded generation                (request, then agent turns)
  5. Quality gate                                 (rejects before caching)
  6. Export with a leak-free held-out split

Usage:
  python run_forge_ultra.py --use-nvidia-llm --samples-per-family 200
  python run_forge_ultra.py --audit-surface          # verify grounding, no API calls
  python run_forge_ultra.py --dry-run                # inspect prompts, no API calls
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _signal_handler(sig, frame):
    print("\n\nInterrupted. Everything generated so far is already in the cache; "
          "re-run to resume from there.", flush=True)
    sys.exit(130)


signal.signal(signal.SIGINT, _signal_handler)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from odoo_agent_forge import agent_surface
from odoo_agent_forge.config import Settings, export_api_keys
from odoo_agent_forge.dataset_generators import DatasetGeneratorFactory
from odoo_agent_forge.exporter import DatasetExporter
from odoo_agent_forge.extractor import OdooCodebaseExtractor
from odoo_agent_forge.knowledge_graph import OdooKnowledgeGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("OdooForge")
console = Console()


def _parse_args():
    p = argparse.ArgumentParser(description="Odoo 19 Agent Dataset Forge")
    p.add_argument("--codebase", type=str,
                   default=os.environ.get("ODOO_SOURCE", "./odoo"),
                   help="Path to the Odoo 19 source tree")
    p.add_argument("--samples-per-family", type=int, default=200,
                   help="Accepted samples to produce per family")
    p.add_argument("--use-nvidia-llm", action="store_true",
                   help="Generate via the NVIDIA teacher model pool")
    p.add_argument("--output-dir", type=str, default="./forge_outputs")
    p.add_argument("--cache", type=str, default="./forge_outputs/generation_cache_v2.jsonl",
                   help="Generation cache. Resumable: cached samples are never regenerated.")
    p.add_argument("--rebuild-db", action="store_true",
                   help="Force a rescan of the codebase and rebuild forge_knowledge.db")
    p.add_argument("--workers", type=int, default=3,
                   help="Concurrent teacher requests. Raising this is the fastest "
                        "way to trigger HTTP 429 from the model pool.")
    p.add_argument("--failure-rate", type=float, default=0.22,
                   help="Share of trajectory samples where a call raises a real Odoo "
                        "exception. 0 trains an agent that believes nothing ever fails.")
    p.add_argument("--seed", type=int, default=20260317)
    p.add_argument("--eval-fraction", type=float, default=0.05)
    p.add_argument("--families", type=str, default=None,
                   help="Comma-separated family names to generate, in order. "
                        "Accepts 'reverse' for the declared order reversed, or "
                        "'empty' to target only families with no cached samples. "
                        "Default: all thirteen.")
    p.add_argument("--preset", type=str, default=None,
                   choices=("writes", "grounding", "new", "reads"),
                   help="Named family group, so you do not have to remember which "
                        "families cover what. "
                        "'writes' = record_update + record_creation (the ORM edit "
                        "paths, where odoo_write had zero coverage until now). "
                        "'grounding' = the families that teach not inventing values: "
                        "record_creation, record_update, clarification_dialogues, "
                        "error_recovery, refusal_and_confirmation. "
                        "'new' = families with nothing cached yet. "
                        "'reads' = search, aggregate and analysis. "
                        "Overrides --families.")
    p.add_argument("--family-target", type=str, default=None, metavar="NAME=N,...",
                   help="Per-family sample targets, overriding --samples-per-family "
                        "for those families. Use it for families with a hard "
                        "ceiling: schema_knowledge is bounded by the size of the "
                        "knowledge base and error_recovery by the number of real "
                        "failure modes, so chasing a higher number just re-derives "
                        "draws already on disk for the rest of the run. "
                        "Example: --family-target schema_knowledge=986,error_recovery=1100")
    p.add_argument("--chunk", type=int, default=100,
                   help="Round-robin chunk size. Each round tops every family up "
                        "to the next multiple of this, so an interrupted run "
                        "leaves balanced coverage instead of a few finished "
                        "families and the rest empty.")
    p.add_argument("--legacy-kb", type=str,
                   default="./forge_outputs/odoo_schema_knowledge_base.jsonl",
                   help="Schema-recall rows generated by the earlier pass. Only "
                        "those about a model in the agent surface are merged; the "
                        "file covers all 2,266 models evenly, so ~97%% of it is "
                        "capacity spent on models no agent touches. Pass '' to skip.")
    p.add_argument("--local-phase1", type=str, default=None, metavar="MODEL",
                   help="Use a local Ollama/LM Studio model for phase 1 (writing "
                        "the user's message), keeping the teacher for the graded "
                        "reasoning. Phase 1 is ~45%% of API calls, so this roughly "
                        "halves load on a rate-limited account. e.g. qwen2.5:7b")
    p.add_argument("--local-url", type=str, default=None,
                   help="Base URL for the local model (default Ollama: "
                        "http://localhost:11434/v1)")
    p.add_argument("--no-wide-surface", dest="wide_surface", action="store_false",
                   help="Restrict generation to the 20 hand-curated models. The "
                        "default also uses the ~50 business documents discovered "
                        "from the Community + Enterprise tree, which is what makes "
                        "large per-family counts viable.")
    p.add_argument("--strict-surface", action="store_true",
                   help="Abort if any curated model or method fails KG verification")
    p.add_argument("--show-endpoints", action="store_true",
                   help="List the provider/key/model endpoints the pool will use "
                        "and exit. Keys are never printed. No generation.")
    p.add_argument("--audit-surface", action="store_true",
                   help="Report the verified agent surface and exit. No API calls.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build contexts and prompts without calling the teacher. "
                        "Use to inspect what would be generated.")
    return p.parse_args()


def _build_kg(settings: Settings, args) -> OdooKnowledgeGraph:
    kg = OdooKnowledgeGraph(db_path=str(settings.db_file))

    populated = False
    if settings.db_file.exists() and not args.rebuild_db:
        try:
            populated = bool(kg.get_model_fields("sale.order"))
        except Exception:
            populated = False

    if populated:
        console.print(f"[green]Knowledge graph loaded from {settings.db_file}.[/green]")
        return kg

    console.print(f"[yellow]Scanning Odoo codebase at {settings.odoo_codebase_path}...[/yellow]")
    if not settings.odoo_codebase_path.exists():
        console.print(f"[red]Codebase not found: {settings.odoo_codebase_path}[/red]")
        console.print("[red]Pass --codebase with the path to your Odoo 19 checkout.[/red]")
        sys.exit(1)

    extractor = OdooCodebaseExtractor(settings.odoo_codebase_path)
    modules = extractor.discover_and_extract_all()
    console.print(f"[green]Extracted {len(modules)} modules.[/green]")
    kg.build_graph_from_modules(modules)
    console.print("[green]Knowledge graph built.[/green]")
    return kg


def _audit_surface(kg, wide: bool = True) -> int:
    """Reports the agent surface after KG verification, both tiers."""
    verified, warnings = agent_surface.verify_against_kg(kg)
    discovered, notes = ({}, [])
    if wide:
        discovered, notes = agent_surface.discover_tier_b(kg, exclude=verified)

    table = Table(title="Verified Agent Surface (curated allowlist x extracted AST)")
    table.add_column("Model", style="cyan")
    table.add_column("Domain")
    table.add_column("Methods", justify="right", style="green")
    table.add_column("Mutating", justify="right")
    table.add_column("With failures", justify="right")
    table.add_column("Weight", justify="right")

    for spec in sorted(verified.values(), key=lambda s: (-s.weight, s.model)):
        mutating = sum(1 for m in spec.methods if m.to_state and not m.returns_action)
        failing = sum(1 for m in spec.methods if m.failures)
        table.add_row(spec.model, spec.domain, str(len(spec.methods)),
                      str(mutating), str(failing), str(spec.weight))
    console.print(table)

    core_methods = sum(len(s.methods) for s in verified.values())
    with_failures = sum(1 for s in verified.values() for m in s.methods if m.failures)
    console.print(f"\n[bold]Tier A (curated): {len(verified)} models, {core_methods} "
                  f"methods, {with_failures} with real failure modes.[/bold]")

    if discovered:
        by_domain: dict = {}
        for spec in discovered.values():
            by_domain.setdefault(spec.domain, []).append(spec.model)
        b_methods = sum(len(s.methods) for s in discovered.values())
        console.print(f"[bold]Tier B (discovered from the codebase): {len(discovered)} "
                      f"models, {b_methods} methods. No invented states or "
                      f"failures.[/bold]")
        for domain in sorted(by_domain):
            console.print(f"  [dim]{domain:14s}[/dim] {', '.join(sorted(by_domain[domain]))}")
        console.print(f"\n[bold green]Total surface: {len(verified) + len(discovered)} "
                      f"models, {core_methods + b_methods} verified callable "
                      f"methods.[/bold green]")
        skipped = [n for n in notes if "skipped" in n]
        if skipped:
            console.print(f"[dim]{len(skipped)} candidates skipped: absent from the "
                          f"graph, or no method survived the UI-only and "
                          f"localisation filters.[/dim]")

    if warnings:
        console.print("\n[yellow]Verification notes:[/yellow]")
        for w in warnings:
            console.print(f"  - {w}")
    return 0 if verified else 1


def main() -> int:
    args = _parse_args()

    settings = Settings(
        odoo_codebase_path=Path(args.codebase),
        output_dir=Path(args.output_dir),
    )
    if settings.nvidia_api_key:
        os.environ["NVIDIA_API_KEY"] = settings.nvidia_api_key
    # Every additional key widens the pool: rate limits are per account, so more
    # keys is the only real throughput lever. pydantic drops undeclared fields,
    # so they are exported here explicitly.
    extra = export_api_keys()
    if extra:
        console.print(f"[cyan]Loaded {extra} additional API key(s) from "
                      f"odoo_agent_forge/.env.[/cyan]")

    console.print(Panel.fit(
        "[bold green]Odoo 19 Agent Dataset Forge[/bold green]\n"
        "[cyan]Grounded MCP agent training data — 13 behavioural families[/cyan]"))

    kg = _build_kg(settings, args)

    if args.show_endpoints:
        return _show_endpoints()

    if args.audit_surface:
        return _audit_surface(kg, wide=args.wide_surface)

    if not args.use_nvidia_llm and not args.dry_run:
        console.print(
            "[red]Refusing to run without a teacher model.[/red]\n"
            "Template-generated text is what produced the 8,470 unusable prompts in "
            "the previous dataset, so there is deliberately no offline fallback.\n"
            "Pass --use-nvidia-llm (with NVIDIA_API_KEY set), or --dry-run to inspect "
            "prompts without generating.")
        return 1

    factory = DatasetGeneratorFactory(
        kg,
        use_nvidia_llm=args.use_nvidia_llm and not args.dry_run,
        api_key=settings.nvidia_api_key if args.use_nvidia_llm else None,
        cache_path=args.cache,
        seed=args.seed,
        failure_rate=args.failure_rate,
        strict_surface=args.strict_surface,
        max_workers=args.workers,
        wide_surface=args.wide_surface,
        use_local_llm=bool(args.local_phase1),
        local_llm_model=args.local_phase1 or "qwen2.5:7b",
        local_llm_base_url=args.local_url,
    )

    if args.dry_run:
        return _dry_run(factory)

    console.print(f"\n[yellow]Generating up to {args.samples_per_family} samples "
                  f"per family across 13 families...[/yellow]")
    family_targets = {}
    if args.family_target:
        for pair in args.family_target.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise SystemExit(
                    f"--family-target expects NAME=N pairs, got {pair!r}. "
                    f"Example: --family-target schema_knowledge=986")
            name, _, value = pair.partition("=")
            try:
                family_targets[name.strip()] = int(value)
            except ValueError:
                raise SystemExit(
                    f"--family-target: {value!r} is not a number (in {pair!r}).")

    # _resolve_families was defined but never wired in, so --families was accepted,
    # validated, and then silently ignored — every run generated all of them. Found
    # by asking for one family and watching the log work through the others.
    datasets = factory.generate_all_families(
        count_per_family=args.samples_per_family,
        families=_resolve_families(args, factory),
        chunk=args.chunk,
        family_targets=family_targets or None,
    )

    console.print(f"\n[yellow]Exporting to {settings.output_dir}...[/yellow]")
    knowledge_rows = _load_legacy_knowledge(args, factory)
    exporter = DatasetExporter(settings.output_dir, eval_fraction=args.eval_fraction,
                               seed=args.seed)
    exported = exporter.export_all(datasets, knowledge_rows=knowledge_rows)

    _summary(datasets, factory, exported)
    return 0


def _load_legacy_knowledge(args, factory) -> list:
    """Pulls in the surface-relevant slice of the earlier schema knowledge base."""
    from odoo_agent_forge.knowledge_pack import filter_legacy_kb

    path = (args.legacy_kb or "").strip()
    if not path:
        return []
    rows = filter_legacy_kb(path, factory.surface)
    if not rows:
        return []

    gate = factory.gate
    kept = [r for r in rows if gate.check({"messages": r["messages"]}).ok]
    console.print(
        f"[cyan]Schema knowledge base: {len(rows)} of its rows concern a model in "
        f"the agent surface; {len(kept)} pass the quality gate and are merged.[/cyan]")
    return [{"messages": r["messages"],
             "_meta": {"family": "schema_knowledge", "shape": "knowledge:legacy",
                       "model": r.get("_model", "?"), "method": r.get("_type", "?"),
                       "domain": "knowledge", "persona": "ERP consultant",
                       "generator_version": "legacy-kb"}}
            for r in kept]


def _show_endpoints() -> int:
    """Reports the endpoint pool. Keys are identified by variable name only."""
    from odoo_agent_forge.llm_client import LLMPool, discover_keys

    keys = discover_keys()
    if not keys:
        console.print("[red]No usable API key found.[/red] Keys are recognised by "
                      "their prefix: nvapi- for NVIDIA, sk-or- for OpenRouter.")
        return 1

    table = Table(title="LLM Endpoint Pool")
    table.add_column("Provider", style="cyan")
    table.add_column("Key variable")
    table.add_column("Models", justify="right", style="green")
    by_key: dict = {}
    pool = LLMPool()
    for e in pool.endpoints:
        by_key.setdefault((e.provider, e.key_name), []).append(e.model)
    for (provider, key_name), models in sorted(by_key.items()):
        table.add_row(provider, key_name, str(len(models)))
    console.print(table)

    console.print(f"\n[bold]{len(pool.endpoints)} endpoints from {len(keys)} "
                  f"distinct keys.[/bold] Rate limits are per account, so this is "
                  f"roughly how much parallel capacity you have.")
    console.print("[dim]Duplicate keys under different variable names are ignored — "
                  "they share one budget. Provider is detected from the key prefix, "
                  "not the variable name.[/dim]")
    return 0


#: Named groups for --preset. Kept here rather than in the factory because they are
#: a convenience for whoever runs the script, not a property of the dataset.
FAMILY_PRESETS = {
    # odoo_write had zero samples across 20,485 rows before record_update existed:
    # build_write_call was in the simulator and no family ever called it.
    "writes": ("record_update", "record_creation"),
    # The families that teach the agent not to make things up — invented names,
    # guessed ids, and claiming a record exists when it does not.
    "grounding": ("record_creation", "record_update", "clarification_dialogues",
                  "error_recovery", "refusal_and_confirmation"),
    "reads": ("business_data_retrieval", "report_analysis", "lookup_then_act",
              "verification"),
}


def _resolve_families(args, factory) -> list:
    """Turns --preset / --families into an explicit ordered list."""
    order = list(factory.FAMILY_ORDER)

    if getattr(args, "preset", None):
        if args.preset == "new":
            chosen = [f for f in order if not factory.by_family.get(f)]
            if not chosen:
                console.print("[yellow]Every family already has samples; "
                              "generating all of them.[/yellow]")
                return order
        else:
            chosen = [f for f in FAMILY_PRESETS[args.preset] if f in order]
        console.print(f"[cyan]--preset {args.preset}: {', '.join(chosen)}[/cyan]")
        return chosen

    raw = (args.families or "").strip().lower()

    if not raw:
        return order
    if raw == "reverse":
        return list(reversed(order))
    if raw == "empty":
        empty = [f for f in order if not factory.by_family.get(f)]
        if not empty:
            console.print("[yellow]No family is empty; generating all of them.[/yellow]")
            return order
        console.print(f"[cyan]Targeting {len(empty)} famil"
                      f"{'y' if len(empty)==1 else 'ies'} with no cached samples: "
                      f"{', '.join(empty)}[/cyan]")
        return empty

    chosen = [f.strip() for f in raw.split(",") if f.strip()]
    unknown = [f for f in chosen if f not in order]
    if unknown:
        console.print(f"[red]Unknown famil{'y' if len(unknown)==1 else 'ies'}: "
                      f"{', '.join(unknown)}[/red]")
        console.print(f"[dim]Valid: {', '.join(order)}[/dim]")
        sys.exit(2)
    return chosen


def _dry_run(factory) -> int:
    """Shows one grounded context and phase-1 prompt per family. No API calls."""
    from odoo_agent_forge import prompts

    console.print("\n[bold]Dry run — grounded contexts, no teacher calls.[/bold]\n")
    for family in ("tool_calling", "error_recovery", "agent_trajectories",
                   "clarification_dialogues", "report_analysis",
                   "refusal_and_confirmation"):
        ctx = factory._draw_context(0, family, mutating_only=True)
        if not ctx:
            console.print(f"[red]{family}: no context could be drawn[/red]")
            continue
        console.print(Panel(
            f"[cyan]model[/cyan]    {ctx['spec'].model}.{ctx['method'].name}\n"
            f"[cyan]states[/cyan]   {ctx['method'].from_state} -> {ctx['method'].to_state}\n"
            f"[cyan]persona[/cyan]  {ctx['persona'].role} ({ctx['persona'].channel})\n"
            f"[cyan]ref[/cyan]      {ctx['doc_ref']}\n"
            f"[cyan]partner[/cyan]  {ctx['partner_name']}\n"
            f"[cyan]situation[/cyan] "
            f"{ctx['situation'].text if ctx['situation'] else '(none)'}",
            title=family))
    console.print("\n[green]Dry run complete. Re-run with --use-nvidia-llm to "
                  "generate.[/green]")
    return 0


def _summary(datasets, factory, exported) -> None:
    table = Table(title="Generation Summary")
    table.add_column("Family", style="cyan")
    table.add_column("Accepted", justify="right", style="green")
    for family, samples in datasets.items():
        table.add_row(family.replace("_", " ").title(), str(len(samples)))
    table.add_row("[bold]Total[/bold]",
                  f"[bold]{sum(len(v) for v in datasets.values())}[/bold]")
    console.print(table)

    if factory.gate.counts:
        rej = Table(title="Rejected by the quality gate")
        rej.add_column("Reason", style="yellow")
        rej.add_column("Count", justify="right")
        for reason, count in sorted(factory.gate.counts.items(), key=lambda kv: -kv[1]):
            rej.add_row(reason, str(count))
        console.print(rej)

    if exported:
        console.print(Panel("\n".join(f"{k}: {v}" for k, v in exported.items()),
                            title="Exported files"))


if __name__ == "__main__":
    sys.exit(main())
