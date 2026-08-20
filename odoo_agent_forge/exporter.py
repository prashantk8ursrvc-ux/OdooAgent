"""
Stage 5: Training Dataset Exporter
==================================

Writes the accepted samples out in the shapes a trainer consumes.

Two things the previous exporter got wrong:

* It wrote every internal field straight into the training file. Samples now
  carry a ``_meta`` block (family, model, method, persona) that exists for
  auditing and cache identity; shipping it to the trainer would put generator
  bookkeeping into the token stream. It is stripped on the way out and kept in a
  sidecar index instead.

* It emitted a single undifferentiated file with no held-out split, so there was
  no honest way to measure the fine-tune. Splits are produced here, grouped so
  that samples about the same model and method cannot straddle train and eval —
  otherwise eval scores measure memorisation of a document reference rather than
  learned behaviour.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

#: Families whose samples belong in the tool-calling-specific export.
_TOOL_FAMILIES = ("tool_calling", "lookup_then_act", "mcp_agent_protocol",
                  "workflow_execution", "business_data_retrieval")

#: Families that are full multi-step agent transcripts.
_TRAJECTORY_FAMILIES = ("agent_trajectories", "multi_turn_memory", "verification")


class DatasetExporter:
    """Exports validated dataset families to JSONL."""

    def __init__(self, output_dir: Path, eval_fraction: float = 0.05,
                 seed: int = 20260317) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_fraction = eval_fraction
        self.seed = seed

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _training_row(sample: Dict[str, Any]) -> Dict[str, Any]:
        """The sample as the trainer should see it: messages only."""
        return {"messages": sample["messages"]}

    @staticmethod
    def _group_key(sample: Dict[str, Any]) -> str:
        """Groups samples that share a model+method so a split cannot leak.

        Two samples about ``sale.order.action_confirm`` differ mainly in their
        document reference. Splitting them across train and eval would let the
        model score well on eval by having memorised the pattern, not the skill.
        """
        meta = sample.get("_meta") or {}
        return f"{meta.get('model', '?')}::{meta.get('method', '?')}"

    def _split(self, samples: List[Dict[str, Any]]
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            groups.setdefault(self._group_key(s), []).append(s)

        keys = sorted(groups)
        random.Random(self.seed).shuffle(keys)

        target = int(len(samples) * self.eval_fraction)
        eval_rows: List[Dict[str, Any]] = []
        eval_keys: set = set()
        for k in keys:
            if len(eval_rows) >= target:
                break
            eval_rows.extend(groups[k])
            eval_keys.add(k)

        train_rows = [s for k in keys if k not in eval_keys for s in groups[k]]
        return train_rows, eval_rows

    @staticmethod
    def _write(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        return n

    # -- main ------------------------------------------------------------------

    def export_all(self, datasets: Dict[str, List[Dict[str, Any]]],
                   dpo_pairs: List[Dict[str, Any]] | None = None,
                   knowledge_rows: List[Dict[str, Any]] | None = None) -> Dict[str, Path]:
        # Schema-recall rows are merged into the SFT split like any other family.
        # They carry no tool calls, so they never reach the tool-calling or
        # trajectory exports, but the model needs them in the same training mix
        # as the behaviour they support.
        if knowledge_rows:
            datasets = dict(datasets)
            datasets.setdefault("schema_knowledge", [])
            datasets["schema_knowledge"] = list(datasets["schema_knowledge"]) + list(knowledge_rows)
            logger.info("Merged %d schema-recall rows into the training mix.",
                        len(knowledge_rows))

        all_samples = [s for family in datasets.values() for s in family]
        if not all_samples:
            logger.warning("Nothing to export: every family is empty.")
            return {}

        exported: Dict[str, Path] = {}
        train, held_out = self._split(all_samples)

        sft = self.output_dir / "odoo19_agent_sft.jsonl"
        n = self._write(sft, (self._training_row(s) for s in train))
        exported["sft"] = sft
        logger.info("SFT train: %d samples -> %s", n, sft)

        eval_path = self.output_dir / "odoo19_agent_eval.jsonl"
        n = self._write(eval_path, (self._training_row(s) for s in held_out))
        exported["eval"] = eval_path
        logger.info("Held-out eval: %d samples -> %s", n, eval_path)

        tc = self.output_dir / "odoo19_mcp_tool_calling.jsonl"
        n = self._write(tc, (self._training_row(s) for f in _TOOL_FAMILIES
                             for s in datasets.get(f, [])))
        exported["tool_calling"] = tc
        logger.info("Tool calling: %d samples -> %s", n, tc)

        traj = self.output_dir / "odoo19_agent_trajectories.jsonl"
        n = self._write(traj, (self._training_row(s) for f in _TRAJECTORY_FAMILIES
                               for s in datasets.get(f, [])))
        exported["trajectories"] = traj
        logger.info("Trajectories: %d samples -> %s", n, traj)

        # Sidecar: the provenance that used to be inlined into the training row.
        index = self.output_dir / "odoo19_sample_index.jsonl"
        with open(index, "w", encoding="utf-8") as fh:
            for split_name, rows in (("train", train), ("eval", held_out)):
                for s in rows:
                    meta = dict(s.get("_meta") or {})
                    meta["split"] = split_name
                    fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        exported["index"] = index

        if dpo_pairs:
            dpo = self.output_dir / "odoo19_dpo_pairs.jsonl"
            n = self._write(dpo, dpo_pairs)
            exported["dpo"] = dpo
            logger.info("DPO pairs: %d -> %s", n, dpo)

        self._write_manifest(datasets, train, held_out, exported)
        return exported

    def _write_manifest(self, datasets, train, held_out, exported) -> None:
        """A human-readable record of what this run actually produced."""
        by_family = Counter()
        by_model = Counter()
        by_persona = Counter()
        by_shape = Counter()
        for family, samples in datasets.items():
            by_family[family] = len(samples)
            for s in samples:
                meta = s.get("_meta") or {}
                by_model[meta.get("model", "?")] += 1
                by_persona[meta.get("persona", "?")] += 1
                by_shape[meta.get("shape", "?")] += 1

        manifest = {
            "totals": {
                "samples": len(train) + len(held_out),
                "train": len(train),
                "eval": len(held_out),
                "families": len(datasets),
            },
            "by_family": dict(by_family.most_common()),
            "by_model": dict(by_model.most_common()),
            "by_persona": dict(by_persona.most_common()),
            "by_shape": dict(by_shape.most_common()),
            "files": {k: str(v) for k, v in exported.items()},
        }
        path = self.output_dir / "dataset_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Manifest -> %s", path)
