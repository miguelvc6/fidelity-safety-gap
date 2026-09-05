#!/usr/bin/env python3
"""Read-only equality audit of validator-v3 labels and shared reconstruction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.evaluation_artifacts import atomic_write_json, repository_relative_path, sha256_file
from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION
from modules.semantics_provenance import expected_semantic_contracts


def _load_eval_module():
    path = SRC / "09_eval.py"
    spec = importlib.util.spec_from_file_location("eval_09_for_label_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import evaluation helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()
FIELDS = (
    ("factor_checkable_pre", "pre_checkable"),
    ("factor_satisfied_pre", "pre_satisfied"),
    ("factor_checkable_post_gold", "post_checkable"),
    ("factor_satisfied_post_gold", "post_satisfied"),
)


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _gold_slots(row: Any) -> list[int]:
    return [
        int(getattr(row, "add_subject", 0)),
        int(getattr(row, "add_predicate", 0)),
        int(getattr(row, "add_object", 0)),
        int(getattr(row, "del_subject", 0)),
        int(getattr(row, "del_predicate", 0)),
        int(getattr(row, "del_object", 0)),
    ]


def _counter() -> dict[str, int]:
    return {"rows": 0, "factor_positions": 0, "length_mismatches": 0, **{f"{a}_drift": 0 for a, _ in FIELDS}}


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    labeled_dir = Path(args.labeled_dir)
    label_manifest = json.loads((labeled_dir / "label_manifest.json").read_text(encoding="utf-8"))
    if label_manifest.get("validator_semantics_version") != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Label audit requires current validator semantics")
    if label_manifest.get("semantic_contracts") != expected_semantic_contracts():
        raise ValueError("Label audit requires current semantic contracts")
    support = EVAL._maybe_prepare_global_support(
        args.dataset,
        args.min_occurrence,
        split="test",
        none_class=0,
        strict=True,
        registry_dataset=args.registry_dataset,
    )
    if support is None:
        raise RuntimeError("Unable to construct corrected symbolic evaluator.")
    if support.evaluator.hierarchy_identity != label_manifest.get("hierarchy"):
        raise ValueError("Label audit hierarchy differs from the shared evaluator hierarchy")

    report: dict[str, object] = {
        "schema_version": 2,
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "semantic_contracts": expected_semantic_contracts(),
        "hierarchy": label_manifest.get("hierarchy"),
        "mode": "read_only",
        "labeled_directory": repository_relative_path(labeled_dir),
        "label_manifest": {
            "path": repository_relative_path(labeled_dir / "label_manifest.json"),
            "sha256": sha256_file(labeled_dir / "label_manifest.json"),
        },
        "mismatch_count": 0,
        "splits": {},
    }
    for split in args.splits:
        path = labeled_dir / f"df_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        dataframe = pd.read_parquet(path)
        overall = _counter()
        by_family: dict[str, dict[str, int]] = defaultdict(_counter)
        for row in dataframe.itertuples(index=False):
            family = str(getattr(row, "constraint_type", None) or "UNKNOWN")
            factor_ids = _sequence(getattr(row, "factor_constraint_ids", None))
            details = support.evaluator.evaluate_full(
                row,
                candidate_slots=_gold_slots(row),
                primary_factor_index=None,
                factor_constraint_ids=[int(value) for value in factor_ids],
            )
            for bucket in (overall, by_family[family]):
                bucket["rows"] += 1
                bucket["factor_positions"] += len(factor_ids)
            for stored_name, corrected_name in FIELDS:
                stored = _sequence(getattr(row, stored_name, None))
                corrected = _sequence(details.get(corrected_name))
                if len(stored) != len(corrected):
                    for bucket in (overall, by_family[family]):
                        bucket["length_mismatches"] += 1
                    continue
                drift = sum(int(bool(left) != bool(right)) for left, right in zip(stored, corrected))
                for bucket in (overall, by_family[family]):
                    bucket[f"{stored_name}_drift"] += drift
        report["splits"][split] = {
            "path": repository_relative_path(path),
            "overall": overall,
            "by_constraint_family": dict(sorted(by_family.items())),
        }
        split_mismatches = overall["length_mismatches"] + sum(
            overall[f"{name}_drift"] for name, _ in FIELDS
        )
        report["mismatch_count"] = int(report["mismatch_count"]) + split_mismatches
        if split_mismatches:
            raise AssertionError(f"Stored labels drift from shared validator reconstruction in {split}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="full_strat1m")
    parser.add_argument("--min-occurrence", type=int, default=100)
    parser.add_argument("--registry-dataset", default="full")
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=Path("data/interim/full_strat1m_minocc100_labeled"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(args)
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
