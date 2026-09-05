#!/usr/bin/env python3
"""Report v3 primary transitions, uncertainty, and single-value failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION
from modules.constraint_identity import resolve_primary_index
from modules.evidence_state import build_post_state_from_slots, build_pre_state
from modules.evaluation_artifacts import atomic_write_json
from modules.semantics_provenance import expected_semantic_contracts


OUTCOMES = ("satisfied", "violated", "unknown")


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _gold_slots(row: Any) -> tuple[int, ...]:
    return tuple(
        int(getattr(row, name, 0) or 0)
        for name in (
            "add_subject",
            "add_predicate",
            "add_object",
            "del_subject",
            "del_predicate",
            "del_object",
        )
    )


def _single_details(row: Any, primary_index: int, post_outcome: str) -> dict[str, Any]:
    pre, p_local = build_pre_state(row, assume_complete=True, cast_int=True)
    post, resolved = build_post_state_from_slots(pre, p_local=p_local, candidate_slots=_gold_slots(row))
    prop = int(getattr(row, "predicate"))
    subject = int(getattr(row, "subject"))
    pre_values = sorted(pre.values_for(subject, prop))
    post_values = sorted(post.values_for(subject, prop))
    deleted = resolved.get("del")
    unknown_reason = str(
        _sequence(getattr(row, "factor_unknown_reason_post_gold", []))[primary_index]
    )
    representation_limit = (
        unknown_reason
        if any(
            marker in unknown_reason.lower()
            for marker in ("p4155", "separator", "representation", "unrepresentable")
        )
        else None
    )
    return {
        "constraint_id": int(getattr(row, "constraint_id")),
        "primary_factor_index": primary_index,
        "post_outcome": post_outcome,
        "pre_cardinality": len(pre_values),
        "post_cardinality": len(post_values),
        "edit_type": "replace" if resolved.get("add") and deleted else "add" if resolved.get("add") else "delete" if deleted else "none",
        "deleted_triple": list(deleted) if deleted else None,
        "deleted_triple_present_pre": bool(deleted and pre.has_statement(*deleted)),
        "surviving_values": post_values,
        "surviving_witness_values": post_values if len(post_values) > 1 else [],
        "unknown_reason": unknown_reason,
        "separator_or_representation_limitation": representation_limit,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.labeled_dir / "label_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("validator_semantics_version", -1)) != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Diagnostics require regenerated labels under current validator semantics")
    if manifest.get("semantic_contracts") != expected_semantic_contracts():
        raise ValueError("Diagnostics require current semantic contract versions")

    report: dict[str, Any] = {
        "schema_version": 1,
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "semantic_contracts": expected_semantic_contracts(),
        "bounded": not args.full_scan,
        "max_rows_per_split": None if args.full_scan else args.max_rows,
        "splits": {},
    }
    for split in args.splits:
        frame = pd.read_parquet(args.labeled_dir / f"df_{split}.parquet")
        if not args.full_scan:
            frame = frame.head(args.max_rows)
        transitions: Counter[str] = Counter()
        by_family: dict[str, Counter[str]] = defaultdict(Counter)
        unknown_reasons: dict[str, Counter[str]] = defaultdict(Counter)
        primary_mismatches: list[dict[str, Any]] = []
        single_failures: list[dict[str, Any]] = []
        for row_number, row in enumerate(frame.itertuples(index=False)):
            ids = [int(value) for value in _sequence(getattr(row, "factor_constraint_ids"))]
            supplied = int(getattr(row, "primary_factor_index"))
            try:
                primary = resolve_primary_index(row, ids, supplied_index=supplied)
            except ValueError as exc:
                primary_mismatches.append({"row": row_number, "error": str(exc)})
                continue
            pre = str(_sequence(getattr(row, "factor_outcome_pre"))[primary])
            post = str(_sequence(getattr(row, "factor_outcome_post_gold"))[primary])
            family = str(getattr(row, "constraint_type", "unknown"))
            key = f"{pre}->{post}"
            transitions[key] += 1
            by_family[family][key] += 1
            if post == "unknown":
                reason = str(_sequence(getattr(row, "factor_unknown_reason_post_gold"))[primary] or "unspecified")
                unknown_reasons[family][reason] += 1
            if family == "single" and post != "satisfied" and len(single_failures) < args.single_examples:
                single_failures.append(_single_details(row, primary, post))
        zero_filled = {
            f"{pre}->{post}": transitions[f"{pre}->{post}"]
            for pre in OUTCOMES
            for post in OUTCOMES
        }
        report["splits"][split] = {
            "rows_scanned": len(frame),
            "transitions_3x3": zero_filled,
            "transitions_by_family": {
                family: {
                    f"{pre}->{post}": counts[f"{pre}->{post}"]
                    for pre in OUTCOMES
                    for post in OUTCOMES
                }
                for family, counts in sorted(by_family.items())
            },
            "post_unknown_reasons_by_family": {
                family: dict(counts.most_common()) for family, counts in sorted(unknown_reasons.items())
            },
            "primary_index_mismatch_count": len(primary_mismatches),
            "primary_index_mismatches": primary_mismatches[: args.mismatch_examples],
            "single_value_failure_examples": single_failures,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=Path("data/interim/full_strat1m_minocc100_labeled"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--full-scan", action="store_true")
    parser.add_argument("--single-examples", type=int, default=20)
    parser.add_argument("--mismatch-examples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    report = run(args)
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
