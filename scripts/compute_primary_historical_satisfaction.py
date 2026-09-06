#!/usr/bin/env python3
"""Stream current primary-constraint outcomes before and after gold edits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION
from modules.data_encoders import GlobalIntEncoder
from modules.evaluation_artifacts import atomic_write_json, sha256_file
from modules.reranker_eval import CandidateConstraintEvaluator
from modules.semantics_provenance import expected_semantic_contracts


OUTCOMES = ("satisfied", "violated", "unknown")
ROW_COLUMNS = (
    "constraint_type",
    "constraint_id",
    "subject",
    "predicate",
    "object",
    "other_subject",
    "other_predicate",
    "other_object",
    "subject_predicates",
    "subject_objects",
    "object_predicates",
    "object_objects",
    "other_entity_predicates",
    "other_entity_objects",
    "add_subject",
    "add_predicate",
    "add_object",
    "del_subject",
    "del_predicate",
    "del_object",
)
GOLD_COLUMNS = (
    "add_subject",
    "add_predicate",
    "add_object",
    "del_subject",
    "del_predicate",
    "del_object",
)


def _gold_slots(row: Any) -> tuple[int, int, int, int, int, int]:
    return tuple(int(getattr(row, name, 0) or 0) for name in GOLD_COLUMNS)  # type: ignore[return-value]


def _update(
    counts: Counter[str],
    transitions: Counter[str],
    pre: str,
    post: str,
) -> None:
    counts["total"] += 1
    counts[f"pre_{pre}"] += 1
    counts[f"post_{post}"] += 1
    counts["pre_checkable"] += int(pre != "unknown")
    counts["post_checkable"] += int(post != "unknown")
    transitions[f"{pre}->{post}"] += 1


def _summary(counts: Counter[str], transitions: Counter[str]) -> dict[str, Any]:
    total = int(counts["total"])
    pre_checkable = int(counts["pre_checkable"])
    post_checkable = int(counts["post_checkable"])
    pre_satisfied = int(counts["pre_satisfied"])
    post_satisfied = int(counts["post_satisfied"])
    return {
        "total": total,
        "pre": {
            "satisfied": pre_satisfied,
            "violated": int(counts["pre_violated"]),
            "unknown": int(counts["pre_unknown"]),
            "checkable": pre_checkable,
            "satisfied_over_total": pre_satisfied / total if total else None,
            "satisfied_over_checkable": (
                pre_satisfied / pre_checkable if pre_checkable else None
            ),
        },
        "post": {
            "satisfied": post_satisfied,
            "violated": int(counts["post_violated"]),
            "unknown": int(counts["post_unknown"]),
            "checkable": post_checkable,
            "satisfied_over_total": post_satisfied / total if total else None,
            "satisfied_over_checkable": (
                post_satisfied / post_checkable if post_checkable else None
            ),
        },
        "transitions": {
            f"{pre}->{post}": int(transitions[f"{pre}->{post}"])
            for pre in OUTCOMES
            for post in OUTCOMES
        },
    }


def _input_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    encoder = GlobalIntEncoder()
    encoder.load(args.encoder)
    encoder.freeze()
    evaluator = CandidateConstraintEvaluator(
        str(args.registry),
        encoder=encoder,
        assume_complete=True,
        constraint_scope="local",
        use_encoded_ids=True,
        hierarchy_path=args.hierarchy,
        require_hierarchy=True,
    )

    combined: dict[str, Counter[str]] = defaultdict(Counter)
    combined_transitions: dict[str, Counter[str]] = defaultdict(Counter)
    split_results: dict[str, Any] = {}
    started = time.monotonic()

    for split in args.splits:
        path = args.benchmark_dir / f"df_{split}.parquet"
        parquet = pq.ParquetFile(path)
        split_counts: dict[str, Counter[str]] = defaultdict(Counter)
        split_transitions: dict[str, Counter[str]] = defaultdict(Counter)
        scanned = 0
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=list(ROW_COLUMNS)):
            frame = batch.to_pandas()
            if args.max_rows is not None:
                remaining = args.max_rows - scanned
                if remaining <= 0:
                    break
                frame = frame.head(remaining)
            for row in frame.itertuples(index=False):
                constraint_id = int(row.constraint_id)
                instance = evaluator.constraint_instance(constraint_id)
                family = (
                    instance.constraint_type
                    if instance is not None and instance.constraint_type
                    else str(row.constraint_type or "unknown")
                )
                details = evaluator.evaluate_full(
                    row,
                    candidate_slots=_gold_slots(row),
                    primary_factor_index=0,
                    factor_constraint_ids=[constraint_id],
                )
                pre = str(details["pre_outcomes"][0])
                post = str(details["post_outcomes"][0])
                _update(split_counts[family], split_transitions[family], pre, post)
                _update(combined[family], combined_transitions[family], pre, post)
            scanned += len(frame)
            print(
                f"split={split} scanned={scanned}/{parquet.metadata.num_rows}",
                file=sys.stderr,
                flush=True,
            )
            if args.max_rows is not None and scanned >= args.max_rows:
                break
        split_results[split] = {
            "source": _input_identity(path),
            "rows_scanned": scanned,
            "families": {
                family: _summary(counts, split_transitions[family])
                for family, counts in sorted(split_counts.items())
            },
        }

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    code_paths = (
        Path("src/modules/constraint_checkers.py"),
        Path("src/modules/evidence_state.py"),
        Path("src/modules/class_hierarchy.py"),
        Path("src/modules/reranker_eval.py"),
        Path(__file__).resolve(),
    )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "semantic_contracts": expected_semantic_contracts(),
        "code": {
            "commit": commit,
            "working_tree_dirty": dirty,
            "files": {
                path.name: _input_identity(path)
                for path in code_paths
            },
        },
        "registry": _input_identity(args.registry),
        "encoder": _input_identity(args.encoder),
        "hierarchy": evaluator.hierarchy_identity,
        "assume_complete_entity_facts": True,
        "elapsed_seconds": time.monotonic() - started,
        "splits": split_results,
        "combined": {
            "rows_scanned": sum(item["rows_scanned"] for item in split_results.values()),
            "families": {
                family: _summary(counts, combined_transitions[family])
                for family, counts in sorted(combined.items())
            },
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/interim/full_strat1m_minocc100"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/interim/constraint_registry_full.parquet"),
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=Path("data/interim/full_strat1m_minocc100/globalintencoder.txt"),
    )
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=Path("data/static/wikidata-p279-2018-07-01.v2.json"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = run(args)
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
