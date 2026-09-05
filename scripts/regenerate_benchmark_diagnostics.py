#!/usr/bin/env python3
"""Regenerate benchmark statistics and validator-v3 provenance sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.class_hierarchy import load_hierarchy_artifact  # noqa: E402
from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION  # noqa: E402
from modules.data_encoders import graph_dataset_filename  # noqa: E402
from modules.evaluation_artifacts import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    atomic_write_json,
    repository_relative_path,
    sha256_file,
)
from modules.evidence_state import build_pre_state  # noqa: E402
from modules.semantics_provenance import expected_semantic_contracts  # noqa: E402
from modules.training_utils import load_graph_dataset  # noqa: E402


DEFAULT_ARCHIVED_REPORT = (
    ROOT
    / "deprecated"
    / "pre-validator-rewrite-2026-09-04"
    / "models"
    / "paper_diagnostics"
    / "benchmark_provenance_and_statistics.json"
)


def _summary(values: Iterable[int]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.int64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty benchmark population")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": int(array.max()),
    }


def _identity(path: Path) -> dict[str, str | int]:
    return {
        "path": repository_relative_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "paper_diagnostics" / "benchmark_provenance_and_statistics.json",
    )
    parser.add_argument("--archived-report", type=Path, default=DEFAULT_ARCHIVED_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archived_report = json.loads(args.archived_report.read_text(encoding="utf-8"))
    benchmark_dir = ROOT / "data" / "interim" / "full_strat1m_minocc100"
    labeled_dir = ROOT / "data" / "interim" / "full_strat1m_minocc100_labeled"
    processed_dir = ROOT / "data" / "processed" / "full_strat1m_minocc100"
    test_path = benchmark_dir / "df_test.parquet"
    labeled_test_path = labeled_dir / "df_test.parquet"
    graph_path = processed_dir / graph_dataset_filename(
        "test", "node_id", constraint_representation="factorized"
    )
    graph_manifest_path = graph_path.with_suffix(graph_path.suffix + ".manifest.json")
    hierarchy_path = ROOT / "data" / "static" / "wikidata-p279-2018-07-01.v2.json"
    _hierarchy_payload, hierarchy = load_hierarchy_artifact(hierarchy_path)

    test_frame = pd.read_parquet(test_path)
    local_statements: list[int] = []
    attached_constraints: list[int] = []
    for row in test_frame.itertuples(index=False):
        state, _p_local = build_pre_state(row, assume_complete=True, cast_int=True)
        local_statements.append(
            sum(len(objects) for facts in state.facts_by_entity.values() for objects in facts.values())
        )
        attached_constraints.append(len(getattr(row, "local_constraint_ids", ())))

    labeled_frame = pd.read_parquet(labeled_test_path, columns=["factor_constraint_ids"])
    executable_factors = [len(value) for value in labeled_frame["factor_constraint_ids"]]
    del labeled_frame

    graph_data = load_graph_dataset(graph_path)
    graph_nodes: list[int] = []
    graph_edges: list[int] = []
    for graph in graph_data:
        graph_nodes.append(int(graph.num_nodes))
        graph_edges.append(int(graph.edge_index.shape[1]))

    sampling_metadata_path = benchmark_dir / "sampling_metadata.json"
    sampling_metadata = json.loads(sampling_metadata_path.read_text(encoding="utf-8"))
    label_manifest_path = labeled_dir / "label_manifest.json"
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    if int(label_manifest.get("validator_semantics_version", -1)) != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Label manifest has incompatible validator semantics")
    if label_manifest.get("semantic_contracts") != expected_semantic_contracts():
        raise ValueError("Label manifest has incompatible semantic contracts")
    if (label_manifest.get("hierarchy") or {}).get("content_sha256") != hierarchy.content_sha256:
        raise ValueError("Label manifest and fixed hierarchy differ")
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    if int(graph_manifest.get("validator_semantics_version", -1)) != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Factor graph manifest has incompatible validator semantics")
    graph_provenance = graph_manifest.get("validator_provenance") or {}
    if graph_provenance.get("semantic_contracts") != expected_semantic_contracts():
        raise ValueError("Factor graph manifest has incompatible semantic contracts")
    if (graph_provenance.get("hierarchy") or {}).get("content_sha256") != hierarchy.content_sha256:
        raise ValueError("Factor graph manifest and fixed hierarchy differ")

    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "semantic_contracts": expected_semantic_contracts(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": archived_report["dataset"],
        # These source-corpus counts concern immutable raw inputs.  Preserve
        # them from the checksummed pre-rewrite audit and identify that source
        # explicitly; all validator/graph-dependent values below are recomputed.
        "source_row_audit": archived_report["source_row_audit"],
        "source_row_audit_provenance": _identity(args.archived_report),
        "test_split_statistics": {
            "rows": len(test_frame),
            "local_a_box_statements": _summary(local_statements),
            "locally_attached_constraints": _summary(attached_constraints),
            "executable_factors": _summary(executable_factors),
            "factorized_graph_nodes": _summary(graph_nodes),
            "factorized_graph_directed_edges": _summary(graph_edges),
        },
        "definitions": archived_report["definitions"],
        "input_identity": {
            "sampling_metadata": _identity(sampling_metadata_path),
            "unlabeled_test_parquet": _identity(test_path),
            "labeled_test_parquet": _identity(labeled_test_path),
            "label_manifest": _identity(label_manifest_path),
            "factorized_test_graph_manifest": _identity(graph_manifest_path),
            "constraint_registry": _identity(ROOT / "data" / "interim" / "constraint_registry_full.parquet"),
            "encoder": _identity(benchmark_dir / "globalintencoder.txt"),
            "hierarchy": {**_identity(hierarchy_path), "content_sha256": hierarchy.content_sha256},
        },
        "sampling_manifest": sampling_metadata,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
