#!/usr/bin/env python3
"""Small cross-path pipeline fixture for validator semantics v3.

The filename is retained so existing automation keeps finding the smoke entry
point; the assertions require the current semantic contracts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from modules.class_hierarchy import (
        HIERARCHY_CACHE_SCHEMA_VERSION,
        HIERARCHY_CUTOFF,
        HIERARCHY_PARSER_VERSION,
        HIERARCHY_SCHEMA_VERSION,
        canonical_sha256,
    )
    from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION
    from modules.data_encoders import GlobalIntEncoder, iter_stream
    from modules.reranker_eval import CandidateConstraintEvaluator
    from modules.semantics_provenance import expected_semantic_contracts
    from torch_geometric.data import Batch

    with tempfile.TemporaryDirectory(prefix="validator-v3-smoke-") as directory:
        work = Path(directory)
        interim = work / "data/interim"
        benchmark = interim / "full_strat1m_minocc100"
        benchmark.mkdir(parents=True)
        shutil.copy2(
            ROOT / "data/interim/constraint_registry_full.parquet",
            interim / "constraint_registry_full.parquet",
        )
        shutil.copy2(
            ROOT / "data/interim/full_strat1m_minocc100/globalintencoder.txt",
            benchmark / "globalintencoder.txt",
        )
        for split in ("train", "val", "test"):
            source = ROOT / f"data/interim/full_strat1m_minocc100/df_{split}.parquet"
            pd.read_parquet(source).head(12).to_parquet(benchmark / source.name, index=False)

        hierarchy_payload = {
            "schema_version": HIERARCHY_SCHEMA_VERSION,
            "parser_version": HIERARCHY_PARSER_VERSION,
            "cache_schema_version": HIERARCHY_CACHE_SCHEMA_VERSION,
            "cutoff": HIERARCHY_CUTOFF,
            "records": {},
            "seed_ids": [],
            "seed_count": 0,
        }
        hierarchy_payload["content_sha256"] = canonical_sha256(hierarchy_payload)
        hierarchy = work / "fixture-hierarchy.json"
        hierarchy.write_text(json.dumps(hierarchy_payload), encoding="utf-8")

        _run(
            [
                sys.executable,
                str(ROOT / "src/05_constraint_labeler.py"),
                "--dataset",
                "full_strat1m",
                "--registry-dataset",
                "full",
                "--min-occurrence",
                "100",
                "--hierarchy",
                str(hierarchy),
            ],
            work,
        )
        _run(
            [
                sys.executable,
                str(ROOT / "src/06_graph.py"),
                "--dataset",
                "full_strat1m",
                "--registry-dataset",
                "full",
                "--min-occurrence",
                "100",
                "--encoding",
                "node_id",
                "--constraint-representation",
                "factorized",
                "--hierarchy",
                str(hierarchy),
                "--persistence-profile",
                "full",
                "--debug-factor-wiring",
            ],
            work,
        )

        label_manifest = json.loads(
            (interim / "full_strat1m_minocc100_labeled/label_manifest.json").read_text()
        )
        graph_manifest = json.loads(
            (work / "data/processed/full_strat1m_minocc100/test_graph-node_id.pkl.manifest.json").read_text()
        )
        assert label_manifest["validator_semantics_version"] == VALIDATOR_SEMANTICS_VERSION
        assert graph_manifest["validator_semantics_version"] == VALIDATOR_SEMANTICS_VERSION
        assert label_manifest["semantic_contracts"] == expected_semantic_contracts()
        assert graph_manifest["validator_provenance"]["semantic_contracts"] == expected_semantic_contracts()
        assert (
            graph_manifest["validator_provenance"]["hierarchy"]["content_sha256"]
            == label_manifest["hierarchy"]["content_sha256"]
        )
        assert graph_manifest["graph_count"] == 12

        labeled = pd.read_parquet(interim / "full_strat1m_minocc100_labeled/df_test.parquet")
        encoder = GlobalIntEncoder()
        encoder.load(benchmark / "globalintencoder.txt")
        evaluator = CandidateConstraintEvaluator(
            str(interim / "constraint_registry_full.parquet"),
            encoder=encoder,
            assume_complete=True,
            constraint_scope="local",
            use_encoded_ids=True,
            hierarchy_path=hierarchy,
            require_hierarchy=True,
        )
        graph_path = work / "data/processed/full_strat1m_minocc100/test_graph-node_id.pkl"
        graphs = list(iter_stream(graph_path))
        restored = Batch.from_data_list(graphs).to_data_list()
        saw_filtered_primary_shift = False
        for row, graph in zip(labeled.itertuples(index=False), restored):
            ids = [int(value) for value in row.factor_constraint_ids]
            primary = int(row.primary_factor_index)
            raw_ids = [int(value) for value in row.local_constraint_ids]
            saw_filtered_primary_shift |= raw_ids.index(int(row.constraint_id)) != primary
            assert ids == graph.factor_constraint_ids.tolist()
            assert ids[primary] == int(row.constraint_id) == int(graph.shape_id)
            assert primary == int(graph.primary_factor_index)
            slots = tuple(
                int(getattr(row, name))
                for name in (
                    "add_subject",
                    "add_predicate",
                    "add_object",
                    "del_subject",
                    "del_predicate",
                    "del_object",
                )
            )
            single = evaluator.evaluate_full(
                row,
                candidate_slots=slots,
                primary_factor_index=primary,
            )
            batched = evaluator.evaluate_candidates(
                row,
                candidates=[slots],
                primary_factor_index=primary,
            )[0]
            metrics = evaluator.evaluate_candidate_metrics(
                row,
                candidates=[slots],
                primary_factor_index=primary,
            )[0]
            assert single == batched
            assert single["pre_outcomes"] == list(row.factor_outcome_pre)
            assert single["post_outcomes"] == list(row.factor_outcome_post_gold)
            assert graph.factor_checkable_pre.tolist() == list(row.factor_checkable_pre)
            assert graph.factor_satisfied_pre.tolist() == list(row.factor_satisfied_pre)
            assert metrics.primary_pre_violated == int(
                row.factor_outcome_pre[primary] == "violated"
            )
            assert metrics.primary_checkable == int(
                row.factor_outcome_post_gold[primary] != "unknown"
            )
            assert metrics.primary_satisfied == int(
                row.factor_outcome_post_gold[primary] == "satisfied"
            )
        assert saw_filtered_primary_shift
    print("validator-v3 smoke pipeline: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
