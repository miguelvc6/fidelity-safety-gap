#!/usr/bin/env python3
"""Small label-to-factor-graph fixture for validator semantics v2."""

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
    from modules.class_hierarchy import HIERARCHY_CUTOFF, canonical_sha256

    with tempfile.TemporaryDirectory(prefix="validator-v2-smoke-") as directory:
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
            "schema_version": 1,
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
        assert label_manifest["validator_semantics_version"] == 2
        assert graph_manifest["validator_semantics_version"] == 2
        assert (
            graph_manifest["validator_provenance"]["hierarchy"]["content_sha256"]
            == label_manifest["hierarchy"]["content_sha256"]
        )
        assert graph_manifest["graph_count"] == 12
    print("validator-v2 smoke pipeline: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
