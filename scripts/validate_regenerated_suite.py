#!/usr/bin/env python3
"""Acceptance gate for the complete validator-v3 paper regeneration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.class_hierarchy import load_hierarchy_artifact, sha256_file  # noqa: E402
from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION  # noqa: E402
from modules.evaluation_artifacts import EVALUATION_SCHEMA_VERSION  # noqa: E402
from modules.evidence_state import build_pre_state  # noqa: E402
from modules.semantics_provenance import expected_semantic_contracts  # noqa: E402


ARCHIVE = ROOT / "deprecated" / "pre-validator-rewrite-2026-09-04"
RUNS = (
    "b0_eswc_reproduction__full_strat1m_minocc100__node_id",
    "a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id",
    "m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id",
    "m1d_safe_factor_direct_compact_grouped__full_strat1m_minocc100__node_id",
    "g0_globalfix_reference_v2__full_strat1m_minocc100__node_id",
)
BASELINES = (
    "baseline-DeleteFocusBaseline",
    "baseline-AddMirrorBaseline",
    "baseline-ConstraintFamilyMajorityBaseline",
    "baseline-ConstraintDefinitionMajorityBaseline",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_immutable_inputs(archive_manifest: dict) -> None:
    for record in archive_manifest["immutable_inputs"]:
        path = ROOT / record["path"]
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Immutable input changed: {record['path']}")


def _verify_base_statements() -> None:
    base = ROOT / "data/interim/full_strat1m_minocc100"
    for path in sorted(base.glob("df_*.parquet")):
        for row in pd.read_parquet(path).itertuples(index=False):
            build_pre_state(row, assume_complete=True, cast_int=True)


def _archived_file_map(manifest: dict) -> dict[str, dict]:
    prefix = "deprecated/pre-validator-rewrite-2026-09-04/"
    return {
        record["path"].removeprefix(prefix): record
        for record in manifest["files"]
    }


def _verify_passive_payloads(archive_manifest: dict) -> None:
    archived = _archived_file_map(archive_manifest)
    active_root = ROOT / "data/processed/full_strat1m_minocc100"
    archived_paths = {
        path
        for path in archived
        if "_graph_repr-eswc_passive-node_id-shard" in path
        and Path(path).suffix in {".pt", ".pkl"}
    }
    active_paths = {
        path.relative_to(ROOT).as_posix()
        for path in active_root.glob("*_graph_repr-eswc_passive-node_id-shard*")
        if path.suffix in {".pt", ".pkl"}
    }
    if active_paths != archived_paths:
        raise ValueError(
            "Passive graph payload set differs from the archived generation: "
            f"active={len(active_paths)} archived={len(archived_paths)}"
        )
    for relative in sorted(active_paths):
        path = ROOT / relative
        expected = archived.get(relative)
        if expected is None or path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"Passive graph payload is not archived-identical: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Passive graph payload checksum changed: {relative}")


def _verify_semantics(hierarchy_checksum: str) -> None:
    expected_contracts = expected_semantic_contracts()
    label_manifest = _load(ROOT / "data/interim/full_strat1m_minocc100_labeled/label_manifest.json")
    if label_manifest.get("validator_semantics_version") != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Label suite has incompatible validator semantics")
    if label_manifest.get("semantic_contracts") != expected_contracts:
        raise ValueError("Label suite has incompatible semantic contracts")
    if (label_manifest.get("hierarchy") or {}).get("content_sha256") != hierarchy_checksum:
        raise ValueError("Label suite hierarchy mismatch")
    for manifest_path in sorted((ROOT / "data/processed/full_strat1m_minocc100").glob("*.manifest.json")):
        manifest = _load(manifest_path)
        if manifest.get("validator_semantics_version") != VALIDATOR_SEMANTICS_VERSION:
            raise ValueError(f"Graph manifest has incompatible validator semantics: {manifest_path}")
        provenance = manifest.get("validator_provenance") or {}
        if provenance.get("semantic_contracts") != expected_contracts:
            raise ValueError(f"Graph manifest has incompatible semantic contracts: {manifest_path}")
        hierarchy = provenance.get("hierarchy") or {}
        if hierarchy.get("content_sha256") != hierarchy_checksum:
            raise ValueError(f"Graph hierarchy mismatch: {manifest_path}")
        build_identity = manifest.get("build_contract") or {}
        build_path_raw = build_identity.get("path")
        if not build_path_raw:
            raise ValueError(f"Graph manifest lacks a build contract: {manifest_path}")
        build_path = Path(str(build_path_raw))
        if not build_path.exists() or sha256_file(build_path) != build_identity.get("sha256"):
            raise ValueError(f"Graph build contract is missing or changed: {build_path}")
        build_contract = _load(build_path)
        if build_contract.get("validator_provenance") != manifest.get("validator_provenance"):
            raise ValueError(f"Graph build contract provenance mismatch: {build_path}")
        source = build_contract.get("source_parquet") or {}
        source_path = Path(str(source.get("path", "")))
        if not source_path.exists() or sha256_file(source_path) != source.get("sha256"):
            raise ValueError(f"Graph source Parquet differs from its build contract: {source_path}")


def _verify_diagnostics(hierarchy_checksum: str, *, require_paper: bool) -> None:
    expected_contracts = expected_semantic_contracts()
    diagnostics = ROOT / "models" / "paper_diagnostics"
    benchmark = _load(diagnostics / "benchmark_provenance_and_statistics.json")
    readiness = _load(diagnostics / "corrected_paper_readiness.json")
    label_audit = _load(diagnostics / "label_semantics_audit.json")
    if benchmark.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Benchmark diagnostic is not schema v3")
    if benchmark.get("validator_semantics_version") != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Benchmark diagnostic has incompatible validator semantics")
    if benchmark.get("semantic_contracts") != expected_contracts:
        raise ValueError("Benchmark diagnostic has incompatible semantic contracts")
    hierarchy = ((benchmark.get("input_identity") or {}).get("hierarchy") or {})
    if hierarchy.get("content_sha256") != hierarchy_checksum:
        raise ValueError("Benchmark diagnostic hierarchy mismatch")
    if readiness.get("schema_version") != EVALUATION_SCHEMA_VERSION or readiness.get("ready") is not True:
        raise ValueError("Corrected paper readiness has not passed under schema v3")
    if readiness.get("validator_semantics_version") != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Corrected paper readiness has incompatible validator semantics")
    if readiness.get("semantic_contracts") != expected_contracts:
        raise ValueError("Corrected paper readiness has incompatible semantic contracts")
    if require_paper and not readiness.get("paper"):
        raise ValueError(
            "Final acceptance requires a paper-bound readiness report; rerun "
            "check_corrected_paper_readiness.py with --paper."
        )
    if label_audit.get("validator_semantics_version") != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Label semantics audit has incompatible validator semantics")
    if label_audit.get("semantic_contracts") != expected_contracts:
        raise ValueError("Label semantics audit has incompatible semantic contracts")
    if (label_audit.get("hierarchy") or {}).get("content_sha256") != hierarchy_checksum:
        raise ValueError("Label semantics audit hierarchy mismatch")
    if int(label_audit.get("mismatch_count", -1)) != 0:
        raise ValueError("Stored labels drift from the shared validator")
    factor_summary = pd.read_csv(diagnostics / "factor_semantics_summary.csv")
    if factor_summary.empty:
        raise ValueError("Factor-semantics summary is empty")
    if set(factor_summary.get("validator_semantics_version", [])) != {
        VALIDATOR_SEMANTICS_VERSION
    }:
        raise ValueError("Factor-semantics summary has incompatible validator semantics")
    if set(factor_summary.get("semantic_contracts", [])) != {
        json.dumps(expected_contracts, sort_keys=True)
    }:
        raise ValueError("Factor-semantics summary has incompatible semantic contracts")
    if set(factor_summary.get("hierarchy_content_sha256", [])) != {hierarchy_checksum}:
        raise ValueError("Factor-semantics summary has incompatible hierarchy")

    g0_evaluation = ROOT / "models" / RUNS[-1] / "evaluations"
    membership = _load(g0_evaluation / "candidate_membership_audit.json")
    if membership.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Candidate-membership audit is not schema v3")
    if membership.get("semantic_contracts") != expected_contracts:
        raise ValueError("Candidate-membership audit has incompatible semantic contracts")
    if membership.get("status") != "ok" or float(membership.get("membership_rate", 0.0)) != 1.0:
        raise ValueError("Candidate-SR predictions are not all label-blind candidates")
    if (membership.get("hierarchy") or {}).get("content_sha256") != hierarchy_checksum:
        raise ValueError("Candidate-membership hierarchy mismatch")
    deletion = _load(
        g0_evaluation / "deletion_degeneracy" / "deletion_degeneracy_summary.json"
    )
    if deletion.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Deletion-degeneracy diagnostic is not schema v3")
    if deletion.get("semantic_contracts") != expected_contracts:
        raise ValueError("Deletion-degeneracy diagnostic has incompatible semantic contracts")
    if (deletion.get("hierarchy") or {}).get("content_sha256") != hierarchy_checksum:
        raise ValueError("Deletion-degeneracy hierarchy mismatch")


def _verify_runs(hierarchy_checksum: str) -> None:
    expected_contracts = expected_semantic_contracts()
    expected_contracts_json = json.dumps(expected_contracts, sort_keys=True)
    active = sorted(
        path.name
        for path in (ROOT / "models").iterdir()
        if path.is_dir() and path.name not in {"baselines", "paper_diagnostics"}
    )
    if active != sorted(RUNS):
        raise ValueError(f"Active learned run set differs from the paper suite: {active}")
    for run in RUNS:
        evaluation = ROOT / "models" / run / "evaluations"
        model = _load(evaluation / "model.json")
        manifest = _load(evaluation / "predictions.manifest.json")
        if model.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"Legacy model metrics entered active run {run}")
        if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"Legacy prediction artifact entered active run {run}")
        if manifest.get("validator_semantics_version") != VALIDATOR_SEMANTICS_VERSION:
            raise ValueError(f"Wrong validator semantics in {run}")
        if model.get("semantic_contracts") != expected_contracts:
            raise ValueError(f"Wrong semantic contracts in {run}/model.json")
        if manifest.get("semantic_contracts") != expected_contracts:
            raise ValueError(f"Wrong semantic contracts in {run}/predictions.manifest.json")
        if (manifest.get("hierarchy") or {}).get("content_sha256") != hierarchy_checksum:
            raise ValueError(f"Wrong hierarchy in {run}")
        if not (evaluation / "historical_strata.csv").exists():
            raise FileNotFoundError(f"Missing historical strata for {run}")
        for csv_name in ("per_constraint.csv", "historical_strata.csv"):
            frame = pd.read_csv(evaluation / csv_name)
            if set(frame.get("validator_semantics_version", [])) != {VALIDATOR_SEMANTICS_VERSION}:
                raise ValueError(f"Wrong validator provenance in {run}/{csv_name}")
            if set(frame.get("semantic_contracts", [])) != {expected_contracts_json}:
                raise ValueError(f"Wrong semantic contracts in {run}/{csv_name}")
            if set(frame.get("hierarchy_content_sha256", [])) != {hierarchy_checksum}:
                raise ValueError(f"Wrong hierarchy provenance in {run}/{csv_name}")

    baseline_root = ROOT / "models/baselines/full_strat1m/parquet"
    for baseline in BASELINES:
        model = _load(baseline_root / f"{baseline}.json")
        manifest = _load(baseline_root / baseline / "predictions.manifest.json")
        if model.get("schema_version") != EVALUATION_SCHEMA_VERSION or manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"Legacy baseline artifact entered {baseline}")
        if model.get("semantic_contracts") != expected_contracts:
            raise ValueError(f"Wrong semantic contracts in {baseline}.json")
        if manifest.get("semantic_contracts") != expected_contracts:
            raise ValueError(f"Wrong semantic contracts in {baseline}/predictions.manifest.json")
        if (manifest.get("hierarchy") or {}).get("content_sha256") != hierarchy_checksum:
            raise ValueError(f"Wrong hierarchy in {baseline}")
        artifact_dir = baseline_root / baseline
        for csv_name in ("per_constraint.csv", "historical_strata.csv"):
            frame = pd.read_csv(artifact_dir / csv_name)
            if set(frame.get("validator_semantics_version", [])) != {VALIDATOR_SEMANTICS_VERSION}:
                raise ValueError(f"Wrong validator provenance in {baseline}/{csv_name}")
            if set(frame.get("semantic_contracts", [])) != {expected_contracts_json}:
                raise ValueError(f"Wrong semantic contracts in {baseline}/{csv_name}")
            if set(frame.get("hierarchy_content_sha256", [])) != {hierarchy_checksum}:
                raise ValueError(f"Wrong hierarchy provenance in {baseline}/{csv_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-expensive-input-checks", action="store_true")
    parser.add_argument(
        "--require-paper",
        action="store_true",
        help="Require readiness to include an exact manuscript-table comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_manifest = _load(ARCHIVE / "ARCHIVE_MANIFEST.json")
    hierarchy_payload, hierarchy = load_hierarchy_artifact(
        ROOT / "data/static/wikidata-p279-2018-07-01.v2.json"
    )
    if int(hierarchy_payload.get("seed_count", -1)) != 3941:
        raise ValueError("Historical hierarchy does not contain the required 3,941 seed IDs")
    if not args.skip_expensive_input_checks:
        _verify_immutable_inputs(archive_manifest)
        _verify_base_statements()
        _verify_passive_payloads(archive_manifest)
    _verify_semantics(hierarchy.content_sha256)
    _verify_runs(hierarchy.content_sha256)
    _verify_diagnostics(hierarchy.content_sha256, require_paper=args.require_paper)
    print("Validator-v3 regenerated suite passes acceptance checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
