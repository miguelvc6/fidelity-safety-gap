#!/usr/bin/env python3
"""Fail unless every corrected paper artifact is complete and mutually consistent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.evaluation_artifacts import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    atomic_write_json,
    sha256_file,
)
from modules.repair_eval import PAPER_METRIC_KEYS  # noqa: E402

EXPECTED_ROWS = 143_316
LEGACY_FIELDS = {
    "gfr",
    "overall_gfr",
    "non_vacuous_primary_fix_rate",
    "primary_fix_rate",
}
RUNS = {
    "B0": "b0_eswc_reproduction__full_strat1m_minocc100__node_id",
    "A1": "a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id",
    "M1C": "m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id",
    "M1D": "m1d_safe_factor_direct_compact_grouped__full_strat1m_minocc100__node_id",
}
G0_RUN = "g0_globalfix_reference__full_strat1m_minocc100__node_id"
SIDECAR_RUNS = ("A1", "M1C", "M1D")
EXPECTED_H2_SELECTION = {
    "A1": "slot_argmax",
    "M1C": "chooser",
    "M1D": "direct_safety",
}
SELECTOR_RERUN_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "paper_diagnostics" / "corrected_paper_readiness.json",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _legacy_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(key for key in value if key in LEGACY_FIELDS)
        for item in value.values():
            found.update(_legacy_fields(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_legacy_fields(item))
    return found


def _check_paper_metrics(system: str, metrics: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(metrics, dict) or set(metrics) != set(PAPER_METRIC_KEYS):
        raise ValueError(f"{system}: {context} paper metric set is incomplete")
    for name in PAPER_METRIC_KEYS:
        metric = metrics[name]
        if not isinstance(metric, dict) or set(metric) != {"value", "numerator", "denominator"}:
            raise ValueError(f"{system}: malformed {context} metric {name}")
    return metrics


def _metric_within_selector_tolerance(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    if observed == expected:
        return True
    expected_denominator = max(1, int(expected.get("denominator", 0)))
    value_drift = abs(float(observed.get("value", 0.0)) - float(expected.get("value", 0.0)))
    numerator_drift = abs(int(observed.get("numerator", 0)) - int(expected.get("numerator", 0)))
    denominator_drift = abs(int(observed.get("denominator", 0)) - int(expected.get("denominator", 0)))
    return (
        value_drift <= SELECTOR_RERUN_TOLERANCE
        and numerator_drift / expected_denominator <= SELECTOR_RERUN_TOLERANCE
        and denominator_drift / expected_denominator <= SELECTOR_RERUN_TOLERANCE
    )


def _evaluation_paths(system: str) -> tuple[Path, Path]:
    if system == "DFB":
        root = ROOT / "models" / "baselines" / "full_strat1m" / "parquet"
        return root / "baseline-DeleteFocusBaseline.json", root / "baseline-DeleteFocusBaseline"
    root = ROOT / "models" / RUNS[system] / "evaluations"
    return root / "model.json", root


def _check_evaluation(system: str) -> dict[str, Any]:
    model_path, artifact_dir = _evaluation_paths(system)
    model = _load(model_path)
    if model.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"{system}: model schema is not v2")
    legacy = _legacy_fields(model)
    if legacy:
        raise ValueError(f"{system}: legacy metric fields remain: {sorted(legacy)}")
    metrics = _check_paper_metrics(system, model.get("paper_metrics"), context="model")

    manifest_path = artifact_dir / "predictions.manifest.json"
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"{system}: prediction manifest schema is not v2")
    if int(manifest.get("row_count", -1)) != EXPECTED_ROWS:
        raise ValueError(f"{system}: expected {EXPECTED_ROWS} predictions")
    predictions = manifest.get("predictions") or {}
    predictions_path = Path(predictions.get("path", ""))
    if not predictions_path.exists():
        raise FileNotFoundError(f"{system}: predictions missing at {predictions_path}")
    if sha256_file(predictions_path) != predictions.get("sha256"):
        raise ValueError(f"{system}: predictions checksum mismatch")
    dataset = manifest.get("dataset") or {}
    artifact = dataset.get("artifact") or {}
    dataset_path = Path(artifact.get("path", ""))
    if not dataset_path.exists() or sha256_file(dataset_path) != artifact.get("sha256"):
        raise ValueError(f"{system}: interim dataset checksum mismatch")
    return {
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": predictions["sha256"],
        "dataset_variant": dataset.get("variant"),
        "dataset_sha256": artifact.get("sha256"),
        "row_identity_sha256": dataset.get("row_identity_sha256"),
        "row_count": manifest["row_count"],
        "paper_metrics": metrics,
    }


def _check_sidecars(system: str) -> dict[str, Any]:
    evaluation_dir = ROOT / "models" / RUNS[system] / "evaluations"
    h2 = _load(evaluation_dir / "h2" / "h2_report.json")
    legacy = _legacy_fields(h2)
    if legacy:
        raise ValueError(f"{system}: legacy metric fields remain in H2: {sorted(legacy)}")
    if h2.get("status") not in {"ok", "partial"} or not h2.get("overall"):
        raise ValueError(f"{system}: incomplete H2 report")
    if h2.get("selection_mode") != EXPECTED_H2_SELECTION[system]:
        raise ValueError(
            f"{system}: H2 selection mode is {h2.get('selection_mode')!r}, "
            f"expected {EXPECTED_H2_SELECTION[system]!r}"
        )
    unsupported = h2.get("unsupported") or {}
    unexpected = set(unsupported) - {"post_gold_factor_semantics"}
    if unexpected:
        raise ValueError(f"{system}: unexpected H2 unsupported sections: {sorted(unexpected)}")
    h2_overall = h2["overall"]
    expected_variants = {
        "normal",
        "no_factor_pressure",
        "primary_only_pressure",
        "secondary_only_pressure",
    }
    if {row.get("variant") for row in h2_overall} != expected_variants:
        raise ValueError(f"{system}: H2 variants are incomplete")
    for row in h2_overall:
        if int(row.get("support", -1)) != EXPECTED_ROWS:
            raise ValueError(f"{system}: incomplete H2 support for {row.get('variant')}")
    normal = next(row for row in h2_overall if row.get("variant") == "normal")
    model_metrics = _load(evaluation_dir / "model.json").get("paper_metrics") or {}
    for name, metric in model_metrics.items():
        observed = {
            "value": normal.get(name),
            "numerator": normal.get(f"{name}_numerator"),
            "denominator": normal.get(f"{name}_denominator"),
        }
        exact_replay = h2.get("normal_prediction_source") == "validated_schema_v2_replay"
        matches = observed == metric if exact_replay else _metric_within_selector_tolerance(observed, metric)
        if not matches:
            mode = "validated replay" if exact_replay else "selector rerun tolerance"
            raise ValueError(f"{system}: normal H2 metrics exceed {mode} for {name}")

    oracle = _load(evaluation_dir / "oracle" / "oracle_summary.json")
    legacy = _legacy_fields(oracle)
    if legacy:
        raise ValueError(f"{system}: legacy metric fields remain in oracle: {sorted(legacy)}")
    if oracle.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"{system}: oracle schema is not v2")
    if (oracle.get("candidate_config") or {}).get("include_gold") is not False:
        raise ValueError(f"{system}: candidate oracle is not label-blind")
    overall = oracle.get("overall") or {}
    if int(overall.get("support", -1)) != EXPECTED_ROWS:
        raise ValueError(f"{system}: oracle support is incomplete")
    for key in ("oracle_paper_metrics", "selected_paper_metrics"):
        _check_paper_metrics(system, overall.get(key), context=key)
    selected_metrics = overall["selected_paper_metrics"]
    exact_selected_replay = (
        (oracle.get("selected_prediction_source") or {}).get("mode")
        == "validated_schema_v2_replay"
    )
    for name, metric in model_metrics.items():
        observed = selected_metrics.get(name) or {}
        matches = observed == metric if exact_selected_replay else _metric_within_selector_tolerance(
            observed, metric
        )
        if not matches:
            raise ValueError(
                f"{system}: candidate-oracle selected metrics do not match the paper prediction for {name}"
            )
    return {
        "h2_status": h2["status"],
        "h2_unsupported": unsupported,
        "oracle_support": overall["support"],
    }


def _check_g0_exclusion() -> dict[str, Any]:
    run_directory = ROOT / "models" / G0_RUN
    evaluation_dir = run_directory / "evaluations"
    if (run_directory / "checkpoint.pth").exists():
        raise ValueError("G0: checkpoint now exists; regenerate it instead of excluding the run")
    audit = _load(evaluation_dir / "candidate_membership_audit.json")
    if audit.get("status") != "failed":
        raise ValueError("G0: expected the retained-prediction exclusion audit to fail")
    if int(audit.get("row_count", -1)) != EXPECTED_ROWS:
        raise ValueError("G0: candidate-membership audit has incomplete support")
    if int(audit.get("membership_count", EXPECTED_ROWS)) >= EXPECTED_ROWS:
        raise ValueError("G0: exclusion audit no longer demonstrates missing label-blind candidates")
    candidate_protocol = audit.get("candidate_protocol") or {}
    if candidate_protocol.get("include_gold") is not False:
        raise ValueError("G0: candidate-membership audit did not exclude gold")
    if candidate_protocol.get("topk_proposal_candidates_used") is not True:
        raise ValueError("G0: candidate-membership audit did not reconstruct the full candidate set")
    predictions = audit.get("predictions") or {}
    source_path = Path(predictions.get("path", ""))
    if not source_path.exists() or sha256_file(source_path) != predictions.get("sha256"):
        raise ValueError("G0: audited source prediction checksum mismatch")
    return {
        "status": "excluded",
        "reason": "missing reranker checkpoint and incomplete gold-excluded candidate membership",
        "row_count": audit["row_count"],
        "membership_rate": audit.get("membership_rate"),
        "predictions_sha256": predictions.get("sha256"),
    }


def main() -> None:
    args = parse_args()
    systems = {system: _check_evaluation(system) for system in (*RUNS, "DFB")}
    reference = systems["A1"]
    for system, report in systems.items():
        if report["row_count"] != reference["row_count"]:
            raise ValueError(f"{system}: row count differs from A1")
        if report["dataset_sha256"] != reference["dataset_sha256"]:
            raise ValueError(f"{system}: dataset differs from A1")
        if report["row_identity_sha256"] != reference["row_identity_sha256"]:
            raise ValueError(f"{system}: row identity differs from A1")
    dfb_metrics = systems["DFB"]["paper_metrics"]
    for metric_name, expected in (
        ("base_deletion_rate", 1.0),
        ("deletes_base_action_rate", 1.0),
        ("eppf", 0.0),
    ):
        actual = float((dfb_metrics.get(metric_name) or {}).get("value", float("nan")))
        if actual != expected:
            raise ValueError(f"DFB {metric_name} is {actual}, expected {expected}")

    sidecars = {system: _check_sidecars(system) for system in SIDECAR_RUNS}
    g0_exclusion = _check_g0_exclusion()
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "ready": True,
        "systems": systems,
        "sidecars": sidecars,
        "excluded_runs": {"G0": g0_exclusion},
    }
    atomic_write_json(args.output, report)
    print(f"Corrected paper artifacts are ready; wrote {args.output}")


if __name__ == "__main__":
    main()
