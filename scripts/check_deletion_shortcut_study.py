#!/usr/bin/env python3
"""Gate the isolated M1D/G0 deletion-shortcut study before paper promotion."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.evaluation_artifacts import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    atomic_write_json,
    repository_relative_path,
    sha256_file,
)

A1_RUN = ROOT / "models" / "a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id"
RUNS = {
    "M1D": "m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id",
    "M1D-BP": "m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id",
    "G0": "g0_globalfix_reference_v2__full_strat1m_minocc100__node_id",
    "G0-BP": "g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id",
}
METRICS = {
    "pfr",
    "local_satisfaction",
    "delta_local_satisfaction",
    "sir",
    "srr",
    "disruption",
    "base_deletion_rate",
    "deletes_base_action_rate",
    "eppf",
    "vacuous_improvement",
}
LEGACY_FIELDS = {
    "gfr",
    "overall_gfr",
    "non_vacuous_primary_fix_rate",
    "primary_fix_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "paper_diagnostics" / "deletion_shortcut_study_readiness.json",
    )
    return parser.parse_args()


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": repository_relative_path(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _assert_no_legacy_fields(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LEGACY_FIELDS:
                raise ValueError(f"Legacy metric field {key!r} found at {path}")
            _assert_no_legacy_fields(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_legacy_fields(item, path=f"{path}[{index}]")


def _assert_metric_schema(system: str, payload: dict[str, Any]) -> None:
    metrics = payload.get("paper_metrics")
    if not isinstance(metrics, dict) or set(metrics) != METRICS:
        raise ValueError(f"{system}: incomplete paper metric schema")
    for name, metric in metrics.items():
        if not isinstance(metric, dict) or set(metric) != {"value", "numerator", "denominator"}:
            raise ValueError(f"{system}: malformed {name} metric")
        numerator = int(metric["numerator"])
        denominator = int(metric["denominator"])
        expected = float(numerator) / denominator if denominator else 0.0
        if not math.isclose(float(metric["value"]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{system}: inconsistent {name} value")


def _numeric_values(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numeric_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numeric_values(item)


def _check_history(system: str, path: Path) -> dict[str, Any]:
    history = _load(path)
    values = list(_numeric_values(history))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{system}: training history contains non-finite values")
    train_loss = [float(value) for value in history.get("train_loss", [])]
    val_loss = [float(value) for value in history.get("val_loss", [])]
    if not train_loss or len(train_loss) != len(val_loss):
        raise ValueError(f"{system}: incomplete train/validation loss history")
    if max(train_loss + val_loss) >= 100.0:
        raise ValueError(f"{system}: total loss crossed the registered stability bound")
    if system.startswith("M1D"):
        valid_logits = [
            *history.get("train_valid_edit_logit_abs_max", []),
            *history.get("val_valid_edit_logit_abs_max", []),
        ]
        if not valid_logits or max(float(value) for value in valid_logits) >= 10_000.0:
            raise ValueError(f"{system}: valid edit logits crossed the registered stability bound")
    if int(history.get("best_epoch", 0)) not in range(1, len(train_loss) + 1):
        raise ValueError(f"{system}: invalid best epoch")
    return {
        "identity": _identity(path),
        "epochs": len(train_loss),
        "best_epoch": int(history["best_epoch"]),
        "max_train_loss": max(train_loss),
        "max_validation_loss": max(val_loss),
    }


def _check_checkpoint(system: str, run: Path, config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = run / "checkpoint.pth"
    last_path = run / "checkpoint.last.pth"
    if not checkpoint_path.exists() or not last_path.exists():
        raise FileNotFoundError(f"{system}: best and last checkpoints are required")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_role") != "best" or int(checkpoint.get("best_epoch", 0)) <= 0:
        raise ValueError(f"{system}: malformed best checkpoint")
    if checkpoint.get("training_cfg", {}).get("seed") != 42:
        raise ValueError(f"{system}: checkpoint seed is not 42")
    provenance = checkpoint.get("training_provenance") or {}
    config_identity = provenance.get("config") or {}
    if config_identity.get("sha256") != sha256_file(run / "config.json"):
        raise ValueError(f"{system}: checkpoint/config checksum mismatch")
    a1_checksum = sha256_file(A1_RUN / "checkpoint.pth")
    source = (
        provenance.get("initialization_checkpoint")
        if system.startswith("M1D")
        else provenance.get("proposal_checkpoint")
    ) or {}
    if source.get("sha256") != a1_checksum:
        raise ValueError(f"{system}: A1 initialization/proposal checksum mismatch")
    expected_weight = 1.0 if system.endswith("-BP") else 0.0
    training = config.get("training_config") or {}
    if system.startswith("M1D"):
        direct = training.get("direct_safety") or {}
        expected = {
            "enabled": True,
            "alpha_primary": 1.0,
            "beta_secondary": 0.5,
            "loss_weight": 0.25,
            "score_temperature": 6.0,
            "focus_deletion_weight": expected_weight,
        }
        observed = {key: direct.get(key, 0.0 if key == "focus_deletion_weight" else None) for key in expected}
        if observed != expected or training.get("learning_rate") != 1e-5:
            raise ValueError(f"{system}: calibrated direct-safety configuration mismatch")
    else:
        observed_weight = float(training.get("focus_deletion_weight", 0.0))
        if training.get("objective") != "global_fix" or observed_weight != expected_weight:
            raise ValueError(f"{system}: global-fix configuration mismatch")
        if training.get("prediction_include_gold") is not False:
            raise ValueError(f"{system}: prediction candidates are not label-blind")
    del checkpoint
    return {
        "best": _identity(checkpoint_path),
        "last": _identity(last_path),
        "a1_source_sha256": a1_checksum,
    }


def _check_evaluation(system: str, run: Path) -> dict[str, Any]:
    evaluation = run / "evaluations"
    model_path = evaluation / "model.json"
    direct_path = evaluation / "model.direct.json"
    manifest_path = evaluation / "predictions.manifest.json"
    predictions_path = evaluation / "predictions.parquet"
    for path in (model_path, direct_path, manifest_path, predictions_path):
        if not path.exists():
            raise FileNotFoundError(f"{system}: missing evaluation artifact {path.name}")
    model = _load(model_path)
    direct = _load(direct_path)
    _assert_no_legacy_fields(model)
    _assert_metric_schema(system, model)
    for key in ("paper_metrics", "global_counts", "micro_f1", "macro_f1"):
        if model.get(key) != direct.get(key):
            raise ValueError(f"{system}: direct/replay mismatch in {key}")
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"{system}: prediction manifest is not schema v2")
    frame = pd.read_parquet(predictions_path)
    if int(manifest.get("row_count", -1)) != len(frame):
        raise ValueError(f"{system}: prediction row-count mismatch")
    prediction_identity = manifest.get("predictions") or {}
    if prediction_identity.get("sha256") != sha256_file(predictions_path):
        raise ValueError(f"{system}: prediction checksum mismatch")
    if system.startswith("G0"):
        audit_path = evaluation / "candidate_membership_audit.json"
        audit = _load(audit_path)
        if (
            audit.get("status") != "ok"
            or int(audit.get("membership_count", -1)) != len(frame)
            or (audit.get("candidate_protocol") or {}).get("include_gold") is not False
        ):
            raise ValueError(f"{system}: label-blind candidate membership audit failed")
    return {
        "model": _identity(model_path),
        "predictions": _identity(predictions_path),
        "row_count": len(frame),
        "paper_metrics": model["paper_metrics"],
    }


def _assert_mitigation(results: dict[str, dict[str, Any]], control: str, mitigation: str) -> None:
    control_metrics = results[control]["evaluation"]["paper_metrics"]
    mitigation_metrics = results[mitigation]["evaluation"]["paper_metrics"]
    for metric in ("base_deletion_rate", "deletes_base_action_rate"):
        if not mitigation_metrics[metric]["value"] < control_metrics[metric]["value"]:
            raise ValueError(f"{mitigation}: {metric} did not improve over {control}")
    if not mitigation_metrics["eppf"]["value"] > control_metrics["eppf"]["value"]:
        raise ValueError(f"{mitigation}: EPPF did not improve over {control}")


def main() -> None:
    args = parse_args()
    results: dict[str, dict[str, Any]] = {}
    common_row_count: int | None = None
    for system, directory in RUNS.items():
        run = ROOT / "models" / directory
        config = _load(run / "config.json")
        result = {
            "run_directory": repository_relative_path(run),
            "config": _identity(run / "config.json"),
            "checkpoint": _check_checkpoint(system, run, config),
            "history": _check_history(system, run / "training_history.json"),
            "evaluation": _check_evaluation(system, run),
        }
        row_count = int(result["evaluation"]["row_count"])
        if common_row_count is None:
            common_row_count = row_count
        elif row_count != common_row_count:
            raise ValueError("Study evaluations do not share the same test-row count")
        results[system] = result
    _assert_mitigation(results, "M1D", "M1D-BP")
    _assert_mitigation(results, "G0", "G0-BP")
    atomic_write_json(
        args.output.resolve(),
        {
            "schema_version": 1,
            "status": "ready",
            "seed": 42,
            "row_count": common_row_count,
            "systems": results,
            "promotion": {
                "canonical_controls": ["M1D", "G0"],
                "mitigation_companions": ["M1D-BP", "G0-BP"],
            },
        },
    )
    print(f"[ok] deletion-shortcut study is ready: {args.output.resolve()}")


if __name__ == "__main__":
    main()
