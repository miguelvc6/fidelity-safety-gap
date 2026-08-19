#!/usr/bin/env python3
"""Fail unless every corrected paper artifact is complete and mutually consistent."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.evaluation_artifacts import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    IDENTITY_COLUMNS,
    PREDICTION_COLUMNS,
    atomic_write_json,
    repository_relative_path,
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
    "G0": "g0_globalfix_reference_v2__full_strat1m_minocc100__node_id",
}
BASELINES = {
    "DFB": "DeleteFocusBaseline",
    "AMB": "AddMirrorBaseline",
    "CFM": "ConstraintFamilyMajorityBaseline",
    "CDM": "ConstraintDefinitionMajorityBaseline",
}
EXPECTED_MODEL_CONFIG = {
    "B0": {
        "model": "GIN",
        "constraint_representation": "eswc_passive",
        "hidden_channels": 128,
        "head_hidden": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "num_embedding_size": 128,
        "use_role_embeddings": True,
        "role_embedding_dim": 16,
        "use_edge_attributes": True,
        "pressure_enabled": False,
    },
    "A1": {
        "model": "GIN_PRESSURE",
        "constraint_representation": "factorized",
        "hidden_channels": 400,
        "head_hidden": 400,
        "num_layers": 4,
        "dropout": 0.17,
        "num_embedding_size": 128,
        "use_role_embeddings": True,
        "role_embedding_dim": 16,
        "use_edge_attributes": True,
        "pressure_enabled": True,
        "pressure_residual_scale": 0.1,
        "factor_executor_impl": "per_type_grouped_v2",
        "gold_edit_embedding_mode": "compact",
        "pressure_module_sharing": "per_type",
        "active_factor_type_ids": [0, 2, 3, 4, 5, 9, 12, 14, 15, 16],
    },
    "M1C": {
        "model": "GIN_PRESSURE",
        "constraint_representation": "factorized",
        "hidden_channels": 400,
        "head_hidden": 400,
        "num_layers": 4,
        "dropout": 0.17,
        "num_embedding_size": 128,
        "use_role_embeddings": True,
        "role_embedding_dim": 16,
        "use_edge_attributes": True,
        "pressure_enabled": True,
        "pressure_residual_scale": 0.1,
        "factor_executor_impl": "per_type_grouped_v2",
        "gold_edit_embedding_mode": "compact",
        "pressure_module_sharing": "per_type",
        "active_factor_type_ids": [0, 2, 3, 4, 5, 9, 12, 14, 15, 16],
    },
    "M1D": {
        "model": "GIN_PRESSURE",
        "constraint_representation": "factorized",
        "hidden_channels": 400,
        "head_hidden": 400,
        "num_layers": 4,
        "dropout": 0.17,
        "num_embedding_size": 128,
        "use_role_embeddings": True,
        "role_embedding_dim": 16,
        "use_edge_attributes": True,
        "pressure_enabled": True,
        "pressure_residual_scale": 0.1,
        "factor_executor_impl": "per_type_grouped_v2",
        "gold_edit_embedding_mode": "compact",
        "pressure_module_sharing": "per_type",
        "active_factor_type_ids": [0, 2, 3, 4, 5, 9, 12, 14, 15, 16],
    },
    "G0": {
        "model": "RERANKER",
        "constraint_representation": "factorized",
        "hidden_channels": 128,
        "head_hidden": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "num_embedding_size": 128,
        "use_role_embeddings": False,
        "use_edge_attributes": False,
        "pressure_enabled": False,
    },
}
EXPECTED_TRAINING_CONFIG = {
    "batch_size": 256,
    "early_stopping_rounds": 2,
    "grad_clip": 0.5,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "validation_subset_size": 25_000,
    "weight_decay": 1.1e-4,
}
EXPECTED_FACTOR_LOSS = {
    "enabled": True,
    "only_checkable": True,
    "per_graph_reduction": "mean",
    "pos_weight": None,
    "weight_post_gold": 0.1,
    "weight_pre": 0.1,
}
EXPECTED_RERANKER_TRAINING_CONFIG = {
    "batch_size": 64,
    "early_stopping_rounds": 2,
    "grad_clip": 0.5,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "validation_subset_size": 25_000,
    "weight_decay": 1e-4,
    "seed": 42,
    "objective": "global_fix",
    "include_gold": True,
    "prediction_include_gold": False,
}
SIDECAR_RUNS = ("A1", "M1C", "M1D")
EXPECTED_H2_SELECTION = {
    "A1": "slot_argmax",
    "M1C": "chooser",
    "M1D": "direct_safety",
}
SELECTOR_RERUN_TOLERANCE = 1e-3
TARGET_COLUMNS = (
    "add_subject",
    "add_predicate",
    "add_object",
    "del_subject",
    "del_predicate",
    "del_object",
)
PAPER_ROW_NAMES = (*RUNS, *BASELINES)
PAPER_LABELS = {
    "B0": "Direct--Passive GNN",
    "A1": "Direct--Factor GNN",
    "M1C": "Candidate--C",
    "M1D": "Candidate--DP",
    "G0": "Candidate--SR",
    "DFB": "Baseline--DB",
    "AMB": "Baseline--AM",
    "CFM": "Baseline--FM",
    "CDM": "Baseline--DM",
}
_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "paper_diagnostics" / "corrected_paper_readiness.json",
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=None,
        help="Optional TeX source whose result tables must match the canonical artifacts.",
    )
    parser.add_argument(
        "--verify-graph-checksums",
        action="store_true",
        help="Hash every unique graph artifact in addition to validating its path and size.",
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
    if system in BASELINES:
        root = ROOT / "models" / "baselines" / "full_strat1m" / "parquet"
        basename = f"baseline-{BASELINES[system]}"
        return root / f"{basename}.json", root / basename
    root = ROOT / "models" / RUNS[system] / "evaluations"
    return root / "model.json", root


def _assert_file_identity(system: str, name: str, identity: Any) -> Path:
    if not isinstance(identity, dict):
        raise ValueError(f"{system}: missing {name} provenance")
    path = Path(identity.get("path", ""))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"{system}: {name} missing at {path}")
    if int(identity.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError(f"{system}: {name} size mismatch")
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    checksum = _SHA256_CACHE.get(cache_key)
    if checksum is None:
        checksum = sha256_file(path)
        _SHA256_CACHE[cache_key] = checksum
    if checksum != identity.get("sha256"):
        raise ValueError(f"{system}: {name} checksum mismatch")
    return path


def _fidelity_counts(predictions: np.ndarray, targets: np.ndarray) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for columns in (slice(0, 3), slice(3, 6)):
        predicted = predictions[:, columns]
        target = targets[:, columns]
        predicted_complete = np.all(predicted != 0, axis=1)
        target_complete = np.all(target != 0, axis=1)
        exact = predicted_complete & target_complete & np.all(predicted == target, axis=1)
        matches = int(exact.sum())
        tp += matches
        fp += int(predicted_complete.sum()) - matches
        fn += int(target_complete.sum()) - matches
    return tp, fp, fn


def _fidelity_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = float(tp) / (tp + fp) if tp + fp else 0.0
    recall = float(tp) / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _assert_close(system: str, field: str, observed: Any, expected: float) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{system}: {field} is {observed}, expected {expected}")


def _check_per_constraint_csv(
    system: str,
    *,
    artifact_dir: Path,
    frame: pd.DataFrame,
    targets: np.ndarray,
) -> None:
    csv_path = artifact_dir / "per_constraint.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{system}: missing per-constraint table at {csv_path}")
    table = pd.read_csv(csv_path).set_index("constraint_type")
    frame_types = frame["constraint_type"].astype(str)
    if set(table.index.astype(str)) != set(frame_types.unique()):
        raise ValueError(f"{system}: per-constraint family set does not match predictions")
    predictions = frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=np.int64)
    for constraint_type, row in table.iterrows():
        mask = (frame_types == str(constraint_type)).to_numpy()
        support = int(mask.sum())
        if int(row["support"]) != support:
            raise ValueError(f"{system}: support mismatch for {constraint_type}")
        tp, fp, fn = _fidelity_counts(predictions[mask], targets[mask])
        fidelity = _fidelity_metrics(tp, fp, fn)
        _assert_close(
            system,
            f"{constraint_type}.fidelity_micro_f1",
            row["fidelity_micro_f1"],
            fidelity["f1"],
        )
        for metric_name in PAPER_METRIC_KEYS:
            numerator = int(frame.loc[mask, f"metric_{metric_name}_numerator"].sum())
            denominator = int(frame.loc[mask, f"metric_{metric_name}_denominator"].sum())
            value = float(numerator) / denominator if denominator else 0.0
            if int(row[f"{metric_name}_numerator"]) != numerator:
                raise ValueError(
                    f"{system}: {constraint_type}.{metric_name} numerator mismatch"
                )
            if int(row[f"{metric_name}_denominator"]) != denominator:
                raise ValueError(
                    f"{system}: {constraint_type}.{metric_name} denominator mismatch"
                )
            _assert_close(
                system,
                f"{constraint_type}.{metric_name}",
                row[f"{metric_name}_value"],
                value,
            )


def _check_evaluation(
    system: str,
    dataset_frame: pd.DataFrame,
    *,
    verify_graph_checksums: bool,
) -> dict[str, Any]:
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
    predictions_path = _assert_file_identity(system, "predictions", predictions)
    dataset = manifest.get("dataset") or {}
    artifact = dataset.get("artifact") or {}
    _assert_file_identity(system, "interim dataset", artifact)
    if dataset.get("variant") != "full_strat1m_minocc100" or dataset.get("split") != "test":
        raise ValueError(f"{system}: wrong dataset variant or split")

    frame = pd.read_parquet(predictions_path)
    required = set(IDENTITY_COLUMNS + PREDICTION_COLUMNS)
    required.update(
        f"metric_{name}_{part}"
        for name in PAPER_METRIC_KEYS
        for part in ("numerator", "denominator")
    )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{system}: prediction columns are incomplete: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS or len(dataset_frame) != EXPECTED_ROWS:
        raise ValueError(f"{system}: prediction or dataset row count is not {EXPECTED_ROWS}")
    expected_identity = dataset_frame.loc[:, list(IDENTITY_COLUMNS[1:])].reset_index(drop=True)
    actual_identity = frame.loc[:, list(IDENTITY_COLUMNS[1:])].reset_index(drop=True)
    if not actual_identity.equals(expected_identity):
        raise ValueError(f"{system}: prediction row identity/order differs from the dataset")
    if not np.array_equal(frame["row_index"].to_numpy(), np.arange(EXPECTED_ROWS)):
        raise ValueError(f"{system}: row_index is not canonical")

    for metric_name, recorded in metrics.items():
        numerator = int(frame[f"metric_{metric_name}_numerator"].sum())
        denominator = int(frame[f"metric_{metric_name}_denominator"].sum())
        value = float(numerator) / denominator if denominator else 0.0
        expected_metric = {
            "value": value,
            "numerator": numerator,
            "denominator": denominator,
        }
        if recorded != expected_metric:
            raise ValueError(f"{system}: {metric_name} does not aggregate from predictions")

    prediction_values = frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=np.int64)
    target_values = dataset_frame.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.int64)
    tp, fp, fn = _fidelity_counts(prediction_values, target_values)
    fidelity = _fidelity_metrics(tp, fp, fn)
    if model.get("global_counts") != {"tp": tp, "fp": fp, "fn": fn}:
        raise ValueError(f"{system}: fidelity counts do not match predictions")
    for field, expected in (
        ("micro_precision", fidelity["precision"]),
        ("micro_recall", fidelity["recall"]),
        ("micro_f1", fidelity["f1"]),
    ):
        _assert_close(system, field, model.get(field), expected)

    model_artifacts = model.get("prediction_artifacts") or {}
    if model_artifacts.get("predictions_sha256") != predictions.get("sha256"):
        raise ValueError(f"{system}: model.json points to different predictions")

    training_summary = None
    if system in RUNS:
        expected_run = (ROOT / "models" / RUNS[system]).resolve()
        config_path = _assert_file_identity(system, "config", manifest.get("config"))
        checkpoint_path = _assert_file_identity(system, "checkpoint", manifest.get("checkpoint"))
        if config_path.resolve() != expected_run / "config.json":
            raise ValueError(f"{system}: manifest references the wrong config")
        if checkpoint_path.resolve() != expected_run / "checkpoint.pth":
            raise ValueError(f"{system}: manifest references the wrong checkpoint")
        full_config = _load(config_path)
        config = full_config.get("model_config") or {}
        for key, expected in EXPECTED_MODEL_CONFIG[system].items():
            if config.get(key) != expected:
                raise ValueError(f"{system}: config {key}={config.get(key)!r}, expected {expected!r}")
        training_config = full_config.get("training_config") or {}
        expected_training = (
            EXPECTED_RERANKER_TRAINING_CONFIG if system == "G0" else EXPECTED_TRAINING_CONFIG
        )
        for key, expected in expected_training.items():
            if training_config.get(key) != expected:
                raise ValueError(
                    f"{system}: training config {key}={training_config.get(key)!r}, "
                    f"expected {expected!r}"
                )
        # A1 predates the explicit seed field; its training entry point hard-coded
        # 42. New configurations record the same value directly.
        if training_config.get("seed", 42) != 42:
            raise ValueError(f"{system}: training seed is not 42")
        if (
            system in {"A1", "M1C", "M1D"}
            and training_config.get("factor_loss") != EXPECTED_FACTOR_LOSS
        ):
            raise ValueError(f"{system}: factor-loss configuration differs from the paper")
        chooser = training_config.get("chooser") or {}
        direct_safety = training_config.get("direct_safety") or {}
        if system == "M1C":
            expected_chooser = {
                "enabled": True,
                "loss_mode": "fix1",
                "loss_weight": 0.25,
                "beta_no_regression": 0.25,
                "gamma_primary": 0.2,
                "topk_candidates": 20,
                "max_candidates_total": 80,
            }
            if chooser != expected_chooser:
                raise ValueError("M1C: chooser configuration differs from the paper")
        elif system != "G0" and chooser.get("enabled") is not False:
            raise ValueError(f"{system}: unexpected learned chooser")
        if system == "M1D":
            expected_direct = {
                "enabled": True,
                "alpha_primary": 1.0,
                "beta_secondary": 0.5,
                "topk_candidates": 20,
                "max_candidates_total": 80,
            }
            if direct_safety != expected_direct:
                raise ValueError("M1D: direct-safety configuration differs from the paper")
        elif system != "G0" and direct_safety.get("enabled") is not False:
            raise ValueError(f"{system}: unexpected direct-safety objective")
        if system == "G0":
            proposal = full_config.get("proposal_config") or {}
            if proposal.get("config_tag") != RUNS["A1"]:
                raise ValueError("G0: proposal configuration is not the paper Direct-Factor run")
            reranker = full_config.get("reranker_config") or {}
            if reranker != {
                "candidate_embedding_dim": 64,
                "candidate_hidden_dim": 128,
                "dropout": 0.1,
            }:
                raise ValueError("G0: reranker architecture differs from the paper")
        graph_artifacts = (manifest.get("graph") or {}).get("artifacts", [])
        graph_paths = [Path(item.get("path", "")) for item in graph_artifacts]
        graph_paths = [path if path.is_absolute() else ROOT / path for path in graph_paths]
        if not graph_paths or any(not path.exists() for path in graph_paths):
            raise ValueError(f"{system}: graph provenance is incomplete")
        for index, (identity, path) in enumerate(zip(graph_artifacts, graph_paths)):
            if int(identity.get("size_bytes", -1)) != path.stat().st_size:
                raise ValueError(f"{system}: graph artifact {index} size mismatch")
            if verify_graph_checksums:
                _assert_file_identity(system, f"graph artifact {index}", identity)
        if system == "B0":
            if any("repr-eswc_passive" not in path.name for path in graph_paths):
                raise ValueError("B0: predictions were not produced from the passive graph suite")
        elif any("repr-eswc_passive" in path.name for path in graph_paths):
            raise ValueError(f"{system}: factorized run references passive graphs")
        history_path = expected_run / "training_history.json"
        history = _load(history_path)
        val_loss = history.get("val_loss")
        if not isinstance(val_loss, list) or not val_loss:
            raise ValueError(f"{system}: validation-loss history is missing")
        if len(val_loss) > int(training_config["num_epochs"]):
            raise ValueError(f"{system}: training history exceeds configured epochs")
        best_epoch_index = min(range(len(val_loss)), key=lambda index: float(val_loss[index]))
        training_summary = {
            "history_path": repository_relative_path(history_path),
            "history_sha256": sha256_file(history_path),
            "epochs_run": len(val_loss),
            "best_epoch": best_epoch_index + 1,
            "best_validation_loss": float(val_loss[best_epoch_index]),
        }
        if system == "M1D" and not (
            len(val_loss) == 3
            and best_epoch_index == 0
            and math.isclose(float(val_loss[0]), 1.48208, abs_tol=1e-12)
            and math.isclose(float(val_loss[1]), 46_378.4448, abs_tol=1e-8)
            and math.isclose(float(val_loss[2]), 1_016_081_888.0512, abs_tol=1e-4)
        ):
            raise ValueError("M1D: training instability record differs from the paper")
    else:
        if manifest.get("config") is not None or manifest.get("checkpoint") is not None:
            raise ValueError(f"{system}: deterministic baseline has learned-model provenance")
        if (manifest.get("graph") or {}).get("artifacts"):
            raise ValueError(f"{system}: parquet baseline unexpectedly references model graphs")

    _check_per_constraint_csv(
        system,
        artifact_dir=artifact_dir,
        frame=frame,
        targets=target_values,
    )
    add_mean = float(np.all(prediction_values[:, :3] != 0, axis=1).mean())
    del_mean = float(np.all(prediction_values[:, 3:] != 0, axis=1).mean())
    return {
        "model_path": repository_relative_path(model_path),
        "predictions_path": repository_relative_path(predictions_path),
        "predictions_sha256": predictions["sha256"],
        "dataset_variant": dataset.get("variant"),
        "dataset_sha256": artifact.get("sha256"),
        "row_identity_sha256": dataset.get("row_identity_sha256"),
        "row_count": manifest["row_count"],
        "graph_checksums_verified": bool(verify_graph_checksums and system in RUNS),
        "training": training_summary,
        "fidelity": fidelity,
        "mean_additions": add_mean,
        "mean_deletions": del_mean,
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
    expected_candidate_config = {
        "heuristic_max_candidates": 30,
        "heuristic_max_values": 3,
        "include_gold": False,
        "max_candidates_total": 80,
        "topk_candidates": 20,
        "topk_per_slot": 5,
    }
    if oracle.get("candidate_config") != expected_candidate_config:
        raise ValueError(f"{system}: candidate oracle configuration differs from the paper")
    if oracle.get("selection_mode") != EXPECTED_H2_SELECTION[system]:
        raise ValueError(
            f"{system}: oracle selection mode is {oracle.get('selection_mode')!r}, "
            f"expected {EXPECTED_H2_SELECTION[system]!r}"
        )
    overall = oracle.get("overall") or {}
    if int(overall.get("support", -1)) != EXPECTED_ROWS:
        raise ValueError(f"{system}: oracle support is incomplete")
    for key in ("oracle_paper_metrics", "selected_paper_metrics"):
        _check_paper_metrics(system, overall.get(key), context=key)
    selected_metrics = overall["selected_paper_metrics"]
    selected_source = oracle.get("selected_prediction_source") or {}
    exact_selected_replay = selected_source.get("mode") == "validated_schema_v2_replay"
    model_manifest = _load(evaluation_dir / "predictions.manifest.json")
    expected_prediction_sha = (model_manifest.get("predictions") or {}).get("sha256")
    if not exact_selected_replay or selected_source.get("sha256") != expected_prediction_sha:
        raise ValueError(f"{system}: candidate oracle did not replay the paper predictions")
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


def _normalise_tex(value: str) -> str:
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r"\\(?:textbf|underline)\{([^{}]*)\}", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def _paper_table(source: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    label_position = source.find(marker)
    if label_position < 0:
        raise ValueError(f"Paper: missing table {label}")
    start = source.rfind(r"\begin{table}", 0, label_position)
    end = source.find(r"\end{table}", label_position)
    if start < 0 or end < 0:
        raise ValueError(f"Paper: malformed table {label}")
    return _normalise_tex(source[start : end + len(r"\end{table}")])


def _require_table_row(table: str, label: str, values: list[str]) -> None:
    expected = _normalise_tex(" & ".join(values) + r" \\")
    if expected not in table:
        raise ValueError(f"Paper: {label} row does not match artifacts: {expected}")


def _check_paper(
    paper_path: Path,
    systems: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not paper_path.exists():
        raise FileNotFoundError(paper_path)
    source = paper_path.read_text(encoding="utf-8")
    forbidden_fragments = (
        "compact A1",
        "Compact A1",
        "parameter-matched B0",
        "there are no retained weights with which to rescore",
        "cannot be regenerated",
    )
    present_forbidden = [value for value in forbidden_fragments if value in source]
    if present_forbidden:
        raise ValueError(f"Paper: stale or unsupported claims remain: {present_forbidden}")
    main_table = _paper_table(source, "tab:main-results")
    symbolic_table = _paper_table(source, "tab:symbolic-results")
    transition_table = _paper_table(source, "tab:transition-results")

    for system in PAPER_ROW_NAMES:
        report = systems[system]
        fidelity = report["fidelity"]
        metrics = report["paper_metrics"]
        _require_table_row(
            main_table,
            "tab:main-results",
            [
                PAPER_LABELS[system],
                f'{fidelity["precision"]:.4f}',
                f'{fidelity["recall"]:.4f}',
                f'{fidelity["f1"]:.4f}',
                f'{report["mean_additions"]:.4f}',
                f'{report["mean_deletions"]:.4f}',
                f'{metrics["disruption"]["value"]:.4f}',
            ],
        )
        _require_table_row(
            symbolic_table,
            "tab:symbolic-results",
            [
                PAPER_LABELS[system],
                f'{metrics["pfr"]["value"]:.4f}',
                f'{metrics["local_satisfaction"]["value"]:.4f}',
                f'{metrics["local_satisfaction"]["denominator"]:,}',
                f'{metrics["delta_local_satisfaction"]["value"]:.4f}',
                f'{metrics["delta_local_satisfaction"]["denominator"]:,}',
                f'{metrics["base_deletion_rate"]["value"]:.4f}',
                f'{metrics["deletes_base_action_rate"]["value"]:.4f}',
                f'{metrics["eppf"]["value"]:.4f}',
                f'{metrics["vacuous_improvement"]["value"]:.4f}',
            ],
        )
        _require_table_row(
            transition_table,
            "tab:transition-results",
            [
                PAPER_LABELS[system],
                f'{metrics["sir"]["value"]:.4f}',
                f'{metrics["sir"]["denominator"]:,}',
                f'{metrics["srr"]["value"]:.4f}',
                f'{metrics["srr"]["denominator"]:,}',
            ],
        )

    h2 = _load(
        ROOT / "models" / RUNS["A1"] / "evaluations" / "h2" / "h2_report.json"
    )
    pressure_table = _paper_table(source, "tab:pressure-masking")
    pressure_names = {
        "no_factor_pressure": "No factor pressure",
        "primary_only_pressure": "Primary-only pressure",
        "secondary_only_pressure": "Secondary-only pressure",
    }
    for row in h2.get("overall_deltas") or []:
        variant = row.get("variant")
        if variant not in pressure_names:
            continue
        _require_table_row(
            pressure_table,
            "tab:pressure-masking",
            [
                pressure_names[variant],
                f'{row["delta_fidelity_f1"]:.4f}',
                f'{row["delta_pfr"]:.4f}',
                f'{row["delta_local_satisfaction"]:.4f}',
                f'{row["delta_disruption"]:.4f}',
                f'{row["prediction_changed_rate"]:.4f}',
            ],
        )

    return {
        "path": repository_relative_path(paper_path),
        "sha256": sha256_file(paper_path),
        "validated_tables": [
            "tab:main-results",
            "tab:symbolic-results",
            "tab:transition-results",
            "tab:pressure-masking",
        ],
    }


def main() -> None:
    args = parse_args()
    dataset_path = (
        ROOT
        / "data"
        / "interim"
        / "full_strat1m_minocc100"
        / "df_test.parquet"
    )
    dataset_frame = pd.read_parquet(
        dataset_path,
        columns=[*IDENTITY_COLUMNS[1:], *TARGET_COLUMNS],
    )
    systems = {
        system: _check_evaluation(
            system,
            dataset_frame,
            verify_graph_checksums=args.verify_graph_checksums,
        )
        for system in PAPER_ROW_NAMES
    }
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
    paper = _check_paper(args.paper.resolve(), systems) if args.paper else None
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "ready": True,
        "canonical_scope": {
            "learned_systems": list(RUNS),
            "deterministic_baselines": list(BASELINES),
        },
        "systems": systems,
        "sidecars": sidecars,
        "paper": paper,
    }
    atomic_write_json(args.output, report)
    print(f"Corrected paper artifacts are ready; wrote {args.output}")


if __name__ == "__main__":
    main()
