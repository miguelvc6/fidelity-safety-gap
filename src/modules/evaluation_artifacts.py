"""Schema-v2 evaluation output and replay artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch


EVALUATION_SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_COLUMNS = (
    "pred_add_subject",
    "pred_add_predicate",
    "pred_add_object",
    "pred_del_subject",
    "pred_del_predicate",
    "pred_del_object",
)
IDENTITY_COLUMNS = (
    "row_index",
    "constraint_id",
    "constraint_type",
    "subject",
    "predicate",
    "object",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_identity(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scalar(value: Any, default: Any = 0) -> Any:
    if value is None:
        return default
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def expected_row_identity(rows: Sequence[object], kinds: Sequence[str] | None = None) -> list[dict[str, Any]]:
    identity: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        kind = kinds[index] if kinds is not None and index < len(kinds) else getattr(row, "constraint_type", "UNKNOWN")
        identity.append(
            {
                "row_index": index,
                "constraint_id": int(_scalar(getattr(row, "constraint_id", -1), -1)),
                "constraint_type": str(kind or "UNKNOWN"),
                "subject": int(_scalar(getattr(row, "subject", 0), 0)),
                "predicate": int(_scalar(getattr(row, "predicate", 0), 0)),
                "object": int(_scalar(getattr(row, "object", 0), 0)),
            }
        )
    return identity


def build_predictions_frame(
    predictions: torch.Tensor,
    *,
    rows: Sequence[object],
    kinds: Sequence[str],
    metric_instances: Sequence[dict[str, object]],
) -> pd.DataFrame:
    tensor = predictions.detach().cpu().to(dtype=torch.long)
    if tensor.ndim != 2 or tensor.shape[1] != 6:
        raise ValueError(f"Predictions must have shape (N, 6), got {tuple(tensor.shape)}")
    if not (tensor.shape[0] == len(rows) == len(kinds) == len(metric_instances)):
        raise ValueError(
            "Prediction artifact inputs differ in length: "
            f"predictions={tensor.shape[0]} rows={len(rows)} kinds={len(kinds)} "
            f"metric_instances={len(metric_instances)}"
        )

    records = expected_row_identity(rows)
    for index, record in enumerate(records):
        values = tensor[index].tolist()
        record.update({column: int(value) for column, value in zip(PREDICTION_COLUMNS, values)})
        instance = metric_instances[index]
        for operation in ("add", "del"):
            resolved = instance.get(f"resolved_{operation}")
            if resolved is None:
                resolved = (0, 0, 0)
            record.update(
                {
                    f"resolved_{operation}_subject": int(resolved[0]),
                    f"resolved_{operation}_predicate": int(resolved[1]),
                    f"resolved_{operation}_object": int(resolved[2]),
                }
            )
        events = instance.get("events") or {}
        for metric_name, event in events.items():
            record[f"metric_{metric_name}_numerator"] = int(event["numerator"])
            record[f"metric_{metric_name}_denominator"] = int(event["denominator"])
    return pd.DataFrame.from_records(records)


def backup_schema_v1_once(path: Path) -> Path | None:
    path = Path(path)
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.pre-schema-v2{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_path(path)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def repository_relative_path(path: Path) -> str:
    """Return a portable repository-relative path when possible."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _file_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return {
        "path": repository_relative_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_prediction_artifacts(
    output_dir: Path,
    frame: pd.DataFrame,
    *,
    config_path: Path | None,
    checkpoint_path: Path | None,
    dataset_path: Path,
    graph_paths: Iterable[Path],
    dataset_variant: str,
    split: str = "test",
    source_predictions_path: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.parquet"
    manifest_path = output_dir / "predictions.manifest.json"
    atomic_write_parquet(predictions_path, frame)

    identity_rows = frame.loc[:, list(IDENTITY_COLUMNS)].to_dict(orient="records")
    graph_files = [_file_identity(Path(path)) for path in graph_paths]
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "row_count": len(frame),
        "split": split,
        "checkpoint": _file_identity(checkpoint_path),
        "config": _file_identity(config_path),
        "dataset": {
            "variant": dataset_variant,
            "split": split,
            "artifact": _file_identity(dataset_path),
            "row_identity_sha256": _json_identity(identity_rows),
        },
        "graph": {
            "artifacts": [item for item in graph_files if item is not None],
        },
        "predictions": _file_identity(predictions_path),
    }
    source_predictions = _file_identity(source_predictions_path)
    if source_predictions is not None:
        manifest["source_predictions"] = source_predictions
    atomic_write_json(manifest_path, manifest)
    return predictions_path, manifest_path, manifest


def load_and_validate_predictions(
    predictions_path: Path,
    *,
    rows: Sequence[object],
    dataset_path: Path,
    graph_paths: Iterable[Path],
    dataset_variant: str,
    split: str = "test",
) -> tuple[torch.Tensor, dict[str, Any]]:
    predictions_path = Path(predictions_path)
    if predictions_path.suffix.lower() != ".parquet":
        raise ValueError("Schema-v2 prediction replay requires a Parquet artifact.")
    manifest_path = predictions_path.with_suffix(".manifest.json")
    if not predictions_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Prediction replay requires {predictions_path} and {manifest_path}."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Prediction manifest is not schema version 2.")
    recorded_prediction = manifest.get("predictions") or {}
    if recorded_prediction.get("sha256") != sha256_file(predictions_path):
        raise ValueError("Prediction Parquet checksum does not match its manifest.")

    frame = pd.read_parquet(predictions_path)
    if len(frame) != len(rows) or int(manifest.get("row_count", -1)) != len(rows):
        raise ValueError(
            f"Prediction count mismatch: artifact={len(frame)} manifest={manifest.get('row_count')} "
            f"dataset={len(rows)}"
        )
    missing = set(IDENTITY_COLUMNS + PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction artifact is missing required columns: {sorted(missing)}")

    expected_identity = expected_row_identity(rows)
    actual_identity = frame.loc[:, list(IDENTITY_COLUMNS)].to_dict(orient="records")
    if actual_identity != expected_identity:
        raise ValueError("Prediction row ordering or row identity does not match the evaluation dataset.")
    dataset_manifest = manifest.get("dataset") or {}
    if dataset_manifest.get("variant") != dataset_variant or dataset_manifest.get("split") != split:
        raise ValueError("Prediction dataset identity does not match the requested evaluation dataset.")
    if dataset_manifest.get("row_identity_sha256") != _json_identity(actual_identity):
        raise ValueError("Prediction row-identity checksum is invalid.")
    recorded_dataset = dataset_manifest.get("artifact") or {}
    if recorded_dataset.get("sha256") != sha256_file(dataset_path):
        raise ValueError("Prediction dataset checksum does not match the current interim dataset.")

    recorded_graphs = (manifest.get("graph") or {}).get("artifacts") or []
    current_graphs = [_file_identity(Path(path)) for path in graph_paths]
    recorded_checksums = [item.get("sha256") for item in recorded_graphs]
    current_checksums = [item.get("sha256") for item in current_graphs if item is not None]
    if recorded_checksums != current_checksums:
        raise ValueError("Prediction graph checksum/order does not match the current graph suite.")

    tensor = torch.tensor(frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(), dtype=torch.long)
    return tensor, manifest
