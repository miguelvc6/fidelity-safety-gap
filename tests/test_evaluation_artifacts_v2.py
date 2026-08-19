from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from modules.evaluation_artifacts import (
    EVALUATION_SCHEMA_VERSION,
    REPOSITORY_ROOT,
    backup_schema_v1_once,
    build_predictions_frame,
    load_and_validate_predictions,
    repository_relative_path,
    write_prediction_artifacts,
)
from modules.repair_eval import PAPER_METRIC_KEYS


def _rows():
    return [
        SimpleNamespace(
            constraint_id=10 + index,
            constraint_type="single",
            subject=1 + index,
            predicate=20,
            object=30 + index,
        )
        for index in range(2)
    ]


def _instances():
    events = {
        key: {"numerator": int(key == "pfr"), "denominator": 1}
        for key in PAPER_METRIC_KEYS
    }
    return [
        {
            "events": events,
            "resolved_add": (1 + index, 20, 40),
            "resolved_del": None,
        }
        for index in range(2)
    ]


def _sources(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config.json"
    checkpoint = tmp_path / "checkpoint.pth"
    dataset = tmp_path / "df_test.parquet"
    graph = tmp_path / "test_graph.pt"
    config.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    pd.DataFrame({"row": [0, 1]}).to_parquet(dataset, index=False)
    graph.write_bytes(b"graph")
    return config, checkpoint, dataset, graph


def _write(tmp_path, *, rows=None, predictions=None):
    rows = rows or _rows()
    predictions = predictions if predictions is not None else torch.tensor(
        [[1, 20, 40, 0, 0, 0], [2, 20, 40, 0, 0, 0]], dtype=torch.long
    )
    frame = build_predictions_frame(
        predictions,
        rows=rows,
        kinds=["single"] * len(rows),
        metric_instances=_instances()[: len(rows)],
    )
    config, checkpoint, dataset, graph = _sources(tmp_path)
    paths = write_prediction_artifacts(
        tmp_path / "evaluations",
        frame,
        config_path=config,
        checkpoint_path=checkpoint,
        dataset_path=dataset,
        graph_paths=[graph],
        dataset_variant="toy_minocc100",
    )
    return paths, (config, checkpoint, dataset, graph), predictions


def test_parquet_replay_provenance_and_direct_equality(tmp_path) -> None:
    (predictions_path, _manifest_path, manifest), sources, direct = _write(tmp_path)
    _config, _checkpoint, dataset, graph = sources
    replayed, replay_manifest = load_and_validate_predictions(
        predictions_path,
        rows=_rows(),
        dataset_path=dataset,
        graph_paths=[graph],
        dataset_variant="toy_minocc100",
    )

    torch.testing.assert_close(replayed, direct)
    assert replay_manifest == manifest
    assert manifest["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert manifest["row_count"] == 2
    assert manifest["config"]["sha256"]
    assert manifest["checkpoint"]["sha256"]


def test_replay_rejects_count_order_checksum_and_dataset_identity(tmp_path) -> None:
    (predictions_path, _manifest_path, _manifest), sources, _ = _write(tmp_path)
    _config, _checkpoint, dataset, graph = sources

    with pytest.raises(ValueError, match="count mismatch"):
        load_and_validate_predictions(
            predictions_path,
            rows=_rows()[:1],
            dataset_path=dataset,
            graph_paths=[graph],
            dataset_variant="toy_minocc100",
        )

    with pytest.raises(ValueError, match="dataset identity"):
        load_and_validate_predictions(
            predictions_path,
            rows=_rows(),
            dataset_path=dataset,
            graph_paths=[graph],
            dataset_variant="different",
        )

    order_root = tmp_path / "order"
    swapped_rows = list(reversed(_rows()))
    (swapped_path, _, _), swapped_sources, _ = _write(order_root, rows=swapped_rows)
    _, _, swapped_dataset, swapped_graph = swapped_sources
    with pytest.raises(ValueError, match="ordering or row identity"):
        load_and_validate_predictions(
            swapped_path,
            rows=_rows(),
            dataset_path=swapped_dataset,
            graph_paths=[swapped_graph],
            dataset_variant="toy_minocc100",
        )

    predictions_path.write_bytes(predictions_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_and_validate_predictions(
            predictions_path,
            rows=_rows(),
            dataset_path=dataset,
            graph_paths=[graph],
            dataset_variant="toy_minocc100",
        )


def test_schema_v2_backups_are_created_once(tmp_path) -> None:
    model = tmp_path / "model.json"
    csv = tmp_path / "per_constraint.csv"
    model.write_text('{"legacy": 1}', encoding="utf-8")
    csv.write_text("legacy\n1\n", encoding="utf-8")

    model_backup = backup_schema_v1_once(model)
    csv_backup = backup_schema_v1_once(csv)
    assert model_backup.name == "model.pre-schema-v2.json"
    assert csv_backup.name == "per_constraint.pre-schema-v2.csv"

    model.write_text('{"schema_version": 2}', encoding="utf-8")
    csv.write_text("schema_version\n2\n", encoding="utf-8")
    backup_schema_v1_once(model)
    backup_schema_v1_once(csv)
    assert model_backup.read_text(encoding="utf-8") == '{"legacy": 1}'
    assert csv_backup.read_text(encoding="utf-8") == "legacy\n1\n"


def test_manifest_records_legacy_prediction_source_checksum(tmp_path) -> None:
    rows = _rows()
    predictions = torch.tensor(
        [[1, 20, 40, 0, 0, 0], [2, 20, 40, 0, 0, 0]], dtype=torch.long
    )
    frame = build_predictions_frame(
        predictions,
        rows=rows,
        kinds=["single"] * len(rows),
        metric_instances=_instances(),
    )
    config, checkpoint, dataset, graph = _sources(tmp_path)
    source = tmp_path / "legacy_predictions.json"
    source.write_text("[]", encoding="utf-8")

    _predictions_path, _manifest_path, manifest = write_prediction_artifacts(
        tmp_path / "evaluations",
        frame,
        config_path=config,
        checkpoint_path=checkpoint,
        dataset_path=dataset,
        graph_paths=[graph],
        dataset_variant="toy_minocc100",
        source_predictions_path=source,
    )

    assert manifest["source_predictions"]["path"] == str(source.resolve())
    assert manifest["source_predictions"]["sha256"]


def test_repository_paths_are_recorded_portably() -> None:
    assert repository_relative_path(REPOSITORY_ROOT / "models" / "run" / "config.json") == (
        "models/run/config.json"
    )
