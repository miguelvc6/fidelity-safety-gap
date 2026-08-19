#!/usr/bin/env python3
"""Audit whether G0 behaves like the delete-focus H1 baseline.

The script is read-only with respect to model artifacts. It compares reranker
predictions against the constructed H1 delete-focus policy and evaluates both
through the shared symbolic candidate evaluator.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig
from modules.data_encoders import (
    dataset_variant_name,
    discover_graph_artifacts,
    graph_dataset_filename,
)
from modules.evaluation_artifacts import (
    EVALUATION_SCHEMA_VERSION,
    atomic_write_json,
    load_and_validate_predictions,
    repository_relative_path,
)
from modules.model_store import config_copy_path
from modules.repair_eval import PaperMetricsAccumulator, evaluate_paper_metric_instance

NONE_CLASS_INDEX = 0


def _load_eval_module():
    module_path = ROOT / "src" / "09_eval.py"
    spec = importlib.util.spec_from_file_location("eval_09_for_deletion_degeneracy", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load eval helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit G0 delete-focus degeneracy against H1.")
    parser.add_argument("--g0-run-directory", required=True, help="G0 run directory under models/.")
    parser.add_argument(
        "--predictions",
        "--reranker-predictions",
        dest="predictions",
        default=None,
        help=(
            "Schema-v2 Parquet predictions. --reranker-predictions is retained as an alias. "
            "Defaults to <g0-run-directory>/evaluations/predictions.parquet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <g0-run-directory>/evaluations/deletion_degeneracy.",
    )
    parser.add_argument(
        "--strict-global-metrics",
        action="store_true",
        help="Require strict symbolic evaluator setup, matching paper-suite evaluation.",
    )
    parser.add_argument(
        "--registry-dataset",
        default="full",
        help="Registry dataset fallback for strict symbolic evaluation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix length for smoke tests.")
    parser.add_argument(
        "--examples-limit",
        type=int,
        default=200,
        help="Maximum number of mismatch examples to write.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _load_model_config(run_directory: Path) -> ModelConfig:
    config_path = config_copy_path(run_directory)
    if not config_path.exists():
        raise FileNotFoundError(f"Stored configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return ModelConfig.from_mapping(payload["model_config"])


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split("|") if part]
    if isinstance(value, Sequence):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    return [value]


def _density_bucket(size: int) -> str:
    if size <= 0:
        return "0"
    if size == 1:
        return "1"
    if size <= 4:
        return "2_4"
    if size <= 16:
        return "5_16"
    if size <= 64:
        return "17_64"
    return "65_plus"


def _row_context(row: Any) -> tuple[str, str, int]:
    constraint_type = getattr(row, "constraint_type", None) or "UNKNOWN"
    local_ids = _as_sequence(getattr(row, "local_constraint_ids", None))
    if not local_ids:
        local_ids = _as_sequence(getattr(row, "factor_constraint_ids", None))
    constraint_id = _as_int(getattr(row, "constraint_id", None), default=-1)
    return str(constraint_type), _density_bucket(len(local_ids)), constraint_id


def _slots_from_prediction(item: Any) -> list[int]:
    if isinstance(item, dict):
        add = item.get("add") or [NONE_CLASS_INDEX, NONE_CLASS_INDEX, NONE_CLASS_INDEX]
        delete = item.get("del") or item.get("delete") or [NONE_CLASS_INDEX, NONE_CLASS_INDEX, NONE_CLASS_INDEX]
        return [int(v) for v in [*add, *delete]]
    values = list(item)
    if len(values) != 6:
        raise ValueError(f"Expected 6-slot prediction, got {len(values)} values.")
    return [int(v) for v in values]


def _h1_slots(row: Any) -> list[int]:
    return [
        NONE_CLASS_INDEX,
        NONE_CLASS_INDEX,
        NONE_CLASS_INDEX,
        _as_int(getattr(row, "subject", None)),
        _as_int(getattr(row, "predicate", None)),
        _as_int(getattr(row, "object", None)),
    ]


def _metric_equivalent(g0: dict[str, Any], h1: dict[str, Any]) -> bool:
    return g0.get("events") == h1.get("events")


class Aggregate:
    def __init__(self) -> None:
        self.support = 0
        self.prediction_exact = 0
        self.metric_equivalent = 0
        self.resolved_operation_equivalent = 0
        self.g0_metrics = PaperMetricsAccumulator()
        self.h1_metrics = PaperMetricsAccumulator()

    def add(self, *, g0_slots: list[int], h1_slots: list[int], g0: dict[str, Any], h1: dict[str, Any]) -> None:
        self.support += 1
        self.prediction_exact += int(g0_slots == h1_slots)
        self.metric_equivalent += int(_metric_equivalent(g0, h1))
        self.resolved_operation_equivalent += int(
            g0.get("resolved_add") == h1.get("resolved_add")
            and g0.get("resolved_del") == h1.get("resolved_del")
        )
        self.g0_metrics.update(g0["events"])
        self.h1_metrics.update(h1["events"])

    def to_dict(self) -> dict[str, object]:
        support = self.support
        return {
            "support": support,
            "prediction_exact_match_rate": self.prediction_exact / support if support else 0.0,
            "metric_equivalent_rate": self.metric_equivalent / support if support else 0.0,
            "resolved_operation_equivalent_rate": (
                self.resolved_operation_equivalent / support if support else 0.0
            ),
            "g0_paper_metrics": self.g0_metrics.as_dict(),
            "dfb_paper_metrics": self.h1_metrics.as_dict(),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_analysis(args: argparse.Namespace) -> None:
    run_directory = Path(args.g0_run_directory).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_directory / "evaluations" / "deletion_degeneracy"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = _load_model_config(run_directory)
    global_support = EVAL._maybe_prepare_global_support(
        model_cfg.dataset_variant,
        model_cfg.min_occurrence,
        split="test",
        none_class=NONE_CLASS_INDEX,
        strict=bool(args.strict_global_metrics),
        registry_dataset=args.registry_dataset,
    )
    if global_support is None:
        raise RuntimeError("Deletion-degeneracy analysis requires symbolic global support.")

    rows = global_support.rows
    if global_support.dataset_path is None:
        raise RuntimeError("Deletion-degeneracy analysis requires a concrete interim dataset artifact.")
    variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
    test_graph_path = (
        ROOT
        / "data"
        / "processed"
        / variant
        / graph_dataset_filename(
            "test",
            model_cfg.encoding,
            constraint_representation=model_cfg.constraint_representation,
        )
    )
    graph_paths = [item.path for item in discover_graph_artifacts(test_graph_path)]
    predictions_path = (
        Path(args.predictions).resolve()
        if args.predictions
        else run_directory / "evaluations" / "predictions.parquet"
    )
    g0_predictions, predictions_manifest = load_and_validate_predictions(
        predictions_path,
        rows=rows,
        dataset_path=global_support.dataset_path,
        graph_paths=graph_paths,
        dataset_variant=variant,
    )
    limit = len(rows)
    if args.limit is not None:
        limit = min(limit, args.limit)

    overall = Aggregate()
    by_density: dict[str, Aggregate] = defaultdict(Aggregate)
    by_type: dict[str, Aggregate] = defaultdict(Aggregate)
    examples: list[dict[str, Any]] = []
    constraint_counter: Counter[str] = Counter()

    for idx in range(limit):
        row = rows[idx]
        g0_slots = _slots_from_prediction(g0_predictions[idx])
        h1_slots = _h1_slots(row)
        constraint_type, density_bucket, constraint_id = _row_context(row)
        g0_detail = evaluate_paper_metric_instance(
            row=row,
            evaluator=global_support.evaluator,
            candidate_slots=g0_slots,
            constraint_type=constraint_type,
            row_index=idx,
            none_class=NONE_CLASS_INDEX,
        )
        h1_detail = evaluate_paper_metric_instance(
            row=row,
            evaluator=global_support.evaluator,
            candidate_slots=h1_slots,
            constraint_type=constraint_type,
            row_index=idx,
            none_class=NONE_CLASS_INDEX,
        )
        constraint_counter[constraint_type] += 1
        overall.add(g0_slots=g0_slots, h1_slots=h1_slots, g0=g0_detail, h1=h1_detail)
        by_density[density_bucket].add(g0_slots=g0_slots, h1_slots=h1_slots, g0=g0_detail, h1=h1_detail)
        by_type[constraint_type].add(g0_slots=g0_slots, h1_slots=h1_slots, g0=g0_detail, h1=h1_detail)

        equivalent = _metric_equivalent(g0_detail, h1_detail)
        if (g0_slots != h1_slots or not equivalent) and len(examples) < args.examples_limit:
            examples.append(
                {
                    "index": idx,
                    "constraint_id": constraint_id,
                    "constraint_type": constraint_type,
                    "density_bucket": density_bucket,
                    "prediction_exact_match": int(g0_slots == h1_slots),
                    "metric_equivalent": int(equivalent),
                    "resolved_operation_equivalent": int(
                        g0_detail.get("resolved_add") == h1_detail.get("resolved_add")
                        and g0_detail.get("resolved_del") == h1_detail.get("resolved_del")
                    ),
                    "g0_slots": json.dumps(g0_slots),
                    "h1_slots": json.dumps(h1_slots),
                    "g0_resolved_add": json.dumps(g0_detail.get("resolved_add")),
                    "g0_resolved_del": json.dumps(g0_detail.get("resolved_del")),
                    "dfb_resolved_add": json.dumps(h1_detail.get("resolved_add")),
                    "dfb_resolved_del": json.dumps(h1_detail.get("resolved_del")),
                    "g0_events": json.dumps(g0_detail.get("events"), sort_keys=True),
                    "dfb_events": json.dumps(h1_detail.get("events"), sort_keys=True),
                }
            )

    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "g0_run_directory": repository_relative_path(run_directory),
        "predictions": {
            "path": repository_relative_path(predictions_path),
            "sha256": predictions_manifest["predictions"]["sha256"],
            "row_count": predictions_manifest["row_count"],
        },
        "dataset_variant": variant,
        "min_occurrence": model_cfg.min_occurrence,
        "encoding": model_cfg.encoding,
        "constraint_type_support": dict(sorted(constraint_counter.items())),
        "overall": overall.to_dict(),
    }
    atomic_write_json(output_dir / "deletion_degeneracy_summary.json", summary)

    slice_fields = ["slice", *overall.to_dict().keys()]
    density_rows = [{"slice": key, **agg.to_dict()} for key, agg in sorted(by_density.items())]
    type_rows = [{"slice": key, **agg.to_dict()} for key, agg in sorted(by_type.items())]
    _write_csv(output_dir / "deletion_degeneracy_by_density.csv", density_rows, slice_fields)
    _write_csv(output_dir / "deletion_degeneracy_by_constraint_type.csv", type_rows, slice_fields)
    _write_csv(
        output_dir / "deletion_degeneracy_examples.csv",
        examples,
        [
            "index",
            "constraint_id",
            "constraint_type",
            "density_bucket",
            "prediction_exact_match",
            "metric_equivalent",
            "resolved_operation_equivalent",
            "g0_slots",
            "h1_slots",
            "g0_resolved_add",
            "g0_resolved_del",
            "dfb_resolved_add",
            "dfb_resolved_del",
            "g0_events",
            "dfb_events",
        ],
    )
    logging.info("Wrote deletion-degeneracy outputs to %s", output_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(message)s")
    run_analysis(args)


if __name__ == "__main__":
    main()
