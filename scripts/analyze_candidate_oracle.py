#!/usr/bin/env python3
"""Analyze candidate-set oracle headroom for a trained proposal run.

The script compares the model-selected edit against the best candidate available
inside the same generated candidate set. It does not train or mutate model
artifacts; outputs are written under the requested oracle directory.
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

import torch
from torch_geometric.loader import DataLoader
from tqdm.autonotebook import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.candidates import (
    CandidateConfig,
    batch_topk_candidate_triples,
    build_candidates,
    score_candidates_from_logits,
)
from modules.config import ModelConfig, TrainingConfig
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
    sha256_file,
)
from modules.model_store import config_copy_path, get_checkpoint_path
from modules.repair_eval import PaperMetricsAccumulator, evaluate_paper_metric_instance
from modules.training_utils import load_graph_dataset

NONE_CLASS_INDEX = 0


def _load_eval_module():
    module_path = ROOT / "src" / "09_eval.py"
    spec = importlib.util.spec_from_file_location("eval_09_for_candidate_oracle", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load eval helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate-oracle headroom for a trained run.")
    parser.add_argument("--run-directory", required=True, help="Trained run directory under models/.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-directory>/evaluations/oracle.",
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix length for smoke tests.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device. Use cpu for lightweight smoke tests on busy GPU machines.",
    )
    parser.add_argument(
        "--max-safe-disruption",
        type=int,
        default=2,
        help="Maximum add+delete operation count for the safe-oracle availability flag.",
    )
    parser.add_argument(
        "--examples-limit",
        type=int,
        default=200,
        help="Maximum number of oracle gap examples to write.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _load_config(run_directory: Path) -> tuple[ModelConfig, TrainingConfig]:
    config_path = config_copy_path(run_directory)
    if not config_path.exists():
        raise FileNotFoundError(f"Stored configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return (
        ModelConfig.from_mapping(payload["model_config"]),
        TrainingConfig.from_mapping(payload.get("training_config", {})),
    )


def _load_test_data(model_cfg: ModelConfig):
    variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
    base_path = ROOT / "data" / "processed" / variant
    graph_path = base_path / graph_dataset_filename(
        "test",
        model_cfg.encoding,
        constraint_representation=model_cfg.constraint_representation,
    )
    return load_graph_dataset(graph_path)


def _test_graph_path(model_cfg: ModelConfig) -> Path:
    variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
    return ROOT / "data" / "processed" / variant / graph_dataset_filename(
        "test",
        model_cfg.encoding,
        constraint_representation=model_cfg.constraint_representation,
    )


def _set_context_indices(test_data: Any) -> None:
    if isinstance(test_data, list):
        for idx, graph in enumerate(test_data):
            setattr(graph, "context_index", idx)


def _candidate_config(training_cfg: TrainingConfig) -> CandidateConfig:
    if training_cfg.chooser.enabled:
        return CandidateConfig(
            topk_candidates=training_cfg.chooser.topk_candidates,
            max_candidates_total=training_cfg.chooser.max_candidates_total,
            include_gold=False,
        )
    if training_cfg.direct_safety.enabled:
        return CandidateConfig(
            topk_candidates=training_cfg.direct_safety.topk_candidates,
            max_candidates_total=training_cfg.direct_safety.max_candidates_total,
            include_gold=False,
        )
    return CandidateConfig(include_gold=False)


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


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].item())
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).detach().cpu().tolist()
    if isinstance(value, str):
        return [part for part in value.split("|") if part]
    try:
        if isinstance(value, float) and math.isnan(value):
            return []
    except Exception:
        pass
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _row_context(row: Any) -> tuple[str, str, int]:
    constraint_type = getattr(row, "constraint_type", None) or "UNKNOWN"
    local_ids = _as_sequence(getattr(row, "local_constraint_ids", None))
    if not local_ids:
        local_ids = _as_sequence(getattr(row, "factor_constraint_ids", None))
    constraint_id = _as_int(getattr(row, "constraint_id", None), default=-1)
    return str(constraint_type), _density_bucket(len(local_ids)), constraint_id


def _event(metrics: dict[str, Any], key: str) -> dict[str, int]:
    return metrics["events"][key]


def _event_rate(metrics: dict[str, Any], key: str) -> float:
    event = _event(metrics, key)
    denominator = int(event["denominator"])
    return float(event["numerator"]) / denominator if denominator else 0.0


def _total_ops(metrics: dict[str, Any]) -> int:
    return int(_event(metrics, "disruption")["numerator"])


def _safe_flag(metrics: dict[str, Any], *, max_disruption: int) -> bool:
    return (
        int(_event(metrics, "pfr")["numerator"]) == 1
        and int(_event(metrics, "srr")["numerator"]) == 0
        and int(_event(metrics, "delta_local_satisfaction")["numerator"]) >= 0
        and _total_ops(metrics) <= max_disruption
    )


def _non_vacuous_safe_flag(metrics: dict[str, Any], *, max_disruption: int) -> bool:
    return _safe_flag(metrics, max_disruption=max_disruption) and bool(
        int(metrics.get("post_base_present", 0))
    )


def _oracle_index(details: list[dict[str, Any]], scores: Sequence[float]) -> int:
    if not details:
        raise ValueError("Cannot choose an oracle candidate from an empty candidate set.")

    def key(idx: int) -> tuple[float, float, float, float, float, float]:
        item = details[idx]
        return (
            float(_event(item, "pfr")["numerator"]),
            1.0 if int(_event(item, "srr")["numerator"]) == 0 else 0.0,
            _event_rate(item, "local_satisfaction"),
            -float(_total_ops(item)),
            -_event_rate(item, "srr"),
            float(scores[idx]) if idx < len(scores) else 0.0,
        )

    return max(range(len(details)), key=key)


def _non_vacuous_oracle_index(
    details: list[dict[str, Any]],
    scores: Sequence[float],
    *,
    max_disruption: int,
) -> int | None:
    valid = [
        idx
        for idx, item in enumerate(details)
        if _non_vacuous_safe_flag(item, max_disruption=max_disruption)
    ]
    if not valid:
        return None

    def key(idx: int) -> tuple[float, float, float, float]:
        item = details[idx]
        return (
            _event_rate(item, "local_satisfaction"),
            -float(_total_ops(item)),
            -_event_rate(item, "srr"),
            float(scores[idx]) if idx < len(scores) else 0.0,
        )

    return max(valid, key=key)


def _slot_list(slots: Sequence[int]) -> list[int]:
    return [int(v) for v in slots]


def _selected_from_argmax(logits: torch.Tensor) -> list[int]:
    return [int(v) for v in torch.argmax(logits, dim=-1).detach().cpu().tolist()]


def _selected_from_candidates(
    *,
    model: Any,
    graph_emb: torch.Tensor | None,
    logits: torch.Tensor,
    candidates: list[tuple[int, int, int, int, int, int]],
    candidate_scores: torch.Tensor,
    training_cfg: TrainingConfig,
) -> tuple[list[int], int | None]:
    if training_cfg.chooser.enabled:
        if graph_emb is None:
            raise RuntimeError("Chooser run requires graph_emb from model outputs.")
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits.device)
        scores = model.score_candidates(graph_emb, candidate_tensor)
        best_idx = int(torch.argmax(scores).item())
        return list(candidates[best_idx]), best_idx
    if training_cfg.direct_safety.enabled:
        best_idx = int(torch.argmax(candidate_scores).item())
        return list(candidates[best_idx]), best_idx
    return _selected_from_argmax(logits), None


class Aggregate:
    def __init__(self) -> None:
        self.support = 0
        self.candidate_nonempty = 0
        self.safe_available = 0
        self.selected_safe = 0
        self.non_vacuous_safe_available = 0
        self.selected_non_vacuous_safe = 0
        self.candidate_count_sum = 0
        self.oracle_metrics = PaperMetricsAccumulator()
        self.selected_metrics = PaperMetricsAccumulator()

    def add(
        self,
        *,
        candidate_count: int,
        oracle: dict[str, Any],
        selected: dict[str, Any],
        oracle_safe: bool,
        selected_safe: bool,
        oracle_non_vacuous_safe: bool,
        selected_non_vacuous_safe: bool,
    ) -> None:
        self.support += 1
        self.candidate_nonempty += int(candidate_count > 0)
        self.safe_available += int(oracle_safe)
        self.selected_safe += int(selected_safe)
        self.non_vacuous_safe_available += int(oracle_non_vacuous_safe)
        self.selected_non_vacuous_safe += int(selected_non_vacuous_safe)
        self.candidate_count_sum += int(candidate_count)
        self.oracle_metrics.update(oracle["events"])
        self.selected_metrics.update(selected["events"])

    def to_dict(self) -> dict[str, object]:
        support = self.support
        return {
            "support": support,
            "candidate_set_nonempty_rate": self.candidate_nonempty / support if support else 0.0,
            "candidate_count_mean": self.candidate_count_sum / support if support else 0.0,
            "oracle_safe_available_rate": self.safe_available / support if support else 0.0,
            "selected_safe_rate": self.selected_safe / support if support else 0.0,
            "oracle_non_vacuous_safe_available_rate": self.non_vacuous_safe_available / support
            if support
            else 0.0,
            "selected_non_vacuous_safe_rate": self.selected_non_vacuous_safe / support if support else 0.0,
            "oracle_paper_metrics": self.oracle_metrics.as_dict(),
            "selected_paper_metrics": self.selected_metrics.as_dict(),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def run_analysis(args: argparse.Namespace) -> None:
    run_directory = Path(args.run_directory).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_directory / "evaluations" / "oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg, training_cfg = _load_config(run_directory)
    test_data = _load_test_data(model_cfg)
    _set_context_indices(test_data)

    repair_support = EVAL._maybe_prepare_repair_support(
        model_cfg.dataset_variant,
        model_cfg.min_occurrence,
        split="test",
        none_class=NONE_CLASS_INDEX,
    )
    global_support = EVAL._maybe_prepare_global_support(
        model_cfg.dataset_variant,
        model_cfg.min_occurrence,
        split="test",
        none_class=NONE_CLASS_INDEX,
        strict=bool(args.strict_global_metrics),
        registry_dataset=args.registry_dataset,
    )
    if global_support is None:
        raise RuntimeError("Candidate-oracle analysis requires symbolic global support.")
    if global_support.dataset_path is None:
        raise RuntimeError("Candidate-oracle analysis requires a concrete interim dataset artifact.")

    if args.device == "auto":
        device = EVAL.get_device()
    else:
        device = torch.device(args.device)
        logging.info("Device: %s", device)
    chooser_support = object() if training_cfg.chooser.enabled else None
    model = EVAL.load_trained_model_for_eval(
        run_directory=run_directory,
        model_cfg=model_cfg,
        device=device,
        chooser_support=chooser_support,
    )
    model.eval()

    candidate_cfg = _candidate_config(training_cfg)
    placeholder_ids = set(repair_support.heuristics.placeholder_ids.values())
    rows = global_support.rows
    contexts = repair_support.contexts
    prediction_path = run_directory / "evaluations" / "predictions.parquet"
    graph_paths = [item.path for item in discover_graph_artifacts(_test_graph_path(model_cfg))]
    selected_predictions, selected_manifest = load_and_validate_predictions(
        prediction_path,
        rows=rows,
        dataset_path=global_support.dataset_path,
        graph_paths=graph_paths,
        dataset_variant=dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence),
    )

    overall = Aggregate()
    by_density: dict[str, Aggregate] = defaultdict(Aggregate)
    by_type: dict[str, Aggregate] = defaultdict(Aggregate)
    examples: list[dict[str, Any]] = []
    constraint_counter: Counter[str] = Counter()

    effective_batch_size = min(args.batch_size, args.limit) if args.limit is not None else args.batch_size
    effective_batch_size = max(1, int(effective_batch_size))
    loader = DataLoader(test_data, batch_size=effective_batch_size)
    processed = 0
    for batch in tqdm(loader, desc="Candidate oracle"):
        graphs = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
        batch = batch.to(device)
        # Candidate analysis needs proposal logits and graph embeddings only.
        # The evaluation-safe path deliberately skips post-gold factor logits,
        # whose test gold IDs may be absent from the compact train/val target
        # vocabulary.
        out = model.forward_for_evaluation(batch)
        if isinstance(out, dict):
            logits_batch = out.get("edit_logits")
            graph_emb_batch = out.get("graph_emb")
            if logits_batch is None:
                raise KeyError("Model output dict missing edit_logits.")
        else:
            logits_batch = out
            graph_emb_batch = None
        logits_batch_cpu = logits_batch.detach().cpu()
        batch_add_topk, batch_del_topk = batch_topk_candidate_triples(
            logits_batch.detach(),
            topk_triples=candidate_cfg.topk_candidates,
            topk_per_slot=candidate_cfg.topk_per_slot,
        )

        for local_idx, graph in enumerate(graphs):
            if args.limit is not None and processed >= args.limit:
                break
            context_index = int(getattr(graph, "context_index", processed))
            if context_index >= len(contexts) or context_index >= len(rows):
                raise RuntimeError(f"Context index {context_index} out of bounds.")
            context = contexts[context_index]
            row = rows[context_index]
            # Move the whole proposal row once before candidate expansion.  Candidate
            # construction and oracle tie-breaking are CPU work; keeping logits on
            # CUDA here would force several tiny device synchronizations per row.
            logits = logits_batch_cpu[local_idx]
            candidates, _gold_index = build_candidates(
                graph=graph,
                context=context,
                heuristics=repair_support.heuristics,
                proposal_logits=logits,
                cfg=candidate_cfg,
                placeholder_ids=placeholder_ids,
                num_target_ids=model.num_target_ids,
                precomputed_add_topk=batch_add_topk[local_idx],
                precomputed_del_topk=batch_del_topk[local_idx],
            )
            if not candidates:
                processed += 1
                continue

            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits.device)
            candidate_scores_tensor = score_candidates_from_logits(logits, candidate_tensor)
            candidate_scores = [float(v) for v in candidate_scores_tensor.detach().cpu().tolist()]
            # Selected-candidate diagnostics must describe the exact paper prediction,
            # including deterministic resolution of near-tied CUDA scores.  Replaying
            # the validated schema-v2 artifact avoids treating an independent rerun as
            # evaluation truth while the candidate set below remains freshly rebuilt.
            selected_slots = [int(value) for value in selected_predictions[context_index].tolist()]
            selected_tuple = tuple(selected_slots)
            selected_candidate_index = (
                candidates.index(selected_tuple) if selected_tuple in candidates else None
            )

            constraint_type, density_bucket, constraint_id = _row_context(row)
            raw_candidate_details = global_support.evaluator.evaluate_candidates(
                row,
                candidates=candidates,
                primary_factor_index=None,
            )
            candidate_details = [
                evaluate_paper_metric_instance(
                    row=row,
                    evaluator=global_support.evaluator,
                    candidate_slots=candidate,
                    constraint_type=constraint_type,
                    row_index=context_index,
                    none_class=NONE_CLASS_INDEX,
                    details=raw_detail,
                )
                for candidate, raw_detail in zip(candidates, raw_candidate_details)
            ]
            raw_selected_detail = global_support.evaluator.evaluate_candidates(
                row,
                candidates=[selected_slots],
                primary_factor_index=None,
            )[0]
            selected_detail = evaluate_paper_metric_instance(
                row=row,
                evaluator=global_support.evaluator,
                candidate_slots=selected_slots,
                constraint_type=constraint_type,
                row_index=context_index,
                none_class=NONE_CLASS_INDEX,
                details=raw_selected_detail,
            )
            oracle_candidate_index = _oracle_index(candidate_details, candidate_scores)
            oracle_detail = candidate_details[oracle_candidate_index]
            non_vacuous_oracle_candidate_index = _non_vacuous_oracle_index(
                candidate_details,
                candidate_scores,
                max_disruption=args.max_safe_disruption,
            )
            non_vacuous_oracle_detail = (
                candidate_details[non_vacuous_oracle_candidate_index]
                if non_vacuous_oracle_candidate_index is not None
                else None
            )
            oracle_safe = _safe_flag(
                oracle_detail,
                max_disruption=args.max_safe_disruption,
            )
            selected_safe = _safe_flag(
                selected_detail,
                max_disruption=args.max_safe_disruption,
            )
            oracle_non_vacuous_safe = non_vacuous_oracle_detail is not None
            selected_non_vacuous_safe = _non_vacuous_safe_flag(
                selected_detail,
                max_disruption=args.max_safe_disruption,
            )

            constraint_counter[constraint_type] += 1
            overall.add(
                candidate_count=len(candidates),
                oracle=oracle_detail,
                selected=selected_detail,
                oracle_safe=oracle_safe,
                selected_safe=selected_safe,
                oracle_non_vacuous_safe=oracle_non_vacuous_safe,
                selected_non_vacuous_safe=selected_non_vacuous_safe,
            )
            by_density[density_bucket].add(
                candidate_count=len(candidates),
                oracle=oracle_detail,
                selected=selected_detail,
                oracle_safe=oracle_safe,
                selected_safe=selected_safe,
                oracle_non_vacuous_safe=oracle_non_vacuous_safe,
                selected_non_vacuous_safe=selected_non_vacuous_safe,
            )
            by_type[constraint_type].add(
                candidate_count=len(candidates),
                oracle=oracle_detail,
                selected=selected_detail,
                oracle_safe=oracle_safe,
                selected_safe=selected_safe,
                oracle_non_vacuous_safe=oracle_non_vacuous_safe,
                selected_non_vacuous_safe=selected_non_vacuous_safe,
            )

            selected_worse = (
                oracle_safe
                and not selected_safe
                or oracle_non_vacuous_safe
                and not selected_non_vacuous_safe
                or _event_rate(oracle_detail, "local_satisfaction")
                > _event_rate(selected_detail, "local_satisfaction")
                or int(_event(oracle_detail, "pfr")["numerator"])
                > int(_event(selected_detail, "pfr")["numerator"])
            )
            if selected_worse and len(examples) < args.examples_limit:
                examples.append(
                    {
                        "index": context_index,
                        "constraint_id": constraint_id,
                        "constraint_type": constraint_type,
                        "density_bucket": density_bucket,
                        "candidate_count": len(candidates),
                        "selected_candidate_index": selected_candidate_index
                        if selected_candidate_index is not None
                        else "",
                        "oracle_candidate_index": oracle_candidate_index,
                        "non_vacuous_oracle_candidate_index": non_vacuous_oracle_candidate_index
                        if non_vacuous_oracle_candidate_index is not None
                        else "",
                        "selected_slots": json.dumps(_slot_list(selected_slots)),
                        "oracle_slots": json.dumps(_slot_list(candidates[oracle_candidate_index])),
                        "non_vacuous_oracle_slots": json.dumps(
                            _slot_list(candidates[non_vacuous_oracle_candidate_index])
                        )
                        if non_vacuous_oracle_candidate_index is not None
                        else "",
                        "selected_safe": int(selected_safe),
                        "oracle_safe": int(oracle_safe),
                        "selected_non_vacuous_safe": int(selected_non_vacuous_safe),
                        "oracle_non_vacuous_safe": int(oracle_non_vacuous_safe),
                        "selected_events": json.dumps(selected_detail["events"], sort_keys=True),
                        "oracle_events": json.dumps(oracle_detail["events"], sort_keys=True),
                        "non_vacuous_oracle_events": (
                            json.dumps(non_vacuous_oracle_detail["events"], sort_keys=True)
                            if non_vacuous_oracle_detail is not None
                            else ""
                        ),
                    }
                )

            processed += 1
        if args.limit is not None and processed >= args.limit:
            break

    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "run_directory": repository_relative_path(run_directory),
        "config": {
            "path": repository_relative_path(config_copy_path(run_directory)),
            "sha256": sha256_file(config_copy_path(run_directory)),
        },
        "checkpoint": {
            "path": repository_relative_path(get_checkpoint_path(run_directory)),
            "sha256": sha256_file(get_checkpoint_path(run_directory)),
        },
        "dataset": {
            "variant": dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence),
            "path": repository_relative_path(global_support.dataset_path),
            "sha256": sha256_file(global_support.dataset_path),
        },
        "min_occurrence": model_cfg.min_occurrence,
        "encoding": model_cfg.encoding,
        "constraint_representation": model_cfg.constraint_representation,
        "selection_mode": "chooser"
        if training_cfg.chooser.enabled
        else ("direct_safety" if training_cfg.direct_safety.enabled else "slot_argmax"),
        "selected_prediction_source": {
            "mode": "validated_schema_v2_replay",
            "path": repository_relative_path(prediction_path),
            "sha256": selected_manifest["predictions"]["sha256"],
        },
        "candidate_config": {
            "topk_candidates": candidate_cfg.topk_candidates,
            "topk_per_slot": candidate_cfg.topk_per_slot,
            "heuristic_max_candidates": candidate_cfg.heuristic_max_candidates,
            "heuristic_max_values": candidate_cfg.heuristic_max_values,
            "include_gold": candidate_cfg.include_gold,
            "max_candidates_total": candidate_cfg.max_candidates_total,
        },
        "max_safe_disruption": args.max_safe_disruption,
        "constraint_type_support": dict(sorted(constraint_counter.items())),
        "overall": overall.to_dict(),
    }
    atomic_write_json(output_dir / "oracle_summary.json", summary)

    slice_fields = ["slice", *overall.to_dict().keys()]
    density_rows = [{"slice": key, **agg.to_dict()} for key, agg in sorted(by_density.items())]
    type_rows = [{"slice": key, **agg.to_dict()} for key, agg in sorted(by_type.items())]
    _write_csv(output_dir / "oracle_by_density.csv", density_rows, slice_fields)
    _write_csv(output_dir / "oracle_by_constraint_type.csv", type_rows, slice_fields)
    example_fields = [
        "index",
        "constraint_id",
        "constraint_type",
        "density_bucket",
        "candidate_count",
        "selected_candidate_index",
        "oracle_candidate_index",
        "non_vacuous_oracle_candidate_index",
        "selected_slots",
        "oracle_slots",
        "non_vacuous_oracle_slots",
        "selected_safe",
        "oracle_safe",
        "selected_non_vacuous_safe",
        "oracle_non_vacuous_safe",
        "selected_events",
        "oracle_events",
        "non_vacuous_oracle_events",
    ]
    _write_csv(output_dir / "oracle_examples.csv", examples, example_fields)
    logging.info("Wrote candidate-oracle outputs to %s", output_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(message)s")
    run_analysis(args)


if __name__ == "__main__":
    main()
