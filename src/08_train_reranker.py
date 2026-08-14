#!/usr/bin/env python3


import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from modules.config import ModelConfig
from modules.data_encoders import (
    GlobalIntEncoder,
    GraphStreamDataset,
    base_dataset_name,
    dataset_variant_name,
    discover_graph_artifacts,
    graph_dataset_filename,
    infer_node_feature_spec,
)
from modules.model_store import (
    atomic_torch_save,
    available_config_tags,
    ensure_run_dir_for_config,
    get_checkpoint_path,
    get_last_checkpoint_path,
    history_path,
    resolve_run_dir,
)
from modules.evaluation_artifacts import atomic_write_json, sha256_file
from modules.models import build_model
from modules.repair_eval import ConstraintRepairHeuristics, ViolationContext, load_violation_contexts
from modules.candidates import CandidateConfig, batch_topk_candidate_triples, build_candidates
from modules.reranker import CandidateReranker, RerankerConfig, build_reranker
from modules.reranker_eval import CandidateConstraintEvaluator
from modules.training_utils import (
    load_graph_dataset,
    placeholder_ids_from_encoder,
    progress_bar,
    set_seed,
)


def _derive_graph_model_cfg_from_state_dict(
    state_dict: dict,
    *,
    model_cfg: ModelConfig,
    proposal_cfg: dict,
) -> ModelConfig:
    payload = model_cfg.to_dict()
    payload["model"] = proposal_cfg.get("model", "GIN")
    payload["use_node_embeddings"] = "graph_encoder.node_embeddings.weight" in state_dict
    payload["use_role_embeddings"] = "graph_encoder.role_embeddings.weight" in state_dict
    if payload["use_role_embeddings"]:
        role_weight = state_dict["graph_encoder.role_embeddings.weight"]
        payload["role_embedding_dim"] = int(role_weight.shape[1])
        payload["num_role_types"] = int(role_weight.shape[0])
    if payload["use_node_embeddings"]:
        emb_weight = state_dict["graph_encoder.node_embeddings.weight"]
        payload["num_embedding_size"] = int(emb_weight.shape[1])
    else:
        init_weight = state_dict.get("graph_encoder.initialization.0.weight")
        if init_weight is not None:
            input_channels = int(init_weight.shape[1])
            role_dim = int(payload.get("role_embedding_dim", 0)) if payload["use_role_embeddings"] else 0
            payload["num_embedding_size"] = max(input_channels - role_dim, 1)
    return ModelConfig.from_mapping(payload)

NUM_SLOTS = 6
NONE_CLASS_INDEX = 0

logger = logging.getLogger(__name__)


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _global_fix_loss(
    probs: torch.Tensor,
    metrics_summary: Sequence[Any],
    *,
    focus_deletion_weight: float,
    device: torch.device,
) -> torch.Tensor:
    satisfaction = torch.tensor(
        [m.global_satisfied_fraction for m in metrics_summary],
        dtype=torch.float32,
        device=device,
    )
    expected_satisfaction = torch.sum(probs * satisfaction)
    if focus_deletion_weight == 0.0:
        return -expected_satisfaction
    focus_deleted = torch.tensor(
        [m.focus_deleted for m in metrics_summary],
        dtype=torch.float32,
        device=device,
    )
    return -expected_satisfaction + focus_deletion_weight * torch.sum(probs * focus_deleted)


class ValidationSubsetStream(IterableDataset):
    """Yield the first ``limit`` graphs from a streamed validation dataset."""

    def __init__(self, dataset: IterableDataset, limit: int) -> None:
        if limit <= 0:
            raise ValueError("Validation subset limit must be positive.")
        self.dataset = dataset
        self.limit = int(limit)

    def __iter__(self):
        for idx, graph in enumerate(self.dataset):
            if idx >= self.limit:
                break
            yield graph

    def __len__(self) -> int:
        return self.limit


@dataclass
class RerankerTrainingConfig:
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 5
    early_stopping_rounds: int = 3
    grad_clip: float | None = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    validation_subset_size: int | None = None
    objective: str = "main"  # main | global_fix
    regression_weight: float = 0.5
    topk_candidates: int = 20
    topk_per_slot: int = 5
    heuristic_max_candidates: int = 30
    heuristic_max_values: int = 3
    include_gold: bool = True
    prediction_include_gold: bool = False
    max_candidates_total: int = 80
    assume_complete_entity_facts: bool = True
    constraint_scope: str = "local"  # local | focus
    focus_deletion_weight: float = 0.0
    save_last_checkpoint: bool = False

    @classmethod
    def from_mapping(cls, data: dict | None) -> "RerankerTrainingConfig":
        payload = dict(data or {})
        objective = str(payload.get("objective", cls.objective)).lower()
        if objective not in {"main", "global_fix"}:
            raise ValueError("training_config.objective must be 'main' or 'global_fix'")
        if "beta" in payload and "regression_weight" not in payload:
            payload["regression_weight"] = payload.get("beta")
        constraint_scope = str(payload.get("constraint_scope", cls.constraint_scope)).lower()
        if constraint_scope not in {"local", "focus"}:
            raise ValueError("training_config.constraint_scope must be 'local' or 'focus'")
        validation_subset_size = payload.get("validation_subset_size", cls.validation_subset_size)
        if validation_subset_size is not None:
            validation_subset_size = int(validation_subset_size)
            if validation_subset_size <= 0:
                raise ValueError("training_config.validation_subset_size must be positive when set")
        focus_deletion_weight = float(
            payload.get("focus_deletion_weight", cls.focus_deletion_weight)
        )
        if focus_deletion_weight < 0.0:
            raise ValueError("training_config.focus_deletion_weight must be non-negative")
        return cls(
            seed=int(payload.get("seed", cls.seed)),
            batch_size=int(payload.get("batch_size", cls.batch_size)),
            num_epochs=int(payload.get("num_epochs", cls.num_epochs)),
            early_stopping_rounds=int(payload.get("early_stopping_rounds", cls.early_stopping_rounds)),
            grad_clip=payload.get("grad_clip", cls.grad_clip),
            learning_rate=float(payload.get("learning_rate", cls.learning_rate)),
            weight_decay=float(payload.get("weight_decay", cls.weight_decay)),
            scheduler_factor=float(payload.get("scheduler_factor", cls.scheduler_factor)),
            scheduler_patience=int(payload.get("scheduler_patience", cls.scheduler_patience)),
            num_workers=int(payload.get("num_workers", cls.num_workers)),
            pin_memory=bool(payload.get("pin_memory", cls.pin_memory)),
            validation_subset_size=validation_subset_size,
            objective=objective,
            regression_weight=float(payload.get("regression_weight", cls.regression_weight)),
            topk_candidates=int(payload.get("topk_candidates", cls.topk_candidates)),
            topk_per_slot=int(payload.get("topk_per_slot", cls.topk_per_slot)),
            heuristic_max_candidates=int(payload.get("heuristic_max_candidates", cls.heuristic_max_candidates)),
            heuristic_max_values=int(payload.get("heuristic_max_values", cls.heuristic_max_values)),
            include_gold=bool(payload.get("include_gold", cls.include_gold)),
            prediction_include_gold=bool(
                payload.get("prediction_include_gold", cls.prediction_include_gold)
            ),
            max_candidates_total=int(payload.get("max_candidates_total", cls.max_candidates_total)),
            assume_complete_entity_facts=bool(
                payload.get("assume_complete_entity_facts", cls.assume_complete_entity_facts)
            ),
            constraint_scope=constraint_scope,
            focus_deletion_weight=focus_deletion_weight,
            save_last_checkpoint=bool(
                payload.get("save_last_checkpoint", cls.save_last_checkpoint)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "early_stopping_rounds": self.early_stopping_rounds,
            "grad_clip": self.grad_clip,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_patience": self.scheduler_patience,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "validation_subset_size": self.validation_subset_size,
            "objective": self.objective,
            "regression_weight": self.regression_weight,
            "topk_candidates": self.topk_candidates,
            "topk_per_slot": self.topk_per_slot,
            "heuristic_max_candidates": self.heuristic_max_candidates,
            "heuristic_max_values": self.heuristic_max_values,
            "include_gold": self.include_gold,
            "prediction_include_gold": self.prediction_include_gold,
            "max_candidates_total": self.max_candidates_total,
            "assume_complete_entity_facts": self.assume_complete_entity_facts,
            "constraint_scope": self.constraint_scope,
            "focus_deletion_weight": self.focus_deletion_weight,
            "save_last_checkpoint": self.save_last_checkpoint,
        }


def _load_experiment_config(path: Path) -> tuple[ModelConfig, RerankerConfig, RerankerTrainingConfig, dict]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a JSON object.")

    model_cfg = ModelConfig.from_mapping(payload.get("model_config", {}))
    reranker_cfg = RerankerConfig.from_mapping(payload.get("reranker_config", {}))
    training_cfg = RerankerTrainingConfig.from_mapping(payload.get("training_config", {}))
    proposal_cfg = dict(payload.get("proposal_config", {}))
    return model_cfg, reranker_cfg, training_cfg, proposal_cfg


def _write_effective_experiment_config(
    config_path: Path,
    original_payload: dict[str, Any],
    *,
    model_cfg: ModelConfig,
    reranker_cfg: RerankerConfig,
    training_cfg: RerankerTrainingConfig,
    proposal_cfg: dict[str, Any],
) -> None:
    payload = dict(original_payload)
    payload["model_config"] = model_cfg.to_dict()
    payload["reranker_config"] = reranker_cfg.to_dict()
    payload["training_config"] = training_cfg.to_dict()
    payload["proposal_config"] = dict(proposal_cfg)
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _load_encoder(interim_path: Path) -> GlobalIntEncoder:
    encoder = GlobalIntEncoder()
    encoder.load(interim_path / "globalintencoder.txt")
    encoder.freeze()
    return encoder


def _load_parquet_rows(interim_path: Path, split: str) -> list:
    import pandas as pd

    path = interim_path / f"df_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet split not found at {path}")
    columns = [
        "constraint_id",
        "constraint_type",
        "subject",
        "predicate",
        "object",
        "other_subject",
        "other_predicate",
        "other_object",
        "constraint_predicates",
        "constraint_objects",
        "subject_predicates",
        "subject_objects",
        "object_predicates",
        "object_objects",
        "other_entity_predicates",
        "other_entity_objects",
        "local_constraint_ids",
        "local_constraint_ids_focus",
    ]
    df = pd.read_parquet(path)
    existing = [col for col in columns if col in df.columns]
    if existing:
        df = df[existing]
    return list(df.itertuples(index=False))


def _resolve_proposal_checkpoint(
    proposal_cfg: dict,
    *,
    model_cfg: ModelConfig,
) -> Path:
    if "checkpoint_path" in proposal_cfg:
        return Path(proposal_cfg["checkpoint_path"])
    if "run_dir" in proposal_cfg:
        return Path(proposal_cfg["run_dir"]) / "checkpoint.pth"
    if "config_tag" in proposal_cfg or "model" in proposal_cfg:
        model_name = proposal_cfg.get("model", model_cfg.model)
        config_tag = proposal_cfg.get("config_tag")
        try:
            run_dir = resolve_run_dir(
                model_cfg.dataset_variant,
                model_cfg.encoding,
                model_name,
                config_tag,
            )
        except FileNotFoundError as exc:
            if not config_tag:
                raise
            candidates = [
                tag
                for tag in available_config_tags(model_cfg.dataset_variant, model_cfg.encoding, model_name)
                if tag == config_tag or tag.startswith(f"{config_tag}__")
            ]
            if len(candidates) == 1:
                run_dir = resolve_run_dir(
                    model_cfg.dataset_variant,
                    model_cfg.encoding,
                    model_name,
                    candidates[0],
                )
            elif len(candidates) > 1:
                joined = ", ".join(candidates)
                raise FileNotFoundError(
                    f"Multiple proposal runs match config_tag {config_tag}: {joined}"
                ) from exc
            else:
                raise
        return run_dir / "checkpoint.pth"
    raise ValueError("proposal_config requires checkpoint_path, run_dir, or model/config_tag.")


def _load_proposal_model(
    proposal_cfg: dict,
    *,
    num_input_graph_nodes: int,
    device: torch.device,
    fallback_model_cfg: ModelConfig,
) -> nn.Module:
    checkpoint_path = _resolve_proposal_checkpoint(proposal_cfg, model_cfg=fallback_model_cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Proposal checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", fallback_model_cfg.model)
    model_cfg_payload = checkpoint.get("model_cfg", None)
    model_cfg = ModelConfig.from_mapping(model_cfg_payload) if model_cfg_payload else fallback_model_cfg
    model = build_model(model_name, num_input_graph_nodes, model_cfg)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


def _gold_candidate(graph: Data) -> tuple[int, int, int, int, int, int]:
    y = getattr(graph, "y", None)
    if y is None:
        raise ValueError("Graph missing y target tensor.")
    if y.dim() == 2:
        y = y[0]
    return tuple(int(v) for v in y.tolist())


def _candidate_from_triple(triple: tuple[int, int, int], *, action: str) -> tuple[int, int, int, int, int, int]:
    if action == "add":
        return (triple[0], triple[1], triple[2], 0, 0, 0)
    return (0, 0, 0, triple[0], triple[1], triple[2])


def _select_values(
    values: Iterable[int] | None, *, placeholder_ids: set[int], none_class: int, max_values: int
) -> list[int]:
    if not values:
        return []
    unique = []
    seen: set[int] = set()
    for value in values:
        if value in (none_class, None):
            continue
        if value in placeholder_ids:
            continue
        if value in seen:
            continue
        seen.add(int(value))
        unique.append(int(value))
        if len(unique) >= max_values:
            return unique
    if unique:
        return unique
    for value in values:
        if value in (none_class, None):
            continue
        if value in seen:
            continue
        seen.add(int(value))
        unique.append(int(value))
        if len(unique) >= max_values:
            break
    return unique


def _instantiate_patterns(
    patterns: Sequence,
    *,
    placeholder_ids: set[int],
    none_class: int,
    max_values: int,
    max_candidates: int,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for pattern in patterns:
        subjects = _select_values(
            pattern.subjects, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values
        )
        predicates = _select_values(
            pattern.predicates, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values
        )
        objects = _select_values(
            pattern.objects, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values
        )
        if not subjects or not predicates or not objects:
            continue
        for s in subjects:
            for p in predicates:
                for o in objects:
                    candidates.append((s, p, o))
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates


def _topk_triples_from_logits(
    logits: torch.Tensor,
    *,
    slots: tuple[int, int, int],
    topk_triples: int,
    topk_per_slot: int,
) -> list[tuple[int, int, int]]:
    topk_per_slot = max(1, min(topk_per_slot, logits.size(-1)))
    slot_vals = []
    slot_ids = []
    for slot in slots:
        vals, ids = torch.topk(logits[slot], k=topk_per_slot)
        slot_vals.append(vals.cpu())
        slot_ids.append(ids.cpu())
    combos: list[tuple[float, int, int, int]] = []
    for i in range(topk_per_slot):
        for j in range(topk_per_slot):
            for k in range(topk_per_slot):
                score = float(slot_vals[0][i] + slot_vals[1][j] + slot_vals[2][k])
                combos.append((score, int(slot_ids[0][i]), int(slot_ids[1][j]), int(slot_ids[2][k])))
    combos.sort(key=lambda x: x[0], reverse=True)
    return [(s, p, o) for _, s, p, o in combos[:topk_triples]]


def _build_candidates(
    *,
    graph: Data,
    context: ViolationContext,
    heuristics: ConstraintRepairHeuristics,
    proposal_logits: torch.Tensor,
    cfg: RerankerTrainingConfig,
    placeholder_ids: set[int],
    num_target_ids: int,
    include_gold: bool | None = None,
    precomputed_add_topk: Sequence[tuple[int, int, int]] | None = None,
    precomputed_del_topk: Sequence[tuple[int, int, int]] | None = None,
) -> tuple[list[tuple[int, int, int, int, int, int]], int | None]:
    candidate_cfg = CandidateConfig(
        topk_candidates=cfg.topk_candidates,
        topk_per_slot=cfg.topk_per_slot,
        heuristic_max_candidates=cfg.heuristic_max_candidates,
        heuristic_max_values=cfg.heuristic_max_values,
        include_gold=cfg.include_gold if include_gold is None else include_gold,
        max_candidates_total=cfg.max_candidates_total,
    )
    return build_candidates(
        graph=graph,
        context=context,
        heuristics=heuristics,
        proposal_logits=proposal_logits,
        cfg=candidate_cfg,
        placeholder_ids=placeholder_ids,
        num_target_ids=num_target_ids,
        precomputed_add_topk=precomputed_add_topk,
        precomputed_del_topk=precomputed_del_topk,
    )


def _evaluate_candidate_set(
    evaluator: CandidateConstraintEvaluator,
    row: Any,
    *,
    candidates: Sequence[Sequence[int]],
    primary_index: int,
) -> list:
    metrics: list = []
    for cand in candidates:
        metrics.append(evaluator.evaluate_full(row, candidate_slots=cand, primary_factor_index=primary_index))
    return metrics


def _aggregate_epoch_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    totals: dict[str, float] = {}
    for record in records:
        for key, value in record.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: total / len(records) for key, total in totals.items()}


def _manifest_graph_count(graph_path: Path) -> int | None:
    manifest_path = graph_path.with_suffix(graph_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        logger.exception("Failed to read graph manifest at %s", manifest_path)
        return None
    graph_count = payload.get("graph_count")
    return int(graph_count) if isinstance(graph_count, int) else None


def _dataset_graph_count(dataset: Sequence[Data] | GraphStreamDataset, graph_path: Path) -> int | None:
    if isinstance(dataset, list):
        return len(dataset)
    return _manifest_graph_count(graph_path)


def _resolve_constraint_registry_path(dataset_variant: str) -> Path:
    names = [dataset_variant, base_dataset_name(dataset_variant)]
    base_name = base_dataset_name(dataset_variant)
    if "_strat" in base_name:
        names.append(base_name.split("_strat", 1)[0])
    candidates = [
        Path("data") / "interim" / f"constraint_registry_{name}.parquet"
        for name in dict.fromkeys(names)
    ]
    for path in candidates:
        if path.exists():
            if path.name != f"constraint_registry_{dataset_variant}.parquet":
                logger.info("Using constraint registry %s for dataset variant %s", path, dataset_variant)
            return path
    raise FileNotFoundError(
        "Constraint registry not found for dataset variant "
        f"{dataset_variant}; checked {', '.join(str(path) for path in candidates)}"
    )


def _candidate_to_slots(candidate: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    if len(candidate) != NUM_SLOTS:
        raise ValueError("Candidate must have 6 slots.")
    return tuple(int(v) for v in candidate)


def _candidate_slots_to_actions(candidate: Sequence[int]) -> dict[str, list[int]]:
    add = list(candidate[:3])
    delete = list(candidate[3:6])
    return {"add": add, "del": delete}


@torch.no_grad()
def _predict_reranker_edits(
    *,
    model: CandidateReranker,
    proposal_model: nn.Module,
    data: Sequence[Data] | GraphStreamDataset,
    contexts: Sequence[ViolationContext],
    rows: Sequence[Any],
    heuristics: ConstraintRepairHeuristics,
    evaluator: CandidateConstraintEvaluator,
    device: torch.device,
    cfg: RerankerTrainingConfig,
    batch_size: int | None = None,
) -> list[dict[str, list[int]]]:
    model.eval()
    proposal_model.eval()
    predictions: list[dict[str, list[int]]] = []

    loader = DataLoader(
        data,
        batch_size=batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    for batch in progress_bar(loader, desc="predict"):
        batch = batch.to(device)
        proposal_outputs = proposal_model(batch)
        proposal_logits = proposal_outputs["edit_logits"].detach()
        proposal_logits_cpu = proposal_logits.cpu()
        batch_add_topk, batch_del_topk = batch_topk_candidate_triples(
            proposal_logits,
            topk_triples=cfg.topk_candidates,
            topk_per_slot=cfg.topk_per_slot,
        )
        graph_emb = model.encode_graphs(batch)
        graphs = batch.to_data_list()

        candidate_groups: list[list[tuple[int, int, int, int, int, int]]] = []
        packed_candidates: list[tuple[int, int, int, int, int, int]] = []
        packed_graph_index: list[int] = []
        for idx, graph in enumerate(graphs):
            context_index = int(getattr(graph, "context_index"))
            context = contexts[context_index]
            row = rows[context_index]
            candidates, _ = _build_candidates(
                graph=graph,
                context=context,
                heuristics=heuristics,
                proposal_logits=proposal_logits_cpu[idx],
                cfg=cfg,
                placeholder_ids=set(heuristics.placeholder_ids.values()),
                num_target_ids=model.num_target_ids,
                include_gold=cfg.prediction_include_gold,
                precomputed_add_topk=batch_add_topk[idx],
                precomputed_del_topk=batch_del_topk[idx],
            )
            if not candidates:
                raise RuntimeError("Reranker prediction produced an empty candidate set.")
            candidate_groups.append(candidates)
            packed_candidates.extend(candidates)
            packed_graph_index.extend([idx] * len(candidates))

        candidate_tensor = torch.tensor(packed_candidates, dtype=torch.long, device=device)
        graph_index_tensor = torch.tensor(packed_graph_index, dtype=torch.long, device=device)
        packed_scores = model.score_candidates_packed(
            graph_emb, candidate_tensor, graph_index_tensor
        ).detach().cpu()
        offset = 0
        for candidates in candidate_groups:
            next_offset = offset + len(candidates)
            best_idx = int(torch.argmax(packed_scores[offset:next_offset]).item())
            best_candidate = _candidate_to_slots(candidates[best_idx])
            predictions.append(_candidate_slots_to_actions(best_candidate))
            _ = evaluator  # keep evaluator in signature for future diagnostics
            offset = next_offset

    return predictions


def _run_epoch(
    *,
    model: CandidateReranker,
    proposal_model: nn.Module,
    loader: DataLoader,
    contexts: Sequence[ViolationContext],
    rows: Sequence[Any],
    heuristics: ConstraintRepairHeuristics,
    evaluator: CandidateConstraintEvaluator,
    device: torch.device,
    cfg: RerankerTrainingConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    proposal_model.eval()

    epoch_loss = 0.0
    epoch_records: list[dict[str, float]] = []
    batch_steps = 0

    for batch in progress_bar(loader, desc="train" if is_train else "val"):
        batch = batch.to(device)
        with torch.no_grad():
            proposal_outputs = proposal_model(batch)
            proposal_logits = proposal_outputs["edit_logits"].detach()

        graph_emb = model.encode_graphs(batch)
        graphs = batch.to_data_list()

        total_loss = 0.0
        batch_count = 0
        for idx, graph in enumerate(graphs):
            context_index = int(getattr(graph, "context_index"))
            context = contexts[context_index]
            row = rows[context_index]
            candidates, gold_index = _build_candidates(
                graph=graph,
                context=context,
                heuristics=heuristics,
                proposal_logits=proposal_logits[idx],
                cfg=cfg,
                placeholder_ids=set(heuristics.placeholder_ids.values()),
                num_target_ids=model.num_target_ids,
            )
            if cfg.objective != "global_fix" and gold_index is None:
                raise RuntimeError(
                    "Reranker training with objective='main' requires include_gold=True."
                )
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(graph_emb[idx], candidate_tensor)
            log_probs = F.log_softmax(scores, dim=0)
            probs = log_probs.exp()

            metrics_summary = evaluator.evaluate_candidate_metrics(
                row,
                candidates=candidates,
                primary_factor_index=int(getattr(graph, "primary_factor_index", 0)),
            )

            primary_oracle = max(m.primary_satisfied for m in metrics_summary)
            global_oracle = max(m.global_satisfied_fraction for m in metrics_summary)

            best_idx = int(torch.argmax(scores).item())
            primary_best = metrics_summary[best_idx].primary_satisfied
            global_best = metrics_summary[best_idx].global_satisfied_fraction
            regress_best = metrics_summary[best_idx].secondary_regressions
            focus_deleted_best = metrics_summary[best_idx].focus_deleted
            deletes_focus_action_best = metrics_summary[best_idx].candidate_deletes_focus

            record = {
                "primary_oracle": primary_oracle,
                "primary_chosen": primary_best,
                "global_oracle": global_oracle,
                "global_chosen": global_best,
                "regressions_chosen": regress_best,
                "focus_deleted_chosen": focus_deleted_best,
                "deletes_focus_action_chosen": deletes_focus_action_best,
                "candidate_count": float(len(candidates)),
            }
            epoch_records.append(record)

            if cfg.objective == "global_fix":
                loss = _global_fix_loss(
                    probs,
                    metrics_summary,
                    focus_deletion_weight=cfg.focus_deletion_weight,
                    device=device,
                )
            else:
                ce_loss = -log_probs[gold_index]
                regression_tensor = torch.tensor(
                    [m.srr for m in metrics_summary], dtype=torch.float32, device=device
                )
                gold_regression = regression_tensor[gold_index]
                reg_penalty = torch.sum(
                    probs * torch.clamp(regression_tensor - gold_regression, min=0.0)
                )
                loss = ce_loss + cfg.regression_weight * reg_penalty

            total_loss += loss
            batch_count += 1

        if batch_count == 0:
            continue

        batch_loss = total_loss / batch_count
        if not torch.isfinite(batch_loss):
            raise FloatingPointError("Non-finite reranker batch loss encountered")
        if is_train:
            optimizer.zero_grad()
            batch_loss.backward()
            if cfg.grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError("Non-finite reranker gradient norm encountered")
            optimizer.step()

        epoch_loss += float(batch_loss.item())
        batch_steps += 1

    avg_loss = epoch_loss / max(batch_steps, 1)
    metrics = _aggregate_epoch_metrics(epoch_records)
    return avg_loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidate-based reranker.")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="Path to reranker experiment config JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random-seed override; must match training_config.seed when supplied.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and only generate reranker_predictions.json from the latest checkpoint.",
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=None,
        help="Optional test-prediction batch size override; does not change the training config.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    with args.experiment_config.open("r", encoding="utf-8") as fh:
        experiment_payload = json.load(fh)
    model_cfg, reranker_cfg, training_cfg, proposal_cfg = _load_experiment_config(args.experiment_config)
    seed_is_explicit = "seed" in experiment_payload.get("training_config", {})
    if seed_is_explicit and args.seed is not None and int(args.seed) != training_cfg.seed:
        raise ValueError(
            f"--seed={args.seed} differs from training_config.seed={training_cfg.seed}"
        )
    if not seed_is_explicit and args.seed is not None:
        training_cfg.seed = int(args.seed)
    set_seed(training_cfg.seed)

    dataset_variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
    processed_root = Path("data") / "processed" / dataset_variant
    interim_path = Path("data") / "interim" / dataset_variant

    train_path = processed_root / graph_dataset_filename(
        "train",
        model_cfg.encoding,
        constraint_representation=model_cfg.constraint_representation,
    )
    val_path = processed_root / graph_dataset_filename(
        "val",
        model_cfg.encoding,
        constraint_representation=model_cfg.constraint_representation,
    )

    train_data = load_graph_dataset(train_path)
    val_data = load_graph_dataset(val_path)
    train_graph_count = _dataset_graph_count(train_data, train_path)
    val_graph_count = _dataset_graph_count(val_data, val_path)

    encoder = _load_encoder(interim_path)
    placeholder_ids = placeholder_ids_from_encoder(encoder)
    heuristics = ConstraintRepairHeuristics(
        encoder=encoder,
        placeholder_ids=placeholder_ids,
        none_class=NONE_CLASS_INDEX,
    )

    train_contexts = load_violation_contexts(interim_path, "train", none_class=NONE_CLASS_INDEX)
    val_contexts = load_violation_contexts(interim_path, "val", none_class=NONE_CLASS_INDEX)
    if train_graph_count is not None and len(train_contexts) != train_graph_count:
        raise RuntimeError("Mismatch between train graph dataset size and violation contexts.")
    if val_graph_count is not None and len(val_contexts) != val_graph_count:
        raise RuntimeError("Mismatch between validation graph dataset size and violation contexts.")

    if isinstance(train_data, list):
        for idx, graph in enumerate(train_data):
            setattr(graph, "context_index", idx)
    else:
        logger.info("Using streamed training graphs for reranker training.")
    if isinstance(val_data, list):
        for idx, graph in enumerate(val_data):
            setattr(graph, "context_index", idx)
    else:
        logger.info("Using streamed validation graphs for reranker training.")

    train_rows = _load_parquet_rows(interim_path, "train")
    val_rows = _load_parquet_rows(interim_path, "val")
    if train_graph_count is not None and len(train_rows) != train_graph_count:
        raise RuntimeError("Mismatch between parquet rows and train graph dataset size.")
    if val_graph_count is not None and len(val_rows) != val_graph_count:
        raise RuntimeError("Mismatch between parquet rows and validation graph dataset size.")

    registry_path = _resolve_constraint_registry_path(model_cfg.dataset_variant)

    use_encoded_ids = True
    try:
        sample_id = getattr(train_rows[0], "constraint_id", None)
        if sample_id is None or isinstance(sample_id, str):
            use_encoded_ids = False
    except Exception:
        use_encoded_ids = True

    evaluator = CandidateConstraintEvaluator(
        str(registry_path),
        encoder=encoder if use_encoded_ids else None,
        assume_complete=training_cfg.assume_complete_entity_facts,
        constraint_scope=training_cfg.constraint_scope,
        use_encoded_ids=use_encoded_ids,
    )

    use_node_embeddings, feature_dim, _, role_spec = infer_node_feature_spec(train_data)

    vocab_from_filtered = len(encoder._global_id_to_unfiltered_global_id)
    if vocab_from_filtered > 0:
        num_input_graph_nodes = vocab_from_filtered + 1
    else:
        num_input_graph_nodes = len(encoder._encoding) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proposal_checkpoint_path = _resolve_proposal_checkpoint(
        proposal_cfg,
        model_cfg=model_cfg,
    )
    proposal_model = _load_proposal_model(
        proposal_cfg,
        num_input_graph_nodes=num_input_graph_nodes,
        device=device,
        fallback_model_cfg=model_cfg,
    )

    if str(model_cfg.encoding).lower() != "node_id" and training_cfg.batch_size > 8:
        logger.info(
            "Reducing reranker batch size from %s to 8 for encoding=%s to avoid OOM.",
            training_cfg.batch_size,
            model_cfg.encoding,
        )
        training_cfg.batch_size = 8

    validation_subset_size = training_cfg.validation_subset_size
    val_loader_data: list[Data] | IterableDataset
    val_num_workers = training_cfg.num_workers
    if validation_subset_size is None:
        val_loader_data = val_data
    elif isinstance(val_data, list):
        val_loader_data = val_data[:validation_subset_size]
        logger.info(
            "Reranker validation subset enabled | using first %s/%s in-memory graphs per epoch",
            len(val_loader_data),
            len(val_data),
        )
    else:
        val_loader_data = ValidationSubsetStream(val_data, validation_subset_size)
        if val_num_workers > 0:
            logger.info(
                "Reranker validation subset uses num_workers=0 so streamed validation emits one global prefix only."
            )
            val_num_workers = 0
        logger.info(
            "Reranker validation subset enabled | using first %s streamed graphs per epoch",
            validation_subset_size,
        )

    graph_model_cfg = model_cfg
    if str(model_cfg.model).upper() == "RERANKER":
        model_payload = model_cfg.to_dict()
        model_payload["model"] = proposal_cfg.get("model", "GIN")
        model_payload["use_node_embeddings"] = bool(use_node_embeddings)
        if not use_node_embeddings:
            model_payload["num_embedding_size"] = int(feature_dim)
        model_payload["use_role_embeddings"] = bool(role_spec.enabled)
        model_payload["num_role_types"] = int(role_spec.num_types)
        graph_model_cfg = ModelConfig.from_mapping(model_payload)
    model = build_reranker(
        num_input_graph_nodes=num_input_graph_nodes,
        model_cfg=graph_model_cfg,
        reranker_cfg=reranker_cfg,
    )
    model.to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=training_cfg.batch_size,
        shuffle=(not isinstance(train_data, IterableDataset)),
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_loader_data,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=training_cfg.pin_memory,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=training_cfg.scheduler_factor,
        patience=training_cfg.scheduler_patience,
    )

    run_dir = ensure_run_dir_for_config(args.experiment_config)
    _write_effective_experiment_config(
        args.experiment_config,
        experiment_payload,
        model_cfg=model_cfg,
        reranker_cfg=reranker_cfg,
        training_cfg=training_cfg,
        proposal_cfg=proposal_cfg,
    )
    logger.info("Updated resolved experiment config at %s", args.experiment_config)
    training_provenance = {
        "schema_version": 2,
        "seed": training_cfg.seed,
        "config": _file_identity(args.experiment_config),
        "proposal_checkpoint": _file_identity(proposal_checkpoint_path),
        "train_graph": str(train_path.resolve()),
        "validation_graph": str(val_path.resolve()),
    }
    if args.predict_only:
        checkpoint_path = get_checkpoint_path(run_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} for --predict-only.")

    best_val = float("inf")
    best_epoch = -1
    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
    }

    if not args.predict_only:
        for epoch in range(training_cfg.num_epochs):
            train_loss, train_metrics = _run_epoch(
                model=model,
                proposal_model=proposal_model,
                loader=train_loader,
                contexts=train_contexts,
                rows=train_rows,
                heuristics=heuristics,
                evaluator=evaluator,
                device=device,
                cfg=training_cfg,
                optimizer=optimizer,
            )
            val_loss, val_metrics = _run_epoch(
                model=model,
                proposal_model=proposal_model,
                loader=val_loader,
                contexts=val_contexts,
                rows=val_rows,
                heuristics=heuristics,
                evaluator=evaluator,
                device=device,
                cfg=training_cfg,
            )

            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_metrics"].append(train_metrics)
            history["val_metrics"].append(val_metrics)

            logger.info(
                "Epoch %s | train_loss=%.4f val_loss=%.4f primary=%.3f/%.3f global=%.3f/%.3f",
                epoch + 1,
                train_loss,
                val_loss,
                train_metrics.get("primary_chosen", 0.0),
                train_metrics.get("primary_oracle", 0.0),
                val_metrics.get("global_chosen", 0.0),
                val_metrics.get("global_oracle", 0.0),
            )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                model_path = get_checkpoint_path(run_dir)
                checkpoint_payload = {
                    "model_state": model.state_dict(),
                    "num_graph_nodes": num_input_graph_nodes,
                    "model_name": "RERANKER",
                    "model_cfg": model_cfg.to_dict(),
                    "training_cfg": training_cfg.to_dict(),
                    "reranker_cfg": reranker_cfg.to_dict(),
                    "best_epoch": epoch + 1,
                    "checkpoint_role": "best",
                    "training_provenance": training_provenance,
                }
                if training_cfg.save_last_checkpoint:
                    atomic_torch_save(checkpoint_payload, model_path)
                else:
                    torch.save(checkpoint_payload, model_path)

            if training_cfg.save_last_checkpoint:
                last_payload = {
                    "model_state": model.state_dict(),
                    "num_graph_nodes": num_input_graph_nodes,
                    "model_name": "RERANKER",
                    "model_cfg": model_cfg.to_dict(),
                    "training_cfg": training_cfg.to_dict(),
                    "reranker_cfg": reranker_cfg.to_dict(),
                    "best_epoch": best_epoch + 1,
                    "completed_epoch": epoch + 1,
                    "checkpoint_role": "last",
                    "training_provenance": training_provenance,
                }
                atomic_torch_save(last_payload, get_last_checkpoint_path(run_dir))

            if epoch - best_epoch >= training_cfg.early_stopping_rounds:
                logger.info("Early stopping at epoch %s", epoch + 1)
                break

    if not args.predict_only:
        history["best_epoch"] = best_epoch + 1
        history["completed_epochs"] = len(history["train_loss"])
        history["training_provenance"] = training_provenance
        history_file = history_path(run_dir)
        if training_cfg.save_last_checkpoint:
            atomic_write_json(history_file, history)
        else:
            with history_file.open("w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)

    # Generate reranker predictions for evaluation.
    test_path = processed_root / graph_dataset_filename(
        "test",
        model_cfg.encoding,
        constraint_representation=model_cfg.constraint_representation,
    )
    if discover_graph_artifacts(test_path):
        test_data = load_graph_dataset(test_path)
        test_graph_count = _dataset_graph_count(test_data, test_path)
        test_contexts = load_violation_contexts(interim_path, "test", none_class=NONE_CLASS_INDEX)
        if test_graph_count is None or len(test_contexts) == test_graph_count:
            if isinstance(test_data, list):
                for idx, graph in enumerate(test_data):
                    setattr(graph, "context_index", idx)
            test_rows = _load_parquet_rows(interim_path, "test")
            if test_graph_count is None or len(test_rows) == test_graph_count:
                checkpoint = torch.load(get_checkpoint_path(run_dir), map_location=device)
                state_dict = checkpoint.get("model_state")
                if state_dict is not None:
                    try:
                        model.load_state_dict(state_dict)
                    except RuntimeError as exc:
                        if args.predict_only:
                            graph_model_cfg = _derive_graph_model_cfg_from_state_dict(
                                state_dict,
                                model_cfg=model_cfg,
                                proposal_cfg=proposal_cfg,
                            )
                            model = build_reranker(
                                num_input_graph_nodes=num_input_graph_nodes,
                                model_cfg=graph_model_cfg,
                                reranker_cfg=reranker_cfg,
                            )
                            model.load_state_dict(state_dict, strict=False)
                        else:
                            raise
                    model.to(device)
                predictions = _predict_reranker_edits(
                    model=model,
                    proposal_model=proposal_model,
                    data=test_data,
                    contexts=test_contexts,
                    rows=test_rows,
                    heuristics=heuristics,
                    evaluator=evaluator,
                    device=device,
                    cfg=training_cfg,
                    batch_size=args.prediction_batch_size,
                )
                pred_path = run_dir / "reranker_predictions.json"
                if training_cfg.save_last_checkpoint:
                    atomic_write_json(pred_path, predictions)
                else:
                    with pred_path.open("w", encoding="utf-8") as fh:
                        json.dump(predictions, fh, indent=2)
                logger.info("Saved reranker predictions to %s", pred_path)
            else:
                logger.warning("Skipping reranker predictions: test rows mismatch.")
        else:
            logger.warning("Skipping reranker predictions: test contexts mismatch.")
    else:
        logger.warning("Skipping reranker predictions: test split not found at %s", test_path)

    logger.info("Training complete. Best val loss=%.4f", best_val)


if __name__ == "__main__":
    main()
