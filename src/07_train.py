#!/usr/bin/env python3

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, cast

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from modules.candidates import CandidateConfig, build_candidates, score_candidates_from_logits
from modules.config import ModelConfig, TrainingConfig
from modules.data_encoders import (
    GlobalIntEncoder,
    GraphStreamDataset,
    base_dataset_name,
    dataset_variant_name,
    graph_dataset_filename,
    infer_node_feature_spec,
)
from modules.model_store import (
    ensure_run_dir_for_config,
    get_checkpoint_path,
    history_path,
)
from modules.models import BaseGraphModel, build_model
from modules.repair_eval import ConstraintRepairHeuristics, ViolationContext, load_violation_contexts
from modules.reranker_eval import CandidateConstraintEvaluator
from modules.policy import POLICY_NAMES, derive_policy_label
from modules.training_utils import (
    ConstraintMetricsAccumulator,
    DynamicConstraintWeighter,
    FixProbabilityScheduler,
    compute_fix_probabilities,
    extract_constraint_types,
    load_graph_dataset,
    load_precomputed_target_vocabs,
    log_cuda_memory,
    placeholder_ids_from_encoder,
    plot_training_history,
    progress_bar,
    set_seed,
    update_per_constraint_history,
)

NUM_SLOTS = 6
NONE_CLASS_INDEX = 0

logger = logging.getLogger(__name__)


ENTITY_SLOT_INDICES: tuple[int, ...] = (0, 2, 3, 5)
PREDICATE_SLOT_INDICES: tuple[int, ...] = (1, 4)


class ValidationSubsetStream(IterableDataset):
    """Yield the first ``limit`` graphs from a streamed validation dataset."""

    def __init__(self, dataset: IterableDataset, limit: int) -> None:
        if limit <= 0:
            raise ValueError("Validation subset limit must be positive.")
        self.dataset = dataset
        self.limit = int(limit)

    def __iter__(self) -> Iterator[Data]:
        for idx, graph in enumerate(self.dataset):
            if idx >= self.limit:
                break
            yield graph

    def __len__(self) -> int:
        return self.limit


def _max_abs_value(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float(torch.nan_to_num(tensor.detach(), nan=0.0, posinf=float("inf"), neginf=float("-inf")).abs().max().item())


def _scalar_float(value: float | torch.Tensor) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _assert_finite_scalar(name: str, value: float | torch.Tensor, *, epoch: int, batch_idx: int | None = None) -> float:
    numeric_value = _scalar_float(value)
    if not math.isfinite(numeric_value):
        location = f"epoch={epoch}"
        if batch_idx is not None:
            location += f" batch={batch_idx}"
        raise FloatingPointError(f"Non-finite {name} detected at {location}: {numeric_value}")
    return numeric_value


def _grad_norm(parameters: Any, norm_type: float = 2.0) -> torch.Tensor:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(g, ord=norm_type).to(device) for g in grads])
    return torch.linalg.vector_norm(norms, ord=norm_type)


def _parameter_norm_stats(model: torch.nn.Module) -> tuple[float, float]:
    total_sq = 0.0
    max_abs = 0.0
    for parameter in model.parameters():
        if parameter.numel() == 0:
            continue
        detached = parameter.detach()
        param_norm = float(torch.linalg.vector_norm(detached.float(), ord=2).item())
        param_abs = _max_abs_value(detached)
        total_sq += param_norm * param_norm
        max_abs = max(max_abs, param_abs)
    return math.sqrt(total_sq), max_abs


# Implementation checklist (factor supervision):
# - In train() and validation loops, insert factor-level losses after graph_loss is computed
#   and before dynamic reweighting (and before weighted_loss is averaged).
# - Reuse per-graph loss vectors (graph_loss) for weighting; add factor losses as per-graph
#   scalars to preserve the same averaging and dynamic constraint weighting path.
# - Use factor label tensors on Data (factor_* fields) aligned to factor_constraint_ids;
#   primary_factor_index identifies the violated constraint within each graph.
# - Mask factor losses with factor_checkable_* to avoid penalizing uncheckable factors.
# - Keep fix-probability loss term order: slot loss -> optional fix loss -> factor losses -> reweight.


def derive_target_class_ids(
    *datasets: list[Data] | GraphStreamDataset | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Aggregate entity/predicate target vocabularies observed in the provided datasets."""

    entity_ids: set[int] = {0}
    predicate_ids: set[int] = {0}

    def update_from_tensor(targets: torch.Tensor) -> None:
        if targets.dim() == 1:
            tensor = targets.view(1, -1)
        elif targets.dim() == 2:
            tensor = targets
        else:
            tensor = targets.view(-1, targets.size(-1))

        entity_values = tensor[:, ENTITY_SLOT_INDICES].reshape(-1)
        predicate_values = tensor[:, PREDICATE_SLOT_INDICES].reshape(-1)

        entity_ids.update(int(v) for v in entity_values.tolist())
        predicate_ids.update(int(v) for v in predicate_values.tolist())

    # Accumulate from all datasets
    for dataset in datasets:
        if dataset is None:
            continue
        iterator = dataset if isinstance(dataset, list) else iter(dataset)
        for graph in iterator:
            targets = getattr(graph, "y", None)
            if targets is None:
                continue
            if targets.dtype not in (torch.long, torch.int64):
                targets = targets.to(dtype=torch.long)
            update_from_tensor(targets)

    entity_sorted = tuple(sorted(entity_ids))
    predicate_sorted = tuple(sorted(predicate_ids))

    return entity_sorted, predicate_sorted


def derive_factor_type_count(
    *datasets: list[Data] | GraphStreamDataset | None,
) -> int:
    """Return max factor type id + 1 across datasets (0 if absent)."""
    max_type = -1

    def update_from_tensor(values: torch.Tensor) -> None:
        nonlocal max_type
        if values.numel() == 0:
            return
        if values.dtype not in (torch.long, torch.int64, torch.int32):
            values = values.to(dtype=torch.long)
        current_max = int(values.max().item())
        if current_max > max_type:
            max_type = current_max

    for dataset in datasets:
        if dataset is None:
            continue
        iterator = dataset if isinstance(dataset, list) else iter(dataset)
        for graph in iterator:
            factor_types = getattr(graph, "factor_types", None)
            if factor_types is None:
                continue
            update_from_tensor(torch.as_tensor(factor_types))

    return max_type + 1 if max_type >= 0 else 0


def _registry_factor_type_count(dataset_variant: str) -> int:
    import pandas as pd

    candidates = _constraint_registry_candidate_paths(dataset_variant)

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            logger.exception("Failed to read constraint registry at %s", path)
            continue

        if "registry_json" in df.columns and len(df) > 0:
            try:
                payload = json.loads(df.iloc[0]["registry_json"])
            except Exception:
                logger.exception("Failed to parse registry_json from %s", path)
                continue
            indices = [
                int(item["constraint_type_index"])
                for item in payload.values()
                if isinstance(item, dict) and item.get("constraint_type_index") is not None
            ]
            if indices:
                return max(indices) + 1

        for column in ("constraint_type_index", "constraint_type_id"):
            if column not in df.columns:
                continue
            series = df[column].dropna()
            if len(series) == 0:
                continue
            return int(series.max()) + 1

    return 0


def _constraint_registry_candidate_paths(dataset_variant: str) -> list[Path]:
    names = [dataset_variant, base_dataset_name(dataset_variant)]
    base_name = base_dataset_name(dataset_variant)
    if "_strat" in base_name:
        names.append(base_name.split("_strat", 1)[0])
    return [
        Path("data") / "interim" / f"constraint_registry_{name}.parquet"
        for name in dict.fromkeys(names)
    ]


def _resolve_constraint_registry_path(dataset_variant: str) -> Path:
    candidates = _constraint_registry_candidate_paths(dataset_variant)
    for path in candidates:
        if path.exists():
            if path.name != f"constraint_registry_{dataset_variant}.parquet":
                logger.info("Using constraint registry %s for dataset variant %s", path, dataset_variant)
            return path
    raise FileNotFoundError(
        "Constraint registry not found for dataset variant "
        f"{dataset_variant}; checked {', '.join(str(path) for path in candidates)}"
    )


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


def _assert_factor_labels(graph: Data, graph_index: int | None = None) -> None:
    prefix = f"graph[{graph_index}] " if graph_index is not None else ""

    factor_ids = getattr(graph, "factor_constraint_ids", None)
    assert factor_ids is not None, f"{prefix}missing factor_constraint_ids"
    factor_ids_tensor = torch.as_tensor(factor_ids).view(-1)
    assert factor_ids_tensor.numel() > 0, f"{prefix}factor_constraint_ids must be non-empty"
    expected_len = int(factor_ids_tensor.numel())

    primary_idx = getattr(graph, "primary_factor_index", None)
    assert primary_idx is not None, f"{prefix}missing primary_factor_index"
    if torch.is_tensor(primary_idx):
        if primary_idx.numel() != 1:
            raise AssertionError(f"{prefix}primary_factor_index must be scalar; got {tuple(primary_idx.shape)}")
        primary_idx = int(primary_idx.item())
    else:
        primary_idx = int(primary_idx)
    assert 0 <= primary_idx < expected_len, (
        f"{prefix}primary_factor_index {primary_idx} out of range for {expected_len} factors"
    )

    def _check_vector(name: str, *, expect_bool: bool = False, expect_int: bool = False) -> torch.Tensor:
        value = getattr(graph, name, None)
        assert value is not None, f"{prefix}missing {name}"
        tensor = torch.as_tensor(value).view(-1)
        assert tensor.numel() == expected_len, (
            f"{prefix}{name} length {tensor.numel()} does not match factor_constraint_ids {expected_len}"
        )
        if expect_bool:
            assert tensor.dtype == torch.bool, f"{prefix}{name} must be bool, got {tensor.dtype}"
        if expect_int:
            assert not torch.is_floating_point(tensor), f"{prefix}{name} must be integer dtype"
        return tensor

    _check_vector("factor_checkable_pre", expect_bool=True)
    _check_vector("factor_checkable_post_gold", expect_bool=True)
    _check_vector("factor_satisfied_pre", expect_int=True)
    _check_vector("factor_satisfied_post_gold", expect_int=True)
    _check_vector("factor_types", expect_int=True)


def _assert_factor_labels_batch(data: Data) -> None:
    graphs = data.to_data_list() if hasattr(data, "to_data_list") else [data]
    for idx, graph in enumerate(graphs):
        _assert_factor_labels(graph, idx)


def _assert_factor_type_ids_supported(data: Data, num_factor_types: int) -> None:
    if num_factor_types <= 0:
        return
    graphs = data.to_data_list() if hasattr(data, "to_data_list") else [data]
    for idx, graph in enumerate(graphs):
        factor_types = getattr(graph, "factor_types", None)
        if factor_types is None:
            continue
        factor_types_tensor = torch.as_tensor(factor_types).view(-1)
        if factor_types_tensor.numel() == 0:
            continue
        invalid_mask = (factor_types_tensor < 0) | (factor_types_tensor >= num_factor_types)
        if not torch.any(invalid_mask):
            continue
        invalid_values = sorted({int(v) for v in factor_types_tensor[invalid_mask].tolist()})
        preview = ", ".join(str(v) for v in invalid_values[:8])
        if len(invalid_values) > 8:
            preview += ", ..."
        raise AssertionError(
            f"graph[{idx}] contains factor_types outside [0, {num_factor_types}) "
            f"(invalid values: {preview}). Rebuild the constraint registry, labeled parquet, and graphs "
            "so factor_types use dense compact constraint_type_item indices."
        )


def _assert_factor_logit_alignment(
    data: Data,
    factor_logits: torch.Tensor,
    factor_graph_index: torch.Tensor,
) -> None:
    graphs = data.to_data_list() if hasattr(data, "to_data_list") else [data]
    offset = 0
    for graph_idx, graph in enumerate(graphs):
        factor_ids = getattr(graph, "factor_constraint_ids", None)
        if factor_ids is None:
            continue
        factor_ids = torch.as_tensor(factor_ids).view(-1)
        count = int(factor_ids.numel())
        if count == 0:
            continue
        end = offset + count
        assert end <= factor_logits.numel(), (
            f"factor_logits_pre length {factor_logits.numel()} too small for graph {graph_idx}"
        )
        slice_graph = factor_graph_index[offset:end]
        assert torch.all(slice_graph == graph_idx), (
            f"factor logits order mismatch for graph {graph_idx} (expected index {graph_idx})"
        )
        primary_idx = getattr(graph, "primary_factor_index", None)
        assert primary_idx is not None, f"missing primary_factor_index for graph {graph_idx}"
        if torch.is_tensor(primary_idx):
            primary_idx = int(primary_idx.item())
        else:
            primary_idx = int(primary_idx)
        assert 0 <= primary_idx < count, (
            f"primary_factor_index {primary_idx} out of range for graph {graph_idx} with {count} factors"
        )
        primary_logit = factor_logits[offset + primary_idx]
        assert torch.isfinite(primary_logit).item(), "primary factor logit is not finite"
        offset = end
    assert offset == factor_logits.numel(), "factor_logits_pre length does not match factor label total"


def _log_factor_debug(
    data: Data,
    factor_logits: torch.Tensor,
    factor_graph_index: torch.Tensor,
    max_per_graph: int = 3,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if factor_logits.numel() == 0:
        return
    graphs = data.to_data_list() if hasattr(data, "to_data_list") else [data]
    total_graphs = len(graphs)
    for graph_idx in range(total_graphs):
        mask = factor_graph_index == graph_idx
        if not mask.any():
            continue
        local_logits = factor_logits[mask]
        probs = torch.sigmoid(local_logits)
        k = min(max_per_graph, local_logits.numel())
        if k == 0:
            continue
        values, indices = torch.topk(1.0 - probs, k=k)
        factor_ids = getattr(graphs[graph_idx], "factor_constraint_ids", None)
        if factor_ids is None:
            continue
        factor_ids = torch.as_tensor(factor_ids).view(-1)
        primary_idx_raw = getattr(graphs[graph_idx], "primary_factor_index", None)
        primary_idx = int(primary_idx_raw) if primary_idx_raw is not None else None
        entries = []
        for rank, (score, local_idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            global_factor_idx = int(local_idx)
            if global_factor_idx >= factor_ids.numel():
                continue
            cid = int(factor_ids[global_factor_idx].item())
            tag = "PRIMARY" if primary_idx is not None and global_factor_idx == primary_idx else ""
            entries.append(f"{rank}:{cid}:{score:.3f}{':' + tag if tag else ''}")
        logger.debug(
            "Graph %s factor violations: %s",
            graph_idx,
            ", ".join(entries) if entries else "none",
        )


def _batch_int_attr(
    data: Data,
    attr_name: str,
    batch_size: int,
    *,
    default_value: int = 0,
    default_to_batch_index: bool = False,
    min_value: int | None = None,
) -> list[int]:
    """
    Convert a batched graph attribute into a Python `list[int]` of length `batch_size`.
    """
    raw = getattr(data, attr_name, None)
    if raw is None:
        if default_to_batch_index:
            return list(range(batch_size))
        return [int(default_value)] * batch_size

    if torch.is_tensor(raw):
        raw_tensor = raw.detach().view(-1)
        # PyG increments attributes whose key contains "index" during batching.
        # Recover per-graph scalar values by subtracting node-prefix offsets.
        if "index" in attr_name and raw_tensor.numel() == batch_size:
            ptr = getattr(data, "ptr", None)
            if torch.is_tensor(ptr) and ptr.numel() == batch_size + 1:
                ptr_offsets = ptr.detach().to(device=raw_tensor.device, dtype=raw_tensor.dtype)[:-1]
                raw_tensor = raw_tensor - ptr_offsets
        flat = raw_tensor.cpu().tolist()
    elif isinstance(raw, (list, tuple)):
        flat = list(raw)
    else:
        flat = [raw]

    if len(flat) != batch_size:
        if len(flat) == 1 and batch_size == 1:
            pass
        else:
            raise RuntimeError(
                f"Batched attribute '{attr_name}' length {len(flat)} does not match batch size {batch_size}."
            )

    values: list[int] = []
    for idx, item in enumerate(flat):
        fallback = idx if default_to_batch_index else int(default_value)
        try:
            value = int(item)
        except (TypeError, ValueError):
            value = fallback
        if min_value is not None and value < min_value:
            raise RuntimeError(
                f"Batched attribute '{attr_name}' contains value {value} below minimum {min_value}."
            )
        values.append(value)
    return values


def _compute_batch_slot_topk(
    proposal_logits: torch.Tensor,
    *,
    topk_per_slot: int,
    slot_allowed_ids: tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
    | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Compute per-slot top-k ids/scores for all graphs in one batched pass.
    """
    if proposal_logits.dim() != 3 or proposal_logits.size(1) != NUM_SLOTS:
        raise ValueError(
            f"Expected proposal_logits shape (B,{NUM_SLOTS},V), got {tuple(proposal_logits.shape)}"
        )
    vocab_size = int(proposal_logits.size(-1))
    k_default = max(1, min(int(topk_per_slot), vocab_size))

    slot_vals_cpu: list[torch.Tensor] = []
    slot_ids_cpu: list[torch.Tensor] = []
    for slot in range(NUM_SLOTS):
        allowed = None if slot_allowed_ids is None else slot_allowed_ids[slot]
        if allowed is not None:
            allowed_ids = allowed
            if allowed_ids.device != proposal_logits.device:
                allowed_ids = allowed_ids.to(device=proposal_logits.device)
            if allowed_ids.dtype != torch.long:
                allowed_ids = allowed_ids.to(dtype=torch.long)
            if allowed_ids.numel() >= topk_per_slot and allowed_ids.numel() > 0:
                restricted = proposal_logits[:, slot, :].index_select(1, allowed_ids)
                k = max(1, min(int(topk_per_slot), int(restricted.size(1))))
                vals_local, idx_local = torch.topk(restricted, k=k, dim=1)
                ids = allowed_ids.index_select(0, idx_local.reshape(-1)).view_as(idx_local)
                vals = vals_local
            else:
                vals, ids = torch.topk(proposal_logits[:, slot, :], k=k_default, dim=1)
        else:
            vals, ids = torch.topk(proposal_logits[:, slot, :], k=k_default, dim=1)
        slot_vals_cpu.append(vals.detach().cpu())
        slot_ids_cpu.append(ids.detach().cpu())
    return slot_vals_cpu, slot_ids_cpu


def _topk_triples_from_slot_topk_row(
    slot_vals: list[torch.Tensor],
    slot_ids: list[torch.Tensor],
    *,
    row_index: int,
    slots: tuple[int, int, int],
    topk_triples: int,
) -> list[tuple[int, int, int]]:
    vals0 = slot_vals[slots[0]][row_index]
    vals1 = slot_vals[slots[1]][row_index]
    vals2 = slot_vals[slots[2]][row_index]
    ids0 = slot_ids[slots[0]][row_index]
    ids1 = slot_ids[slots[1]][row_index]
    ids2 = slot_ids[slots[2]][row_index]

    k_combos = min(int(vals0.numel()), int(vals1.numel()), int(vals2.numel()))
    if k_combos <= 0:
        return []
    limit = min(max(int(topk_triples), 0), k_combos * k_combos * k_combos)
    if limit <= 0:
        return []

    vals0 = vals0[:k_combos]
    vals1 = vals1[:k_combos]
    vals2 = vals2[:k_combos]
    ids0 = ids0[:k_combos]
    ids1 = ids1[:k_combos]
    ids2 = ids2[:k_combos]

    combo_scores = (
        vals0.view(k_combos, 1, 1)
        + vals1.view(1, k_combos, 1)
        + vals2.view(1, 1, k_combos)
    ).reshape(-1)

    # Keep legacy tie behavior by using a stable descending sort over row-major (i,j,k) order.
    order = torch.argsort(combo_scores, descending=True, stable=True)[:limit]
    k_sq = k_combos * k_combos
    i_idx = torch.div(order, k_sq, rounding_mode="floor")
    rem = torch.remainder(order, k_sq)
    j_idx = torch.div(rem, k_combos, rounding_mode="floor")
    k_idx = torch.remainder(rem, k_combos)

    top_ids0 = ids0.index_select(0, i_idx).tolist()
    top_ids1 = ids1.index_select(0, j_idx).tolist()
    top_ids2 = ids2.index_select(0, k_idx).tolist()
    return [(int(s), int(p), int(o)) for s, p, o in zip(top_ids0, top_ids1, top_ids2)]


def train(
    model: BaseGraphModel,
    train_data: list[Data] | GraphStreamDataset,
    val_data: list[Data] | GraphStreamDataset,
    num_graph_nodes: int,
    train_cfg: TrainingConfig,
    device: torch.device | str = torch.device("cpu"),
    fix_loss_state: dict[str, object] | None = None,
    chooser_state: dict[str, object] | None = None,
    direct_safety_state: dict[str, object] | None = None,
    policy_state: dict[str, object] | None = None,
    estimated_train_batches: int | None = None,
    estimated_val_batches: int | None = None,
):
    # Normalise the device argument because config may pass strings.
    if isinstance(device, str):
        device = torch.device(device)

    # Optional fix-probability regulariser state (scheduler, heuristics, contexts).
    fix_scheduler: FixProbabilityScheduler | None = None
    fix_heuristics: ConstraintRepairHeuristics | None = None
    train_contexts: list[ViolationContext] | None = None
    val_contexts: list[ViolationContext] | None = None

    if fix_loss_state:
        fix_scheduler = cast(FixProbabilityScheduler | None, fix_loss_state.get("scheduler"))
        fix_heuristics = cast(ConstraintRepairHeuristics | None, fix_loss_state.get("heuristics"))
        train_contexts = cast(list[ViolationContext] | None, fix_loss_state.get("train_contexts"))
        val_contexts = cast(list[ViolationContext] | None, fix_loss_state.get("val_contexts"))

    chooser_cfg = train_cfg.chooser
    chooser_enabled = bool(chooser_cfg.enabled)
    chooser_heuristics = None
    chooser_train_rows = None
    chooser_val_rows = None
    chooser_train_contexts = None
    chooser_val_contexts = None
    chooser_evaluator = None
    chooser_candidate_cfg = None
    chooser_placeholder_ids: set[int] | None = None
    if chooser_enabled and chooser_state:
        chooser_heuristics = cast(ConstraintRepairHeuristics | None, chooser_state.get("heuristics"))
        chooser_train_rows = cast(list | None, chooser_state.get("train_rows"))
        chooser_val_rows = cast(list | None, chooser_state.get("val_rows"))
        chooser_train_contexts = cast(list[ViolationContext] | None, chooser_state.get("train_contexts"))
        chooser_val_contexts = cast(list[ViolationContext] | None, chooser_state.get("val_contexts"))
        chooser_evaluator = cast(CandidateConstraintEvaluator | None, chooser_state.get("evaluator"))
        chooser_candidate_cfg = cast(CandidateConfig | None, chooser_state.get("candidate_cfg"))
        chooser_placeholder_ids = cast(set[int] | None, chooser_state.get("placeholder_ids_set"))

    direct_safety_cfg = train_cfg.direct_safety
    direct_safety_enabled = bool(direct_safety_cfg.enabled)
    direct_safety_heuristics = None
    direct_safety_train_rows = None
    direct_safety_val_rows = None
    direct_safety_train_contexts = None
    direct_safety_val_contexts = None
    direct_safety_evaluator = None
    direct_safety_candidate_cfg = None
    direct_safety_placeholder_ids: set[int] | None = None
    if direct_safety_enabled and direct_safety_state:
        direct_safety_heuristics = cast(
            ConstraintRepairHeuristics | None, direct_safety_state.get("heuristics")
        )
        direct_safety_train_rows = cast(list | None, direct_safety_state.get("train_rows"))
        direct_safety_val_rows = cast(list | None, direct_safety_state.get("val_rows"))
        direct_safety_train_contexts = cast(
            list[ViolationContext] | None, direct_safety_state.get("train_contexts")
        )
        direct_safety_val_contexts = cast(
            list[ViolationContext] | None, direct_safety_state.get("val_contexts")
        )
        direct_safety_evaluator = cast(
            CandidateConstraintEvaluator | None, direct_safety_state.get("evaluator")
        )
        direct_safety_candidate_cfg = cast(
            CandidateConfig | None, direct_safety_state.get("candidate_cfg")
        )
        direct_safety_placeholder_ids = cast(
            set[int] | None, direct_safety_state.get("placeholder_ids_set")
        )

    policy_enabled = bool(getattr(model, "_policy_enabled", False))
    policy_train_contexts = None
    policy_val_contexts = None
    if policy_enabled and policy_state:
        policy_train_contexts = cast(list[ViolationContext] | None, policy_state.get("train_contexts"))
        policy_val_contexts = cast(list[ViolationContext] | None, policy_state.get("val_contexts"))

    if chooser_enabled and direct_safety_enabled:
        raise RuntimeError("chooser and direct_safety cannot be enabled at the same time.")

    # Unpack training configuration.
    batch_size = train_cfg.batch_size
    num_epochs = train_cfg.num_epochs
    early_stopping_rounds = train_cfg.early_stopping_rounds
    grad_clip = train_cfg.grad_clip if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0 else None
    pin_memory = device.type == "cuda" if train_cfg.pin_memory is None else train_cfg.pin_memory

    # Device introspection (CUDA memory tracking/debugging).
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        # Fast-path defaults for NVIDIA Tensor Cores with stable numerics.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info(f"GPU: {torch.cuda.get_device_name(device_index)}")
        log_cuda_memory("Initial GPU state", device)
    amp_disabled = os.environ.get("TRAIN_DISABLE_AMP", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    amp_enabled = device.type == "cuda" and not amp_disabled
    amp_dtype = torch.bfloat16 if amp_enabled else None
    if amp_enabled:
        logger.info("Automatic mixed precision enabled | dtype=%s", str(amp_dtype).replace("torch.", ""))
    elif device.type == "cuda":
        logger.info("Automatic mixed precision disabled (env: TRAIN_DISABLE_AMP=1)")

    # Dataset bookkeeping / loader construction.
    train_dataset_info = "streaming" if isinstance(train_data, IterableDataset) else "in-memory"
    val_dataset_info = "streaming" if isinstance(val_data, IterableDataset) else "in-memory"
    validation_subset_size = train_cfg.validation_subset_size

    val_loader_data: list[Data] | IterableDataset
    if validation_subset_size is None:
        val_loader_data = val_data
    elif isinstance(val_data, list):
        val_loader_data = val_data[:validation_subset_size]
        logger.info(
            "Validation subset enabled | using first %s/%s in-memory graphs per epoch",
            len(val_loader_data),
            len(val_data),
        )
    else:
        val_loader_data = ValidationSubsetStream(cast(IterableDataset, val_data), validation_subset_size)
        logger.info(
            "Validation subset enabled | using first %s streamed graphs per epoch",
            validation_subset_size,
        )

    try:
        train_size = len(train_data)
        logger.info(f"Train dataset ({train_dataset_info}) contains {train_size} graphs")
    except TypeError:
        logger.info(f"Train dataset is {train_dataset_info}; size will be determined lazily")

    try:
        val_size = len(val_data)
        logger.info(f"Validation dataset ({val_dataset_info}) contains {val_size} graphs")
    except TypeError:
        logger.info(f"Validation dataset is {val_dataset_info}; size will be determined lazily")

    train_is_iterable = isinstance(train_data, IterableDataset)

    def _loader_kwargs_for_split(split: str, dataset: list[Data] | IterableDataset) -> dict[str, Any]:
        dataset_is_iterable = isinstance(dataset, IterableDataset)
        num_workers = train_cfg.num_workers
        if split == "val" and validation_subset_size is not None and dataset_is_iterable and num_workers > 0:
            logger.info(
                "Validation subset uses num_workers=0 so streamed validation emits one global prefix only."
            )
            num_workers = 0
        effective_pin_memory = pin_memory
        if dataset_is_iterable and effective_pin_memory and device.type == "cuda":
            logger.warning(
                "Disabling pin_memory for %s loader because streamed graph datasets with worker queues "
                "can exhaust host/shared memory on CUDA.",
                split,
            )
            effective_pin_memory = False

        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "pin_memory": effective_pin_memory,
            "num_workers": num_workers,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = split == "train" and not dataset_is_iterable
            kwargs["prefetch_factor"] = 1 if dataset_is_iterable else 2

        logger.info(
            "%s loader settings | batch_size=%s num_workers=%s pin_memory=%s persistent_workers=%s prefetch_factor=%s",
            split.capitalize(),
            kwargs["batch_size"],
            kwargs["num_workers"],
            kwargs["pin_memory"],
            kwargs.get("persistent_workers", False),
            kwargs.get("prefetch_factor", "n/a"),
        )
        return kwargs

    train_loader = DataLoader(
        cast(Any, train_data),
        shuffle=(not train_is_iterable),
        **_loader_kwargs_for_split("train", train_data),
    )
    val_loader = DataLoader(
        cast(Any, val_loader_data),
        shuffle=False,
        **_loader_kwargs_for_split("val", val_loader_data),
    )

    # Optimiser / scheduler / loss boilerplate.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.scheduler_factor,
        patience=train_cfg.scheduler_patience,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    entity_allowed_mask: torch.Tensor | None = None
    predicate_allowed_mask: torch.Tensor | None = None
    if (
        hasattr(model, "entity_class_ids")
        and torch.is_tensor(getattr(model, "entity_class_ids"))
        and int(getattr(model, "entity_class_ids").numel()) < model.num_target_ids
    ):
        entity_allowed_mask = torch.zeros(model.num_target_ids, dtype=torch.bool, device=device)
        entity_ids = getattr(model, "entity_class_ids").to(device=device, dtype=torch.long).view(-1)
        entity_allowed_mask[entity_ids] = True
    if (
        hasattr(model, "predicate_class_ids")
        and torch.is_tensor(getattr(model, "predicate_class_ids"))
        and int(getattr(model, "predicate_class_ids").numel()) < model.num_target_ids
    ):
        predicate_allowed_mask = torch.zeros(model.num_target_ids, dtype=torch.bool, device=device)
        predicate_ids = getattr(model, "predicate_class_ids").to(device=device, dtype=torch.long).view(-1)
        predicate_allowed_mask[predicate_ids] = True

    def _assert_targets_supported_by_heads(targets: torch.Tensor, *, split: str, batch_idx: int) -> None:
        if entity_allowed_mask is not None:
            entity_targets = targets[:, ENTITY_SLOT_INDICES].reshape(-1)
            entity_ok = entity_allowed_mask[entity_targets]
            if not bool(entity_ok.all()):
                bad_ids = torch.unique(entity_targets[~entity_ok].detach().cpu())[:10].tolist()
                raise AssertionError(
                    f"{split} batch {batch_idx}: found entity target ids outside entity_class_ids: {bad_ids}"
                )
        if predicate_allowed_mask is not None:
            predicate_targets = targets[:, PREDICATE_SLOT_INDICES].reshape(-1)
            predicate_ok = predicate_allowed_mask[predicate_targets]
            if not bool(predicate_ok.all()):
                bad_ids = torch.unique(predicate_targets[~predicate_ok].detach().cpu())[:10].tolist()
                raise AssertionError(
                    f"{split} batch {batch_idx}: found predicate target ids outside predicate_class_ids: {bad_ids}"
                )

    factor_cfg = train_cfg.factor_loss
    factor_criterion: torch.nn.BCEWithLogitsLoss | None = None
    if factor_cfg.enabled:
        pos_weight_tensor = None
        if factor_cfg.pos_weight is not None and factor_cfg.pos_weight > 0:
            pos_weight_tensor = torch.tensor([float(factor_cfg.pos_weight)], device=device)
        factor_criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)

    # Dynamic constraint reweighting / early stopping setup.
    dynamic_weighter = DynamicConstraintWeighter(train_cfg.constraint_loss.dynamic_reweighting)
    try:
        train_total_batches = len(train_loader)
    except TypeError:
        train_total_batches = estimated_train_batches
    try:
        val_total_batches = len(val_loader)
    except TypeError:
        val_total_batches = estimated_val_batches

    def _should_log_batch(batch_idx: int, total_batches: int | None, last_logged_percent: int) -> tuple[bool, int]:
        """
        Emit per-batch logs for:
        - the first 10 iterations
        - then each new 5% progress point within the epoch (when total is known)
        """
        if batch_idx <= 10:
            return True, last_logged_percent
        if total_batches is None or total_batches <= 0:
            return False, last_logged_percent
        progress_percent = int((batch_idx * 100) / total_batches)
        progress_bucket = min((progress_percent // 5) * 5, 100)
        if progress_bucket > last_logged_percent:
            return True, progress_bucket
        return False, last_logged_percent

    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Invalid integer for %s=%r; using default %s", name, raw, default)
            return default
        return max(value, minimum)

    prefetch_enabled = bool(
        device.type == "cuda" and _env_flag("TRAIN_CUDA_PREFETCH", default=True)
    )

    def _iter_batches(loader: DataLoader) -> Iterator[tuple[Data, bool]]:
        """
        Yield `(batch, on_device)` where `on_device=True` indicates CUDA-prefetched
        batches already moved to GPU.
        """
        if not prefetch_enabled:
            for batch in loader:
                yield batch, False
            return

        assert device.type == "cuda"
        prefetch_stream = torch.cuda.Stream(device=device)
        loader_iter = iter(loader)
        next_batch: Data | None = None

        def _preload_next() -> None:
            nonlocal next_batch
            try:
                candidate = next(loader_iter)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(prefetch_stream):
                candidate = candidate.to(device, non_blocking=True)
            next_batch = candidate

        _preload_next()
        while next_batch is not None:
            torch.cuda.current_stream(device).wait_stream(prefetch_stream)
            batch = next_batch
            record_stream = getattr(batch, "record_stream", None)
            if callable(record_stream):
                record_stream(torch.cuda.current_stream(device))
            _preload_next()
            yield batch, True

    if prefetch_enabled:
        logger.info("CUDA batch prefetch enabled (env: TRAIN_CUDA_PREFETCH=1)")

    timing_enabled = _env_flag("TRAIN_TIMING_PROFILE", default=False)
    timing_warmup_batches = _env_int("TRAIN_TIMING_WARMUP_BATCHES", 10, minimum=0)
    timing_log_every = _env_int("TRAIN_TIMING_LOG_EVERY", 100, minimum=1)

    train_timing_keys = (
        "data_s",
        "forward_s",
        "base_loss_s",
        "policy_s",
        "fix_s",
        "factor_s",
        "chooser_s",
        "chooser_candidate_build_s",
        "chooser_packed_score_s",
        "chooser_evaluator_terms_s",
        "chooser_eval_constraint_s",
        "chooser_eval_math_s",
        "reweight_s",
        "backward_s",
        "optim_s",
        "metrics_s",
        "total_s",
    )
    val_timing_keys = (
        "data_s",
        "forward_s",
        "base_loss_s",
        "policy_s",
        "fix_s",
        "factor_s",
        "chooser_s",
        "chooser_candidate_build_s",
        "chooser_packed_score_s",
        "chooser_evaluator_terms_s",
        "chooser_eval_constraint_s",
        "chooser_eval_math_s",
        "metrics_s",
        "total_s",
    )

    def _new_timing_bucket(keys: tuple[str, ...]) -> dict[str, float]:
        return {key: 0.0 for key in keys}

    def _timing_bucket_ms(bucket: dict[str, float], count: int, key: str) -> float:
        if count <= 0:
            return 0.0
        return (bucket.get(key, 0.0) / count) * 1000.0

    if timing_enabled:
        logger.info(
            "Timing profiler enabled | warmup_batches=%s log_every=%s (env: TRAIN_TIMING_PROFILE=1)",
            timing_warmup_batches,
            timing_log_every,
        )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    # Rolling training history for logging + persistence.
    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc_all6": [],
        "val_acc_all6": [],
        "train_acc_slot": [],
        "val_acc_slot": [],
        "learning_rate": [],
        "train_grad_norm_mean": [],
        "train_grad_norm_max": [],
        "train_param_norm": [],
        "train_param_abs_max": [],
        "train_edit_logit_abs_max": [],
        "val_edit_logit_abs_max": [],
        "train_factor_logit_abs_max": [],
        "val_factor_logit_abs_max": [],
        "train_chooser_score_abs_max": [],
        "val_chooser_score_abs_max": [],
    }

    chooser_loss_mode = chooser_cfg.loss_mode
    chooser_loss_weight = max(float(getattr(chooser_cfg, "loss_weight", 0.5)), 0.0)
    chooser_need_regression = chooser_loss_mode == "fix1" and chooser_cfg.beta_no_regression > 0
    chooser_need_primary = chooser_cfg.gamma_primary > 0
    chooser_placeholder_ids_for_candidates = (
        chooser_placeholder_ids
        if chooser_placeholder_ids is not None
        else (set(chooser_heuristics.placeholder_ids.values()) if chooser_heuristics is not None else set())
    )
    chooser_entity_allowed_ids = (
        model.entity_class_ids
        if hasattr(model, "entity_class_ids")
        and torch.is_tensor(getattr(model, "entity_class_ids"))
        and int(getattr(model, "entity_class_ids").numel()) < model.num_target_ids
        else None
    )
    chooser_predicate_allowed_ids = (
        model.predicate_class_ids
        if hasattr(model, "predicate_class_ids")
        and torch.is_tensor(getattr(model, "predicate_class_ids"))
        and int(getattr(model, "predicate_class_ids").numel()) < model.num_target_ids
        else None
    )
    chooser_slot_allowed_ids = (
        chooser_entity_allowed_ids,
        chooser_predicate_allowed_ids,
        chooser_entity_allowed_ids,
        chooser_entity_allowed_ids,
        chooser_predicate_allowed_ids,
        chooser_entity_allowed_ids,
    )
    direct_safety_placeholder_ids_for_candidates = (
        direct_safety_placeholder_ids
        if direct_safety_placeholder_ids is not None
        else (
            set(direct_safety_heuristics.placeholder_ids.values())
            if direct_safety_heuristics is not None
            else set()
        )
    )
    direct_safety_slot_allowed_ids = chooser_slot_allowed_ids

    logger.info("Starting training loop for %d epochs", num_epochs)
    epoch_iter = progress_bar(range(num_epochs), desc="Training Epochs", leave=True)
    sanity_checked = False

    for epoch in epoch_iter:
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started")
        if device.type == "cuda" and logger.isEnabledFor(logging.DEBUG):
            log_cuda_memory(f"Epoch {epoch + 1} pre-forward", device, level=logging.DEBUG)
        train_epoch_start = time.perf_counter()

        # ------- Training -------
        model.train()
        train_graph_loss_sum = 0.0  # combined loss (per graph)
        train_graph_count = 0  # graphs processed (for averaging)
        train_slots = 0  # total number of slots processed
        train_correct = 0  # all-6 correct predictions
        train_total = 0  # total number of graphs
        train_slot_correct = 0  # per-slot correct predictions
        train_slot_total = 0  # total number of slots processed
        train_slot_loss_sums = [0.0] * NUM_SLOTS  # per-slot loss sums
        train_slot_counts = [0] * NUM_SLOTS  # per-slot sample counts
        train_slot_correct_slots = [0] * NUM_SLOTS  # per-slot correct predictions
        train_factor_loss_sum = 0.0
        train_factor_count = 0
        train_factor_correct = 0
        train_chooser_loss_sum = 0.0
        train_chooser_count = 0
        train_direct_safety_loss_sum = 0.0
        train_direct_safety_count = 0
        train_policy_loss_sum = 0.0
        train_policy_count = 0
        train_policy_correct = 0
        train_grad_norm_sum = 0.0
        train_grad_norm_count = 0
        train_grad_norm_max = 0.0
        train_edit_logit_abs_max = 0.0
        train_factor_logit_abs_max = 0.0
        train_chooser_score_abs_max = 0.0

        train_constraint_metrics = ConstraintMetricsAccumulator()
        train_last_logged_percent = 0
        train_timing_window = _new_timing_bucket(train_timing_keys)
        train_timing_epoch = _new_timing_bucket(train_timing_keys)
        train_timing_window_count = 0
        train_timing_epoch_count = 0
        for batch_idx, (data, data_on_device) in enumerate(_iter_batches(train_loader), start=1):
            batch_t0 = time.perf_counter() if timing_enabled else 0.0
            phase_t0 = time.perf_counter() if timing_enabled else 0.0
            if not data_on_device:
                data = data.to(device, non_blocking=True)
            _assert_factor_type_ids_supported(data, int(getattr(model, "_num_factor_types", 0)))
            if train_cfg.validate_factor_labels:
                _assert_factor_labels_batch(data)
            targets = data.y.long()

            # Validation checks
            assert targets.dim() == 2 and targets.size(1) == NUM_SLOTS, (
                f"targets must be (B,{NUM_SLOTS}), got {tuple(targets.shape)}"
            )
            t_min, t_max = targets.min().item(), targets.max().item()
            assert 0 <= t_min and t_max < model.num_target_ids, (
                f"Expected targets in [0,{model.num_target_ids}), got [{t_min},{t_max}]"
            )
            _assert_targets_supported_by_heads(targets, split="train", batch_idx=batch_idx)
            data_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            optimizer.zero_grad(set_to_none=True)

            phase_t0 = time.perf_counter() if timing_enabled else 0.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model(data)
            forward_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0
            if torch.is_tensor(outputs):
                outputs = {
                    "edit_logits": outputs,
                    "node_emb": None,
                    "graph_emb": None,
                }
            out = outputs.get("edit_logits") if isinstance(outputs, dict) else None
            assert out is not None, "Model output must include 'edit_logits'."
            if not sanity_checked:
                assert isinstance(outputs, dict), "Model output must be a dict."
                assert "edit_logits" in outputs, "Model output must include 'edit_logits'."
                sanity_checked = True
            assert out.dim() == 3 and out.size(1) == NUM_SLOTS and out.size(2) == model.num_target_ids, (
                f"Expected out (batch_size,{NUM_SLOTS},{model.num_target_ids}), got {tuple(out.shape)}"
            )
            train_edit_logit_abs_max = max(train_edit_logit_abs_max, _max_abs_value(out))

            out_flat = out.reshape(-1, out.size(-1))
            targets_flat = targets.reshape(-1)

            # Constraint type extraction and loss computation
            phase_t0 = time.perf_counter() if timing_enabled else 0.0
            per_slot_loss = criterion(out_flat, targets_flat)
            loss_matrix = per_slot_loss.view(-1, NUM_SLOTS)
            graph_loss = loss_matrix.mean(dim=1)
            base_loss_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0
            policy_s = 0.0
            fix_s = 0.0
            factor_s = 0.0
            chooser_s = 0.0
            chooser_candidate_build_s = 0.0
            chooser_packed_score_s = 0.0
            chooser_evaluator_terms_s = 0.0
            chooser_eval_constraint_s = 0.0
            chooser_eval_math_s = 0.0
            reweight_s = 0.0

            if policy_enabled:
                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                policy_logits = outputs.get("policy_logits")
                if policy_logits is None:
                    raise RuntimeError("Policy choice enabled but model output missing policy_logits.")
                if policy_train_contexts is None:
                    raise RuntimeError("Policy choice enabled but training contexts are missing.")
                batch_context_indices = _batch_int_attr(
                    data,
                    "context_index",
                    graph_loss.size(0),
                    default_to_batch_index=True,
                    min_value=0,
                )
                targets_rows = targets.detach().cpu().tolist()
                policy_labels = []
                for idx, context_index in enumerate(batch_context_indices):
                    if context_index < 0 or context_index >= len(policy_train_contexts):
                        raise RuntimeError("Policy context index out of bounds.")
                    context = policy_train_contexts[context_index]
                    decision = derive_policy_label(targets_rows[idx], context, none_class=NONE_CLASS_INDEX)
                    policy_labels.append(int(decision.policy_id))
                policy_targets = torch.tensor(policy_labels, dtype=torch.long, device=graph_loss.device)
                policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets, reduction="none")
                graph_loss = graph_loss + policy_loss
                train_policy_loss_sum += float(policy_loss.sum().item())
                train_policy_count += policy_loss.numel()
                train_policy_correct += int((policy_logits.argmax(dim=-1) == policy_targets).sum().item())
                policy_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            # Optional - Constraint-is-fixed loss term
            fix_weight = 0.0
            if fix_scheduler and fix_scheduler.enabled and fix_heuristics is not None and train_contexts is not None:
                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                context_attr = getattr(data, "context_index", None)
                if context_attr is not None:
                    context_indices = _batch_int_attr(
                        data,
                        "context_index",
                        graph_loss.size(0),
                        default_to_batch_index=True,
                    )
                    if any(idx < 0 or idx >= len(train_contexts) for idx in context_indices):
                        raise RuntimeError("Fix-loss context index out of bounds.")
                    batch_contexts = [train_contexts[idx] for idx in context_indices]
                    if train_total_batches:
                        batch_progress = (batch_idx - 1) / max(train_total_batches, 1)
                    else:
                        batch_progress = 0.0
                    progress = epoch + max(batch_progress, 0.0)
                    fix_weight = fix_scheduler.weight_for_progress(progress)
                    if fix_weight > 0.0:
                        fix_probs = compute_fix_probabilities(out, batch_contexts, fix_heuristics)
                        fix_penalty = 1.0 - fix_probs
                        graph_loss = graph_loss + fix_weight * fix_penalty
                fix_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            # Optional - Factor satisfaction loss (pre-state)
            if factor_cfg.enabled:
                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                assert factor_criterion is not None
                factor_logits = outputs.get("factor_logits_pre")
                factor_logits_post_gold = outputs.get("factor_logits_post_gold")
                factor_mask = outputs.get("factor_mask_pre")
                factor_graph_index = outputs.get("factor_graph_index")
                factor_targets = getattr(data, "factor_satisfied_pre", None)
                assert factor_targets is not None, "Missing factor_satisfied_pre for factor loss."
                factor_targets = torch.as_tensor(
                    factor_targets, dtype=torch.float32, device=graph_loss.device
                ).view(-1)
                if factor_mask is None:
                    factor_mask = torch.ones_like(factor_targets, dtype=torch.bool)
                else:
                    factor_mask = torch.as_tensor(
                        factor_mask, dtype=torch.bool, device=graph_loss.device
                    ).view(-1)
                assert factor_logits is not None, "Model output missing factor_logits_pre."
                factor_logits = torch.as_tensor(factor_logits, device=graph_loss.device).view(-1)
                train_factor_logit_abs_max = max(train_factor_logit_abs_max, _max_abs_value(factor_logits))
                assert factor_logits.numel() == factor_targets.numel(), (
                    "factor_logits_pre length must match factor_satisfied_pre length."
                )
                if factor_graph_index is None:
                    if graph_loss.numel() == 1:
                        factor_graph_index = torch.zeros(
                            factor_targets.numel(), device=graph_loss.device, dtype=torch.long
                        )
                    else:
                        raise AssertionError("factor_graph_index is required for batched factor loss.")
                factor_graph_index = torch.as_tensor(factor_graph_index, device=graph_loss.device).view(-1)
                if train_cfg.validate_factor_labels:
                    _assert_factor_logit_alignment(data, factor_logits, factor_graph_index)
                if factor_cfg.only_checkable:
                    checkable = getattr(data, "factor_checkable_pre", None)
                    if checkable is None:
                        active_mask = factor_mask
                    else:
                        checkable = torch.as_tensor(
                            checkable, dtype=torch.bool, device=graph_loss.device
                        ).view(-1)
                        if checkable.numel() != factor_targets.numel():
                            raise AssertionError(
                                "factor_checkable_pre length must match factor_satisfied_pre length."
                            )
                        active_mask = factor_mask & checkable
                else:
                    active_mask = torch.ones_like(factor_mask, dtype=torch.bool)
                if active_mask.numel() != factor_targets.numel():
                    raise AssertionError("factor_mask_pre length must match factor_satisfied_pre length.")
                _log_factor_debug(data, factor_logits, factor_graph_index)

                graph_loss_add = torch.zeros(
                    graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                )
                factor_loss_total = 0.0
                factor_count_total = 0
                factor_correct_total = 0

                active_float = active_mask.to(dtype=graph_loss.dtype)
                per_factor_loss = factor_criterion(factor_logits, factor_targets).to(dtype=graph_loss.dtype)
                per_factor_loss = per_factor_loss * active_float
                active_count = int(active_mask.sum().item())
                if factor_cfg.weight_pre > 0.0:
                    pre_graph_add = torch.zeros(
                        graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                    )
                    pre_graph_add.scatter_add_(0, factor_graph_index, per_factor_loss)
                    if factor_cfg.per_graph_reduction == "mean":
                        per_graph_counts = torch.zeros(
                            graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                        )
                        per_graph_counts.scatter_add_(0, factor_graph_index, active_float)
                        pre_graph_add = pre_graph_add / per_graph_counts.clamp(min=1.0)
                    graph_loss_add = graph_loss_add + (factor_cfg.weight_pre * pre_graph_add)
                factor_loss_total += float(per_factor_loss.sum().item())
                factor_count_total += active_count
                if active_count > 0:
                    preds = (factor_logits > 0).to(dtype=torch.long)
                    targets_long = factor_targets.to(dtype=torch.long)
                    factor_correct_total += int((preds[active_mask] == targets_long[active_mask]).sum().item())

                if factor_logits_post_gold is not None and factor_cfg.weight_post_gold > 0.0:
                    post_targets = getattr(data, "factor_satisfied_post_gold", None)
                    assert post_targets is not None, "Missing factor_satisfied_post_gold for factor post-gold loss."
                    post_targets = torch.as_tensor(
                        post_targets, dtype=torch.float32, device=graph_loss.device
                    ).view(-1)
                    factor_logits_post_gold = torch.as_tensor(
                        factor_logits_post_gold, device=graph_loss.device
                    ).view(-1)
                    train_factor_logit_abs_max = max(
                        train_factor_logit_abs_max,
                        _max_abs_value(factor_logits_post_gold),
                    )
                    if factor_logits_post_gold.numel() != post_targets.numel():
                        raise AssertionError(
                            "factor_logits_post_gold length must match factor_satisfied_post_gold length."
                        )
                    if factor_cfg.only_checkable:
                        post_checkable = getattr(data, "factor_checkable_post_gold", None)
                        if post_checkable is None:
                            post_active_mask = torch.ones_like(post_targets, dtype=torch.bool)
                        else:
                            post_active_mask = torch.as_tensor(
                                post_checkable, dtype=torch.bool, device=graph_loss.device
                            ).view(-1)
                            if post_active_mask.numel() != post_targets.numel():
                                raise AssertionError(
                                    "factor_checkable_post_gold length must match factor_satisfied_post_gold length."
                                )
                    else:
                        post_active_mask = torch.ones_like(post_targets, dtype=torch.bool)
                    post_active_float = post_active_mask.to(dtype=graph_loss.dtype)
                    per_factor_post_loss = factor_criterion(
                        factor_logits_post_gold, post_targets
                    ).to(dtype=graph_loss.dtype)
                    per_factor_post_loss = per_factor_post_loss * post_active_float
                    post_graph_add = torch.zeros(
                        graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                    )
                    post_graph_add.scatter_add_(0, factor_graph_index, per_factor_post_loss)
                    if factor_cfg.per_graph_reduction == "mean":
                        post_graph_counts = torch.zeros(
                            graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                        )
                        post_graph_counts.scatter_add_(0, factor_graph_index, post_active_float)
                        post_graph_add = post_graph_add / post_graph_counts.clamp(min=1.0)
                    graph_loss_add = graph_loss_add + (factor_cfg.weight_post_gold * post_graph_add)
                    post_active_count = int(post_active_mask.sum().item())
                    factor_loss_total += float(per_factor_post_loss.sum().item())
                    factor_count_total += post_active_count
                    if post_active_count > 0:
                        post_preds = (factor_logits_post_gold > 0).to(dtype=torch.long)
                        post_targets_long = post_targets.to(dtype=torch.long)
                        factor_correct_total += int(
                            (post_preds[post_active_mask] == post_targets_long[post_active_mask]).sum().item()
                        )

                graph_loss = graph_loss + graph_loss_add

                train_factor_loss_sum += factor_loss_total
                train_factor_count += factor_count_total
                train_factor_correct += factor_correct_total
                factor_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            if chooser_enabled:
                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                if chooser_evaluator is None or chooser_heuristics is None:
                    raise RuntimeError("Chooser enabled but evaluator/heuristics are not initialized.")
                if chooser_candidate_cfg is None:
                    chooser_candidate_cfg = CandidateConfig(
                        topk_candidates=chooser_cfg.topk_candidates,
                        max_candidates_total=chooser_cfg.max_candidates_total,
                    )
                graph_emb = outputs.get("graph_emb")
                if graph_emb is None:
                    raise RuntimeError("Model output missing graph_emb required for chooser.")
                if chooser_train_contexts is None or chooser_train_rows is None:
                    raise RuntimeError("Chooser enabled but training contexts/rows are missing.")
                batch_context_indices = _batch_int_attr(
                    data,
                    "context_index",
                    graph_loss.size(0),
                    default_to_batch_index=True,
                    min_value=0,
                )
                batch_primary_indices = _batch_int_attr(
                    data,
                    "primary_factor_index",
                    graph_loss.size(0),
                    default_value=0,
                    min_value=0,
                )
                gold_rows = targets.detach().cpu().tolist()
                out_detached = out.detach()
                batch_slot_vals, batch_slot_ids = _compute_batch_slot_topk(
                    out_detached,
                    topk_per_slot=chooser_candidate_cfg.topk_per_slot,
                    slot_allowed_ids=chooser_slot_allowed_ids,
                )
                chooser_losses = torch.zeros(
                    graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                )
                candidate_groups: list[list[tuple[int, int, int, int, int, int]]] = []
                gold_indices: list[int] = []
                candidate_rows: list[Any] = []
                primary_indices: list[int] = []
                packed_candidates: list[tuple[int, int, int, int, int, int]] = []
                packed_graph_index: list[int] = []
                chooser_build_t0 = time.perf_counter() if timing_enabled else 0.0
                for idx, context_index in enumerate(batch_context_indices):
                    if (
                        context_index < 0
                        or context_index >= len(chooser_train_contexts)
                        or context_index >= len(chooser_train_rows)
                    ):
                        raise RuntimeError("Chooser context index out of bounds.")
                    context = chooser_train_contexts[context_index]
                    row = chooser_train_rows[context_index]
                    primary_index = int(batch_primary_indices[idx])
                    add_topk = _topk_triples_from_slot_topk_row(
                        batch_slot_vals,
                        batch_slot_ids,
                        row_index=idx,
                        slots=(0, 1, 2),
                        topk_triples=chooser_candidate_cfg.topk_candidates,
                    )
                    del_topk = _topk_triples_from_slot_topk_row(
                        batch_slot_vals,
                        batch_slot_ids,
                        row_index=idx,
                        slots=(3, 4, 5),
                        topk_triples=chooser_candidate_cfg.topk_candidates,
                    )
                    candidates, gold_index = build_candidates(
                        gold_slots=gold_rows[idx],
                        context=context,
                        heuristics=chooser_heuristics,
                        proposal_logits=out_detached[idx],
                        cfg=chooser_candidate_cfg,
                        placeholder_ids=chooser_placeholder_ids_for_candidates,
                        num_target_ids=model.num_target_ids,
                        slot_allowed_ids=chooser_slot_allowed_ids,
                        precomputed_add_topk=add_topk,
                        precomputed_del_topk=del_topk,
                    )
                    if gold_index is None:
                        raise RuntimeError("Chooser training requires include_gold=True.")
                    candidate_groups.append(candidates)
                    gold_indices.append(gold_index)
                    candidate_rows.append(row)
                    primary_indices.append(primary_index)
                    packed_candidates.extend(candidates)
                    packed_graph_index.extend([idx] * len(candidates))
                chooser_candidate_build_s = (time.perf_counter() - chooser_build_t0) if timing_enabled else 0.0

                if not packed_candidates:
                    raise RuntimeError("Chooser produced no candidates for the current batch.")

                chooser_score_t0 = time.perf_counter() if timing_enabled else 0.0
                packed_candidate_tensor = torch.tensor(
                    packed_candidates, dtype=torch.long, device=graph_loss.device
                )
                packed_graph_index_tensor = torch.tensor(
                    packed_graph_index, dtype=torch.long, device=graph_loss.device
                )
                packed_scores = model.score_candidates_packed(
                    graph_emb, packed_candidate_tensor, packed_graph_index_tensor
                )
                train_chooser_score_abs_max = max(train_chooser_score_abs_max, _max_abs_value(packed_scores))
                chooser_packed_score_s = (time.perf_counter() - chooser_score_t0) if timing_enabled else 0.0

                chooser_eval_t0 = time.perf_counter() if timing_enabled else 0.0
                offset = 0
                for idx, candidates in enumerate(candidate_groups):
                    candidate_count = len(candidates)
                    scores = packed_scores[offset : offset + candidate_count]
                    offset += candidate_count
                    row = candidate_rows[idx]
                    gold_index = gold_indices[idx]
                    primary_index = primary_indices[idx]
                    math_t0 = time.perf_counter() if timing_enabled else 0.0
                    log_probs = F.log_softmax(scores, dim=0)
                    probs = log_probs.exp()
                    chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                    if chooser_loss_mode == "global_fix":
                        constraint_t0 = time.perf_counter() if timing_enabled else 0.0
                        metrics_summary = chooser_evaluator.evaluate_candidate_metrics(
                            row,
                            candidates=candidates,
                            primary_factor_index=primary_index,
                        )
                        chooser_eval_constraint_s += (time.perf_counter() - constraint_t0) if timing_enabled else 0.0
                        math_t0 = time.perf_counter() if timing_enabled else 0.0
                        satisfaction = torch.tensor(
                            [m.global_satisfied_fraction for m in metrics_summary],
                            dtype=graph_loss.dtype,
                            device=graph_loss.device,
                        )
                        chooser_loss = -torch.sum(probs * satisfaction)
                        chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                    else:
                        math_t0 = time.perf_counter() if timing_enabled else 0.0
                        ce_loss = -log_probs[gold_index]
                        chooser_loss = ce_loss
                        chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                        if chooser_need_regression or chooser_need_primary:
                            constraint_t0 = time.perf_counter() if timing_enabled else 0.0
                            metrics_summary = chooser_evaluator.evaluate_candidate_metrics(
                                row,
                                candidates=candidates,
                                primary_factor_index=primary_index,
                            )
                            chooser_eval_constraint_s += (time.perf_counter() - constraint_t0) if timing_enabled else 0.0
                        else:
                            metrics_summary = []
                        if chooser_need_regression and metrics_summary:
                            math_t0 = time.perf_counter() if timing_enabled else 0.0
                            regression_tensor = torch.tensor(
                                [m.srr for m in metrics_summary],
                                dtype=graph_loss.dtype,
                                device=graph_loss.device,
                            )
                            gold_regression = regression_tensor[gold_index]
                            reg_penalty = torch.sum(
                                probs * torch.clamp(regression_tensor - gold_regression, min=0.0)
                            )
                            chooser_loss = chooser_loss + chooser_cfg.beta_no_regression * reg_penalty
                            chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                        if chooser_need_primary and metrics_summary:
                            math_t0 = time.perf_counter() if timing_enabled else 0.0
                            primary_tensor = torch.tensor(
                                [float(m.primary_satisfied) for m in metrics_summary],
                                dtype=graph_loss.dtype,
                                device=graph_loss.device,
                            )
                            primary_penalty = torch.sum(probs * (1.0 - primary_tensor))
                            chooser_loss = chooser_loss + chooser_cfg.gamma_primary * primary_penalty
                            chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                    chooser_losses[idx] = chooser_loss
                chooser_evaluator_terms_s = (time.perf_counter() - chooser_eval_t0) if timing_enabled else 0.0
                assert offset == packed_scores.numel(), "Packed chooser score size mismatch."

                graph_loss = graph_loss + (chooser_loss_weight * chooser_losses)
                train_chooser_loss_sum += float(chooser_losses.sum().item())
                train_chooser_count += chooser_losses.numel()
                chooser_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            if direct_safety_enabled:
                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                if direct_safety_evaluator is None or direct_safety_heuristics is None:
                    raise RuntimeError("Direct safety enabled but evaluator/heuristics are not initialized.")
                if direct_safety_candidate_cfg is None:
                    direct_safety_candidate_cfg = CandidateConfig(
                        topk_candidates=direct_safety_cfg.topk_candidates,
                        max_candidates_total=direct_safety_cfg.max_candidates_total,
                    )
                if direct_safety_train_contexts is None or direct_safety_train_rows is None:
                    raise RuntimeError("Direct safety enabled but training contexts/rows are missing.")
                batch_context_indices = _batch_int_attr(
                    data,
                    "context_index",
                    graph_loss.size(0),
                    default_to_batch_index=True,
                    min_value=0,
                )
                batch_primary_indices = _batch_int_attr(
                    data,
                    "primary_factor_index",
                    graph_loss.size(0),
                    default_value=0,
                    min_value=0,
                )
                out_detached = out.detach()
                batch_slot_vals, batch_slot_ids = _compute_batch_slot_topk(
                    out_detached,
                    topk_per_slot=direct_safety_candidate_cfg.topk_per_slot,
                    slot_allowed_ids=direct_safety_slot_allowed_ids,
                )
                direct_safety_losses = torch.zeros(
                    graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                )
                for idx, context_index in enumerate(batch_context_indices):
                    if (
                        context_index < 0
                        or context_index >= len(direct_safety_train_contexts)
                        or context_index >= len(direct_safety_train_rows)
                    ):
                        raise RuntimeError("Direct safety context index out of bounds.")
                    context = direct_safety_train_contexts[context_index]
                    row = direct_safety_train_rows[context_index]
                    primary_index = int(batch_primary_indices[idx])
                    add_topk = _topk_triples_from_slot_topk_row(
                        batch_slot_vals,
                        batch_slot_ids,
                        row_index=idx,
                        slots=(0, 1, 2),
                        topk_triples=direct_safety_candidate_cfg.topk_candidates,
                    )
                    del_topk = _topk_triples_from_slot_topk_row(
                        batch_slot_vals,
                        batch_slot_ids,
                        row_index=idx,
                        slots=(3, 4, 5),
                        topk_triples=direct_safety_candidate_cfg.topk_candidates,
                    )
                    candidates, _ = build_candidates(
                        gold_slots=targets[idx].detach().cpu().tolist(),
                        context=context,
                        heuristics=direct_safety_heuristics,
                        proposal_logits=out_detached[idx],
                        cfg=direct_safety_candidate_cfg,
                        placeholder_ids=direct_safety_placeholder_ids_for_candidates,
                        num_target_ids=model.num_target_ids,
                        slot_allowed_ids=direct_safety_slot_allowed_ids,
                        precomputed_add_topk=add_topk,
                        precomputed_del_topk=del_topk,
                    )
                    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=graph_loss.device)
                    scores = score_candidates_from_logits(out[idx], candidate_tensor)
                    log_probs = F.log_softmax(scores, dim=0)
                    probs = log_probs.exp()
                    metrics_summary = direct_safety_evaluator.evaluate_candidate_metrics(
                        row,
                        candidates=candidates,
                        primary_factor_index=primary_index,
                    )
                    primary_tensor = torch.tensor(
                        [float(m.primary_satisfied) for m in metrics_summary],
                        dtype=graph_loss.dtype,
                        device=graph_loss.device,
                    )
                    secondary_tensor = torch.tensor(
                        [m.srr for m in metrics_summary],
                        dtype=graph_loss.dtype,
                        device=graph_loss.device,
                    )
                    primary_penalty = torch.sum(probs * (1.0 - primary_tensor))
                    secondary_penalty = torch.sum(probs * secondary_tensor)
                    direct_safety_losses[idx] = (
                        direct_safety_cfg.alpha_primary * primary_penalty
                        + direct_safety_cfg.beta_secondary * secondary_penalty
                    )
                graph_loss = graph_loss + direct_safety_losses
                train_direct_safety_loss_sum += float(direct_safety_losses.sum().item())
                train_direct_safety_count += direct_safety_losses.numel()

            # Optional - Rebalance weights based on constraint types
            phase_t0 = time.perf_counter() if timing_enabled else 0.0
            constraint_types = extract_constraint_types(data, loss_matrix.size(0))
            sample_weights = dynamic_weighter.weights_for(constraint_types)

            if dynamic_weighter.enabled:
                weight_tensor = torch.tensor(
                    sample_weights,
                    device=graph_loss.device,
                    dtype=graph_loss.dtype,
                )
                weighted_loss = (graph_loss * weight_tensor).mean()
            else:
                weighted_loss = graph_loss.mean()
            reweight_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            _assert_finite_scalar("weighted_loss", weighted_loss, epoch=epoch + 1, batch_idx=batch_idx)
            phase_t0 = time.perf_counter() if timing_enabled else 0.0
            weighted_loss.backward()
            backward_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            train_graph_loss_sum += graph_loss.detach().sum().item()
            train_graph_count += graph_loss.numel()

            phase_t0 = time.perf_counter() if timing_enabled else 0.0
            if grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                grad_norm = _grad_norm(model.parameters())
            grad_norm_value = _assert_finite_scalar("grad_norm", grad_norm, epoch=epoch + 1, batch_idx=batch_idx)
            train_grad_norm_sum += grad_norm_value
            train_grad_norm_count += 1
            train_grad_norm_max = max(train_grad_norm_max, grad_norm_value)

            optimizer.step()
            optim_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            if logger.isEnabledFor(logging.DEBUG):
                if batch_idx == 1:
                    logger.debug(
                        "First training batch stats: num_graphs=%s num_nodes=%s num_edges=%s",
                        getattr(data, "num_graphs", "?"),
                        getattr(data, "num_nodes", "?"),
                        getattr(data, "num_edges", "?"),
                    )
                if device.type == "cuda" and batch_idx % 10 == 0:
                    log_cuda_memory(
                        f"Epoch {epoch + 1} batch {batch_idx}",
                        device,
                        level=logging.DEBUG,
                    )

            # Accuracy (all-6 and per-slot)
            phase_t0 = time.perf_counter() if timing_enabled else 0.0
            _, predicted = torch.max(out_flat, 1)
            predicted_reshaped = predicted.reshape(-1, NUM_SLOTS)
            targets_reshaped = targets_flat.reshape(-1, NUM_SLOTS)

            all_correct = (predicted_reshaped == targets_reshaped).all(dim=1)
            train_correct += all_correct.sum().item()
            train_total += all_correct.size(0)

            train_slot_correct += (predicted == targets_flat).sum().item()
            train_slot_total += targets_flat.numel()

            # Loss accumulation (per-slot averaging) uses unweighted losses.
            train_slots += targets_flat.numel()

            slot_loss_sums = loss_matrix.sum(dim=0).detach().cpu().tolist()
            batch_graphs = loss_matrix.size(0)
            slot_correct_matrix = (predicted_reshaped == targets_reshaped).to(dtype=torch.long)
            slot_correct_counts = slot_correct_matrix.sum(dim=0).detach().cpu().tolist()
            for idx in range(NUM_SLOTS):
                train_slot_loss_sums[idx] += slot_loss_sums[idx]
                train_slot_counts[idx] += batch_graphs
                train_slot_correct_slots[idx] += int(slot_correct_counts[idx])

            # Per-constraint aggregation
            loss_per_graph = loss_matrix.detach().cpu().sum(dim=1).tolist()
            slot_correct_per_graph = slot_correct_matrix.sum(dim=1).detach().cpu().tolist()
            all_correct_per_graph = all_correct.to(dtype=torch.long).detach().cpu().tolist()
            train_constraint_metrics.update(
                constraint_types,
                loss_per_graph,
                slot_correct_per_graph,
                all_correct_per_graph,
                NUM_SLOTS,
            )
            dynamic_weighter.observe_batch(constraint_types, loss_matrix)
            metrics_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

            if timing_enabled and batch_idx > timing_warmup_batches:
                timing_snapshot = {
                    "data_s": data_s,
                    "forward_s": forward_s,
                    "base_loss_s": base_loss_s,
                    "policy_s": policy_s,
                    "fix_s": fix_s,
                    "factor_s": factor_s,
                    "chooser_s": chooser_s,
                    "chooser_candidate_build_s": chooser_candidate_build_s,
                    "chooser_packed_score_s": chooser_packed_score_s,
                    "chooser_evaluator_terms_s": chooser_evaluator_terms_s,
                    "chooser_eval_constraint_s": chooser_eval_constraint_s,
                    "chooser_eval_math_s": chooser_eval_math_s,
                    "reweight_s": reweight_s,
                    "backward_s": backward_s,
                    "optim_s": optim_s,
                    "metrics_s": metrics_s,
                    "total_s": time.perf_counter() - batch_t0,
                }
                for key, value in timing_snapshot.items():
                    train_timing_window[key] += value
                    train_timing_epoch[key] += value
                train_timing_window_count += 1
                train_timing_epoch_count += 1

            should_log, train_last_logged_percent = _should_log_batch(
                batch_idx,
                train_total_batches,
                train_last_logged_percent,
            )
            if should_log:
                if train_total_batches is not None and train_total_batches > 0:
                    progress = int((batch_idx * 100) / train_total_batches)
                    progress_label = (
                        f"batch {batch_idx}/{train_total_batches} ({min(progress, 100)}%)"
                    )
                else:
                    progress_label = f"batch {batch_idx}"
                elapsed_s = max(time.perf_counter() - train_epoch_start, 1e-9)
                it_per_s = batch_idx / elapsed_s
                if it_per_s >= 1.0:
                    speed_label = f"{it_per_s:.2f} it/s"
                else:
                    speed_label = f"{(1.0 / it_per_s):.2f} s/it"
                if train_total_batches is not None and train_total_batches > 0:
                    remaining_batches = max(train_total_batches - batch_idx, 0)
                    eta_s = remaining_batches / max(it_per_s, 1e-9)
                    eta_label = time.strftime("%H:%M:%S", time.gmtime(max(int(eta_s), 0)))
                else:
                    eta_label = "n/a"
                logger.info(
                    "Epoch %s train %s | dt=%s loss=%.4f speed=%s eta_epoch=%s all6_acc=%.2f%% slot_acc=%.2f%% factor_loss=%s chooser_loss=%s direct_safety_loss=%s policy_loss=%s",
                    epoch + 1,
                    progress_label,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    weighted_loss.item(),
                    speed_label,
                    eta_label,
                    100 * train_correct / max(train_total, 1),
                    100 * train_slot_correct / max(train_slot_total, 1),
                    f"{train_factor_loss_sum / max(train_factor_count, 1):.4f}" if factor_cfg.enabled else "n/a",
                    f"{train_chooser_loss_sum / max(train_chooser_count, 1):.4f}" if chooser_enabled else "n/a",
                    (
                        f"{train_direct_safety_loss_sum / max(train_direct_safety_count, 1):.4f}"
                        if direct_safety_enabled
                        else "n/a"
                    ),
                    f"{train_policy_loss_sum / max(train_policy_count, 1):.4f}" if policy_enabled else "n/a",
                )
            if timing_enabled and train_timing_window_count >= timing_log_every:
                logger.info(
                    "Epoch %s train timing (batch=%s, window=%s) | data=%.1fms forward=%.1fms base_loss=%.1fms chooser=%.1fms chooser_build=%.1fms chooser_score=%.1fms chooser_eval=%.1fms chooser_eval_constraints=%.1fms chooser_eval_math=%.1fms factor=%.1fms policy=%.1fms fix=%.1fms reweight=%.1fms backward=%.1fms optim=%.1fms metrics=%.1fms total=%.1fms",
                    epoch + 1,
                    batch_idx,
                    train_timing_window_count,
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "data_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "forward_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "base_loss_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "chooser_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "chooser_candidate_build_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "chooser_packed_score_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "chooser_evaluator_terms_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "chooser_eval_constraint_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "chooser_eval_math_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "factor_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "policy_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "fix_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "reweight_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "backward_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "optim_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "metrics_s"),
                    _timing_bucket_ms(train_timing_window, train_timing_window_count, "total_s"),
                )
                train_timing_window = _new_timing_bucket(train_timing_keys)
                train_timing_window_count = 0

        # Epoch training metrics
        avg_train_loss = train_graph_loss_sum / max(train_graph_count, 1)
        train_acc_all6 = 100 * train_correct / max(train_total, 1)
        train_acc_slot = 100 * train_slot_correct / max(train_slot_total, 1)
        train_slot_avg_loss = [train_slot_loss_sums[idx] / max(train_slot_counts[idx], 1) for idx in range(NUM_SLOTS)]
        train_slot_acc_per_slot = [
            100 * train_slot_correct_slots[idx] / max(train_slot_counts[idx], 1) for idx in range(NUM_SLOTS)
        ]
        train_per_constraint = train_constraint_metrics.as_epoch_metrics()
        train_factor_loss = train_factor_loss_sum / max(train_factor_count, 1)
        train_factor_acc = 100 * train_factor_correct / max(train_factor_count, 1)
        train_chooser_loss = train_chooser_loss_sum / max(train_chooser_count, 1)
        train_direct_safety_loss = train_direct_safety_loss_sum / max(train_direct_safety_count, 1)
        train_policy_loss = train_policy_loss_sum / max(train_policy_count, 1)
        train_policy_acc = 100 * train_policy_correct / max(train_policy_count, 1)
        train_grad_norm_mean = train_grad_norm_sum / max(train_grad_norm_count, 1)
        train_param_norm, train_param_abs_max = _parameter_norm_stats(model)
        _assert_finite_scalar("parameter_norm", train_param_norm, epoch=epoch + 1)
        _assert_finite_scalar("parameter_abs_max", train_param_abs_max, epoch=epoch + 1)
        if timing_enabled and train_timing_epoch_count > 0:
            logger.info(
                "Epoch %s train timing summary (%s batches, warmup=%s) | data=%.1fms forward=%.1fms base_loss=%.1fms chooser=%.1fms chooser_build=%.1fms chooser_score=%.1fms chooser_eval=%.1fms chooser_eval_constraints=%.1fms chooser_eval_math=%.1fms factor=%.1fms policy=%.1fms fix=%.1fms reweight=%.1fms backward=%.1fms optim=%.1fms metrics=%.1fms total=%.1fms",
                epoch + 1,
                train_timing_epoch_count,
                timing_warmup_batches,
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "data_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "forward_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "base_loss_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "chooser_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "chooser_candidate_build_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "chooser_packed_score_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "chooser_evaluator_terms_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "chooser_eval_constraint_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "chooser_eval_math_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "factor_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "policy_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "fix_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "reweight_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "backward_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "optim_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "metrics_s"),
                _timing_bucket_ms(train_timing_epoch, train_timing_epoch_count, "total_s"),
            )

        # ------- Validation -------
        model.eval()
        val_constraint_metrics = ConstraintMetricsAccumulator()
        val_graph_loss_sum = 0.0  # combined validation loss
        val_graph_count = 0  # graphs processed
        val_slots = 0  # total number of slots processed
        val_correct = 0  # all-6 correct predictions
        val_total = 0  # total number of graphs
        val_slot_correct = 0  # per-slot correct predictions
        val_slot_total = 0  # total number of slots processed
        val_slot_loss_sums = [0.0] * NUM_SLOTS
        val_slot_counts = [0] * NUM_SLOTS
        val_slot_correct_slots = [0] * NUM_SLOTS
        val_factor_loss_sum = 0.0
        val_factor_count = 0
        val_factor_correct = 0
        val_chooser_loss_sum = 0.0
        val_chooser_count = 0
        val_direct_safety_loss_sum = 0.0
        val_direct_safety_count = 0
        val_policy_loss_sum = 0.0
        val_policy_count = 0
        val_policy_correct = 0
        val_edit_logit_abs_max = 0.0
        val_factor_logit_abs_max = 0.0
        val_chooser_score_abs_max = 0.0
        val_epoch_start = time.perf_counter()
        val_timing_window = _new_timing_bucket(val_timing_keys)
        val_timing_epoch = _new_timing_bucket(val_timing_keys)
        val_timing_window_count = 0
        val_timing_epoch_count = 0

        with torch.no_grad():
            val_last_logged_percent = 0
            for batch_idx, (data, data_on_device) in enumerate(_iter_batches(val_loader), start=1):
                batch_t0 = time.perf_counter() if timing_enabled else 0.0
                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                if not data_on_device:
                    data = data.to(device, non_blocking=True)
                _assert_factor_type_ids_supported(data, int(getattr(model, "_num_factor_types", 0)))
                if train_cfg.validate_factor_labels:
                    _assert_factor_labels_batch(data)
                targets = data.y.long()
                _assert_targets_supported_by_heads(targets, split="val", batch_idx=batch_idx)
                data_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    outputs = model(data)
                forward_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0
                if torch.is_tensor(outputs):
                    outputs = {
                        "edit_logits": outputs,
                        "node_emb": None,
                        "graph_emb": None,
                    }
                out = outputs.get("edit_logits") if isinstance(outputs, dict) else None
                assert out is not None, "Model output must include 'edit_logits'."
                assert out.dim() == 3 and out.size(1) == NUM_SLOTS and out.size(2) == model.num_target_ids, (
                    f"Expected out (B,{NUM_SLOTS},{model.num_target_ids}), got {tuple(out.shape)}"
                )
                val_edit_logit_abs_max = max(val_edit_logit_abs_max, _max_abs_value(out))

                out_flat = out.reshape(-1, out.size(-1))
                targets_flat = targets.reshape(-1)

                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                per_slot_loss = criterion(out_flat, targets_flat)
                loss_matrix = per_slot_loss.view(-1, NUM_SLOTS)
                graph_loss = loss_matrix.mean(dim=1)
                base_loss_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0
                policy_s = 0.0
                fix_s = 0.0
                factor_s = 0.0
                chooser_s = 0.0
                chooser_candidate_build_s = 0.0
                chooser_packed_score_s = 0.0
                chooser_evaluator_terms_s = 0.0
                chooser_eval_constraint_s = 0.0
                chooser_eval_math_s = 0.0

                if policy_enabled:
                    phase_t0 = time.perf_counter() if timing_enabled else 0.0
                    policy_logits = outputs.get("policy_logits")
                    if policy_logits is None:
                        raise RuntimeError("Policy choice enabled but model output missing policy_logits.")
                    if policy_val_contexts is None:
                        raise RuntimeError("Policy choice enabled but validation contexts are missing.")
                    batch_context_indices = _batch_int_attr(
                        data,
                        "context_index",
                        graph_loss.size(0),
                        default_to_batch_index=True,
                        min_value=0,
                    )
                    targets_rows = targets.detach().cpu().tolist()
                    policy_labels = []
                    for idx, context_index in enumerate(batch_context_indices):
                        if context_index < 0 or context_index >= len(policy_val_contexts):
                            raise RuntimeError("Policy context index out of bounds.")
                        context = policy_val_contexts[context_index]
                        decision = derive_policy_label(targets_rows[idx], context, none_class=NONE_CLASS_INDEX)
                        policy_labels.append(int(decision.policy_id))
                    policy_targets = torch.tensor(policy_labels, dtype=torch.long, device=graph_loss.device)
                    policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets, reduction="none")
                    graph_loss = graph_loss + policy_loss
                    val_policy_loss_sum += float(policy_loss.sum().item())
                    val_policy_count += policy_loss.numel()
                    val_policy_correct += int((policy_logits.argmax(dim=-1) == policy_targets).sum().item())
                    policy_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

                # Optional - Constraint-is-fixed loss term
                if (
                    fix_scheduler
                    and fix_scheduler.enabled
                    and fix_heuristics is not None
                    and val_contexts is not None
                ):
                    phase_t0 = time.perf_counter() if timing_enabled else 0.0
                    context_attr = getattr(data, "context_index", None)
                    if context_attr is not None:
                        context_indices = _batch_int_attr(
                            data,
                            "context_index",
                            graph_loss.size(0),
                            default_to_batch_index=True,
                        )
                        if any(idx < 0 or idx >= len(val_contexts) for idx in context_indices):
                            raise RuntimeError("Fix-loss context index out of bounds.")
                        batch_contexts = [val_contexts[idx] for idx in context_indices]
                        progress = epoch + 1.0
                        fix_weight = fix_scheduler.weight_for_progress(progress)
                        if fix_weight > 0.0:
                            fix_probs = compute_fix_probabilities(out, batch_contexts, fix_heuristics)
                            graph_loss = graph_loss + fix_weight * (1.0 - fix_probs)
                    fix_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

                # Optional - Factor satisfaction loss (pre-state)
                if factor_cfg.enabled:
                    phase_t0 = time.perf_counter() if timing_enabled else 0.0
                    assert factor_criterion is not None
                    factor_logits = outputs.get("factor_logits_pre")
                    factor_logits_post_gold = outputs.get("factor_logits_post_gold")
                    factor_mask = outputs.get("factor_mask_pre")
                    factor_graph_index = outputs.get("factor_graph_index")
                    factor_targets = getattr(data, "factor_satisfied_pre", None)
                    assert factor_targets is not None, "Missing factor_satisfied_pre for factor loss."
                    factor_targets = torch.as_tensor(
                        factor_targets, dtype=torch.float32, device=graph_loss.device
                    ).view(-1)
                    if factor_mask is None:
                        factor_mask = torch.ones_like(factor_targets, dtype=torch.bool)
                    else:
                        factor_mask = torch.as_tensor(
                            factor_mask, dtype=torch.bool, device=graph_loss.device
                        ).view(-1)
                    assert factor_logits is not None, "Model output missing factor_logits_pre."
                    factor_logits = torch.as_tensor(factor_logits, device=graph_loss.device).view(-1)
                    val_factor_logit_abs_max = max(val_factor_logit_abs_max, _max_abs_value(factor_logits))
                    assert factor_logits.numel() == factor_targets.numel(), (
                        "factor_logits_pre length must match factor_satisfied_pre length."
                    )
                    if factor_graph_index is None:
                        if graph_loss.numel() == 1:
                            factor_graph_index = torch.zeros(
                                factor_targets.numel(), device=graph_loss.device, dtype=torch.long
                            )
                        else:
                            raise AssertionError("factor_graph_index is required for batched factor loss.")
                    factor_graph_index = torch.as_tensor(factor_graph_index, device=graph_loss.device).view(-1)
                    if train_cfg.validate_factor_labels:
                        _assert_factor_logit_alignment(data, factor_logits, factor_graph_index)
                    if factor_cfg.only_checkable:
                        checkable = getattr(data, "factor_checkable_pre", None)
                        if checkable is None:
                            active_mask = factor_mask
                        else:
                            checkable = torch.as_tensor(
                                checkable, dtype=torch.bool, device=graph_loss.device
                            ).view(-1)
                            if checkable.numel() != factor_targets.numel():
                                raise AssertionError(
                                    "factor_checkable_pre length must match factor_satisfied_pre length."
                                )
                            active_mask = factor_mask & checkable
                    else:
                        active_mask = torch.ones_like(factor_mask, dtype=torch.bool)
                    if active_mask.numel() != factor_targets.numel():
                        raise AssertionError("factor_mask_pre length must match factor_satisfied_pre length.")
                    _log_factor_debug(data, factor_logits, factor_graph_index)

                    graph_loss_add = torch.zeros(
                        graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                    )
                    factor_loss_total = 0.0
                    factor_count_total = 0
                    factor_correct_total = 0

                    active_float = active_mask.to(dtype=graph_loss.dtype)
                    per_factor_loss = factor_criterion(factor_logits, factor_targets).to(dtype=graph_loss.dtype)
                    per_factor_loss = per_factor_loss * active_float
                    active_count = int(active_mask.sum().item())
                    if factor_cfg.weight_pre > 0.0:
                        pre_graph_add = torch.zeros(
                            graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                        )
                        pre_graph_add.scatter_add_(0, factor_graph_index, per_factor_loss)
                        if factor_cfg.per_graph_reduction == "mean":
                            per_graph_counts = torch.zeros(
                                graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                            )
                            per_graph_counts.scatter_add_(0, factor_graph_index, active_float)
                            pre_graph_add = pre_graph_add / per_graph_counts.clamp(min=1.0)
                        graph_loss_add = graph_loss_add + (factor_cfg.weight_pre * pre_graph_add)
                    factor_loss_total += float(per_factor_loss.sum().item())
                    factor_count_total += active_count
                    if active_count > 0:
                        preds = (factor_logits > 0).to(dtype=torch.long)
                        targets_long = factor_targets.to(dtype=torch.long)
                        factor_correct_total += int((preds[active_mask] == targets_long[active_mask]).sum().item())

                    if factor_logits_post_gold is not None and factor_cfg.weight_post_gold > 0.0:
                        post_targets = getattr(data, "factor_satisfied_post_gold", None)
                        assert post_targets is not None, "Missing factor_satisfied_post_gold for factor post-gold loss."
                        post_targets = torch.as_tensor(
                            post_targets, dtype=torch.float32, device=graph_loss.device
                        ).view(-1)
                        factor_logits_post_gold = torch.as_tensor(
                            factor_logits_post_gold, device=graph_loss.device
                        ).view(-1)
                        val_factor_logit_abs_max = max(
                            val_factor_logit_abs_max,
                            _max_abs_value(factor_logits_post_gold),
                        )
                        if factor_logits_post_gold.numel() != post_targets.numel():
                            raise AssertionError(
                                "factor_logits_post_gold length must match factor_satisfied_post_gold length."
                            )
                        if factor_cfg.only_checkable:
                            post_checkable = getattr(data, "factor_checkable_post_gold", None)
                            if post_checkable is None:
                                post_active_mask = torch.ones_like(post_targets, dtype=torch.bool)
                            else:
                                post_active_mask = torch.as_tensor(
                                    post_checkable, dtype=torch.bool, device=graph_loss.device
                                ).view(-1)
                                if post_active_mask.numel() != post_targets.numel():
                                    raise AssertionError(
                                        "factor_checkable_post_gold length must match factor_satisfied_post_gold length."
                                    )
                        else:
                            post_active_mask = torch.ones_like(post_targets, dtype=torch.bool)
                        post_active_float = post_active_mask.to(dtype=graph_loss.dtype)
                        per_factor_post_loss = factor_criterion(
                            factor_logits_post_gold, post_targets
                        ).to(dtype=graph_loss.dtype)
                        per_factor_post_loss = per_factor_post_loss * post_active_float
                        post_graph_add = torch.zeros(
                            graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                        )
                        post_graph_add.scatter_add_(0, factor_graph_index, per_factor_post_loss)
                        if factor_cfg.per_graph_reduction == "mean":
                            post_graph_counts = torch.zeros(
                                graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                            )
                            post_graph_counts.scatter_add_(0, factor_graph_index, post_active_float)
                            post_graph_add = post_graph_add / post_graph_counts.clamp(min=1.0)
                        graph_loss_add = graph_loss_add + (factor_cfg.weight_post_gold * post_graph_add)
                        post_active_count = int(post_active_mask.sum().item())
                        factor_loss_total += float(per_factor_post_loss.sum().item())
                        factor_count_total += post_active_count
                        if post_active_count > 0:
                            post_preds = (factor_logits_post_gold > 0).to(dtype=torch.long)
                            post_targets_long = post_targets.to(dtype=torch.long)
                            factor_correct_total += int(
                                (post_preds[post_active_mask] == post_targets_long[post_active_mask]).sum().item()
                            )

                    graph_loss = graph_loss + graph_loss_add

                    val_factor_loss_sum += factor_loss_total
                    val_factor_count += factor_count_total
                    val_factor_correct += factor_correct_total
                    factor_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

                if chooser_enabled:
                    phase_t0 = time.perf_counter() if timing_enabled else 0.0
                    if chooser_evaluator is None or chooser_heuristics is None:
                        raise RuntimeError("Chooser enabled but evaluator/heuristics are not initialized.")
                    if chooser_candidate_cfg is None:
                        chooser_candidate_cfg = CandidateConfig(
                            topk_candidates=chooser_cfg.topk_candidates,
                            max_candidates_total=chooser_cfg.max_candidates_total,
                        )
                    graph_emb = outputs.get("graph_emb")
                    if graph_emb is None:
                        raise RuntimeError("Model output missing graph_emb required for chooser.")
                    if chooser_val_contexts is None or chooser_val_rows is None:
                        raise RuntimeError("Chooser enabled but validation contexts/rows are missing.")
                    batch_context_indices = _batch_int_attr(
                        data,
                        "context_index",
                        graph_loss.size(0),
                        default_to_batch_index=True,
                        min_value=0,
                    )
                    batch_primary_indices = _batch_int_attr(
                        data,
                        "primary_factor_index",
                        graph_loss.size(0),
                        default_value=0,
                        min_value=0,
                    )
                    gold_rows = targets.detach().cpu().tolist()
                    out_detached = out.detach()
                    batch_slot_vals, batch_slot_ids = _compute_batch_slot_topk(
                        out_detached,
                        topk_per_slot=chooser_candidate_cfg.topk_per_slot,
                        slot_allowed_ids=chooser_slot_allowed_ids,
                    )
                    chooser_losses = torch.zeros(
                        graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                    )
                    candidate_groups: list[list[tuple[int, int, int, int, int, int]]] = []
                    gold_indices: list[int] = []
                    candidate_rows: list[Any] = []
                    primary_indices: list[int] = []
                    packed_candidates: list[tuple[int, int, int, int, int, int]] = []
                    packed_graph_index: list[int] = []
                    chooser_build_t0 = time.perf_counter() if timing_enabled else 0.0
                    for idx, context_index in enumerate(batch_context_indices):
                        if (
                            context_index < 0
                            or context_index >= len(chooser_val_contexts)
                            or context_index >= len(chooser_val_rows)
                        ):
                            raise RuntimeError("Chooser context index out of bounds.")
                        context = chooser_val_contexts[context_index]
                        row = chooser_val_rows[context_index]
                        primary_index = int(batch_primary_indices[idx])
                        add_topk = _topk_triples_from_slot_topk_row(
                            batch_slot_vals,
                            batch_slot_ids,
                            row_index=idx,
                            slots=(0, 1, 2),
                            topk_triples=chooser_candidate_cfg.topk_candidates,
                        )
                        del_topk = _topk_triples_from_slot_topk_row(
                            batch_slot_vals,
                            batch_slot_ids,
                            row_index=idx,
                            slots=(3, 4, 5),
                            topk_triples=chooser_candidate_cfg.topk_candidates,
                        )
                        candidates, gold_index = build_candidates(
                            gold_slots=gold_rows[idx],
                            context=context,
                            heuristics=chooser_heuristics,
                            proposal_logits=out_detached[idx],
                            cfg=chooser_candidate_cfg,
                            placeholder_ids=chooser_placeholder_ids_for_candidates,
                            num_target_ids=model.num_target_ids,
                            slot_allowed_ids=chooser_slot_allowed_ids,
                            precomputed_add_topk=add_topk,
                            precomputed_del_topk=del_topk,
                        )
                        if gold_index is None:
                            raise RuntimeError("Chooser validation requires include_gold=True.")
                        candidate_groups.append(candidates)
                        gold_indices.append(gold_index)
                        candidate_rows.append(row)
                        primary_indices.append(primary_index)
                        packed_candidates.extend(candidates)
                        packed_graph_index.extend([idx] * len(candidates))
                    chooser_candidate_build_s = (time.perf_counter() - chooser_build_t0) if timing_enabled else 0.0

                    if not packed_candidates:
                        raise RuntimeError("Chooser produced no candidates for the current validation batch.")

                    chooser_score_t0 = time.perf_counter() if timing_enabled else 0.0
                    packed_candidate_tensor = torch.tensor(
                        packed_candidates, dtype=torch.long, device=graph_loss.device
                    )
                    packed_graph_index_tensor = torch.tensor(
                        packed_graph_index, dtype=torch.long, device=graph_loss.device
                    )
                    packed_scores = model.score_candidates_packed(
                        graph_emb, packed_candidate_tensor, packed_graph_index_tensor
                    )
                    val_chooser_score_abs_max = max(val_chooser_score_abs_max, _max_abs_value(packed_scores))
                    chooser_packed_score_s = (time.perf_counter() - chooser_score_t0) if timing_enabled else 0.0

                    chooser_eval_t0 = time.perf_counter() if timing_enabled else 0.0
                    offset = 0
                    for idx, candidates in enumerate(candidate_groups):
                        candidate_count = len(candidates)
                        scores = packed_scores[offset : offset + candidate_count]
                        offset += candidate_count
                        row = candidate_rows[idx]
                        gold_index = gold_indices[idx]
                        primary_index = primary_indices[idx]
                        math_t0 = time.perf_counter() if timing_enabled else 0.0
                        log_probs = F.log_softmax(scores, dim=0)
                        probs = log_probs.exp()
                        chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                        if chooser_loss_mode == "global_fix":
                            constraint_t0 = time.perf_counter() if timing_enabled else 0.0
                            metrics_summary = chooser_evaluator.evaluate_candidate_metrics(
                                row,
                                candidates=candidates,
                                primary_factor_index=primary_index,
                            )
                            chooser_eval_constraint_s += (time.perf_counter() - constraint_t0) if timing_enabled else 0.0
                            math_t0 = time.perf_counter() if timing_enabled else 0.0
                            satisfaction = torch.tensor(
                                [m.global_satisfied_fraction for m in metrics_summary],
                                dtype=graph_loss.dtype,
                                device=graph_loss.device,
                            )
                            chooser_loss = -torch.sum(probs * satisfaction)
                            chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                        else:
                            math_t0 = time.perf_counter() if timing_enabled else 0.0
                            ce_loss = -log_probs[gold_index]
                            chooser_loss = ce_loss
                            chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                            if chooser_need_regression or chooser_need_primary:
                                constraint_t0 = time.perf_counter() if timing_enabled else 0.0
                                metrics_summary = chooser_evaluator.evaluate_candidate_metrics(
                                    row,
                                    candidates=candidates,
                                    primary_factor_index=primary_index,
                                )
                                chooser_eval_constraint_s += (time.perf_counter() - constraint_t0) if timing_enabled else 0.0
                            else:
                                metrics_summary = []
                            if chooser_need_regression and metrics_summary:
                                math_t0 = time.perf_counter() if timing_enabled else 0.0
                                regression_tensor = torch.tensor(
                                    [m.srr for m in metrics_summary],
                                    dtype=graph_loss.dtype,
                                    device=graph_loss.device,
                                )
                                gold_regression = regression_tensor[gold_index]
                                reg_penalty = torch.sum(
                                    probs * torch.clamp(regression_tensor - gold_regression, min=0.0)
                                )
                                chooser_loss = chooser_loss + chooser_cfg.beta_no_regression * reg_penalty
                                chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                            if chooser_need_primary and metrics_summary:
                                math_t0 = time.perf_counter() if timing_enabled else 0.0
                                primary_tensor = torch.tensor(
                                    [float(m.primary_satisfied) for m in metrics_summary],
                                    dtype=graph_loss.dtype,
                                    device=graph_loss.device,
                                )
                                primary_penalty = torch.sum(probs * (1.0 - primary_tensor))
                                chooser_loss = chooser_loss + chooser_cfg.gamma_primary * primary_penalty
                                chooser_eval_math_s += (time.perf_counter() - math_t0) if timing_enabled else 0.0
                        chooser_losses[idx] = chooser_loss
                    chooser_evaluator_terms_s = (time.perf_counter() - chooser_eval_t0) if timing_enabled else 0.0
                    assert offset == packed_scores.numel(), "Packed chooser score size mismatch."

                    graph_loss = graph_loss + (chooser_loss_weight * chooser_losses)
                    val_chooser_loss_sum += float(chooser_losses.sum().item())
                    val_chooser_count += chooser_losses.numel()
                    chooser_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

                if direct_safety_enabled:
                    phase_t0 = time.perf_counter() if timing_enabled else 0.0
                    if direct_safety_evaluator is None or direct_safety_heuristics is None:
                        raise RuntimeError("Direct safety enabled but evaluator/heuristics are not initialized.")
                    if direct_safety_candidate_cfg is None:
                        direct_safety_candidate_cfg = CandidateConfig(
                            topk_candidates=direct_safety_cfg.topk_candidates,
                            max_candidates_total=direct_safety_cfg.max_candidates_total,
                        )
                    if direct_safety_val_contexts is None or direct_safety_val_rows is None:
                        raise RuntimeError("Direct safety enabled but validation contexts/rows are missing.")
                    batch_context_indices = _batch_int_attr(
                        data,
                        "context_index",
                        graph_loss.size(0),
                        default_to_batch_index=True,
                        min_value=0,
                    )
                    batch_primary_indices = _batch_int_attr(
                        data,
                        "primary_factor_index",
                        graph_loss.size(0),
                        default_value=0,
                        min_value=0,
                    )
                    out_detached = out.detach()
                    batch_slot_vals, batch_slot_ids = _compute_batch_slot_topk(
                        out_detached,
                        topk_per_slot=direct_safety_candidate_cfg.topk_per_slot,
                        slot_allowed_ids=direct_safety_slot_allowed_ids,
                    )
                    direct_safety_losses = torch.zeros(
                        graph_loss.size(0), device=graph_loss.device, dtype=graph_loss.dtype
                    )
                    for idx, context_index in enumerate(batch_context_indices):
                        if (
                            context_index < 0
                            or context_index >= len(direct_safety_val_contexts)
                            or context_index >= len(direct_safety_val_rows)
                        ):
                            raise RuntimeError("Direct safety context index out of bounds.")
                        context = direct_safety_val_contexts[context_index]
                        row = direct_safety_val_rows[context_index]
                        primary_index = int(batch_primary_indices[idx])
                        add_topk = _topk_triples_from_slot_topk_row(
                            batch_slot_vals,
                            batch_slot_ids,
                            row_index=idx,
                            slots=(0, 1, 2),
                            topk_triples=direct_safety_candidate_cfg.topk_candidates,
                        )
                        del_topk = _topk_triples_from_slot_topk_row(
                            batch_slot_vals,
                            batch_slot_ids,
                            row_index=idx,
                            slots=(3, 4, 5),
                            topk_triples=direct_safety_candidate_cfg.topk_candidates,
                        )
                        candidates, _ = build_candidates(
                            gold_slots=targets[idx].detach().cpu().tolist(),
                            context=context,
                            heuristics=direct_safety_heuristics,
                            proposal_logits=out_detached[idx],
                            cfg=direct_safety_candidate_cfg,
                            placeholder_ids=direct_safety_placeholder_ids_for_candidates,
                            num_target_ids=model.num_target_ids,
                            slot_allowed_ids=direct_safety_slot_allowed_ids,
                            precomputed_add_topk=add_topk,
                            precomputed_del_topk=del_topk,
                        )
                        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=graph_loss.device)
                        scores = score_candidates_from_logits(out[idx], candidate_tensor)
                        log_probs = F.log_softmax(scores, dim=0)
                        probs = log_probs.exp()
                        metrics_summary = direct_safety_evaluator.evaluate_candidate_metrics(
                            row,
                            candidates=candidates,
                            primary_factor_index=primary_index,
                        )
                        primary_tensor = torch.tensor(
                            [float(m.primary_satisfied) for m in metrics_summary],
                            dtype=graph_loss.dtype,
                            device=graph_loss.device,
                        )
                        secondary_tensor = torch.tensor(
                            [m.srr for m in metrics_summary],
                            dtype=graph_loss.dtype,
                            device=graph_loss.device,
                        )
                        primary_penalty = torch.sum(probs * (1.0 - primary_tensor))
                        secondary_penalty = torch.sum(probs * secondary_tensor)
                        direct_safety_losses[idx] = (
                            direct_safety_cfg.alpha_primary * primary_penalty
                            + direct_safety_cfg.beta_secondary * secondary_penalty
                        )
                    graph_loss = graph_loss + direct_safety_losses
                    val_direct_safety_loss_sum += float(direct_safety_losses.sum().item())
                    val_direct_safety_count += direct_safety_losses.numel()

                _assert_finite_scalar("validation_graph_loss_mean", graph_loss.mean(), epoch=epoch + 1, batch_idx=batch_idx)

                # Accumulate validation metrics
                phase_t0 = time.perf_counter() if timing_enabled else 0.0
                val_graph_loss_sum += graph_loss.sum().item()
                val_graph_count += graph_loss.numel()
                val_slots += targets_flat.numel()

                _, predicted = torch.max(out_flat, 1)
                predicted_reshaped = predicted.reshape(-1, NUM_SLOTS)
                targets_reshaped = targets_flat.reshape(-1, NUM_SLOTS)

                all_correct = (predicted_reshaped == targets_reshaped).all(dim=1)
                val_correct += all_correct.sum().item()
                val_total += all_correct.size(0)

                val_slot_correct += (predicted == targets_flat).sum().item()
                val_slot_total += targets_flat.numel()

                slot_loss_sums = loss_matrix.sum(dim=0).detach().cpu().tolist()
                batch_graphs = loss_matrix.size(0)
                slot_correct_matrix = (predicted_reshaped == targets_reshaped).to(dtype=torch.long)
                slot_correct_counts = slot_correct_matrix.sum(dim=0).detach().cpu().tolist()
                for idx in range(NUM_SLOTS):
                    val_slot_loss_sums[idx] += slot_loss_sums[idx]
                    val_slot_counts[idx] += batch_graphs
                    val_slot_correct_slots[idx] += int(slot_correct_counts[idx])

                loss_per_graph = loss_matrix.detach().cpu().sum(dim=1).tolist()
                slot_correct_per_graph = slot_correct_matrix.sum(dim=1).detach().cpu().tolist()
                all_correct_per_graph = all_correct.to(dtype=torch.long).detach().cpu().tolist()
                constraint_types = extract_constraint_types(data, len(loss_per_graph))
                val_constraint_metrics.update(
                    constraint_types,
                    loss_per_graph,
                    slot_correct_per_graph,
                    all_correct_per_graph,
                    NUM_SLOTS,
                )
                metrics_s = (time.perf_counter() - phase_t0) if timing_enabled else 0.0

                if timing_enabled and batch_idx > timing_warmup_batches:
                    timing_snapshot = {
                        "data_s": data_s,
                        "forward_s": forward_s,
                        "base_loss_s": base_loss_s,
                        "policy_s": policy_s,
                        "fix_s": fix_s,
                        "factor_s": factor_s,
                        "chooser_s": chooser_s,
                        "chooser_candidate_build_s": chooser_candidate_build_s,
                        "chooser_packed_score_s": chooser_packed_score_s,
                        "chooser_evaluator_terms_s": chooser_evaluator_terms_s,
                        "chooser_eval_constraint_s": chooser_eval_constraint_s,
                        "chooser_eval_math_s": chooser_eval_math_s,
                        "metrics_s": metrics_s,
                        "total_s": time.perf_counter() - batch_t0,
                    }
                    for key, value in timing_snapshot.items():
                        val_timing_window[key] += value
                        val_timing_epoch[key] += value
                    val_timing_window_count += 1
                    val_timing_epoch_count += 1

                if logger.isEnabledFor(logging.DEBUG):
                    if batch_idx == 1:
                        logger.debug(
                            "First validation batch stats: num_graphs=%s num_nodes=%s num_edges=%s",
                            getattr(data, "num_graphs", "?"),
                            getattr(data, "num_nodes", "?"),
                            getattr(data, "num_edges", "?"),
                        )
                    if device.type == "cuda" and batch_idx % 5 == 0:
                        log_cuda_memory(
                            f"Epoch {epoch + 1} validation batch {batch_idx}",
                            device,
                            level=logging.DEBUG,
                        )

                should_log, val_last_logged_percent = _should_log_batch(
                    batch_idx,
                    val_total_batches,
                    val_last_logged_percent,
                )
                if should_log:
                    if val_total_batches is not None and val_total_batches > 0:
                        progress = int((batch_idx * 100) / val_total_batches)
                        progress_label = (
                            f"batch {batch_idx}/{val_total_batches} ({min(progress, 100)}%)"
                        )
                    else:
                        progress_label = f"batch {batch_idx}"
                    avg_val_batch_loss = val_graph_loss_sum / max(val_graph_count, 1)
                    elapsed_s = max(time.perf_counter() - val_epoch_start, 1e-9)
                    it_per_s = batch_idx / elapsed_s
                    if it_per_s >= 1.0:
                        speed_label = f"{it_per_s:.2f} it/s"
                    else:
                        speed_label = f"{(1.0 / it_per_s):.2f} s/it"
                    if val_total_batches is not None and val_total_batches > 0:
                        remaining_batches = max(val_total_batches - batch_idx, 0)
                        eta_s = remaining_batches / max(it_per_s, 1e-9)
                        eta_label = time.strftime("%H:%M:%S", time.gmtime(max(int(eta_s), 0)))
                    else:
                        eta_label = "n/a"
                    logger.info(
                        "Epoch %s val %s | dt=%s loss=%.4f speed=%s eta_epoch=%s all6_acc=%.2f%% slot_acc=%.2f%% factor_loss=%s chooser_loss=%s direct_safety_loss=%s policy_loss=%s",
                        epoch + 1,
                        progress_label,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        avg_val_batch_loss,
                        speed_label,
                        eta_label,
                        100 * val_correct / max(val_total, 1),
                        100 * val_slot_correct / max(val_slot_total, 1),
                        f"{val_factor_loss_sum / max(val_factor_count, 1):.4f}" if factor_cfg.enabled else "n/a",
                        f"{val_chooser_loss_sum / max(val_chooser_count, 1):.4f}" if chooser_enabled else "n/a",
                        (
                            f"{val_direct_safety_loss_sum / max(val_direct_safety_count, 1):.4f}"
                            if direct_safety_enabled
                            else "n/a"
                        ),
                        f"{val_policy_loss_sum / max(val_policy_count, 1):.4f}" if policy_enabled else "n/a",
                    )
                if timing_enabled and val_timing_window_count >= timing_log_every:
                    logger.info(
                        "Epoch %s val timing (batch=%s, window=%s) | data=%.1fms forward=%.1fms base_loss=%.1fms chooser=%.1fms chooser_build=%.1fms chooser_score=%.1fms chooser_eval=%.1fms chooser_eval_constraints=%.1fms chooser_eval_math=%.1fms factor=%.1fms policy=%.1fms fix=%.1fms metrics=%.1fms total=%.1fms",
                        epoch + 1,
                        batch_idx,
                        val_timing_window_count,
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "data_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "forward_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "base_loss_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "chooser_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "chooser_candidate_build_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "chooser_packed_score_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "chooser_evaluator_terms_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "chooser_eval_constraint_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "chooser_eval_math_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "factor_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "policy_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "fix_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "metrics_s"),
                        _timing_bucket_ms(val_timing_window, val_timing_window_count, "total_s"),
                    )
                    val_timing_window = _new_timing_bucket(val_timing_keys)
                    val_timing_window_count = 0

        # Epoch validation metrics
        avg_val_loss = val_graph_loss_sum / max(val_graph_count, 1)
        val_acc_all6 = 100 * val_correct / max(val_total, 1)
        val_acc_slot = 100 * val_slot_correct / max(val_slot_total, 1)
        val_slot_avg_loss = [val_slot_loss_sums[idx] / max(val_slot_counts[idx], 1) for idx in range(NUM_SLOTS)]
        val_slot_acc_per_slot = [
            100 * val_slot_correct_slots[idx] / max(val_slot_counts[idx], 1) for idx in range(NUM_SLOTS)
        ]
        val_per_constraint = val_constraint_metrics.as_epoch_metrics()
        val_factor_loss = val_factor_loss_sum / max(val_factor_count, 1)
        val_factor_acc = 100 * val_factor_correct / max(val_factor_count, 1)
        val_chooser_loss = val_chooser_loss_sum / max(val_chooser_count, 1)
        val_direct_safety_loss = val_direct_safety_loss_sum / max(val_direct_safety_count, 1)
        val_policy_loss = val_policy_loss_sum / max(val_policy_count, 1)
        val_policy_acc = 100 * val_policy_correct / max(val_policy_count, 1)
        if timing_enabled and val_timing_epoch_count > 0:
            logger.info(
                "Epoch %s val timing summary (%s batches, warmup=%s) | data=%.1fms forward=%.1fms base_loss=%.1fms chooser=%.1fms chooser_build=%.1fms chooser_score=%.1fms chooser_eval=%.1fms chooser_eval_constraints=%.1fms chooser_eval_math=%.1fms factor=%.1fms policy=%.1fms fix=%.1fms metrics=%.1fms total=%.1fms",
                epoch + 1,
                val_timing_epoch_count,
                timing_warmup_batches,
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "data_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "forward_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "base_loss_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "chooser_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "chooser_candidate_build_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "chooser_packed_score_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "chooser_evaluator_terms_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "chooser_eval_constraint_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "chooser_eval_math_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "factor_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "policy_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "fix_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "metrics_s"),
                _timing_bucket_ms(val_timing_epoch, val_timing_epoch_count, "total_s"),
            )

        logger.info(
            "Epoch %s samples | train=%s validation=%s",
            epoch + 1,
            train_total,
            val_total,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(avg_val_loss)

        # Snapshot aggregate metrics for this epoch.
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc_all6"].append(train_acc_all6)
        history["val_acc_all6"].append(val_acc_all6)
        history["train_acc_slot"].append(train_acc_slot)
        history["val_acc_slot"].append(val_acc_slot)
        history["learning_rate"].append(current_lr)
        history["train_grad_norm_mean"].append(train_grad_norm_mean)
        history["train_grad_norm_max"].append(train_grad_norm_max)
        history["train_param_norm"].append(train_param_norm)
        history["train_param_abs_max"].append(train_param_abs_max)
        history["train_edit_logit_abs_max"].append(train_edit_logit_abs_max)
        history["val_edit_logit_abs_max"].append(val_edit_logit_abs_max)
        history["train_factor_logit_abs_max"].append(train_factor_logit_abs_max)
        history["val_factor_logit_abs_max"].append(val_factor_logit_abs_max)
        history["train_chooser_score_abs_max"].append(train_chooser_score_abs_max)
        history["val_chooser_score_abs_max"].append(val_chooser_score_abs_max)
        history.setdefault("train_chooser_loss", []).append(train_chooser_loss)
        history.setdefault("val_chooser_loss", []).append(val_chooser_loss)
        history.setdefault("train_direct_safety_loss", []).append(train_direct_safety_loss)
        history.setdefault("val_direct_safety_loss", []).append(val_direct_safety_loss)
        history.setdefault("train_policy_loss", []).append(train_policy_loss)
        history.setdefault("val_policy_loss", []).append(val_policy_loss)
        history.setdefault("train_policy_acc", []).append(train_policy_acc)
        history.setdefault("val_policy_acc", []).append(val_policy_acc)

        # Record per-slot diagnostics (helps inspect slot-specific drift).
        per_slot_history = history.setdefault("per_slot", {})
        for slot_idx in range(NUM_SLOTS):
            slot_key = str(slot_idx)
            slot_entry = per_slot_history.setdefault(
                slot_key,
                {
                    "train_loss": [],
                    "val_loss": [],
                    "train_acc": [],
                    "val_acc": [],
                },
            )
            slot_entry["train_loss"].append(train_slot_avg_loss[slot_idx])
            slot_entry["val_loss"].append(val_slot_avg_loss[slot_idx])
            slot_entry["train_acc"].append(train_slot_acc_per_slot[slot_idx])
            slot_entry["val_acc"].append(val_slot_acc_per_slot[slot_idx])

        # Merge per-constraint metrics for later plotting / reweighting.
        per_constraint_history = history.setdefault("per_constraint", {})
        epoch_index = len(history["train_loss"]) - 1
        update_per_constraint_history(
            per_constraint_history,
            train_per_constraint,
            val_per_constraint,
            epoch_index,
        )
        dynamic_weighter.update_from_metrics(val_per_constraint)

        if train_per_constraint:
            top_train = sorted(train_per_constraint.items(), key=lambda item: item[1]["graph_count"], reverse=True)[
                :3
            ]
            logger.debug(
                "Top train constraint metrics: %s",
                "; ".join(
                    f"{name}: loss={metrics['loss']:.4f} acc_all6={metrics['acc_all6']:.2f}%"
                    f" acc_slot={metrics['acc_slot']:.2f}% (graphs={metrics['graph_count']})"
                    for name, metrics in top_train
                ),
            )
        if val_per_constraint:
            top_val = sorted(val_per_constraint.items(), key=lambda item: item[1]["graph_count"], reverse=True)[:3]
            logger.info(
                "Val constraint metrics: %s",
                "; ".join(
                    f"{name}: loss={metrics['loss']:.4f} acc_all6={metrics['acc_all6']:.2f}%"
                    f" acc_slot={metrics['acc_slot']:.2f}% (graphs={metrics['graph_count']})"
                    for name, metrics in top_val
                ),
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_rounds:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        epoch_iter.set_postfix(
            {
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "train_all6": f"{train_acc_all6:.2f}%",
                "val_all6": f"{val_acc_all6:.2f}%",
                "train_slot": f"{train_acc_slot:.2f}%",
                "val_slot": f"{val_acc_slot:.2f}%",
                "best_val": f"{best_val_loss:.4f}",
            }
        )

        logger.info(
            "Epoch %s summary | train_loss=%.4f val_loss=%.4f train_all6=%.2f%% val_all6=%.2f%%"
            " train_slot=%.2f%% val_slot=%.2f%%",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            train_acc_all6,
            val_acc_all6,
            train_acc_slot,
            val_acc_slot,
        )
        logger.info(
            "Epoch %s stability | lr=%.3g grad_norm_mean=%.4f grad_norm_max=%.4f "
            "param_norm=%.4f param_abs_max=%.4f edit_logit_abs_max=train %.4f/val %.4f "
            "factor_logit_abs_max=train %.4f/val %.4f chooser_score_abs_max=train %.4f/val %.4f",
            epoch + 1,
            current_lr,
            train_grad_norm_mean,
            train_grad_norm_max,
            train_param_norm,
            train_param_abs_max,
            train_edit_logit_abs_max,
            val_edit_logit_abs_max,
            train_factor_logit_abs_max,
            val_factor_logit_abs_max,
            train_chooser_score_abs_max,
            val_chooser_score_abs_max,
        )
        if factor_cfg.enabled:
            logger.info(
                "Epoch %s factor metrics | train_loss=%.4f val_loss=%.4f train_acc=%.2f%% val_acc=%.2f%%",
                epoch + 1,
                train_factor_loss,
                val_factor_loss,
                train_factor_acc,
                val_factor_acc,
            )
        if direct_safety_enabled:
            logger.info(
                "Epoch %s direct safety metrics | train_loss=%.4f val_loss=%.4f",
                epoch + 1,
                train_direct_safety_loss,
                val_direct_safety_loss,
            )

        if device.type == "cuda":
            log_cuda_memory(f"Epoch {epoch + 1} post-epoch", device)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    if device.type == "cuda":
        log_cuda_memory("Training complete", device)
        torch.cuda.empty_cache()

    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the graphs dataset.")

    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        required=True,
        help="JSON file with model hyperparameters.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Verbosity level for logging output.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    return args


def _write_effective_experiment_config(
    config_path: Path,
    original_payload: dict[str, Any],
    *,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
) -> None:
    payload = dict(original_payload)
    payload["model_config"] = model_cfg.to_dict()
    payload["training_config"] = training_cfg.to_dict()
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main():
    args = parse_args()

    # Load experiment configuration (model + training sections).
    config_path = Path(args.experiment_config)
    with config_path.open("r", encoding="utf-8") as f:
        experiment_config = json.load(f)
    model_cfg = ModelConfig.from_mapping(experiment_config["model_config"])
    training_cfg = TrainingConfig.from_mapping(experiment_config["training_config"])
    set_seed(training_cfg.seed)

    # Prepare run directory
    run_directory = ensure_run_dir_for_config(config_path)
    logger.info("Artifacts will be stored in %s", run_directory)

    logger.info(
        "Starting experiment | dataset=%s min_occurrance=%s encoding=%s model=%s log_level=%s",
        model_cfg.dataset_variant,
        model_cfg.min_occurrence,
        model_cfg.encoding,
        model_cfg.model,
        args.log_level,
    )

    # Dataset paths
    dataset_variant = model_cfg.dataset_variant
    if "_minocc" not in dataset_variant:
        dataset_variant = dataset_variant_name(model_cfg.dataset_variant, model_cfg.min_occurrence)
    path = Path("data/processed") / dataset_variant
    logger.debug("Resolved dataset base path to %s", path)

    train_data_path = path / graph_dataset_filename(
        "train",
        model_cfg.encoding,
        constraint_representation=model_cfg.constraint_representation,
    )
    val_data_path = path / graph_dataset_filename(
        "val",
        model_cfg.encoding,
        constraint_representation=model_cfg.constraint_representation,
    )
    train_data = load_graph_dataset(train_data_path)
    val_data = load_graph_dataset(val_data_path)

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
        if isinstance(graph_count, int):
            return graph_count
        return None

    # Infer node feature spec and target vocabularies
    use_node_embeddings, feature_dim, feature_dtype, role_spec = infer_node_feature_spec(train_data, val_data)
    logger.info(
        "Detected node features | use_node_embeddings=%s feature_dim=%s dtype=%s role_flags=%s (types=%s)",
        use_node_embeddings,
        feature_dim,
        feature_dtype,
        "enabled" if role_spec.enabled else "absent",
        role_spec.num_types if role_spec.enabled else 0,
    )

    # Target vocabularies for distinct entity and predicate predictions
    precomputed_targets = load_precomputed_target_vocabs(path, splits=("train", "val"))
    if precomputed_targets is not None:
        entity_class_ids, predicate_class_ids = precomputed_targets
    else:
        entity_class_ids, predicate_class_ids = derive_target_class_ids(train_data, val_data)

    model_cfg = model_cfg.updated(
        use_node_embeddings=use_node_embeddings,
        entity_class_ids=entity_class_ids,
        predicate_class_ids=predicate_class_ids,
        use_role_embeddings=role_spec.enabled,
        num_role_types=role_spec.num_types if role_spec.enabled else 0,
        num_embedding_size=feature_dim if not use_node_embeddings else model_cfg.num_embedding_size,
    )
    if model_cfg.constraint_representation == "factorized":
        registry_factor_types = _registry_factor_type_count(model_cfg.dataset_variant)

        # Avoid scanning all graphs when num_factor_types is already explicit in config.
        if model_cfg.num_factor_types > 0:
            num_factor_types = int(model_cfg.num_factor_types)
            if registry_factor_types > num_factor_types:
                logger.warning(
                    "Config num_factor_types=%s is below registry-derived count=%s; using registry value instead.",
                    num_factor_types,
                    registry_factor_types,
                )
                num_factor_types = registry_factor_types
                model_cfg = model_cfg.updated(num_factor_types=num_factor_types)
            logger.info("Using preconfigured num_factor_types=%s (skipping dataset scan).", num_factor_types)
        else:
            num_factor_types = registry_factor_types
            if num_factor_types > 0:
                model_cfg = model_cfg.updated(num_factor_types=num_factor_types)
                logger.info("Inferred num_factor_types=%s from constraint registry.", num_factor_types)
            else:
                num_factor_types = derive_factor_type_count(train_data, val_data)
                if num_factor_types > 0:
                    model_cfg = model_cfg.updated(num_factor_types=num_factor_types)
                    logger.info("Inferred num_factor_types=%s from dataset scan.", num_factor_types)
    elif model_cfg.num_factor_types != 0:
        logger.info(
            "Clearing num_factor_types=%s for passive constraint representation.",
            model_cfg.num_factor_types,
        )
        model_cfg = model_cfg.updated(num_factor_types=0)

    _write_effective_experiment_config(
        config_path,
        experiment_config,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
    )
    logger.info("Updated resolved experiment config at %s", config_path)

    # Load and freeze int encoder
    encoder = GlobalIntEncoder()
    interim_path = Path("data/interim/") / dataset_variant
    encoder.load(interim_path / "globalintencoder.txt")
    encoder.freeze()

    # Optional - Constraint-is-fixed loss components (heuristics + contexts).
    fix_loss_cfg = training_cfg.fix_probability_loss
    fix_loss_state: dict[str, object] | None = None
    fix_scheduler: FixProbabilityScheduler | None = None

    if fix_loss_cfg.enabled:
        fix_scheduler = FixProbabilityScheduler(fix_loss_cfg)
        placeholder_ids = placeholder_ids_from_encoder(encoder)
        heuristics = ConstraintRepairHeuristics(
            encoder=encoder,
            placeholder_ids=placeholder_ids,
            none_class=NONE_CLASS_INDEX,
        )

        def _prepare_contexts(
            split: str,
            dataset_obj: list[Data] | GraphStreamDataset,
            graph_path: Path,
        ) -> list | None:
            contexts = load_violation_contexts(interim_path, split, none_class=NONE_CLASS_INDEX)
            if isinstance(dataset_obj, list):
                expected_count = len(dataset_obj)
            else:
                expected_count = _manifest_graph_count(graph_path)

            if expected_count is not None and len(contexts) != expected_count:
                logger.warning(
                    "Fix probability loss disabled: split=%s has %s graphs but %s contexts.",
                    split,
                    expected_count,
                    len(contexts),
                )
                return None

            if isinstance(dataset_obj, list):
                for idx, graph in enumerate(dataset_obj):
                    setattr(graph, "context_index", idx)
            else:
                logger.info(
                    "Fix probability loss for %s split will use streamed datasets with on-the-fly context_index assignment.",
                    split,
                )
            return contexts

        train_contexts = _prepare_contexts("train", train_data, train_data_path)
        val_contexts = _prepare_contexts("val", val_data, val_data_path)

        if train_contexts and val_contexts:
            fix_loss_state = {
                "scheduler": fix_scheduler,
                "heuristics": heuristics,
                "train_contexts": train_contexts,
                "val_contexts": val_contexts,
            }
            logger.info(
                "Fix probability loss enabled | initial_weight=%.3f final_weight=%.3f decay=%s schedule=%s",
                fix_loss_cfg.initial_weight,
                fix_loss_cfg.final_weight,
                fix_loss_cfg.decay_epochs,
                fix_loss_cfg.schedule,
            )
        else:
            logger.warning("Fix probability loss disabled due to missing contexts.")
            fix_loss_state = None
            fix_scheduler = None
    else:
        fix_scheduler = None

    chooser_state: dict[str, object] | None = None
    chooser_cfg = training_cfg.chooser
    if chooser_cfg.enabled:
        placeholder_ids = placeholder_ids_from_encoder(encoder)
        chooser_heuristics = ConstraintRepairHeuristics(
            encoder=encoder,
            placeholder_ids=placeholder_ids,
            none_class=NONE_CLASS_INDEX,
        )
        train_contexts = load_violation_contexts(interim_path, "train", none_class=NONE_CLASS_INDEX)
        val_contexts = load_violation_contexts(interim_path, "val", none_class=NONE_CLASS_INDEX)
        train_rows = _load_parquet_rows(interim_path, "train")
        val_rows = _load_parquet_rows(interim_path, "val")
        if len(train_rows) != len(train_contexts) or len(val_rows) != len(val_contexts):
            raise RuntimeError("Mismatch between parquet rows and violation contexts for chooser.")
        if isinstance(train_data, list) and isinstance(val_data, list):
            if len(train_contexts) != len(train_data) or len(val_contexts) != len(val_data):
                raise RuntimeError("Mismatch between graph dataset size and violation contexts for chooser.")
            for idx, graph in enumerate(train_data):
                setattr(graph, "context_index", idx)
            for idx, graph in enumerate(val_data):
                setattr(graph, "context_index", idx)
        else:
            train_graph_count = _manifest_graph_count(train_data_path)
            val_graph_count = _manifest_graph_count(val_data_path)
            if train_graph_count is not None and len(train_contexts) != train_graph_count:
                raise RuntimeError(
                    f"Mismatch between train manifest graph_count={train_graph_count} and "
                    f"chooser contexts={len(train_contexts)}."
                )
            if val_graph_count is not None and len(val_contexts) != val_graph_count:
                raise RuntimeError(
                    f"Mismatch between val manifest graph_count={val_graph_count} and "
                    f"chooser contexts={len(val_contexts)}."
                )
            logger.info("Chooser training will use streamed datasets with on-the-fly context_index assignment.")
        registry_path = _resolve_constraint_registry_path(model_cfg.dataset_variant)
        evaluator = CandidateConstraintEvaluator(
            str(registry_path),
            encoder=encoder,
            assume_complete=True,
            constraint_scope="local",
            use_encoded_ids=True,
        )
        candidate_cfg = CandidateConfig(
            topk_candidates=chooser_cfg.topk_candidates,
            max_candidates_total=chooser_cfg.max_candidates_total,
            include_gold=True,
        )
        chooser_state = {
            "heuristics": chooser_heuristics,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_contexts": train_contexts,
            "val_contexts": val_contexts,
            "evaluator": evaluator,
            "candidate_cfg": candidate_cfg,
            "placeholder_ids_set": set(chooser_heuristics.placeholder_ids.values()),
        }
        logger.info(
            "Chooser enabled | topk_candidates=%s max_candidates_total=%s loss_mode=%s loss_weight=%.3f",
            chooser_cfg.topk_candidates,
            chooser_cfg.max_candidates_total,
            chooser_cfg.loss_mode,
            float(getattr(chooser_cfg, "loss_weight", 0.5)),
        )

    direct_safety_state: dict[str, object] | None = None
    direct_safety_cfg = training_cfg.direct_safety
    if direct_safety_cfg.enabled:
        if chooser_cfg.enabled:
            raise RuntimeError("Chooser and direct safety cannot both be enabled in the same training config.")
        placeholder_ids = placeholder_ids_from_encoder(encoder)
        direct_safety_heuristics = ConstraintRepairHeuristics(
            encoder=encoder,
            placeholder_ids=placeholder_ids,
            none_class=NONE_CLASS_INDEX,
        )
        train_contexts = load_violation_contexts(interim_path, "train", none_class=NONE_CLASS_INDEX)
        val_contexts = load_violation_contexts(interim_path, "val", none_class=NONE_CLASS_INDEX)
        train_rows = _load_parquet_rows(interim_path, "train")
        val_rows = _load_parquet_rows(interim_path, "val")
        if len(train_rows) != len(train_contexts) or len(val_rows) != len(val_contexts):
            raise RuntimeError("Mismatch between parquet rows and violation contexts for direct safety.")
        if isinstance(train_data, list) and isinstance(val_data, list):
            if len(train_contexts) != len(train_data) or len(val_contexts) != len(val_data):
                raise RuntimeError("Mismatch between graph dataset size and violation contexts for direct safety.")
            for idx, graph in enumerate(train_data):
                if not hasattr(graph, "context_index"):
                    setattr(graph, "context_index", idx)
            for idx, graph in enumerate(val_data):
                if not hasattr(graph, "context_index"):
                    setattr(graph, "context_index", idx)
        else:
            train_graph_count = _manifest_graph_count(train_data_path)
            val_graph_count = _manifest_graph_count(val_data_path)
            if train_graph_count is not None and len(train_contexts) != train_graph_count:
                raise RuntimeError(
                    f"Mismatch between train manifest graph_count={train_graph_count} and "
                    f"direct safety contexts={len(train_contexts)}."
                )
            if val_graph_count is not None and len(val_contexts) != val_graph_count:
                raise RuntimeError(
                    f"Mismatch between val manifest graph_count={val_graph_count} and "
                    f"direct safety contexts={len(val_contexts)}."
                )
            logger.info("Direct safety training will use streamed datasets with on-the-fly context_index assignment.")
        registry_path = _resolve_constraint_registry_path(model_cfg.dataset_variant)
        evaluator = CandidateConstraintEvaluator(
            str(registry_path),
            encoder=encoder,
            assume_complete=True,
            constraint_scope="local",
            use_encoded_ids=True,
        )
        candidate_cfg = CandidateConfig(
            topk_candidates=direct_safety_cfg.topk_candidates,
            max_candidates_total=direct_safety_cfg.max_candidates_total,
            include_gold=True,
        )
        direct_safety_state = {
            "heuristics": direct_safety_heuristics,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_contexts": train_contexts,
            "val_contexts": val_contexts,
            "evaluator": evaluator,
            "candidate_cfg": candidate_cfg,
            "placeholder_ids_set": set(direct_safety_heuristics.placeholder_ids.values()),
        }
        logger.info(
            "Direct safety enabled | topk_candidates=%s max_candidates_total=%s alpha_primary=%.3f beta_secondary=%.3f",
            direct_safety_cfg.topk_candidates,
            direct_safety_cfg.max_candidates_total,
            direct_safety_cfg.alpha_primary,
            direct_safety_cfg.beta_secondary,
        )

    policy_state: dict[str, object] | None = None
    if model_cfg.enable_policy_choice:
        if isinstance(train_data, GraphStreamDataset):
            logger.info("Materializing training stream dataset into memory for policy choice.")
            train_data = list(train_data)
        if isinstance(val_data, GraphStreamDataset):
            logger.info("Materializing validation stream dataset into memory for policy choice.")
            val_data = list(val_data)
        if not isinstance(train_data, list) or not isinstance(val_data, list):
            raise RuntimeError("Policy choice training requires in-memory datasets (list[Data]).")
        train_contexts = load_violation_contexts(interim_path, "train", none_class=NONE_CLASS_INDEX)
        val_contexts = load_violation_contexts(interim_path, "val", none_class=NONE_CLASS_INDEX)
        if len(train_contexts) != len(train_data) or len(val_contexts) != len(val_data):
            raise RuntimeError("Mismatch between graph dataset size and violation contexts for policy choice.")
        for idx, graph in enumerate(train_data):
            if not hasattr(graph, "context_index"):
                setattr(graph, "context_index", idx)
        for idx, graph in enumerate(val_data):
            if not hasattr(graph, "context_index"):
                setattr(graph, "context_index", idx)
        policy_state = {
            "train_contexts": train_contexts,
            "val_contexts": val_contexts,
        }
        logger.info(
            "Policy choice enabled | classes=%s | policies=%s",
            model_cfg.policy_num_classes,
            ", ".join(POLICY_NAMES),
        )

    estimated_train_batches: int | None = None
    estimated_val_batches: int | None = None

    if isinstance(train_data, list):
        estimated_train_batches = max(1, math.ceil(len(train_data) / max(training_cfg.batch_size, 1)))
    else:
        train_graph_count = _manifest_graph_count(train_data_path)
        if train_graph_count is not None and train_graph_count > 0:
            estimated_train_batches = max(1, math.ceil(train_graph_count / max(training_cfg.batch_size, 1)))

    if isinstance(val_data, list):
        val_graph_count = len(val_data)
        if training_cfg.validation_subset_size is not None:
            val_graph_count = min(val_graph_count, training_cfg.validation_subset_size)
        estimated_val_batches = max(1, math.ceil(val_graph_count / max(training_cfg.batch_size, 1)))
    else:
        val_graph_count = _manifest_graph_count(val_data_path)
        if val_graph_count is not None and val_graph_count > 0:
            if training_cfg.validation_subset_size is not None:
                val_graph_count = min(val_graph_count, training_cfg.validation_subset_size)
            estimated_val_batches = max(1, math.ceil(val_graph_count / max(training_cfg.batch_size, 1)))

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Selected compute device: %s", device)

    vocab_from_filtered = len(encoder._global_id_to_unfiltered_global_id)
    if vocab_from_filtered > 0:
        num_input_graph_nodes = vocab_from_filtered + 1
    else:
        num_input_graph_nodes = len(encoder._encoding) + 1
    logger.info("Encoder vocabulary size (graph nodes): %s", num_input_graph_nodes)

    # Build model
    model = build_model(
        model_cfg.model,
        num_input_graph_nodes,
        model_cfg,
    )
    model.to(device)
    if chooser_cfg.enabled:
        model.enable_chooser()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters | total=%s trainable=%s", total_params, trainable_params)
    logger.debug("Model architecture: %s", model)
    if hasattr(model, "head_hidden"):
        logger.info(
            "Model dimensions | mp_hidden=%s head_hidden=%s branch_hidden=%s",
            getattr(model, "hidden_channels", "?"),
            model.head_hidden,
            getattr(model, "branch_hidden", "?"),
        )
        logger.info(
            "Model regularisation | dropout=%.3f num_layers=%s",
            getattr(model, "dropout", float("nan")),
            model.num_layers,
        )
    logger.info(
        "Role features | enabled=%s num_role_types=%s embedding_dim=%s",
        model_cfg.use_role_embeddings,
        model_cfg.num_role_types,
        getattr(model, "role_embedding_dim", 0),
    )
    if device.type == "cuda":
        log_cuda_memory("Model moved to device", device, level=logging.DEBUG)

    # Train model
    model, history = train(
        model,
        train_data,
        val_data,
        num_input_graph_nodes,
        train_cfg=training_cfg,
        device=device,
        fix_loss_state=fix_loss_state,
        chooser_state=chooser_state,
        direct_safety_state=direct_safety_state,
        policy_state=policy_state,
        estimated_train_batches=estimated_train_batches,
        estimated_val_batches=estimated_val_batches,
    )

    if device.type == "cuda":
        log_cuda_memory("Post-training state", device)

    model_path = get_checkpoint_path(run_directory)
    # Save model state and minimal config
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_graph_nodes": num_input_graph_nodes,
            "model_name": model_cfg.model,
            "model_cfg": model_cfg.to_dict(),
            "training_cfg": training_cfg.to_dict(),
        },
        model_path,
    )
    logger.info("Saved model checkpoint to %s", model_path)

    # Save training history
    history_file = history_path(run_directory)
    with history_file.open("w", encoding="utf-8") as f:
        json.dump(history, f)
    logger.info("Persisted training history to %s", history_file)
    plot_training_history(history, history_file)


if __name__ == "__main__":
    main()
