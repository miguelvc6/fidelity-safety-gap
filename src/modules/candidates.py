from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch_geometric.data import Data

from .repair_eval import ConstraintRepairHeuristics, ViolationContext


NUM_SLOTS = 6
NONE_CLASS_INDEX = 0


@dataclass(frozen=True)
class CandidateConfig:
    topk_candidates: int = 20
    topk_per_slot: int = 5
    heuristic_max_candidates: int = 30
    heuristic_max_values: int = 3
    include_gold: bool = True
    max_candidates_total: int = 80


def gold_candidate(graph: Data) -> tuple[int, int, int, int, int, int]:
    y = getattr(graph, "y", None)
    if y is None:
        raise ValueError("Graph missing y target tensor.")
    if y.dim() == 2:
        y = y[0]
    return tuple(int(v) for v in y.tolist())


def _coerce_gold_candidate(
    *,
    graph: Data | None,
    gold_slots: Sequence[int] | None,
) -> tuple[int, int, int, int, int, int]:
    if gold_slots is not None:
        if len(gold_slots) != NUM_SLOTS:
            raise ValueError(f"Expected gold_slots length {NUM_SLOTS}, got {len(gold_slots)}")
        return tuple(int(v) for v in gold_slots)
    if graph is None:
        raise ValueError("Either graph or gold_slots must be provided to build_candidates().")
    return gold_candidate(graph)


def candidate_from_triple(triple: tuple[int, int, int], *, action: str) -> tuple[int, int, int, int, int, int]:
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
            break
    return unique


def _instantiate_patterns(
    patterns,
    *,
    placeholder_ids: set[int],
    none_class: int,
    max_values: int,
    max_candidates: int,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for pattern in patterns:
        subj_vals = _select_values(pattern.subjects, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        pred_vals = _select_values(pattern.predicates, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        obj_vals = _select_values(pattern.objects, placeholder_ids=placeholder_ids, none_class=none_class, max_values=max_values)
        if not subj_vals or not pred_vals or not obj_vals:
            continue
        for s in subj_vals:
            for p in pred_vals:
                for o in obj_vals:
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
    slot_allowed_ids: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None,
) -> list[tuple[int, int, int]]:
    slot_vals = []
    slot_ids = []
    for local_idx, slot in enumerate(slots):
        allowed_ids = None if slot_allowed_ids is None else slot_allowed_ids[local_idx]
        if allowed_ids is None:
            k = max(1, min(topk_per_slot, logits.size(-1)))
            vals, ids = torch.topk(logits[slot], k=k)
        else:
            allowed = allowed_ids
            if allowed.device != logits.device:
                allowed = allowed.to(device=logits.device)
            if allowed.dtype != torch.long:
                allowed = allowed.to(dtype=torch.long)
            if allowed.numel() >= topk_per_slot and allowed.numel() > 0:
                restricted = logits[slot].index_select(0, allowed)
                k = max(1, min(topk_per_slot, restricted.size(0)))
                vals_local, idx_local = torch.topk(restricted, k=k)
                ids = allowed.index_select(0, idx_local)
                vals = vals_local
            else:
                # Preserve legacy behavior when the allowed-id set is too small:
                # old logic would still return top-k over full vocabulary.
                k = max(1, min(topk_per_slot, logits.size(-1)))
                vals, ids = torch.topk(logits[slot], k=k)
        slot_vals.append(vals.cpu())
        slot_ids.append(ids.cpu())
    if len(slot_vals) != 3:
        return []
    k_combos = min(len(slot_vals[0]), len(slot_vals[1]), len(slot_vals[2]))
    if k_combos <= 0:
        return []
    combos: list[tuple[float, int, int, int]] = []
    for i in range(k_combos):
        for j in range(k_combos):
            for k in range(k_combos):
                score = float(slot_vals[0][i] + slot_vals[1][j] + slot_vals[2][k])
                combos.append((score, int(slot_ids[0][i]), int(slot_ids[1][j]), int(slot_ids[2][k])))
    combos.sort(key=lambda x: x[0], reverse=True)
    return [(s, p, o) for _, s, p, o in combos[:topk_triples]]


def batch_topk_candidate_triples(
    proposal_logits: torch.Tensor,
    *,
    topk_triples: int,
    topk_per_slot: int,
) -> tuple[list[list[tuple[int, int, int]]], list[list[tuple[int, int, int]]]]:
    """Precompute add/delete top-k triples for a batch with one vocabulary scan."""
    if proposal_logits.dim() != 3 or proposal_logits.size(1) != NUM_SLOTS:
        raise ValueError(
            f"proposal_logits must be shaped (batch, {NUM_SLOTS}, vocab), "
            f"got {tuple(proposal_logits.shape)}"
        )
    k = max(1, min(topk_per_slot, proposal_logits.size(-1)))
    slot_values, slot_ids = torch.topk(proposal_logits, k=k, dim=-1)
    slot_values = slot_values.detach().cpu()
    slot_ids = slot_ids.detach().cpu()

    def triples_for(slot_offset: int) -> list[list[tuple[int, int, int]]]:
        scores = (
            slot_values[:, slot_offset, :, None, None]
            + slot_values[:, slot_offset + 1, None, :, None]
            + slot_values[:, slot_offset + 2, None, None, :]
        ).reshape(proposal_logits.size(0), -1)
        keep = min(topk_triples, scores.size(1))
        # Stable descending order matches the legacy Python sort's i/j/k tie order.
        flat_indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :keep]
        first_index = flat_indices // (k * k)
        second_index = (flat_indices // k) % k
        third_index = flat_indices % k
        first = slot_ids[:, slot_offset].gather(1, first_index)
        second = slot_ids[:, slot_offset + 1].gather(1, second_index)
        third = slot_ids[:, slot_offset + 2].gather(1, third_index)
        rows = torch.stack((first, second, third), dim=-1).tolist()
        return [[tuple(int(value) for value in triple) for triple in row] for row in rows]

    return triples_for(0), triples_for(3)


def build_candidates(
    *,
    graph: Data | None = None,
    gold_slots: Sequence[int] | None = None,
    context: ViolationContext,
    heuristics: ConstraintRepairHeuristics,
    proposal_logits: torch.Tensor,
    cfg: CandidateConfig,
    placeholder_ids: set[int],
    num_target_ids: int,
    slot_allowed_ids: tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
    | None = None,
    precomputed_add_topk: Sequence[tuple[int, int, int]] | None = None,
    precomputed_del_topk: Sequence[tuple[int, int, int]] | None = None,
) -> tuple[list[tuple[int, int, int, int, int, int]], int | None]:
    candidates: list[tuple[int, int, int, int, int, int]] = []
    gold = (
        _coerce_gold_candidate(graph=graph, gold_slots=gold_slots)
        if cfg.include_gold
        else None
    )

    if gold is not None:
        candidates.append(gold)

    candidate_map = heuristics.candidates_for(context)
    add_triples = _instantiate_patterns(
        candidate_map.add,
        placeholder_ids=placeholder_ids,
        none_class=NONE_CLASS_INDEX,
        max_values=cfg.heuristic_max_values,
        max_candidates=cfg.heuristic_max_candidates,
    )
    del_triples = _instantiate_patterns(
        candidate_map.delete,
        placeholder_ids=placeholder_ids,
        none_class=NONE_CLASS_INDEX,
        max_values=cfg.heuristic_max_values,
        max_candidates=cfg.heuristic_max_candidates,
    )
    candidates.extend(candidate_from_triple(triple, action="add") for triple in add_triples)
    candidates.extend(candidate_from_triple(triple, action="delete") for triple in del_triples)

    if precomputed_add_topk is None:
        add_slots = (0, 1, 2)
        add_topk = _topk_triples_from_logits(
            proposal_logits,
            slots=add_slots,
            topk_triples=cfg.topk_candidates,
            topk_per_slot=cfg.topk_per_slot,
            slot_allowed_ids=(slot_allowed_ids[0], slot_allowed_ids[1], slot_allowed_ids[2])
            if slot_allowed_ids is not None
            else None,
        )
    else:
        add_topk = [(int(s), int(p), int(o)) for s, p, o in precomputed_add_topk]

    if precomputed_del_topk is None:
        del_slots = (3, 4, 5)
        del_topk = _topk_triples_from_logits(
            proposal_logits,
            slots=del_slots,
            topk_triples=cfg.topk_candidates,
            topk_per_slot=cfg.topk_per_slot,
            slot_allowed_ids=(slot_allowed_ids[3], slot_allowed_ids[4], slot_allowed_ids[5])
            if slot_allowed_ids is not None
            else None,
        )
    else:
        del_topk = [(int(s), int(p), int(o)) for s, p, o in precomputed_del_topk]

    candidates.extend(candidate_from_triple(triple, action="add") for triple in add_topk)
    candidates.extend(candidate_from_triple(triple, action="delete") for triple in del_topk)

    deduped: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for cand in candidates:
        if any(v < 0 or v >= num_target_ids for v in cand):
            continue
        if cand in seen:
            continue
        seen.add(cand)
        deduped.append(cand)
        if len(deduped) >= cfg.max_candidates_total:
            break

    if gold is None:
        return deduped, None

    if any(v < 0 or v >= num_target_ids for v in gold):
        raise ValueError("Gold candidate contains out-of-range ids for target vocabulary.")
    if gold not in seen:
        deduped.insert(0, gold)
    return deduped, deduped.index(gold)


def score_candidates_from_logits(
    proposal_logits: torch.Tensor,
    candidate_slots: torch.Tensor,
) -> torch.Tensor:
    """Score candidate edits by summing the corresponding per-slot proposal logits."""
    if proposal_logits.dim() != 2 or proposal_logits.size(0) != NUM_SLOTS:
        raise ValueError(
            f"proposal_logits must be shaped ({NUM_SLOTS}, vocab), got {tuple(proposal_logits.shape)}"
        )
    if candidate_slots.dim() != 2 or candidate_slots.size(-1) != NUM_SLOTS:
        raise ValueError(
            f"candidate_slots must be shaped (num_candidates, {NUM_SLOTS}), got {tuple(candidate_slots.shape)}"
        )
    candidate_slots = candidate_slots.to(device=proposal_logits.device, dtype=torch.long)
    if candidate_slots.numel() == 0:
        return proposal_logits.new_zeros((0,))
    if int(candidate_slots.min().item()) < 0 or int(candidate_slots.max().item()) >= proposal_logits.size(-1):
        raise ValueError("candidate_slots contains out-of-range ids for proposal logits.")
    gathered = proposal_logits.gather(1, candidate_slots.transpose(0, 1)).transpose(0, 1)
    return gathered.sum(dim=-1)


def score_candidates_from_logits_packed(
    proposal_logits: torch.Tensor,
    candidate_slots: torch.Tensor,
    candidate_graph_index: torch.Tensor,
) -> torch.Tensor:
    """Score packed candidate rows against their corresponding batched logits."""
    if proposal_logits.dim() != 3 or proposal_logits.size(1) != NUM_SLOTS:
        raise ValueError(
            f"proposal_logits must be shaped (batch, {NUM_SLOTS}, vocab), "
            f"got {tuple(proposal_logits.shape)}"
        )
    if candidate_slots.dim() != 2 or candidate_slots.size(-1) != NUM_SLOTS:
        raise ValueError(
            f"candidate_slots must be shaped (num_candidates, {NUM_SLOTS}), "
            f"got {tuple(candidate_slots.shape)}"
        )
    candidate_slots = candidate_slots.to(device=proposal_logits.device, dtype=torch.long)
    candidate_graph_index = candidate_graph_index.to(
        device=proposal_logits.device, dtype=torch.long
    ).view(-1)
    if candidate_graph_index.numel() != candidate_slots.size(0):
        raise ValueError("candidate_graph_index length must match candidate_slots rows.")
    if candidate_graph_index.numel() == 0:
        return proposal_logits.new_empty((0,))
    if (
        int(candidate_graph_index.min().item()) < 0
        or int(candidate_graph_index.max().item()) >= proposal_logits.size(0)
    ):
        raise ValueError("candidate_graph_index contains out-of-range graph ids.")
    if int(candidate_slots.min().item()) < 0 or int(candidate_slots.max().item()) >= proposal_logits.size(-1):
        raise ValueError("candidate_slots contain out-of-range vocabulary ids.")

    slot_index = torch.arange(NUM_SLOTS, device=proposal_logits.device).view(1, -1)
    selected = proposal_logits[
        candidate_graph_index.view(-1, 1),
        slot_index,
        candidate_slots,
    ]
    return selected.sum(dim=-1)
