from __future__ import annotations

import torch

from modules.candidates import (
    CandidateConfig,
    batch_topk_candidate_triples,
    build_candidates,
    score_candidates_from_logits,
    score_candidates_from_logits_packed,
)
from modules.repair_eval import CandidateRepairs, ViolationContext


class _NoHeuristicCandidates:
    placeholder_ids = {"subject": 7, "predicate": 8, "object": 9}

    def candidates_for(self, _context: ViolationContext) -> CandidateRepairs:
        return CandidateRepairs()


class _GoldReadTrap:
    @property
    def y(self):
        raise AssertionError("evaluation candidate generation read graph.y")


def _context() -> ViolationContext:
    return ViolationContext(
        constraint_type="single",
        constraint_id=1,
        subject=1,
        predicate=2,
        object=3,
        other_subject=0,
        other_predicate=0,
        other_object=0,
        constraint_predicates=(),
        constraint_objects=(),
    )


def _proposal_logits() -> torch.Tensor:
    logits = torch.zeros((6, 10), dtype=torch.float32)
    for slot, target_id in enumerate((1, 2, 3, 4, 5, 6)):
        logits[slot, target_id] = 10.0
    return logits


def test_build_candidates_does_not_read_or_force_gold_when_excluded() -> None:
    gold = (9, 8, 7, 6, 5, 4)
    candidates, gold_index = build_candidates(
        graph=None,
        gold_slots=gold,
        context=_context(),
        heuristics=_NoHeuristicCandidates(),
        proposal_logits=_proposal_logits(),
        cfg=CandidateConfig(
            include_gold=False,
            topk_candidates=1,
            topk_per_slot=1,
            heuristic_max_candidates=0,
        ),
        placeholder_ids=set(),
        num_target_ids=10,
    )

    assert gold_index is None
    assert gold not in candidates
    assert candidates == [(1, 2, 3, 0, 0, 0), (0, 0, 0, 4, 5, 6)]


def test_build_candidates_does_not_access_graph_target_when_excluded() -> None:
    candidates, gold_index = build_candidates(
        graph=_GoldReadTrap(),  # type: ignore[arg-type]
        context=_context(),
        heuristics=_NoHeuristicCandidates(),
        proposal_logits=_proposal_logits(),
        cfg=CandidateConfig(
            include_gold=False,
            topk_candidates=1,
            topk_per_slot=1,
            heuristic_max_candidates=0,
        ),
        placeholder_ids=set(),
        num_target_ids=10,
    )

    assert gold_index is None
    assert candidates == [(1, 2, 3, 0, 0, 0), (0, 0, 0, 4, 5, 6)]


def test_build_candidates_includes_gold_and_returns_its_index_for_training() -> None:
    gold = (9, 8, 7, 6, 5, 4)
    candidates, gold_index = build_candidates(
        graph=None,
        gold_slots=gold,
        context=_context(),
        heuristics=_NoHeuristicCandidates(),
        proposal_logits=_proposal_logits(),
        cfg=CandidateConfig(
            include_gold=True,
            topk_candidates=1,
            topk_per_slot=1,
            heuristic_max_candidates=0,
        ),
        placeholder_ids=set(),
        num_target_ids=10,
    )

    assert gold_index is not None
    assert candidates[gold_index] == gold


def test_packed_logit_scoring_matches_independent_graph_scoring() -> None:
    logits = torch.arange(2 * 6 * 10, dtype=torch.float32).view(2, 6, 10)
    first = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    second = torch.tensor([[0, 1, 2, 3, 4, 5]])
    packed = torch.cat((first, second), dim=0)
    graph_index = torch.tensor([0, 0, 1])

    expected = torch.cat(
        (
            score_candidates_from_logits(logits[0], first),
            score_candidates_from_logits(logits[1], second),
        )
    )
    observed = score_candidates_from_logits_packed(logits, packed, graph_index)

    assert torch.equal(observed, expected)


def test_batched_topk_candidate_generation_matches_per_row_generation() -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn((3, 6, 23), generator=generator)
    cfg = CandidateConfig(
        include_gold=False,
        topk_candidates=7,
        topk_per_slot=4,
        heuristic_max_candidates=0,
    )
    batch_add, batch_delete = batch_topk_candidate_triples(
        logits,
        topk_triples=cfg.topk_candidates,
        topk_per_slot=cfg.topk_per_slot,
    )

    for row_index in range(logits.size(0)):
        expected, _ = build_candidates(
            context=_context(),
            heuristics=_NoHeuristicCandidates(),
            proposal_logits=logits[row_index],
            cfg=cfg,
            placeholder_ids=set(),
            num_target_ids=23,
        )
        observed, _ = build_candidates(
            context=_context(),
            heuristics=_NoHeuristicCandidates(),
            proposal_logits=logits[row_index],
            cfg=cfg,
            placeholder_ids=set(),
            num_target_ids=23,
            precomputed_add_topk=batch_add[row_index],
            precomputed_del_topk=batch_delete[row_index],
        )
        assert observed == expected
