from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from modules.constraint_checkers import ConstraintInstance
from modules.constraint_identity import resolve_primary_index, select_constraint_ids
from modules.reranker_eval import CandidateConstraintEvaluator


def _definition(constraint_id: int, *, supported: bool = True) -> ConstraintInstance:
    instance = ConstraintInstance(
        constraint_id=constraint_id,
        constraint_type="oneOf",
        constraint_type_id=1,
        constrained_property=10,
        required_properties=set(),
        allowed_items={5},
        allowed_classes=set(),
        relation_predicates=[],
        inverse_properties=[],
        conflict_properties=set(),
    )
    if supported:
        return instance
    return replace(
        instance,
        constraint_type="unsupported:test",
        definition_valid=False,
        invalid_reason="unsupported constraint family",
    )


def _row(factor_ids=(900,)) -> SimpleNamespace:
    return SimpleNamespace(
        constraint_id=900,
        constraint_type="oneOf",
        subject=1,
        predicate=10,
        object=6,
        other_subject=0,
        other_predicate=0,
        other_object=0,
        subject_predicates=[10],
        subject_objects=[6],
        object_predicates=[],
        object_objects=[],
        other_entity_predicates=[],
        other_entity_objects=[],
        local_constraint_ids=[100, 900],
        local_constraint_ids_focus=[100, 900],
        factor_constraint_ids=list(factor_ids),
    )


def _evaluator() -> CandidateConstraintEvaluator:
    definitions = {100: _definition(100, supported=False), 900: _definition(900)}
    evaluator = object.__new__(CandidateConstraintEvaluator)
    evaluator._encoder = None
    evaluator._assume_complete = True
    evaluator._constraint_scope = "local"
    evaluator._use_encoded_ids = True
    evaluator._placeholder_token_ids = {}
    evaluator._hierarchy = None
    evaluator._get_constraint_instance = lambda constraint_id: definitions.get(int(constraint_id))
    return evaluator


def test_filtered_primary_position_is_resolved_in_filtered_vector() -> None:
    row = _row()
    assert select_constraint_ids(row, constraint_scope="local") == [900]
    assert resolve_primary_index(row, [900], supplied_index=0) == 0
    candidate = (1, 10, 5, 1, 10, 6)
    single = _evaluator().evaluate_full(row, candidate_slots=candidate, primary_factor_index=0)
    batch = _evaluator().evaluate_candidate_metrics(
        row, candidates=[candidate], primary_factor_index=0
    )[0]
    assert single["local_constraint_ids"] == [900]
    assert single["primary_factor_index"] == 0
    assert single["primary_satisfied"] == batch.primary_satisfied == 1
    assert batch.primary_pre_violated == 1


def test_constraint_vector_permutation_preserves_primary_result_by_id() -> None:
    candidate = (1, 10, 5, 1, 10, 6)
    first = _evaluator().evaluate_full(
        _row((900, 100)), candidate_slots=candidate, primary_factor_index=0
    )
    second = _evaluator().evaluate_full(
        _row((100, 900)), candidate_slots=candidate, primary_factor_index=1
    )
    assert first["primary_satisfied"] == second["primary_satisfied"] == 1
    assert first["post_outcomes"][0] == second["post_outcomes"][1] == "satisfied"


@pytest.mark.parametrize(
    ("ids", "index", "message"),
    [
        ([100], None, "missing"),
        ([900, 900], None, "occurs 2 times"),
        ([100, 900], 0, "mismatch"),
    ],
)
def test_primary_identity_errors_are_actionable(ids, index, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_primary_index(_row(ids), ids, supplied_index=index)


def test_batch_roundtrip_keeps_factor_id_index_alignment() -> None:
    graphs = []
    for ids, primary in (([100, 900], 1), ([900, 100], 0)):
        graph = Data(x=torch.ones((2, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
        graph.factor_constraint_ids = torch.tensor(ids, dtype=torch.long)
        graph.primary_factor_index = primary
        graph.shape_id = 900
        graphs.append(graph)
    restored = Batch.from_data_list(graphs).to_data_list()
    for graph in restored:
        ids = graph.factor_constraint_ids.tolist()
        primary = int(graph.primary_factor_index)
        assert ids[primary] == int(graph.shape_id) == 900
