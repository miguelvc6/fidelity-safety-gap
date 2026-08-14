from __future__ import annotations

from types import SimpleNamespace

from modules.evidence_state import apply_evidence_edits, build_pre_state
from modules.repair_eval import (
    PAPER_METRIC_KEYS,
    RepairSample,
    evaluate_global_repair_samples,
    evaluate_paper_metric_instance,
)


def _row(**overrides):
    values = {
        "constraint_id": 100,
        "constraint_type": "single",
        "subject": 1,
        "predicate": 10,
        "object": 2,
        "other_subject": 0,
        "other_predicate": 0,
        "other_object": 0,
        "subject_predicates": [],
        "subject_objects": [],
        "object_predicates": [],
        "object_objects": [],
        "other_entity_predicates": [],
        "other_entity_objects": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_role_aliases_merge_and_base_is_mandatory() -> None:
    row = _row(
        object=1,
        subject_predicates=[20],
        subject_objects=[3],
        object_predicates=[30],
        object_objects=[4],
        other_subject=1,
        other_entity_predicates=[40],
        other_entity_objects=[5],
    )
    state, _ = build_pre_state(row, assume_complete=True, cast_int=True)

    assert state.facts_by_entity[1][20] == {3}
    assert state.facts_by_entity[1][30] == {4}
    assert state.facts_by_entity[1][40] == {5}
    assert state.facts_by_entity[1][10] == {1}
    assert state.focus_statement_present()


def test_base_and_non_base_deletion_and_delete_then_reinsert() -> None:
    state, p_local = build_pre_state(
        _row(subject_predicates=[20], subject_objects=[3]),
        assume_complete=True,
        cast_int=True,
    )

    non_base, _ = apply_evidence_edits(
        state,
        p_local=p_local,
        delete=(1, 20, 3),
        add=None,
    )
    assert not non_base.has_statement(1, 20, 3)
    assert non_base.focus_statement_present()

    deleted, _ = apply_evidence_edits(
        state,
        p_local=p_local,
        delete=(1, 10, 2),
        add=None,
    )
    assert not deleted.focus_statement_present()

    reinserted, resolved = apply_evidence_edits(
        state,
        p_local=p_local,
        delete=(1, 10, 2),
        add=(1, 10, 2),
    )
    assert reinserted.focus_statement_present()
    assert resolved == {"del": (1, 10, 2), "add": (1, 10, 2)}


class _Evaluator:
    def __init__(self, details):
        self.details = iter(details)

    def evaluate_full(self, *args, **kwargs):
        del args, kwargs
        return next(self.details)


def _sample(delete=None):
    return RepairSample(
        constraint_type="single",
        predicted={"add": None, "del": delete},
        gold={"add": None, "del": None},
    )


def test_paper_metrics_use_eligibility_common_support_and_pooled_rates() -> None:
    details = {
        "local_constraint_ids": [100, 200, 300],
        "primary_factor_index": 0,
        "pre_checkable": [True, True, True],
        "pre_satisfied": [0, 1, 0],
        "post_checkable": [True, True, True],
        "post_satisfied": [1, 0, 1],
        "pre_focus_present": 1,
        "post_focus_present": 1,
        "candidate_deletes_focus": 0,
    }
    report = evaluate_global_repair_samples(
        samples=[_sample()],
        rows=[_row()],
        evaluator=_Evaluator([details]),
        none_class=0,
    )
    paper = report["paper_metrics"]

    assert paper["pfr"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert paper["local_satisfaction"] == {
        "value": 2 / 3,
        "numerator": 2,
        "denominator": 3,
    }
    assert paper["delta_local_satisfaction"] == {
        "value": 1 / 3,
        "numerator": 1,
        "denominator": 3,
    }
    assert paper["sir"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert paper["srr"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert paper["eppf"] == {"value": 1.0, "numerator": 1, "denominator": 1}


def test_metric_keys_are_unique_and_partial_slots_are_not_operations() -> None:
    assert len(PAPER_METRIC_KEYS) == len(set(PAPER_METRIC_KEYS))

    details = {
        "local_constraint_ids": [100],
        "primary_factor_index": 0,
        "pre_checkable": [True],
        "pre_satisfied": [0],
        "post_checkable": [True],
        "post_satisfied": [0],
        "pre_focus_present": 1,
        "post_focus_present": 1,
        "candidate_deletes_focus": 0,
    }
    instance = evaluate_paper_metric_instance(
        row=_row(),
        evaluator=_Evaluator([details]),
        candidate_slots=(1, 10, 0, 1, 0, 2),
        constraint_type="single",
        row_index=0,
        none_class=0,
    )

    assert instance["events"]["disruption"] == {"numerator": 0, "denominator": 1}


def test_empty_support_and_delete_focus_baseline_events() -> None:
    empty = {
        "local_constraint_ids": [100],
        "primary_factor_index": 0,
        "pre_checkable": [False],
        "pre_satisfied": [0],
        "post_checkable": [False],
        "post_satisfied": [0],
        "pre_focus_present": 1,
        "post_focus_present": 1,
        "candidate_deletes_focus": 0,
    }
    report = evaluate_global_repair_samples(
        samples=[_sample()],
        rows=[_row()],
        evaluator=_Evaluator([empty]),
        none_class=0,
    )
    assert report["paper_metrics"]["pfr"] == {
        "value": 0.0,
        "numerator": 0,
        "denominator": 0,
    }
    assert report["paper_metrics"]["local_satisfaction"]["denominator"] == 0

    dfb = {
        "local_constraint_ids": [100],
        "primary_factor_index": 0,
        "pre_checkable": [True],
        "pre_satisfied": [0],
        "post_checkable": [True],
        "post_satisfied": [1],
        "pre_focus_present": 1,
        "post_focus_present": 0,
        "candidate_deletes_focus": 1,
    }
    report = evaluate_global_repair_samples(
        samples=[_sample((1, 10, 2))],
        rows=[_row()],
        evaluator=_Evaluator([dfb]),
        none_class=0,
    )
    paper = report["paper_metrics"]
    assert paper["base_deletion_rate"]["value"] == 1.0
    assert paper["deletes_base_action_rate"]["value"] == 1.0
    assert paper["eppf"]["value"] == 0.0
    assert paper["vacuous_improvement"]["value"] == 1.0


def test_stored_pre_vectors_are_rejected_as_evaluation_truth() -> None:
    try:
        evaluate_global_repair_samples(
            samples=[_sample()],
            rows=[_row()],
            evaluator=_Evaluator([]),
            none_class=0,
            pre_vectors=[{"pre_satisfied": [1]}],
        )
    except ValueError as exc:
        assert "not valid paper-metric truth" in str(exc)
    else:
        raise AssertionError("Expected stored pre-vector rejection")
