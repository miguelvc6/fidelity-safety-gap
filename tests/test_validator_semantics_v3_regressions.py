from __future__ import annotations

from dataclasses import replace

import pytest

from modules.class_hierarchy import ClassHierarchy
from modules.constraint_checkers import (
    ConstraintInstance,
    EvidenceState,
    UnresolvedEdit,
    ValidationOutcome as O,
    evaluate_constraint_detailed,
    evaluate_constraint_outcome,
)
from modules.evidence_state import apply_evidence_edits


def _constraint(
    family: str,
    *,
    prop: int = 10,
    selector: str | None = None,
    exceptions: frozenset[int] = frozenset(),
) -> ConstraintInstance:
    return ConstraintInstance(
        constraint_id=900,
        constraint_type=family,
        constraint_type_id=1,
        constrained_property=prop,
        required_properties={20},
        allowed_items={5},
        allowed_classes={5},
        relation_predicates=[31, 279],
        inverse_properties=[20],
        conflict_properties={20},
        selector=selector,
        exceptions=exceptions,
        p31_predicate=31,
        p279_predicate=279,
    )


def _state(facts: dict[int, dict[int, set[int]]], *, complete: bool = True) -> EvidenceState:
    return EvidenceState(
        facts_by_entity=facts,
        predicates_present={subject: set(values) for subject, values in facts.items()},
        assume_complete=complete,
        missing_edits=set(),
        focus_subject=1,
        focus_predicate=10,
        focus_object=6,
        other_subject=2,
        other_predicate=10,
        other_object=7,
    )


def test_p279_deletion_keeps_surviving_alternate_path() -> None:
    hierarchy = ClassHierarchy(
        parents={1: {5, 7}, 7: {5}, 5: set()},
        complete={1: True, 7: True, 5: True},
    )
    pre = _state({1: {10: {6}, 279: {5, 7}}, 7: {279: {5}}})
    post, _ = apply_evidence_edits(pre, p_local={10, 279}, delete=(1, 279, 5), add=None)
    assert evaluate_constraint_outcome(
        post, _constraint("type", selector="subclass"), hierarchy=hierarchy
    ) == O.SATISFIED


def test_p279_tombstone_with_incomplete_remaining_ancestry_is_unknown() -> None:
    hierarchy = ClassHierarchy(
        parents={1: {5, 7}, 7: set(), 5: set()},
        complete={1: True, 7: False, 5: True},
        direct_complete={1: True, 7: False, 5: True},
    )
    pre = _state({1: {10: {6}, 279: {5, 7}}, 7: {}}, complete=False)
    post, _ = apply_evidence_edits(pre, p_local={10, 279}, delete=(1, 279, 5), add=None)
    assert evaluate_constraint_outcome(
        post, _constraint("type", selector="subclass"), hierarchy=hierarchy
    ) == O.UNKNOWN


def test_candidate_p279_overlays_are_isolated_and_delete_reinsert_survives() -> None:
    hierarchy = ClassHierarchy(parents={1: {5}, 5: set()}, complete={1: True, 5: True})
    pre = _state({1: {10: {6}, 279: {5}}})
    deleted, _ = apply_evidence_edits(pre, p_local={10, 279}, delete=(1, 279, 5), add=None)
    unchanged, _ = apply_evidence_edits(pre, p_local={10, 279}, delete=None, add=None)
    reinserted, _ = apply_evidence_edits(
        pre, p_local={10, 279}, delete=(1, 279, 5), add=(1, 279, 5)
    )
    constraint = _constraint("type", selector="subclass")
    assert evaluate_constraint_outcome(deleted, constraint, hierarchy=hierarchy) == O.VIOLATED
    assert evaluate_constraint_outcome(unchanged, constraint, hierarchy=hierarchy) == O.SATISFIED
    assert evaluate_constraint_outcome(reinserted, constraint, hierarchy=hierarchy) == O.SATISFIED
    assert pre.has_statement(1, 279, 5)


def test_unresolved_edit_metadata_is_specific_and_diagnostic() -> None:
    pre = _state({1: {10: {6}}})
    post, _ = apply_evidence_edits(pre, p_local={10}, delete=None, add=(9, 10, 6))
    assert post.unresolved_edits == (
        UnresolvedEdit("add", 9, 10, 6, "subject_outside_bounded_scope"),
    )
    result = evaluate_constraint_detailed(post, _constraint("distinct"))
    assert result.outcome == O.UNKNOWN
    assert result.applicable is True
    assert "unresolved add" in result.unknown_reasons[0]


def test_unrelated_unresolved_edit_does_not_change_a_result() -> None:
    pre = _state({1: {10: {6}}})
    post, _ = apply_evidence_edits(pre, p_local={10}, delete=None, add=(9, 99, 8))
    assert evaluate_constraint_outcome(post, _constraint("oneOf")) == O.VIOLATED


def test_unresolved_secondary_new_anchor_is_unknown_but_existing_violation_dominates() -> None:
    pre = _state({1: {10: {6}}, 2: {30: {5}}})
    post, _ = apply_evidence_edits(pre, p_local={10, 30}, delete=None, add=(9, 30, 7))
    allowed = replace(_constraint("oneOf", prop=30), allowed_items={5})
    assert evaluate_constraint_outcome(post, allowed, primary=False) == O.UNKNOWN

    already_bad = _state({1: {10: {6}}, 2: {30: {7}}})
    still_bad, _ = apply_evidence_edits(already_bad, p_local={10, 30}, delete=None, add=(9, 30, 8))
    assert evaluate_constraint_outcome(still_bad, allowed, primary=False) == O.VIOLATED


@pytest.mark.parametrize(
    "family",
    ["inverse", "symmetric", "valueRequiresStatement", "oneOf", "valueType", "distinct"],
)
def test_primary_occurrence_exemption_precedes_vacuity_for_every_family(family: str) -> None:
    constraint = _constraint(family, selector="instance", exceptions=frozenset({1}))
    pre = _state({1: {10: {6}}, 6: {}})
    deleted, _ = apply_evidence_edits(pre, p_local={10}, delete=(1, 10, 6), add=None)
    reinserted, _ = apply_evidence_edits(
        pre, p_local={10}, delete=(1, 10, 6), add=(1, 10, 6)
    )
    assert evaluate_constraint_outcome(pre, constraint) == O.UNKNOWN
    assert evaluate_constraint_outcome(deleted, constraint) == O.UNKNOWN
    assert evaluate_constraint_outcome(reinserted, constraint) == O.UNKNOWN


def test_mixed_exempt_anchors_evaluate_only_nonexempt_anchor() -> None:
    constraint = _constraint("oneOf", exceptions=frozenset({1}))
    evidence = _state({1: {10: {6}}, 2: {10: {5}}})
    assert evaluate_constraint_outcome(evidence, constraint, primary=False) == O.SATISFIED


@pytest.mark.parametrize("remaining,expected", [(1, O.SATISFIED), (2, O.VIOLATED)])
def test_single_value_deletion_cardinality_controls(remaining: int, expected: O) -> None:
    values = set(range(5, 5 + remaining + 1))
    pre = _state({1: {10: values}})
    post, _ = apply_evidence_edits(pre, p_local={10}, delete=(1, 10, 5), add=None)
    assert evaluate_constraint_outcome(post, _constraint("single")) == expected


@pytest.mark.parametrize(
    ("family", "prop", "selector", "edited_predicate", "primary", "expected"),
    [
        ("itemRequiresStatement", 10, None, 20, True, O.SATISFIED),
        ("conflictWith", 10, None, 20, True, O.VIOLATED),
        ("type", 10, "subclass", 279, True, O.SATISFIED),
        ("type", 10, "instance", 31, True, O.SATISFIED),
        ("oneOf", 20, None, 20, False, O.SATISFIED),
    ],
)
def test_unresolved_delete_is_replayed_before_successful_addition(
    family: str,
    prop: int,
    selector: str | None,
    edited_predicate: int,
    primary: bool,
    expected: O,
) -> None:
    pre = _state({1: {10: {6}}})
    triple = (1, edited_predicate, 5)
    post, _ = apply_evidence_edits(pre, p_local={10}, delete=triple, add=triple)
    hierarchy = ClassHierarchy(parents={1: set(), 5: set()}, complete={1: True, 5: True})

    assert [(event.kind, event.applied) for event in post.edit_events] == [
        ("del", False),
        ("add", True),
    ]
    assert post.has_statement(*triple)
    assert evaluate_constraint_outcome(
        post,
        _constraint(family, prop=prop, selector=selector),
        hierarchy=hierarchy,
        primary=primary,
    ) == expected


@pytest.mark.parametrize(
    ("family", "values", "expected"),
    [
        ("single", {5}, O.SATISFIED),
        ("oneOf", {5}, O.SATISFIED),
        ("oneOf", {6}, O.VIOLATED),
    ],
)
def test_partial_add_on_known_other_subject_is_irrelevant_to_primary(
    family: str,
    values: set[int],
    expected: O,
) -> None:
    pre = _state({1: {10: values}})
    post, _ = apply_evidence_edits(pre, p_local={10}, delete=None, add=(2, 10, 0))

    assert evaluate_constraint_outcome(post, _constraint(family)) == expected
