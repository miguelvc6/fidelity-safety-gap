from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.class_hierarchy import ClassHierarchy
from modules.constraint_checkers import (
    ConstraintInstance,
    EvidenceState,
    ValidationOutcome,
    evaluate_constraint,
    evaluate_constraint_outcome,
    normalize_token,
    parse_constraint_instance,
)
from modules.evidence_state import apply_evidence_edits, build_pre_state
from modules.reranker_eval import _ReusableEvidenceState


IDS = {
    "P10": 10,
    "P20": 20,
    "P31": 31,
    "P279": 279,
    "Q5": 5,
    "Q6": 6,
    "Q7": 7,
    "Q8": 8,
    "Q9": 9,
    "Q21503252": 1001,
    "Q21514624": 1002,
    "Q30208840": 1003,
    "Q54828448": 1004,
    "Q54828449": 1005,
}


def resolve(raw: str | None) -> int:
    return IDS.get(normalize_token(raw) or "", 0)


def definition(family: str, parameters: list[tuple[str, str]] = ()) -> ConstraintInstance:
    return parse_constraint_instance(
        constraint_id=900,
        constraint_type=family,
        constraint_type_id=1,
        constrained_property_raw="P10",
        param_predicates_raw=[predicate for predicate, _value in parameters],
        param_objects_raw=[value for _predicate, value in parameters],
        resolve_id=resolve,
        p31_predicate=31,
        p279_predicate=279,
    )


def state(
    facts: dict[int, dict[int, set[int]]],
    *,
    complete: bool = True,
    missing: set[tuple[int, int]] | None = None,
) -> EvidenceState:
    return EvidenceState(
        facts_by_entity=facts,
        predicates_present={entity: set(values) for entity, values in facts.items()},
        assume_complete=complete,
        missing_edits=missing or set(),
        focus_subject=1,
        focus_predicate=10,
        focus_object=5,
        other_subject=2,
        other_predicate=10,
        other_object=5,
    )


@pytest.mark.parametrize(
    ("family", "parameters", "facts", "primary", "expected"),
    [
        ("conflictWith", [("P2306", "P20")], {1: {10: {5}, 20: {6}}}, True, ValidationOutcome.VIOLATED),
        ("itemRequiresStatement", [("P2306", "P20")], {1: {10: {5}, 20: {6}}}, True, ValidationOutcome.SATISFIED),
        ("valueRequiresStatement", [("P2306", "P20")], {1: {10: {5}}, 5: {20: {6}}}, True, ValidationOutcome.SATISFIED),
        ("inverse", [("P2306", "P20")], {1: {10: {5}}, 5: {20: {1}}}, True, ValidationOutcome.SATISFIED),
        ("symmetric", [], {1: {10: {5}}, 5: {10: {1}}}, True, ValidationOutcome.SATISFIED),
        ("oneOf", [("P2305", "Q5")], {1: {10: {5}}}, True, ValidationOutcome.SATISFIED),
        ("single", [], {1: {10: {5, 6}}}, True, ValidationOutcome.VIOLATED),
        ("distinct", [], {1: {10: {5}}, 2: {10: {5}}}, False, ValidationOutcome.VIOLATED),
    ],
)
def test_family_semantics(family, parameters, facts, primary, expected) -> None:
    assert evaluate_constraint_outcome(state(facts), definition(family, parameters), primary=primary) == expected


def test_optional_value_restrictions_apply_to_conflict_and_required() -> None:
    conflict = definition("conflictWith", [("P2306", "P20"), ("P2305", "Q7")])
    required = definition("itemRequiresStatement", [("P2306", "P20"), ("P2305", "Q7")])
    evidence = state({1: {10: {5}, 20: {6}}})
    assert evaluate_constraint_outcome(evidence, conflict) == ValidationOutcome.SATISFIED
    assert evaluate_constraint_outcome(evidence, required) == ValidationOutcome.VIOLATED

    value_required = definition("valueRequiresStatement", [("P2306", "P20"), ("P2305", "Q7")])
    target_evidence = state({1: {10: {5}}, 5: {20: {6}}})
    assert evaluate_constraint_outcome(target_evidence, value_required) == ValidationOutcome.VIOLATED


@pytest.mark.parametrize(
    ("selector", "entity_facts", "expected"),
    [
        ("Q21503252", {1: {10: {5}, 31: {6}}}, ValidationOutcome.SATISFIED),
        ("Q21514624", {1: {10: {5}}}, ValidationOutcome.VIOLATED),
        ("Q30208840", {1: {10: {5}, 31: {6}}}, ValidationOutcome.SATISFIED),
    ],
)
def test_subject_type_selectors(selector, entity_facts, expected) -> None:
    constraint = definition("type", [("P2308", "Q5"), ("P2309", selector)])
    hierarchy = ClassHierarchy(parents={1: set(), 6: {5}, 5: set()}, complete={1: True, 5: True, 6: True})
    assert evaluate_constraint_outcome(state(entity_facts), constraint, hierarchy=hierarchy) == expected


def test_value_type_uses_each_replacement_value_and_transitive_hierarchy() -> None:
    constraint = definition("valueType", [("P2308", "Q5"), ("P2309", "Q21514624")])
    hierarchy = ClassHierarchy(
        parents={7: {6}, 6: {5}, 5: set(), 8: set()},
        complete={5: True, 6: True, 7: True, 8: True},
    )
    pre = state({1: {10: {8}}})
    post, _ = apply_evidence_edits(pre, p_local={10}, delete=(1, 10, 8), add=(1, 10, 7))
    assert evaluate_constraint_outcome(pre, constraint, hierarchy=hierarchy) == ValidationOutcome.VIOLATED
    assert evaluate_constraint_outcome(post, constraint, hierarchy=hierarchy) == ValidationOutcome.SATISFIED


def test_parser_rejects_missing_multiple_malformed_scope_and_separator() -> None:
    assert not definition("inverse").definition_valid
    assert not definition("inverse", [("P2306", "P20"), ("P2306", "P10")]).definition_valid
    assert not definition("inverse", [("P2306", "Q5")]).definition_valid
    assert not definition("single", [("P4155", "Q5")]).definition_valid
    assert not definition("single", [("P4680", "Q54828449")]).definition_valid


def test_parser_does_not_drop_unrepresentable_value_restrictions() -> None:
    constraint = definition(
        "itemRequiresStatement",
        [("P2306", "P20"), ("P2305", "Q999999")],
    )
    assert not constraint.definition_valid
    assert constraint.invalid_reason == "P2305 value is outside the fixed representation"


def test_exception_removes_anchor_and_all_exempt_is_unknown() -> None:
    exempt = definition("single", [("P2303", "Q5")])
    exempt = ConstraintInstance(**{**exempt.__dict__, "exceptions": frozenset({1})})
    assert evaluate_constraint_outcome(state({1: {10: {5, 6}}}), exempt) == ValidationOutcome.UNKNOWN


def test_distinct_ignores_exempt_competing_subjects() -> None:
    constraint = definition("distinct", [("P2303", "Q5")])
    constraint = ConstraintInstance(**{**constraint.__dict__, "exceptions": frozenset({2})})
    evidence = state({1: {10: {5}}, 2: {10: {5}}})
    assert evaluate_constraint_outcome(evidence, constraint, primary=False) == ValidationOutcome.SATISFIED


def test_multi_anchor_aggregation_and_secondary_binding() -> None:
    constraint = definition("single")
    evidence = state({1: {10: {5}}, 2: {10: {6, 7}}})
    assert evaluate_constraint_outcome(evidence, constraint, primary=True) == ValidationOutcome.SATISFIED
    assert evaluate_constraint_outcome(evidence, constraint, primary=False) == ValidationOutcome.VIOLATED


def test_unknown_anchor_mixed_with_satisfied_is_unknown() -> None:
    constraint = definition("itemRequiresStatement", [("P2306", "P20")])
    evidence = state({1: {10: {5}, 20: {6}}, 2: {10: {7}}}, complete=False)
    assert evaluate_constraint_outcome(evidence, constraint, primary=False) == ValidationOutcome.UNKNOWN


def test_delete_then_reinsert_and_unresolved_edit() -> None:
    constraint = definition("oneOf", [("P2305", "Q5")])
    pre = state({1: {10: {5}}})
    same, _ = apply_evidence_edits(pre, p_local={10}, delete=(1, 10, 5), add=(1, 10, 5))
    assert same.focus_statement_present()
    assert evaluate_constraint(same, constraint) == (True, 1)
    unresolved, _ = apply_evidence_edits(pre, p_local={10}, delete=(9, 10, 5), add=None)
    assert (9, 10) in unresolved.missing_edits


def test_candidate_edit_does_not_mutate_shared_predicate_scope() -> None:
    pre = state({1: {10: {5}}})
    p_local = {10}
    apply_evidence_edits(pre, p_local=p_local, delete=None, add=(1, 20, 6))
    assert p_local == {10}


def test_reusable_candidate_state_treats_failed_edit_as_incomplete() -> None:
    reusable = _ReusableEvidenceState(
        facts_by_entity={1: {10: {5}}},
        predicates_present={1: {10}},
        assume_complete=True,
        missing_edits={(1, 20)},
        focus_subject=1,
        focus_predicate=10,
        focus_object=5,
        other_subject=0,
        other_predicate=0,
        other_object=0,
    )
    assert not reusable.property_complete(1, 20)


def test_primary_deletion_is_vacuously_satisfied() -> None:
    constraint = definition("oneOf", [("P2305", "Q5")])
    pre = state({1: {10: {5}}})
    post, _ = apply_evidence_edits(pre, p_local={10}, delete=(1, 10, 5), add=None)
    assert evaluate_constraint_outcome(post, constraint, primary=True) == ValidationOutcome.SATISFIED


def test_explicit_other_statement_and_correct_auxiliary_role_are_retained() -> None:
    row = SimpleNamespace(
        subject=1,
        predicate=10,
        object=2,
        other_subject=1,
        other_predicate=20,
        other_object=3,
        subject_predicates=[],
        subject_objects=[],
        object_predicates=[],
        object_objects=[],
        other_entity_predicates=[31],
        other_entity_objects=[6],
    )
    evidence, _ = build_pre_state(row, assume_complete=True, cast_int=True)
    assert evidence.has_statement(1, 20, 3)
    assert evidence.has_statement(3, 31, 6)
