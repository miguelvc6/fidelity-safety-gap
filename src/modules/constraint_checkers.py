#!/usr/bin/env python3
"""
Constraint checking utilities for the 05_constraint_labeler stage.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


def normalize_token(raw: str | None) -> str | None:
    """Normalize Wikidata tokens to bare Q/P ids when possible."""
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.startswith("^"):
        value = value[1:].strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    if value.startswith("http://www.wikidata.org/prop/direct/"):
        value = value.replace("http://www.wikidata.org/prop/direct/", "http://www.wikidata.org/entity/")
    if value.startswith("http://www.wikidata.org/entity/"):
        return value.rsplit("/", 1)[-1]
    return value


def normalize_property_id(raw: str | None) -> str | None:
    token = normalize_token(raw)
    if token and token.startswith("P") and token[1:].isdigit():
        return token
    return None


@dataclass(frozen=True)
class EvidenceState:
    facts_by_entity: Dict[int, Dict[int, Set[int]]]
    predicates_present: Dict[int, Set[int]]
    assume_complete: bool
    missing_edits: Set[Tuple[int, int]]
    focus_subject: int
    focus_predicate: int
    focus_object: int
    other_subject: int
    other_predicate: int
    other_object: int

    def entity_in_scope(self, entity_id: int) -> bool:
        return entity_id in self.facts_by_entity

    def property_complete(self, entity_id: int, predicate_id: int) -> bool:
        if self.assume_complete:
            return self.entity_in_scope(entity_id)
        return predicate_id in self.predicates_present.get(entity_id, set())

    def has_property(self, entity_id: int, predicate_id: int) -> bool:
        return len(self.facts_by_entity.get(entity_id, {}).get(predicate_id, set())) > 0

    def values_for(self, entity_id: int, predicate_id: int) -> Set[int]:
        return self.facts_by_entity.get(entity_id, {}).get(predicate_id, set())

    def has_statement(self, entity_id: int, predicate_id: int, object_id: int) -> bool:
        return object_id in self.facts_by_entity.get(entity_id, {}).get(predicate_id, set())

    def edit_unknown(self, entity_id: int, predicate_id: int) -> bool:
        return (entity_id, predicate_id) in self.missing_edits

    def focus_statement_present(self) -> bool:
        return self.has_statement(self.focus_subject, self.focus_predicate, self.focus_object)


@dataclass(frozen=True)
class ConstraintInstance:
    constraint_id: int
    constraint_type: str
    constraint_type_id: int
    constrained_property: int
    required_properties: Set[int]
    allowed_items: Set[int]
    allowed_classes: Set[int]
    relation_predicates: List[int]
    inverse_properties: List[int]
    conflict_properties: Set[int]


def _needs_edit_guard(state: EvidenceState, entity_id: int, predicate_id: int) -> bool:
    return state.edit_unknown(entity_id, predicate_id)


def is_checkable_conflict_with(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    if subject == 0 or constraint.constrained_property == 0:
        return False
    if _needs_edit_guard(state, subject, constraint.constrained_property):
        return False
    if constraint.constrained_property not in p_local:
        return False

    conflict_props = set(constraint.conflict_properties)
    if not conflict_props and state.other_predicate:
        conflict_props.add(state.other_predicate)
    if not conflict_props:
        return False

    if not state.entity_in_scope(subject):
        return False

    for prop in conflict_props:
        if prop == 0:
            continue
        if prop not in p_local:
            return False
        if _needs_edit_guard(state, subject, prop):
            return False
        if not state.property_complete(subject, prop) and not state.has_property(subject, prop):
            return False

    if not state.property_complete(subject, constraint.constrained_property) and not state.has_property(
        subject, constraint.constrained_property
    ):
        return False
    return True


def is_satisfied_conflict_with(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    if subject == 0:
        return True
    has_p = state.has_property(subject, constraint.constrained_property)
    if not has_p:
        return True
    conflict_props = set(constraint.conflict_properties)
    if not conflict_props and state.other_predicate:
        conflict_props.add(state.other_predicate)
    has_q = False
    for prop in conflict_props:
        if prop == 0:
            continue
        if state.has_property(subject, prop):
            has_q = True
    return not (has_p and has_q)


def is_checkable_inverse(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    predicate = state.focus_predicate
    obj = state.focus_object
    if subject == 0 or predicate == 0:
        return False
    if predicate != constraint.constrained_property:
        return False
    if _needs_edit_guard(state, subject, constraint.constrained_property):
        return False
    has_trigger = state.focus_statement_present()
    if not has_trigger:
        return state.property_complete(subject, constraint.constrained_property)
    if obj == 0:
        return False
    if not state.entity_in_scope(obj):
        return False
    inverse_props = constraint.inverse_properties or [constraint.constrained_property]
    for inv_prop in inverse_props:
        if inv_prop == 0:
            return False
        if inv_prop not in p_local:
            return False
        if _needs_edit_guard(state, obj, inv_prop):
            return False
        if not state.property_complete(obj, inv_prop) and not state.has_property(obj, inv_prop):
            return False
    return True


def is_satisfied_inverse(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    obj = state.focus_object
    if not state.focus_statement_present():
        return True
    inverse_props = constraint.inverse_properties or [constraint.constrained_property]
    for inv_prop in inverse_props:
        if state.has_statement(obj, inv_prop, subject):
            return True
    return False


def is_checkable_item_requires_statement(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    if subject == 0:
        return False
    if not constraint.required_properties:
        return False
    if constraint.constrained_property == 0:
        return False
    if constraint.constrained_property not in p_local:
        return False
    if _needs_edit_guard(state, subject, constraint.constrained_property):
        return False
    if not state.entity_in_scope(subject):
        return False
    has_trigger = state.has_property(subject, constraint.constrained_property)
    if not has_trigger:
        return state.property_complete(subject, constraint.constrained_property)
    for req_prop in constraint.required_properties:
        if req_prop == 0:
            return False
        if req_prop not in p_local:
            return False
        if _needs_edit_guard(state, subject, req_prop):
            return False
        if not state.property_complete(subject, req_prop) and not state.has_property(subject, req_prop):
            return False
    return True


def is_satisfied_item_requires_statement(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    if not state.has_property(subject, constraint.constrained_property):
        return True
    for req_prop in constraint.required_properties:
        if not state.has_property(subject, req_prop):
            return False
    return True


def is_checkable_value_requires_statement(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    obj = state.focus_object
    if constraint.constrained_property == 0:
        return False
    if state.focus_predicate != constraint.constrained_property:
        return False
    if _needs_edit_guard(state, subject, constraint.constrained_property):
        return False
    has_trigger = state.focus_statement_present()
    if not has_trigger:
        return state.property_complete(subject, constraint.constrained_property)
    if obj == 0:
        return False
    if not constraint.required_properties:
        return False
    if not state.entity_in_scope(obj):
        return False
    for req_prop in constraint.required_properties:
        if req_prop == 0:
            return False
        if req_prop not in p_local:
            return False
        if _needs_edit_guard(state, obj, req_prop):
            return False
        if not state.property_complete(obj, req_prop) and not state.has_property(obj, req_prop):
            return False
    return True


def is_satisfied_value_requires_statement(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    obj = state.focus_object
    if not state.focus_statement_present():
        return True
    for req_prop in constraint.required_properties:
        if not state.has_property(obj, req_prop):
            return False
    return True


def is_checkable_one_of(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    if not constraint.allowed_items:
        return False
    if state.focus_predicate != constraint.constrained_property:
        return False
    if constraint.constrained_property not in p_local:
        return False
    if _needs_edit_guard(state, state.focus_subject, constraint.constrained_property):
        return False
    has_trigger = state.focus_statement_present()
    if not has_trigger:
        return state.property_complete(state.focus_subject, constraint.constrained_property)
    if state.focus_object == 0:
        return False
    return True


def is_satisfied_one_of(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    if not state.focus_statement_present():
        return True
    return state.focus_object in constraint.allowed_items


def is_checkable_single(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    prop = constraint.constrained_property
    if subject == 0 or prop == 0:
        return False
    if prop not in p_local:
        return False
    if _needs_edit_guard(state, subject, prop):
        return False
    if not state.entity_in_scope(subject):
        return False
    if not state.property_complete(subject, prop):
        return False
    return True


def is_satisfied_single(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    prop = constraint.constrained_property
    return len(state.values_for(subject, prop)) <= 1


def _type_relation_predicates(constraint: ConstraintInstance) -> List[int]:
    return constraint.relation_predicates or []


def is_checkable_type(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    if subject == 0:
        return False
    if constraint.constrained_property == 0:
        return False
    if constraint.constrained_property not in p_local:
        return False
    if _needs_edit_guard(state, subject, constraint.constrained_property):
        return False
    if not constraint.allowed_classes:
        return False
    if not state.entity_in_scope(subject):
        return False
    has_trigger = state.has_property(subject, constraint.constrained_property)
    if not has_trigger:
        return state.property_complete(subject, constraint.constrained_property)
    rel_preds = _type_relation_predicates(constraint)
    if not rel_preds:
        return False
    for rel in rel_preds:
        if rel not in p_local:
            return False
        if _needs_edit_guard(state, subject, rel):
            return False
        if not state.property_complete(subject, rel) and not state.has_property(subject, rel):
            return False
    if not any(state.has_property(subject, rel) for rel in rel_preds):
        return False
    return True


def is_satisfied_type(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    if not state.has_property(subject, constraint.constrained_property):
        return True
    rel_preds = _type_relation_predicates(constraint)
    for rel in rel_preds:
        if state.values_for(subject, rel) & constraint.allowed_classes:
            return True
    return False


def is_checkable_value_type(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    obj = state.focus_object
    if obj == 0:
        return False
    if constraint.constrained_property == 0:
        return False
    if constraint.constrained_property not in p_local:
        return False
    if _needs_edit_guard(state, state.focus_subject, constraint.constrained_property):
        return False
    if not constraint.allowed_classes:
        return False
    has_trigger = state.has_property(state.focus_subject, constraint.constrained_property)
    if not has_trigger:
        return state.property_complete(state.focus_subject, constraint.constrained_property)
    if not state.entity_in_scope(obj):
        return False
    rel_preds = _type_relation_predicates(constraint)
    if not rel_preds:
        return False
    for rel in rel_preds:
        if rel not in p_local:
            return False
        if _needs_edit_guard(state, obj, rel):
            return False
        if not state.property_complete(obj, rel) and not state.has_property(obj, rel):
            return False
    if not any(state.has_property(obj, rel) for rel in rel_preds):
        return False
    return True


def is_satisfied_value_type(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    obj = state.focus_object
    if not state.has_property(state.focus_subject, constraint.constrained_property):
        return True
    rel_preds = _type_relation_predicates(constraint)
    for rel in rel_preds:
        if state.values_for(obj, rel) & constraint.allowed_classes:
            return True
    return False


def is_checkable_distinct(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    subject = state.focus_subject
    prop = constraint.constrained_property
    if subject == 0 or prop == 0:
        return False
    if prop not in p_local:
        return False
    if _needs_edit_guard(state, subject, prop):
        return False
    has_trigger = state.focus_statement_present()
    if not has_trigger:
        return state.property_complete(subject, prop)
    if state.other_subject == 0 or state.other_predicate == 0 or state.other_object == 0:
        return False
    if state.other_predicate != prop:
        return False
    if state.other_predicate not in p_local:
        return False
    return True


def is_satisfied_distinct(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> bool:
    if not state.focus_statement_present():
        return True
    if state.other_subject == 0 or state.other_predicate == 0 or state.other_object == 0:
        return True
    if state.other_predicate != constraint.constrained_property:
        return True
    if state.other_object != state.focus_object:
        return True
    if state.other_subject == state.focus_subject:
        return True
    return False


CHECKERS = {
    "conflictWith": (is_checkable_conflict_with, is_satisfied_conflict_with),
    "inverse": (is_checkable_inverse, is_satisfied_inverse),
    "symmetric": (is_checkable_inverse, is_satisfied_inverse),
    "itemRequiresStatement": (is_checkable_item_requires_statement, is_satisfied_item_requires_statement),
    "valueRequiresStatement": (is_checkable_value_requires_statement, is_satisfied_value_requires_statement),
    "oneOf": (is_checkable_one_of, is_satisfied_one_of),
    "single": (is_checkable_single, is_satisfied_single),
    "type": (is_checkable_type, is_satisfied_type),
    "valueType": (is_checkable_value_type, is_satisfied_value_type),
    "distinct": (is_checkable_distinct, is_satisfied_distinct),
}


def evaluate_constraint(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int],
) -> Tuple[bool, int]:
    """Return (checkable, satisfied) for a constraint instance."""
    checker = CHECKERS.get(constraint.constraint_type)
    if checker is None:
        return False, 0
    is_checkable, is_satisfied = checker
    checkable = is_checkable(state, constraint, p_local)
    if not checkable:
        return False, 0
    satisfied = is_satisfied(state, constraint, p_local)
    return True, 1 if satisfied else 0
