#!/usr/bin/env python3
"""Shared parser and three-valued symbolic validator (semantics version 3)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


VALIDATOR_SEMANTICS_VERSION = 3
P2303 = "P2303"
P2305 = "P2305"
P2306 = "P2306"
P2308 = "P2308"
P2309 = "P2309"
P4155 = "P4155"
P4680 = "P4680"

SELECTOR_BY_ITEM = {
    "Q21503252": "instance",
    "Q21514624": "subclass",
    "Q30208840": "either",
}
# The expanded July-2018 registry uses the Q548... identifier. Q46466787
# is the later canonical item used by the current constraint documentation.
MAIN_VALUE_SCOPE_ITEMS = {"Q54828448", "Q46466787"}
SUPPORTED_FAMILIES = frozenset(
    {
        "conflictWith",
        "inverse",
        "symmetric",
        "itemRequiresStatement",
        "valueRequiresStatement",
        "oneOf",
        "single",
        "type",
        "valueType",
        "distinct",
    }
)


class ValidationOutcome(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


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
    direct = "http://www.wikidata.org/prop/direct/"
    entity = "http://www.wikidata.org/entity/"
    if value.startswith(direct):
        value = entity + value[len(direct) :]
    if value.startswith(entity):
        return value.rsplit("/", 1)[-1]
    return value


def normalize_property_id(raw: str | None) -> str | None:
    token = normalize_token(raw)
    if token and token.startswith("P") and token[1:].isdigit() and int(token[1:]) > 0:
        return token
    return None


def normalize_item_id(raw: str | None) -> str | None:
    token = normalize_token(raw)
    if token and token.startswith("Q") and token[1:].isdigit() and int(token[1:]) > 0:
        return token
    return None


@dataclass(frozen=True)
class UnresolvedEdit:
    """A requested edit that could not be projected into bounded evidence."""

    kind: str
    subject: int
    predicate: int
    object: int
    reason: str


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
    unresolved_edits: Tuple[UnresolvedEdit, ...] = ()
    applied_additions: frozenset[Tuple[int, int, int]] = frozenset()
    applied_deletions: frozenset[Tuple[int, int, int]] = frozenset()

    def entity_in_scope(self, entity_id: int) -> bool:
        return entity_id in self.facts_by_entity

    def property_complete(self, entity_id: int, predicate_id: int) -> bool:
        if self.edit_unknown(entity_id, predicate_id):
            return False
        if self.assume_complete:
            return self.entity_in_scope(entity_id)
        return predicate_id in self.predicates_present.get(entity_id, set())

    def has_property(self, entity_id: int, predicate_id: int) -> bool:
        return bool(self.facts_by_entity.get(entity_id, {}).get(predicate_id, set()))

    def values_for(self, entity_id: int, predicate_id: int) -> Set[int]:
        return self.facts_by_entity.get(entity_id, {}).get(predicate_id, set())

    def has_statement(self, entity_id: int, predicate_id: int, object_id: int) -> bool:
        return object_id in self.values_for(entity_id, predicate_id)

    def edit_unknown(self, entity_id: int, predicate_id: int) -> bool:
        return (entity_id, predicate_id) in self.missing_edits or any(
            edit.subject == entity_id and edit.predicate == predicate_id
            for edit in self.unresolved_edits
        )

    def focus_statement_present(self) -> bool:
        return self.has_statement(self.focus_subject, self.focus_predicate, self.focus_object)


@dataclass(frozen=True)
class ConstraintInstance:
    """Resolved constraint definition shared by every validator consumer."""

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
    selector: str | None = None
    exceptions: frozenset[int] = frozenset()
    value_restrictions: frozenset[int] = frozenset()
    definition_valid: bool = True
    invalid_reason: str | None = None
    p31_predicate: int = 0
    p279_predicate: int = 0


@dataclass(frozen=True)
class ValidationResult:
    """Three-valued outcome plus applicability and auditable uncertainty."""

    outcome: ValidationOutcome
    applicable: bool
    unknown_reasons: Tuple[str, ...] = ()


def _invalid(instance: ConstraintInstance, reason: str) -> ConstraintInstance:
    values = dict(instance.__dict__)
    values["definition_valid"] = False
    values["invalid_reason"] = reason
    return ConstraintInstance(**values)


def parse_constraint_instance(
    *,
    constraint_id: int,
    constraint_type: str,
    constraint_type_id: int,
    constrained_property_raw: str,
    param_predicates_raw: Sequence[str],
    param_objects_raw: Sequence[str],
    resolve_id: Callable[[str | None], int],
    p31_predicate: int = 0,
    p279_predicate: int = 0,
) -> ConstraintInstance:
    """Parse one registry definition according to Wikidata parameter roles."""

    base = ConstraintInstance(
        constraint_id=constraint_id,
        constraint_type=constraint_type,
        constraint_type_id=constraint_type_id,
        constrained_property=(
            resolve_id(constrained_property_raw)
            if normalize_property_id(constrained_property_raw) is not None
            else 0
        ),
        required_properties=set(),
        allowed_items=set(),
        allowed_classes=set(),
        relation_predicates=[],
        inverse_properties=[],
        conflict_properties=set(),
        p31_predicate=p31_predicate,
        p279_predicate=p279_predicate,
    )
    if len(param_predicates_raw) != len(param_objects_raw):
        return _invalid(base, "parameter predicate/object length mismatch")
    if normalize_property_id(constrained_property_raw) is None:
        return _invalid(base, "constrained property is not a valid property ID")
    if not base.constrained_property:
        return _invalid(base, "missing or unrepresentable constrained property")

    raw_values: dict[str, list[str]] = {}
    for predicate_raw, object_raw in zip(param_predicates_raw, param_objects_raw):
        predicate = normalize_property_id(predicate_raw)
        obj = normalize_token(object_raw)
        if predicate is None or obj is None:
            return _invalid(base, "malformed parameter")
        raw_values.setdefault(predicate, []).append(obj)

    scopes = raw_values.get(P4680, [])
    if scopes and any(scope not in MAIN_VALUE_SCOPE_ITEMS for scope in scopes):
        return _invalid(base, "constraint scope is not representable by main-value triples")

    def resolved_items(
        parameter: str,
        *,
        property_values: bool = False,
        unresolved_is_error: bool = True,
    ) -> tuple[set[int], str | None]:
        resolved: set[int] = set()
        for raw in raw_values.get(parameter, []):
            if property_values and normalize_property_id(raw) is None:
                return set(), f"{parameter} requires a property value"
            if not property_values and normalize_item_id(raw) is None:
                return set(), f"{parameter} requires an item value"
            value = resolve_id(raw)
            if not value:
                if unresolved_is_error:
                    return set(), f"{parameter} value is outside the fixed representation"
                continue
            resolved.add(value)
        return resolved, None

    properties, error = resolved_items(P2306, property_values=True)
    if error:
        return _invalid(base, error)
    # P2305 is optional for conflicts/requires constraints, but when it is
    # present it changes the meaning of the definition.  Dropping an
    # out-of-vocabulary value would silently turn a value-qualified rule into
    # an unqualified property rule, so fail the whole definition closed.
    values, error = resolved_items(P2305)
    if error:
        return _invalid(base, error)
    classes, error = resolved_items(P2308)
    if error:
        return _invalid(base, error)
    exceptions, error = resolved_items(P2303, unresolved_is_error=False)
    if error:
        return _invalid(base, error)

    selector_values = raw_values.get(P2309, [])
    selector: str | None = None
    if selector_values:
        if len(selector_values) != 1:
            return _invalid(base, "P2309 is singular")
        selector = SELECTOR_BY_ITEM.get(selector_values[0])
        if selector is None:
            return _invalid(base, "unsupported P2309 selector")

    instance = ConstraintInstance(
        constraint_id=constraint_id,
        constraint_type=constraint_type,
        constraint_type_id=constraint_type_id,
        constrained_property=base.constrained_property,
        required_properties=set(properties),
        allowed_items=set(values),
        allowed_classes=set(classes),
        relation_predicates=(
            [p31_predicate]
            if selector == "instance" and p31_predicate
            else [p279_predicate]
            if selector == "subclass" and p279_predicate
            else [value for value in (p31_predicate, p279_predicate) if value]
            if selector == "either"
            else []
        ),
        inverse_properties=list(properties),
        conflict_properties=set(properties),
        selector=selector,
        exceptions=frozenset(exceptions),
        value_restrictions=frozenset(values),
        p31_predicate=p31_predicate,
        p279_predicate=p279_predicate,
    )

    singular_property_families = {
        "conflictWith",
        "itemRequiresStatement",
        "valueRequiresStatement",
        "inverse",
    }
    if constraint_type in singular_property_families and len(raw_values.get(P2306, [])) != 1:
        return _invalid(instance, f"{constraint_type} requires exactly one P2306 value")
    if constraint_type in {"type", "valueType"}:
        if not classes:
            return _invalid(instance, f"{constraint_type} requires P2308")
        if len(selector_values) != 1:
            return _invalid(instance, f"{constraint_type} requires exactly one P2309 selector")
        if not p279_predicate or (selector in {"instance", "either"} and not p31_predicate):
            return _invalid(instance, "type selector predicates are not representable")
    if constraint_type == "oneOf" and not values:
        return _invalid(instance, "oneOf requires at least one P2305 value")
    if constraint_type in {"single", "distinct"} and raw_values.get(P4155):
        return _invalid(instance, "P4155 separators are not retained by the triple abstraction")
    if constraint_type not in SUPPORTED_FAMILIES:
        return _invalid(instance, "unsupported constraint family")
    return instance


class ClassHierarchyProtocol:
    def reachable(
        self,
        child: int,
        ancestors: Set[int],
        *,
        state: EvidenceState | None = None,
        p279_predicate: int = 0,
    ) -> ValidationOutcome:  # pragma: no cover
        raise NotImplementedError


def _aggregate(results: Iterable[ValidationOutcome]) -> ValidationOutcome:
    values = list(results)
    if not values:
        return ValidationOutcome.UNKNOWN
    if ValidationOutcome.VIOLATED in values:
        return ValidationOutcome.VIOLATED
    if all(value == ValidationOutcome.SATISFIED for value in values):
        return ValidationOutcome.SATISFIED
    return ValidationOutcome.UNKNOWN


def _subjects_with_property(state: EvidenceState, predicate: int) -> list[int]:
    return sorted(entity for entity, facts in state.facts_by_entity.items() if facts.get(predicate))


def _occurrences(state: EvidenceState, predicate: int) -> list[tuple[int, int]]:
    return sorted(
        (entity, value)
        for entity, facts in state.facts_by_entity.items()
        for value in facts.get(predicate, set())
    )


def _required_statement(
    state: EvidenceState,
    entity: int,
    predicate: int,
    allowed_values: frozenset[int],
) -> ValidationOutcome:
    if state.edit_unknown(entity, predicate) or not state.entity_in_scope(entity):
        return ValidationOutcome.UNKNOWN
    values = state.values_for(entity, predicate)
    if (values & allowed_values) if allowed_values else bool(values):
        return ValidationOutcome.SATISFIED
    return ValidationOutcome.VIOLATED if state.property_complete(entity, predicate) else ValidationOutcome.UNKNOWN


def _type_relation(
    state: EvidenceState,
    entity: int,
    constraint: ConstraintInstance,
    hierarchy: ClassHierarchyProtocol | None,
) -> ValidationOutcome:
    selector = constraint.selector
    allowed = set(constraint.allowed_classes)
    if selector is None or not allowed:
        return ValidationOutcome.UNKNOWN

    checks: list[ValidationOutcome] = []
    if selector in {"subclass", "either"}:
        if entity in allowed:
            return ValidationOutcome.SATISFIED
        checks.append(
            hierarchy.reachable(
                entity,
                allowed,
                state=state,
                p279_predicate=constraint.p279_predicate,
            )
            if hierarchy is not None
            else ValidationOutcome.UNKNOWN
        )
    if selector in {"instance", "either"}:
        p31 = constraint.p31_predicate
        if not p31 or state.edit_unknown(entity, p31) or not state.entity_in_scope(entity):
            checks.append(ValidationOutcome.UNKNOWN)
        else:
            classes = state.values_for(entity, p31)
            if not classes:
                checks.append(
                    ValidationOutcome.VIOLATED
                    if state.property_complete(entity, p31)
                    else ValidationOutcome.UNKNOWN
                )
            else:
                class_checks: list[ValidationOutcome] = []
                for cls in classes:
                    if cls in allowed:
                        return ValidationOutcome.SATISFIED
                    class_checks.append(
                        hierarchy.reachable(
                            cls,
                            allowed,
                            state=state,
                            p279_predicate=constraint.p279_predicate,
                        )
                        if hierarchy is not None
                        else ValidationOutcome.UNKNOWN
                    )
                if any(value == ValidationOutcome.SATISFIED for value in class_checks):
                    return ValidationOutcome.SATISFIED
                checks.append(
                    ValidationOutcome.VIOLATED
                    if class_checks and all(value == ValidationOutcome.VIOLATED for value in class_checks)
                    else ValidationOutcome.UNKNOWN
                )
    if any(value == ValidationOutcome.SATISFIED for value in checks):
        return ValidationOutcome.SATISFIED
    if checks and all(value == ValidationOutcome.VIOLATED for value in checks):
        return ValidationOutcome.VIOLATED
    return ValidationOutcome.UNKNOWN


def _evaluate_without_unresolved(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int] | None = None,
    *,
    hierarchy: ClassHierarchyProtocol | None = None,
    primary: bool = True,
) -> ValidationOutcome:
    """Evaluate a definition over its correctly bound local anchors."""

    del p_local
    if not constraint.definition_valid or not constraint.constrained_property:
        return ValidationOutcome.UNKNOWN
    family = constraint.constraint_type
    prop = constraint.constrained_property
    if family in {"conflictWith", "itemRequiresStatement", "valueRequiresStatement", "inverse"}:
        relevant = constraint.conflict_properties if family == "conflictWith" else constraint.required_properties
        if family == "inverse":
            relevant = set(constraint.inverse_properties)
        if len(relevant) != 1:
            return ValidationOutcome.UNKNOWN
    subject_families = {"conflictWith", "itemRequiresStatement", "single", "type"}
    occurrence_families = {
        "inverse",
        "symmetric",
        "valueRequiresStatement",
        "oneOf",
        "valueType",
        "distinct",
    }
    if family not in subject_families | occurrence_families:
        return ValidationOutcome.UNKNOWN
    if primary and state.focus_predicate != prop:
        return ValidationOutcome.UNKNOWN
    # P2303 exempts the entity being checked.  Apply this before primary
    # vacuity: deleting the last occurrence must not turn an exempt check into
    # an apparent success.
    if primary and state.focus_subject in constraint.exceptions:
        return ValidationOutcome.UNKNOWN

    if family in subject_families:
        anchors = [state.focus_subject] if primary else _subjects_with_property(state, prop)
        anchors = [entity for entity in anchors if entity and entity not in constraint.exceptions]
        if not anchors:
            return ValidationOutcome.UNKNOWN
        results: list[ValidationOutcome] = []
        for entity in anchors:
            if state.edit_unknown(entity, prop) or not state.entity_in_scope(entity):
                results.append(ValidationOutcome.UNKNOWN)
                continue
            if not state.has_property(entity, prop):
                results.append(
                    ValidationOutcome.SATISFIED
                    if primary and state.property_complete(entity, prop)
                    else ValidationOutcome.UNKNOWN
                )
                continue
            if family == "conflictWith":
                conflict = next(iter(constraint.conflict_properties))
                values = state.values_for(entity, conflict)
                violation = bool(values & constraint.value_restrictions) if constraint.value_restrictions else bool(values)
                if violation:
                    results.append(ValidationOutcome.VIOLATED)
                elif state.property_complete(entity, conflict):
                    results.append(ValidationOutcome.SATISFIED)
                else:
                    results.append(ValidationOutcome.UNKNOWN)
            elif family == "itemRequiresStatement":
                required = next(iter(constraint.required_properties))
                results.append(_required_statement(state, entity, required, constraint.value_restrictions))
            elif family == "single":
                results.append(
                    ValidationOutcome.SATISFIED
                    if len(state.values_for(entity, prop)) <= 1
                    else ValidationOutcome.VIOLATED
                )
            else:
                results.append(_type_relation(state, entity, constraint, hierarchy))
        return _aggregate(results)

    occurrences = _occurrences(state, prop)
    if primary:
        occurrences = [(s, o) for s, o in occurrences if s == state.focus_subject]
        if not occurrences:
            return (
                ValidationOutcome.SATISFIED
                if state.property_complete(state.focus_subject, prop)
                else ValidationOutcome.UNKNOWN
            )
    occurrences = [(s, o) for s, o in occurrences if s not in constraint.exceptions]
    if not occurrences:
        return ValidationOutcome.UNKNOWN

    results: list[ValidationOutcome] = []
    if family == "distinct":
        owners: dict[int, set[int]] = {}
        for subject, value in _occurrences(state, prop):
            # Exceptions remove anchors from checking, not their statements
            # from the evidence used while checking a non-exempt anchor.  This
            # matches WikibaseQualityConstraints' UniqueValueChecker.
            owners.setdefault(value, set()).add(subject)
        return _aggregate(
            ValidationOutcome.VIOLATED if len(owners.get(value, set())) > 1 else ValidationOutcome.SATISFIED
            for _subject, value in occurrences
        )

    for subject, value in occurrences:
        if state.edit_unknown(subject, prop):
            results.append(ValidationOutcome.UNKNOWN)
        elif family == "oneOf":
            results.append(ValidationOutcome.SATISFIED if value in constraint.allowed_items else ValidationOutcome.VIOLATED)
        elif family == "valueRequiresStatement":
            required = next(iter(constraint.required_properties))
            results.append(_required_statement(state, value, required, constraint.value_restrictions))
        elif family in {"inverse", "symmetric"}:
            inverse = prop if family == "symmetric" else next(iter(constraint.inverse_properties))
            if state.has_statement(value, inverse, subject):
                results.append(ValidationOutcome.SATISFIED)
            elif state.property_complete(value, inverse):
                results.append(ValidationOutcome.VIOLATED)
            else:
                results.append(ValidationOutcome.UNKNOWN)
        else:
            results.append(_type_relation(state, value, constraint, hierarchy))
    return _aggregate(results)


def _relevant_predicates(constraint: ConstraintInstance) -> set[int]:
    predicates = {constraint.constrained_property}
    predicates.update(constraint.required_properties)
    predicates.update(constraint.inverse_properties)
    predicates.update(constraint.conflict_properties)
    if constraint.constraint_type in {"type", "valueType"}:
        predicates.update(value for value in (constraint.p31_predicate, constraint.p279_predicate) if value)
    return {value for value in predicates if value}


def _force_edit(state: EvidenceState, edit: UnresolvedEdit) -> EvidenceState:
    """Return one possible world in which an unresolved operation succeeded."""

    facts = {
        entity: {predicate: set(values) for predicate, values in entity_facts.items()}
        for entity, entity_facts in state.facts_by_entity.items()
    }
    present = {entity: set(predicates) for entity, predicates in state.predicates_present.items()}
    additions = set(getattr(state, "applied_additions", frozenset()))
    deletions = set(getattr(state, "applied_deletions", frozenset()))
    triple = (edit.subject, edit.predicate, edit.object)
    if edit.kind == "add":
        facts.setdefault(edit.subject, {}).setdefault(edit.predicate, set()).add(edit.object)
        present.setdefault(edit.subject, set()).add(edit.predicate)
        additions.add(triple)
        deletions.discard(triple)
    elif edit.kind == "del":
        facts.setdefault(edit.subject, {}).setdefault(edit.predicate, set()).discard(edit.object)
        deletions.add(triple)
        additions.discard(triple)
    return replace(
        state,
        facts_by_entity=facts,
        predicates_present=present,
        missing_edits=set(),
        unresolved_edits=(),
        applied_additions=frozenset(additions),
        applied_deletions=frozenset(deletions),
    )


def _applicable(state: EvidenceState, constraint: ConstraintInstance, *, primary: bool) -> bool:
    if not constraint.definition_valid or not constraint.constrained_property:
        return False
    if primary:
        return state.focus_predicate == constraint.constrained_property and state.focus_subject not in constraint.exceptions
    if any(subject not in constraint.exceptions for subject, _value in _occurrences(state, constraint.constrained_property)):
        return True
    return any(
        edit.kind == "add"
        and edit.predicate == constraint.constrained_property
        and edit.subject not in constraint.exceptions
        for edit in getattr(state, "unresolved_edits", ())
    )


def _legacy_missing_leaves_a_violation(
    state: EvidenceState,
    constraint: ConstraintInstance,
    missing: set[tuple[int, int]],
    *,
    primary: bool,
) -> bool:
    """Recognize a known violation independent of pair-only legacy metadata."""

    family = constraint.constraint_type
    prop = constraint.constrained_property
    subject_anchors = [state.focus_subject] if primary else _subjects_with_property(state, prop)
    subject_anchors = [anchor for anchor in subject_anchors if anchor not in constraint.exceptions]
    for anchor in subject_anchors:
        if (anchor, prop) in missing:
            continue
        if family == "conflictWith":
            conflict = next(iter(constraint.conflict_properties))
            values = state.values_for(anchor, conflict)
            violates = bool(values & constraint.value_restrictions) if constraint.value_restrictions else bool(values)
            if violates and (anchor, conflict) not in missing:
                return True
        elif family == "itemRequiresStatement":
            required = next(iter(constraint.required_properties))
            values = state.values_for(anchor, required)
            has_required = bool(values & constraint.value_restrictions) if constraint.value_restrictions else bool(values)
            if not has_required and state.property_complete(anchor, required) and (anchor, required) not in missing:
                return True
        elif family == "single" and len(state.values_for(anchor, prop)) > 1:
            return True

    occurrences = _occurrences(state, prop)
    if primary:
        occurrences = [(subject, value) for subject, value in occurrences if subject == state.focus_subject]
    occurrences = [(subject, value) for subject, value in occurrences if subject not in constraint.exceptions]
    if family == "oneOf":
        return any(value not in constraint.allowed_items and (subject, prop) not in missing for subject, value in occurrences)
    if family == "valueRequiresStatement":
        required = next(iter(constraint.required_properties))
        for subject, value in occurrences:
            required_values = state.values_for(value, required)
            has_required = (
                bool(required_values & constraint.value_restrictions)
                if constraint.value_restrictions
                else bool(required_values)
            )
            if (
                not has_required
                and state.property_complete(value, required)
                and (subject, prop) not in missing
                and (value, required) not in missing
            ):
                return True
    if family in {"inverse", "symmetric"}:
        inverse = prop if family == "symmetric" else next(iter(constraint.inverse_properties))
        return any(
            not state.has_statement(value, inverse, subject)
            and state.property_complete(value, inverse)
            and (subject, prop) not in missing
            and (value, inverse) not in missing
            for subject, value in occurrences
        )
    if family == "distinct":
        owners: dict[int, set[int]] = {}
        for subject, value in _occurrences(state, prop):
            owners.setdefault(value, set()).add(subject)
        for subject, value in occurrences:
            stable_owners = {
                owner for owner in owners.get(value, set()) if (owner, prop) not in missing
            }
            if subject in stable_owners and len(stable_owners) > 1:
                return True
    # Pair-only ancestry uncertainty cannot safely be localized through an
    # arbitrary P279 path, so type-family violations remain unknown.
    return False


def evaluate_constraint_detailed(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int] | None = None,
    *,
    hierarchy: ClassHierarchyProtocol | None = None,
    primary: bool = True,
) -> ValidationResult:
    """Evaluate a definition and retain why a result is not checkable.

    Unresolved edits are handled as bounded possible worlds.  A definite value
    is returned only when the projected state and every relevant successful-edit
    world agree.  This preserves unaffected violations without certifying an
    edit whose unresolved effect could change the result.
    """

    del p_local
    applicable = _applicable(state, constraint, primary=primary)
    if not constraint.definition_valid or not constraint.constrained_property:
        return ValidationResult(
            ValidationOutcome.UNKNOWN,
            False,
            (constraint.invalid_reason or "invalid constraint definition",),
        )
    if primary and state.focus_predicate != constraint.constrained_property:
        return ValidationResult(ValidationOutcome.UNKNOWN, False, ("primary property mismatch",))
    if primary and state.focus_subject in constraint.exceptions:
        return ValidationResult(ValidationOutcome.UNKNOWN, False, ("primary anchor is exempt",))

    outcome = _evaluate_without_unresolved(state, constraint, hierarchy=hierarchy, primary=primary)
    relevant_predicates = _relevant_predicates(constraint)
    detailed = tuple(
        edit
        for edit in getattr(state, "unresolved_edits", ())
        if not edit.predicate or edit.predicate in relevant_predicates
    )
    described_pairs = {(edit.subject, edit.predicate) for edit in getattr(state, "unresolved_edits", ())}
    legacy_relevant = {
        pair
        for pair in state.missing_edits
        if pair[1] in relevant_predicates and pair not in described_pairs
    }
    if legacy_relevant:
        if outcome == ValidationOutcome.VIOLATED and _legacy_missing_leaves_a_violation(
            state,
            constraint,
            legacy_relevant,
            primary=primary,
        ):
            return ValidationResult(ValidationOutcome.VIOLATED, applicable)
        return ValidationResult(
            ValidationOutcome.UNKNOWN,
            applicable,
            ("relevant unresolved edit lacks operation details",),
        )

    if detailed:
        if any(not edit.subject or not edit.predicate or not edit.object for edit in detailed):
            return ValidationResult(
                ValidationOutcome.UNKNOWN,
                applicable,
                ("relevant unresolved edit is malformed or partial",),
            )
        clean_state = replace(
            state,
            missing_edits=set(state.missing_edits) - described_pairs,
            unresolved_edits=(),
        )
        outcome = _evaluate_without_unresolved(
            clean_state,
            constraint,
            hierarchy=hierarchy,
            primary=primary,
        )
        possible = {outcome}
        # A benchmark candidate has at most one add and one delete.  Enumerate
        # every success/failure combination without mutating the shared state.
        worlds = [clean_state]
        for edit in detailed:
            worlds += [_force_edit(world, edit) for world in list(worlds)]
        possible.update(
            _evaluate_without_unresolved(world, constraint, hierarchy=hierarchy, primary=primary)
            for world in worlds
        )
        if len(possible) != 1:
            reasons = tuple(
                sorted({f"unresolved {edit.kind} affects {edit.predicate}: {edit.reason}" for edit in detailed})
            )
            return ValidationResult(ValidationOutcome.UNKNOWN, applicable, reasons)
        outcome = next(iter(possible))

    reasons: Tuple[str, ...] = ()
    if outcome == ValidationOutcome.UNKNOWN:
        if not applicable:
            reasons = ("no applicable non-exempt anchor",)
        else:
            reasons = ("incomplete local or hierarchy evidence",)
    return ValidationResult(outcome, applicable, reasons)


def evaluate_constraint_outcome(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int] | None = None,
    *,
    hierarchy: ClassHierarchyProtocol | None = None,
    primary: bool = True,
) -> ValidationOutcome:
    """Compatibility wrapper returning only the semantics-v3 outcome."""

    return evaluate_constraint_detailed(
        state,
        constraint,
        p_local,
        hierarchy=hierarchy,
        primary=primary,
    ).outcome


def evaluate_constraint(
    state: EvidenceState,
    constraint: ConstraintInstance,
    p_local: Set[int] | None = None,
    *,
    hierarchy: ClassHierarchyProtocol | None = None,
    primary: bool = True,
) -> Tuple[bool, int]:
    """Compatibility result: ``(checkable, satisfied)`` from v3 outcome."""

    outcome = evaluate_constraint_outcome(state, constraint, p_local, hierarchy=hierarchy, primary=primary)
    if outcome == ValidationOutcome.UNKNOWN:
        return False, 0
    return True, int(outcome == ValidationOutcome.SATISFIED)


# Kept only so old imports fail closed instead of dispatching to v1 semantics.
CHECKERS: Mapping[str, tuple[Callable[..., bool], Callable[..., bool]]] = {}
