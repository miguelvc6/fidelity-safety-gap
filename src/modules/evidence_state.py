"""Canonical construction and editing of local symbolic evidence states.

The benchmark stores the same entity through several graph roles (focus subject,
focus object, and an auxiliary entity).  Those role projections are views over
one local fact set, not independent entities.  This module is deliberately
shared by future label generation and paper-metric evaluation so the two paths
cannot silently drift again.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from modules.constraint_checkers import EvidenceState


EMPTY_VALUES = (None, "", 0)


def coerce_sequence(value: Any, *, cast_int: bool = True) -> list[Any]:
    """Flatten common parquet sequence payloads into a Python list."""

    def _flatten(item: Any) -> list[Any]:
        if item is None:
            return []
        if isinstance(item, np.ndarray):
            return _flatten(item.tolist())
        if isinstance(item, (list, tuple)):
            flattened: list[Any] = []
            for nested in item:
                flattened.extend(_flatten(nested))
            return flattened
        return [item]

    values = _flatten(value)
    if not cast_int:
        return values
    coerced: list[Any] = []
    for item in values:
        try:
            coerced.append(int(item))
        except (TypeError, ValueError):
            coerced.append(0)
    return coerced


def coerce_value(value: Any, *, cast_int: bool = True) -> Any:
    if not cast_int:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def compute_p_local(row: Any, *, cast_int: bool = True) -> set[Any]:
    """Return the predicate closure represented by an interim row."""

    predicates: set[Any] = set()
    for name in ("predicate", "other_predicate"):
        value = coerce_value(getattr(row, name, None), cast_int=cast_int)
        if value not in EMPTY_VALUES:
            predicates.add(value)
    for name in ("subject_predicates", "object_predicates", "other_entity_predicates"):
        for value in coerce_sequence(getattr(row, name, None), cast_int=cast_int):
            if value not in EMPTY_VALUES:
                predicates.add(value)
    return predicates


def _other_entity_id(row: Any, *, cast_int: bool) -> Any:
    other_subject = coerce_value(getattr(row, "other_subject", None), cast_int=cast_int)
    if other_subject not in EMPTY_VALUES:
        return other_subject
    other_object = coerce_value(getattr(row, "other_object", None), cast_int=cast_int)
    if other_object not in EMPTY_VALUES:
        return other_object
    return 0


def _merge_role_facts(
    facts_by_entity: dict[Any, dict[Any, set[Any]]],
    predicates_present: dict[Any, set[Any]],
    *,
    entity_id: Any,
    predicates: Any,
    objects: Any,
    p_local: set[Any],
    cast_int: bool,
) -> None:
    """Merge one role projection into the canonical state for its entity."""

    if entity_id in EMPTY_VALUES:
        return
    entity_facts = facts_by_entity.setdefault(entity_id, {})
    entity_predicates = predicates_present.setdefault(entity_id, set())
    for predicate, obj in zip(
        coerce_sequence(predicates, cast_int=cast_int),
        coerce_sequence(objects, cast_int=cast_int),
    ):
        if predicate in EMPTY_VALUES or obj in EMPTY_VALUES or predicate not in p_local:
            continue
        entity_facts.setdefault(predicate, set()).add(obj)
        entity_predicates.add(predicate)


def build_facts_state(
    row: Any,
    *,
    p_local: set[Any] | None = None,
    assume_complete: bool,
    cast_int: bool,
) -> tuple[dict[Any, dict[Any, set[Any]]], dict[Any, set[Any]]]:
    """Build the corrected pre-edit fact maps for an interim row.

    Role aliases are merged, and the focus/base statement is inserted even when
    a historical sidecar omitted it.  The latter invariant is required because
    every benchmark row describes a violation attached to that statement.
    """

    del assume_complete  # Kept in the signature for both historical call sites.
    local_predicates = p_local if p_local is not None else compute_p_local(row, cast_int=cast_int)
    facts_by_entity: dict[Any, dict[Any, set[Any]]] = {}
    predicates_present: dict[Any, set[Any]] = {}

    subject = coerce_value(getattr(row, "subject", None), cast_int=cast_int)
    obj = coerce_value(getattr(row, "object", None), cast_int=cast_int)
    other_entity = _other_entity_id(row, cast_int=cast_int)

    for entity_id, predicate_column, object_column in (
        (subject, "subject_predicates", "subject_objects"),
        (obj, "object_predicates", "object_objects"),
        (other_entity, "other_entity_predicates", "other_entity_objects"),
    ):
        _merge_role_facts(
            facts_by_entity,
            predicates_present,
            entity_id=entity_id,
            predicates=getattr(row, predicate_column, None),
            objects=getattr(row, object_column, None),
            p_local=local_predicates,
            cast_int=cast_int,
        )

    focus_predicate = coerce_value(getattr(row, "predicate", None), cast_int=cast_int)
    focus_object = coerce_value(getattr(row, "object", None), cast_int=cast_int)
    if subject not in EMPTY_VALUES:
        facts_by_entity.setdefault(subject, {})
        predicates_present.setdefault(subject, set())
    if (
        subject not in EMPTY_VALUES
        and focus_predicate not in EMPTY_VALUES
        and focus_object not in EMPTY_VALUES
    ):
        local_predicates.add(focus_predicate)
        facts_by_entity[subject].setdefault(focus_predicate, set()).add(focus_object)
        predicates_present[subject].add(focus_predicate)

    return facts_by_entity, predicates_present


def build_pre_state(
    row: Any,
    *,
    p_local: set[Any] | None = None,
    assume_complete: bool,
    cast_int: bool,
) -> tuple[EvidenceState, set[Any]]:
    local_predicates = p_local if p_local is not None else compute_p_local(row, cast_int=cast_int)
    facts, present = build_facts_state(
        row,
        p_local=local_predicates,
        assume_complete=assume_complete,
        cast_int=cast_int,
    )
    state = EvidenceState(
        facts_by_entity=facts,
        predicates_present=present,
        assume_complete=assume_complete,
        missing_edits=set(),
        focus_subject=coerce_value(getattr(row, "subject", 0), cast_int=cast_int),
        focus_predicate=coerce_value(getattr(row, "predicate", 0), cast_int=cast_int),
        focus_object=coerce_value(getattr(row, "object", 0), cast_int=cast_int),
        other_subject=coerce_value(getattr(row, "other_subject", 0), cast_int=cast_int),
        other_predicate=coerce_value(getattr(row, "other_predicate", 0), cast_int=cast_int),
        other_object=coerce_value(getattr(row, "other_object", 0), cast_int=cast_int),
    )
    if not state.focus_statement_present():
        raise AssertionError("Corrected pre-state is missing its mandatory base statement")
    return state, local_predicates


def clone_fact_maps(
    facts_by_entity: Mapping[Any, Mapping[Any, set[Any]]],
    predicates_present: Mapping[Any, set[Any]],
) -> tuple[dict[Any, dict[Any, set[Any]]], dict[Any, set[Any]]]:
    return (
        {
            entity: {predicate: set(values) for predicate, values in facts.items()}
            for entity, facts in facts_by_entity.items()
        },
        {entity: set(predicates) for entity, predicates in predicates_present.items()},
    )


def resolve_triple(
    triple: Sequence[Any] | None,
    *,
    resolver: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any, Any] | None:
    if triple is None or len(triple) < 3:
        return None
    resolve = resolver or (lambda value: value)
    resolved = tuple(resolve(value) for value in triple[:3])
    if any(value in EMPTY_VALUES for value in resolved):
        return None
    return resolved  # type: ignore[return-value]


def apply_evidence_edits(
    pre_state: EvidenceState,
    *,
    p_local: set[Any],
    delete: Sequence[Any] | None,
    add: Sequence[Any] | None,
    resolver: Callable[[Any], Any] | None = None,
) -> tuple[EvidenceState, dict[str, tuple[Any, Any, Any] | None]]:
    """Apply a repair in canonical delete-then-add order.

    Returning the resolved operations makes both metrics and persisted
    prediction artifacts describe the exact symbolic edit that was applied.
    """

    facts, present = clone_fact_maps(pre_state.facts_by_entity, pre_state.predicates_present)
    missing_edits: set[tuple[Any, Any]] = set()
    resolved = {
        "del": resolve_triple(delete, resolver=resolver),
        "add": resolve_triple(add, resolver=resolver),
    }

    def _apply(kind: str, triple: tuple[Any, Any, Any] | None) -> None:
        if triple is None:
            return
        subject, predicate, obj = triple
        if subject not in facts or predicate not in p_local:
            missing_edits.add((subject, predicate))
            return
        if not pre_state.assume_complete and predicate not in pre_state.predicates_present.get(subject, set()):
            missing_edits.add((subject, predicate))
            return
        entity_facts = facts[subject]
        if kind == "del":
            entity_facts.get(predicate, set()).discard(obj)
        else:
            entity_facts.setdefault(predicate, set()).add(obj)
            present.setdefault(subject, set()).add(predicate)

    _apply("del", resolved["del"])
    _apply("add", resolved["add"])

    post_state = EvidenceState(
        facts_by_entity=facts,
        predicates_present=present,
        assume_complete=pre_state.assume_complete,
        missing_edits=missing_edits,
        focus_subject=pre_state.focus_subject,
        focus_predicate=pre_state.focus_predicate,
        focus_object=pre_state.focus_object,
        other_subject=pre_state.other_subject,
        other_predicate=pre_state.other_predicate,
        other_object=pre_state.other_object,
    )
    return post_state, resolved


def slots_to_operations(candidate_slots: Sequence[Any]) -> tuple[Sequence[Any], Sequence[Any]]:
    if len(candidate_slots) < 6:
        return (), ()
    return candidate_slots[3:6], candidate_slots[0:3]


def build_post_state_from_slots(
    pre_state: EvidenceState,
    *,
    p_local: set[Any],
    candidate_slots: Sequence[Any],
    resolver: Callable[[Any], Any] | None = None,
) -> tuple[EvidenceState, dict[str, tuple[Any, Any, Any] | None]]:
    delete, add = slots_to_operations(candidate_slots)
    return apply_evidence_edits(
        pre_state,
        p_local=p_local,
        delete=delete,
        add=add,
        resolver=resolver,
    )


def row_gold_operations(row: Any) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    return (
        (
            getattr(row, "del_subject", 0),
            getattr(row, "del_predicate", 0),
            getattr(row, "del_object", 0),
        ),
        (
            getattr(row, "add_subject", 0),
            getattr(row, "add_predicate", 0),
            getattr(row, "add_object", 0),
        ),
    )
