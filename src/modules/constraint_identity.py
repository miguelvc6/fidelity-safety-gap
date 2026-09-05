"""Shared constraint-vector selection and primary-identity invariants."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable, Mapping


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return _flatten(value.tolist())
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def select_constraint_ids(
    row: Any,
    *,
    constraint_scope: str,
    explicit_ids: Sequence[Any] | None = None,
    coerce: Callable[[Any], Any] = lambda value: int(value),
) -> list[Any]:
    """Select the exact definition vector used for both IDs and positions.

    Explicit IDs take precedence, followed by labeler-filtered factor IDs, then
    the configured raw local closure.  A positional index from one vector must
    never be applied to another vector.
    """

    raw: Any = explicit_ids
    if raw is None:
        raw = _field(row, "factor_constraint_ids")
    if raw is None and constraint_scope == "focus":
        raw = _field(row, "local_constraint_ids_focus")
    if raw is None:
        raw = _field(row, "local_constraint_ids")
    values = [coerce(value) for value in _flatten(raw) if value is not None]
    if not values:
        raise ValueError("Constraint vector is empty; cannot resolve the primary constraint")
    return values


def resolve_primary_index(
    row: Any,
    constraint_ids: Sequence[Any],
    *,
    supplied_index: int | None = None,
    coerce: Callable[[Any], Any] = lambda value: int(value),
) -> int:
    """Resolve and validate ``row.constraint_id`` inside ``constraint_ids``."""

    if not constraint_ids:
        raise ValueError("Constraint vector is empty; cannot resolve the primary constraint")
    try:
        primary_id = coerce(_field(row, "constraint_id"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Row has no valid constraint_id for primary resolution") from exc
    matches = [index for index, value in enumerate(constraint_ids) if value == primary_id]
    if not matches:
        raise ValueError(
            f"Primary constraint_id {primary_id!r} is missing from evaluated constraint vector {list(constraint_ids)!r}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Primary constraint_id {primary_id!r} occurs {len(matches)} times in evaluated constraint vector"
        )
    resolved = matches[0]
    if supplied_index is not None:
        try:
            supplied = int(supplied_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid supplied primary_factor_index {supplied_index!r}") from exc
        if supplied < 0 or supplied >= len(constraint_ids):
            raise ValueError(
                f"primary_factor_index {supplied} is outside evaluated vector of length {len(constraint_ids)}"
            )
        if supplied != resolved:
            raise ValueError(
                "primary_factor_index/constraint_id mismatch: "
                f"index {supplied} identifies {constraint_ids[supplied]!r}, "
                f"but row.constraint_id is {primary_id!r} at index {resolved}"
            )
    return resolved
