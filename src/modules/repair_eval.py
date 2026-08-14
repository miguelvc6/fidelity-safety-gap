"""Utilities that credit model predictions which resolve the violation even if they differ from gold."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from modules.data_encoders import GlobalIntEncoder
from modules.reranker_eval import CandidateConstraintEvaluator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriplePattern:
    """Match triples against optional per-component value sets."""

    subjects: frozenset[int] | None = None
    predicates: frozenset[int] | None = None
    objects: frozenset[int] | None = None

    def matches(self, triple: tuple[int, int, int]) -> bool:
        s, p, o = triple
        if self.subjects is not None and s not in self.subjects:
            return False
        if self.predicates is not None and p not in self.predicates:
            return False
        if self.objects is not None and o not in self.objects:
            return False
        return True


@dataclass
class CandidateRepairs:
    """Keeps the pattern lists produced by the heuristics for a single violation."""

    add: list[TriplePattern] = field(default_factory=list)
    delete: list[TriplePattern] = field(default_factory=list)


@dataclass
class ViolationContext:
    """Light-weight view over the original parquet row for a graph sample."""

    constraint_type: str
    constraint_id: int
    subject: int
    predicate: int
    object: int
    other_subject: int
    other_predicate: int
    other_object: int
    constraint_predicates: tuple[int, ...]
    constraint_objects: tuple[int, ...]
    row_index: int | None = None

    def values_for_predicate(self, predicate_id: int, none_class: int) -> list[int]:
        """Extract all object values associated with a given predicate in the constraint."""
        values: list[int] = []
        if not self.constraint_predicates:
            return values
        for pred, obj in zip(self.constraint_predicates, self.constraint_objects):
            if pred == predicate_id and obj != none_class and obj is not None:
                values.append(int(obj))
        return values


def _to_int(value: object, none_class: int) -> int:
    """Coerce a value to int, returning none_class on failure."""
    if value is None:
        return none_class
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return none_class
    if isinstance(value, (bytes, bytearray, memoryview)):
        return none_class
    if hasattr(value, "__int__"):
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return none_class
    return none_class


def _coerce_sequence(value: object) -> tuple[int, ...]:
    """Coerce a value to a tuple of ints, or empty tuple on failure."""
    if value is None:
        return ()
    items: object
    if isinstance(value, (tuple, list)):
        items = value
    elif isinstance(value, (str, bytes, bytearray, memoryview)):
        return ()
    elif hasattr(value, "tolist"):
        items = value.tolist()
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        return ()

    if not isinstance(items, (tuple, list)):
        items = [items]

    normalized: list[int] = []
    for item in items:
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(normalized)


def load_violation_contexts(base_path: Path, split: str, *, none_class: int = 0) -> list[ViolationContext]:
    """
    Load per-row metadata necessary for repair heuristics.

    Parameters
    ----------
    base_path : Path
        Base directory containing parquet split files.
    split : str
        Name of the split to load (e.g., "validation" or "test").
    none_class : int, optional
        ID used to represent missing values in the parquet file.

    Returns
    -------
    list[ViolationContext]
        List of violation contexts, one per row in the parquet file.
    """
    base_path = Path(base_path)
    path = base_path / f"df_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet split not found at {path}")

    columns = [
        "constraint_type",
        "constraint_id",
        "subject",
        "predicate",
        "object",
        "other_subject",
        "other_predicate",
        "other_object",
        "constraint_predicates",
        "constraint_objects",
    ]
    dataframe = pd.read_parquet(path, columns=columns)

    contexts: list[ViolationContext] = []
    for row_index, row in enumerate(dataframe.itertuples(index=False)):
        contexts.append(
            ViolationContext(
                constraint_type=str(getattr(row, "constraint_type")),
                constraint_id=int(getattr(row, "constraint_id")),
                subject=_to_int(getattr(row, "subject"), none_class),
                predicate=_to_int(getattr(row, "predicate"), none_class),
                object=_to_int(getattr(row, "object"), none_class),
                other_subject=_to_int(getattr(row, "other_subject"), none_class),
                other_predicate=_to_int(getattr(row, "other_predicate"), none_class),
                other_object=_to_int(getattr(row, "other_object"), none_class),
                constraint_predicates=_coerce_sequence(getattr(row, "constraint_predicates")),
                constraint_objects=_coerce_sequence(getattr(row, "constraint_objects")),
                row_index=row_index,
            )
        )

    del dataframe
    return contexts


def load_global_eval_rows(base_path: Path, split: str) -> list:
    """Load parquet rows with enough metadata for global repair metrics."""
    base_path = Path(base_path)
    path = base_path / f"df_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet split not found at {path}")

    columns = [
        "constraint_id",
        "constraint_type",
        "subject",
        "predicate",
        "object",
        "other_subject",
        "other_predicate",
        "other_object",
        "constraint_predicates",
        "constraint_objects",
        "subject_predicates",
        "subject_objects",
        "object_predicates",
        "object_objects",
        "other_entity_predicates",
        "other_entity_objects",
        "local_constraint_ids",
        "local_constraint_ids_focus",
        "factor_checkable_pre",
        "factor_satisfied_pre",
        "primary_factor_index",
        "factor_constraint_ids",
        "factor_types",
    ]
    dataframe = pd.read_parquet(path)
    existing = [col for col in columns if col in dataframe.columns]
    if existing:
        dataframe = dataframe[existing]
    rows = list(dataframe.itertuples(index=False))
    del dataframe
    return rows


@dataclass
class OutcomeCounts:
    """Counts of repair outcomes for a given action."""

    total: int = 0
    exact: int = 0
    alternative: int = 0
    missing: int = 0
    failed: int = 0

    def update(self, outcome: str) -> None:
        self.total += 1
        if outcome == RepairOutcome.EXACT:
            self.exact += 1
        elif outcome == RepairOutcome.ALTERNATIVE:
            self.alternative += 1
        elif outcome == RepairOutcome.MISSING:
            self.missing += 1
        else:
            self.failed += 1

    def as_dict(self) -> dict[str, float | int]:
        def _safe_div(num: int, denom: int) -> float:
            return float(num) / denom if denom else 0.0

        return {
            "total": self.total,
            "exact": self.exact,
            "alternative": self.alternative,
            "missing": self.missing,
            "failed": self.failed,
            "exact_rate": _safe_div(self.exact, self.total),
            "alternative_rate": _safe_div(self.alternative, self.total),
            "fix_rate": _safe_div(self.exact + self.alternative, self.total),
        }


class RepairMetricsAggregator:
    """Aggregate repair outcomes globally and per constraint type."""

    def __init__(self, actions: Sequence[str]) -> None:
        self.actions = tuple(actions)
        self.overall = {action: OutcomeCounts() for action in self.actions}
        self.per_constraint: dict[str, dict[str, OutcomeCounts]] = {}

    def add(self, constraint_type: str, action: str, outcome: str) -> None:
        if action not in self.overall:
            return
        self.overall[action].update(outcome)
        bucket = self.per_constraint.setdefault(
            constraint_type,
            {act: OutcomeCounts() for act in self.actions},
        )
        bucket[action].update(outcome)

    def as_dict(self) -> dict[str, object]:
        summary_counts = OutcomeCounts()
        for counts in self.overall.values():
            summary_counts.total += counts.total
            summary_counts.exact += counts.exact
            summary_counts.alternative += counts.alternative
            summary_counts.missing += counts.missing
            summary_counts.failed += counts.failed

        return {
            "overall": summary_counts.as_dict(),
            "by_action": {action: counts.as_dict() for action, counts in self.overall.items()},
            "per_constraint_type": {
                constraint: {action: counts.as_dict() for action, counts in action_counts.items()}
                for constraint, action_counts in sorted(self.per_constraint.items())
            },
        }


class ConstraintRepairHeuristics:
    """Derive candidate repair triples for each constraint type."""

    def __init__(
        self,
        *,
        encoder: GlobalIntEncoder,
        placeholder_ids: dict[str, int],
        none_class: int,
    ) -> None:
        self.encoder = encoder
        self.placeholder_ids = placeholder_ids
        self.none_class = none_class

        # Frequently used parameter/predicate ids from constraint templates.
        self._instance_of = self._maybe_id("<http://www.wikidata.org/entity/P31>")
        self._subclass_of = self._maybe_id("<http://www.wikidata.org/entity/P279>")
        self._class_param = self._maybe_id("<http://www.wikidata.org/entity/P2308>")
        self._relation_param = self._maybe_id("<http://www.wikidata.org/entity/P2309>")
        self._items_param = self._maybe_id("<http://www.wikidata.org/entity/P2305>")
        self._property_param = self._maybe_id("<http://www.wikidata.org/entity/P2306>")
        self._inverse_param = self._maybe_id("<http://www.wikidata.org/entity/P1696>")

    def _maybe_id(self, token: str) -> int | None:
        """Return the encoder id for *token*, or None if missing from the vocabulary."""
        idx = self.encoder.encode(token, add_new=False)
        return idx if idx else None

    def _has_value(self, value: int | None) -> bool:
        return value is not None and value != self.none_class

    def _component_values(self, placeholder: str, actual: int) -> frozenset[int] | None:
        """Return a set containing the placeholder id, the actual id, or both."""
        values: set[int] = set()
        placeholder_id = self.placeholder_ids.get(placeholder)
        if placeholder_id is not None:
            values.add(placeholder_id)
        if self._has_value(actual):
            values.add(int(actual))
        return frozenset(values) if values else None

    def _placeholder_only(self, placeholder: str) -> frozenset[int] | None:
        pid = self.placeholder_ids.get(placeholder)
        return frozenset({pid}) if pid is not None else None

    def _focus_predicate_values(self, ctx: ViolationContext) -> frozenset[int] | None:
        """Allow predicates encoded as either the placeholder or the focus value."""
        values: set[int] = set()
        placeholder_val = self.placeholder_ids.get("predicate")
        if placeholder_val is not None:
            values.add(placeholder_val)
        if self._has_value(ctx.predicate):
            values.add(ctx.predicate)
        return frozenset(values) if values else None

    def _value_sets(self, values: Iterable[int]) -> list[frozenset[int]]:
        """Wrap scalar ids in frozensets so they can populate TriplePattern components."""
        out: list[frozenset[int]] = []
        for value in values:
            if value == self.none_class or value is None:
                continue
            out.append(frozenset({int(value)}))
        return out

    def candidates_for(self, context: ViolationContext) -> CandidateRepairs:
        """Return heuristic additions/deletions that should count as valid repairs."""
        candidates = CandidateRepairs()
        candidates.delete.extend(self._base_deletions(context))
        candidates.add.extend(self._base_additions(context))

        ctype = context.constraint_type
        # Each constraint family contributes custom patterns to the relevant action(s).
        if ctype == "conflictWith":
            candidates.delete.extend(self._conflict_with_patterns(context))
        elif ctype == "distinct":
            candidates.delete.extend(self._distinct_patterns(context))
        elif ctype == "single":
            candidates.delete.extend(self._single_patterns(context))
        elif ctype == "oneOf":
            candidates.delete.extend(self._one_of_patterns(context))
            candidates.add.extend(self._one_of_additions(context))
        elif ctype == "type":
            candidates.add.extend(self._type_additions(context))
        elif ctype == "valueType":
            candidates.add.extend(self._value_type_additions(context))
        elif ctype == "itemRequiresStatement":
            candidates.add.extend(self._item_requires_additions(context))
        elif ctype == "valueRequiresStatement":
            candidates.add.extend(self._value_requires_additions(context))
        elif ctype == "inverse":
            candidates.add.extend(self._inverse_additions(context))

        return candidates

    # Base heuristics for every constraint type
    def _base_deletions(self, ctx: ViolationContext) -> list[TriplePattern]:
        """
        All types - deleting the focus triple; deleting the `other_*` triple when the violation
        references a conflicting statement
        """
        patterns: list[TriplePattern] = []
        focus = TriplePattern(
            subjects=self._component_values("subject", ctx.subject),
            predicates=self._component_values("predicate", ctx.predicate),
            objects=self._component_values("object", ctx.object),
        )
        patterns.append(focus)

        if self._has_value(ctx.other_predicate):
            other = TriplePattern(
                subjects=self._component_values("other_subject", ctx.other_subject),
                predicates=self._component_values("other_predicate", ctx.other_predicate),
                objects=self._component_values("other_object", ctx.other_object),
            )
            patterns.append(other)
        return [p for p in patterns if any((p.subjects, p.predicates, p.objects))]

    def _base_additions(self, ctx: ViolationContext) -> list[TriplePattern]:
        # Default additions are empty unless overridden per constraint.
        return []

    # Constraint-specific heuristics
    def _distinct_patterns(self, ctx: ViolationContext) -> list[TriplePattern]:
        """deleting the duplicate `(other_subject, predicate, object)` triple belonging to the competing item."""
        patterns: list[TriplePattern] = []
        if (
            self._has_value(ctx.other_subject)
            and self._has_value(ctx.other_object)
            and ctx.other_predicate == ctx.predicate
            and ctx.other_object == ctx.object
            and ctx.other_subject != ctx.subject
        ):
            patterns.append(
                TriplePattern(
                    subjects=self._component_values("other_subject", ctx.other_subject),
                    predicates=self._component_values("predicate", ctx.predicate),
                    objects=self._component_values("object", ctx.object),
                )
            )
        return patterns

    # oneOf-specific additions
    def _one_of_additions(self, ctx: ViolationContext) -> list[TriplePattern]:
        """
        adding `(subject, predicate, value)` where `value` is listed in the constraint's
        `items (P2305)` parameter (either via the literal predicate ID or the `predicate` placeholder).
        """
        if not self._items_param:
            return []
        allowed_items = ctx.values_for_predicate(self._items_param, self.none_class)
        if not allowed_items:
            return []
        predicate_values = self._focus_predicate_values(ctx)
        subject_values = self._component_values("subject", ctx.subject)
        return [
            TriplePattern(
                subjects=subject_values,
                predicates=predicate_values,
                objects=frozenset({item}),
            )
            for item in allowed_items
        ]

    # These three constraints share similar deletion heuristics
    def _one_of_patterns(self, ctx: ViolationContext) -> list[TriplePattern]:
        return self._consistency_overlaps(ctx)

    def _single_patterns(self, ctx: ViolationContext) -> list[TriplePattern]:
        return self._consistency_overlaps(ctx)

    def _conflict_with_patterns(self, ctx: ViolationContext) -> list[TriplePattern]:
        return self._consistency_overlaps(ctx)

    def _consistency_overlaps(self, ctx: ViolationContext) -> list[TriplePattern]:
        """
        Heuristics shared by conflictWith/oneOf/single style constraints.

        deletions of mixed triples such as `(subject, predicate, other_object)`, `(subject, other_predicate, object)`, or
        `(other_subject, predicate, object)` depending on whether the clash is per-value or per-property.
        """
        patterns: list[TriplePattern] = []
        same_subject = self._has_value(ctx.other_subject) and ctx.other_subject == ctx.subject
        same_predicate = self._has_value(ctx.other_predicate) and ctx.other_predicate == ctx.predicate

        other_object_values = self._component_values("other_object", ctx.other_object)
        if same_subject and same_predicate and other_object_values is not None:
            patterns.append(
                TriplePattern(
                    subjects=self._component_values("subject", ctx.subject),
                    predicates=self._component_values("predicate", ctx.predicate),
                    objects=other_object_values,
                )
            )

        if same_subject and self._has_value(ctx.other_predicate):
            patterns.append(
                TriplePattern(
                    subjects=self._component_values("subject", ctx.subject),
                    predicates=self._component_values("other_predicate", ctx.other_predicate),
                    objects=self._component_values("object", ctx.object),
                )
            )
            if other_object_values is not None:
                patterns.append(
                    TriplePattern(
                        subjects=self._component_values("subject", ctx.subject),
                        predicates=self._component_values("other_predicate", ctx.other_predicate),
                        objects=other_object_values,
                    )
                )

        if (
            self._has_value(ctx.other_subject)
            and same_predicate
            and self._has_value(ctx.other_object)
            and ctx.other_object == ctx.object
            and ctx.other_subject != ctx.subject
        ):
            patterns.append(
                TriplePattern(
                    subjects=self._component_values("other_subject", ctx.other_subject),
                    predicates=self._component_values("predicate", ctx.predicate),
                    objects=self._component_values("object", ctx.object),
                )
            )

        return [p for p in patterns if any((p.subjects, p.predicates, p.objects))]

    # type/valueType-specific additions
    def _type_additions(self, ctx: ViolationContext) -> list[TriplePattern]:
        """adding `(subject, relation, class)` where `class` comes from `class (P2308)` and
        `relation` is either the constraint's `relation (P2309)` parameter or the defaults
        `instance of (P31)` / `subclass of (P279)`."""
        return self._class_based_additions(ctx, subject_placeholder="subject")

    def _value_type_additions(self, ctx: ViolationContext) -> list[TriplePattern]:
        """same as `type` but anchored on the object placeholder, i.e. adding statements
        about the value entity."""
        return self._class_based_additions(ctx, subject_placeholder="object")

    def _class_based_additions(self, ctx: ViolationContext, *, subject_placeholder: str) -> list[TriplePattern]:
        if not self._class_param:
            return []
        allowed_classes = ctx.values_for_predicate(self._class_param, self.none_class)
        if not allowed_classes:
            return []

        relation_candidates = ctx.values_for_predicate(self._relation_param or -1, self.none_class)
        if not relation_candidates:
            relation_candidates = [pid for pid in (self._instance_of, self._subclass_of) if pid]

        subject_values = self._component_values(
            subject_placeholder, getattr(ctx, subject_placeholder, self.none_class)
        )
        patterns: list[TriplePattern] = []
        for relation in relation_candidates:
            if not self._has_value(relation):
                continue
            predicate_values = frozenset({int(relation)})
            for cls in allowed_classes:
                patterns.append(
                    TriplePattern(
                        subjects=subject_values,
                        predicates=predicate_values,
                        objects=frozenset({int(cls)}),
                    )
                )
        return patterns

    # itemRequiresStatement-specific additions
    def _item_requires_additions(self, ctx: ViolationContext) -> list[TriplePattern]:
        """adding `(subject, property, ANY)` whenever `property` appears in the
        constraint's `property (P2306)` list; any value satisfies the requirement."""
        if not self._property_param:
            return []
        required_props = ctx.values_for_predicate(self._property_param, self.none_class)
        if not required_props:
            return []
        subject_values = self._component_values("subject", ctx.subject)
        patterns: list[TriplePattern] = []
        for prop in required_props:
            if not self._has_value(prop):
                continue
            predicate_values = frozenset({int(prop)})
            patterns.append(
                TriplePattern(
                    subjects=subject_values,
                    predicates=predicate_values,
                    objects=None,  # any value for the required statement is acceptable
                )
            )
        return patterns

    def _value_requires_additions(self, ctx: ViolationContext) -> list[TriplePattern]:
        """adding `(object, property, ANY)` for every required property in
        `P2306`, thereby satisfying obligations on the target item."""
        if not self._property_param:
            return []
        required_props = ctx.values_for_predicate(self._property_param, self.none_class)
        if not required_props:
            return []
        subject_values = self._component_values("object", ctx.object)
        patterns: list[TriplePattern] = []
        for prop in required_props:
            if not self._has_value(prop):
                continue
            patterns.append(
                TriplePattern(
                    subjects=subject_values,
                    predicates=frozenset({int(prop)}),
                    objects=None,
                )
            )
        return patterns

    def _inverse_additions(self, ctx: ViolationContext) -> list[TriplePattern]:
        """
        adding `(object, inverse_predicate, subject)` where `inverse_predicate` is taken
        from `inverse property (P1696)` or, if unspecified, defaults to the original predicate (covering symmetric
        properties).
        """
        predicate_ids = ctx.values_for_predicate(self._inverse_param or -1, self.none_class)
        if not predicate_ids:
            predicate_ids = [ctx.predicate]

        subject_values = self._component_values("object", ctx.object)
        object_values = self._component_values("subject", ctx.subject)
        patterns: list[TriplePattern] = []
        for predicate_id in predicate_ids:
            if not self._has_value(predicate_id):
                continue
            values = {int(predicate_id)}
            if predicate_id == ctx.predicate:
                placeholder = self.placeholder_ids.get("predicate")
                if placeholder is not None:
                    values.add(placeholder)
            patterns.append(
                TriplePattern(
                    subjects=subject_values,
                    predicates=frozenset(values),
                    objects=object_values,
                )
            )
        return patterns


@dataclass
class RepairSample:
    """Predicted/gold triples for a single graph, grouped by action."""

    constraint_type: str
    predicted: dict[str, tuple[int, int, int] | None]
    gold: dict[str, tuple[int, int, int] | None]


def evaluate_repair_samples(
    *,
    samples: Sequence[RepairSample],
    contexts: Sequence[ViolationContext],
    heuristics: ConstraintRepairHeuristics,
    actions: Sequence[str],
) -> dict[str, object]:
    if len(samples) != len(contexts):
        raise ValueError(
            f"Mismatch between prediction samples ({len(samples)}) and contexts ({len(contexts)}). "
            "Ensure the parquet rows and graph samples are aligned."
        )

    aggregator = RepairMetricsAggregator(actions)
    for sample, context in zip(samples, contexts):
        candidate_map = heuristics.candidates_for(context)
        for action in actions:
            triple = sample.predicted.get(action)
            gold = sample.gold.get(action)
            patterns = getattr(candidate_map, "add" if action == "add" else "delete")
            outcome = _classify_triple(triple, gold, patterns)
            aggregator.add(sample.constraint_type, action, outcome)
    return aggregator.as_dict()


PAPER_METRIC_KEYS: tuple[str, ...] = (
    "pfr",
    "local_satisfaction",
    "delta_local_satisfaction",
    "sir",
    "srr",
    "disruption",
    "base_deletion_rate",
    "deletes_base_action_rate",
    "eppf",
    "vacuous_improvement",
)


@dataclass
class PaperMetricsAccumulator:
    numerators: dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in PAPER_METRIC_KEYS}
    )
    denominators: dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in PAPER_METRIC_KEYS}
    )

    def update(self, events: dict[str, dict[str, int]]) -> None:
        for key in PAPER_METRIC_KEYS:
            event = events[key]
            self.numerators[key] += int(event["numerator"])
            self.denominators[key] += int(event["denominator"])

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        metrics: dict[str, dict[str, float | int]] = {}
        for key in PAPER_METRIC_KEYS:
            numerator = self.numerators[key]
            denominator = self.denominators[key]
            metrics[key] = {
                "value": float(numerator) / denominator if denominator else 0.0,
                "numerator": numerator,
                "denominator": denominator,
            }
        return metrics


def _coerce_factor_sequence(value: object) -> list[int] | list[bool] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())  # type: ignore[call-arg]
        except Exception:
            pass
    return [value]


def _candidate_slots_from_sample(sample: RepairSample, none_class: int) -> tuple[int, int, int, int, int, int]:
    add = sample.predicted.get("add")
    delete = sample.predicted.get("del")
    if add is None:
        add_slot = (none_class, none_class, none_class)
    else:
        add_slot = add
    if delete is None:
        del_slot = (none_class, none_class, none_class)
    else:
        del_slot = delete
    return (
        int(add_slot[0]),
        int(add_slot[1]),
        int(add_slot[2]),
        int(del_slot[0]),
        int(del_slot[1]),
        int(del_slot[2]),
    )


def _resolve_primary_index_from_row(row: object, local_constraint_ids: Sequence[int]) -> int:
    value = getattr(row, "primary_factor_index", None)
    if value is not None:
        try:
            idx = int(value)
            if 0 <= idx < len(local_constraint_ids):
                return idx
        except (TypeError, ValueError):
            pass
    constraint_id = getattr(row, "constraint_id", None)
    try:
        constraint_id = int(constraint_id)
    except (TypeError, ValueError):
        return -1
    try:
        return list(local_constraint_ids).index(constraint_id)
    except ValueError:
        return -1


def evaluate_paper_metric_instance(
    *,
    row: object,
    evaluator: CandidateConstraintEvaluator,
    candidate_slots: Sequence[int],
    constraint_type: str,
    row_index: int,
    none_class: int,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    """Reconstruct one edit and return its corrected paper-metric events.

    This is the single source of truth used by model evaluation and sidecar
    diagnostics.  In particular, it recomputes both symbolic states from the
    interim row and never consumes stored graph factor-label tensors.
    """

    slots = tuple(int(value) for value in candidate_slots)
    if len(slots) != 6:
        raise ValueError(f"Candidate slots must contain six values, got {len(slots)}.")
    if any(value < 0 for value in slots):
        raise ValueError("Candidate slots contain negative class indices.")

    if details is None:
        details = evaluator.evaluate_full(
            row,
            candidate_slots=slots,
            primary_factor_index=None,
            factor_constraint_ids=None,
        )

    local_constraint_ids = details["local_constraint_ids"]
    primary_index = details["primary_factor_index"]
    pre_checkable = details["pre_checkable"]
    pre_satisfied = details["pre_satisfied"]
    post_checkable = details["post_checkable"]
    post_satisfied = details["post_satisfied"]

    if primary_index < 0:
        primary_index = _resolve_primary_index_from_row(row, local_constraint_ids)

    if not (
        len(pre_checkable)
        == len(pre_satisfied)
        == len(post_checkable)
        == len(post_satisfied)
    ):
        raise AssertionError("Pre/post factor arrays differ in length during paper evaluation.")

    local_num = sum(
        int(post_satisfied[i])
        for i, checkable in enumerate(post_checkable)
        if checkable
    )
    local_denom = sum(int(bool(checkable)) for checkable in post_checkable)

    delta_num = 0
    delta_denom = 0
    srr_num = srr_denom = 0
    sir_num = sir_denom = 0
    for i in range(len(local_constraint_ids)):
        common_support = bool(pre_checkable[i]) and bool(post_checkable[i])
        if common_support:
            delta_num += int(post_satisfied[i]) - int(pre_satisfied[i])
            delta_denom += 1
        if i == primary_index or not common_support:
            continue
        if pre_satisfied[i]:
            srr_denom += 1
            if not post_satisfied[i]:
                srr_num += 1
        else:
            sir_denom += 1
            if post_satisfied[i]:
                sir_num += 1

    # A slot group is an operation only when all three components resolve.
    # This matches `_triples_from_indices` and the resolved operations written
    # to schema-v2 prediction artifacts. Partially populated slot groups do not
    # mutate the symbolic state and therefore do not count as disruption.
    add_count = int(all(value != none_class for value in slots[:3]))
    del_count = int(all(value != none_class for value in slots[3:]))
    pre_base_present = int(details.get("pre_focus_present", 0))
    post_base_present = int(details.get("post_focus_present", 0))
    base_deleted = int(bool(pre_base_present) and not bool(post_base_present))
    deletes_base_action = int(details.get("candidate_deletes_focus", 0))

    pfr_eligible = int(
        0 <= primary_index < len(pre_checkable)
        and bool(pre_checkable[primary_index])
        and not bool(pre_satisfied[primary_index])
    )
    pfr_event = int(
        pfr_eligible
        and bool(post_checkable[primary_index])
        and bool(post_satisfied[primary_index])
    )
    eppf_event = int(pfr_event and bool(post_base_present))
    vacuous_event = int(base_deleted and delta_denom > 0 and delta_num > 0)

    events = {
        "pfr": {"numerator": pfr_event, "denominator": pfr_eligible},
        "local_satisfaction": {"numerator": local_num, "denominator": local_denom},
        "delta_local_satisfaction": {"numerator": delta_num, "denominator": delta_denom},
        "sir": {"numerator": sir_num, "denominator": sir_denom},
        "srr": {"numerator": srr_num, "denominator": srr_denom},
        "disruption": {"numerator": add_count + del_count, "denominator": 1},
        "base_deletion_rate": {"numerator": base_deleted, "denominator": pre_base_present},
        "deletes_base_action_rate": {"numerator": deletes_base_action, "denominator": 1},
        "eppf": {"numerator": eppf_event, "denominator": pfr_eligible},
        "vacuous_improvement": {"numerator": vacuous_event, "denominator": 1},
    }
    return {
        "row_index": row_index,
        "constraint_type": constraint_type,
        "events": events,
        "pre_checkable": [bool(value) for value in pre_checkable],
        "pre_satisfied": [int(value) for value in pre_satisfied],
        "post_checkable": [bool(value) for value in post_checkable],
        "post_satisfied": [int(value) for value in post_satisfied],
        "primary_factor_index": int(primary_index),
        "pre_base_present": pre_base_present,
        "post_base_present": post_base_present,
        "resolved_add": details.get("resolved_add"),
        "resolved_del": details.get("resolved_del"),
    }


def evaluate_global_repair_samples(
    *,
    samples: Sequence[RepairSample],
    rows: Sequence[object],
    evaluator: CandidateConstraintEvaluator,
    none_class: int,
    pre_vectors: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    if len(samples) != len(rows):
        raise ValueError(
            f"Mismatch between prediction samples ({len(samples)}) and parquet rows ({len(rows)})."
        )

    if pre_vectors is not None:
        raise ValueError(
            "Stored graph pre-label tensors are not valid paper-metric truth; "
            "recompute both states from interim rows."
        )

    overall = PaperMetricsAccumulator()
    per_constraint: dict[str, PaperMetricsAccumulator] = {}
    per_instance: list[dict[str, object]] = []

    for idx, (sample, row) in enumerate(zip(samples, rows)):
        instance = evaluate_paper_metric_instance(
            row=row,
            evaluator=evaluator,
            candidate_slots=_candidate_slots_from_sample(sample, none_class),
            constraint_type=sample.constraint_type,
            row_index=idx,
            none_class=none_class,
        )
        events = instance["events"]
        overall.update(events)
        per_constraint.setdefault(sample.constraint_type, PaperMetricsAccumulator()).update(events)
        per_instance.append(instance)

    return {
        "paper_metrics": overall.as_dict(),
        "per_constraint_type": {
            key: accumulator.as_dict()
            for key, accumulator in sorted(per_constraint.items())
        },
        "per_instance": per_instance,
    }


class RepairOutcome:
    """Labels returned by `_classify_triple`."""

    EXACT = "exact"
    ALTERNATIVE = "alternative_fix"
    NO_FIX = "no_fix"
    MISSING = "missing_prediction"


def _classify_triple(
    triple: tuple[int, int, int] | None,
    gold: tuple[int, int, int] | None,
    patterns: Sequence[TriplePattern],
) -> str:
    if triple is None:
        return RepairOutcome.EXACT if gold is None else RepairOutcome.MISSING
    if gold is not None and triple == gold:
        return RepairOutcome.EXACT
    for pattern in patterns:
        if pattern.matches(triple):
            return RepairOutcome.ALTERNATIVE
    return RepairOutcome.NO_FIX
