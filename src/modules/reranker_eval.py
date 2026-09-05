"""Shared candidate evaluation over validator-semantics-v3 evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd
import numpy as np

from modules.constraint_checkers import (
    VALIDATOR_SEMANTICS_VERSION,
    ConstraintInstance,
    EvidenceState,
    evaluate_constraint_detailed,
    normalize_token,
    parse_constraint_instance,
)
from modules.class_hierarchy import ClassHierarchy, HIERARCHY_CUTOFF
from modules.constraint_identity import resolve_primary_index, select_constraint_ids
from modules.data_encoders import GlobalIntEncoder
from modules.evidence_state import (
    apply_evidence_edits as _shared_apply_evidence_edits,
    build_facts_state as _shared_build_facts_state,
    compute_p_local as _shared_compute_p_local,
)

PLACEHOLDER_LABELS: tuple[str, ...] = (
    "subject",
    "predicate",
    "object",
    "other_subject",
    "other_predicate",
    "other_object",
)

@dataclass(frozen=True)
class RegistryEntry:
    constraint_type_raw: str
    constraint_type_item: str
    constraint_type_index: int
    constraint_family: str
    constraint_label: str
    constraint_family_supported: bool
    constrained_property_raw: str
    param_predicates_raw: Tuple[str, ...]
    param_objects_raw: Tuple[str, ...]


@dataclass
class CandidateMetrics:
    primary_satisfied: int
    primary_checkable: int
    primary_pre_violated: int
    global_satisfied_fraction: float
    secondary_regressions: int
    secondary_regressions_denom: int
    secondary_improvements: int
    secondary_improvements_denom: int
    srr: float
    sir: float
    add_count: int
    del_count: int
    focus_preserved: int = 0
    focus_deleted: int = 0
    candidate_deletes_focus: int = 0


def _load_registry(path: str | None) -> Dict[str, RegistryEntry]:
    if path is None:
        return {}
    registry_df = pd.read_parquet(path)
    registry_json = registry_df["registry_json"].iloc[0]
    if isinstance(registry_json, str):
        registry = json.loads(registry_json)
    else:
        registry = registry_json
    type_items = sorted(
        {
            str(entry.get("constraint_type_item", "")).strip()
            for entry in registry.values()
            if str(entry.get("constraint_type_item", "")).strip()
        }
    )
    fallback_type_index = {type_item: idx for idx, type_item in enumerate(type_items)}
    parsed: Dict[str, RegistryEntry] = {}
    for constraint_id, entry in registry.items():
        constraint_family = entry.get("constraint_family")
        if not constraint_family:
            constraint_family = entry.get("constraint_type_name", "")
        constraint_family_supported = entry.get("constraint_family_supported")
        if constraint_family_supported is None:
            constraint_family_supported = entry.get("constraint_type_supported", False)
        constraint_type_item = str(entry.get("constraint_type_item", ""))
        constraint_type_index = entry.get("constraint_type_index")
        if constraint_type_index is None:
            constraint_type_index = fallback_type_index.get(constraint_type_item.strip(), -1)
        parsed[constraint_id] = RegistryEntry(
            constraint_type_raw=str(entry.get("constraint_type", "")),
            constraint_type_item=constraint_type_item,
            constraint_type_index=int(constraint_type_index),
            constraint_family=str(constraint_family or ""),
            constraint_label=str(entry.get("constraint_label", "")),
            constraint_family_supported=bool(constraint_family_supported),
            constrained_property_raw=str(entry.get("constrained_property", "")),
            param_predicates_raw=tuple(entry.get("param_predicates") or ()),
            param_objects_raw=tuple(entry.get("param_objects") or ()),
        )
    return parsed


def _resolve_registry_id(raw_id: str | None, encoder: GlobalIntEncoder | None) -> int:
    if encoder is None or not raw_id:
        return 0
    raw = raw_id.strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1].strip()
    if raw.startswith("http://www.wikidata.org/prop/direct/"):
        raw = raw.replace("http://www.wikidata.org/prop/direct/", "http://www.wikidata.org/entity/")
    candidates: List[str] = []
    seen: Set[str] = set()
    if raw.startswith("http://") or raw.startswith("https://"):
        candidates.extend([raw, f"<{raw}>"])
        tail = raw.rsplit("/", 1)[-1]
        if tail and tail[0] in ("P", "Q") and tail[1:].isdigit():
            candidates.append(tail)
    else:
        if raw and raw[0] in ("P", "Q") and raw[1:].isdigit():
            entity_uri = f"http://www.wikidata.org/entity/{raw}"
            candidates.extend([entity_uri, f"<{entity_uri}>"])
        candidates.append(raw)
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        token_id = encoder.encode(candidate, add_new=False)
        if token_id:
            return token_id
    return 0


def _resolve_registry_mapping(
    registry: Dict[str, RegistryEntry],
    *,
    encoder: GlobalIntEncoder | None,
    use_encoded_ids: bool,
) -> Dict[int, RegistryEntry] | Dict[str, RegistryEntry]:
    if use_encoded_ids:
        if encoder is None:
            raise ValueError("Encoder required to map registry constraint ids to dataset ids.")
        mapped: Dict[int, RegistryEntry] = {}
        for constraint_id, entry in registry.items():
            cid = _resolve_registry_id(constraint_id, encoder)
            if cid == 0:
                continue
            mapped[cid] = entry
        return mapped
    mapped_str: Dict[str, RegistryEntry] = {}
    for constraint_id, entry in registry.items():
        key = normalize_token(constraint_id) or constraint_id
        mapped_str[key] = entry
    return mapped_str


def _lookup_registry_entry(
    constraint_id: Any,
    registry_by_id: Dict[int, RegistryEntry] | Dict[str, RegistryEntry],
    *,
    use_encoded_ids: bool,
) -> RegistryEntry | None:
    if use_encoded_ids:
        try:
            cid = int(constraint_id)
        except (TypeError, ValueError):
            return None
        return registry_by_id.get(cid)  # type: ignore[arg-type]
    key = normalize_token(str(constraint_id)) or str(constraint_id)
    return registry_by_id.get(key)  # type: ignore[arg-type]


def _coerce_value(value: Any, *, cast_int: bool = True) -> Any:
    if not cast_int:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _resolve_placeholder_token_ids(encoder: GlobalIntEncoder | None) -> Dict[str, int]:
    if encoder is None:
        return {}
    token_ids: Dict[str, int] = {}
    for label in PLACEHOLDER_LABELS:
        token_id = encoder.encode(label, add_new=False)
        if token_id:
            token_ids[label] = int(token_id)
    return token_ids


def _compute_p_local(row: Any, *, cast_int: bool = True) -> Set[Any]:
    return _shared_compute_p_local(row, cast_int=cast_int)


def _build_facts_state(
    row: Any,
    *,
    p_local: Set[Any],
    assume_complete: bool,
    cast_int: bool,
) -> Tuple[Dict[int, Dict[int, Set[int]]], Dict[int, Set[int]]]:
    facts, present = _shared_build_facts_state(
        row,
        p_local=p_local,
        assume_complete=assume_complete,
        cast_int=cast_int,
    )
    return facts, present


def _build_placeholder_map(
    encoder: GlobalIntEncoder | None,
    row: Any,
    placeholder_token_ids: Dict[str, int] | None = None,
) -> Dict[Any, Any]:
    mapping: Dict[Any, Any] = {}
    if encoder is None:
        return {
            "subject": getattr(row, "subject", 0),
            "predicate": getattr(row, "predicate", 0),
            "object": getattr(row, "object", 0),
            "other_subject": getattr(row, "other_subject", 0),
            "other_predicate": getattr(row, "other_predicate", 0),
            "other_object": getattr(row, "other_object", 0),
        }

    token_ids = placeholder_token_ids if placeholder_token_ids is not None else _resolve_placeholder_token_ids(encoder)
    for label, token_id in token_ids.items():
        mapping[token_id] = int(getattr(row, label, 0) or 0)
    return mapping


def _resolve_placeholder(value: Any, placeholder_map: Dict[Any, Any]) -> Any:
    if value in placeholder_map:
        return placeholder_map[value]
    return value


def _build_post_state_for_candidate(
    base_facts_by_entity: Dict[Any, Dict[Any, Set[Any]]],
    base_predicates_present: Dict[Any, Set[Any]],
    p_local: Set[Any],
    *,
    candidate_slots: Sequence[int],
    placeholder_map: Dict[Any, Any],
    assume_complete: bool,
) -> Tuple[
    Dict[Any, Dict[Any, Set[Any]]],
    Dict[Any, Set[Any]],
    Set[Tuple[Any, Any]],
    tuple,
    frozenset,
    frozenset,
]:
    pre_state = EvidenceState(
        facts_by_entity=base_facts_by_entity,
        predicates_present=base_predicates_present,
        assume_complete=assume_complete,
        missing_edits=set(),
        focus_subject=0,
        focus_predicate=0,
        focus_object=0,
        other_subject=0,
        other_predicate=0,
        other_object=0,
    )
    post_state, _ = _shared_apply_evidence_edits(
        pre_state,
        p_local=p_local,
        delete=candidate_slots[3:6],
        add=candidate_slots[0:3],
        resolver=lambda value: _resolve_placeholder(int(value), placeholder_map),
    )
    return (
        post_state.facts_by_entity,
        post_state.predicates_present,
        post_state.missing_edits,
        post_state.unresolved_edits,
        post_state.applied_additions,
        post_state.applied_deletions,
    )


def _local_satisfied_fraction(checkable: Sequence[bool], satisfied: Sequence[int]) -> float:
    denom = sum(1 for flag in checkable if flag)
    if not denom:
        return 0.0
    return float(sum(int(value) for value, flag in zip(satisfied, checkable) if flag)) / float(denom)


def _resolved_candidate_triples(
    candidate_slots: Sequence[int],
    placeholder_map: Dict[Any, Any],
) -> Dict[str, Tuple[Any, Any, Any] | None]:
    if len(candidate_slots) < 6:
        return {"add": None, "del": None}

    def _triple(base_idx: int) -> Tuple[Any, Any, Any] | None:
        subj = _resolve_placeholder(int(candidate_slots[base_idx]), placeholder_map)
        pred = _resolve_placeholder(int(candidate_slots[base_idx + 1]), placeholder_map)
        obj = _resolve_placeholder(int(candidate_slots[base_idx + 2]), placeholder_map)
        if subj in (None, "", 0) or pred in (None, "", 0) or obj in (None, "", 0):
            return None
        return subj, pred, obj

    return {"add": _triple(0), "del": _triple(3)}


def _evidence_preservation_details(
    *,
    pre_state: EvidenceState,
    post_state: EvidenceState,
    candidate_slots: Sequence[int],
    placeholder_map: Dict[Any, Any],
    primary_satisfied: int,
    pre_global_satisfied_fraction: float,
    post_global_satisfied_fraction: float,
) -> Dict[str, Any]:
    pre_focus_present = bool(pre_state.focus_statement_present())
    post_focus_present = bool(post_state.focus_statement_present())
    focus_preserved = pre_focus_present and post_focus_present
    focus_deleted = pre_focus_present and not post_focus_present
    resolved = _resolved_candidate_triples(candidate_slots, placeholder_map)
    candidate_deletes_focus = resolved.get("del") == (
        pre_state.focus_subject,
        pre_state.focus_predicate,
        pre_state.focus_object,
    )
    return {
        "pre_global_satisfied_fraction": pre_global_satisfied_fraction,
        "post_global_satisfied_fraction": post_global_satisfied_fraction,
        "pre_focus_present": int(pre_focus_present),
        "post_focus_present": int(post_focus_present),
        "focus_preserved": int(focus_preserved),
        "focus_deleted": int(focus_deleted),
        "candidate_deletes_focus": int(candidate_deletes_focus),
        "resolved_add": resolved.get("add"),
        "resolved_del": resolved.get("del"),
        "non_vacuous_primary_fix": int(int(primary_satisfied) == 1 and focus_preserved),
        "vacuous_satisfaction_improvement": int(
            focus_deleted and post_global_satisfied_fraction > pre_global_satisfied_fraction
        ),
    }


def _resolve_default_relations(encoder: GlobalIntEncoder | None) -> List[int]:
    if encoder is None:
        return []
    p31 = _resolve_registry_id("P31", encoder)
    p279 = _resolve_registry_id("P279", encoder)
    defaults = [pid for pid in (p31, p279) if pid]
    return defaults


def _constraint_type_id_from_registry(
    registry_entry: RegistryEntry,
    encoder: GlobalIntEncoder | None,
) -> int:
    del encoder
    return int(registry_entry.constraint_type_index)


def _build_constraint_instance(
    constraint_id: int,
    registry_entry: RegistryEntry,
    *,
    encoder: GlobalIntEncoder | None,
    constraint_type_name: str,
    constraint_type_id: int,
    default_relation_predicates: List[int],
) -> ConstraintInstance:
    p31 = default_relation_predicates[0] if default_relation_predicates else 0
    p279 = default_relation_predicates[1] if len(default_relation_predicates) > 1 else 0
    return parse_constraint_instance(
        constraint_id=constraint_id,
        constraint_type=constraint_type_name,
        constraint_type_id=constraint_type_id,
        constrained_property_raw=registry_entry.constrained_property_raw,
        param_predicates_raw=registry_entry.param_predicates_raw,
        param_objects_raw=registry_entry.param_objects_raw,
        resolve_id=lambda raw: _resolve_registry_id(raw, encoder),
        p31_predicate=p31,
        p279_predicate=p279,
    )


class CandidateConstraintEvaluator:
    def __init__(
        self,
        registry_path: str,
        *,
        encoder: GlobalIntEncoder | None,
        assume_complete: bool,
        constraint_scope: str,
        use_encoded_ids: bool,
        hierarchy_path: str | Path | None = "data/static/wikidata-p279-2018-07-01.v2.json",
        require_hierarchy: bool = False,
    ) -> None:
        registry_raw = _load_registry(registry_path)
        self._registry_by_id = _resolve_registry_mapping(
            registry_raw, encoder=encoder, use_encoded_ids=use_encoded_ids
        )
        self._encoder = encoder
        self._assume_complete = assume_complete
        self._constraint_scope = constraint_scope
        self._use_encoded_ids = use_encoded_ids
        self._default_relations = _resolve_default_relations(encoder)
        self._placeholder_token_ids = _resolve_placeholder_token_ids(encoder)
        self._constraint_cache: Dict[str, ConstraintInstance] = {}
        self.validator_semantics_version = VALIDATOR_SEMANTICS_VERSION
        self._hierarchy: ClassHierarchy | None = None
        if hierarchy_path is not None and Path(hierarchy_path).exists():
            self._hierarchy = ClassHierarchy.from_artifact(
                Path(hierarchy_path),
                resolve_id=lambda raw: _resolve_registry_id(raw, encoder),
                expected_cutoff=HIERARCHY_CUTOFF,
            )
        elif require_hierarchy:
            raise FileNotFoundError(f"Validator semantics v3 requires hierarchy artifact {hierarchy_path}")

    @property
    def hierarchy_identity(self) -> dict[str, object] | None:
        if self._hierarchy is None or self._hierarchy.identity is None:
            return None
        return dict(self._hierarchy.identity.__dict__)

    def _get_constraint_instance(self, constraint_id: Any) -> ConstraintInstance | None:
        entry = _lookup_registry_entry(
            constraint_id, self._registry_by_id, use_encoded_ids=self._use_encoded_ids
        )
        if entry is None:
            return None
        cache_key = str(int(constraint_id)) if self._use_encoded_ids else str(constraint_id)
        cached = self._constraint_cache.get(cache_key)
        if cached is not None:
            return cached
        type_name = entry.constraint_family or ""
        constraint_type_id = _constraint_type_id_from_registry(entry, self._encoder)
        instance = _build_constraint_instance(
            int(constraint_id) if self._use_encoded_ids else 0,
            entry,
            encoder=self._encoder,
            constraint_type_name=type_name,
            constraint_type_id=constraint_type_id,
            default_relation_predicates=self._default_relations,
        )
        self._constraint_cache[cache_key] = instance
        return instance

    def constraint_instance(self, constraint_id: Any) -> ConstraintInstance | None:
        """Return the shared-parser definition used by every evaluator path."""

        return self._get_constraint_instance(constraint_id)

    def evaluate(
        self,
        row: Any,
        *,
        candidate_slots: Sequence[int],
        primary_factor_index: int,
    ) -> CandidateMetrics:
        details = self.evaluate_full(
            row,
            candidate_slots=candidate_slots,
            primary_factor_index=primary_factor_index,
        )
        return _metrics_from_details(details)

    def evaluate_candidates_loss_terms(
        self,
        row: Any,
        *,
        candidates: Sequence[Sequence[int]],
        gold_index: int,
        primary_factor_index: int | None = None,
        need_regression: bool = True,
        need_primary: bool = False,
    ) -> Tuple[List[float], List[float] | None]:
        """
        Compute only the chooser loss terms required by training.

        Returns:
            regression_rates: per-candidate secondary regression rates (zeros when disabled).
            primary_flags: per-candidate primary satisfaction flags (or None when disabled).
        """
        if not candidates:
            return [], [] if need_primary else None
        if gold_index < 0 or gold_index >= len(candidates):
            raise ValueError(
                f"gold_index {gold_index} out of range for {len(candidates)} candidates."
            )
        if not need_regression and not need_primary:
            return [0.0] * len(candidates), None
        details = self.evaluate_candidates(
            row,
            candidates=candidates,
            primary_factor_index=primary_factor_index,
        )
        primary_index = int(details[0]["primary_factor_index"])
        tracked = [
            index
            for index, (checkable, satisfied) in enumerate(
                zip(details[gold_index]["post_checkable"], details[gold_index]["post_satisfied"])
            )
            if index != primary_index and checkable and satisfied
        ]
        regression_rates: List[float] = []
        primary_flags: List[float] | None = [] if need_primary else None
        for detail in details:
            if need_regression:
                supported = [index for index in tracked if detail["post_checkable"][index]]
                regressions = sum(not bool(detail["post_satisfied"][index]) for index in supported)
                regression_rates.append(float(regressions) / len(supported) if supported else 0.0)
            else:
                regression_rates.append(0.0)
            if primary_flags is not None:
                primary_flags.append(
                    float(
                        bool(detail["post_checkable"][primary_index])
                        and bool(detail["post_satisfied"][primary_index])
                    )
                )
        return regression_rates, primary_flags

    def evaluate_full(
        self,
        row: Any,
        *,
        candidate_slots: Sequence[int],
        primary_factor_index: int | None = None,
        factor_constraint_ids: Sequence[int] | None = None,
    ) -> Dict[str, Any]:
        p_local = _compute_p_local(row, cast_int=self._use_encoded_ids)
        p_local_set = p_local
        facts_by_entity, predicates_present = _build_facts_state(
            row,
            p_local=p_local,
            assume_complete=self._assume_complete,
            cast_int=self._use_encoded_ids,
        )
        subject = _coerce_value(getattr(row, "subject", 0), cast_int=self._use_encoded_ids)
        predicate = _coerce_value(getattr(row, "predicate", 0), cast_int=self._use_encoded_ids)
        obj = _coerce_value(getattr(row, "object", 0), cast_int=self._use_encoded_ids)
        other_subject = _coerce_value(getattr(row, "other_subject", 0), cast_int=self._use_encoded_ids)
        other_predicate = _coerce_value(getattr(row, "other_predicate", 0), cast_int=self._use_encoded_ids)
        other_object = _coerce_value(getattr(row, "other_object", 0), cast_int=self._use_encoded_ids)

        pre_state = EvidenceState(
            facts_by_entity=facts_by_entity,
            predicates_present=predicates_present,
            assume_complete=self._assume_complete,
            missing_edits=set(),
            focus_subject=subject,
            focus_predicate=predicate,
            focus_object=obj,
            other_subject=other_subject,
            other_predicate=other_predicate,
            other_object=other_object,
        )
        if not pre_state.focus_statement_present():
            raise AssertionError("Corrected evaluation pre-state is missing its base statement")

        placeholder_map = _build_placeholder_map(self._encoder, row, self._placeholder_token_ids)
        (
            post_facts,
            post_predicates,
            missing_edits,
            unresolved_edits,
            applied_additions,
            applied_deletions,
        ) = _build_post_state_for_candidate(
            facts_by_entity,
            predicates_present,
            p_local,
            candidate_slots=candidate_slots,
            placeholder_map=placeholder_map,
            assume_complete=self._assume_complete,
        )
        post_state = EvidenceState(
            facts_by_entity=post_facts,
            predicates_present=post_predicates,
            assume_complete=self._assume_complete,
            missing_edits=missing_edits,
            focus_subject=subject,
            focus_predicate=predicate,
            focus_object=obj,
            other_subject=other_subject,
            other_predicate=other_predicate,
            other_object=other_object,
            unresolved_edits=unresolved_edits,
            applied_additions=applied_additions,
            applied_deletions=applied_deletions,
        )

        coerce_id = lambda value: _coerce_value(value, cast_int=self._use_encoded_ids)
        local_constraint_ids = select_constraint_ids(
            row,
            constraint_scope=self._constraint_scope,
            explicit_ids=factor_constraint_ids,
            coerce=coerce_id,
        )

        pre_checkable: List[bool] = []
        pre_satisfied: List[int] = []
        post_checkable: List[bool] = []
        post_satisfied: List[int] = []

        resolved_primary_index = resolve_primary_index(
            row,
            local_constraint_ids,
            supplied_index=primary_factor_index,
            coerce=coerce_id,
        )

        pre_outcomes: List[str] = []
        post_outcomes: List[str] = []
        pre_applicable: List[bool] = []
        post_applicable: List[bool] = []
        pre_unknown_reasons: List[List[str]] = []
        post_unknown_reasons: List[List[str]] = []

        for index, constraint_id in enumerate(local_constraint_ids):
            instance = self._get_constraint_instance(constraint_id)
            if instance is None:
                pre_checkable.append(False)
                pre_satisfied.append(0)
                post_checkable.append(False)
                post_satisfied.append(0)
                pre_outcomes.append("unknown")
                post_outcomes.append("unknown")
                pre_applicable.append(False)
                post_applicable.append(False)
                pre_unknown_reasons.append(["constraint missing from registry"])
                post_unknown_reasons.append(["constraint missing from registry"])
                continue
            is_primary = index == resolved_primary_index
            pre_result = evaluate_constraint_detailed(
                pre_state, instance, p_local_set, hierarchy=self._hierarchy, primary=is_primary
            )
            post_result = evaluate_constraint_detailed(
                post_state, instance, p_local_set, hierarchy=self._hierarchy, primary=is_primary
            )
            pre_checkable.append(pre_result.outcome.value != "unknown")
            pre_satisfied.append(int(pre_result.outcome.value == "satisfied"))
            post_checkable.append(post_result.outcome.value != "unknown")
            post_satisfied.append(int(post_result.outcome.value == "satisfied"))
            pre_outcomes.append(pre_result.outcome.value)
            post_outcomes.append(post_result.outcome.value)
            pre_applicable.append(pre_result.applicable)
            post_applicable.append(post_result.applicable)
            pre_unknown_reasons.append(list(pre_result.unknown_reasons))
            post_unknown_reasons.append(list(post_result.unknown_reasons))

        primary_satisfied = 0
        if 0 <= resolved_primary_index < len(post_satisfied):
            primary_satisfied = post_satisfied[resolved_primary_index]

        pre_global_satisfied_fraction = _local_satisfied_fraction(pre_checkable, pre_satisfied)
        global_satisfied_fraction = _local_satisfied_fraction(post_checkable, post_satisfied)

        secondary_regressions = 0
        secondary_improvements = 0
        secondary_regressions_denom = 0
        secondary_improvements_denom = 0
        for idx in range(len(local_constraint_ids)):
            if idx == resolved_primary_index:
                continue
            if not pre_checkable[idx]:
                continue
            if not post_checkable[idx]:
                continue
            if pre_satisfied[idx]:
                secondary_regressions_denom += 1
                if not post_satisfied[idx]:
                    secondary_regressions += 1
            else:
                secondary_improvements_denom += 1
                if post_satisfied[idx]:
                    secondary_improvements += 1

        srr = float(secondary_regressions) / secondary_regressions_denom if secondary_regressions_denom else 0.0
        sir = float(secondary_improvements) / secondary_improvements_denom if secondary_improvements_denom else 0.0

        add_count = 0
        del_count = 0
        if len(candidate_slots) >= 6:
            if all(int(v) != 0 for v in candidate_slots[:3]):
                add_count = 1
            if all(int(v) != 0 for v in candidate_slots[3:6]):
                del_count = 1

        evidence_details = _evidence_preservation_details(
            pre_state=pre_state,
            post_state=post_state,
            candidate_slots=candidate_slots,
            placeholder_map=placeholder_map,
            primary_satisfied=primary_satisfied,
            pre_global_satisfied_fraction=pre_global_satisfied_fraction,
            post_global_satisfied_fraction=global_satisfied_fraction,
        )

        return {
            "local_constraint_ids": local_constraint_ids,
            "primary_factor_index": resolved_primary_index,
            "pre_checkable": pre_checkable,
            "pre_satisfied": pre_satisfied,
            "post_checkable": post_checkable,
            "post_satisfied": post_satisfied,
            "pre_outcomes": pre_outcomes,
            "post_outcomes": post_outcomes,
            "pre_applicable": pre_applicable,
            "post_applicable": post_applicable,
            "pre_unknown_reasons": pre_unknown_reasons,
            "post_unknown_reasons": post_unknown_reasons,
            "edit_applicable": not bool(unresolved_edits),
            "unresolved_edits": [edit.__dict__ for edit in unresolved_edits],
            "primary_satisfied": primary_satisfied,
            "primary_pre_violated": int(
                pre_outcomes[resolved_primary_index] == "violated"
            ),
            "global_satisfied_fraction": global_satisfied_fraction,
            "secondary_regressions": secondary_regressions,
            "secondary_improvements": secondary_improvements,
            "secondary_regressions_denom": secondary_regressions_denom,
            "secondary_improvements_denom": secondary_improvements_denom,
            "srr": srr,
            "sir": sir,
            "add_count": add_count,
            "del_count": del_count,
            **evidence_details,
        }

    def evaluate_candidates(
        self,
        row: Any,
        *,
        candidates: Sequence[Sequence[int]],
        primary_factor_index: int | None = None,
    ) -> List[Dict[str, Any]]:
        return [
            self.evaluate_full(
                row,
                candidate_slots=candidate,
                primary_factor_index=primary_factor_index,
            )
            for candidate in candidates
        ]

    def evaluate_candidate_metrics(
        self,
        row: Any,
        *,
        candidates: Sequence[Sequence[int]],
        primary_factor_index: int | None = None,
    ) -> List[CandidateMetrics]:
        details = self.evaluate_candidates(
            row,
            candidates=candidates,
            primary_factor_index=primary_factor_index,
        )
        return [_metrics_from_details(detail) for detail in details]


def _metrics_from_details(details: Dict[str, Any]) -> CandidateMetrics:
    primary_index = int(details.get("primary_factor_index", -1))
    post_checkable = details.get("post_checkable") or []
    primary_checkable = int(
        0 <= primary_index < len(post_checkable) and bool(post_checkable[primary_index])
    )
    return CandidateMetrics(
        primary_satisfied=int(details.get("primary_satisfied", 0)),
        primary_checkable=primary_checkable,
        primary_pre_violated=int(details.get("primary_pre_violated", 0)),
        global_satisfied_fraction=float(details.get("global_satisfied_fraction", 0.0)),
        secondary_regressions=int(details.get("secondary_regressions", 0)),
        secondary_regressions_denom=int(details.get("secondary_regressions_denom", 0)),
        secondary_improvements=int(details.get("secondary_improvements", 0)),
        secondary_improvements_denom=int(details.get("secondary_improvements_denom", 0)),
        srr=float(details.get("srr", 0.0)),
        sir=float(details.get("sir", 0.0)),
        add_count=int(details.get("add_count", 0)),
        del_count=int(details.get("del_count", 0)),
        focus_preserved=int(details.get("focus_preserved", 0)),
        focus_deleted=int(details.get("focus_deleted", 0)),
        candidate_deletes_focus=int(details.get("candidate_deletes_focus", 0)),
    )
