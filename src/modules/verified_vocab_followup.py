"""Pure helpers for the verified-training output-vocabulary follow-up.

The helpers in this module never run a symbolic validator and never use neural
feature IDs as semantic identity.  They consume the lossless terms and fixed
audited-v3 flags already recorded by the identity-preserving audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from modules.identity_audit import (
    ENTITY_SLOT_INDICES,
    PREDICATE_SLOT_INDICES,
    ROLE_NAMES,
    choose_role_reference,
    roundtrip_slots,
)


SLOT_NAMES: tuple[str, ...] = (
    "add_subject",
    "add_predicate",
    "add_object",
    "del_subject",
    "del_predicate",
    "del_object",
)
ACTION_CATEGORIES: tuple[str, ...] = (
    "addition_only",
    "deletion_only",
    "paired_replacement_same_subject_property",
    "paired_other",
)
SUPPORT_BUCKETS: tuple[str, ...] = ("0", "1", "2-4", "5-9", "10-49", "50-99", ">=100")


def slot_type(index: int) -> str:
    return "predicate" if index in PREDICATE_SLOT_INDICES else "entity"


def slots_from_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(name) or "") for name in SLOT_NAMES)


def support_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 9:
        return "5-9"
    if value <= 49:
        return "10-49"
    if value <= 99:
        return "50-99"
    return ">=100"


def validation_support_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 9:
        return "1-9"
    if value <= 99:
        return "10-99"
    return ">=100"


def term_kind(term: str) -> str:
    if term.startswith("wd:Q"):
        return "Q-item"
    if term.startswith("wd:P"):
        return "P-property"
    if term.startswith("lit:"):
        return "literal"
    if term.startswith("iri:"):
        return "IRI"
    if term.startswith("bnode:"):
        return "blank-node"
    if term.startswith("special:"):
        return "special-snak"
    return "other"


_DATATYPE_RE = re.compile(r"\^\^<([^>]+)>\s*$")
_LANG_RE = re.compile(r"@([A-Za-z0-9-]+)\s*$")


def literal_datatype(term: str) -> str:
    if not term.startswith("lit:"):
        return "not_literal"
    match = _DATATYPE_RE.search(term)
    if match:
        return match.group(1)
    match = _LANG_RE.search(term)
    if match:
        return f"language:{match.group(1).lower()}"
    return "plain_or_source_untyped"


def _lowest_mask_index(mask: int) -> int | None:
    if not mask:
        return None
    return int((int(mask) & -int(mask)).bit_length() - 1)


@dataclass
class VerifiedVocabularyBuilder:
    """Fit output references from parent TRAIN rows whose fixed v3 F flag is true."""

    role_encoder_ids: Mapping[str, int]
    input_encoder_size: int
    entity_ids: set[int] = field(default_factory=lambda: {0})
    predicate_ids: set[int] = field(default_factory=lambda: {0})
    entity_terms: dict[str, int] = field(default_factory=dict)
    predicate_terms: dict[str, int] = field(default_factory=dict)
    entity_roles: set[int] = field(default_factory=set)
    predicate_roles: set[int] = field(default_factory=set)
    fit_rows: int = 0
    target_slots: int = 0
    none_slots: int = 0
    role_slots: int = 0
    concrete_slots: int = 0
    unrepresentable_slots: int = 0
    raw_support: Counter[tuple[int, str, str]] = field(default_factory=Counter)
    train_seen_terms: set[str] = field(default_factory=set)

    def add_row(self, row: Mapping[str, Any]) -> bool:
        """Add a row only when it belongs to the explicitly frozen fitting cohort."""

        if str(row.get("split")) != "train" or not bool(row.get("v3_F")):
            return False
        self.fit_rows += 1
        slots = slots_from_row(row)
        masks = [int(value) for value in (row.get("target_role_masks") or ())]
        encoder_ids = [int(value) for value in (row.get("target_encoder_ids") or ())]
        if len(masks) != 6 or len(encoder_ids) != 6:
            raise ValueError("target role/id arrays must contain six slots")
        for index, term in enumerate(slots):
            self.target_slots += 1
            if not term:
                self.none_slots += 1
                self.raw_support[(index, "none", "NONE")] += 1
                continue
            self.train_seen_terms.add(term)
            role_index = _lowest_mask_index(masks[index])
            target_ids = self.predicate_ids if index in PREDICATE_SLOT_INDICES else self.entity_ids
            target_roles = (
                self.predicate_roles if index in PREDICATE_SLOT_INDICES else self.entity_roles
            )
            target_terms = (
                self.predicate_terms if index in PREDICATE_SLOT_INDICES else self.entity_terms
            )
            if role_index is not None:
                role_name = ROLE_NAMES[role_index]
                role_id = int(self.role_encoder_ids.get(role_name, 0))
                if role_id <= 0:
                    raise ValueError(f"role {role_name!r} is absent from the fixed encoder")
                target_roles.add(role_index)
                target_ids.add(role_id)
                self.role_slots += 1
                self.raw_support[(index, "role", role_name)] += 1
                continue
            encoder_id = encoder_ids[index]
            if encoder_id > 0:
                previous = target_terms.setdefault(term, encoder_id)
                if previous != encoder_id:
                    raise AssertionError(f"semantic term {term!r} maps to multiple fixed encoder IDs")
                target_ids.add(encoder_id)
                self.concrete_slots += 1
                self.raw_support[(index, "constant", term)] += 1
            else:
                self.unrepresentable_slots += 1
                self.raw_support[(index, "unrepresentable", term)] += 1
        return True

    def build(self) -> dict[str, Any]:
        entity_classes = [
            {"semantic_term": term, "encoder_id": value}
            for term, value in sorted(self.entity_terms.items())
        ]
        predicate_classes = [
            {"semantic_term": term, "encoder_id": value}
            for term, value in sorted(self.predicate_terms.items())
        ]
        entity_role_indices = sorted(self.entity_roles)
        predicate_role_indices = sorted(self.predicate_roles)
        core = {
            "profile": "verified_train_minocc100",
            "threshold": 100,
            "input_encoder_size": int(self.input_encoder_size),
            "fit_population": "parent_pool/train/audited_v3_bounded:F",
            "fit_predicate": "split == train AND v3_F == true",
            "fit_rows": self.fit_rows,
            "entity_class_ids": sorted(self.entity_ids),
            "predicate_class_ids": sorted(self.predicate_ids),
            "entity_terms": sorted(self.entity_terms),
            "predicate_terms": sorted(self.predicate_terms),
            "entity_constant_classes": entity_classes,
            "predicate_constant_classes": predicate_classes,
            "entity_role_indices": entity_role_indices,
            "predicate_role_indices": predicate_role_indices,
            "role_tokens": [
                {
                    "role_index": index,
                    "role_name": name,
                    "encoder_id": int(self.role_encoder_ids[name]),
                }
                for index, name in enumerate(ROLE_NAMES)
                if index in self.entity_roles or index in self.predicate_roles
            ],
            "none": {"name": "NONE", "class_id": 0},
            "fit_slot_counts": {
                "all": self.target_slots,
                "none": self.none_slots,
                "role": self.role_slots,
                "concrete": self.concrete_slots,
                "unrepresentable": self.unrepresentable_slots,
            },
            "policy": {
                "input_feature_vocabulary": "fixed existing threshold-100 encoder",
                "semantic_equality": "lossless canonical terms; never neural UNK",
                "role_precedence": "lowest-index true semantic alias before concrete constant",
                "concrete_eligibility": "nonzero fixed encoder ID; no target-frequency refit",
                "validation_or_test_targets_used": False,
                "copy_references": False,
            },
        }
        per_slot: list[dict[str, Any]] = []
        for index, name in enumerate(SLOT_NAMES):
            kind = slot_type(index)
            per_slot.append(
                {
                    "slot_index": index,
                    "slot": name,
                    "term_type": kind,
                    "class_ids": core[f"{kind}_class_ids"],
                    "constant_terms": core[f"{kind}_terms"],
                    "role_indices": core[f"{kind}_role_indices"],
                    "role_tokens": [
                        ROLE_NAMES[role_index] for role_index in core[f"{kind}_role_indices"]
                    ],
                    "none_class_id": 0,
                }
            )
        core["per_slot"] = per_slot
        fingerprint_payload = {
            key: core[key]
            for key in (
                "profile",
                "threshold",
                "input_encoder_size",
                "fit_population",
                "entity_class_ids",
                "predicate_class_ids",
                "entity_terms",
                "predicate_terms",
                "entity_role_indices",
                "predicate_role_indices",
                "policy",
            )
        }
        core["fingerprint"] = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return core


def permitted_role_indices(vocabulary: Mapping[str, Any], slot_index: int) -> set[int]:
    key = "predicate_role_indices" if slot_index in PREDICATE_SLOT_INDICES else "entity_role_indices"
    cache_key = f"_{key}_set"
    cached = vocabulary.get(cache_key)
    if cached is not None:
        return cached
    return {int(value) for value in vocabulary[key]}


def constant_terms(vocabulary: Mapping[str, Any], slot_index: int) -> set[str]:
    key = "predicate_terms" if slot_index in PREDICATE_SLOT_INDICES else "entity_terms"
    cache_key = f"_{key}_set"
    cached = vocabulary.get(cache_key)
    if cached is not None:
        return cached
    return set(vocabulary[key])


def compile_vocabulary(vocabulary: Mapping[str, Any]) -> dict[str, Any]:
    """Attach immutable lookup sets without changing the persisted vocabulary."""

    compiled = dict(vocabulary)
    for key in (
        "entity_terms",
        "predicate_terms",
        "entity_role_indices",
        "predicate_role_indices",
    ):
        compiled[f"_{key}_set"] = frozenset(vocabulary[key])
    return compiled


def resolve_reference(
    *, term: str, role_mask: int, slot_index: int, vocabulary: Mapping[str, Any]
) -> dict[str, Any]:
    if not term:
        return {"covered": True, "kind": "none", "reference": "NONE", "term": term}
    permitted = permitted_role_indices(vocabulary, slot_index)
    role_index = choose_role_reference(int(role_mask), permitted)
    if role_index is not None:
        return {
            "covered": True,
            "kind": "role",
            "reference": ROLE_NAMES[role_index],
            "role_index": role_index,
            "term": term,
        }
    if term in constant_terms(vocabulary, slot_index):
        return {"covered": True, "kind": "constant", "reference": term, "term": term}
    return {"covered": False, "kind": "unresolved", "reference": None, "term": term}


def edit_coverage(row: Mapping[str, Any], vocabulary: Mapping[str, Any]) -> dict[str, Any]:
    slots = slots_from_row(row)
    masks = [int(value) for value in (row.get("target_role_masks") or ())]
    if len(masks) != 6:
        raise ValueError("target_role_masks must contain six entries")
    references = [
        resolve_reference(term=term, role_mask=masks[index], slot_index=index, vocabulary=vocabulary)
        for index, term in enumerate(slots)
    ]
    covered = bool(row.get("R", True)) and all(reference["covered"] for reference in references)
    if covered:
        role_sets = [permitted_role_indices(vocabulary, index) for index in range(6)]
        term_sets = [constant_terms(vocabulary, index) for index in range(6)]
        roles = tuple(str(row.get(name) or "") for name in ROLE_NAMES)
        exact, resolved = roundtrip_slots(
            slots,
            roles,
            role_indices_by_slot=role_sets,
            constants_by_slot=term_sets,
        )
        if not exact or tuple(resolved) != slots:
            raise AssertionError("claimed V1 coverage failed exact semantic round-trip")
    return {"covered": covered, "references": references}


def action_features(row: Mapping[str, Any]) -> dict[str, Any]:
    slots = slots_from_row(row)
    add = slots[:3]
    delete = slots[3:]
    has_add = all(add)
    has_delete = all(delete)
    if has_add and not has_delete:
        category = "addition_only"
    elif has_delete and not has_add:
        category = "deletion_only"
    elif has_add and has_delete:
        same_sp = add[0] == delete[0] and add[1] == delete[1] and add[2] != delete[2]
        category = "paired_replacement_same_subject_property" if same_sp else "paired_other"
    else:
        raise ValueError("compatible historical edit is neither addition, deletion, nor paired")
    base = tuple(str(row.get(name) or "") for name in ROLE_NAMES[:3])
    base_deleted_operation = has_delete and delete == base
    base_reinserted = base_deleted_operation and has_add and add == base
    base_ultimately_preserved = not base_deleted_operation or base_reinserted
    return {
        "category": category,
        "paired": has_add and has_delete,
        "same_subject_property_replacement": category
        == "paired_replacement_same_subject_property",
        "base_deleted_operation": base_deleted_operation,
        "base_reinserted": base_reinserted,
        "base_ultimately_preserved": base_ultimately_preserved,
        "base_deleting": not base_ultimately_preserved,
        "changed_subject": bool(has_add and has_delete and add[0] != delete[0]),
        "changed_property": bool(has_add and has_delete and add[1] != delete[1]),
    }


def failure_tags(
    *,
    term: str,
    role_mask: int,
    slot_index: int,
    vocabulary: Mapping[str, Any],
    encoder_id: int,
) -> list[str]:
    """Return deterministic diagnostic tags for one unresolved slot."""

    tags: list[str] = []
    if role_mask and choose_role_reference(role_mask, permitted_role_indices(vocabulary, slot_index)) is None:
        tags.append("role_exists_semantically_but_illegal_for_slot")
    kind = term_kind(term)
    if kind == "literal":
        tags.append("literal_target_not_representable")
    if slot_index in PREDICATE_SLOT_INDICES:
        tags.append("predicate_target_not_in_predicate_output_vocabulary")
    elif kind == "Q-item":
        tags.append("entity_item_target_not_in_entity_output_vocabulary")
    if encoder_id > 0:
        tags.append("fixed_encoder_term_not_in_v1_output_mask")
    elif not role_mask:
        tags.append("outside_fixed_input_output_vocabulary_and_roles")
    if not tags:
        tags.append("other_unclassified")
    return tags


PRIMARY_REASON_PRECEDENCE: tuple[str, ...] = (
    "role_exists_semantically_but_illegal_for_slot",
    "literal_target_not_representable",
    "predicate_target_not_in_predicate_output_vocabulary",
    "entity_item_target_not_in_entity_output_vocabulary",
    "fixed_encoder_term_not_in_v1_output_mask",
    "outside_fixed_input_output_vocabulary_and_roles",
    "other_unclassified",
)


def primary_exclusion_reason(failures: Sequence[Mapping[str, Any]]) -> str:
    present = {tag for failure in failures for tag in failure["tags"]}
    for reason in PRIMARY_REASON_PRECEDENCE:
        if reason in present:
            return reason
    return "other_unclassified"


def js_divergence(counts_a: Mapping[str, int], counts_b: Mapping[str, int]) -> float | None:
    """Return Jensen-Shannon divergence in bits (base-2, range 0..1)."""

    keys = sorted(set(counts_a) | set(counts_b))
    total_a = sum(int(counts_a.get(key, 0)) for key in keys)
    total_b = sum(int(counts_b.get(key, 0)) for key in keys)
    if not total_a or not total_b:
        return None
    result = 0.0
    for key in keys:
        pa = int(counts_a.get(key, 0)) / total_a
        pb = int(counts_b.get(key, 0)) / total_b
        midpoint = (pa + pb) / 2.0
        if pa:
            result += 0.5 * pa * math.log2(pa / midpoint)
        if pb:
            result += 0.5 * pb * math.log2(pb / midpoint)
    return result


def largest_remainder_allocation(weights: Mapping[str, int], total: int) -> dict[str, int]:
    """Allocate an integer total proportionally without exceeding availability."""

    available = {key: max(0, int(value)) for key, value in weights.items()}
    capacity = sum(available.values())
    if total < 0 or total > capacity:
        raise ValueError("requested total exceeds no-oversampling capacity")
    if total == 0:
        return {key: 0 for key in available}
    quotas = {key: total * value / capacity for key, value in available.items()}
    result = {key: min(available[key], int(math.floor(quota))) for key, quota in quotas.items()}
    remaining = total - sum(result.values())
    order = sorted(
        available,
        key=lambda key: (-(quotas[key] - math.floor(quotas[key])), key),
    )
    for key in order:
        if remaining == 0:
            break
        if result[key] < available[key]:
            result[key] += 1
            remaining -= 1
    if remaining:
        raise AssertionError("largest-remainder allocation failed to reconcile")
    return result


def capped_target_allocation(
    desired: Mapping[str, int], available: Mapping[str, int], total: int
) -> dict[str, int]:
    """Preserve desired counts where feasible, then fill deficits without oversampling."""

    keys = sorted(set(desired) | set(available))
    capacity = sum(max(0, int(available.get(key, 0))) for key in keys)
    if total > capacity:
        raise ValueError("requested total exceeds no-oversampling capacity")
    result = {
        key: min(max(0, int(desired.get(key, 0))), max(0, int(available.get(key, 0))))
        for key in keys
    }
    current = sum(result.values())
    if current > total:
        return largest_remainder_allocation(result, total)
    remaining = total - current
    while remaining:
        spare = {key: int(available.get(key, 0)) - result[key] for key in keys}
        spare = {key: value for key, value in spare.items() if value > 0}
        if not spare:
            raise AssertionError("no capacity remains while filling capped allocation")
        addition = largest_remainder_allocation(spare, remaining)
        for key, value in addition.items():
            result[key] += value
        remaining = total - sum(result.values())
    return result


def nearest_rank(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(float(quantile) * len(ordered)))
    return ordered[rank - 1]


def assert_v1_subset_v0(v1: Mapping[str, Any], v0: Mapping[str, Any]) -> None:
    checks = (
        ("entity_terms", set(v1["entity_terms"]), set(v0["entity_terms"])),
        ("predicate_terms", set(v1["predicate_terms"]), set(v0["predicate_terms"])),
        (
            "entity_role_indices",
            set(v1["entity_role_indices"]),
            set(v0["entity_role_indices"]),
        ),
        (
            "predicate_role_indices",
            set(v1["predicate_role_indices"]),
            set(v0["predicate_role_indices"]),
        ),
    )
    failures = {name: sorted(left - right) for name, left, right in checks if left - right}
    if failures:
        raise AssertionError(f"V1 is not a subset of V0: {failures}")
