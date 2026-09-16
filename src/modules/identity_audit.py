"""Lossless parsing and grounding helpers for the decoder/validator audit.

This module deliberately keeps semantic terms as canonical strings.  Neural
encoder IDs are annotations on those strings; they are never used for
equality.  The production validators are called in isolated processes through
``SemanticEncoder``, whose integer representation is a reversible encoding of
the canonical string rather than a frequency-pruned model index.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ADD = "<http://wikiba.se/history/ontology#addition>"
DELETE = "<http://wikiba.se/history/ontology#deletion>"
PAGE_STATUS = "<http://wikiba.se/history/ontology#pageStatusCode>"
PAGE_CONTAINS_LABEL = "<http://wikiba.se/history/ontology#pageContainsLabel>"
ROLE_NAMES: tuple[str, ...] = (
    "subject",
    "predicate",
    "object",
    "other_subject",
    "other_predicate",
    "other_object",
)
ENTITY_SLOT_INDICES = (0, 2, 3, 5)
PREDICATE_SLOT_INDICES = (1, 4)
EMPTY = ""

_ENTITY_RE = re.compile(r"^(?:https?://www\.wikidata\.org/entity/)?([PQ][1-9]\d*)$")
_DIRECT_RE = re.compile(r"^https?://www\.wikidata\.org/prop/direct/([P][1-9]\d*)$")


def strip_brackets(value: str) -> str:
    value = value.strip()
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1].strip()
    return value


def canonical_term(raw: Any, *, blank_scope: str | None = None) -> str:
    """Return a lossless, type-explicit semantic spelling.

    Wikidata entity and direct-property aliases are normalized to ``wd:Q…``
    and ``wd:P…``.  Literals retain their complete RDF spelling, including
    datatype/language.  Blank nodes are scoped to their source record or
    constraint definition.
    """

    if raw is None:
        return EMPTY
    value = str(raw).strip()
    if not value:
        return EMPTY
    if value.startswith(("wd:", "iri:", "lit:", "bnode:", "special:", "role:")):
        return value
    unwrapped = strip_brackets(value)
    direct = _DIRECT_RE.match(unwrapped)
    if direct:
        return f"wd:{direct.group(1)}"
    entity = _ENTITY_RE.match(unwrapped)
    if entity:
        return f"wd:{entity.group(1)}"
    if unwrapped.startswith("_:"):
        return f"bnode:{blank_scope or 'unscoped'}:{unwrapped[2:]}"
    if value.startswith(('"', "'")):
        return f"lit:{value}"
    if unwrapped.startswith(("http://", "https://")):
        return f"iri:{unwrapped}"
    # The flattened registry represents special snaks with non-Q/P tokens.
    if blank_scope and ("somevalue" in unwrapped.lower() or "novalue" in unwrapped.lower()):
        return f"special:{blank_scope}:{unwrapped}"
    return f"lit:{value}"


def semantic_id(raw_or_canonical: Any, *, blank_scope: str | None = None) -> int:
    """Reversibly encode one canonical term as a positive Python integer."""

    term = canonical_term(raw_or_canonical, blank_scope=blank_scope)
    if not term:
        return 0
    # The leading non-zero byte makes int.from_bytes injective over UTF-8
    # strings, including strings with leading NUL bytes.
    return int.from_bytes(b"\x01" + term.encode("utf-8"), "big", signed=False)


def semantic_id_to_term(value: int) -> str:
    if not value:
        return EMPTY
    payload = int(value).to_bytes((int(value).bit_length() + 7) // 8, "big")
    if not payload.startswith(b"\x01"):
        raise ValueError("not an identity-audit semantic ID")
    return payload[1:].decode("utf-8")


class SemanticEncoder:
    """Minimal GlobalIntEncoder-compatible lossless identity adapter."""

    def __init__(self) -> None:
        self._filtered_ids: set[int] = set()
        self._frozen = True

    def encode(self, value: str | None, add_new: bool | None = None) -> int:
        del add_new
        if value in ROLE_NAMES:
            return semantic_id(f"role:{value}")
        return semantic_id(value)

    def decode(self, value: int | None, use_filtered_id_mapping: bool = False) -> str | None:
        del use_filtered_id_mapping
        if value is None:
            raise ValueError("semantic ID is None")
        return semantic_id_to_term(value)

    def freeze(self) -> None:
        self._frozen = True


def stable_record_key(path: str | Path, line_number: int) -> str:
    return f"{Path(path).name}:{int(line_number)}"


def is_json_blob(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def split_description_tail(elements: Sequence[str]) -> tuple[int, list[str]]:
    """Return the exclusive operation boundary and up to three descriptions."""

    idx = len(elements) - 1
    tail: list[str] = []
    while idx >= 0:
        value = elements[idx]
        if value.strip() == "" or is_json_blob(value):
            tail.append(value)
            idx -= 1
            continue
        break
    tail.reverse()
    return len(elements) - len(tail), tail[-3:] if tail else []


@dataclass(frozen=True)
class ParsedOperation:
    kind: str
    subject: str
    predicate: str
    object: str
    source_position: int

    @property
    def complete(self) -> bool:
        return bool(self.subject and self.predicate and self.object)

    def triple(self) -> tuple[str, str, str]:
        return self.subject, self.predicate, self.object


@dataclass
class ParsedRecord:
    source_key: str
    constraint_id: str
    revision: str
    roles: tuple[str, str, str, str, str, str]
    facts_by_owner: dict[str, list[tuple[str, str]]]
    described_owners: set[str]
    operations: list[ParsedOperation]
    compatible: bool
    exclusion_reason: str
    parse_warnings: list[str]

    @property
    def additions(self) -> list[ParsedOperation]:
        return [operation for operation in self.operations if operation.kind == "add"]

    @property
    def deletions(self) -> list[ParsedOperation]:
        return [operation for operation in self.operations if operation.kind == "del"]

    def six_slots(self) -> tuple[str, str, str, str, str, str]:
        add = self.additions[0].triple() if len(self.additions) == 1 else (EMPTY,) * 3
        delete = self.deletions[0].triple() if len(self.deletions) == 1 else (EMPTY,) * 3
        return (*add, *delete)


def _status_iri(code: Any) -> str:
    try:
        phrase = HTTPStatus(int(code)).phrase
    except Exception:
        phrase = f"Status{code}"
    name = phrase.title().replace(" ", "").replace("-", "")
    return canonical_term(f"<http://www.w3.org/2011/http-statusCodes#{name}>")


def _description_payload(raw: str, *, source_key: str) -> tuple[dict[str, Any] | None, str | None]:
    if not raw.strip():
        return None, None
    if not is_json_blob(raw):
        return None, "non-json description field"
    try:
        value = json.loads(raw)
    except ValueError:
        return None, "invalid description JSON"
    if not isinstance(value, dict):
        return None, "description JSON is not an object"
    kind = value.get("type")
    if kind not in {"entity", "page"}:
        return value, f"unsupported description type {kind!r}"
    return value, None


def _entity_facts(desc: Mapping[str, Any], *, blank_scope: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    facts = desc.get("facts") or {}
    if not isinstance(facts, Mapping):
        return result
    for predicate, objects in facts.items():
        pred = canonical_term(predicate, blank_scope=blank_scope)
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        for obj in objects:
            result.append((pred, canonical_term(obj, blank_scope=blank_scope)))
    return result


def parse_raw_record(elements: Sequence[str], *, source_key: str) -> ParsedRecord:
    """Parse one accepted raw correction row without neural-ID compression."""

    warnings: list[str] = []
    if len(elements) < 9:
        return ParsedRecord(
            source_key,
            canonical_term(elements[0], blank_scope=source_key) if elements else EMPTY,
            canonical_term(elements[1]) if len(elements) > 1 else EMPTY,
            (EMPTY,) * 6,
            {},
            set(),
            [],
            False,
            "malformed_tsv_short",
            [f"expected at least 9 fields, got {len(elements)}"],
        )

    constraint_id = canonical_term(elements[0], blank_scope=f"constraint:{elements[0]}")
    revision = canonical_term(elements[1])
    roles = tuple(canonical_term(value, blank_scope=source_key) for value in elements[2:8])
    op_end, desc_fields = split_description_tail(elements)
    operations: list[ParsedOperation] = []
    bad_layout = op_end < 9 or (op_end - 9) % 4 != 0
    if bad_layout:
        warnings.append(f"operation field layout 9:{op_end} is not divisible into four-field groups")
    else:
        for pos in range(9, op_end, 4):
            subject, predicate, obj, raw_kind = elements[pos : pos + 4]
            if raw_kind == ADD:
                kind = "add"
            elif raw_kind == DELETE:
                kind = "del"
            else:
                kind = "unsupported"
            operations.append(
                ParsedOperation(
                    kind=kind,
                    subject=canonical_term(subject, blank_scope=source_key),
                    predicate=canonical_term(predicate, blank_scope=source_key),
                    object=canonical_term(obj, blank_scope=source_key),
                    source_position=pos,
                )
            )

    subject, _predicate, obj, other_subject, _other_predicate, other_obj = roles
    facts_by_owner: dict[str, list[tuple[str, str]]] = {}
    described_owners: set[str] = set()
    labels_by_owner: dict[str, list[str]] = {}
    page_descriptions: list[dict[str, Any]] = []

    for desc_index, raw_desc in enumerate(desc_fields):
        desc, warning = _description_payload(raw_desc, source_key=source_key)
        if warning:
            warnings.append(f"description[{desc_index}]: {warning}")
        if not desc:
            continue
        if desc.get("type") == "entity":
            owner = canonical_term(desc.get("id"), blank_scope=source_key)
            if not owner:
                warnings.append(f"description[{desc_index}] entity has no id")
                continue
            described_owners.add(owner)
            facts_by_owner.setdefault(owner, []).extend(
                _entity_facts(desc, blank_scope=f"{source_key}:description:{desc_index}")
            )
            labels = desc.get("labels") or {}
            if isinstance(labels, Mapping):
                labels_by_owner.setdefault(owner, []).extend(str(label) for label in labels.values())
        else:
            page_descriptions.append(desc)

    # A page description has no semantic owner in the source.  The corpus
    # schema places it in the focus-object evidence slot, which is the only
    # ownership assertion we can preserve without invention.
    for page in page_descriptions:
        if not obj:
            warnings.append("page description has no focus-object owner")
            continue
        described_owners.add(obj)
        facts_by_owner.setdefault(obj, []).append((canonical_term(PAGE_STATUS), _status_iri(page.get("statusCode"))))
        content = str(page.get("content") or "")
        for owner in (subject, other_subject, obj, other_obj):
            if not owner:
                continue
            if any(label and label in content for label in labels_by_owner.get(owner, ())):
                facts_by_owner[obj].append((canonical_term(PAGE_CONTAINS_LABEL), owner))

    additions = [operation for operation in operations if operation.kind == "add"]
    deletions = [operation for operation in operations if operation.kind == "del"]
    unsupported = [operation for operation in operations if operation.kind == "unsupported"]
    incomplete = [operation for operation in operations if not operation.complete]
    reasons: list[str] = []
    if bad_layout:
        reasons.append("malformed_operation_layout")
    if unsupported:
        reasons.append("unsupported_operation_kind")
    if incomplete:
        reasons.append("incomplete_operation")
    if len(additions) > 1:
        reasons.append("multiple_additions")
    if len(deletions) > 1:
        reasons.append("multiple_deletions")
    compatible = not reasons
    return ParsedRecord(
        source_key=source_key,
        constraint_id=constraint_id,
        revision=revision,
        roles=roles,  # type: ignore[arg-type]
        facts_by_owner=facts_by_owner,
        described_owners=described_owners,
        operations=operations,
        compatible=compatible,
        exclusion_reason=";".join(reasons),
        parse_warnings=warnings,
    )


def role_alias_mask(term: str, roles: Sequence[str]) -> int:
    mask = 0
    if not term:
        return mask
    for index, role_term in enumerate(roles):
        if role_term and role_term == term:
            mask |= 1 << index
    return mask


def choose_role_reference(mask: int, permitted_roles: Iterable[int]) -> int | None:
    permitted = set(int(value) for value in permitted_roles)
    for index in range(len(ROLE_NAMES)):
        if mask & (1 << index) and index in permitted:
            return index
    return None


def other_entity(roles: Sequence[str]) -> str:
    subject, _predicate, obj, other_subject, _other_predicate, other_obj = roles
    if other_subject and other_subject == subject and other_obj:
        return other_obj
    if other_obj and other_obj == obj and other_subject:
        return other_subject
    if other_subject:
        return other_subject
    if other_obj:
        return other_obj
    return EMPTY


def represented_fact_triples(record: ParsedRecord) -> list[tuple[str, str, str]]:
    """Return the exact pre-edit triples supplied to graph/state construction."""

    subject, predicate, obj, other_subject, other_predicate, other_obj = record.roles
    triples: list[tuple[str, str, str]] = []
    if subject and predicate and obj:
        triples.append((subject, predicate, obj))
    if other_subject and other_predicate and other_obj:
        triples.append((other_subject, other_predicate, other_obj))
    owners = {subject, obj, other_entity(record.roles)} - {EMPTY}
    for owner in owners:
        for pred, value in record.facts_by_owner.get(owner, ()):
            if owner and pred and value:
                triples.append((owner, pred, value))
    return triples


def local_predicates(record: ParsedRecord) -> set[str]:
    return {predicate for _subject, predicate, _object in represented_fact_triples(record)}


def definition_terms(
    constraint_ids: Iterable[str],
    registry: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Return factor IDs, parameter pairs, and malformed-definition reasons."""

    factors: list[str] = []
    pairs: list[tuple[str, str]] = []
    reasons: list[str] = []
    for constraint_id in constraint_ids:
        factors.append(constraint_id)
        entry = registry.get(constraint_id)
        if entry is None:
            reasons.append(f"missing registry definition: {constraint_id}")
            continue
        predicates = list(entry.get("param_predicates") or ())
        objects = list(entry.get("param_objects") or ())
        if len(predicates) != len(objects):
            reasons.append(f"parameter length mismatch: {constraint_id}")
        scope = f"constraint:{constraint_id}"
        for pred, obj in zip(predicates, objects):
            pairs.append(
                (
                    canonical_term(pred, blank_scope=scope),
                    canonical_term(obj, blank_scope=scope),
                )
            )
    return factors, pairs, reasons


def copy_domain(record: ParsedRecord, parameter_pairs: Iterable[tuple[str, str]]) -> set[str]:
    terms: set[str] = set()
    for subject, predicate, obj in represented_fact_triples(record):
        terms.update((subject, predicate, obj))
    for predicate, obj in parameter_pairs:
        if predicate:
            terms.add(predicate)
        if obj:
            terms.add(obj)
    return terms


def feature_known(term: str, canonical_to_encoder_id: Mapping[str, int]) -> bool:
    return bool(term and canonical_to_encoder_id.get(term, 0))


def input_topology(
    record: ParsedRecord,
    *,
    factor_ids: Sequence[str],
    parameter_pairs: Sequence[tuple[str, str]],
    canonical_to_encoder_id: Mapping[str, int],
    wiring_edges: int = 0,
) -> dict[str, int]:
    """Measure nodes/edges without allocating a PyG graph tensor.

    Predicate occurrences and parameter predicates are occurrence nodes, while
    subjects, objects, and parameter objects are identity-deduplicated.  Factor
    nodes are synthetic and are excluded from ``distinct_semantic_terms``.
    """

    triples = represented_fact_triples(record)
    endpoint_terms: set[str] = set()
    predicate_occurrences: list[str] = []
    for subject, predicate, obj in triples:
        endpoint_terms.update((subject, obj))
        predicate_occurrences.append(predicate)
    parameter_predicates: list[str] = []
    for predicate, obj in parameter_pairs:
        endpoint_terms.add(obj)
        parameter_predicates.append(predicate)

    semantic_terms = endpoint_terms | set(predicate_occurrences) | set(parameter_predicates)
    unknown_endpoint_nodes = sum(not feature_known(term, canonical_to_encoder_id) for term in endpoint_terms)
    unknown_predicate_nodes = sum(
        not feature_known(term, canonical_to_encoder_id)
        for term in (*predicate_occurrences, *parameter_predicates)
    )
    nodes = len(endpoint_terms) + len(predicate_occurrences) + len(parameter_predicates) + len(factor_ids)
    edges = 2 * (len(triples) + len(parameter_pairs)) + int(wiring_edges)
    return {
        "pre_triples": len(triples),
        "distinct_semantic_terms": len(semantic_terms),
        "nodes": nodes,
        "edges": edges,
        "unknown_feature_nodes": unknown_endpoint_nodes + unknown_predicate_nodes,
        "factor_nodes": len(factor_ids),
        "parameter_pairs": len(parameter_pairs),
    }


def factorized_wiring_edge_count(
    record: ParsedRecord,
    *,
    constraint_ids: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
    primary_constraint_id: str,
) -> int:
    """Count validator-v3 factor wiring edges without creating tensors."""

    triples = represented_fact_triples(record)
    focus_subject = record.roles[0]
    focus_object = record.roles[2]
    p31 = canonical_term("P31")
    p279 = canonical_term("P279")
    p2306 = canonical_term("P2306")
    p2309 = canonical_term("P2309")
    total_edges = 0

    for constraint_id in constraint_ids:
        entry = registry.get(constraint_id)
        if entry is None:
            continue
        family = str(entry.get("constraint_family") or "")
        constrained = canonical_term(entry.get("constrained_property"))
        all_constrained = [
            (index, subject, predicate, obj)
            for index, (subject, predicate, obj) in enumerate(triples)
            if predicate == constrained
        ]
        constrained_triples = all_constrained
        primary = constraint_id == primary_constraint_id
        if primary:
            if family == "distinct":
                focus_values = {
                    obj for _index, subject, _predicate, obj in all_constrained if subject == focus_subject
                }
                constrained_triples = [
                    triple
                    for triple in all_constrained
                    if triple[1] == focus_subject or triple[3] in focus_values
                ]
            else:
                constrained_triples = [
                    triple for triple in all_constrained if triple[1] == focus_subject
                ]

        raw_predicates = list(entry.get("param_predicates") or ())
        raw_objects = list(entry.get("param_objects") or ())
        pairs = [
            (
                canonical_term(predicate, blank_scope=f"constraint:{constraint_id}"),
                canonical_term(obj, blank_scope=f"constraint:{constraint_id}"),
            )
            for predicate, obj in zip(raw_predicates, raw_objects)
        ]
        observed: set[str] = set()
        if family in {
            "conflictWith",
            "inverse",
            "symmetric",
            "itemRequiresStatement",
            "valueRequiresStatement",
        }:
            observed.update(obj for predicate, obj in pairs if predicate == p2306)
            if family == "symmetric" and not observed and constrained:
                observed.add(constrained)
        elif family in {"type", "valueType"}:
            selectors = [obj for predicate, obj in pairs if predicate == p2309]
            if selectors == [canonical_term("Q21503252")]:
                observed.add(p31)
            elif selectors == [canonical_term("Q21514624")]:
                observed.add(p279)
            elif selectors == [canonical_term("Q30208840")]:
                observed.update((p31, p279))

        subject_scope: set[str] | None
        if primary and family in {"conflictWith", "itemRequiresStatement", "type"}:
            subject_scope = {focus_subject} if focus_subject else set()
        elif primary and family in {
            "valueRequiresStatement",
            "valueType",
            "inverse",
            "symmetric",
        }:
            subject_scope = {obj for _index, _subject, _predicate, obj in constrained_triples}
        elif not primary and family in {"conflictWith", "itemRequiresStatement", "type"}:
            subject_scope = {subject for _index, subject, _predicate, _obj in constrained_triples}
        elif not primary and family in {
            "valueRequiresStatement",
            "valueType",
            "inverse",
            "symmetric",
        }:
            subject_scope = {obj for _index, _subject, _predicate, obj in constrained_triples}
        else:
            subject_scope = None

        predicate_occurrences = {index for index, *_rest in constrained_triples}
        for index, (subject, predicate, _obj) in enumerate(triples):
            if predicate not in observed:
                continue
            if subject_scope is not None and subject not in subject_scope:
                continue
            predicate_occurrences.add(index)
        total_edges += 2 * len(predicate_occurrences)
        total_edges += 4 * len(constrained_triples)
    return total_edges


def grounding_reason(
    *,
    term: str,
    role_mask: int,
    permitted_role_indices: Iterable[int],
    constant_terms: set[str],
    input_terms: set[str],
    primary_parameter_terms: set[str],
    local_parameter_terms: set[str],
    canonical_to_encoder_id: Mapping[str, int],
) -> tuple[bool, bool, str]:
    """Return ``(fixed, fixed_or_copy, diagnostic reason)`` for one slot."""

    if not term:
        return True, True, "none"
    if choose_role_reference(role_mask, permitted_role_indices) is not None:
        return True, True, "role"
    if term in constant_terms:
        return True, True, "output_constant"
    if term in input_terms:
        if term in primary_parameter_terms or term in local_parameter_terms:
            return False, True, "local_constraint_parameter"
        return False, True, "missing_role_present_local_neighbor"
    if canonical_to_encoder_id.get(term, 0):
        return False, False, "known_constant_blocked_by_output_mask"
    return False, False, "outside_fixed_vocabulary_and_input"


def roundtrip_slots(
    slots: Sequence[str],
    roles: Sequence[str],
    *,
    role_indices_by_slot: Sequence[set[int]],
    constants_by_slot: Sequence[set[str]],
    copies_by_slot: Sequence[set[str]] | None = None,
) -> tuple[bool, list[str]]:
    """Encode and resolve all slots, asserting exact semantic round-trip."""

    resolved: list[str] = []
    for index, term in enumerate(slots):
        if not term:
            resolved.append(EMPTY)
            continue
        alias = choose_role_reference(role_alias_mask(term, roles), role_indices_by_slot[index])
        if alias is not None:
            resolved.append(roles[alias])
            continue
        if term in constants_by_slot[index]:
            resolved.append(term)
            continue
        if copies_by_slot is not None and term in copies_by_slot[index]:
            resolved.append(term)
            continue
        resolved.append("<UNRESOLVED>")
    covered = list(slots) == resolved
    if covered and tuple(resolved) != tuple(slots):
        raise AssertionError("covered edit failed semantic round-trip")
    return covered, resolved


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_from_histogram(histogram: Mapping[int, int], q: float) -> float | None:
    total = sum(int(value) for value in histogram.values())
    if not total:
        return None
    rank = max(0, min(total - 1, int(round((total - 1) * q))))
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += int(count)
        if cumulative > rank:
            return float(value)
    raise AssertionError("histogram percentile did not reconcile")
