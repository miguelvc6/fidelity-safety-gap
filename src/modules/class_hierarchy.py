"""Fixed-cutoff Wikidata P279 evidence with semantics-v3 state overlays."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Hashable, Mapping, Set

from modules.constraint_checkers import ValidationOutcome

if TYPE_CHECKING:
    from modules.constraint_checkers import EvidenceState


HIERARCHY_SCHEMA_VERSION = 2
HIERARCHY_PARSER_VERSION = 2
HIERARCHY_CACHE_SCHEMA_VERSION = 2
HIERARCHY_CUTOFF = "2018-07-01T00:00:00Z"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HierarchyIdentity:
    path: str
    sha256: str
    content_sha256: str
    cutoff: str
    schema_version: int
    parser_version: int
    cache_schema_version: int


class ClassHierarchy:
    """Encoded hierarchy with conservative reachability under missing ancestry."""

    def __init__(
        self,
        *,
        parents: Mapping[int, Set[int]],
        complete: Mapping[int, bool],
        direct_complete: Mapping[int, bool] | None = None,
        identity: HierarchyIdentity | None = None,
        raw_parents: Mapping[str, Set[str]] | None = None,
        raw_complete: Mapping[str, bool] | None = None,
        raw_direct_complete: Mapping[str, bool] | None = None,
        raw_by_encoded: Mapping[int, str] | None = None,
    ) -> None:
        self.parents = {int(child): {int(parent) for parent in values} for child, values in parents.items()}
        self.complete = {int(node): bool(value) for node, value in complete.items()}
        self.direct_complete = {
            int(node): bool(value)
            for node, value in (direct_complete if direct_complete is not None else complete).items()
        }
        self.identity = identity
        self.raw_parents = {str(child): {str(parent) for parent in values} for child, values in (raw_parents or {}).items()}
        self.raw_complete = {str(node): bool(value) for node, value in (raw_complete or {}).items()}
        self.raw_direct_complete = {
            str(node): bool(value)
            for node, value in (
                raw_direct_complete if raw_direct_complete is not None else (raw_complete or {})
            ).items()
        }
        self.raw_by_encoded = {int(node): str(raw) for node, raw in (raw_by_encoded or {}).items()}
        self.encoded_by_raw = {raw: node for node, raw in self.raw_by_encoded.items()}

    def reachable(
        self,
        child: int,
        ancestors: Set[int],
        *,
        state: "EvidenceState | None" = None,
        p279_predicate: int = 0,
    ) -> ValidationOutcome:
        """Traverse background parents plus a state-local overlay.

        Local P279 statements add direct edges.  Successful state deletions are
        tombstones and therefore take precedence over the immutable background;
        all other background edges remain visible.  Completeness is assessed
        per direct adjacency, so a known path is sufficient for satisfaction
        while a negative requires every visited adjacency to be complete.
        """

        targets = {int(value) for value in ancestors}
        if child in targets:
            return ValidationOutcome.SATISFIED
        deleted = set(getattr(state, "applied_deletions", frozenset())) if state is not None else set()

        def _key_for_raw(raw: str) -> Hashable:
            return self.encoded_by_raw.get(raw, raw)

        def _background_parents(node: Hashable) -> set[Hashable]:
            if isinstance(node, int) and self.raw_parents:
                raw = self.raw_by_encoded.get(node)
                if raw is None:
                    return set()
                candidates = {_key_for_raw(parent) for parent in self.raw_parents.get(raw, set())}
            elif isinstance(node, str):
                candidates = {_key_for_raw(parent) for parent in self.raw_parents.get(node, set())}
            elif isinstance(node, int):
                candidates = set(self.parents.get(node, set()))
            else:
                candidates = set()
            if not isinstance(node, int) or not p279_predicate:
                return candidates
            return {
                parent
                for parent in candidates
                if not isinstance(parent, int) or (node, p279_predicate, parent) not in deleted
            }

        def _local_parents(node: Hashable) -> set[Hashable]:
            if state is None or not p279_predicate or not isinstance(node, int):
                return set()
            return set(state.values_for(node, p279_predicate))

        def _adjacency_complete(node: Hashable) -> bool:
            if isinstance(node, str):
                return self.raw_direct_complete.get(node, False)
            background_complete = self.direct_complete.get(int(node), False)
            local_complete = bool(
                state is not None
                and p279_predicate
                and state.property_complete(int(node), p279_predicate)
            )
            return background_complete or local_complete

        stack: list[Hashable] = [int(child)]
        visited: set[Hashable] = set()
        incomplete = False
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if not _adjacency_complete(node):
                incomplete = True
            for parent in _background_parents(node) | _local_parents(node):
                if isinstance(parent, int) and parent in targets:
                    return ValidationOutcome.SATISFIED
                if parent not in visited:
                    stack.append(parent)
        return ValidationOutcome.UNKNOWN if incomplete else ValidationOutcome.VIOLATED

    @classmethod
    def from_artifact(
        cls,
        path: Path,
        *,
        resolve_id: Callable[[str | None], int],
        expected_cutoff: str = HIERARCHY_CUTOFF,
    ) -> "ClassHierarchy":
        payload, identity = load_hierarchy_artifact(path, expected_cutoff=expected_cutoff)

        records = payload.get("records") or {}
        parents: dict[int, set[int]] = {}
        complete: dict[int, bool] = {}
        direct_complete: dict[int, bool] = {}
        raw_parents: dict[str, set[str]] = {}
        raw_complete: dict[str, bool] = {}
        raw_direct_complete: dict[str, bool] = {}
        raw_by_encoded: dict[int, str] = {}
        for raw_node, record in records.items():
            node = resolve_id(raw_node)
            status = record.get("status")
            node_complete = status == "ok" and bool(record.get("closure_complete", False))
            node_direct_complete = status == "ok" and bool(record.get("direct_adjacency_complete", False))
            raw_complete[raw_node] = node_complete
            raw_direct_complete[raw_node] = node_direct_complete
            raw_parents[raw_node] = set(record.get("parents", []))
            if not node:
                continue
            raw_by_encoded[node] = raw_node
            complete[node] = node_complete
            direct_complete[node] = node_direct_complete
            encoded_parents = {resolve_id(raw) for raw in record.get("parents", [])}
            parents[node] = {value for value in encoded_parents if value}
        return cls(
            parents=parents,
            complete=complete,
            direct_complete=direct_complete,
            identity=identity,
            raw_parents=raw_parents,
            raw_complete=raw_complete,
            raw_direct_complete=raw_direct_complete,
            raw_by_encoded=raw_by_encoded,
        )


def load_hierarchy_artifact(
    path: Path,
    *,
    expected_cutoff: str = HIERARCHY_CUTOFF,
) -> tuple[dict[str, object], HierarchyIdentity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != HIERARCHY_SCHEMA_VERSION:
        raise ValueError("Unsupported hierarchy schema")
    if payload.get("parser_version") != HIERARCHY_PARSER_VERSION:
        raise ValueError("Unsupported hierarchy parser version")
    if payload.get("cache_schema_version") != HIERARCHY_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported hierarchy cache provenance")
    if payload.get("cutoff") != expected_cutoff:
        raise ValueError("Hierarchy cutoff does not match validator semantics")
    recorded = payload.get("content_sha256")
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    actual = canonical_sha256(content)
    if recorded != actual:
        raise ValueError("Hierarchy content checksum mismatch")
    return payload, HierarchyIdentity(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        content_sha256=actual,
        cutoff=str(payload["cutoff"]),
        schema_version=int(payload["schema_version"]),
        parser_version=int(payload["parser_version"]),
        cache_schema_version=int(payload["cache_schema_version"]),
    )
