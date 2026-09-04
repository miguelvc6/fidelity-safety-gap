"""Fixed-cutoff Wikidata P279 hierarchy evidence for validator semantics v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Set

from modules.constraint_checkers import ValidationOutcome


HIERARCHY_SCHEMA_VERSION = 1
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


class ClassHierarchy:
    """Encoded hierarchy with conservative reachability under missing ancestry."""

    def __init__(
        self,
        *,
        parents: Mapping[int, Set[int]],
        complete: Mapping[int, bool],
        identity: HierarchyIdentity | None = None,
        raw_parents: Mapping[str, Set[str]] | None = None,
        raw_complete: Mapping[str, bool] | None = None,
        raw_by_encoded: Mapping[int, str] | None = None,
    ) -> None:
        self.parents = {int(child): {int(parent) for parent in values} for child, values in parents.items()}
        self.complete = {int(node): bool(value) for node, value in complete.items()}
        self.identity = identity
        self.raw_parents = {str(child): {str(parent) for parent in values} for child, values in (raw_parents or {}).items()}
        self.raw_complete = {str(node): bool(value) for node, value in (raw_complete or {}).items()}
        self.raw_by_encoded = {int(node): str(raw) for node, raw in (raw_by_encoded or {}).items()}

    def reachable(self, child: int, ancestors: Set[int]) -> ValidationOutcome:
        targets = {int(value) for value in ancestors}
        if child in targets:
            return ValidationOutcome.SATISFIED
        if self.raw_parents:
            raw_child = self.raw_by_encoded.get(int(child))
            raw_targets = {self.raw_by_encoded[value] for value in targets if value in self.raw_by_encoded}
            if raw_child is None or len(raw_targets) != len(targets):
                return ValidationOutcome.UNKNOWN
            stack_raw = [raw_child]
            visited_raw: set[str] = set()
            incomplete_raw = False
            while stack_raw:
                node = stack_raw.pop()
                if node in visited_raw:
                    continue
                visited_raw.add(node)
                if not self.raw_complete.get(node, False):
                    incomplete_raw = True
                for parent in self.raw_parents.get(node, set()):
                    if parent in raw_targets:
                        return ValidationOutcome.SATISFIED
                    if parent not in visited_raw:
                        stack_raw.append(parent)
            return ValidationOutcome.UNKNOWN if incomplete_raw else ValidationOutcome.VIOLATED
        stack = [int(child)]
        visited: set[int] = set()
        incomplete = False
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if not self.complete.get(node, False):
                incomplete = True
            for parent in self.parents.get(node, set()):
                if parent in targets:
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
        raw_parents: dict[str, set[str]] = {}
        raw_complete: dict[str, bool] = {}
        raw_by_encoded: dict[int, str] = {}
        for raw_node, record in records.items():
            node = resolve_id(raw_node)
            status = record.get("status")
            node_complete = status == "ok" and bool(record.get("closure_complete", False))
            raw_complete[raw_node] = node_complete
            raw_parents[raw_node] = set(record.get("parents", []))
            if not node:
                continue
            raw_by_encoded[node] = raw_node
            complete[node] = node_complete
            encoded_parents = {resolve_id(raw) for raw in record.get("parents", [])}
            parents[node] = {value for value in encoded_parents if value}
        return cls(
            parents=parents,
            complete=complete,
            identity=identity,
            raw_parents=raw_parents,
            raw_complete=raw_complete,
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
    )
