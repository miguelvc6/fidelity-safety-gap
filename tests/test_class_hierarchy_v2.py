from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from modules.class_hierarchy import (
    ClassHierarchy,
    HIERARCHY_CUTOFF,
    HIERARCHY_CACHE_SCHEMA_VERSION,
    HIERARCHY_PARSER_VERSION,
    HIERARCHY_SCHEMA_VERSION,
    canonical_sha256,
)
from modules.constraint_checkers import ValidationOutcome
from modules.constraint_checkers import EvidenceState


def test_equality_transitive_cycles_and_incomplete_ancestry() -> None:
    hierarchy = ClassHierarchy(
        parents={1: {2}, 2: {3}, 3: {2}, 4: {5}},
        complete={1: True, 2: True, 3: True, 4: False, 5: False},
    )
    assert hierarchy.reachable(1, {1}) == ValidationOutcome.SATISFIED
    assert hierarchy.reachable(1, {3}) == ValidationOutcome.SATISFIED
    assert hierarchy.reachable(1, {9}) == ValidationOutcome.VIOLATED
    assert hierarchy.reachable(4, {9}) == ValidationOutcome.UNKNOWN


def test_artifact_checksum_cutoff_and_unencoded_intermediate(tmp_path: Path) -> None:
    payload = {
        "schema_version": HIERARCHY_SCHEMA_VERSION,
        "parser_version": HIERARCHY_PARSER_VERSION,
        "cache_schema_version": HIERARCHY_CACHE_SCHEMA_VERSION,
        "cutoff": HIERARCHY_CUTOFF,
        "records": {
            "Q1": {"status": "ok", "parents": ["Q999"], "closure_complete": True, "direct_adjacency_complete": True},
            "Q999": {"status": "ok", "parents": ["Q2"], "closure_complete": True, "direct_adjacency_complete": True},
            "Q2": {"status": "ok", "parents": [], "closure_complete": True, "direct_adjacency_complete": True},
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    path = tmp_path / "hierarchy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    ids = {"Q1": 1, "Q2": 2}
    hierarchy = ClassHierarchy.from_artifact(path, resolve_id=lambda raw: ids.get(str(raw), 0))
    assert hierarchy.reachable(1, {2}) == ValidationOutcome.SATISFIED

    broken = dict(payload)
    broken["cutoff"] = "2019-01-01T00:00:00Z"
    broken["content_sha256"] = canonical_sha256({k: v for k, v in broken.items() if k != "content_sha256"})
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="cutoff"):
        ClassHierarchy.from_artifact(path, resolve_id=lambda raw: ids.get(str(raw), 0))


def test_artifact_loader_rejects_pre_v3_parser_contract(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "cutoff": HIERARCHY_CUTOFF,
        "records": {},
    }
    payload["content_sha256"] = canonical_sha256(payload)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        ClassHierarchy.from_artifact(path, resolve_id=lambda _raw: 0)


def _downloader_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_historical_class_hierarchy.py"
    spec = importlib.util.spec_from_file_location("hierarchy_builder_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        entity = {
            "claims": {
                "P279": [
                    {"rank": "deprecated", "mainsnak": {"datavalue": {"value": {"id": "Q8"}}}},
                    {"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": "Q7"}}}},
                    {"rank": "preferred", "mainsnak": {"datavalue": {"value": {"id": "Q6"}}}},
                ]
            }
        }
        return {
            "query": {
                "pages": [
                    {
                        "revisions": [
                            {
                                "revid": 12,
                                "parentid": 11,
                                "timestamp": "2018-06-30T00:00:00Z",
                                "slots": {"main": {"content": json.dumps(entity)}},
                            }
                        ]
                    }
                ]
            }
        }


class _Session:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return _Response()


def test_historical_client_uses_cutoff_and_truthy_rank() -> None:
    module = _downloader_module()
    client = module.HistoricalRevisionClient(timeout=1.0, retries=0)
    client.session = _Session()
    result = client.fetch("Q5", HIERARCHY_CUTOFF)
    assert result["parents"] == ["Q6"]
    params = client.session.calls[0][1]
    assert params["rvstart"] == HIERARCHY_CUTOFF
    assert params["rvdir"] == "older"
    assert params["maxlag"] == "10"
    assert result["revision_id"] == 12


def test_historical_client_honors_maxlag_retry(monkeypatch) -> None:
    module = _downloader_module()
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))

    class MaxlagResponse(_Response):
        headers = {"Retry-After": "17"}

        def json(self):
            return {"error": {"code": "maxlag", "info": "replica lagged"}}

    class RecoveringSession(_Session):
        def get(self, url, params, timeout):
            self.calls.append((url, params, timeout))
            return MaxlagResponse() if len(self.calls) == 1 else _Response()

    client = module.HistoricalRevisionClient(timeout=1.0, retries=1)
    client.session = RecoveringSession()
    result = client.fetch("Q5", HIERARCHY_CUTOFF)

    assert result["revision_id"] == 12
    assert sleeps == [60.0]


def test_historical_client_accepts_legacy_empty_claims_list() -> None:
    module = _downloader_module()
    page = {
        "revisions": [
            {
                "revid": 13,
                "parentid": 12,
                "timestamp": "2018-06-30T00:00:00Z",
                "slots": {"main": {"content": json.dumps({"type": "item", "claims": []})}},
            }
        ]
    }
    result = module.HistoricalRevisionClient._record_from_page("Q5", page)
    assert result["status"] == "ok"
    assert result["parents"] == []
    assert result["direct_adjacency_complete"] is True


@pytest.mark.parametrize(
    ("value", "parents", "complete"),
    [
        ({"entity-type": "item", "numeric-id": 5}, ["Q5"], True),
        ({"entity-type": "item", "id": "Q5", "numeric-id": 5}, ["Q5"], True),
        ({"entity-type": "item", "id": "Q5", "numeric-id": 6}, [], False),
        ({"entity-type": "property", "numeric-id": 5}, [], False),
        ({"entity-type": "item", "id": "bad", "numeric-id": 5}, [], False),
        ({"entity-type": "item", "id": "Q5", "numeric-id": "5"}, [], False),
        ({"entity-type": "item", "id": "Q0"}, [], False),
    ],
)
def test_historical_parent_entity_id_formats(value, parents, complete) -> None:
    module = _downloader_module()
    entity = {
        "claims": {
            "P279": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"type": "wikibase-entityid", "value": value},
                    },
                }
            ]
        }
    }
    page = {
        "revisions": [
            {
                "revid": 15,
                "timestamp": "2018-06-30T00:00:00Z",
                "slots": {"main": {"content": json.dumps(entity)}},
            }
        ]
    }
    result = module.HistoricalRevisionClient._record_from_page("Q7", page)
    assert result["parents"] == parents
    assert result["direct_adjacency_complete"] is complete


def test_valid_parent_is_preserved_alongside_unknown_parent() -> None:
    module = _downloader_module()
    entity = {
        "claims": {
            "P279": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q5"}},
                    },
                },
                {"rank": "normal", "mainsnak": {"snaktype": "somevalue"}},
            ]
        }
    }
    page = {
        "revisions": [
            {
                "revid": 16,
                "timestamp": "2018-06-30T00:00:00Z",
                "slots": {"main": {"content": json.dumps(entity)}},
            }
        ]
    }
    result = module.HistoricalRevisionClient._record_from_page("Q7", page)
    assert result["parents"] == ["Q5"]
    assert result["direct_adjacency_complete"] is False
    assert module._closure_complete("Q7", {"Q7": result, "Q5": {"status": "ok", "parents": [], "direct_adjacency_complete": True}}) is False


def test_valid_parent_is_preserved_alongside_malformed_rank() -> None:
    module = _downloader_module()
    entity = {
        "claims": {
            "P279": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q5"}},
                    },
                },
                {
                    "rank": "invalid-rank",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q6"}},
                    },
                },
            ]
        }
    }
    page = {
        "revisions": [
            {
                "revid": 18,
                "timestamp": "2018-06-30T00:00:00Z",
                "slots": {"main": {"content": json.dumps(entity)}},
            }
        ]
    }
    result = module.HistoricalRevisionClient._record_from_page("Q7", page)
    assert result["parents"] == ["Q5"]
    assert result["direct_adjacency_complete"] is False


def test_novalue_is_complete_absence_under_truthy_projection() -> None:
    module = _downloader_module()
    entity = {
        "claims": {
            "P279": [
                {"rank": "normal", "mainsnak": {"snaktype": "novalue"}},
            ]
        }
    }
    page = {
        "revisions": [
            {
                "revid": 17,
                "timestamp": "2018-06-30T00:00:00Z",
                "slots": {"main": {"content": json.dumps(entity)}},
            }
        ]
    }
    result = module.HistoricalRevisionClient._record_from_page("Q7", page)
    assert result["parents"] == []
    assert result["direct_adjacency_complete"] is True
    assert result["snak_counts"]["novalue"] == 1


def test_exhaustive_effective_hierarchy_matches_independent_bfs() -> None:
    """Finite reference check independent of ClassHierarchy.reachable."""

    possible_edges = ((1, 2), (1, 3), (2, 4), (3, 4), (4, 1), (4, 5))
    hierarchy = ClassHierarchy(
        parents={node: set() for node in range(1, 6)},
        complete={node: True for node in range(1, 6)},
    )

    def reference(edges: set[tuple[int, int]], child: int, target: int) -> bool:
        stack = [child]
        visited: set[int] = set()
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(parent for source, parent in edges if source == node)
        return False

    for flags in itertools.product((False, True), repeat=len(possible_edges)):
        edges = {edge for edge, enabled in zip(possible_edges, flags) if enabled}
        facts = {
            node: {279: {parent for source, parent in edges if source == node}}
            for node in range(1, 6)
        }
        state = EvidenceState(
            facts_by_entity=facts,
            predicates_present={node: {279} for node in facts},
            assume_complete=True,
            missing_edits=set(),
            focus_subject=1,
            focus_predicate=279,
            focus_object=2,
            other_subject=0,
            other_predicate=0,
            other_object=0,
        )
        expected = (
            ValidationOutcome.SATISFIED
            if reference(edges, 1, 5)
            else ValidationOutcome.VIOLATED
        )
        assert hierarchy.reachable(1, {5}, state=state, p279_predicate=279) == expected


def test_candidate_reachability_does_not_scan_or_mutate_whole_hierarchy() -> None:
    class NoFullScanDict(dict):
        def items(self):  # pragma: no cover - failure guard
            raise AssertionError("candidate reachability scanned the whole hierarchy")

    hierarchy = ClassHierarchy(
        parents={node: ({node + 1} if node < 1000 else set()) for node in range(1, 1001)},
        complete={node: True for node in range(1, 1001)},
    )
    snapshot = {node: set(parents) for node, parents in hierarchy.parents.items()}
    hierarchy.parents = NoFullScanDict(hierarchy.parents)
    for _ in range(100):
        assert hierarchy.reachable(1, {5}) == ValidationOutcome.SATISFIED
    assert dict(hierarchy.parents) == snapshot


def test_historical_client_records_unrepresentable_revision_as_incomplete() -> None:
    module = _downloader_module()
    page = {
        "revisions": [
            {
                "revid": 14,
                "parentid": 13,
                "timestamp": "2018-06-30T00:00:00Z",
                "slots": {"main": {"content": "#REDIRECT [[Q6]]"}},
            }
        ]
    }
    result = module.HistoricalRevisionClient._record_from_page("Q5", page)
    assert result["status"] == "historical_content_unrepresentable"
    assert result["parents"] == []
    assert result["revision_id"] == 14
    assert result["revision_content_sha256"]


def test_resumable_cache_avoids_network(tmp_path: Path) -> None:
    module = _downloader_module()
    client = module.HistoricalRevisionClient(timeout=1.0, retries=0)
    client.session = _Session()
    first = module._load_or_fetch(client, tmp_path, "Q5", HIERARCHY_CUTOFF)
    assert len(client.session.calls) == 1
    second = module._load_or_fetch(client, tmp_path, "Q5", HIERARCHY_CUTOFF)
    assert first == second
    assert len(client.session.calls) == 1


def test_stale_success_cache_without_content_hash_is_refetched(tmp_path: Path) -> None:
    module = _downloader_module()
    stale = {
        "cutoff": HIERARCHY_CUTOFF,
        "record": {
            "entity_id": "Q5",
            "status": "ok",
            "revision_id": 1,
            "revision_timestamp": "2018-01-01T00:00:00Z",
            "parents": [],
        },
    }
    (tmp_path / "Q5.json").write_text(json.dumps(stale), encoding="utf-8")
    client = module.HistoricalRevisionClient(timeout=1.0, retries=0)
    client.session = _Session()

    refreshed = module._load_or_fetch(client, tmp_path, "Q5", HIERARCHY_CUTOFF)

    assert len(client.session.calls) == 1
    assert refreshed["revision_id"] == 12
    assert refreshed["revision_content_sha256"]
    assert (tmp_path / "Q5.pre-parser-v2.json").exists()
    cache_payload = json.loads((tmp_path / "Q5.json").read_text(encoding="utf-8"))
    assert cache_payload["parser_version"] == module.HIERARCHY_PARSER_VERSION
    assert cache_payload["cache_schema_version"] == module.HIERARCHY_CACHE_SCHEMA_VERSION
    second = module._load_or_fetch(client, tmp_path, "Q5", HIERARCHY_CUTOFF)
    assert refreshed == second
    assert len(client.session.calls) == 1


def test_stale_cache_with_raw_page_is_reparsed_without_network(tmp_path: Path) -> None:
    module = _downloader_module()
    response_page = _Response().json()["query"]["pages"][0]
    stale = {
        "cutoff": HIERARCHY_CUTOFF,
        "parser_version": 1,
        "source_page": response_page,
        "record": {"entity_id": "Q5", "status": "ok", "parents": []},
    }
    (tmp_path / "Q5.json").write_text(json.dumps(stale), encoding="utf-8")
    client = module.HistoricalRevisionClient(timeout=1.0, retries=0)
    client.session = _Session()

    reparsed = module._load_or_fetch(client, tmp_path, "Q5", HIERARCHY_CUTOFF)

    assert reparsed["parents"] == ["Q6"]
    assert client.session.calls == []
    assert (tmp_path / "Q5.pre-parser-v2.json").exists()


def test_batched_historical_retrieval_and_per_entity_resume(tmp_path: Path) -> None:
    module = _downloader_module()
    client = module.HistoricalRevisionClient(timeout=1.0, retries=0)
    session = _Session()
    original_get = session.get

    def get(url, params, timeout):
        if params["titles"] == "Q9":
            session.calls.append((url, params, timeout))
            response = _Response()
            response.json = lambda: {"query": {"pages": [{"title": "Q9", "missing": True}]}}
            return response
        return original_get(url, params, timeout)

    session.get = get
    client.session = session
    records = module._load_or_fetch_many(client, tmp_path, ["Q5", "Q9"], HIERARCHY_CUTOFF)
    assert records["Q5"]["revision_id"] == 12
    assert records["Q9"]["status"] == "historically_unavailable"
    assert (tmp_path / "Q5.json").exists()
    assert (tmp_path / "Q9.json").exists()
    assert [call[1]["titles"] for call in session.calls] == ["Q5", "Q9"]
    assert all("|" not in call[1]["titles"] for call in session.calls)


def test_legacy_revision_provenance_enables_verified_batch_reparse(tmp_path: Path) -> None:
    module = _downloader_module()
    legacy_cache = tmp_path / "legacy"
    current_cache = tmp_path / "current"
    legacy_cache.mkdir()

    def page(entity_id: str, revision_id: int, parent: str) -> dict:
        entity = {
            "claims": {
                "P279": [
                    {
                        "rank": "normal",
                        "mainsnak": {
                            "snaktype": "value",
                            "datavalue": {
                                "type": "wikibase-entityid",
                                "value": {"entity-type": "item", "id": parent},
                            },
                        },
                    }
                ]
            }
        }
        return {
            "title": entity_id,
            "revisions": [
                {
                    "revid": revision_id,
                    "parentid": revision_id - 1,
                    "timestamp": "2018-06-30T00:00:00Z",
                    "slots": {"main": {"content": json.dumps(entity)}},
                }
            ],
        }

    pages = {
        "Q5": page("Q5", 15, "Q7"),
        "Q9": page("Q9", 19, "Q11"),
    }
    for entity_id, source_page in pages.items():
        record = module.HistoricalRevisionClient._record_from_page(entity_id, source_page)
        (legacy_cache / f"{entity_id}.json").write_text(
            json.dumps({"cutoff": HIERARCHY_CUTOFF, "record": record}),
            encoding="utf-8",
        )

    class RevisionResponse:
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"query": {"pages": list(pages.values())}}

    class RevisionSession:
        def __init__(self):
            self.headers = {}
            self.calls = []

        def get(self, url, params, timeout):
            self.calls.append((url, params, timeout))
            return RevisionResponse()

    client = module.HistoricalRevisionClient(timeout=1.0, retries=0)
    client.session = RevisionSession()
    records = module._load_or_fetch_many(
        client,
        current_cache,
        ["Q5", "Q9"],
        HIERARCHY_CUTOFF,
        legacy_cache,
    )

    assert records["Q5"]["parents"] == ["Q7"]
    assert records["Q9"]["parents"] == ["Q11"]
    assert len(client.session.calls) == 1
    params = client.session.calls[0][1]
    assert "revids" in params
    assert "rvstart" not in params
    assert set(params["revids"].split("|")) == {"15", "19"}
    for entity_id in pages:
        cached = json.loads((current_cache / f"{entity_id}.json").read_text())
        assert cached["parser_version"] == module.HIERARCHY_PARSER_VERSION
        assert cached["source_page"] == pages[entity_id]
