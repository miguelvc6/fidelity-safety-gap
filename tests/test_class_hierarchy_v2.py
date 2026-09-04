from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from modules.class_hierarchy import ClassHierarchy, HIERARCHY_CUTOFF, canonical_sha256
from modules.constraint_checkers import ValidationOutcome


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
        "schema_version": 1,
        "cutoff": HIERARCHY_CUTOFF,
        "records": {
            "Q1": {"status": "ok", "parents": ["Q999"], "closure_complete": True},
            "Q999": {"status": "ok", "parents": ["Q2"], "closure_complete": True},
            "Q2": {"status": "ok", "parents": [], "closure_complete": True},
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
    second = module._load_or_fetch(client, tmp_path, "Q5", HIERARCHY_CUTOFF)
    assert refreshed == second
    assert len(client.session.calls) == 1


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
