#!/usr/bin/env python3
"""Build the resumable 2018-07-01 P279 hierarchy used by validator v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import requests

from modules.class_hierarchy import (
    HIERARCHY_CACHE_SCHEMA_VERSION,
    HIERARCHY_CUTOFF,
    HIERARCHY_PARSER_VERSION,
    HIERARCHY_SCHEMA_VERSION,
    canonical_sha256,
)
from modules.constraint_checkers import normalize_item_id, normalize_property_id
from modules.data_encoders import GlobalIntEncoder


API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = (
    "fidelity-safety-gap-historical-hierarchy/3.0 "
    "(https://github.com/miguelvc6/fidelity-safety-gap)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entity_id(encoder: GlobalIntEncoder, value: object) -> str | None:
    try:
        decoded = encoder.decode(int(value))
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    return normalize_item_id(str(decoded))


def benchmark_seed_ids(registry_path: Path, benchmark_dir: Path, encoder_path: Path) -> set[str]:
    """Collect the benchmark's fixed 3,941 hierarchy-query seed IDs.

    The seed population is defined by the immutable benchmark inputs: P2308
    targets, P31/P279 values in the focus subject and focus object
    descriptions, and the child of a focus P279 statement.  Gold repair output
    fields and the auxiliary-neighbour description are deliberately excluded:
    they are possible edits/context, not members of the benchmark's fixed
    hierarchy-query population.  The recursive download still retains every
    parent needed to close these seeds at the cutoff.
    """
    encoder = GlobalIntEncoder()
    encoder.load(encoder_path)
    registry_frame = pd.read_parquet(registry_path)
    registry_raw = registry_frame["registry_json"].iloc[0]
    registry = json.loads(registry_raw) if isinstance(registry_raw, str) else registry_raw
    seeds = {
        item
        for entry in registry.values()
        for predicate, obj in zip(entry.get("param_predicates") or [], entry.get("param_objects") or [])
        if normalize_property_id(predicate) == "P2308"
        if (item := normalize_item_id(obj)) is not None
    }
    p31 = encoder.encode("<http://www.wikidata.org/entity/P31>", add_new=False)
    p279 = encoder.encode("<http://www.wikidata.org/entity/P279>", add_new=False)
    for path in sorted(benchmark_dir.glob("df_*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "subject", "predicate",
                "subject_predicates", "subject_objects",
                "object_predicates", "object_objects",
            ],
        )
        for row in frame.itertuples(index=False):
            if int(row.predicate or 0) == p279:
                item = _entity_id(encoder, row.subject)
                if item:
                    seeds.add(item)
            for predicate_column, object_column in (
                (row.subject_predicates, row.subject_objects),
                (row.object_predicates, row.object_objects),
            ):
                for predicate, obj in zip(predicate_column, object_column):
                    if int(predicate) in {p31, p279}:
                        item = _entity_id(encoder, obj)
                        if item:
                            seeds.add(item)
    return seeds


class HistoricalRevisionClient:
    def __init__(
        self,
        *,
        timeout: float,
        retries: int,
        request_delay: float = 0.0,
        maxlag: int = 10,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.request_delay = request_delay
        self.maxlag = maxlag
        self._last_request_started: float | None = None
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_pages: dict[str, dict[str, Any]] = {}

    def _pace_request(self) -> None:
        if self._last_request_started is not None:
            remaining = self.request_delay - (time.monotonic() - self._last_request_started)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started = time.monotonic()

    @staticmethod
    def _record_from_page(entity_id: str, page: dict[str, Any]) -> dict[str, Any]:
        revisions = page.get("revisions") or []
        if page.get("missing") is not None or not revisions:
            return {"entity_id": entity_id, "status": "historically_unavailable", "parents": []}
        revision = revisions[0]
        provenance = {
            "entity_id": entity_id,
            "revision_id": int(revision["revid"]),
            "parent_revision_id": int(revision.get("parentid", 0)),
            "revision_timestamp": revision["timestamp"],
        }
        content = revision.get("slots", {}).get("main", {}).get("content")
        if content is None:
            content = revision.get("content")
        if content is None:
            return {
                **provenance,
                "status": "historical_content_unavailable",
                "parents": [],
            }
        content_bytes = (
            content.encode("utf-8")
            if isinstance(content, str)
            else json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        provenance["revision_content_sha256"] = hashlib.sha256(content_bytes).hexdigest()
        try:
            entity = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError) as exc:
            return {
                **provenance,
                "status": "historical_content_unrepresentable",
                "status_reason": type(exc).__name__,
                "parents": [],
            }
        if not isinstance(entity, dict):
            return {
                **provenance,
                "status": "historical_content_unrepresentable",
                "status_reason": f"entity payload is {type(entity).__name__}",
                "parents": [],
            }
        claims_payload = entity.get("claims") or {}
        # Older Wikibase JSON serializes an empty claims map as [] rather than
        # {}. Accept that representation, but reject a non-empty non-map so a
        # malformed revision cannot be recorded as a parentless class.
        if isinstance(claims_payload, list) and not claims_payload:
            claims_payload = {}
        if not isinstance(claims_payload, dict):
            return {
                **provenance,
                "status": "historical_content_unrepresentable",
                "status_reason": f"claims payload is {type(claims_payload).__name__}",
                "parents": [],
            }
        claims = claims_payload.get("P279", [])
        if not isinstance(claims, list):
            return {
                **provenance,
                "status": "historical_content_unrepresentable",
                "status_reason": f"P279 payload is {type(claims).__name__}",
                "parents": [],
            }
        malformed_claim_shape = any(
            not isinstance(claim, dict)
            or claim.get("rank") not in {"deprecated", "normal", "preferred"}
            for claim in claims
        )
        shaped_claims = [claim for claim in claims if isinstance(claim, dict)]
        nondeprecated = [claim for claim in shaped_claims if claim.get("rank") != "deprecated"]
        preferred = [claim for claim in nondeprecated if claim.get("rank") == "preferred"]
        selected = preferred if preferred else [claim for claim in nondeprecated if claim.get("rank") == "normal"]
        if nondeprecated and not selected:
            malformed_claim_shape = True
        parents: set[str] = set()
        adjacency_complete = not malformed_claim_shape
        snak_counts = {"value": 0, "somevalue": 0, "novalue": 0, "malformed": 0}
        for claim in selected:
            mainsnak = claim.get("mainsnak")
            if not isinstance(mainsnak, dict):
                adjacency_complete = False
                snak_counts["malformed"] += 1
                continue
            snaktype = mainsnak.get("snaktype")
            # Old revisions sometimes omit snaktype on ordinary value snaks.
            if snaktype is None and "datavalue" in mainsnak:
                snaktype = "value"
            if snaktype == "novalue":
                snak_counts["novalue"] += 1
                continue
            if snaktype == "somevalue":
                adjacency_complete = False
                snak_counts["somevalue"] += 1
                continue
            if snaktype != "value":
                adjacency_complete = False
                snak_counts["malformed"] += 1
                continue
            snak_counts["value"] += 1
            datavalue = mainsnak.get("datavalue")
            value = datavalue.get("value") if isinstance(datavalue, dict) else None
            parent: str | None = None
            datavalue_type_valid = (
                isinstance(datavalue, dict)
                and datavalue.get("type", "wikibase-entityid") == "wikibase-entityid"
            )
            if (
                datavalue_type_valid
                and isinstance(value, dict)
                and value.get("entity-type", "item") == "item"
            ):
                has_explicit = "id" in value
                explicit = normalize_item_id(value.get("id")) if has_explicit else None
                has_numeric = "numeric-id" in value
                numeric = value.get("numeric-id")
                numeric_item = None
                if has_numeric and isinstance(numeric, int) and not isinstance(numeric, bool) and numeric > 0:
                    numeric_item = f"Q{numeric}"
                if has_explicit and explicit is None:
                    parent = None
                elif has_numeric and numeric_item is None:
                    parent = None
                elif explicit is not None and numeric_item is not None and explicit != numeric_item:
                    parent = None
                else:
                    parent = explicit or numeric_item
            if parent is not None:
                parents.add(parent)
            else:
                adjacency_complete = False
                snak_counts["malformed"] += 1
        return {
            **provenance,
            "status": "ok",
            "parents": sorted(parents),
            "direct_adjacency_complete": adjacency_complete,
            "snak_counts": snak_counts,
            "rank_policy": "preferred_if_present_else_normal;deprecated_ignored",
        }

    def fetch_many(self, entity_ids: list[str], cutoff: str) -> dict[str, dict[str, Any]]:
        # MediaWiki forbids rvstart/rvdir=older when a query supplies multiple
        # pages. Keep this convenience method strictly single-page per request.
        return {entity_id: self.fetch(entity_id, cutoff) for entity_id in entity_ids}

    def fetch(self, entity_id: str, cutoff: str) -> dict[str, Any]:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "titles": entity_id,
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "rvlimit": "1",
            "rvstart": cutoff,
            "rvdir": "older",
            "maxlag": str(int(self.maxlag)),
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self._pace_request()
                response = self.session.get(API_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if payload.get("error"):
                    error = payload["error"]
                    if error.get("code") == "maxlag":
                        retry_after = response.headers.get("Retry-After", "")
                        try:
                            delay = max(60.0, float(retry_after))
                        except (TypeError, ValueError):
                            delay = 60.0
                        raise MaxlagError(str(error), retry_after=delay)
                    raise RuntimeError(str(error))
                pages = payload["query"]["pages"]
                if len(pages) != 1:
                    raise RuntimeError(f"MediaWiki response returned {len(pages)} pages for {entity_id}")
                self._last_pages[entity_id] = pages[0]
                return self._record_from_page(entity_id, pages[0])
            except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                retry_delay = min(60.0, (2**attempt) + random.random())
                response = getattr(exc, "response", None)
                if isinstance(exc, MaxlagError):
                    retry_delay = exc.retry_after
                elif response is not None and int(getattr(response, "status_code", 0)) == 429:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        retry_delay = max(60.0, float(retry_after))
                    except (TypeError, ValueError):
                        retry_delay = 60.0
                time.sleep(retry_delay)
        raise RuntimeError(f"Historical retrieval failed for {entity_id}; construction stopped") from last_error


class MaxlagError(RuntimeError):
    """MediaWiki maxlag response carrying its requested retry delay."""

    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _cache_path(cache_dir: Path, entity_id: str) -> Path:
    return cache_dir / f"{entity_id}.json"


def _load_or_fetch(
    client: HistoricalRevisionClient,
    cache_dir: Path,
    entity_id: str,
    cutoff: str,
) -> dict[str, Any]:
    path = _cache_path(cache_dir, entity_id)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("cache_schema_version") == HIERARCHY_CACHE_SCHEMA_VERSION
            and cached.get("parser_version") == HIERARCHY_PARSER_VERSION
            and cached.get("cutoff") == cutoff
        ):
            return cached["record"]
        # A stale parse can only be migrated by reparsing retained raw source.
        # A revision-content checksum proves source identity, not parser
        # correctness.  Preserve the legacy entry before writing v2.
        source_page = cached.get("source_page")
        if cached.get("cutoff") == cutoff and isinstance(source_page, dict):
            record = client._record_from_page(entity_id, source_page)
            legacy = path.with_name(f"{path.stem}.pre-parser-v{HIERARCHY_PARSER_VERSION}{path.suffix}")
            if not legacy.exists():
                shutil.copy2(path, legacy)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "cache_schema_version": HIERARCHY_CACHE_SCHEMA_VERSION,
                        "parser_version": HIERARCHY_PARSER_VERSION,
                        "cutoff": cutoff,
                        "source_page": source_page,
                        "record": record,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
            return record
        legacy = path.with_name(f"{path.stem}.pre-parser-v{HIERARCHY_PARSER_VERSION}{path.suffix}")
        if not legacy.exists():
            shutil.copy2(path, legacy)
    record = client.fetch(entity_id, cutoff)
    source_page = client._last_pages.get(entity_id)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "cache_schema_version": HIERARCHY_CACHE_SCHEMA_VERSION,
                "parser_version": HIERARCHY_PARSER_VERSION,
                "cutoff": cutoff,
                "source_page": source_page,
                "record": record,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return record


def _load_or_fetch_many(
    client: HistoricalRevisionClient,
    cache_dir: Path,
    entity_ids: list[str],
    cutoff: str,
) -> dict[str, dict[str, Any]]:
    # Persist each response immediately. If a later entity fails, every prior
    # revision in this queue chunk remains available to a safe restart.
    return {
        entity_id: _load_or_fetch(client, cache_dir, entity_id, cutoff)
        for entity_id in entity_ids
    }


def _closure_complete(node: str, records: dict[str, dict[str, Any]]) -> bool:
    stack = [node]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        record = records.get(current)
        if (
            record is None
            or record.get("status") != "ok"
            or not record.get("direct_adjacency_complete", False)
        ):
            return False
        stack.extend(record.get("parents", []))
    return True


def build(args: argparse.Namespace) -> dict[str, Any]:
    seeds = benchmark_seed_ids(args.registry, args.benchmark, args.encoder)
    if args.expected_seed_count is not None and len(seeds) != args.expected_seed_count:
        raise RuntimeError(f"Expected {args.expected_seed_count} seed classes, found {len(seeds)}")
    args.cache.mkdir(parents=True, exist_ok=True)
    client = HistoricalRevisionClient(
        timeout=args.timeout,
        retries=args.retries,
        request_delay=args.request_delay,
        maxlag=args.maxlag,
    )
    records: dict[str, dict[str, Any]] = {}
    pending = deque(sorted(seeds))
    queued = set(pending)
    while pending:
        batch = [pending.popleft() for _ in range(min(args.batch_size, len(pending)))]
        batch_records = _load_or_fetch_many(client, args.cache, batch, args.cutoff)
        for entity_id in batch:
            record = batch_records[entity_id]
            records[entity_id] = record
            for parent in record.get("parents", []):
                if parent not in queued:
                    queued.add(parent)
                    pending.append(parent)
        if len(records) % 100 == 0:
            print(f"retrieved={len(records)} queued={len(queued)}", flush=True)

    graph = nx.DiGraph()
    graph.add_nodes_from(records)
    graph.add_edges_from(
        (child, parent)
        for child, record in records.items()
        for parent in record.get("parents", [])
    )
    cycles = sorted(
        sorted(component)
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1 or any(graph.has_edge(node, node) for node in component)
    )
    for node, record in records.items():
        record["closure_complete"] = _closure_complete(node, records)
    edges = sorted((child, parent) for child, record in records.items() for parent in record.get("parents", []))
    payload: dict[str, Any] = {
        "schema_version": HIERARCHY_SCHEMA_VERSION,
        "parser_version": HIERARCHY_PARSER_VERSION,
        "cache_schema_version": HIERARCHY_CACHE_SCHEMA_VERSION,
        "artifact": "wikidata-p279-hierarchy",
        "cutoff": args.cutoff,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "retrieval": {
            "api": API_URL,
            "query_semantics": "latest revision at or before cutoff via rvstart and rvdir=older",
            "user_agent": USER_AGENT,
            "rank_policy": "preferred_if_present_else_normal;deprecated_ignored",
            "present_day_fallback": False,
            "maxlag": args.maxlag,
        },
        "seed_ids": sorted(seeds),
        "seed_count": len(seeds),
        "node_count": len(records),
        "edge_count": len(edges),
        "cycles": cycles,
        "records": {node: records[node] for node in sorted(records)},
        "checksums": {
            "registry_sha256": sha256_file(args.registry),
            "encoder_sha256": sha256_file(args.encoder),
            "benchmark_inputs": {path.name: sha256_file(path) for path in sorted(args.benchmark.glob("df_*.parquet"))},
            "direct_edges_sha256": canonical_sha256(edges),
            "seed_ids_sha256": canonical_sha256(sorted(seeds)),
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("data/interim/constraint_registry_full.parquet"))
    parser.add_argument("--benchmark", type=Path, default=Path("data/interim/full_strat1m_minocc100"))
    parser.add_argument("--encoder", type=Path, default=Path("data/interim/full_strat1m_minocc100/globalintencoder.txt"))
    parser.add_argument("--cache", type=Path, default=Path("data/interim/hierarchy_cache_2018-07-01.v2"))
    parser.add_argument("--output", type=Path, default=Path("data/static/wikidata-p279-2018-07-01.v2.json"))
    parser.add_argument("--cutoff", default=HIERARCHY_CUTOFF)
    parser.add_argument("--expected-seed-count", type=int, default=3941)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--maxlag",
        type=int,
        default=10,
        help="MediaWiki maxlag threshold; maxlag responses wait at least 60 seconds before retry.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Minimum seconds between historical revision requests (default: 1.0).",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seeds-only", action="store_true", help="Print the seed count without network retrieval.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.batch_size <= 50:
        raise ValueError("--batch-size must be between 1 and the MediaWiki titles limit of 50")
    if args.request_delay < 0:
        raise ValueError("--request-delay must be non-negative")
    if args.maxlag <= 0:
        raise ValueError("--maxlag must be positive")
    if args.seeds_only:
        seeds = benchmark_seed_ids(args.registry, args.benchmark, args.encoder)
        print(f"seed_count={len(seeds)}")
        if args.expected_seed_count is not None and len(seeds) != args.expected_seed_count:
            raise RuntimeError(f"Expected {args.expected_seed_count} seed classes, found {len(seeds)}")
        return 0
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {args.output} ({payload['node_count']} nodes, {payload['edge_count']} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
