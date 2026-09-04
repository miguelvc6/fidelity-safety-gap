"""Compatibility gates for validator-derived datasets, graphs, and runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from modules.class_hierarchy import HIERARCHY_CUTOFF, load_hierarchy_artifact, sha256_file
from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION


DEFAULT_HIERARCHY_PATH = Path("data/static/wikidata-p279-2018-07-01.v1.json")


def graph_manifest_path(graph_path: Path) -> Path:
    return graph_path.with_suffix(graph_path.suffix + ".manifest.json")


def validate_graph_semantics(
    graph_paths: Iterable[Path],
    *,
    hierarchy_path: Path = DEFAULT_HIERARCHY_PATH,
) -> dict[str, object]:
    _payload, identity = load_hierarchy_artifact(hierarchy_path, expected_cutoff=HIERARCHY_CUTOFF)
    manifests: list[dict[str, object]] = []
    for graph_path in graph_paths:
        manifest_path = graph_manifest_path(Path(graph_path))
        if not manifest_path.exists():
            raise FileNotFoundError(f"Graph suite lacks semantics manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("validator_semantics_version", -1)) != VALIDATOR_SEMANTICS_VERSION:
            raise ValueError(f"Graph suite has incompatible validator semantics: {manifest_path}")
        provenance = manifest.get("validator_provenance") or {}
        graph_hierarchy = provenance.get("hierarchy") or {}
        if graph_hierarchy.get("content_sha256") != identity.content_sha256:
            raise ValueError(f"Graph suite has incompatible hierarchy: {manifest_path}")
        if manifest.get("constraint_representation") == "factorized":
            labels = provenance.get("labels") or {}
            if not labels.get("sha256"):
                raise ValueError(f"Factor graph lacks label-manifest identity: {manifest_path}")
        build_contract = manifest.get("build_contract") or {}
        contract_path_raw = build_contract.get("path")
        if not contract_path_raw:
            raise ValueError(f"Graph suite lacks build-contract identity: {manifest_path}")
        contract_path = Path(str(contract_path_raw))
        if not contract_path.exists():
            raise FileNotFoundError(f"Graph build contract is missing: {contract_path}")
        if build_contract.get("sha256") != sha256_file(contract_path):
            raise ValueError(f"Graph build contract checksum mismatch: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("validator_provenance") != provenance:
            raise ValueError(f"Graph build contract has incompatible provenance: {contract_path}")
        manifests.append(
            {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
                "split": manifest.get("split"),
                "constraint_representation": manifest.get("constraint_representation"),
            }
        )
    return {
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "hierarchy": dict(identity.__dict__),
        "graph_manifests": manifests,
    }
