from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from modules.class_hierarchy import (
    HIERARCHY_CACHE_SCHEMA_VERSION,
    HIERARCHY_CUTOFF,
    HIERARCHY_PARSER_VERSION,
    HIERARCHY_SCHEMA_VERSION,
    canonical_sha256,
    load_hierarchy_artifact,
    sha256_file,
)
from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION
from modules.semantics_provenance import (
    expected_semantic_contracts,
    validate_graph_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


def _hierarchy(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": HIERARCHY_SCHEMA_VERSION,
        "parser_version": HIERARCHY_PARSER_VERSION,
        "cache_schema_version": HIERARCHY_CACHE_SCHEMA_VERSION,
        "cutoff": HIERARCHY_CUTOFF,
        "records": {},
    }
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _scheduler_module():
    path = ROOT / "src" / "10_scheduler.py"
    spec = importlib.util.spec_from_file_location("scheduler_10_contract_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_manifest_rejects_cross_objective_contract(tmp_path: Path) -> None:
    hierarchy_path = tmp_path / "hierarchy.json"
    _hierarchy(hierarchy_path)
    _payload, identity = load_hierarchy_artifact(hierarchy_path)
    graph = tmp_path / "train_graph.pt"
    graph.write_bytes(b"fixture")
    provenance = {
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "hierarchy": dict(identity.__dict__),
        "semantic_contracts": expected_semantic_contracts(),
    }
    contract_path = tmp_path / "train_graph.build-contract.json"
    contract_path.write_text(
        json.dumps({"validator_provenance": provenance}),
        encoding="utf-8",
    )
    manifest_path = graph.with_suffix(graph.suffix + ".manifest.json")
    manifest = {
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "constraint_representation": "factorized",
        "validator_provenance": {**provenance, "labels": {"sha256": "labels"}},
        "build_contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
    }
    # Build contract and manifest must carry the identical provenance object.
    contract_path.write_text(
        json.dumps({"validator_provenance": manifest["validator_provenance"]}),
        encoding="utf-8",
    )
    manifest["build_contract"]["sha256"] = sha256_file(contract_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validated = validate_graph_semantics([graph], hierarchy_path=hierarchy_path)
    assert validated["semantic_contracts"] == expected_semantic_contracts()

    manifest["validator_provenance"]["semantic_contracts"] = {
        **expected_semantic_contracts(),
        "candidate_objective_version": 999,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic contracts"):
        validate_graph_semantics([graph], hierarchy_path=hierarchy_path)


def test_checkpoint_rejects_cross_objective_contract(tmp_path: Path, monkeypatch) -> None:
    scheduler = _scheduler_module()
    hierarchy_path = tmp_path / "hierarchy.json"
    _hierarchy(hierarchy_path)
    _payload, identity = load_hierarchy_artifact(hierarchy_path)
    monkeypatch.setattr(scheduler, "DEFAULT_HIERARCHY_PATH", hierarchy_path)
    checkpoint = tmp_path / "checkpoint.pth"
    provenance = {
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "semantic_contracts": expected_semantic_contracts(),
        "hierarchy": dict(identity.__dict__),
    }
    torch.save({"training_provenance": provenance}, checkpoint)
    scheduler._validate_checkpoint_semantics(checkpoint, allow_archived_passive=False)

    torch.save(
        {
            "training_provenance": {
                **provenance,
                "semantic_contracts": {
                    **expected_semantic_contracts(),
                    "effective_hierarchy_policy_version": 999,
                },
            }
        },
        checkpoint,
    )
    with pytest.raises(scheduler.ExperimentError, match="semantic contracts"):
        scheduler._validate_checkpoint_semantics(checkpoint, allow_archived_passive=False)
