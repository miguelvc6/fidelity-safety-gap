from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION
from modules.semantics_provenance import expected_semantic_contracts


ROOT = Path(__file__).resolve().parents[1]


def _diagnostic_module():
    path = ROOT / "scripts" / "diagnose_validator_transitions.py"
    spec = importlib.util.spec_from_file_location("validator_transition_diagnostic_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transition_report_includes_primary_identity_and_single_witnesses(tmp_path: Path) -> None:
    labeled = tmp_path / "labeled"
    labeled.mkdir()
    manifest = {
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "semantic_contracts": expected_semantic_contracts(),
    }
    (labeled / "label_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    row = {
        "constraint_id": 900,
        "constraint_type": "single",
        "subject": 1,
        "predicate": 10,
        "object": 5,
        "other_subject": 0,
        "other_predicate": 0,
        "other_object": 0,
        "subject_predicates": [10, 10],
        "subject_objects": [5, 6],
        "object_predicates": [],
        "object_objects": [],
        "other_entity_predicates": [],
        "other_entity_objects": [],
        "add_subject": 0,
        "add_predicate": 0,
        "add_object": 0,
        "del_subject": 1,
        "del_predicate": 10,
        "del_object": 6,
        "factor_constraint_ids": [900],
        "primary_factor_index": 0,
        "factor_outcome_pre": ["violated"],
        "factor_outcome_post_gold": ["satisfied"],
        "factor_unknown_reason_post_gold": [""],
    }
    pd.DataFrame([row]).to_parquet(labeled / "df_test.parquet", index=False)
    args = argparse.Namespace(
        labeled_dir=labeled,
        splits=["test"],
        full_scan=False,
        max_rows=1000,
        single_examples=20,
        mismatch_examples=20,
    )
    report = _diagnostic_module().run(args)
    split = report["splits"]["test"]
    assert split["transitions_3x3"]["violated->satisfied"] == 1
    assert split["primary_index_mismatch_count"] == 0
    failure = split["single_value_failure_examples"]
    assert failure == []

    row["subject_predicates"] = [10, 10, 10]
    row["subject_objects"] = [5, 6, 7]
    row["factor_outcome_post_gold"] = ["violated"]
    pd.DataFrame([row]).to_parquet(labeled / "df_test.parquet", index=False)
    failure = _diagnostic_module().run(args)["splits"]["test"]["single_value_failure_examples"][0]
    assert failure["pre_cardinality"] == 3
    assert failure["post_cardinality"] == 2
    assert failure["deleted_triple_present_pre"] is True
    assert failure["surviving_values"] == [5, 7]
