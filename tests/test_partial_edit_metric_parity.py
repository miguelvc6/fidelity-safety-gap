from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from modules.constraint_checkers import (
    ConstraintInstance,
    ValidationOutcome,
    evaluate_constraint_outcome,
)
from modules.evaluation_artifacts import (
    build_predictions_frame,
    load_and_validate_predictions,
    write_prediction_artifacts,
)
from modules.reranker_eval import CandidateConstraintEvaluator


ROOT = Path(__file__).resolve().parents[1]


def _evaluation_module():
    path = ROOT / "src" / "09_eval.py"
    spec = importlib.util.spec_from_file_location("partial_edit_metric_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _labeler_module():
    path = ROOT / "src" / "05_constraint_labeler.py"
    spec = importlib.util.spec_from_file_location("ordered_edit_labeler", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluator() -> CandidateConstraintEvaluator:
    constraint = ConstraintInstance(
        constraint_id=900,
        constraint_type="single",
        constraint_type_id=0,
        constrained_property=10,
        required_properties=set(),
        allowed_items=set(),
        allowed_classes=set(),
        relation_predicates=[],
        inverse_properties=[],
        conflict_properties=set(),
    )
    evaluator = CandidateConstraintEvaluator.__new__(CandidateConstraintEvaluator)
    evaluator._encoder = None
    evaluator._assume_complete = True
    evaluator._constraint_scope = "local"
    evaluator._use_encoded_ids = True
    evaluator._placeholder_token_ids = {}
    evaluator._hierarchy = None
    evaluator._get_constraint_instance = lambda constraint_id: (
        constraint if constraint_id == 900 else None
    )
    return evaluator


def _row() -> SimpleNamespace:
    return SimpleNamespace(
        constraint_id=900,
        constraint_type="single",
        subject=1,
        predicate=10,
        object=5,
        other_subject=0,
        other_predicate=0,
        other_object=0,
        subject_predicates=[10, 10],
        subject_objects=[5, 6],
        object_predicates=[],
        object_objects=[],
        other_entity_predicates=[],
        other_entity_objects=[],
        local_constraint_ids=[900],
        factor_constraint_ids=[900],
        primary_factor_index=0,
    )


def test_partial_slots_match_candidate_final_metric_and_replay(tmp_path: Path) -> None:
    module = _evaluation_module()
    evaluator = _evaluator()
    row = _row()
    raw = torch.tensor([[1, 10, 0, 1, 10, 6]])

    direct = evaluator.evaluate_full(row, candidate_slots=raw[0].tolist())
    callback, output = module.GlobalMetricsSupport(rows=[row], evaluator=evaluator).build_postprocess()
    callback(raw, torch.zeros_like(raw), ["single"])
    instance = output["paper_metric_instances"][0]

    assert direct["post_outcomes"] == ["unknown"]
    assert instance["post_checkable"] == direct["post_checkable"]
    assert instance["post_satisfied"] == direct["post_satisfied"]
    assert instance["events"]["pfr"] == {"numerator": 0, "denominator": 1}

    frame = build_predictions_frame(
        raw,
        rows=[row],
        kinds=["single"],
        metric_instances=[instance],
    )
    config = tmp_path / "config.json"
    checkpoint = tmp_path / "checkpoint.pth"
    dataset = tmp_path / "df_test.parquet"
    graph = tmp_path / "test_graph.pt"
    config.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_bytes(b"dataset")
    graph.write_bytes(b"graph")
    predictions_path, _manifest_path, _manifest = write_prediction_artifacts(
        tmp_path / "evaluations",
        frame,
        config_path=config,
        checkpoint_path=checkpoint,
        dataset_path=dataset,
        graph_paths=[graph],
        dataset_variant="fixture_minocc100",
    )
    replayed, _ = load_and_validate_predictions(
        predictions_path,
        rows=[row],
        dataset_path=dataset,
        graph_paths=[graph],
        dataset_variant="fixture_minocc100",
    )
    replay_callback, replay_output = module.GlobalMetricsSupport(
        rows=[row],
        evaluator=evaluator,
    ).build_postprocess()
    replay_callback(replayed, torch.zeros_like(replayed), ["single"])

    replay_instance = replay_output["paper_metric_instances"][0]
    assert replay_instance["post_checkable"] == instance["post_checkable"]
    assert replay_instance["post_satisfied"] == instance["post_satisfied"]
    assert replay_instance["events"] == instance["events"]


def test_labeler_and_candidate_evaluator_preserve_ordered_edit_events() -> None:
    required = ConstraintInstance(
        constraint_id=900,
        constraint_type="itemRequiresStatement",
        constraint_type_id=0,
        constrained_property=10,
        required_properties={20},
        allowed_items=set(),
        allowed_classes=set(),
        relation_predicates=[],
        inverse_properties=[],
        conflict_properties=set(),
    )
    row = _row()
    row.del_subject, row.del_predicate, row.del_object = 1, 20, 5
    row.add_subject, row.add_predicate, row.add_object = 1, 20, 5

    labeler = _labeler_module()
    labeler_state = labeler._apply_edit(
        {1: {10: {5}}},
        {1: {10}},
        {10},
        row,
        placeholder_map={},
        assume_complete=True,
        cast_int=True,
    )
    assert evaluate_constraint_outcome(labeler_state, required) == ValidationOutcome.SATISFIED

    evaluator = _evaluator()
    evaluator._get_constraint_instance = lambda constraint_id: (
        required if constraint_id == 900 else None
    )
    details = evaluator.evaluate_full(
        row,
        candidate_slots=(1, 20, 5, 1, 20, 5),
    )
    assert details["post_outcomes"] == ["satisfied"]
