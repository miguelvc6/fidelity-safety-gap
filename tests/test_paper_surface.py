import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import modules.model_store as model_store
from modules.constraint_checkers import ConstraintInstance, EvidenceState, evaluate_constraint
from modules.constraint_type_map import canonicalize_constraint_type
from modules.config import ModelConfig, TrainingConfig
from modules.data_encoders import graph_dataset_filename
from modules.model_store import config_tag_from_path, ensure_run_dir_for_config


def test_graph_dataset_filename_by_representation() -> None:
    assert graph_dataset_filename("train", "node_id") == "train_graph-node_id.pkl"
    assert (
        graph_dataset_filename("train", "node_id", constraint_representation="eswc_passive")
        == "train_graph_repr-eswc_passive-node_id.pkl"
    )


def test_config_tag_uses_parent_directory_for_config_json() -> None:
    config_path = Path("models/m1d_safe_factor_direct__full_minocc100__node_id/config.json")
    assert config_tag_from_path(config_path) == "m1d_safe_factor_direct__full_minocc100__node_id"


def test_run_dir_for_config_uses_config_parent(tmp_path) -> None:
    config_path = tmp_path / "models" / "hp_m1c_c1__full_minocc100__node_id" / "config.json"
    run_dir = ensure_run_dir_for_config(config_path)
    assert run_dir == config_path.parent
    assert run_dir.exists()


def test_resolve_run_dir_accepts_config_directory_layout(tmp_path) -> None:
    models_root = tmp_path / "models"
    model_dir = models_root / "hp_m1c_c1__full_minocc100__node_id"
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": {
            "dataset_variant": "full_minocc100",
            "encoding": "node_id",
            "model": "GIN_PRESSURE",
        },
        "training_config": {},
    }
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    original_root = model_store.MODELS_ROOT
    model_store.MODELS_ROOT = models_root
    try:
        resolved = model_store.resolve_run_dir(
            "full_minocc100",
            "node_id",
            "GIN_PRESSURE",
            "hp_m1c_c1__full_minocc100__node_id",
        )
        assert resolved == model_dir
        assert model_store.available_config_tags("full_minocc100", "node_id", "GIN_PRESSURE") == [
            "hp_m1c_c1__full_minocc100__node_id"
        ]
    finally:
        model_store.MODELS_ROOT = original_root


def test_paper_surface_configs_accept_new_fields() -> None:
    model_cfg = ModelConfig.from_mapping({"constraint_representation": "eswc_passive"})
    training_cfg = TrainingConfig.from_mapping(
        {
            "direct_safety": {
                "enabled": True,
                "alpha_primary": 2.0,
                "beta_secondary": 0.25,
                "loss_weight": 0.2,
                "score_temperature": 6.0,
                "focus_deletion_weight": 1.0,
                "topk_candidates": 10,
                "max_candidates_total": 40,
            }
        }
    )

    assert model_cfg.constraint_representation == "eswc_passive"
    assert training_cfg.direct_safety.enabled is True
    assert training_cfg.direct_safety.alpha_primary == 2.0
    assert training_cfg.direct_safety.beta_secondary == 0.25
    assert training_cfg.direct_safety.loss_weight == 0.2
    assert training_cfg.direct_safety.score_temperature == 6.0
    assert training_cfg.direct_safety.focus_deletion_weight == 1.0


def test_symmetric_constraint_type_is_supported() -> None:
    family, supported = canonicalize_constraint_type("Q21510862")
    assert family == "symmetric"
    assert supported is True


def test_symmetric_reuses_inverse_checker_semantics() -> None:
    state = EvidenceState(
        facts_by_entity={1: {10: {2}}, 2: {10: {1}}},
        predicates_present={1: {10}, 2: {10}},
        assume_complete=True,
        missing_edits=set(),
        focus_subject=1,
        focus_predicate=10,
        focus_object=2,
        other_subject=0,
        other_predicate=0,
        other_object=0,
    )
    constraint = ConstraintInstance(
        constraint_id=99,
        constraint_type="symmetric",
        constraint_type_id=4,
        constrained_property=10,
        required_properties=set(),
        allowed_items=set(),
        allowed_classes=set(),
        relation_predicates=[],
        inverse_properties=[],
        conflict_properties=set(),
    )

    assert evaluate_constraint(state, constraint, {10}) == (True, True)

    broken_state = EvidenceState(
        facts_by_entity={1: {10: {2}}, 2: {}},
        predicates_present={1: {10}, 2: {10}},
        assume_complete=True,
        missing_edits=set(),
        focus_subject=1,
        focus_predicate=10,
        focus_object=2,
        other_subject=0,
        other_predicate=0,
        other_object=0,
    )
    assert evaluate_constraint(broken_state, constraint, {10}) == (True, False)


if __name__ == "__main__":
    test_graph_dataset_filename_by_representation()
    test_config_tag_uses_parent_directory_for_config_json()
    test_paper_surface_configs_accept_new_fields()
    test_symmetric_constraint_type_is_supported()
    test_symmetric_reuses_inverse_checker_semantics()
    print("paper surface tests passed")
