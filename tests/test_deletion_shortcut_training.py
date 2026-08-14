from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig, TrainingConfig
from modules.model_store import atomic_torch_save, get_last_checkpoint_path


def _load_training_script():
    path = SRC / "07_train.py"
    spec = importlib.util.spec_from_file_location("train_07_for_study_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reranker_script():
    path = SRC / "08_train_reranker.py"
    spec = importlib.util.spec_from_file_location("train_08_for_study_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model_config() -> ModelConfig:
    return ModelConfig.from_mapping(
        {
            "dataset_variant": "toy_minocc100",
            "encoding": "node_id",
            "model": "GIN",
            "constraint_representation": "factorized",
            "num_layers": 2,
            "hidden_channels": 8,
            "head_hidden": 8,
            "factor_executor_impl": "per_type_grouped_v2",
            "gold_edit_embedding_mode": "compact",
            "pressure_module_sharing": "per_type",
            "num_factor_types": 3,
            "active_factor_type_ids": [0, 2],
        }
    )


def test_extended_direct_safety_defaults_preserve_legacy_configuration() -> None:
    direct = TrainingConfig().direct_safety
    assert direct.loss_weight == 1.0
    assert direct.score_temperature == 1.0
    assert direct.focus_deletion_weight == 0.0
    assert TrainingConfig().initialization_checkpoint is None
    assert TrainingConfig().save_last_checkpoint is False
    assert TrainingConfig().max_valid_edit_logit_abs is None


def test_invalid_stability_controls_are_rejected() -> None:
    with pytest.raises(ValueError, match="score_temperature"):
        TrainingConfig.from_mapping({"direct_safety": {"score_temperature": 0.0}})
    with pytest.raises(ValueError, match="max_valid_edit_logit_abs"):
        TrainingConfig.from_mapping({"max_valid_edit_logit_abs": -1.0})


def test_global_fix_base_preservation_penalty_is_opt_in() -> None:
    reranker = _load_reranker_script()
    probs = torch.tensor([0.5, 0.5])
    metrics = [
        SimpleNamespace(global_satisfied_fraction=1.0, focus_deleted=1),
        SimpleNamespace(global_satisfied_fraction=0.8, focus_deleted=0),
    ]

    control = reranker._global_fix_loss(
        probs,
        metrics,
        focus_deletion_weight=0.0,
        device=torch.device("cpu"),
    )
    preserving = reranker._global_fix_loss(
        probs,
        metrics,
        focus_deletion_weight=1.0,
        device=torch.device("cpu"),
    )

    assert control.item() == pytest.approx(-0.9)
    assert preserving.item() == pytest.approx(-0.4)
    assert reranker.RerankerTrainingConfig().seed == 42
    assert reranker.RerankerTrainingConfig().focus_deletion_weight == 0.0


def test_atomic_torch_save_replaces_complete_checkpoint(tmp_path: Path) -> None:
    destination = get_last_checkpoint_path(tmp_path)
    atomic_torch_save({"epoch": 1}, destination)
    atomic_torch_save({"epoch": 2}, destination)

    assert torch.load(destination, map_location="cpu") == {"epoch": 2}
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_strict_initialization_loads_matching_state_and_records_identity(tmp_path: Path) -> None:
    training = _load_training_script()
    model = torch.nn.Linear(3, 2)
    expected = {key: value.detach().clone() + 1.0 for key, value in model.state_dict().items()}
    config = _model_config()
    config_path = tmp_path / "run" / "config.json"
    config_path.parent.mkdir()
    checkpoint_path = tmp_path / "source" / "checkpoint.pth"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "model_state": expected,
            "model_cfg": config.to_dict(),
            "model_name": config.model,
        },
        checkpoint_path,
    )

    identity = training._strict_initialize_model(
        model,
        model_cfg=config,
        config_path=config_path,
        checkpoint_reference="../source/checkpoint.pth",
    )

    assert identity["path"] == str(checkpoint_path.resolve())
    assert len(identity["sha256"]) == 64
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_strict_initialization_rejects_model_configuration_mismatch(tmp_path: Path) -> None:
    training = _load_training_script()
    model = torch.nn.Linear(3, 2)
    config = _model_config()
    checkpoint_config = config.updated(hidden_channels=16)
    config_path = tmp_path / "run" / "config.json"
    config_path.parent.mkdir()
    checkpoint_path = tmp_path / "source" / "checkpoint.pth"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": checkpoint_config.to_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="model configuration mismatch"):
        training._strict_initialize_model(
            model,
            model_cfg=config,
            config_path=config_path,
            checkpoint_reference="../source/checkpoint.pth",
        )
