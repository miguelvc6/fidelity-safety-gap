from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    path = ROOT / "scripts" / "make_experiment_configs.py"
    spec = importlib.util.spec_from_file_location("make_corrected_configs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_generator(
    tmp_path: Path,
    *,
    test_types=(0, 2),
    include_experimental: bool = False,
) -> Path:
    processed = tmp_path / "processed" / "toy_minocc100"
    processed.mkdir(parents=True)
    graph = Data(factor_types=torch.tensor([0, 2], dtype=torch.long))
    torch.save([graph], processed / "train_graph-node_id-shard000.pt")
    torch.save([graph], processed / "train_graph_repr-eswc_passive-node_id-shard000.pt")

    interim = tmp_path / "interim" / "toy_minocc100_labeled"
    interim.mkdir(parents=True)
    pd.DataFrame({"factor_types": [[0], [2]]}).to_parquet(interim / "df_train.parquet", index=False)
    pd.DataFrame({"factor_types": [[2]]}).to_parquet(interim / "df_val.parquet", index=False)
    pd.DataFrame({"factor_types": [list(test_types)]}).to_parquet(interim / "df_test.parquet", index=False)

    models = tmp_path / "models"
    generator = _load_generator()
    old_argv = sys.argv
    sys.argv = [
        "make_experiment_configs.py",
        "--processed-root",
        str(tmp_path / "processed"),
        "--interim-root",
        str(tmp_path / "interim"),
        "--models-root",
        str(models),
    ]
    if include_experimental:
        sys.argv.append("--include-experimental")
    try:
        generator.main()
    finally:
        sys.argv = old_argv
    return models


def _run_deletion_study_generator(tmp_path: Path) -> Path:
    models = _run_generator(tmp_path)
    canonical_path = models / "a1_factorized_imitation_compact_grouped__toy_minocc100__node_id" / "config.json"
    canonical_before = canonical_path.read_bytes()
    generator = _load_generator()
    old_argv = sys.argv
    sys.argv = [
        "make_experiment_configs.py",
        "--processed-root",
        str(tmp_path / "processed"),
        "--interim-root",
        str(tmp_path / "interim"),
        "--models-root",
        str(models),
        "--study",
        "deletion-shortcut-v2",
        "--variant",
        "toy_minocc100",
        "--encoding",
        "node_id",
    ]
    try:
        generator.main()
    finally:
        sys.argv = old_argv
    assert canonical_path.read_bytes() == canonical_before
    return models


def _config(models: Path, name: str) -> dict:
    return json.loads((models / f"{name}__toy_minocc100__node_id" / "config.json").read_text())


def test_canonical_bundle_defines_the_five_paper_systems(tmp_path) -> None:
    models = _run_generator(tmp_path)
    original = _config(models, "b0_eswc_reproduction")
    a1 = _config(models, "a1_factorized_imitation_compact_grouped")
    m1c = _config(models, "m1c_safe_factor_chooser_compact_grouped")
    m1d = _config(models, "m1d_safe_factor_direct_compact_grouped")
    g0 = _config(models, "g0_globalfix_reference_v2")

    assert (
        original["model_config"]["hidden_channels"],
        original["model_config"]["num_layers"],
    ) == (128, 2)
    assert original["model_config"]["dropout"] == 0.5
    assert a1["expected_trainable_parameters"] == 50_072_465

    for payload in (a1, m1c, m1d):
        cfg = payload["model_config"]
        assert cfg["factor_executor_impl"] == "per_type_grouped_v2"
        assert cfg["gold_edit_embedding_mode"] == "compact"
        assert cfg["pressure_module_sharing"] == "per_type"
        assert cfg["active_factor_type_ids"] == [0, 2]
        assert payload["training_config"]["seed"] == 42

    assert m1c["training_config"]["chooser"]["gamma_primary"] == 0.2
    assert g0["proposal_config"]["config_tag"].startswith(
        "a1_factorized_imitation_compact_grouped__"
    )
    assert not (models / "b0_parameter_matched__toy_minocc100__node_id").exists()


def test_parameter_matched_b0_is_opt_in_and_within_tolerance(tmp_path) -> None:
    models = _run_generator(tmp_path, include_experimental=True)
    matched = _config(models, "b0_parameter_matched")
    a1 = _config(models, "a1_factorized_imitation_compact_grouped")

    assert (
        matched["model_config"]["hidden_channels"],
        matched["model_config"]["num_layers"],
    ) == (304, 4)
    assert matched["model_config"]["dropout"] == 0.17
    assert matched["expected_trainable_parameters"] == 50_110_102
    relative_difference = abs(
        matched["expected_trainable_parameters"] - a1["expected_trainable_parameters"]
    ) / a1["expected_trainable_parameters"]
    assert relative_difference <= 0.001


def test_bundle_rejects_factor_type_seen_only_in_test(tmp_path) -> None:
    with pytest.raises(ValueError, match="absent from train/validation"):
        _run_generator(tmp_path, test_types=(0, 3))


def test_deletion_study_is_additive_and_defines_calibrated_matched_pairs(tmp_path) -> None:
    models = _run_deletion_study_generator(tmp_path)
    m1d = _config(models, "m1d_safe_factor_direct_v2")
    m1d_bp = _config(models, "m1d_safe_factor_direct_base_preserving_v2")
    g0 = _config(models, "g0_globalfix_reference_v2")
    g0_bp = _config(models, "g0_globalfix_base_preserving_v2")

    for payload in (m1d, m1d_bp):
        assert payload["model_config"]["factor_executor_impl"] == "per_type_grouped_v2"
        assert payload["model_config"]["gold_edit_embedding_mode"] == "compact"
        assert payload["training_config"]["seed"] == 42
        assert payload["training_config"]["learning_rate"] == 1e-5
        assert payload["training_config"]["initialization_checkpoint"].endswith(
            "a1_factorized_imitation_compact_grouped__toy_minocc100__node_id/checkpoint.pth"
        )
        direct = payload["training_config"]["direct_safety"]
        assert direct["loss_weight"] == 0.25
        assert direct["score_temperature"] == 6.0
        assert payload["training_config"]["save_last_checkpoint"] is True
        assert payload["training_config"]["max_valid_edit_logit_abs"] == 10_000.0
    assert "focus_deletion_weight" not in m1d["training_config"]["direct_safety"]
    assert m1d_bp["training_config"]["direct_safety"]["focus_deletion_weight"] == 1.0

    for payload in (g0, g0_bp):
        assert payload["training_config"]["objective"] == "global_fix"
        assert payload["training_config"]["seed"] == 42
        assert payload["training_config"]["prediction_include_gold"] is False
        assert payload["training_config"]["save_last_checkpoint"] is True
        assert payload["proposal_config"]["config_tag"].startswith(
            "a1_factorized_imitation_compact_grouped__"
        )
    assert "focus_deletion_weight" not in g0["training_config"]
    assert g0_bp["training_config"]["focus_deletion_weight"] == 1.0
