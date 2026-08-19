import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import pandas as pd
import pytest
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig
from modules.h2_eval import (
    aggregate_semantic_records,
    _select_batch_predictions,
    clone_with_factor_pressure_mask,
    collect_h2_variant_outputs,
    count_train_factor_exposure,
    density_bucket,
    exposure_bucket,
    factor_pressure_overlap,
)
from modules.models import build_model


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _factor_graph() -> Data:
    graph = Data(
        x=torch.arange(5, dtype=torch.long),
        edge_index=torch.tensor(
            [
                [0, 1, 3, 3, 4, 4],
                [1, 2, 0, 1, 0, 2],
            ],
            dtype=torch.long,
        ),
        edge_type=torch.tensor([0, 1, 4, 5, 4, 6], dtype=torch.long),
        y=torch.zeros((1, 6), dtype=torch.long),
    )
    graph.factor_node_index = torch.tensor([3, 4], dtype=torch.long)
    graph.factor_constraint_ids = torch.tensor([101, 202], dtype=torch.long)
    graph.factor_types = torch.tensor([0, 1], dtype=torch.long)
    graph.primary_factor_index = 0
    return graph


def test_semantic_metrics_handle_pre_post_and_single_class() -> None:
    records = [
        {"state": "pre", "factor_family": "single", "checkable": True, "label": 1, "score": 0.9},
        {"state": "pre", "factor_family": "single", "checkable": True, "label": 1, "score": 0.8},
        {"state": "pre", "factor_family": "single", "checkable": False, "label": 0, "score": 0.7},
        {"state": "post_gold", "factor_family": "single", "checkable": True, "label": 0, "score": 0.2},
        {"state": "post_gold", "factor_family": "single", "checkable": True, "label": 1, "score": 0.7},
    ]

    rows = aggregate_semantic_records(records, ("state", "factor_family"))
    by_state = {row["state"]: row for row in rows}

    assert by_state["pre"]["support"] == 2
    assert by_state["pre"]["auroc"] is None
    assert by_state["pre"]["accuracy"] == 1.0
    assert by_state["post_gold"]["support"] == 2
    assert by_state["post_gold"]["f1"] == 1.0


def test_transfer_and_density_helpers_are_deterministic() -> None:
    train_graphs = []
    for ids in ([1, 2, 2], [2, 3]):
        graph = Data(x=torch.zeros((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
        graph.factor_constraint_ids = torch.tensor(ids, dtype=torch.long)
        train_graphs.append(graph)

    counts = count_train_factor_exposure(train_graphs)
    assert counts[1] == 1
    assert counts[2] == 3
    assert exposure_bucket(0) == "unseen"
    assert exposure_bucket(3) == "low_1_10"
    assert exposure_bucket(42) == "medium_11_100"
    assert exposure_bucket(101) == "high_gt100"
    assert density_bucket(1) == "1"
    assert density_bucket(5) == "5_16"


def test_train_factor_exposure_can_be_read_from_labeled_parquet(tmp_path: Path) -> None:
    path = tmp_path / "df_train.parquet"
    pd.DataFrame({"factor_constraint_ids": [[1, 2, 2], [2, 3]]}).to_parquet(path, index=False)

    counts = count_train_factor_exposure(path)

    assert counts == {1: 1, 2: 3, 3: 1}


def test_h2_prediction_selection_matches_configured_selector(monkeypatch) -> None:
    graphs = [Data(context_index=0)]
    logits = torch.tensor([[[3.0, 1.0], [3.0, 1.0], [3.0, 1.0], [3.0, 1.0], [3.0, 1.0], [3.0, 1.0]]])
    candidates = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    support = SimpleNamespace(
        contexts=[object()],
        heuristics=SimpleNamespace(placeholder_ids={"x": 99}),
        candidate_cfg=object(),
    )

    monkeypatch.setattr("modules.h2_eval.build_candidates", lambda **_: (candidates, {}))

    class FakeModel:
        num_target_ids = 2

        @staticmethod
        def score_candidates(graph_emb, candidate_tensor):
            return torch.tensor([0.0, 1.0], device=candidate_tensor.device)

    chooser = _select_batch_predictions(
        model=FakeModel(),
        graphs=graphs,
        logits=logits,
        graph_emb=torch.ones((1, 2)),
        chooser_support=support,
        direct_safety_support=None,
    )
    assert chooser.tolist() == [candidates[1]]

    monkeypatch.setattr(
        "modules.h2_eval.score_candidates_from_logits",
        lambda proposal_logits, candidate_tensor: torch.tensor([1.0, 0.0], device=candidate_tensor.device),
    )
    direct = _select_batch_predictions(
        model=FakeModel(),
        graphs=graphs,
        logits=logits,
        graph_emb=None,
        chooser_support=None,
        direct_safety_support=support,
    )
    assert direct.tolist() == [candidates[0]]


def test_h2_normal_prediction_replay_is_exact_and_count_checked() -> None:
    graphs = []
    for idx in range(2):
        graph = Data(
            x=torch.tensor([idx], dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_type=torch.empty((0,), dtype=torch.long),
            y=torch.zeros((1, 6), dtype=torch.long),
        )
        graph.constraint_type = "single"
        graphs.append(graph)

    class FakeModel:
        num_target_ids = 3

        @staticmethod
        def eval():
            return None

        @staticmethod
        def forward_for_evaluation(batch):
            return {
                "edit_logits": torch.zeros((batch.num_graphs, 6, 3)),
                "graph_emb": torch.zeros((batch.num_graphs, 2)),
            }

    expected = torch.tensor([[1, 1, 1, 0, 0, 0], [2, 2, 2, 0, 0, 0]], dtype=torch.long)
    output = collect_h2_variant_outputs(
        model=FakeModel(),
        dataset=graphs,
        device=torch.device("cpu"),
        variant_name="normal",
        exposure_counts={},
        family_lookup=lambda constraint_id, factor_type: "single",
        batch_size=2,
        collect_factor_records=False,
        precomputed_predictions=expected,
    )
    assert torch.equal(output["predictions"], expected)

    with pytest.raises(ValueError, match="shorter than the graph dataset"):
        collect_h2_variant_outputs(
            model=FakeModel(),
            dataset=graphs,
            device=torch.device("cpu"),
            variant_name="normal",
            exposure_counts={},
            family_lookup=lambda constraint_id, factor_type: "single",
            batch_size=2,
            collect_factor_records=False,
            precomputed_predictions=expected[:1],
        )


def test_counterfactual_masking_removes_expected_edges_without_mutation() -> None:
    graph = _factor_graph()
    original_edge_count = int(graph.edge_index.size(1))
    no_pressure = clone_with_factor_pressure_mask(graph, "no_factor_pressure")
    primary_only = clone_with_factor_pressure_mask(graph, "primary_only_pressure")
    secondary_only = clone_with_factor_pressure_mask(graph, "secondary_only_pressure")

    assert int(graph.edge_index.size(1)) == original_edge_count
    assert no_pressure.edge_type.tolist() == [0, 1]
    assert primary_only.edge_type.tolist() == [0, 1, 4, 5]
    assert secondary_only.edge_type.tolist() == [0, 1, 4, 6]

    overlap = factor_pressure_overlap(graph)
    assert overlap["factor_pressure_edges"] == 4
    assert overlap["shared_pressure_target_nodes"] == 1


def test_model_config_accepts_pressure_module_sharing() -> None:
    cfg = ModelConfig.from_mapping({"pressure_module_sharing": "shared"})
    assert cfg.pressure_module_sharing == "shared"

    model_cfg = ModelConfig.from_mapping(
        {
            "model": "GIN_PRESSURE",
            "num_layers": 2,
            "hidden_channels": 8,
            "head_hidden": 8,
            "dropout": 0.0,
            "entity_class_ids": [0, 1],
            "predicate_class_ids": [0, 1],
            "num_factor_types": 3,
            "pressure_enabled": True,
            "pressure_module_sharing": "shared",
        }
    )
    model = build_model("GIN_PRESSURE", 8, model_cfg)
    assert len(model._pressure_role_modules["0"]) == 1


def test_h2_ablation_config_generation_is_opt_in() -> None:
    module = _load_module(ROOT / "scripts" / "make_experiment_configs.py", "make_configs_h2_test")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        processed = root / "data" / "processed" / "toy_minocc100"
        processed.mkdir(parents=True)
        graph = _factor_graph()
        torch.save([graph], processed / "train_graph-node_id-shard000.pt")
        interim = root / "data" / "interim" / "toy_minocc100_labeled"
        interim.mkdir(parents=True)
        for split in ("train", "val", "test"):
            pd.DataFrame({"factor_types": [[0, 1]]}).to_parquet(
                interim / f"df_{split}.parquet",
                index=False,
            )
        models = root / "models"

        argv_backup = list(sys.argv)
        try:
            sys.argv = [
                "make_experiment_configs.py",
                "--processed-root",
                str(root / "data" / "processed"),
                "--models-root",
                str(models),
                "--interim-root",
                str(root / "data" / "interim"),
            ]
            module.main()
            default_dirs = sorted(path.name for path in models.iterdir() if path.is_dir())
            assert not any(name.startswith("h2_") for name in default_dirs)
            assert len(default_dirs) == 5

            sys.argv = [
                "make_experiment_configs.py",
                "--processed-root",
                str(root / "data" / "processed"),
                "--models-root",
                str(models),
                "--interim-root",
                str(root / "data" / "interim"),
                "--include-h2-ablations",
            ]
            module.main()
        finally:
            sys.argv = argv_backup

        h2_dirs = sorted(path.name for path in models.iterdir() if path.name.startswith("h2_"))
        assert h2_dirs == [
            "h2_a1_legacy_shared_executor__toy_minocc100__node_id",
            "h2_a1_no_factor_loss__toy_minocc100__node_id",
            "h2_a1_shared_pressure__toy_minocc100__node_id",
        ]
        cfg = json.loads((models / "h2_a1_shared_pressure__toy_minocc100__node_id" / "config.json").read_text())
        assert cfg["model_config"]["pressure_module_sharing"] == "shared"
        cfg = json.loads((models / "h2_a1_no_factor_loss__toy_minocc100__node_id" / "config.json").read_text())
        assert cfg["training_config"]["factor_loss"]["enabled"] is False
