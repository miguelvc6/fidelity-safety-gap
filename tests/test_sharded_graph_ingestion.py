from __future__ import annotations

import importlib.util
import pickle
import sys
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.data_encoders import GraphStreamDataset, graph_dataset_filename
from modules.training_utils import load_graph_dataset


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_graph(label: int, constraint_type: str) -> Data:
    graph = Data(
        x=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([[label, 0, 0, 0, 0, 0]], dtype=torch.long),
    )
    graph.constraint_type = constraint_type
    return graph


def _write_torch_shards(base_path: Path, shards: list[list[Data]]) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, shard in enumerate(shards):
        shard_path = base_path.with_name(f"{base_path.stem}-shard{idx:03d}.pt")
        torch.save(shard, shard_path)
    manifest_path = base_path.with_suffix(base_path.suffix + ".manifest.json")
    manifest_path.write_text(
        f'{{"graph_count": {sum(len(shard) for shard in shards)}, "sharded": true}}',
        encoding="utf-8",
    )


def test_load_graph_dataset_streams_torch_shards() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base_path = root / graph_dataset_filename("train", "node_id")
        _write_torch_shards(
            base_path,
            [
                [_make_graph(1, "single"), _make_graph(2, "single")],
                [_make_graph(3, "conflictWith")],
            ],
        )

        dataset = load_graph_dataset(base_path)

        assert isinstance(dataset, GraphStreamDataset)
        context_indices = [int(getattr(graph, "context_index")) for graph in dataset]
        labels = [int(graph.y[0, 0].item()) for graph in dataset]

    assert context_indices == [0, 1, 2]
    assert labels == [1, 2, 3]


def test_eval_accepts_sharded_test_split_without_monolith() -> None:
    eval_module = _load_module(ROOT / "src" / "09_eval.py", "eval_09_shard_test")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        processed_dir = root / "data" / "processed" / "full_minocc100"
        base_path = processed_dir / graph_dataset_filename("test", "node_id")
        graphs = [
            _make_graph(1, "single"),
            _make_graph(2, "single"),
            _make_graph(3, "conflictWith"),
        ]
        _write_torch_shards(base_path, [graphs[:2], graphs[2:]])

        dataset = eval_module.load_split(processed_dir, "node_id", "test")
        predictions = torch.cat([graph.y for graph in graphs], dim=0)
        metrics = eval_module.eval(
            None,
            dataset,
            precomputed_predictions=predictions,
        )

    assert "micro_f1" in metrics
    assert metrics["support_per_constraint_type"] == {"single": 2, "conflictWith": 1}


def test_global_metrics_postprocess_accepts_streaming_test_data() -> None:
    eval_module = _load_module(ROOT / "src" / "09_eval.py", "eval_09_stream_global_test")

    with tempfile.TemporaryDirectory() as tmpdir:
        stream_path = Path(tmpdir) / "test_graph-node_id.pkl"
        graphs = [_make_graph(1, "single"), _make_graph(2, "conflictWith")]
        for idx, graph in enumerate(graphs):
            graph.factor_constraint_ids = torch.tensor([idx + 10], dtype=torch.long)
            graph.factor_checkable_pre = torch.tensor([True])
            graph.factor_satisfied_pre = torch.tensor([idx % 2], dtype=torch.long)
            graph.primary_factor_index = 0
        with stream_path.open("wb") as handle:
            for graph in graphs:
                pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

        captured: dict[str, object] = {}

        def fake_evaluate_global_repair_samples(**kwargs):
            captured["kwargs"] = kwargs
            return {"paper_metrics": {}, "per_constraint_type": {}, "per_instance": []}

        original = eval_module.evaluate_global_repair_samples
        eval_module.evaluate_global_repair_samples = fake_evaluate_global_repair_samples
        try:
            support = eval_module.GlobalMetricsSupport(
                rows=[object(), object()],
                evaluator=object(),
            )
            postprocess, state = support.build_postprocess(GraphStreamDataset(stream_path))
            predictions = torch.cat([graph.y for graph in graphs], dim=0)
            kinds = [graph.constraint_type for graph in graphs]
            postprocess(predictions, predictions, kinds)
        finally:
            eval_module.evaluate_global_repair_samples = original

    assert "paper_metrics" in state
    assert "pre_vectors" not in captured["kwargs"]


def test_eval_loads_chooser_checkpoint_without_chooser_mode() -> None:
    eval_module = _load_module(ROOT / "src" / "09_eval.py", "eval_09_chooser_checkpoint_test")

    model_cfg = eval_module.ModelConfig(
        model="GIN_PRESSURE",
        num_embedding_size=8,
        hidden_channels=8,
        head_hidden=8,
        num_layers=2,
        dropout=0.0,
        entity_class_ids=[0, 1, 2],
        predicate_class_ids=[0, 1, 2],
        num_factor_types=1,
    )
    model = eval_module.build_model("GIN_PRESSURE", 3, model_cfg)
    model.enable_chooser()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        torch.save(
            {
                "model_state": model.state_dict(),
                "num_graph_nodes": 3,
                "model_name": "GIN_PRESSURE",
                "model_cfg": model_cfg.to_dict(),
            },
            run_dir / "checkpoint.pth",
        )

        captured: dict[str, object] = {}

        def fake_run_and_save(**kwargs):
            captured["model"] = kwargs["model"]

        original = eval_module._run_and_save
        eval_module._run_and_save = fake_run_and_save
        try:
            eval_module.evaluate_trained_model(
                run_directory=run_dir,
                model_cfg=model_cfg,
                device=torch.device("cpu"),
                test_data=[],
            )
        finally:
            eval_module._run_and_save = original

    loaded = captured["model"]
    assert loaded.chooser_enabled


if __name__ == "__main__":
    test_load_graph_dataset_streams_torch_shards()
    test_eval_accepts_sharded_test_split_without_monolith()
    test_global_metrics_postprocess_accepts_streaming_test_data()
    test_eval_loads_chooser_checkpoint_without_chooser_mode()
    print("sharded graph ingestion tests passed")
