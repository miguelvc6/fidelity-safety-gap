from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.config import ModelConfig
from modules.factor_dispatch import build_grouped_dispatch
from modules.factor_types import scan_factor_type_ids
from modules.models import (
    FactorPostEditHead,
    FactorTypeExecutor,
    GroupedFactorPostEditHead,
    GroupedFactorTypeExecutor,
    GroupedPressureRole,
    RepairGINFactorPressure,
    build_model,
)


ACTIVE_IDS = (0, 2)


def _load_config_generator():
    path = ROOT / "scripts" / "make_compact_a1_config.py"
    spec = importlib.util.spec_from_file_location("make_compact_a1_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compact_model(*, gold_mode: str = "compact") -> RepairGINFactorPressure:
    return RepairGINFactorPressure(
        num_input_graph_nodes=16,
        num_embedding_size=8,
        num_layers=2,
        hidden=8,
        head_hidden=8,
        dropout=0.0,
        use_node_embeddings=True,
        use_edge_attributes=False,
        entity_class_ids=(0, 2, 7),
        predicate_class_ids=(0, 5),
        num_factor_types=3,
        active_factor_type_ids=ACTIVE_IDS,
        factor_executor_impl="per_type_grouped_v2",
        gold_edit_embedding_mode=gold_mode,
        pressure_enabled=True,
        pressure_module_sharing="per_type",
    )


def _factor_graph(factor_type: int) -> Data:
    graph = Data(
        x=torch.tensor([2, 5, 7, 9], dtype=torch.long),
        edge_index=torch.tensor(
            [[0, 1, 3, 3, 3], [1, 2, 1, 0, 2]],
            dtype=torch.long,
        ),
        edge_type=torch.tensor([0, 1, 4, 5, 6], dtype=torch.long),
        y=torch.tensor([[2, 5, 7, 0, 0, 0]], dtype=torch.long),
    )
    graph.batch = torch.zeros(4, dtype=torch.long)
    graph.factor_node_index = torch.tensor([3], dtype=torch.long)
    graph.is_factor_node = torch.tensor([False, False, False, True])
    graph.factor_constraint_ids = torch.tensor([100])
    graph.factor_types = torch.tensor([factor_type])
    graph.factor_checkable_pre = torch.tensor([True])
    graph.factor_satisfied_pre = torch.tensor([0])
    graph.factor_checkable_post_gold = torch.tensor([True])
    graph.factor_satisfied_post_gold = torch.tensor([1])
    graph.primary_factor_index = 0
    return graph


def _copy_executor(
    legacy: nn.ModuleList,
    grouped: GroupedFactorTypeExecutor,
) -> None:
    for type_index, source in enumerate(legacy):
        grouped.input_layer.weight.data[type_index].copy_(source.state_mlp[0].weight)
        grouped.input_layer.bias.data[type_index].copy_(source.state_mlp[0].bias)
        grouped.state_layer.weight.data[type_index].copy_(source.state_mlp[2].weight)
        grouped.state_layer.bias.data[type_index].copy_(source.state_mlp[2].bias)
        grouped.pre_head.weight.data[type_index].copy_(source.pre_head.weight)
        grouped.pre_head.bias.data[type_index].copy_(source.pre_head.bias)


def _copy_post_heads(
    legacy: nn.ModuleList,
    grouped: GroupedFactorPostEditHead,
) -> None:
    for type_index, source in enumerate(legacy):
        grouped.hidden.weight.data[type_index].copy_(source.net[0].weight)
        grouped.hidden.bias.data[type_index].copy_(source.net[0].bias)
        grouped.output.weight.data[type_index].copy_(source.net[2].weight)
        grouped.output.bias.data[type_index].copy_(source.net[2].bias)


def _copy_pressure(
    legacy: nn.ModuleList,
    grouped: GroupedPressureRole,
) -> None:
    for type_index, source in enumerate(legacy):
        grouped.hidden.weight.data[type_index].copy_(source[0].weight)
        grouped.hidden.bias.data[type_index].copy_(source[0].bias)
        grouped.output.weight.data[type_index].copy_(source[2].weight)
        grouped.output.bias.data[type_index].copy_(source[2].bias)


def _write_factor_types(path: Path, rows: list[list[int]]) -> None:
    pq.write_table(pa.table({"factor_types": pa.array(rows)}), path)


def test_factor_type_scan_reads_nested_parquet_lists(tmp_path: Path) -> None:
    path = tmp_path / "df_train.parquet"
    _write_factor_types(path, [[2, 0], [2], [], [5, 0]])
    assert scan_factor_type_ids([path]) == (0, 2, 5)


def test_model_config_validates_explicit_active_mapping() -> None:
    cfg = ModelConfig.from_mapping(
        {
            "num_factor_types": 3,
            "active_factor_type_ids": [0, 2],
            "factor_executor_impl": "per_type_grouped_v2",
            "gold_edit_embedding_mode": "compact",
        }
    )
    assert cfg.active_factor_type_ids == ACTIVE_IDS
    with pytest.raises(ValueError, match="strictly increasing"):
        ModelConfig.from_mapping(
            {"num_factor_types": 3, "active_factor_type_ids": [2, 0]}
        )
    with pytest.raises(ValueError, match="address space"):
        ModelConfig.from_mapping(
            {"num_factor_types": 3, "active_factor_type_ids": [0, 3]}
        )
    with pytest.raises(ValueError, match="must be explicit"):
        ModelConfig.from_mapping(
            {"num_factor_types": 3, "factor_executor_impl": "per_type_grouped_v2"}
        )


def test_compact_model_allocates_and_routes_only_active_types() -> None:
    model = _compact_model()
    assert model.factor_type_ids_compact_to_stable.tolist() == [0, 2]
    assert model.factor_type_id_to_compact.tolist() == [0, -1, 1]
    assert model._num_factor_executor_modules == 2
    assert model.gold_edit_class_ids.tolist() == [0, 2, 5, 7]
    assert model._gold_edit_embeddings.num_embeddings == 4
    assert isinstance(model._pressure_role_modules["0"], GroupedPressureRole)

    outputs = model(_factor_graph(2))
    assert outputs["factor_logits_pre"] is not None
    assert outputs["factor_logits_post_gold"] is not None
    loss = (
        outputs["edit_logits"].sum()
        + outputs["factor_logits_pre"].sum()
        + outputs["factor_logits_post_gold"].sum()
    )
    loss.backward()
    assert model._factor_executors.input_layer.weight.grad is not None
    assert model._gold_edit_embeddings.weight.grad is not None
    assert model._pressure_role_modules["0"].hidden.weight.grad is not None
    with pytest.raises(ValueError, match="absent from active_factor_type_ids"):
        model(_factor_graph(1))


def test_build_model_forwards_compact_configuration() -> None:
    config = ModelConfig(
        num_embedding_size=8,
        num_layers=2,
        hidden_channels=8,
        head_hidden=8,
        dropout=0.0,
        use_edge_attributes=False,
        entity_class_ids=(0, 2, 7),
        predicate_class_ids=(0, 5),
        num_factor_types=3,
        active_factor_type_ids=ACTIVE_IDS,
        factor_executor_impl="per_type_grouped_v2",
        gold_edit_embedding_mode="compact",
        pressure_enabled=True,
    )
    model = build_model("GIN_PRESSURE", 16, config)
    assert model.factor_type_ids_compact_to_stable.tolist() == [0, 2]
    assert model._gold_edit_embeddings.num_embeddings == 4


def test_grouped_executor_post_and_pressure_match_independent_modules_cpu() -> None:
    torch.manual_seed(7)
    input_dim = 11
    state_dim = 8
    compact_types = torch.tensor([1, 0, 1, 1, 0], dtype=torch.long)
    dispatch = build_grouped_dispatch(compact_types, num_types=len(ACTIVE_IDS))
    inputs = torch.randn(compact_types.numel(), input_dim)

    legacy_executors = nn.ModuleList(
        [FactorTypeExecutor(input_dim, state_dim) for _ in ACTIVE_IDS]
    )
    grouped_executor = GroupedFactorTypeExecutor(
        len(ACTIVE_IDS), input_dim, state_dim
    )
    _copy_executor(legacy_executors, grouped_executor)
    expected_states = torch.empty(compact_types.numel(), state_dim)
    expected_logits = torch.empty(compact_types.numel())
    for type_index, executor in enumerate(legacy_executors):
        mask = compact_types == type_index
        state, logit = executor(inputs[mask])
        expected_states[mask] = state
        expected_logits[mask] = logit
    actual_states, actual_logits = grouped_executor(inputs, dispatch)
    torch.testing.assert_close(actual_states, expected_states)
    torch.testing.assert_close(actual_logits, expected_logits)

    legacy_post = nn.ModuleList(
        [FactorPostEditHead(state_dim, state_dim) for _ in ACTIVE_IDS]
    )
    grouped_post = GroupedFactorPostEditHead(
        len(ACTIVE_IDS), state_dim, state_dim
    )
    _copy_post_heads(legacy_post, grouped_post)
    edits = torch.randn_like(actual_states)
    expected_post = torch.empty(compact_types.numel())
    for type_index, head in enumerate(legacy_post):
        mask = compact_types == type_index
        expected_post[mask] = head(actual_states[mask], edits[mask])
    torch.testing.assert_close(
        grouped_post(actual_states, edits, dispatch),
        expected_post,
    )

    pressure_input_dim = state_dim * 2
    legacy_pressure = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(pressure_input_dim, state_dim),
                nn.ReLU(),
                nn.Linear(state_dim, state_dim),
            )
            for _ in ACTIVE_IDS
        ]
    )
    grouped_pressure = GroupedPressureRole(
        len(ACTIVE_IDS), pressure_input_dim, state_dim
    )
    _copy_pressure(legacy_pressure, grouped_pressure)
    pressure_inputs = torch.randn(compact_types.numel(), pressure_input_dim)
    expected_pressure = torch.empty(compact_types.numel(), state_dim)
    for type_index, pressure in enumerate(legacy_pressure):
        mask = compact_types == type_index
        expected_pressure[mask] = pressure(pressure_inputs[mask])
    torch.testing.assert_close(
        grouped_pressure(pressure_inputs, dispatch),
        expected_pressure,
    )


def test_compact_gold_embedding_matches_full_table_for_reachable_ids() -> None:
    torch.manual_seed(11)
    full = _compact_model(gold_mode="full")
    compact = _compact_model(gold_mode="compact")
    compact._gold_edit_embeddings.weight.data.copy_(
        full._gold_edit_embeddings.weight.data.index_select(
            0, compact.gold_edit_class_ids
        )
    )
    targets = torch.tensor(
        [[0, 5, 7, 2, 0, 7], [2, 0, 0, 7, 5, 2]],
        dtype=torch.long,
    )
    graph_index = torch.tensor([0, 1, 1], dtype=torch.long)
    expected = full._gold_edit_representation(targets, graph_index)
    actual = compact._gold_edit_representation(targets, graph_index)
    assert expected is not None and actual is not None
    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="compact target vocabulary"):
        compact._gold_edit_representation(
            torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )


def test_evaluation_skips_post_gold_for_test_only_target_ids() -> None:
    model = _compact_model()
    graph = _factor_graph(2)
    graph.y = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="compact target vocabulary"):
        model(graph)

    outputs = model.forward_for_evaluation(graph)
    assert outputs["edit_logits"].shape == (1, 6, model.num_target_ids)
    assert outputs["factor_logits_pre"] is not None
    assert outputs["factor_logits_post_gold"] is None


def test_legacy_v1_state_dict_remains_strictly_loadable() -> None:
    kwargs = {
        "num_input_graph_nodes": 16,
        "num_embedding_size": 8,
        "num_layers": 2,
        "hidden": 8,
        "head_hidden": 8,
        "dropout": 0.0,
        "use_node_embeddings": True,
        "use_edge_attributes": False,
        "entity_class_ids": (0, 2, 7),
        "predicate_class_ids": (0, 5),
        "num_factor_types": 3,
        "factor_executor_impl": "per_type_v1",
        "gold_edit_embedding_mode": "full",
        "pressure_enabled": True,
    }
    original = RepairGINFactorPressure(**kwargs)
    restored = RepairGINFactorPressure(**kwargs)
    restored.load_state_dict(original.state_dict(), strict=True)
    assert not any(
        key.startswith(("factor_type_id_to_compact", "gold_edit_class_ids"))
        for key in original.state_dict()
    )


def test_config_generator_is_non_overwriting_and_keeps_training_config(
    tmp_path: Path,
) -> None:
    generator = _load_config_generator()
    source = tmp_path / "source.json"
    output = tmp_path / "run" / "config.json"
    interim = tmp_path / "interim"
    interim.mkdir()
    source_payload = {
        "model_config": {
            "dataset_variant": "full_strat1m_minocc100",
            "encoding": "node_id",
            "constraint_representation": "factorized",
            "model": "GIN_PRESSURE",
            "num_factor_types": 5,
            "factor_executor_impl": "per_type_v1",
            "pressure_module_sharing": "per_type",
        },
        "training_config": {"num_epochs": 10, "sentinel": {"unchanged": True}},
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    _write_factor_types(interim / "df_train.parquet", [[3], [0, 3]])
    _write_factor_types(interim / "df_val.parquet", [[0]])
    _write_factor_types(interim / "df_test.parquet", [[3, 0]])

    active_ids = generator.write_compact_a1_config(source, output, interim)
    assert active_ids == (0, 3)
    generated = json.loads(output.read_text(encoding="utf-8"))
    assert generated["training_config"] == source_payload["training_config"]
    assert generated["model_config"]["active_factor_type_ids"] == [0, 3]
    assert generated["model_config"]["factor_executor_impl"] == "per_type_grouped_v2"
    assert generated["model_config"]["gold_edit_embedding_mode"] == "compact"
    assert json.loads(source.read_text(encoding="utf-8")) == source_payload

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generator.write_compact_a1_config(source, output, interim)


def test_config_generator_rejects_test_only_factor_type(tmp_path: Path) -> None:
    generator = _load_config_generator()
    source = tmp_path / "source.json"
    interim = tmp_path / "interim"
    interim.mkdir()
    source.write_text(
        json.dumps(
            {
                "model_config": {
                    "dataset_variant": "full_strat1m_minocc100",
                    "encoding": "node_id",
                    "constraint_representation": "factorized",
                    "model": "GIN_PRESSURE",
                    "num_factor_types": 5,
                },
                "training_config": {},
            }
        ),
        encoding="utf-8",
    )
    _write_factor_types(interim / "df_train.parquet", [[0]])
    _write_factor_types(interim / "df_val.parquet", [[0]])
    _write_factor_types(interim / "df_test.parquet", [[4]])

    with pytest.raises(ValueError, match="absent from train/validation"):
        generator.build_compact_a1_config(source, interim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_executor_uses_bf16_grouped_mm_on_sm80_cuda() -> None:
    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip("grouped_mm requires SM80 or newer")
    executor = GroupedFactorTypeExecutor(2, 13, 8).cuda()
    inputs = torch.randn(9, 13, device="cuda")
    compact_types = torch.tensor([1, 0, 1, 0, 0, 1, 1, 1, 0], device="cuda")
    dispatch = build_grouped_dispatch(compact_types, num_types=2)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        states, logits = executor(inputs, dispatch)
        loss = states.float().square().mean() + logits.float().square().mean()
    loss.backward()
    assert executor.last_backend == "grouped_mm_bf16"
    assert executor.input_layer.weight.grad is not None
