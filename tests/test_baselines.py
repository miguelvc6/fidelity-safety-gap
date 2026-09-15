from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.baselines import (  # noqa: E402
    BASELINE_NAMES,
    BaselineAdapter,
    ConstraintDefinitionMajorityBaseline,
    ConstraintFamilyMajorityBaseline,
    DeleteFocusBaseline,
)


def _graph(constraint_id: int, constraint_type: str, target: list[int]) -> Data:
    graph = Data(
        x=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor(target, dtype=torch.long).view(1, 6),
    )
    graph.shape_id = constraint_id
    graph.constraint_type = constraint_type
    return graph


def test_definition_and_family_majority_use_different_keys() -> None:
    train = [
        _graph(1, "single", [0, 0, 0, 11, 12, 13]),
        _graph(1, "single", [0, 0, 0, 11, 12, 13]),
        _graph(2, "single", [21, 22, 23, 0, 0, 0]),
    ]
    test_graph = _graph(2, "single", [0, 0, 0, 0, 0, 0])

    definition_majority = ConstraintDefinitionMajorityBaseline(num_graph_nodes=30)
    definition_majority.fit(train)

    family_majority = ConstraintFamilyMajorityBaseline(num_graph_nodes=30)
    family_majority.fit(train)

    assert definition_majority.predict_one(test_graph).tolist() == [21, 22, 23, 0, 0, 0]
    assert family_majority.predict_one(test_graph).tolist() == [0, 0, 0, 11, 12, 13]


def test_baseline_registry_exposes_explicit_majority_names() -> None:
    assert "ConstraintDefinitionMajorityBaseline" in BASELINE_NAMES
    assert "ConstraintFamilyMajorityBaseline" in BASELINE_NAMES
    assert "ConstraintShapeMajorityBaseline" not in BASELINE_NAMES


def test_baseline_adapter_evaluation_hook_uses_plain_forward() -> None:
    graph = _graph(1, "single", [0, 0, 0, 0, 0, 0])
    graph.focus_triple = torch.tensor([1, 2, 3], dtype=torch.long)
    batch = Batch.from_data_list([graph])
    adapter = BaselineAdapter(DeleteFocusBaseline(num_graph_nodes=30))

    logits = adapter.forward_for_evaluation(batch)

    assert logits.shape == (1, 6, 30)
    assert logits.argmax(dim=-1).tolist() == [[0, 0, 0, 1, 2, 3]]


if __name__ == "__main__":
    test_definition_and_family_majority_use_different_keys()
    test_baseline_registry_exposes_explicit_majority_names()
    test_baseline_adapter_evaluation_hook_uses_plain_forward()
    print("baseline tests passed")
