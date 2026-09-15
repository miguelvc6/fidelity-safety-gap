from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LABELER = _load_module(ROOT / "src" / "05_constraint_labeler.py", "labeler_05_for_policy_test")
GRAPH = _load_module(ROOT / "src" / "06_graph.py", "graph_06_for_policy_test")


def _registry_entry(family: str, supported: bool, type_index: int) -> object:
    return LABELER.RegistryEntry(
        constraint_type_raw=family,
        constraint_type_item=f"Q{type_index}",
        constraint_type_index=type_index,
        constraint_family=family,
        constraint_label=family,
        constraint_family_supported=supported,
        constrained_property_raw="",
        param_predicates_raw=(),
        param_objects_raw=(),
    )


def _row(constraint_id: int, local_constraint_ids: list[int]) -> dict[str, object]:
    return {
        "constraint_id": constraint_id,
        "constraint_type": "single",
        "subject": 10,
        "predicate": 20,
        "object": 30,
        "other_subject": 0,
        "other_predicate": 0,
        "other_object": 0,
        "add_subject": 0,
        "add_predicate": 0,
        "add_object": 0,
        "del_subject": 0,
        "del_predicate": 0,
        "del_object": 0,
        "subject_predicates": [20],
        "subject_objects": [30],
        "object_predicates": [],
        "object_objects": [],
        "other_entity_predicates": [],
        "other_entity_objects": [],
        "local_constraint_ids": local_constraint_ids,
        "local_constraint_ids_focus": local_constraint_ids,
    }


def _process(policy: str) -> tuple[pd.DataFrame, dict, object]:
    df = pd.DataFrame(
        [
            _row(1, [1, 2, 3]),
            _row(2, [2, 1]),
        ]
    )
    registry = {
        1: _registry_entry("single", True, 0),
        2: _registry_entry("unsupported:QX", False, 1),
        3: _registry_entry("distinct", True, 2),
    }
    return LABELER._process_dataframe(
        df,
        registry,
        encoder=None,
        assume_complete=True,
        use_encoded_ids=True,
        constraint_scope="local",
        factor_family_policy=policy,
    )


def test_supported_only_filters_unsupported_secondary_and_keeps_primary() -> None:
    labeled, _, stats = _process("supported_only")

    assert list(labeled["factor_constraint_ids"].iloc[0]) == [1, 3]
    assert list(labeled["factor_types"].iloc[0]) == [0, 2]
    assert len(labeled["factor_checkable_pre"].iloc[0]) == 2

    assert list(labeled["factor_constraint_ids"].iloc[1]) == [2, 1]
    assert list(labeled["factor_types"].iloc[1]) == [1, 0]
    assert list(labeled["factor_checkable_pre"].iloc[1])[0] is False

    assert stats["raw_factor_total"] == 5
    assert stats["retained_factor_total"] == 4
    assert stats["unsupported_filtered"] == 1
    assert stats["unsupported_primary_retained"] == 1


def test_all_factor_policy_preserves_local_constraint_ids() -> None:
    labeled, _, stats = _process("all")

    assert list(labeled["factor_constraint_ids"].iloc[0]) == [1, 2, 3]
    assert list(labeled["factor_types"].iloc[0]) == [0, 1, 2]
    assert stats["raw_factor_total"] == 5
    assert stats["retained_factor_total"] == 5
    assert stats["unsupported_filtered"] == 0


def test_graph_uses_filtered_factor_ids_and_registry_family() -> None:
    graph = {
        "constraint_id": 1,
        "local_constraint_ids": [1, 2, 3],
        "local_constraint_ids_focus": [1, 2],
        "factor_constraint_ids": [1, 3],
    }
    assert GRAPH._factor_ids_for_graph(graph, "local") == [1, 3]
    assert GRAPH._constraint_family_from_registry_entry(
        {
            "constraint_type": "<http://www.wikidata.org/entity/Q21510862>",
            "constraint_family": "symmetric",
        }
    ) == "symmetric"


if __name__ == "__main__":
    test_supported_only_filters_unsupported_secondary_and_keeps_primary()
    test_all_factor_policy_preserves_local_constraint_ids()
    test_graph_uses_filtered_factor_ids_and_registry_family()
    print("supported factor policy tests passed")
