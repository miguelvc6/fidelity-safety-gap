from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.data_encoders import dataset_variant_name


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


EVAL_MODULE = _load_module(ROOT / "src" / "09_eval.py", "eval_09_for_test")
_resolve_baseline_interim_paths = EVAL_MODULE._resolve_baseline_interim_paths
load_baseline_split_from_parquet = EVAL_MODULE.load_baseline_split_from_parquet
_strict_global_requires_factor_fields = EVAL_MODULE._strict_global_requires_factor_fields
GlobalMetricsSupport = EVAL_MODULE.GlobalMetricsSupport


def test_baseline_resolution_prefers_labeled_and_loads_factor_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        variant = dataset_variant_name("full", 100)
        base_dir = root / "data" / "interim" / variant
        labeled_dir = root / "data" / "interim" / f"{variant}_labeled"
        base_dir.mkdir(parents=True, exist_ok=True)
        labeled_dir.mkdir(parents=True, exist_ok=True)

        row = {
            "add_subject": 1,
            "add_predicate": 2,
            "add_object": 3,
            "del_subject": 4,
            "del_predicate": 5,
            "del_object": 6,
            "subject": 7,
            "predicate": 8,
            "object": 9,
            "constraint_id": 17,
            "constraint_type": "single_value",
            "factor_constraint_ids": [17, 21],
            "factor_types": [0, 3],
            "factor_checkable_pre": [True, True],
            "factor_satisfied_pre": [0, 1],
            "factor_checkable_post_gold": [True, False],
            "factor_satisfied_post_gold": [1, 0],
            "primary_factor_index": 0,
        }
        pd.DataFrame([row]).to_parquet(labeled_dir / "df_train.parquet", index=False)
        pd.DataFrame([row]).to_parquet(labeled_dir / "df_test.parquet", index=False)

        _write_text(base_dir / "globalintencoder.txt", "subject\t1\n")
        _write_text(labeled_dir / "globalintencoder.txt", "subject\t11\n")

        with _pushd(root):
            data_path, encoder_path = _resolve_baseline_interim_paths("full", 100)
            assert data_path.resolve() == labeled_dir.resolve()
            assert encoder_path.resolve() == (labeled_dir / "globalintencoder.txt").resolve()

            graphs, max_index = load_baseline_split_from_parquet(data_path, "test")

        assert len(graphs) == 1
        graph = graphs[0]
        assert graph.factor_constraint_ids.tolist() == [17, 21]
        assert graph.factor_types.tolist() == [0, 3]
        assert graph.factor_checkable_pre.tolist() == [True, True]
        assert graph.factor_satisfied_pre.tolist() == [0, 1]
        assert graph.factor_checkable_post_gold.tolist() == [True, False]
        assert graph.factor_satisfied_post_gold.tolist() == [1, 0]
        assert graph.primary_factor_index == 0
        assert max_index == 9


def test_baseline_resolution_falls_back_to_unlabeled() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        variant = dataset_variant_name("full", 100)
        base_dir = root / "data" / "interim" / variant
        base_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "add_subject": 0,
                    "add_predicate": 0,
                    "add_object": 0,
                    "del_subject": 0,
                    "del_predicate": 0,
                    "del_object": 0,
                    "subject": 0,
                    "predicate": 0,
                    "object": 0,
                    "constraint_type": "conflict_with",
                }
            ]
        ).to_parquet(base_dir / "df_train.parquet", index=False)
        _write_text(base_dir / "globalintencoder.txt", "subject\t1\n")

        with _pushd(root):
            data_path, encoder_path = _resolve_baseline_interim_paths("full", 100)
            assert data_path.resolve() == base_dir.resolve()
            assert encoder_path.resolve() == (base_dir / "globalintencoder.txt").resolve()


def test_strict_global_factor_field_requirement_is_representation_aware() -> None:
    passive_cfg = EVAL_MODULE.ModelConfig.from_mapping({"constraint_representation": "eswc_passive"})
    factorized_cfg = EVAL_MODULE.ModelConfig.from_mapping({"constraint_representation": "factorized"})

    assert _strict_global_requires_factor_fields(passive_cfg) is False
    assert _strict_global_requires_factor_fields(factorized_cfg) is True


def test_global_support_ignores_passive_graph_factor_ids_without_labels() -> None:
    graph = Data(x=torch.zeros((1,), dtype=torch.long))
    graph.factor_constraint_ids = torch.tensor([123], dtype=torch.long)
    graph.primary_factor_index = 0

    calls: list[dict[str, object]] = []

    class DummyEvaluator:
        def evaluate_full(self, row, *, candidate_slots, primary_factor_index=None, factor_constraint_ids=None):
            calls.append(
                {
                    "primary_factor_index": primary_factor_index,
                    "factor_constraint_ids": factor_constraint_ids,
                }
            )
            return {
                "local_constraint_ids": [123, 456],
                "primary_factor_index": 0,
                "pre_checkable": [True, True],
                "pre_satisfied": [1, 1],
                "post_checkable": [True, True],
                "post_satisfied": [1, 0],
            }

    support = GlobalMetricsSupport(rows=[object()], evaluator=DummyEvaluator())
    postprocess, state = support.build_postprocess([graph])
    predictions = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)
    targets = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)

    postprocess(predictions, targets, ["single"])

    assert calls == [{"primary_factor_index": None, "factor_constraint_ids": None}]
    paper = state["paper_metrics"]
    assert paper["srr"]["denominator"] == 1
    assert paper["srr"]["numerator"] == 1


def test_make_experiment_configs_empty_processed_root_message() -> None:
    module = _load_module(ROOT / "scripts" / "make_experiment_configs.py", "make_experiment_configs_for_test")
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_root = Path(tmpdir) / "processed"
        processed_root.mkdir(parents=True, exist_ok=True)

        argv_backup = list(sys.argv)
        sys.argv = [
            "make_experiment_configs.py",
            "--processed-root",
            str(processed_root),
        ]
        try:
            try:
                module.main()
            except SystemExit as exc:
                message = str(exc)
            else:
                raise AssertionError("Expected SystemExit for empty processed root")
        finally:
            sys.argv = argv_backup

    assert "No graph artifacts found under" in message
    assert "src/02b_stratified_benchmark_sampler.py --source-dataset full --output-dataset full_strat1m --min-occurrence 100" in message
    assert "src/05_constraint_labeler.py --dataset full_strat1m --min-occurrence 100 --constraint-scope local --registry-dataset full --factor-family-policy supported_only" in message
    assert "src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-scope local --constraint-representation factorized --registry-dataset full" in message
    assert "src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation eswc_passive --registry-dataset full" in message


if __name__ == "__main__":
    test_baseline_resolution_prefers_labeled_and_loads_factor_fields()
    test_baseline_resolution_falls_back_to_unlabeled()
    test_strict_global_factor_field_requirement_is_representation_aware()
    test_global_support_ignores_passive_graph_factor_ids_without_labels()
    test_make_experiment_configs_empty_processed_root_message()
    print("paper run readiness tests passed")
