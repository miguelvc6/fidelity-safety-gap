from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = ROOT / "scripts" / "check_deletion_shortcut_study.py"
    spec = importlib.util.spec_from_file_location("check_deletion_shortcut_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner():
    path = ROOT / "scripts" / "run_deletion_shortcut_study.py"
    spec = importlib.util.spec_from_file_location("run_deletion_shortcut_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _system(base: float, action: float, eppf: float) -> dict:
    return {
        "evaluation": {
            "paper_metrics": {
                "base_deletion_rate": {"value": base},
                "deletes_base_action_rate": {"value": action},
                "eppf": {"value": eppf},
            }
        }
    }


def test_mitigation_gate_requires_all_registered_directions() -> None:
    checker = _load_checker()
    results = {
        "control": _system(0.9, 0.8, 0.0),
        "mitigation": _system(0.2, 0.3, 0.4),
    }
    checker._assert_mitigation(results, "control", "mitigation")

    results["mitigation"] = _system(0.2, 0.9, 0.4)
    with pytest.raises(ValueError, match="deletes_base_action_rate"):
        checker._assert_mitigation(results, "control", "mitigation")


def test_metric_gate_rejects_legacy_and_inconsistent_fields() -> None:
    checker = _load_checker()
    with pytest.raises(ValueError, match="Legacy metric"):
        checker._assert_no_legacy_fields({"nested": {"gfr": 1.0}})

    payload = {
        "paper_metrics": {
            name: {"value": 0.5, "numerator": 1, "denominator": 2}
            for name in checker.METRICS
        }
    }
    checker._assert_metric_schema("toy", payload)
    payload["paper_metrics"]["pfr"]["value"] = 0.6
    with pytest.raises(ValueError, match="inconsistent pfr"):
        checker._assert_metric_schema("toy", payload)


def test_study_runner_has_an_exact_isolated_order() -> None:
    runner = _load_runner()
    assert [run.name for run in runner.RUNS] == ["M1D", "M1D-BP", "G0", "G0-BP"]
    assert [run.name for run in runner._ordered_runs(g0_first=False)] == [
        "M1D",
        "M1D-BP",
        "G0",
        "G0-BP",
    ]
    assert [run.name for run in runner._ordered_runs(g0_first=True)] == [
        "G0",
        "G0-BP",
        "M1D",
        "M1D-BP",
    ]
    assert len({run.directory for run in runner.RUNS}) == 4
    assert all(directory.endswith("_v2__full_strat1m_minocc100__node_id") for directory in (
        run.directory for run in runner.RUNS
    ))
