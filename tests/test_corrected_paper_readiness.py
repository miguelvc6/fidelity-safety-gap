import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_readiness_module():
    path = ROOT / "scripts" / "check_corrected_paper_readiness.py"
    spec = importlib.util.spec_from_file_location("corrected_paper_readiness_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selector_rerun_tolerance_accepts_ties_but_rejects_selector_drift() -> None:
    module = _load_readiness_module()
    expected = {"value": 0.6, "numerator": 60_000, "denominator": 100_000}
    tied_rerun = {"value": 0.6002, "numerator": 60_020, "denominator": 100_000}
    wrong_selector = {"value": 0.62, "numerator": 62_000, "denominator": 100_000}

    assert module._metric_within_selector_tolerance(expected, expected)
    assert module._metric_within_selector_tolerance(tied_rerun, expected)
    assert not module._metric_within_selector_tolerance(wrong_selector, expected)
