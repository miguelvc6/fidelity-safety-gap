import importlib.util
from pathlib import Path

import numpy as np
import pytest


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


def test_fidelity_counts_require_complete_exact_operations() -> None:
    module = _load_readiness_module()
    targets = np.array(
        [
            [1, 2, 3, 0, 0, 0],
            [0, 0, 0, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=np.int64,
    )
    predictions = np.array(
        [
            [1, 2, 3, 0, 0, 0],
            [0, 0, 0, 4, 99, 6],
            [7, 8, 0, 10, 11, 12],
        ],
        dtype=np.int64,
    )

    assert module._fidelity_counts(predictions, targets) == (2, 1, 2)


def test_table_row_guard_accepts_whitespace_but_rejects_metric_drift() -> None:
    module = _load_readiness_module()
    table = module._normalise_tex(r"A1  & 0.6890 & 0.6689 & 0.6788 \\")

    module._require_table_row(
        table,
        "tab:main-results",
        ["A1", "0.6890", "0.6689", "0.6788"],
    )
    with pytest.raises(ValueError, match="does not match artifacts"):
        module._require_table_row(
            table,
            "tab:main-results",
            ["A1", "0.6890", "0.6689", "0.6787"],
        )
