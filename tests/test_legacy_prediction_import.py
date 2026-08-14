from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.data_encoders import GlobalIntEncoder


def _load_eval_module():
    path = SRC / "09_eval.py"
    spec = importlib.util.spec_from_file_location("eval_09_legacy_import_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encoder_path(tmp_path: Path) -> Path:
    encoder = GlobalIntEncoder()
    encoder.encode("one")
    encoder.encode("two")
    path = tmp_path / "globalintencoder.txt"
    encoder.save(path)
    return path


def test_legacy_prediction_import_validates_count_and_class_range(tmp_path: Path) -> None:
    evaluation = _load_eval_module()
    encoder_path = _encoder_path(tmp_path)
    predictions = torch.tensor([[0, 1, 2, 0, 0, 0]], dtype=torch.long)

    validated = evaluation._validate_legacy_prediction_import(
        predictions,
        expected_count=1,
        encoder_path=encoder_path,
    )
    torch.testing.assert_close(validated, predictions)

    with pytest.raises(ValueError, match="count mismatch"):
        evaluation._validate_legacy_prediction_import(
            predictions,
            expected_count=2,
            encoder_path=encoder_path,
        )

    with pytest.raises(ValueError, match="outside the stored encoder"):
        evaluation._validate_legacy_prediction_import(
            torch.tensor([[0, 1, 99, 0, 0, 0]], dtype=torch.long),
            expected_count=1,
            encoder_path=encoder_path,
        )
