from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _train_module():
    path = Path(__file__).resolve().parents[1] / "src" / "07_train.py"
    spec = importlib.util.spec_from_file_location("train_candidate_mask_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_masked_candidate_mean_ignores_unknown_probability_mass() -> None:
    fn = _train_module()._masked_candidate_mean
    probs = torch.tensor([0.25, 0.75])
    penalties = torch.tensor([1.0, 1.0])
    eligible = torch.tensor([True, False])
    assert fn(probs, penalties, eligible).item() == 1.0


def test_masked_candidate_mean_is_zero_when_every_outcome_is_unknown() -> None:
    fn = _train_module()._masked_candidate_mean
    probs = torch.tensor([0.25, 0.75])
    penalties = torch.tensor([1.0, 1.0])
    eligible = torch.tensor([False, False])
    assert fn(probs, penalties, eligible).item() == 0.0
