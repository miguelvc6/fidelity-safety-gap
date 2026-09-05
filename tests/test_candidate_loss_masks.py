from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from types import SimpleNamespace


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


def test_conditional_candidate_mean_is_stable_and_ignores_ineligible_logits() -> None:
    fn = _train_module()._conditional_candidate_mean
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.tensor([-80.0, 80.0, -79.0], dtype=dtype, requires_grad=True)
        values = torch.tensor([1.0, 99.0, 0.0], dtype=dtype)
        eligible = torch.tensor([True, False, True])
        loss = fn(logits, values, eligible)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad.float()).all()
        reference = float(loss.detach())

        changed = torch.tensor([-80.0, -8000.0, -79.0], dtype=dtype)
        assert float(fn(changed, values, eligible)) == reference


def test_conditional_candidate_mean_all_and_none_are_finite() -> None:
    fn = _train_module()._conditional_candidate_mean
    logits = torch.tensor([1000.0, -1000.0], requires_grad=True)
    values = torch.tensor([0.0, 1.0])
    all_loss = fn(logits, values, torch.tensor([True, True]))
    none_loss = fn(logits, values, torch.tensor([False, False]))
    total = all_loss + none_loss
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(logits.grad).all()


def test_proven_fix_penalty_counts_unknown_as_failure_on_previolated_row() -> None:
    fn = _train_module()._proven_fix_candidate_penalty
    metrics = [
        SimpleNamespace(primary_pre_violated=1, primary_checkable=1, primary_satisfied=1),
        SimpleNamespace(primary_pre_violated=1, primary_checkable=0, primary_satisfied=0),
    ]
    successful_high = fn(torch.tensor([5.0, 0.0]), metrics)
    unknown_high = fn(torch.tensor([0.0, 5.0]), metrics)
    assert float(unknown_high) > float(successful_high)


def test_proven_fix_penalty_is_zero_for_pre_unknown_or_satisfied() -> None:
    fn = _train_module()._proven_fix_candidate_penalty
    logits = torch.tensor([1000.0, -1000.0], requires_grad=True)
    metrics = [
        SimpleNamespace(primary_pre_violated=0, primary_checkable=1, primary_satisfied=1),
        SimpleNamespace(primary_pre_violated=0, primary_checkable=0, primary_satisfied=0),
    ]
    loss = fn(logits, metrics)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
