from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_scheduler():
    path = ROOT / "src" / "10_scheduler.py"
    spec = importlib.util.spec_from_file_location("paper_scheduler_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_suite_dry_run_is_exact_and_has_no_side_effects(
    tmp_path, monkeypatch, capsys
) -> None:
    scheduler = _load_scheduler()
    monkeypatch.setattr(scheduler, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(scheduler, "LOG_DIR", tmp_path / "logs")

    for name in scheduler.PAPER_RUN_DIRECTORIES:
        model = "RERANKER" if name.startswith("g0_") else "GIN"
        directory = scheduler.MODELS_ROOT / name
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "model_config": {"model": model},
                    "training_config": {},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(sys, "argv", ["10_scheduler.py", "--paper-suite", "--dry-run"])
    assert scheduler.main() == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split("\t", 1)[0] for line in lines] == list(
        scheduler.PAPER_RUN_DIRECTORIES
    )
    actions = {line.split("\t")[0]: line.split("\t")[1] for line in lines}
    assert actions[scheduler.PAPER_RETAINED_CHECKPOINT] == "restore"
    assert all(
        action == "train"
        for name, action in actions.items()
        if name != scheduler.PAPER_RETAINED_CHECKPOINT
    )
    assert not scheduler.LOG_DIR.exists()
