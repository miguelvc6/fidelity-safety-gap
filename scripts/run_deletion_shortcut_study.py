#!/usr/bin/env python3
"""Run and resume the four registered M1D/G0 deletion-shortcut experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
A1_RUN = ROOT / "models" / "a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id"
LOG_ROOT = ROOT / "logs" / "deletion_shortcut_v2"
STATUS_PATH = LOG_ROOT / "status.json"


@dataclass(frozen=True)
class StudyRun:
    name: str
    directory: str
    kind: str


RUNS = (
    StudyRun("M1D", "m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id", "proposal"),
    StudyRun(
        "M1D-BP",
        "m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id",
        "proposal",
    ),
    StudyRun("G0", "g0_globalfix_reference_v2__full_strat1m_minocc100__node_id", "reranker"),
    StudyRun(
        "G0-BP",
        "g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id",
        "reranker",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even when a versioned checkpoint already exists.",
    )
    parser.add_argument(
        "--only",
        choices=tuple(run.name for run in RUNS),
        default=None,
        help="Run one registered experiment instead of the complete sequential suite.",
    )
    return parser.parse_args()


def _write_status(payload: dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, STATUS_PATH)


def _run_step(run: StudyRun, step: str, command: Sequence[str]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{run.name.lower()}_{step}.log"
    started = datetime.now(UTC).isoformat()
    _write_status(
        {
            "status": "running",
            "system": run.name,
            "step": step,
            "command": list(command),
            "started_at": started,
            "log": str(log_path),
        }
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n# Started {started}\n# Command: {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(
            f"# Finished {datetime.now(UTC).isoformat()} return_code={completed.returncode}\n"
        )
    if completed.returncode != 0:
        _write_status(
            {
                "status": "failed",
                "system": run.name,
                "step": step,
                "return_code": completed.returncode,
                "log": str(log_path),
            }
        )
        raise subprocess.CalledProcessError(completed.returncode, command)


def _python(*arguments: str) -> list[str]:
    return [sys.executable, *arguments]


def _evaluate(run: StudyRun, run_directory: Path) -> None:
    evaluation_directory = run_directory / "evaluations"
    common = [
        "--strict-global-metrics",
        "--per-constraint-csv",
        "--batch-size",
        "256",
    ]
    if run.kind == "proposal":
        direct_command = _python(
            "src/09_eval.py",
            "--run-directory",
            str(run_directory),
            *common,
        )
    else:
        direct_command = _python(
            "src/09_eval.py",
            "--run-directory",
            str(run_directory),
            "--legacy-predictions-json",
            str(run_directory / "reranker_predictions.json"),
            *common,
        )
    _run_step(run, "evaluate_direct", direct_command)
    shutil.copy2(evaluation_directory / "model.json", evaluation_directory / "model.direct.json")
    _run_step(
        run,
        "evaluate_replay",
        _python(
            "src/09_eval.py",
            "--run-directory",
            str(run_directory),
            "--predictions",
            str(evaluation_directory / "predictions.parquet"),
            *common,
        ),
    )

    if run.kind == "proposal":
        _run_step(
            run,
            "h2",
            _python(
                "src/09_eval.py",
                "--run-directory",
                str(run_directory),
                "--strict-global-metrics",
                "--h2-eval",
                "--h2-batch-size",
                "256",
            ),
        )
        _run_step(
            run,
            "candidate_oracle",
            _python(
                "scripts/analyze_candidate_oracle.py",
                "--run-directory",
                str(run_directory),
                "--strict-global-metrics",
                "--batch-size",
                "256",
            ),
        )
    else:
        _run_step(
            run,
            "candidate_membership",
            _python(
                "scripts/audit_prediction_candidate_membership.py",
                "--run-directory",
                str(run_directory),
                "--proposal-run-directory",
                str(A1_RUN),
                "--predictions",
                str(run_directory / "reranker_predictions.json"),
                "--output",
                str(evaluation_directory / "candidate_membership_audit.json"),
                "--batch-size",
                "256",
            ),
        )
        _run_step(
            run,
            "deletion_degeneracy",
            _python(
                "scripts/analyze_deletion_degeneracy.py",
                "--g0-run-directory",
                str(run_directory),
                "--predictions",
                str(evaluation_directory / "predictions.parquet"),
                "--strict-global-metrics",
            ),
        )


def main() -> None:
    args = parse_args()
    selected = [run for run in RUNS if args.only is None or run.name == args.only]
    for run in selected:
        run_directory = ROOT / "models" / run.directory
        config = run_directory / "config.json"
        checkpoint = run_directory / "checkpoint.pth"
        if args.force_train or not checkpoint.exists():
            training_script = "src/07_train.py" if run.kind == "proposal" else "src/08_train_reranker.py"
            command = _python(training_script, "--experiment-config", str(config))
            if run.kind == "reranker":
                command.extend(("--prediction-batch-size", "256", "--seed", "42"))
            _run_step(run, "train", command)
        elif run.kind == "reranker" and not (run_directory / "reranker_predictions.json").exists():
            _run_step(
                run,
                "predict",
                _python(
                    "src/08_train_reranker.py",
                    "--experiment-config",
                    str(config),
                    "--predict-only",
                    "--prediction-batch-size",
                    "256",
                    "--seed",
                    "42",
                ),
            )
        _evaluate(run, run_directory)

    if args.only is None:
        gate_run = StudyRun("gate", "paper_diagnostics", "gate")
        _run_step(gate_run, "readiness", _python("scripts/check_deletion_shortcut_study.py"))
        _write_status(
            {
                "status": "ready",
                "seed": 42,
                "completed_at": datetime.now(UTC).isoformat(),
                "readiness": str(
                    ROOT / "models" / "paper_diagnostics" / "deletion_shortcut_study_readiness.json"
                ),
            }
        )


if __name__ == "__main__":
    main()
