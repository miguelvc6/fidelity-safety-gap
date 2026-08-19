#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

from modules.model_store import sanitize_fragment

LOG_DIR = Path("logs")
RAW_LOG_DIR = LOG_DIR / "runs"
HISTORY_PATH = LOG_DIR / "scheduler_history.jsonl"
MODELS_ROOT = Path("models")
CONFIG_FILENAME = "config.json"
CHECKPOINT_FILENAME = "checkpoint.pth"
PAPER_RUN_DIRECTORIES = (
    "b0_eswc_reproduction__full_strat1m_minocc100__node_id",
    "a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id",
    "m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id",
    "m1d_safe_factor_direct_compact_grouped__full_strat1m_minocc100__node_id",
    "g0_globalfix_reference_v2__full_strat1m_minocc100__node_id",
)
PAPER_RETAINED_CHECKPOINT = (
    "a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id"
)


class ExperimentError(Exception):
    """Raised when an experiment cannot be prepared or executed."""


def configure_logging(verbose: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "scheduler.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def discover_model_directories(root: Path) -> List[Path]:
    configs = sorted(root.rglob(CONFIG_FILENAME))
    return sorted({cfg.parent for cfg in configs})


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "training_config" not in data or "model_config" not in data:
        raise FileNotFoundError(f"Configuration at {path} missing required sections.")
    return data


def infer_model_field(config: dict[str, Any], field: str) -> str:
    value = config.get("model_config", {}).get(field)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Configuration missing required field '{field}'.")


def sanitize_run_name(model_dir: Path) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{sanitize_fragment(model_dir.name)}"


def write_history(record: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True))
        fh.write("\n")


def run_command(cmd: List[str], log_path: Path) -> int:
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("runner")
    logger.debug("Executing command: %s", " ".join(cmd))
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("# Command: " + " ".join(cmd) + "\n")
        log_file.write("# Started: " + datetime.now(UTC).isoformat() + "Z\n")
        log_file.flush()
        env = os.environ.copy()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # capture logs
            stderr=None,  # inherit parent's stderr -> shows bars in terminal
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            timestamp = datetime.now(UTC).isoformat() + "Z"
            log_file.write(f"{timestamp} {line}\n")
        return_code = process.wait()
        elapsed = time.monotonic() - start
        log_file.write(f"# Finished: {datetime.now(UTC).isoformat()}Z\n")
        log_file.write(f"# DurationSeconds: {elapsed:.2f}\n")
    logger.debug("Command finished with return code %s", return_code)
    return return_code


def run_evaluation(
    run_directory: Path,
    run_name: str,
    logger: logging.Logger,
    *,
    extra_flags: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Execute the evaluation script for the completed run."""
    eval_log_path = RAW_LOG_DIR / f"{run_name}_eval.log"
    command = [
        sys.executable,
        "src/09_eval.py",
        "--run-directory",
        str(run_directory),
    ]
    if extra_flags:
        command.extend(list(extra_flags))
    logger.info("Launching evaluation for run directory %s", run_directory)
    return_code = run_command(command, eval_log_path)
    if return_code == 0:
        logger.info("Evaluation completed for %s", run_directory.name)
    else:
        logger.error("Evaluation failed for %s with return code %s", run_directory.name, return_code)
    return {
        "status": "completed" if return_code == 0 else "failed",
        "command": command,
        "log_file": str(eval_log_path),
        "return_code": return_code,
    }


def ensure_reranker_predictions(
    experiment_config: Path,
    resolved_run_dir: Path,
    run_name: str,
    logger: logging.Logger,
) -> bool:
    pred_path = resolved_run_dir / "reranker_predictions.json"
    if pred_path.exists():
        return True
    predict_log_path = RAW_LOG_DIR / f"{run_name}_predict.log"
    command = [
        sys.executable,
        "src/08_train_reranker.py",
        "--experiment-config",
        str(experiment_config),
        "--predict-only",
    ]
    logger.info("Generating reranker predictions: %s", " ".join(command))
    return_code = run_command(command, predict_log_path)
    if return_code != 0:
        logger.error("Reranker prediction generation failed with return code %s", return_code)
        return False
    if not pred_path.exists():
        logger.error("Reranker prediction generation finished but %s is missing", pred_path)
        return False
    return True


def _infer_experiment_kind(config: dict[str, Any]) -> str:
    model_name = str(config.get("model_config", {}).get("model", "")).upper()
    if "reranker_config" in config or "proposal_config" in config or model_name == "RERANKER":
        return "reranker"
    return "proposal"


def _build_train_command(kind: str, experiment_config: Path) -> List[str]:
    if kind == "reranker":
        return [sys.executable, "src/08_train_reranker.py", "--experiment-config", str(experiment_config)]
    return [sys.executable, "src/07_train.py", "--experiment-config", str(experiment_config)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule training/evaluation across experiment configs.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional substring filter to run only matching experiment directories.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running remaining experiments after failures.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--eval-global-metrics",
        action="store_true",
        help="Enable corrected symbolic paper metrics during evaluation.",
    )
    parser.add_argument(
        "--eval-per-constraint-csv",
        action="store_true",
        help="Write per-constraint evaluation CSV.",
    )
    parser.add_argument(
        "--paper-suite",
        action="store_true",
        help="Run only the five learned paper systems with strict paper evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected run directories and their next command without writing logs or running them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    def _order_key(model_dir: Path) -> tuple[int, str]:
        config_path = model_dir / CONFIG_FILENAME
        if not config_path.exists():
            return (2, model_dir.name)
        try:
            cfg = load_config(config_path)
        except Exception:
            return (2, model_dir.name)
        kind = _infer_experiment_kind(cfg)
        priority = 1 if kind == "reranker" else 0
        return (priority, model_dir.name)

    if args.paper_suite:
        model_dirs = [MODELS_ROOT / name for name in PAPER_RUN_DIRECTORIES]
        missing = [
            path / CONFIG_FILENAME
            for path in model_dirs
            if not (path / CONFIG_FILENAME).exists()
        ]
        if missing:
            print("Missing paper configuration(s):", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            return 2
    else:
        model_dirs = sorted(discover_model_directories(MODELS_ROOT), key=_order_key)
    if args.only:
        model_dirs = [path for path in model_dirs if args.only in path.name]

    if args.dry_run:
        for model_dir in model_dirs:
            config_path = model_dir / CONFIG_FILENAME
            cfg = load_config(config_path)
            kind = _infer_experiment_kind(cfg)
            checkpoint_path = model_dir / CHECKPOINT_FILENAME
            if checkpoint_path.exists():
                command = [
                    sys.executable,
                    "src/09_eval.py",
                    "--run-directory",
                    str(model_dir),
                ]
                action = "evaluate"
            elif args.paper_suite and model_dir.name == PAPER_RETAINED_CHECKPOINT:
                command = ["restore", str(checkpoint_path)]
                action = "restore"
            else:
                command = _build_train_command(kind, config_path)
                action = "train"
            print(f"{model_dir.name}\t{action}\t{' '.join(command)}")
        return 0

    configure_logging(verbose=args.verbose)
    logger = logging.getLogger("scheduler")
    logger.info("Discovered %s model directories", len(model_dirs))

    if args.paper_suite:
        retained_checkpoint = MODELS_ROOT / PAPER_RETAINED_CHECKPOINT / CHECKPOINT_FILENAME
        if not retained_checkpoint.exists():
            logger.error(
                "Paper suite requires the retained Direct-Factor checkpoint at %s",
                retained_checkpoint,
            )
            return 2
    planned: list[str] = []
    for model_dir in model_dirs:
        config_path = model_dir / CONFIG_FILENAME
        if not config_path.exists():
            continue
        try:
            cfg = load_config(config_path)
        except Exception:
            continue
        if cfg.get("disabled") is True:
            continue
        planned.append(model_dir.name)
    if planned:
        logger.info("Planned runs (first %s): %s", min(5, len(planned)), ", ".join(planned[:5]))

    for index, model_dir in enumerate(model_dirs, start=1):
        config_path = model_dir / CONFIG_FILENAME
        checkpoint_path = model_dir / CHECKPOINT_FILENAME

        logger.info("[%s/%s] Processing %s", index, len(model_dirs), model_dir.name)

        if checkpoint_path.exists():
            logger.info("Checkpoint already exists at %s; skipping", checkpoint_path)
            record = {
                "model_dir": str(model_dir),
                "config": str(config_path),
                "status": "skipped",
                "reason": "checkpoint_exists",
                "timestamp": datetime.now(UTC).isoformat() + "Z",
            }
            if config_path.exists():
                try:
                    cfg = load_config(config_path)
                except Exception as exc:
                    logger.error("Skipping eval for %s: %s", model_dir, exc)
                    write_history(record)
                    continue
                kind = _infer_experiment_kind(cfg)
                if kind == "reranker":
                    try:
                        dataset_variant = infer_model_field(cfg, "dataset_variant")
                        encoding = infer_model_field(cfg, "encoding")
                    except Exception as exc:
                        logger.error("Skipping eval for %s: %s", model_dir, exc)
                        write_history(record)
                        continue
                    resolved_run_dir = model_dir
                    run_name = sanitize_run_name(model_dir)
                    run_prefix = f"{sanitize_fragment(dataset_variant)}-{sanitize_fragment(encoding)}_RERANKER"
                    if model_dir.name.startswith(run_prefix):
                        write_history(record)
                        continue
                    eval_flags: list[str] = []
                    if args.eval_global_metrics:
                        eval_flags.append("--global-metrics")
                    if args.eval_per_constraint_csv or "--per-constraint-csv" not in eval_flags:
                        eval_flags.append("--per-constraint-csv")
                    if args.paper_suite and "--strict-global-metrics" not in eval_flags:
                        eval_flags.append("--strict-global-metrics")
                    if ensure_reranker_predictions(config_path, resolved_run_dir, run_name, logger):
                        reranker_predictions = resolved_run_dir / "reranker_predictions.json"
                        evaluation = run_evaluation(
                            resolved_run_dir,
                            run_name,
                            logger,
                            extra_flags=eval_flags
                            + ["--legacy-predictions-json", str(reranker_predictions)],
                        )
                        record["evaluation"] = evaluation
                        if evaluation.get("return_code") not in (0, None) and not args.keep_going:
                            write_history(record)
                            logger.error(
                                "Stopping scheduler after eval failure (use --keep-going to continue)."
                            )
                            return 1
                else:
                    eval_flags: list[str] = []
                    if args.eval_global_metrics:
                        eval_flags.append("--global-metrics")
                    if args.eval_per_constraint_csv or "--per-constraint-csv" not in eval_flags:
                        eval_flags.append("--per-constraint-csv")
                    if args.paper_suite and "--strict-global-metrics" not in eval_flags:
                        eval_flags.append("--strict-global-metrics")
                    chooser_enabled = bool(
                        cfg.get("training_config", {}).get("chooser", {}).get("enabled", False)
                    )
                    policy_enabled = bool(cfg.get("model_config", {}).get("enable_policy_choice", False))
                    if chooser_enabled:
                        eval_flags.append("--use-chooser")
                    if policy_enabled:
                        eval_flags.append("--use-policy-choice")
                    evaluation = run_evaluation(
                        model_dir,
                        sanitize_run_name(model_dir),
                        logger,
                        extra_flags=eval_flags,
                    )
                    record["evaluation"] = evaluation
                    if evaluation.get("return_code") not in (0, None) and not args.keep_going:
                        write_history(record)
                        logger.error("Stopping scheduler after eval failure (use --keep-going to continue).")
                        return 1
            write_history(record)
            continue

        try:
            cfg = load_config(config_path)
        except Exception as exc:
            logger.error("Skipping %s: %s", model_dir, exc)
            write_history(
                {
                    "model_dir": str(model_dir),
                    "config": str(config_path),
                    "status": "failed",
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat() + "Z",
                }
            )
            continue

        if cfg.get("disabled") is True:
            logger.info("Skipping %s (disabled in config)", model_dir.name)
            write_history(
                {
                    "model_dir": str(model_dir),
                    "config": str(config_path),
                    "status": "skipped",
                    "reason": "disabled",
                    "timestamp": datetime.now(UTC).isoformat() + "Z",
                }
            )
            continue

        try:
            dataset_variant = infer_model_field(cfg, "dataset_variant")
            encoding = infer_model_field(cfg, "encoding")
        except Exception as exc:
            logger.error("Skipping %s: %s", model_dir, exc)
            write_history(
                {
                    "model_dir": str(model_dir),
                    "config": str(config_path),
                    "status": "failed",
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat() + "Z",
                }
            )
            continue

        kind = _infer_experiment_kind(cfg)
        if kind == "reranker":
            model_name = "RERANKER"
        else:
            try:
                model_name = infer_model_field(cfg, "model")
            except Exception as exc:
                logger.error("Skipping %s: %s", model_dir, exc)
                write_history(
                    {
                        "model_dir": str(model_dir),
                        "config": str(config_path),
                        "status": "failed",
                        "error": str(exc),
                        "timestamp": datetime.now(UTC).isoformat() + "Z",
                    }
                )
                continue

        resolved_run_dir = model_dir

        run_name = sanitize_run_name(model_dir)
        raw_log_path = RAW_LOG_DIR / f"{run_name}.log"
        command = _build_train_command(kind, config_path)

        logger.info(
            "Launching training | experiment=%s kind=%s model=%s dataset=%s encoding=%s",
            model_dir.name,
            kind,
            model_name,
            dataset_variant,
            encoding,
        )
        logger.info("Run dir: %s", resolved_run_dir)
        logger.info("Train command: %s", " ".join(command))

        start_ts = datetime.now(UTC)
        return_code = run_command(command, raw_log_path)
        end_ts = datetime.now(UTC)

        record: dict[str, Any] = {
            "model_dir": str(model_dir),
            "config": str(config_path),
            "command": command,
            "log_file": str(raw_log_path),
            "started_at": start_ts.isoformat() + "Z",
            "finished_at": end_ts.isoformat() + "Z",
        }

        if return_code == 0:
            logger.info("Training completed for %s", model_dir.name)
            record.update(
                {
                    "status": "completed",
                    "run_dir": str(resolved_run_dir),
                    "kind": kind,
                }
            )
            eval_flags: list[str] = []
            if args.eval_global_metrics:
                eval_flags.append("--global-metrics")
            if args.eval_per_constraint_csv or "--per-constraint-csv" not in eval_flags:
                eval_flags.append("--per-constraint-csv")
            if args.paper_suite and "--strict-global-metrics" not in eval_flags:
                eval_flags.append("--strict-global-metrics")
            chooser_enabled = bool(
                cfg.get("training_config", {}).get("chooser", {}).get("enabled", False)
            )
            policy_enabled = bool(cfg.get("model_config", {}).get("enable_policy_choice", False))

            if kind == "proposal":
                if chooser_enabled:
                    eval_flags.append("--use-chooser")
                if policy_enabled:
                    eval_flags.append("--use-policy-choice")
                eval_command = [
                    sys.executable,
                    "src/09_eval.py",
                    "--run-directory",
                    str(resolved_run_dir),
                ] + eval_flags
                logger.info("Eval command: %s", " ".join(eval_command))
                evaluation = run_evaluation(resolved_run_dir, run_name, logger, extra_flags=eval_flags)
                record["evaluation"] = evaluation
                if evaluation.get("return_code") not in (0, None) and not args.keep_going:
                    write_history(record)
                    logger.error("Stopping scheduler after eval failure (use --keep-going to continue).")
                    return 1
            else:
                if ensure_reranker_predictions(config_path, resolved_run_dir, run_name, logger):
                    reranker_predictions = resolved_run_dir / "reranker_predictions.json"
                    eval_command = [
                        sys.executable,
                        "src/09_eval.py",
                        "--run-directory",
                        str(resolved_run_dir),
                        "--legacy-predictions-json",
                        str(reranker_predictions),
                    ] + eval_flags
                    logger.info("Eval command: %s", " ".join(eval_command))
                    evaluation = run_evaluation(
                        resolved_run_dir,
                        run_name,
                        logger,
                        extra_flags=eval_flags
                        + ["--legacy-predictions-json", str(reranker_predictions)],
                    )
                    record["evaluation"] = evaluation
                    if evaluation.get("return_code") not in (0, None) and not args.keep_going:
                        write_history(record)
                        logger.error("Stopping scheduler after eval failure (use --keep-going to continue).")
                        return 1
                else:
                    logger.info("Eval command: (skipped for reranker)")
                    logger.warning(
                        "Skipping evaluation for reranker experiment %s (missing reranker_predictions.json).",
                        model_dir.name,
                    )
        else:
            logger.error("Training failed for %s with return code %s", model_dir.name, return_code)
            record.update(
                {
                    "status": "failed",
                    "error": f"Process exited with code {return_code}",
                    "kind": kind,
                }
            )
            if not args.keep_going:
                write_history(record)
                logger.error("Stopping scheduler after failure (use --keep-going to continue).")
                return 1

        write_history(record)

    logger.info("Scheduler run finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
