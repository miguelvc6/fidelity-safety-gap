#!/usr/bin/env python3
"""Collect H2 factor semantic calibration metrics across runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize factor semantics from H2 outputs.")
    parser.add_argument(
        "--run-directory",
        action="append",
        required=True,
        help="Run directory containing evaluations/h2/factor_semantics.csv. May be repeated.",
    )
    parser.add_argument("--output-csv", required=True, help="Destination combined CSV.")
    return parser.parse_args()


def _float_or_blank(value: str | None) -> float | str:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def _imbalance_flag(positive_rate: Any) -> str:
    try:
        value = float(positive_rate)
    except (TypeError, ValueError):
        return ""
    return "true" if value < 0.05 or value > 0.95 else "false"


def _run_model_name(run_dir: Path) -> str:
    name = run_dir.name
    return name.split("__", 1)[0] if "__" in name else name


def _read_rows(run_dir: Path) -> list[dict[str, Any]]:
    source = run_dir / "evaluations" / "h2" / "factor_semantics.csv"
    if not source.exists():
        raise FileNotFoundError(f"H2 factor semantics file not found: {source}")
    with source.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            positive_rate = _float_or_blank(row.get("positive_rate"))
            rows.append(
                {
                    "run": run_dir.name,
                    "model": _run_model_name(run_dir),
                    "state": row.get("state", ""),
                    "factor_family": row.get("factor_family", row.get("constraint_type", "")),
                    "support": row.get("support", ""),
                    "positive_rate": positive_rate,
                    "accuracy": _float_or_blank(row.get("accuracy")),
                    "precision": _float_or_blank(row.get("precision")),
                    "recall": _float_or_blank(row.get("recall")),
                    "f1": _float_or_blank(row.get("f1")),
                    "auroc": _float_or_blank(row.get("auroc")),
                    "auprc": _float_or_blank(row.get("auprc")),
                    "ece": _float_or_blank(row.get("ece")),
                    "imbalance_flag": _imbalance_flag(positive_rate),
                }
            )
        return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output_csv)
    rows: list[dict[str, Any]] = []
    for entry in args.run_directory:
        rows.extend(_read_rows(Path(entry).resolve()))

    fieldnames = [
        "run",
        "model",
        "state",
        "factor_family",
        "support",
        "positive_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "ece",
        "imbalance_flag",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
