#!/usr/bin/env python3
"""Write an immutable checksum inventory for run directories before additive experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.evaluation_artifacts import atomic_write_json, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    for run_directory in args.run_directory:
        run = run_directory.resolve()
        if not run.is_dir():
            raise FileNotFoundError(run)
        artifacts = []
        for path in sorted(item for item in run.rglob("*") if item.is_file()):
            stat = path.stat()
            artifacts.append(
                {
                    "relative_path": str(path.relative_to(run)),
                    "size_bytes": stat.st_size,
                    "sha256": sha256_file(path),
                }
            )
        runs.append(
            {
                "run_directory": str(run),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            }
        )
    atomic_write_json(
        args.output.resolve(),
        {
            "schema_version": 1,
            "purpose": "pre-m1d-g0-stability-v2 rollback inventory",
            "runs": runs,
        },
    )
    print(f"[ok] wrote artifact inventory to {args.output.resolve()}")


if __name__ == "__main__":
    main()
