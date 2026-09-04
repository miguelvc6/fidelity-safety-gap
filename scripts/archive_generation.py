#!/usr/bin/env python3
"""Checksummed, recoverable archival of the pre-validator-v2 generation.

This command deliberately uses filesystem renames for large generated trees.
It never deletes an artifact and refuses to overwrite an existing archive.
After archival it restores the five canonical paper configurations and the
Direct--Passive checkpoint/training history, whose learned parameters do not
depend on factors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


ARCHIVE_NAME = "pre-validator-rewrite-2026-09-04"
EXPECTED_COMMIT = "a9724319f2cac42ed495b1c4a8ead601edd81c25"
TAG = ARCHIVE_NAME
PAPER_SYSTEMS = {
    "Direct--Passive GNN": "b0_eswc_reproduction__full_strat1m_minocc100__node_id",
    "Direct--Factor GNN": "a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id",
    "Candidate--C": "m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id",
    "Candidate--DP": "m1d_safe_factor_direct_compact_grouped__full_strat1m_minocc100__node_id",
    "Candidate--SR": "g0_globalfix_reference_v2__full_strat1m_minocc100__node_id",
}
PASSIVE_RUN = PAPER_SYSTEMS["Direct--Passive GNN"]
ARCHIVE_TARGETS = (
    Path("models"),
    Path("data/interim/full_strat1m_minocc100_labeled"),
    Path("data/processed/full_strat1m_minocc100"),
)
IMMUTABLE_INPUTS = (
    Path("data/raw"),
    Path("data/interim/constraint_registry_full.parquet"),
    Path("data/interim/full_strat1m_minocc100"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in paths:
        absolute = root / relative
        candidates = [absolute] if absolute.is_file() else sorted(absolute.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            records.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    return records


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def verify_release(root: Path) -> None:
    release = root / "release/zenodo/v1.0.0"
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=release, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-release-check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    commit = git(root, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Expected {EXPECTED_COMMIT}, found {commit}; refusing archival")
    if git(root, "rev-list", "-n", "1", TAG) != EXPECTED_COMMIT:
        raise RuntimeError(f"Annotated tag {TAG} does not resolve to {EXPECTED_COMMIT}")
    if not args.skip_release_check:
        verify_release(root)

    archive = root / "deprecated" / ARCHIVE_NAME
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    archive.mkdir(parents=True)

    immutable_before = inventory(root, IMMUTABLE_INPUTS)
    moved_roots: list[dict[str, str]] = []
    for source_rel in ARCHIVE_TARGETS:
        source = root / source_rel
        if not source.exists():
            continue
        destination = archive / source_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        moved_roots.append(
            {"active_path": source_rel.as_posix(), "archive_path": destination.relative_to(root).as_posix()}
        )

    archived_files = inventory(root, [archive.relative_to(root)])
    archived_by_path = {record["path"]: record for record in archived_files}

    # Recreate only canonical active run skeletons. Factor-dependent checkpoints
    # remain archived and must be retrained under validator semantics v2.
    active_models = root / "models"
    active_models.mkdir()
    for run_name in PAPER_SYSTEMS.values():
        source_config = archive / "models" / run_name / "config.json"
        destination_dir = active_models / run_name
        destination_dir.mkdir()
        shutil.copy2(source_config, destination_dir / "config.json")
    passive_checkpoint = archive / "models" / PASSIVE_RUN / "checkpoint.pth"
    shutil.copy2(passive_checkpoint, active_models / PASSIVE_RUN / "checkpoint.pth")
    passive_history = archive / "models" / PASSIVE_RUN / "training_history.json"
    shutil.copy2(passive_history, active_models / PASSIVE_RUN / "training_history.json")
    (active_models / "baselines").mkdir()
    (active_models / "paper_diagnostics").mkdir()
    (root / "data/processed/full_strat1m_minocc100").mkdir(parents=True)

    immutable_after = inventory(root, IMMUTABLE_INPUTS)
    if immutable_after != immutable_before:
        raise RuntimeError("Immutable inputs changed during archival")

    passive_archive_rel = passive_checkpoint.relative_to(root).as_posix()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": {"tag": TAG, "commit": commit, "tag_object": git(root, "rev-parse", TAG)},
        "zenodo_v1_0_0_checksums_verified": not args.skip_release_check,
        "zenodo_v1_0_0_checksums_file_sha256": sha256_file(
            root / "release/zenodo/v1.0.0/SHA256SUMS"
        ),
        "moved_roots": moved_roots,
        "paper_system_mapping": {
            "learned": {name: f"models/{path}" for name, path in PAPER_SYSTEMS.items()},
            "baselines": {
                "Baseline--DB": "baseline-DeleteFocusBaseline",
                "Baseline--AM": "baseline-AddMirrorBaseline",
                "Baseline--FM": "baseline-ConstraintFamilyMajorityBaseline",
                "Baseline--DM": "baseline-ConstraintDefinitionMajorityBaseline",
            },
        },
        "retained_active_checkpoint": {
            "system": "Direct--Passive GNN",
            "source": passive_archive_rel,
            "active_path": f"models/{PASSIVE_RUN}/checkpoint.pth",
            "size_bytes": archived_by_path[passive_archive_rel]["size_bytes"],
            "sha256": archived_by_path[passive_archive_rel]["sha256"],
        },
        "immutable_inputs": immutable_before,
        "files": archived_files,
        "file_count": len(archived_files),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in archived_files),
    }
    manifest_path = archive / "ARCHIVE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Archived {manifest['file_count']} files at {archive}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
