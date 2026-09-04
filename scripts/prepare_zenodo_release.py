#!/usr/bin/env python3
"""Assemble the immutable artifact bundles uploaded with the paper.

The generated directory is intentionally ignored by Git.  The archives retain
repository-relative paths so that they can be extracted directly into a clean
checkout of the matching release.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable


REPOSITORY_URL = "https://github.com/miguelvc6/fidelity-safety-gap"
UPSTREAM_DATASET_DOI = "https://doi.org/10.6084/m9.figshare.13338743.v2"
ZENODO_DOI = "10.5281/zenodo.22013512"

LEARNED_RUNS = {
    "Direct--Passive GNN": (
        "models/b0_eswc_reproduction__full_strat1m_minocc100__node_id"
    ),
    "Direct--Factor GNN": (
        "models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id"
    ),
    "Candidate--C": (
        "models/m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id"
    ),
    "Candidate--DP": (
        "models/m1d_safe_factor_direct_compact_grouped__full_strat1m_minocc100__node_id"
    ),
    "Candidate--SR": (
        "models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id"
    ),
}

BASELINE_RUNS = {
    "Baseline--DB": "baseline-DeleteFocusBaseline",
    "Baseline--AM": "baseline-AddMirrorBaseline",
    "Baseline--FM": "baseline-ConstraintFamilyMajorityBaseline",
    "Baseline--DM": "baseline-ConstraintDefinitionMajorityBaseline",
}

GRAPH_METADATA = (
    "data/processed/full_strat1m_minocc100/target_vocabs.json",
    "data/processed/full_strat1m_minocc100/train_graph-node_id.pkl.manifest.json",
    "data/processed/full_strat1m_minocc100/val_graph-node_id.pkl.manifest.json",
    "data/processed/full_strat1m_minocc100/test_graph-node_id.pkl.manifest.json",
    "data/processed/full_strat1m_minocc100/"
    "train_graph_repr-eswc_passive-node_id.pkl.manifest.json",
    "data/processed/full_strat1m_minocc100/"
    "val_graph_repr-eswc_passive-node_id.pkl.manifest.json",
    "data/processed/full_strat1m_minocc100/"
    "test_graph_repr-eswc_passive-node_id.pkl.manifest.json",
)

BENCHMARK_INPUTS = (
    "data/interim/full_strat1m_minocc100/df_test.parquet",
    "data/interim/full_strat1m_minocc100/df_train.parquet",
    "data/interim/full_strat1m_minocc100/df_val.parquet",
    "data/interim/full_strat1m_minocc100/globalintencoder.txt",
    "data/interim/full_strat1m_minocc100/hist_local_constraint_ids.csv",
    "data/interim/full_strat1m_minocc100/hist_local_constraint_ids_by_split.csv",
    "data/interim/full_strat1m_minocc100/sampling_metadata.json",
    "data/interim/full_strat1m_minocc100/sampling_report.csv",
    "data/interim/full_strat1m_minocc100/sampling_report.md",
    "data/interim/full_strat1m_minocc100_labeled/coverage_local.csv",
    "data/interim/full_strat1m_minocc100_labeled/coverage_local.md",
    "data/interim/full_strat1m_minocc100_labeled/df_test.parquet",
    "data/interim/full_strat1m_minocc100_labeled/df_train.parquet",
    "data/interim/full_strat1m_minocc100_labeled/df_val.parquet",
    "data/interim/full_strat1m_minocc100_labeled/filtered_factor_families_local.csv",
    "data/interim/full_strat1m_minocc100_labeled/filtered_factors_local.csv",
    "data/interim/full_strat1m_minocc100_labeled/filtered_factors_local.md",
)

REQUIRED_DIAGNOSTICS = (
    "models/paper_diagnostics/benchmark_provenance_and_statistics.json",
    "models/paper_diagnostics/corrected_paper_readiness.json",
    "models/paper_diagnostics/factor_semantics_summary.csv",
    "models/paper_diagnostics/label_semantics_audit.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Zenodo upload bundles for the paper."
    )
    parser.add_argument(
        "--release-version",
        default="v1.0.0",
        help="Git release tag represented by the archive (default: v1.0.0).",
    )
    parser.add_argument(
        "--doi",
        default=ZENODO_DOI,
        help="Reserved Zenodo DOI, for example 10.5281/zenodo.1234567.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: release/zenodo/<release-version>).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after a successful rebuild.",
    )
    return parser.parse_args()


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(root: Path, paths: Iterable[str]) -> list[str]:
    selected = sorted(dict.fromkeys(paths))
    missing = [relative for relative in selected if not (root / relative).is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"required release files are missing:\n{formatted}")
    return selected


def tracked_files(root: Path, pathspecs: Iterable[str]) -> list[str]:
    output = run_git(root, "ls-files", "--", *pathspecs)
    return sorted(line for line in output.splitlines() if line)


def normalized_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.pax_headers = {}
    return info


def write_tar_gz(root: Path, destination: Path, paths: Iterable[str]) -> None:
    with destination.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            compresslevel=6,
            mtime=0,
        ) as gzip_handle:
            with tarfile.open(
                fileobj=gzip_handle,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for relative in sorted(paths):
                    source = root / relative
                    info = normalized_tarinfo(
                        archive.gettarinfo(source.as_posix(), arcname=relative)
                    )
                    with source.open("rb") as source_handle:
                        archive.addfile(info, source_handle)


def verify_archive_members(destination: Path, expected_paths: Iterable[str]) -> None:
    expected = sorted(expected_paths)
    with tarfile.open(destination, mode="r:gz") as archive:
        members = archive.getmembers()
    actual = [member.name for member in members]
    if actual != expected or any(not member.isfile() for member in members):
        raise RuntimeError(f"archive member validation failed: {destination}")


def inventory(root: Path, paths: Iterable[str]) -> list[dict[str, object]]:
    entries = []
    for relative in sorted(paths):
        path = root / relative
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return entries


def archive_record(
    archive_path: Path,
    purpose: str,
    files: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "filename": archive_path.name,
        "purpose": purpose,
        "size_bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
        "file_count": len(files),
        "uncompressed_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
    }


def ensure_selected_paths_clean(root: Path, selected: Iterable[str]) -> None:
    tracked = [path for path in selected if path.startswith("models/")]
    tracked.append("latex_paper/supplement.pdf")
    status = run_git(root, "status", "--porcelain", "--", *tracked)
    if status:
        raise RuntimeError(
            "tracked release artifacts have uncommitted changes; commit or revert them "
            f"before packaging:\n{status}"
        )


def build_readme(
    version: str,
    doi: str | None,
    archive_records: list[dict[str, object]],
    supplement_size: int,
    license_size: int,
    license_scope_size: int,
    commit: str,
    tag_exists: bool,
) -> str:
    doi_text = f"https://doi.org/{doi}" if doi else "not yet reserved"
    table_rows = "\n".join(
        f"| `{item['filename']}` | {item['purpose']} | {item['size_bytes']:,} |"
        for item in archive_records
    )
    tag_note = (
        "The matching Git tag was present when these files were prepared."
        if tag_exists
        else "The matching Git tag was not present when these files were prepared. "
        "Create and verify it before publishing the Zenodo record."
    )
    return f"""# Reproduction artifacts

These files accompany *The Fidelity--Safety Gap in Neural Wikidata Constraint
Repair*. They correspond to software release `{version}` and repository commit
`{commit}`.

- Repository: {REPOSITORY_URL}
- Intended release: {REPOSITORY_URL}/releases/tag/{version}
- Zenodo DOI: {doi_text}
- Upstream source dataset: {UPSTREAM_DATASET_DOI}

{tag_note}

## Upload files

| File | Contents | Bytes |
|---|---|---:|
{table_rows}
| `supplement.pdf` | Supplementary results document | {supplement_size:,} |
| `LICENSE.txt` | CC BY 4.0 identifier and authoritative licence links | {license_size:,} |
| `LICENSE_SCOPE.md` | Licence scope for software, artifacts, and upstream data | {license_scope_size:,} |
| `ARTIFACT_MANIFEST.json` | Per-file provenance and SHA-256 checksums | generated |
| `SHA256SUMS` | Checksums of every other upload file | generated |

Verify the download before extraction:

```bash
sha256sum --check SHA256SUMS
```

Extract the three archives into a clean checkout of `{version}`:

```bash
tar -xzf fidelity-safety-gap-{version}-benchmark-inputs.tar.gz -C /path/to/fidelity-safety-gap
tar -xzf fidelity-safety-gap-{version}-paper-checkpoints.tar.gz -C /path/to/fidelity-safety-gap
tar -xzf fidelity-safety-gap-{version}-paper-results.tar.gz -C /path/to/fidelity-safety-gap
```

The benchmark archive contains both the exact rows used to reconstruct
evaluation states and the recorded labels used to construct the training
graphs. Do not rerun the constraint labeler for this benchmark.

## Generated graph caches

The 89.4 GB of materialised PyTorch graph payloads are intentionally excluded.
Their manifests and target vocabulary are included in the benchmark archive.
After extracting the recorded inputs, reconstruct the factor and passive graph
suites with the pinned software environment:

```bash
uv sync --frozen
uv run src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation factorized --registry-dataset full --constraint-scope local --shard-size 200000 --use-torch-save --persistence-profile research_safe --overwrite atomic
uv run src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation eswc_passive --registry-dataset full --constraint-scope local --shard-size 10000 --use-torch-save --persistence-profile research_safe --overwrite atomic
```

The release documentation describes training, prediction replay, diagnostics,
and the final readiness check.

## Licences

Repository software is MIT licensed. Author-created benchmark annotations,
model checkpoints, predictions, results, diagnostics, and the supplement are
CC BY 4.0. The upstream Wikidata Constraint Violations dataset remains CC0;
these archives do not relicense that source dataset. See `LICENSES/README.md`
in the software release for the complete scope statement.
"""


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", args.release_version):
        raise ValueError("--release-version must be a version tag such as v1.0.0")
    if args.doi and not re.fullmatch(r"10\.[0-9]{4,9}/[-._;()/A-Za-z0-9]+", args.doi):
        raise ValueError("--doi must be a DOI without the https://doi.org/ prefix")

    root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "release" / "zenodo" / args.release_version
    )
    release_root = (root / "release").resolve()
    if not output_dir.is_relative_to(release_root) or output_dir == release_root:
        raise ValueError(f"--output-dir must be a child of {release_root}")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; use --overwrite to replace it"
        )

    benchmark_paths = list(BENCHMARK_INPUTS)
    benchmark_paths.extend(GRAPH_METADATA)
    benchmark_paths.append("data/interim/constraint_registry_full.parquet")
    benchmark_paths = require_files(root, benchmark_paths)

    checkpoint_paths = require_files(
        root,
        [f"{run_path}/checkpoint.pth" for run_path in LEARNED_RUNS.values()],
    )

    result_roots = [*LEARNED_RUNS.values(), "models/baselines/full_strat1m/parquet"]
    result_paths = tracked_files(root, [*result_roots, "models/paper_diagnostics"])
    result_paths = require_files(root, result_paths)

    required_results = list(REQUIRED_DIAGNOSTICS)
    for run_path in LEARNED_RUNS.values():
        required_results.extend(
            (
                f"{run_path}/config.json",
                f"{run_path}/evaluations/model.json",
                f"{run_path}/evaluations/per_constraint.csv",
                f"{run_path}/evaluations/predictions.manifest.json",
                f"{run_path}/evaluations/predictions.parquet",
            )
        )
    baseline_root = "models/baselines/full_strat1m/parquet"
    for directory in BASELINE_RUNS.values():
        required_results.extend(
            (
                f"{baseline_root}/{directory}.json",
                f"{baseline_root}/{directory}/per_constraint.csv",
                f"{baseline_root}/{directory}/predictions.manifest.json",
                f"{baseline_root}/{directory}/predictions.parquet",
            )
        )
    require_files(root, required_results)
    missing_from_results = sorted(set(required_results) - set(result_paths))
    if missing_from_results:
        raise RuntimeError(
            "required results are not tracked by Git:\n"
            + "\n".join(f"  - {path}" for path in missing_from_results)
        )

    supplement = root / "latex_paper/supplement.pdf"
    if not supplement.is_file():
        raise FileNotFoundError("required supplement is missing: latex_paper/supplement.pdf")

    ensure_selected_paths_clean(root, result_paths)
    commit = run_git(root, "rev-parse", "HEAD")
    worktree_clean = not bool(run_git(root, "status", "--porcelain"))
    tag_exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{args.release_version}"],
        cwd=root,
        check=False,
        capture_output=True,
    ).returncode == 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{args.release_version}-", dir=output_dir.parent)
    )
    try:
        prefix = f"fidelity-safety-gap-{args.release_version}"
        archive_specs = (
            (
                temporary_dir / f"{prefix}-benchmark-inputs.tar.gz",
                "Exact benchmark inputs and graph metadata",
                benchmark_paths,
            ),
            (
                temporary_dir / f"{prefix}-paper-checkpoints.tar.gz",
                "Five selected learned-model checkpoints",
                checkpoint_paths,
            ),
            (
                temporary_dir / f"{prefix}-paper-results.tar.gz",
                "Predictions, metrics, diagnostics, configurations, and histories",
                result_paths,
            ),
        )

        records = []
        for archive_path, purpose, paths in archive_specs:
            print(f"Inventorying {archive_path.name} ({len(paths)} files)...", flush=True)
            file_inventory = inventory(root, paths)
            print(f"Writing {archive_path.name}...", flush=True)
            write_tar_gz(root, archive_path, paths)
            verify_archive_members(archive_path, paths)
            records.append(archive_record(archive_path, purpose, file_inventory))

        supplement_output = temporary_dir / "supplement.pdf"
        shutil.copyfile(supplement, supplement_output)
        license_output = temporary_dir / "LICENSE.txt"
        shutil.copyfile(root / "LICENSES/CC-BY-4.0.txt", license_output)
        license_scope_output = temporary_dir / "LICENSE_SCOPE.md"
        shutil.copyfile(root / "LICENSES/README.md", license_scope_output)

        manifest = {
            "schema_version": 1,
            "title": "The Fidelity--Safety Gap in Neural Wikidata Constraint Repair: reproduction artifacts",
            "release_version": args.release_version,
            "zenodo_doi": args.doi,
            "repository": {
                "url": REPOSITORY_URL,
                "commit": commit,
                "release_url": f"{REPOSITORY_URL}/releases/tag/{args.release_version}",
                "release_tag_present_at_preparation": tag_exists,
                "worktree_clean_at_preparation": worktree_clean,
                "selected_artifact_paths_clean": True,
            },
            "upstream_dataset": {
                "title": "Wikidata Constraints Violations - July 2018 - expanded",
                "version": 2,
                "doi": UPSTREAM_DATASET_DOI,
                "license": "CC0-1.0",
            },
            "licenses": {
                "software": "MIT",
                "author_created_artifacts": "CC-BY-4.0",
                "scope_document": "LICENSES/README.md in the software release",
            },
            "paper_system_mapping": {
                "learned": LEARNED_RUNS,
                "deterministic_baselines": BASELINE_RUNS,
            },
            "archives": records,
            "standalone_files": [
                {
                    "filename": path.name,
                    "purpose": purpose,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path, purpose in (
                    (supplement_output, "Supplementary results document"),
                    (license_output, "CC BY 4.0 identifier and authoritative links"),
                    (license_scope_output, "Licence scope statement"),
                )
            ],
            "excluded_generated_artifacts": {
                "materialized_graph_payloads": {
                    "reason": "Generated cache; exact inputs, construction code, manifests, and target vocabulary are archived.",
                    "local_size_bytes_at_preparation": 89418338063,
                },
                "other": [
                    "duplicate checkpoint.last.pth",
                    "legacy reranker_predictions.json",
                    "pre-schema-v2 backups",
                    "training plots and execution logs",
                    "stale and exploratory model runs",
                ],
            },
        }
        manifest_path = temporary_dir / "ARTIFACT_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        readme_path = temporary_dir / "README.md"
        readme_path.write_text(
            build_readme(
                args.release_version,
                args.doi,
                records,
                supplement_output.stat().st_size,
                license_output.stat().st_size,
                license_scope_output.stat().st_size,
                commit,
                tag_exists,
            ),
            encoding="utf-8",
        )

        checksum_targets = sorted(
            path
            for path in temporary_dir.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        )
        checksum_text = "".join(
            f"{sha256(path)}  {path.name}\n" for path in checksum_targets
        )
        (temporary_dir / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(f"Prepared Zenodo upload at {output_dir}")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
