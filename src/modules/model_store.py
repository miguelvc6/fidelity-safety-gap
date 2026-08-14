"""Utilities for organizing model artifacts under the unified ``models/`` directory."""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

import torch

MODELS_ROOT = Path("models")
TEMPLATES_DIR = MODELS_ROOT / "templates"
DEFAULT_CONFIG_TAG = "default"
DEFAULT_CHECKPOINT_NAME = "checkpoint.pth"
TRAINING_HISTORY_NAME = "training_history.json"
LAST_CHECKPOINT_NAME = "checkpoint.last.pth"
CONFIG_FILE_NAME = "config.json"
EVAL_DIR_NAME = "evaluations"

_SANITIZE_REGEX = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_fragment(name: str | Path | None, fallback: str = DEFAULT_CONFIG_TAG) -> str:
    """Return a filesystem-friendly fragment derived from ``name``."""
    if name is None:
        return fallback
    if isinstance(name, Path):
        name = name.stem
    if not isinstance(name, str):
        name = str(name)
    cleaned = _SANITIZE_REGEX.sub("_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def config_tag_from_path(path: Path | None) -> str:
    """Derive a config tag from a given path."""
    if path is None:
        return DEFAULT_CONFIG_TAG
    candidate = Path(path)
    if candidate.name == CONFIG_FILE_NAME and candidate.parent.name:
        return sanitize_fragment(candidate.parent.name, fallback=DEFAULT_CONFIG_TAG)
    return sanitize_fragment(candidate, fallback=DEFAULT_CONFIG_TAG)


def run_slug(dataset_variant: str, encoding: str, model_name: str, config_tag: str | None = None) -> str:
    """Build a normalized slug identifying a model run."""
    parts = [
        sanitize_fragment(dataset_variant, fallback="dataset"),
        sanitize_fragment(encoding, fallback="encoding"),
        sanitize_fragment(model_name, fallback="model").upper(),
        sanitize_fragment(config_tag, fallback=DEFAULT_CONFIG_TAG),
    ]
    return "-".join(parts[:2]) + f"_{parts[2]}_{parts[3]}"


def run_dir(dataset_variant: str, encoding: str, model_name: str, config_tag: str | None = None) -> Path:
    """Return the directory path for a specific model run."""
    slug = run_slug(dataset_variant, encoding, model_name, config_tag)
    return MODELS_ROOT / slug


def ensure_run_dir(dataset_variant: str, encoding: str, model_name: str, config_tag: str | None = None) -> Path:
    """Create the run directory if needed and return it."""
    directory = run_dir(dataset_variant, encoding, model_name, config_tag)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_dir_from_config_path(config_path: Path) -> Path:
    """Return the artifact directory associated with an experiment config path."""
    candidate = Path(config_path)
    if candidate.suffix.lower() == ".json":
        return candidate.parent
    return candidate


def ensure_run_dir_for_config(config_path: Path) -> Path:
    """Create the artifact directory for ``config_path`` if needed and return it."""
    directory = run_dir_from_config_path(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_checkpoint_path(run_directory: Path) -> Path:
    """Return the checkpoint file location within ``run_directory``."""
    return run_directory / DEFAULT_CHECKPOINT_NAME


def get_last_checkpoint_path(run_directory: Path) -> Path:
    """Return the optional final-epoch checkpoint location within ``run_directory``."""

    return run_directory / LAST_CHECKPOINT_NAME


def atomic_torch_save(payload: object, destination: Path) -> None:
    """Write a torch artifact atomically without exposing a partial checkpoint."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def history_path(run_directory: Path) -> Path:
    """Return the training history file path for a run."""
    return run_directory / TRAINING_HISTORY_NAME


def config_copy_path(run_directory: Path) -> Path:
    """Return the stored config copy path for a run."""
    return run_directory / CONFIG_FILE_NAME


def evaluations_dir(run_directory: Path, create: bool = False) -> Path:
    """Return the evaluations directory, optionally creating it."""
    path = run_directory / EVAL_DIR_NAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def baseline_dir(dataset_variant: str, encoding: str, create: bool = False) -> Path:
    """Return the baseline directory for a dataset/encoding pair."""
    directory = MODELS_ROOT / "baselines" / sanitize_fragment(dataset_variant) / sanitize_fragment(encoding)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_run_dirs(dataset_variant: str, encoding: str, model_name: str) -> list[Path]:
    """List all stored run directories matching the given identifiers."""
    slug_with_placeholder = run_slug(dataset_variant, encoding, model_name, config_tag="placeholder")
    prefix = slug_with_placeholder.rsplit("_", 1)[0] + "_"
    candidates: set[Path] = set()
    if not MODELS_ROOT.exists():
        return []
    for entry in MODELS_ROOT.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(prefix):
            candidates.add(entry)
            continue
        if _run_directory_matches(entry, dataset_variant, encoding, model_name):
            candidates.add(entry)
    return sorted(candidates)


def resolve_run_dir(
    dataset_variant: str,
    encoding: str,
    model_name: str,
    config_tag: str | None = None,
) -> Path:
    """Resolve a unique run directory, optionally constrained by ``config_tag``."""
    if config_tag:
        direct_candidate = MODELS_ROOT / sanitize_fragment(config_tag, fallback=DEFAULT_CONFIG_TAG)
        if direct_candidate.exists() and _run_directory_matches(direct_candidate, dataset_variant, encoding, model_name):
            return direct_candidate

        legacy_candidate = run_dir(dataset_variant, encoding, model_name, config_tag)
        if legacy_candidate.exists():
            return legacy_candidate

        raise FileNotFoundError(
            f"Model run directory not found for dataset={dataset_variant}, encoding={encoding}, "
            f"model={model_name}, config_tag={config_tag}. Tried {direct_candidate} and {legacy_candidate}"
        )

    candidates = list_run_dirs(dataset_variant, encoding, model_name)
    if not candidates:
        raise FileNotFoundError(
            f"No model runs stored for dataset={dataset_variant}, encoding={encoding}, model={model_name}."
        )
    if len(candidates) == 1:
        return candidates[0]
    joined = ", ".join(entry.name for entry in candidates)
    raise FileNotFoundError("Multiple model runs found; please specify --config-tag. Candidates: " + joined)


def available_config_tags(dataset_variant: str, encoding: str, model_name: str) -> list[str]:
    """Return the discovered config tags for the given model runs."""
    prefix = run_slug(dataset_variant, encoding, model_name, config_tag="placeholder")
    prefix = prefix.rsplit("_", 1)[0] + "_"
    tags: set[str] = set()
    for directory in list_run_dirs(dataset_variant, encoding, model_name):
        if directory.name.startswith(prefix):
            tags.add(directory.name[len(prefix) :])
        else:
            tags.add(sanitize_fragment(directory.name, fallback=DEFAULT_CONFIG_TAG))
    return sorted(tags)


def iter_eval_files(dataset_variant: str, encoding: str, model_name: str) -> Iterable[Path]:
    """Yield evaluation result files for each stored run."""
    for run in list_run_dirs(dataset_variant, encoding, model_name):
        eval_dir = run / EVAL_DIR_NAME
        if not eval_dir.exists():
            continue
        yield from sorted(eval_dir.glob("*.json"))


def _run_directory_matches(run_directory: Path, dataset_variant: str, encoding: str, model_name: str) -> bool:
    config_path = run_directory / CONFIG_FILE_NAME
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(payload, dict):
        return False
    model_cfg = payload.get("model_config")
    if not isinstance(model_cfg, dict):
        return False

    candidate_dataset = model_cfg.get("dataset_variant")
    candidate_encoding = model_cfg.get("encoding")
    candidate_model = model_cfg.get("model")
    if not all(isinstance(value, str) for value in (candidate_dataset, candidate_encoding, candidate_model)):
        return False

    return (
        candidate_dataset == dataset_variant
        and candidate_encoding == encoding
        and str(candidate_model).upper() == str(model_name).upper()
    )
