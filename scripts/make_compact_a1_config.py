#!/usr/bin/env python3
"""Create an opt-in compact/grouped A1 config from the existing A1 config.

The active factor vocabulary is derived from train and validation labels only.
The test split is checked for unseen factor types but never influences the
mapping, which keeps test information out of model construction.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from modules.config import ModelConfig
from modules.factor_types import dataset_factor_type_ids


DEFAULT_SOURCE_CONFIG = Path(
    "models/a1_factorized_imitation__full_strat1m_minocc100__node_id/config.json"
)
DEFAULT_OUTPUT_CONFIG = Path(
    "models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id/config.json"
)
DEFAULT_INTERIM_DIRECTORY = Path(
    "data/interim/full_strat1m_minocc100_labeled"
)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Source A1 config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    if not isinstance(payload.get("model_config"), dict):
        raise ValueError(f"Missing model_config object in {path}")
    if not isinstance(payload.get("training_config"), dict):
        raise ValueError(f"Missing training_config object in {path}")
    return payload


def _require_reference_a1(model_config: Mapping[str, Any]) -> None:
    expected = {
        "dataset_variant": "full_strat1m_minocc100",
        "encoding": "node_id",
        "constraint_representation": "factorized",
        "model": "GIN_PRESSURE",
    }
    mismatches = {
        key: (model_config.get(key), value)
        for key, value in expected.items()
        if model_config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in sorted(mismatches.items())
        )
        raise ValueError(f"Source config is not the expected A1 experiment: {details}")


def build_compact_a1_config(
    source_config: Path,
    interim_directory: Path,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Return a copied A1 payload with only low-risk storage/dispatch changes."""

    payload = _load_config(source_config)
    model_config = payload["model_config"]
    _require_reference_a1(model_config)

    active_ids = dataset_factor_type_ids(
        interim_directory,
        splits=("train", "val"),
    )
    if not active_ids:
        raise ValueError("No factor types were observed in the train/validation splits.")

    test_ids = dataset_factor_type_ids(interim_directory, splits=("test",))
    unseen_test_ids = sorted(set(test_ids) - set(active_ids))
    if unseen_test_ids:
        raise ValueError(
            "Test data contains factor types absent from train/validation: "
            f"{unseen_test_ids}. Regenerate or audit the split before training."
        )

    num_factor_types = int(model_config.get("num_factor_types", 0))
    if num_factor_types <= 0 or active_ids[-1] >= num_factor_types:
        raise ValueError(
            "Observed factor type ids do not fit the source config's stable "
            f"address space [0, {num_factor_types}): max={active_ids[-1]}"
        )

    compact_payload = deepcopy(payload)
    compact_model_config = compact_payload["model_config"]
    compact_model_config.update(
        {
            "active_factor_type_ids": list(active_ids),
            "factor_executor_impl": "per_type_grouped_v2",
            "gold_edit_embedding_mode": "compact",
            "pressure_module_sharing": "per_type",
        }
    )

    # Exercise the same parser used by training before anything is written.
    ModelConfig.from_mapping(compact_model_config)
    return compact_payload, active_ids


def write_compact_a1_config(
    source_config: Path,
    output_config: Path,
    interim_directory: Path,
) -> tuple[int, ...]:
    """Build and exclusively create the config, refusing to overwrite a run."""

    if output_config.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing compact A1 config: {output_config}"
        )
    payload, active_ids = build_compact_a1_config(
        source_config,
        interim_directory,
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_config.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing compact A1 config: {output_config}"
        ) from exc
    return active_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-overwriting compact/grouped A1 config using factor "
            "types observed in train and validation parquet artifacts."
        )
    )
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument(
        "--interim-directory",
        type=Path,
        default=DEFAULT_INTERIM_DIRECTORY,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    active_ids = write_compact_a1_config(
        args.source_config,
        args.output_config,
        args.interim_directory,
    )
    print(f"Wrote {args.output_config}")
    print(
        f"Active factor types ({len(active_ids)}): "
        + ", ".join(str(value) for value in active_ids)
    )


if __name__ == "__main__":
    main()
