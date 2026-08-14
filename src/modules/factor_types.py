"""Factor-type vocabulary discovery and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.compute as pc
import pyarrow.parquet as pq


def normalize_active_factor_type_ids(
    values: Iterable[int] | None,
    *,
    num_factor_types: int,
    require_explicit: bool = False,
) -> tuple[int, ...]:
    """Validate a deterministic stable-id vocabulary for compact model dispatch."""

    if values is None:
        if require_explicit:
            raise ValueError("active_factor_type_ids must be explicit for this factor executor.")
        return tuple(range(max(int(num_factor_types), 0)))

    normalized = tuple(int(value) for value in values)
    if not normalized:
        raise ValueError("active_factor_type_ids must not be empty for factorized execution.")
    if tuple(sorted(normalized)) != normalized:
        raise ValueError("active_factor_type_ids must be strictly increasing.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("active_factor_type_ids must not contain duplicates.")
    if normalized[0] < 0:
        raise ValueError("active_factor_type_ids must be non-negative.")
    if num_factor_types <= 0:
        raise ValueError("num_factor_types must be positive when active_factor_type_ids are set.")
    if normalized[-1] >= int(num_factor_types):
        raise ValueError(
            "active_factor_type_ids contains an id outside the stable registry address space "
            f"[0, {int(num_factor_types)})."
        )
    return normalized


def scan_factor_type_ids(parquet_paths: Sequence[Path]) -> tuple[int, ...]:
    """Return the sorted union of nested ``factor_types`` values in parquet files."""

    observed: set[int] = set()
    for path in parquet_paths:
        if not path.exists():
            raise FileNotFoundError(f"Factor-type source parquet not found: {path}")
        parquet = pq.ParquetFile(path)
        if "factor_types" not in parquet.schema_arrow.names:
            raise ValueError(f"Parquet file lacks factor_types: {path}")
        for batch in parquet.iter_batches(columns=["factor_types"], batch_size=65536):
            flattened = pc.list_flatten(batch.column(0))
            if len(flattened) == 0:
                continue
            observed.update(int(value) for value in flattened.to_pylist())
    return tuple(sorted(observed))


def dataset_factor_type_ids(
    interim_directory: Path,
    *,
    splits: Sequence[str] = ("train", "val"),
) -> tuple[int, ...]:
    paths = [interim_directory / f"df_{split}.parquet" for split in splits]
    return scan_factor_type_ids(paths)


__all__ = [
    "dataset_factor_type_ids",
    "normalize_active_factor_type_ids",
    "scan_factor_type_ids",
]
