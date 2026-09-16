#!/usr/bin/env python3
"""Identity-preserving decoder coverage and bounded verified-fix audit.

The command is resumable by stage.  ``--stage all`` performs the complete
parent-corpus scan, runs both pinned validator profiles in isolated processes,
and writes row-level and aggregate artifacts.  It never changes a model,
training configuration, or production dataset.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import pickle
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.identity_audit import (  # noqa: E402
    EMPTY,
    ENTITY_SLOT_INDICES,
    PREDICATE_SLOT_INDICES,
    ROLE_NAMES,
    SemanticEncoder,
    canonical_term,
    copy_domain,
    definition_terms,
    factorized_wiring_edge_count,
    grounding_reason,
    input_topology,
    local_predicates,
    parse_raw_record,
    percentile_from_histogram,
    represented_fact_triples,
    role_alias_mask,
    roundtrip_slots,
    semantic_id,
    sha256_file,
    stable_record_key,
)


TARGETS: tuple[str, ...] = (
    "conflictWith",
    "distinct",
    "inverse",
    "itemRequiresStatement",
    "oneOf",
    "single",
    "type",
    "valueRequiresStatement",
    "valueType",
)
RAW_SPLITS: tuple[str, ...] = ("train", "dev", "test")
PARENT_SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_CODE = {name: index for index, name in enumerate(PARENT_SPLITS)}
OUTCOME_CODE = {"satisfied": 0, "violated": 1, "unknown": 2, "not_evaluated": -1}
OUTCOME_NAME = {value: key for key, value in OUTCOME_CODE.items()}
DEFAULT_OUTPUT = REPO_ROOT / "audits" / "decoder_verified_fix_2026-09-16"
DEFAULT_CHECKED = Path("/tmp/fsg-audit-checked")
DEFAULT_V3 = Path("/tmp/fsg-audit-v3")
DEFAULT_HIERARCHY = (
    REPO_ROOT
    / "deprecated/post-validator-conversation-2026-09-15/data/static/wikidata-p279-2018-07-01.v2.json"
)

SAMPLE_BINS: tuple[tuple[int, int | None, str], ...] = (
    (1, 32, "1-32"),
    (33, 64, "33-64"),
    (65, 83, "65-83"),
    (84, 107, "84-107"),
    (108, 108, "108"),
    (109, 160, "109-160"),
    (161, 267, "161-267"),
    (268, None, "268+"),
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def raw_paths(raw_root: Path) -> list[tuple[str, str, Path]]:
    return [
        (
            raw_split,
            target,
            raw_root / f"constraint-corrections-{target}.tsv.gz.full.{raw_split}.tsv.gz",
        )
        for raw_split in RAW_SPLITS
        for target in TARGETS
    ]


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = pd.read_parquet(path, columns=["registry_json"])["registry_json"].iloc[0]
    if isinstance(payload, str):
        payload = json.loads(payload)
    result: dict[str, dict[str, Any]] = {}
    for raw_id, entry in payload.items():
        result[canonical_term(raw_id, blank_scope=f"constraint:{raw_id}")] = dict(entry)
    return result


def load_encoder(path: Path) -> tuple[dict[str, int], dict[int, str], dict[str, int]]:
    with path.open("rb") as handle:
        encoding, decoding, _filtered, _unfiltered = pickle.load(handle)
    canonical_to_id: dict[str, int] = {}
    canonical_priority: dict[str, int] = {}

    def priority(raw: str) -> int:
        value = raw.strip()
        if value.startswith("^<http://www.wikidata.org/entity/"):
            return 10
        if value.startswith("<http://www.wikidata.org/entity/"):
            return 100
        if value.startswith("http://www.wikidata.org/entity/"):
            return 90
        if value.startswith("<http://www.wikidata.org/prop/direct/"):
            return 80
        if value.startswith("http://www.wikidata.org/prop/direct/"):
            return 70
        if re.fullmatch(r"[PQ][1-9]\d*", value):
            return 50
        return 60

    for raw, value in encoding.items():
        if not raw or raw in ROLE_NAMES or raw == "unknown" or raw.startswith("constraint_factor::"):
            continue
        term = canonical_term(raw)
        raw_priority = priority(raw)
        if raw_priority > canonical_priority.get(term, -1):
            canonical_to_id[term] = int(value)
            canonical_priority[term] = raw_priority
    return dict(encoding), {int(key): value for key, value in decoding.items()}, canonical_to_id


def _scan_raw_file_count(task: tuple[str, str, str, set[str]]) -> dict[str, Any]:
    raw_split, target, raw_path_str, registry_raw_ids = task
    raw_path = Path(raw_path_str)
    total = accepted = unknown = malformed = 0
    operation_patterns: Counter[str] = Counter()
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            total += 1
            elements = line.rstrip("\n").split("\t")
            if not elements or elements[0] not in registry_raw_ids:
                unknown += 1
                continue
            accepted += 1
            if len(elements) < 9:
                malformed += 1
                continue
            # This is diagnostic only; the lossless parser repeats the check.
            idx = len(elements) - 1
            while idx >= 0 and (
                elements[idx].strip() == ""
                or (elements[idx].lstrip().startswith("{") and elements[idx].rstrip().endswith("}"))
            ):
                idx -= 1
            op_end = idx + 1
            if op_end < 9 or (op_end - 9) % 4:
                malformed += 1
                continue
            kinds = []
            for pos in range(12, op_end, 4):
                kinds.append(elements[pos].rsplit("#", 1)[-1].rstrip(">"))
            operation_patterns["+".join(kinds) if kinds else "noop"] += 1
    return {
        "raw_split": raw_split,
        "source_family": target,
        "path": str(raw_path.resolve()),
        "size_bytes": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
        "source_rows": total,
        "accepted_rows": accepted,
        "unknown_constraint_rows": unknown,
        "malformed_rows": malformed,
        "operation_patterns": dict(operation_patterns),
    }


def _sample_bin(length: int) -> str:
    for lower, upper, label in SAMPLE_BINS:
        if length >= lower and (upper is None or length <= upper):
            return label
    return "0"


def replay_sample(parent_root: Path, *, seed: int = 42, fraction: float = 0.5) -> dict[str, np.ndarray]:
    strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for split in PARENT_SPLITS:
        path = parent_root / f"df_{split}.parquet"
        offset = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=100_000,
            columns=["constraint_type", "local_constraint_ids"],
        ):
            families = batch.column(0).to_pylist()
            lengths = pc.fill_null(pc.list_value_length(batch.column(1)), 0).to_numpy(zero_copy_only=False)
            for local_index, (family, length) in enumerate(zip(families, lengths)):
                strata[(split, str(family), _sample_bin(int(length)))].append(offset + local_index)
            offset += batch.num_rows

    rng = np.random.default_rng(seed)
    masks = {
        split: np.zeros(pq.ParquetFile(parent_root / f"df_{split}.parquet").metadata.num_rows, dtype=np.bool_)
        for split in PARENT_SPLITS
    }
    for split, family, bin_name in sorted(strata):
        indices = strata[(split, family, bin_name)]
        target = min(len(indices), max(1, int(round(len(indices) * fraction))))
        if target == len(indices):
            sampled = np.asarray(indices, dtype=np.int64)
        else:
            positions = rng.choice(len(indices), size=target, replace=False)
            sampled = np.asarray(indices, dtype=np.int64)[positions]
        masks[split][sampled] = True
    return masks


def _column_pylist(path: Path, column: str) -> list[Any]:
    return pq.read_table(path, columns=[column])[column].to_pylist()


def prepare_lineage(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    registry_frame = pd.read_parquet(args.registry, columns=["registry_json"])
    raw_registry = registry_frame["registry_json"].iloc[0]
    if isinstance(raw_registry, str):
        raw_registry = json.loads(raw_registry)
    registry_ids = set(raw_registry)

    jobs = [
        (raw_split, target, str(path), registry_ids)
        for raw_split, target, path in raw_paths(args.raw_root)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(_scan_raw_file_count, jobs))

    start = 0
    labels: list[str] = []
    for record in results:
        record["accepted_start"] = start
        start += int(record["accepted_rows"])
        record["accepted_stop"] = start
        labels.extend([str(record["source_family"])] * int(record["accepted_rows"]))
    total = start
    label_array = np.asarray(labels, dtype=object)
    del labels
    all_indices = np.arange(total, dtype=np.int64)
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=0.30,
        stratify=label_array,
        random_state=42,
        shuffle=True,
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=label_array[temp_indices],
        random_state=42,
        shuffle=True,
    )
    split_indices = {"train": train_indices, "val": val_indices, "test": test_indices}
    split_code = np.empty(total, dtype=np.uint8)
    parent_row = np.empty(total, dtype=np.int32)
    for split, indices in split_indices.items():
        split_code[indices] = SPLIT_CODE[split]
        parent_row[indices] = np.arange(len(indices), dtype=np.int32)
        parent_path = args.parent_root / f"df_{split}.parquet"
        parquet_count = pq.ParquetFile(parent_path).metadata.num_rows
        if parquet_count != len(indices):
            raise AssertionError(f"parent {split} count {parquet_count} != replay {len(indices)}")
        actual_families = np.asarray(_column_pylist(parent_path, "constraint_type"), dtype=object)
        expected_families = label_array[indices]
        if not np.array_equal(actual_families, expected_families):
            mismatch = int(np.flatnonzero(actual_families != expected_families)[0])
            raise AssertionError(f"parent lineage family mismatch in {split} at row {mismatch}")

    sample_masks = replay_sample(args.parent_root, seed=42, fraction=0.5)
    sampled = np.zeros(total, dtype=np.bool_)
    sample_verification: dict[str, Any] = {}
    for split, indices in split_indices.items():
        sampled[indices] = sample_masks[split]
        sample_path = args.sample_root / f"df_{split}.parquet"
        parent_path = args.parent_root / f"df_{split}.parquet"
        actual_count = pq.ParquetFile(sample_path).metadata.num_rows
        replayed_count = int(sample_masks[split].sum())
        if actual_count != replayed_count:
            raise AssertionError(f"sample {split} count {actual_count} != replay {replayed_count}")
        # Verify record identity, not just counts.  These scalar columns retain
        # row occurrences and the exact six target slots; repeated content is
        # compared in preserved parent order rather than used as a join key.
        verify_columns = [
            "constraint_type",
            "constraint_id",
            "subject",
            "predicate",
            "object",
            "other_subject",
            "other_predicate",
            "other_object",
            "add_subject",
            "add_predicate",
            "add_object",
            "del_subject",
            "del_predicate",
            "del_object",
        ]
        parent_table = pq.read_table(parent_path, columns=verify_columns)
        selected_table = parent_table.take(pa.array(np.flatnonzero(sample_masks[split]), type=pa.int64()))
        sample_table = pq.read_table(sample_path, columns=verify_columns)
        if not selected_table.equals(sample_table):
            raise AssertionError(f"sample replay content mismatch for {split}")
        sample_verification[split] = {
            "rows": actual_count,
            "columns": verify_columns,
            "exact_arrow_equality": True,
        }

    np.save(output / "source_to_split_code.npy", split_code, allow_pickle=False)
    np.save(output / "source_to_parent_row.npy", parent_row, allow_pickle=False)
    np.save(output / "source_to_existing_sample.npy", sampled, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "full_scan": True,
        "source_rows": sum(int(record["source_rows"]) for record in results),
        "parent_rows": total,
        "parent_split_counts": {split: len(indices) for split, indices in split_indices.items()},
        "sample_split_counts": {split: int(mask.sum()) for split, mask in sample_masks.items()},
        "discarded_split_counts": {
            split: int(len(mask) - mask.sum()) for split, mask in sample_masks.items()
        },
        "unknown_constraint_rows": sum(int(record["unknown_constraint_rows"]) for record in results),
        "malformed_rows": sum(int(record["malformed_rows"]) for record in results),
        "split_replay": {
            "implementation": "sklearn.model_selection.train_test_split",
            "sklearn_version": __import__("sklearn").__version__,
            "random_state": 42,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "stratify": "source constraint_type",
        },
        "sample_replay": {
            "seed": 42,
            "fraction": 0.5,
            "strata": ["split", "constraint_type", "len(local_constraint_ids) bin"],
            "bins": [dict(lower=lo, upper=hi, label=label) for lo, hi, label in SAMPLE_BINS],
            "content_verification": sample_verification,
        },
        "files": results,
        "runtime_seconds": time.monotonic() - started,
        "parent_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "worker_max_rss_kib": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
    }
    atomic_json(output / "lineage.json", metadata)
    return metadata


def _registry_property_index(registry: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for constraint_id, entry in registry.items():
        prop = canonical_term(entry.get("constrained_property"))
        if prop.startswith("wd:P"):
            result[prop].append(constraint_id)
    return {key: sorted(values) for key, values in result.items()}


def _slot_type(index: int) -> str:
    return "predicate" if index in PREDICATE_SLOT_INDICES else "entity"


def _decode_output_vocab(
    encoder_encoding: Mapping[str, int],
    encoder_decoding: Mapping[int, str],
    target_vocab_path: Path,
) -> dict[str, Any]:
    payload = json.loads(target_vocab_path.read_text(encoding="utf-8"))
    per_split = payload.get("per_split") or {}
    entity_ids = set(per_split["train"]["entity_class_ids"]) | set(per_split["val"]["entity_class_ids"])
    predicate_ids = set(per_split["train"]["predicate_class_ids"]) | set(
        per_split["val"]["predicate_class_ids"]
    )
    role_ids = {role: int(encoder_encoding.get(role, 0)) for role in ROLE_NAMES}

    def terms(ids: set[int]) -> set[str]:
        values: set[str] = set()
        for value in ids:
            raw = encoder_decoding.get(int(value), "")
            if not raw or raw == "unknown" or raw in ROLE_NAMES or raw.startswith("constraint_factor::"):
                continue
            values.add(canonical_term(raw))
        return values

    return {
        "entity_ids": {int(value) for value in entity_ids},
        "predicate_ids": {int(value) for value in predicate_ids},
        "entity_terms": terms({int(value) for value in entity_ids}),
        "predicate_terms": terms({int(value) for value in predicate_ids}),
        "entity_role_indices": {
            index for index, role in enumerate(ROLE_NAMES) if role_ids[role] in entity_ids
        },
        "predicate_role_indices": {
            index for index, role in enumerate(ROLE_NAMES) if role_ids[role] in predicate_ids
        },
        "role_ids": role_ids,
    }


_SEMANTIC_CONTEXT: dict[str, Any] = {}


def _semantic_worker_init(config_path: str) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    registry = load_registry(Path(config["registry"]))
    encoding, decoding, canonical_to_id = load_encoder(Path(config["encoder"]))
    output_vocab = _decode_output_vocab(encoding, decoding, Path(config["target_vocabs"]))
    _SEMANTIC_CONTEXT.clear()
    _SEMANTIC_CONTEXT.update(
        config=config,
        registry=registry,
        property_index=_registry_property_index(registry),
        canonical_to_id=canonical_to_id,
        output_vocab=output_vocab,
        split_code=np.load(config["split_code"], mmap_mode="r"),
        parent_row=np.load(config["parent_row"], mmap_mode="r"),
        sampled=np.load(config["sampled"], mmap_mode="r"),
    )


def _serialize_facts(record: Any) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    subject, _predicate, obj, _os, _op, _oo = record.roles
    other = __import__("modules.identity_audit", fromlist=["other_entity"]).other_entity(record.roles)

    def pair(owner: str) -> tuple[list[str], list[str]]:
        facts = list(record.facts_by_owner.get(owner, ())) if owner else []
        return [value[0] for value in facts], [value[1] for value in facts]

    sp, so = pair(subject)
    op, oo = pair(obj)
    xp, xo = pair(other)
    return sp, so, op, oo, xp, xo


def _semantic_chunk(task: Mapping[str, Any]) -> dict[str, Any]:
    context = _SEMANTIC_CONTEXT
    registry = context["registry"]
    property_index = context["property_index"]
    canonical_to_id = context["canonical_to_id"]
    output_vocab = context["output_vocab"]
    raw_path = Path(task["path"])
    output_path = Path(task["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()

    start = int(task["accepted_start"])
    accepted_local = 0
    source_line = 0
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    existing_constants = [
        output_vocab["predicate_terms"] if index in PREDICATE_SLOT_INDICES else output_vocab["entity_terms"]
        for index in range(6)
    ]
    existing_roles = [
        output_vocab["predicate_role_indices"]
        if index in PREDICATE_SLOT_INDICES
        else output_vocab["entity_role_indices"]
        for index in range(6)
    ]

    def flush() -> None:
        nonlocal writer, rows
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", use_dictionary=True)
        writer.write_table(table)
        rows = []

    try:
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            for source_line, line in enumerate(handle, start=1):
                elements = line.rstrip("\n").split("\t")
                raw_constraint = elements[0] if elements else ""
                constraint_id = canonical_term(raw_constraint, blank_scope=f"constraint:{raw_constraint}")
                if constraint_id not in registry:
                    counters["unknown_constraint"] += 1
                    continue
                source_index = start + accepted_local
                accepted_local += 1
                key = stable_record_key(raw_path, source_line)
                record = parse_raw_record(elements, source_key=key)
                family = str(registry[constraint_id].get("constraint_family") or task["source_family"])
                predicates = local_predicates(record)
                local_ids: set[str] = {constraint_id}
                for predicate in predicates:
                    local_ids.update(property_index.get(predicate, ()))
                # Production factorized inputs retain supported secondary
                # definitions and always retain the primary.
                retained_local_ids = sorted(
                    cid
                    for cid in local_ids
                    if cid == constraint_id or bool(registry.get(cid, {}).get("constraint_family_supported", False))
                )
                _primary_factors, primary_pairs, primary_definition_warnings = definition_terms(
                    [constraint_id], registry
                )
                factor_ids, factor_pairs, factor_definition_warnings = definition_terms(
                    retained_local_ids, registry
                )
                passive_domain = copy_domain(record, primary_pairs)
                factorized_domain = copy_domain(record, factor_pairs)
                primary_parameter_terms = {value for pair in primary_pairs for value in pair}
                local_parameter_terms = {value for pair in factor_pairs for value in pair} - primary_parameter_terms
                slots = record.six_slots()
                role_masks = [role_alias_mask(term, record.roles) for term in slots]
                target_encoder_ids = [int(canonical_to_id.get(term, 0)) if term else 0 for term in slots]
                slot_in_passive_input = [bool(term and term in passive_domain) for term in slots]
                slot_in_factorized_input = [bool(term and term in factorized_domain) for term in slots]
                slot_in_primary_parameters = [
                    bool(term and term in primary_parameter_terms) for term in slots
                ]
                slot_in_local_parameters = [
                    bool(term and term in local_parameter_terms) for term in slots
                ]

                existing_slot_fixed: list[bool] = []
                existing_slot_passive: list[bool] = []
                existing_slot_factorized: list[bool] = []
                existing_reason_passive: list[str] = []
                existing_reason_factorized: list[str] = []
                for index, term in enumerate(slots):
                    fixed, copied_passive, reason_passive = grounding_reason(
                        term=term,
                        role_mask=role_masks[index],
                        permitted_role_indices=existing_roles[index],
                        constant_terms=existing_constants[index],
                        input_terms=passive_domain,
                        primary_parameter_terms=primary_parameter_terms,
                        local_parameter_terms=set(),
                        canonical_to_encoder_id=canonical_to_id,
                    )
                    _fixed_again, copied_factorized, reason_factorized = grounding_reason(
                        term=term,
                        role_mask=role_masks[index],
                        permitted_role_indices=existing_roles[index],
                        constant_terms=existing_constants[index],
                        input_terms=factorized_domain,
                        primary_parameter_terms=primary_parameter_terms,
                        local_parameter_terms=local_parameter_terms,
                        canonical_to_encoder_id=canonical_to_id,
                    )
                    if fixed != _fixed_again:
                        raise AssertionError("fixed grounding depends on copy profile")
                    existing_slot_fixed.append(fixed)
                    existing_slot_passive.append(copied_passive)
                    existing_slot_factorized.append(copied_factorized)
                    existing_reason_passive.append(reason_passive)
                    existing_reason_factorized.append(reason_factorized)

                existing_a = bool(record.compatible and all(existing_slot_fixed))
                existing_b_passive = bool(record.compatible and all(existing_slot_passive))
                existing_b_factorized = bool(record.compatible and all(existing_slot_factorized))
                if existing_a and (not existing_b_passive or not existing_b_factorized):
                    raise AssertionError("no-copy coverage is not a subset of copy coverage")
                if existing_a:
                    ok, _ = roundtrip_slots(
                        slots,
                        record.roles,
                        role_indices_by_slot=existing_roles,
                        constants_by_slot=existing_constants,
                    )
                    if not ok:
                        raise AssertionError("existing no-copy edit failed round-trip")
                if existing_b_passive:
                    ok, _ = roundtrip_slots(
                        slots,
                        record.roles,
                        role_indices_by_slot=existing_roles,
                        constants_by_slot=existing_constants,
                        copies_by_slot=[passive_domain] * 6,
                    )
                    if not ok:
                        raise AssertionError("existing passive-copy edit failed round-trip")
                if existing_b_factorized:
                    ok, _ = roundtrip_slots(
                        slots,
                        record.roles,
                        role_indices_by_slot=existing_roles,
                        constants_by_slot=existing_constants,
                        copies_by_slot=[factorized_domain] * 6,
                    )
                    if not ok:
                        raise AssertionError("existing factorized-copy edit failed round-trip")

                passive_topology = input_topology(
                    record,
                    factor_ids=[constraint_id],
                    parameter_pairs=primary_pairs,
                    canonical_to_encoder_id=canonical_to_id,
                )
                factorized_topology = input_topology(
                    record,
                    factor_ids=factor_ids,
                    parameter_pairs=factor_pairs,
                    canonical_to_encoder_id=canonical_to_id,
                    wiring_edges=factorized_wiring_edge_count(
                        record,
                        constraint_ids=factor_ids,
                        registry=registry,
                        primary_constraint_id=constraint_id,
                    ),
                )
                sp, so, op, oo, xp, xo = _serialize_facts(record)
                split = PARENT_SPLITS[int(context["split_code"][source_index])]
                sampled = bool(context["sampled"][source_index])
                additions = record.additions
                deletions = record.deletions
                edit_kind = (
                    "paired"
                    if additions and deletions
                    else "addition_only"
                    if additions
                    else "deletion_only"
                    if deletions
                    else "noop"
                )
                replacement = bool(
                    additions
                    and deletions
                    and additions[0].subject == deletions[0].subject
                    and additions[0].predicate == deletions[0].predicate
                    and additions[0].object != deletions[0].object
                )
                base_triple = record.roles[:3]
                base_removed = bool(deletions and deletions[0].triple() == tuple(base_triple))
                rows.append(
                    {
                        "source_index": source_index,
                        "source_key": key,
                        "diagnostic_rank": int.from_bytes(
                            hashlib.blake2b(f"1729:{key}".encode(), digest_size=8).digest(),
                            "big",
                        )
                        & ((1 << 63) - 1),
                        "source_file": raw_path.name,
                        "source_line": source_line,
                        "original_split": task["raw_split"],
                        "split": split,
                        "parent_row": int(context["parent_row"][source_index]),
                        "existing_sample": sampled,
                        "population": "existing_sample" if sampled else "previously_discarded",
                        "source_family": task["source_family"],
                        "family": family,
                        "constraint_id": record.constraint_id,
                        "revision": record.revision,
                        **{name: value for name, value in zip(ROLE_NAMES, record.roles)},
                        "subject_predicates": sp,
                        "subject_objects": so,
                        "object_predicates": op,
                        "object_objects": oo,
                        "other_entity_predicates": xp,
                        "other_entity_objects": xo,
                        "described_owners": sorted(record.described_owners),
                        "add_subject": slots[0],
                        "add_predicate": slots[1],
                        "add_object": slots[2],
                        "del_subject": slots[3],
                        "del_predicate": slots[4],
                        "del_object": slots[5],
                        "target_encoder_ids": target_encoder_ids,
                        "target_role_masks": role_masks,
                        "slot_in_passive_input": slot_in_passive_input,
                        "slot_in_factorized_input": slot_in_factorized_input,
                        "slot_in_primary_parameters": slot_in_primary_parameters,
                        "slot_in_local_parameters": slot_in_local_parameters,
                        "R": bool(record.compatible),
                        "exclusion_reason": record.exclusion_reason,
                        "parse_warnings": json.dumps(record.parse_warnings, sort_keys=True),
                        "num_additions": len(additions),
                        "num_deletions": len(deletions),
                        "edit_kind": edit_kind,
                        "same_subject_property_replacement": replacement,
                        "base_removed": base_removed,
                        "base_preserved_source_intent": not base_removed,
                        "local_constraint_ids": retained_local_ids,
                        "primary_definition_warnings": json.dumps(
                            primary_definition_warnings, sort_keys=True
                        ),
                        "factor_definition_warnings": json.dumps(
                            factor_definition_warnings, sort_keys=True
                        ),
                        "existing_slot_fixed": existing_slot_fixed,
                        "existing_slot_passive": existing_slot_passive,
                        "existing_slot_factorized": existing_slot_factorized,
                        "existing_reason_passive": existing_reason_passive,
                        "existing_reason_factorized": existing_reason_factorized,
                        "existing_A": existing_a,
                        "existing_B_passive": existing_b_passive,
                        "existing_B_factorized": existing_b_factorized,
                        **{f"passive_{key}": value for key, value in passive_topology.items()},
                        **{f"factorized_{key}": value for key, value in factorized_topology.items()},
                    }
                )
                counters["rows"] += 1
                counters["R"] += int(record.compatible)
                counters["existing_A"] += int(existing_a)
                counters["existing_B_passive"] += int(existing_b_passive)
                counters["existing_B_factorized"] += int(existing_b_factorized)
                if len(rows) >= int(context["config"]["batch_size"]):
                    flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    if accepted_local != int(task["accepted_rows"]):
        raise AssertionError(
            f"{raw_path}: parsed {accepted_local} accepted rows, expected {task['accepted_rows']}"
        )
    temporary.replace(output_path)
    return {
        "path": str(output_path.resolve()),
        "rows": accepted_local,
        "source_start": start,
        "source_stop": start + accepted_local,
        "counters": dict(counters),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def build_semantic_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    lineage = json.loads((args.output / "lineage.json").read_text(encoding="utf-8"))
    chunk_root = args.output / "semantic_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    config = {
        "registry": str(args.registry.resolve()),
        "encoder": str(args.encoder.resolve()),
        "target_vocabs": str(args.target_vocabs.resolve()),
        "split_code": str((args.output / "source_to_split_code.npy").resolve()),
        "parent_row": str((args.output / "source_to_parent_row.npy").resolve()),
        "sampled": str((args.output / "source_to_existing_sample.npy").resolve()),
        "batch_size": args.batch_size,
    }
    config_path = args.output / "semantic_worker_config.json"
    atomic_json(config_path, config)
    tasks = []
    for index, file_record in enumerate(lineage["files"]):
        output_path = chunk_root / f"{index:02d}-{Path(file_record['path']).name}.parquet"
        tasks.append({**file_record, "output": str(output_path)})

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_semantic_worker_init,
        initargs=(str(config_path),),
    ) as executor:
        future_map = {executor.submit(_semantic_chunk, task): task for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(
                f"semantic rows={result['rows']:,} path={Path(result['path']).name}",
                file=sys.stderr,
                flush=True,
            )
    results.sort(key=lambda item: int(item["source_start"]))
    cursor = 0
    for result in results:
        if int(result["source_start"]) != cursor:
            raise AssertionError("semantic chunks do not form a contiguous source partition")
        cursor = int(result["source_stop"])
    if cursor != int(lineage["parent_rows"]):
        raise AssertionError("semantic sidecar does not cover the parent population")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "full_scan": True,
        "rows": cursor,
        "identity": "canonical raw RDF/Wikidata term strings",
        "runtime_seconds": time.monotonic() - started,
        "parent_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "worker_max_rss_kib": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        "chunks": results,
    }
    atomic_json(args.output / "semantic_manifest.json", manifest)
    return manifest


def semantic_chunk_paths(output: Path) -> list[Path]:
    manifest = json.loads((output / "semantic_manifest.json").read_text(encoding="utf-8"))
    return [Path(item["path"]) for item in manifest["chunks"]]


def write_semantic_registry_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    """Persist lossless normalized definitions referenced by semantic rows."""

    rows: list[dict[str, Any]] = []
    for constraint_id, entry in sorted(load_registry(args.registry).items()):
        scope = f"constraint:{constraint_id}"
        raw_predicates = list(entry.get("param_predicates") or ())
        raw_objects = list(entry.get("param_objects") or ())
        rows.append(
            {
                "constraint_id": constraint_id,
                "constraint_family": str(entry.get("constraint_family") or ""),
                "constraint_family_supported": bool(
                    entry.get("constraint_family_supported", False)
                ),
                "constrained_property": canonical_term(
                    entry.get("constrained_property"), blank_scope=scope
                ),
                "parameter_predicates": [
                    canonical_term(value, blank_scope=scope) for value in raw_predicates
                ],
                "parameter_objects": [
                    canonical_term(value, blank_scope=scope) for value in raw_objects
                ],
                "parameter_lengths_match": len(raw_predicates) == len(raw_objects),
            }
        )
    path = args.output / "semantic_constraint_definitions.parquet"
    temporary = path.with_suffix(".tmp.parquet")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    temporary.replace(path)
    return file_identity(path)


def derive_clean_vocab(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    definition_sidecar = write_semantic_registry_sidecar(args)
    encoder_encoding, encoder_decoding, canonical_to_id = load_encoder(args.encoder)
    entity_ids: set[int] = {0}
    predicate_ids: set[int] = {0}
    entity_roles: set[int] = set()
    predicate_roles: set[int] = set()
    for chunk_path in semantic_chunk_paths(args.output):
        parquet = pq.ParquetFile(chunk_path)
        columns = [
            "split",
            "R",
            *ROLE_NAMES,
            "add_subject",
            "add_predicate",
            "add_object",
            "del_subject",
            "del_predicate",
            "del_object",
        ]
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
            for row in batch.to_pylist():
                if row["split"] != "train" or not row["R"]:
                    continue
                roles = tuple(row[name] or EMPTY for name in ROLE_NAMES)
                slots = tuple(
                    row[name] or EMPTY
                    for name in (
                        "add_subject",
                        "add_predicate",
                        "add_object",
                        "del_subject",
                        "del_predicate",
                        "del_object",
                    )
                )
                for index, term in enumerate(slots):
                    target_set = predicate_ids if index in PREDICATE_SLOT_INDICES else entity_ids
                    target_roles = predicate_roles if index in PREDICATE_SLOT_INDICES else entity_roles
                    if not term:
                        target_set.add(0)
                        continue
                    mask = role_alias_mask(term, roles)
                    if mask:
                        # Match the builder's first-alias precedence when
                        # fitting the clean output mask.
                        target_roles.add(int(math.log2(mask & -mask)))
                        continue
                    encoder_id = int(canonical_to_id.get(term, 0))
                    if encoder_id:
                        target_set.add(encoder_id)

    role_encoder_ids = {role: int(encoder_encoding.get(role, 0)) for role in ROLE_NAMES}
    for role_index in entity_roles:
        entity_ids.add(role_encoder_ids[ROLE_NAMES[role_index]])
    for role_index in predicate_roles:
        predicate_ids.add(role_encoder_ids[ROLE_NAMES[role_index]])
    entity_terms = {
        canonical_term(encoder_decoding[value])
        for value in entity_ids
        if value in encoder_decoding
        and encoder_decoding[value]
        and encoder_decoding[value] not in ROLE_NAMES
        and encoder_decoding[value] != "unknown"
    }
    predicate_terms = {
        canonical_term(encoder_decoding[value])
        for value in predicate_ids
        if value in encoder_decoding
        and encoder_decoding[value]
        and encoder_decoding[value] not in ROLE_NAMES
        and encoder_decoding[value] != "unknown"
    }
    payload = {
        "schema_version": 1,
        "threshold": 100,
        "fit_population": "parent_pool/train",
        "fit_before_verified-fix filtering": True,
        "input_encoder_reused": str(args.encoder.resolve()),
        "input_encoder_size": len(encoder_encoding),
        "semantic_constraint_definitions": definition_sidecar,
        "entity_class_ids": sorted(entity_ids),
        "predicate_class_ids": sorted(predicate_ids),
        "entity_terms": sorted(entity_terms),
        "predicate_terms": sorted(predicate_terms),
        "entity_role_indices": sorted(entity_roles),
        "predicate_role_indices": sorted(predicate_roles),
        "fingerprint": hashlib.sha256(
            json.dumps(
                {"entity": sorted(entity_ids), "predicate": sorted(predicate_ids)},
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "runtime_seconds": time.monotonic() - started,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    atomic_json(args.output / "clean_parent_train_vocab.json", payload)
    return payload


_PROFILE_CONTEXT: dict[str, Any] = {}


def _install_profile(profile_root: Path, profile_name: str, registry: Path, hierarchy: Path) -> None:
    profile_src = str((profile_root / "src").resolve())
    if profile_src not in sys.path:
        sys.path.insert(0, profile_src)
    # No production validator module is imported by the parent before this
    # point.  Each profile command is a fresh process, and pool children fork
    # only after the selected revision is installed.
    from modules import reranker_eval as profile_eval

    encoder = SemanticEncoder()
    kwargs: dict[str, Any] = {
        "encoder": encoder,
        "assume_complete": True,
        "constraint_scope": "local",
        "use_encoded_ids": True,
    }
    if profile_name == "audited_v3_bounded":
        kwargs.update(hierarchy_path=hierarchy, require_hierarchy=True)
    evaluator = profile_eval.CandidateConstraintEvaluator(str(registry), **kwargs)
    _PROFILE_CONTEXT.clear()
    _PROFILE_CONTEXT.update(
        profile_name=profile_name,
        profile_root=str(profile_root.resolve()),
        evaluator=evaluator,
        profile_eval=profile_eval,
    )


def _profile_worker_init(
    profile_root: str,
    profile_name: str,
    registry: str,
    hierarchy: str,
) -> None:
    _install_profile(Path(profile_root), profile_name, Path(registry), Path(hierarchy))


def _semantic_row_to_namespace(row: Mapping[str, Any]) -> SimpleNamespace:
    def sid(value: Any) -> int:
        return semantic_id(value) if value else 0

    values: dict[str, Any] = {
        "constraint_type": row.get("family") or row.get("source_family") or "unknown",
        "constraint_id": sid(row.get("constraint_id")),
    }
    for name in ROLE_NAMES:
        values[name] = sid(row.get(name))
    for name in (
        "subject_predicates",
        "subject_objects",
        "object_predicates",
        "object_objects",
        "other_entity_predicates",
        "other_entity_objects",
    ):
        values[name] = [sid(value) for value in (row.get(name) or ())]
    for name in (
        "add_subject",
        "add_predicate",
        "add_object",
        "del_subject",
        "del_predicate",
        "del_object",
    ):
        values[name] = sid(row.get(name))
    constraint_id = values["constraint_id"]
    values["local_constraint_ids"] = [constraint_id]
    values["local_constraint_ids_focus"] = [constraint_id]
    return SimpleNamespace(**values)


def _checked_applicable(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    roles = tuple(str(row.get(name) or "") for name in ROLE_NAMES)
    subject, predicate, obj, other_subject, other_predicate, other_obj = roles
    owners = {subject, obj} - {EMPTY}
    if other_subject == subject and other_obj:
        owners.add(other_obj)
    elif other_obj == obj and other_subject:
        owners.add(other_subject)
    elif other_subject:
        owners.add(other_subject)
    elif other_obj:
        owners.add(other_obj)
    predicates = {predicate, other_predicate} - {EMPTY}
    for name in ("subject_predicates", "object_predicates", "other_entity_predicates"):
        predicates.update(value for value in (row.get(name) or ()) if value)
    reasons: list[str] = []
    for kind in ("del", "add"):
        triple = tuple(str(row.get(f"{kind}_{slot}") or "") for slot in ("subject", "predicate", "object"))
        if not any(triple):
            continue
        if not all(triple):
            reasons.append(f"{kind}:invalid_or_partial_operation")
            continue
        if triple[0] not in owners:
            reasons.append(f"{kind}:subject_outside_bounded_scope")
        if triple[1] not in predicates:
            reasons.append(f"{kind}:predicate_outside_bounded_scope")
    return not reasons, reasons


def _run_profile_chunk(task: Mapping[str, Any]) -> dict[str, Any]:
    semantic_path = Path(task["semantic_path"])
    output_path = Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    evaluator = _PROFILE_CONTEXT["evaluator"]
    profile_name = _PROFILE_CONTEXT["profile_name"]
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    def flush() -> None:
        nonlocal rows, writer
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", use_dictionary=True)
        writer.write_table(table)
        rows = []

    columns = [
        "source_index",
        "R",
        "family",
        "constraint_id",
        *ROLE_NAMES,
        "subject_predicates",
        "subject_objects",
        "object_predicates",
        "object_objects",
        "other_entity_predicates",
        "other_entity_objects",
        "add_subject",
        "add_predicate",
        "add_object",
        "del_subject",
        "del_predicate",
        "del_object",
    ]
    try:
        parquet = pq.ParquetFile(semantic_path)
        for batch in parquet.iter_batches(batch_size=2048, columns=columns):
            for raw_row in batch.to_pylist():
                source_index = int(raw_row["source_index"])
                if not raw_row["R"]:
                    result_row = {
                        "source_index": source_index,
                        "pre_outcome": OUTCOME_CODE["not_evaluated"],
                        "post_outcome": OUTCOME_CODE["not_evaluated"],
                        "edit_applicable": False,
                        "pre_unknown_reason": "outside_R",
                        "post_unknown_reason": "outside_R",
                        "unresolved_operations": json.dumps(["outside_R"]),
                    }
                else:
                    row = _semantic_row_to_namespace(raw_row)
                    slots = [
                        row.add_subject,
                        row.add_predicate,
                        row.add_object,
                        row.del_subject,
                        row.del_predicate,
                        row.del_object,
                    ]
                    details = evaluator.evaluate_full(
                        row,
                        candidate_slots=slots,
                        primary_factor_index=0,
                        factor_constraint_ids=[row.constraint_id],
                    )
                    if profile_name == "audited_v3_bounded":
                        pre_name = str(details["pre_outcomes"][0])
                        post_name = str(details["post_outcomes"][0])
                        applicable = bool(details["edit_applicable"])
                        pre_reasons = "; ".join(details["pre_unknown_reasons"][0])
                        post_reasons = "; ".join(details["post_unknown_reasons"][0])
                        unresolved = [
                            json.dumps(value, sort_keys=True) for value in details["unresolved_edits"]
                        ]
                    else:
                        pre_checkable = bool(details["pre_checkable"][0])
                        post_checkable = bool(details["post_checkable"][0])
                        pre_name = (
                            "satisfied"
                            if pre_checkable and bool(details["pre_satisfied"][0])
                            else "violated"
                            if pre_checkable
                            else "unknown"
                        )
                        post_name = (
                            "satisfied"
                            if post_checkable and bool(details["post_satisfied"][0])
                            else "violated"
                            if post_checkable
                            else "unknown"
                        )
                        applicable, unresolved = _checked_applicable(raw_row)
                        pre_reasons = "" if pre_checkable else "profile returned uncheckable"
                        post_reasons = "" if post_checkable else "profile returned uncheckable"
                    result_row = {
                        "source_index": source_index,
                        "pre_outcome": OUTCOME_CODE[pre_name],
                        "post_outcome": OUTCOME_CODE[post_name],
                        "edit_applicable": applicable,
                        "pre_unknown_reason": pre_reasons,
                        "post_unknown_reason": post_reasons,
                        "unresolved_operations": json.dumps(unresolved, sort_keys=True),
                    }
                    counters[f"transition::{pre_name}->{post_name}"] += 1
                    counters["edit_applicable"] += int(applicable)
                rows.append(result_row)
                counters["rows"] += 1
                if len(rows) >= 4096:
                    flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)
    return {
        "semantic_path": str(semantic_path.resolve()),
        "path": str(output_path.resolve()),
        "rows": int(counters["rows"]),
        "counters": dict(counters),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    profile_name = args.profile_name
    profile_root = args.checked_worktree if profile_name == "checked_out_bounded" else args.v3_worktree
    expected_revision = (
        "6c9f1818e2436e5bdfdc2023d01b3ec022af7596"
        if profile_name == "checked_out_bounded"
        else "018e2845de43bdb9480a1aa9a7791befb8b317ee"
    )
    actual_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=profile_root, text=True
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(f"{profile_name} worktree is {actual_revision}, expected {expected_revision}")
    if profile_name == "audited_v3_bounded" and not args.hierarchy.exists():
        raise FileNotFoundError(f"required hierarchy artifact is unavailable: {args.hierarchy}")

    semantic_paths = semantic_chunk_paths(args.output)
    result_root = args.output / "profile_chunks" / profile_name
    tasks = [
        {
            "semantic_path": str(path),
            "output_path": str(result_root / path.name),
        }
        for path in semantic_paths
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_profile_worker_init,
        initargs=(
            str(profile_root),
            profile_name,
            str(args.registry),
            str(args.hierarchy),
        ),
    ) as executor:
        future_map = {executor.submit(_run_profile_chunk, task): task for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(
                f"profile={profile_name} rows={result['rows']:,} path={Path(result['path']).name}",
                file=sys.stderr,
                flush=True,
            )
    by_semantic = {item["semantic_path"]: item for item in results}
    ordered = [by_semantic[str(path.resolve())] for path in semantic_paths]
    total = sum(int(item["rows"]) for item in ordered)
    expected = json.loads((args.output / "semantic_manifest.json").read_text(encoding="utf-8"))["rows"]
    if total != int(expected):
        raise AssertionError(f"profile {profile_name} rows {total} != semantic rows {expected}")
    code_files = [
        profile_root / "src/modules/constraint_checkers.py",
        profile_root / "src/modules/evidence_state.py",
        profile_root / "src/modules/reranker_eval.py",
    ]
    if profile_name == "audited_v3_bounded":
        code_files.append(profile_root / "src/modules/class_hierarchy.py")
    hierarchy_identity: dict[str, Any] | None = None
    if profile_name == "audited_v3_bounded":
        hierarchy_payload = json.loads(args.hierarchy.read_text(encoding="utf-8"))
        hierarchy_identity = {
            **file_identity(args.hierarchy),
            "content_sha256": hierarchy_payload.get("content_sha256"),
            "cutoff": hierarchy_payload.get("cutoff"),
            "seed_count": hierarchy_payload.get("seed_count"),
            "node_count": hierarchy_payload.get("node_count"),
            "edge_count": hierarchy_payload.get("edge_count"),
            "cycles": len(hierarchy_payload.get("cycles") or ()),
            "present_day_fallback": (hierarchy_payload.get("retrieval") or {}).get(
                "present_day_fallback"
            ),
        }
    manifest = {
        "schema_version": 1,
        "profile": profile_name,
        "revision": actual_revision,
        "full_scan": True,
        "rows": total,
        "configuration": {
            "assume_complete_entity_facts": True,
            "constraint_scope": "local",
            "primary_only": True,
            "identity_adapter": "reversible canonical-term integer encoding",
            "edit_order": "delete_then_add",
        },
        "code": {path.name: file_identity(path) for path in code_files},
        "registry": file_identity(args.registry),
        "hierarchy": hierarchy_identity,
        "runtime_seconds": time.monotonic() - started,
        "parent_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "worker_max_rss_kib": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        "chunks": ordered,
    }
    atomic_json(args.output / f"profile_{profile_name}.json", manifest)
    return manifest


def _profile_chunk_map(output: Path, profile: str) -> dict[str, Path]:
    manifest = json.loads((output / f"profile_{profile}.json").read_text(encoding="utf-8"))
    return {Path(item["semantic_path"]).name: Path(item["path"]) for item in manifest["chunks"]}


def _clean_slot_details(
    row: Mapping[str, Any],
    *,
    constants: Sequence[set[str]],
    roles: Sequence[set[int]],
    canonical_to_id: Mapping[str, int],
    input_profile: str,
) -> tuple[list[bool], list[bool], list[str]]:
    slots = [
        str(row.get(name) or "")
        for name in (
            "add_subject",
            "add_predicate",
            "add_object",
            "del_subject",
            "del_predicate",
            "del_object",
        )
    ]
    masks = list(row["target_role_masks"])
    input_flags = list(row[f"slot_in_{input_profile}_input"])
    primary_flags = list(row["slot_in_primary_parameters"])
    local_flags = list(row["slot_in_local_parameters"])
    fixed_flags: list[bool] = []
    copied_flags: list[bool] = []
    reasons: list[str] = []
    for index, term in enumerate(slots):
        permitted_alias = any(
            (int(masks[index]) & (1 << role_index)) != 0 for role_index in roles[index]
        )
        fixed = not term or permitted_alias or term in constants[index]
        copied = fixed or bool(input_flags[index])
        if not term:
            reason = "none"
        elif permitted_alias:
            reason = "role"
        elif term in constants[index]:
            reason = "output_constant"
        elif input_flags[index] and (primary_flags[index] or local_flags[index]):
            reason = "local_constraint_parameter"
        elif input_flags[index]:
            reason = "missing_role_present_local_neighbor"
        elif canonical_to_id.get(term, 0):
            reason = "known_constant_blocked_by_output_mask"
        else:
            reason = "outside_fixed_vocabulary_and_input"
        fixed_flags.append(fixed)
        copied_flags.append(copied)
        reasons.append(reason)
    return fixed_flags, copied_flags, reasons


def finalize_row_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    clean = json.loads((args.output / "clean_parent_train_vocab.json").read_text(encoding="utf-8"))
    _encoding, _decoding, canonical_to_id = load_encoder(args.encoder)
    clean_constants = [
        set(clean["predicate_terms"] if index in PREDICATE_SLOT_INDICES else clean["entity_terms"])
        for index in range(6)
    ]
    clean_roles = [
        set(clean["predicate_role_indices"])
        if index in PREDICATE_SLOT_INDICES
        else set(clean["entity_role_indices"])
        for index in range(6)
    ]
    profiles = ("checked_out_bounded", "audited_v3_bounded")
    profile_maps = {profile: _profile_chunk_map(args.output, profile) for profile in profiles}
    final_path = args.output / "row_level_audit.parquet"
    temporary = final_path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    final_columns = [
        "source_index",
        "source_key",
        "diagnostic_rank",
        "source_file",
        "source_line",
        "original_split",
        "split",
        "parent_row",
        "existing_sample",
        "population",
        "source_family",
        "family",
        "constraint_id",
        "revision",
        *ROLE_NAMES,
        "add_subject",
        "add_predicate",
        "add_object",
        "del_subject",
        "del_predicate",
        "del_object",
        "target_encoder_ids",
        "target_role_masks",
        "R",
        "exclusion_reason",
        "parse_warnings",
        "num_additions",
        "num_deletions",
        "edit_kind",
        "same_subject_property_replacement",
        "base_removed",
        "base_preserved_source_intent",
        "local_constraint_ids",
        "existing_slot_fixed",
        "existing_slot_passive",
        "existing_slot_factorized",
        "existing_reason_passive",
        "existing_reason_factorized",
        "existing_A",
        "existing_B_passive",
        "existing_B_factorized",
        "slot_in_passive_input",
        "slot_in_factorized_input",
        "slot_in_primary_parameters",
        "slot_in_local_parameters",
        "passive_pre_triples",
        "passive_distinct_semantic_terms",
        "passive_nodes",
        "passive_edges",
        "passive_unknown_feature_nodes",
        "factorized_pre_triples",
        "factorized_distinct_semantic_terms",
        "factorized_nodes",
        "factorized_edges",
        "factorized_unknown_feature_nodes",
    ]
    total = 0
    for semantic_path in semantic_chunk_paths(args.output):
        profile_data: dict[str, dict[str, list[Any]]] = {}
        for profile in profiles:
            table = pq.read_table(profile_maps[profile][semantic_path.name])
            profile_data[profile] = table.to_pydict()
        offset = 0
        parquet = pq.ParquetFile(semantic_path)
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=final_columns):
            batch_rows = batch.to_pylist()
            for local_index, row in enumerate(batch_rows):
                absolute = offset + local_index
                clean_results: dict[str, Any] = {}
                clean_fixed_reference: list[bool] | None = None
                for input_profile in ("passive", "factorized"):
                    fixed, copied, reasons = _clean_slot_details(
                        row,
                        constants=clean_constants,
                        roles=clean_roles,
                        canonical_to_id=canonical_to_id,
                        input_profile=input_profile,
                    )
                    if clean_fixed_reference is None:
                        clean_fixed_reference = fixed
                    elif clean_fixed_reference != fixed:
                        raise AssertionError("clean fixed coverage depends on input profile")
                    a = bool(row["R"] and all(fixed))
                    b = bool(row["R"] and all(copied))
                    if a and not b:
                        raise AssertionError("clean A is not a subset of B")
                    clean_results[f"clean_slot_{input_profile}"] = copied
                    clean_results[f"clean_reason_{input_profile}"] = reasons
                    clean_results[f"clean_B_{input_profile}"] = b
                clean_results["clean_slot_fixed"] = clean_fixed_reference
                clean_results["clean_A"] = bool(row["R"] and all(clean_fixed_reference or ()))
                slots = tuple(
                    str(row[name] or "")
                    for name in (
                        "add_subject",
                        "add_predicate",
                        "add_object",
                        "del_subject",
                        "del_predicate",
                        "del_object",
                    )
                )
                role_terms = tuple(str(row[name] or "") for name in ROLE_NAMES)
                if clean_results["clean_A"]:
                    exact, _resolved = roundtrip_slots(
                        slots,
                        role_terms,
                        role_indices_by_slot=clean_roles,
                        constants_by_slot=clean_constants,
                    )
                    if not exact:
                        raise AssertionError("clean no-copy edit failed semantic round-trip")
                for input_profile in ("passive", "factorized"):
                    if not clean_results[f"clean_B_{input_profile}"]:
                        continue
                    input_flags = row[f"slot_in_{input_profile}_input"]
                    copies = [
                        {term} if term and bool(input_flags[index]) else set()
                        for index, term in enumerate(slots)
                    ]
                    exact, _resolved = roundtrip_slots(
                        slots,
                        role_terms,
                        role_indices_by_slot=clean_roles,
                        constants_by_slot=clean_constants,
                        copies_by_slot=copies,
                    )
                    if not exact:
                        raise AssertionError(
                            f"clean {input_profile} copy edit failed semantic round-trip"
                        )
                row.update(clean_results)
                for profile in profiles:
                    pdata = profile_data[profile]
                    if int(pdata["source_index"][absolute]) != int(row["source_index"]):
                        raise AssertionError(f"{profile} output is not aligned with semantic sidecar")
                    prefix = "checked" if profile == "checked_out_bounded" else "v3"
                    row[f"{prefix}_profile_id"] = profile
                    row[f"{prefix}_pre_outcome"] = int(pdata["pre_outcome"][absolute])
                    row[f"{prefix}_post_outcome"] = int(pdata["post_outcome"][absolute])
                    row[f"{prefix}_edit_applicable"] = bool(pdata["edit_applicable"][absolute])
                    row[f"{prefix}_pre_unknown_reason"] = pdata["pre_unknown_reason"][absolute]
                    row[f"{prefix}_post_unknown_reason"] = pdata["post_unknown_reason"][absolute]
                    row[f"{prefix}_unresolved_operations"] = pdata["unresolved_operations"][absolute]
                    row[f"{prefix}_both_checkable"] = bool(
                        row["R"]
                        and row[f"{prefix}_pre_outcome"] != OUTCOME_CODE["unknown"]
                        and row[f"{prefix}_post_outcome"] != OUTCOME_CODE["unknown"]
                    )
                    row[f"{prefix}_E"] = bool(
                        row["R"] and row[f"{prefix}_pre_outcome"] == OUTCOME_CODE["violated"]
                    )
                    row[f"{prefix}_F"] = bool(
                        row[f"{prefix}_E"]
                        and row[f"{prefix}_post_outcome"] == OUTCOME_CODE["satisfied"]
                    )
                    row[f"{prefix}_F_fully_applied"] = bool(
                        row[f"{prefix}_F"] and row[f"{prefix}_edit_applicable"]
                    )
            table = pa.Table.from_pylist(batch_rows)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", use_dictionary=True)
            elif table.schema != writer.schema:
                # A whole source family can legitimately have no values for a
                # list field, which Arrow infers as list<null>.  The row-level
                # sidecar has one stable cross-family schema; casting an empty
                # list to the first chunk's concrete element type is lossless.
                table = table.cast(writer.schema)
            writer.write_table(table)
            offset += len(batch_rows)
            total += len(batch_rows)
        expected_rows = pq.ParquetFile(semantic_path).metadata.num_rows
        if offset != expected_rows:
            raise AssertionError("finalizer did not consume an entire semantic chunk")
    if writer is not None:
        writer.close()
    temporary.replace(final_path)
    expected = json.loads((args.output / "semantic_manifest.json").read_text(encoding="utf-8"))["rows"]
    if total != int(expected):
        raise AssertionError(f"final sidecar rows {total} != expected {expected}")
    manifest = {
        "path": str(final_path.resolve()),
        "rows": total,
        "sha256": sha256_file(final_path),
        "size_bytes": final_path.stat().st_size,
        "full_scan": True,
        "runtime_seconds": time.monotonic() - started,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    atomic_json(args.output / "row_level_manifest.json", manifest)
    return manifest


def validate_row_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    """Full-scan exact round-trip and subset validation of the persisted rows."""

    started = time.monotonic()
    encoding, decoding, _canonical_to_id = load_encoder(args.encoder)
    existing = _decode_output_vocab(encoding, decoding, args.target_vocabs)
    clean = json.loads((args.output / "clean_parent_train_vocab.json").read_text(encoding="utf-8"))
    vocabularies = {
        "existing": (
            [
                set(existing["predicate_terms"] if index in PREDICATE_SLOT_INDICES else existing["entity_terms"])
                for index in range(6)
            ],
            [
                set(existing["predicate_role_indices"] if index in PREDICATE_SLOT_INDICES else existing["entity_role_indices"])
                for index in range(6)
            ],
        ),
        "clean": (
            [
                set(clean["predicate_terms"] if index in PREDICATE_SLOT_INDICES else clean["entity_terms"])
                for index in range(6)
            ],
            [
                set(clean["predicate_role_indices"] if index in PREDICATE_SLOT_INDICES else clean["entity_role_indices"])
                for index in range(6)
            ],
        ),
    }
    columns = [
        *ROLE_NAMES,
        "add_subject",
        "add_predicate",
        "add_object",
        "del_subject",
        "del_predicate",
        "del_object",
        "existing_A",
        "existing_B_passive",
        "existing_B_factorized",
        "clean_A",
        "clean_B_passive",
        "clean_B_factorized",
        "slot_in_passive_input",
        "slot_in_factorized_input",
    ]
    counters: Counter[str] = Counter()
    parquet = pq.ParquetFile(args.output / "row_level_audit.parquet")
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        for row in batch.to_pylist():
            slots = tuple(
                str(row[name] or "")
                for name in (
                    "add_subject",
                    "add_predicate",
                    "add_object",
                    "del_subject",
                    "del_predicate",
                    "del_object",
                )
            )
            roles = tuple(str(row[name] or "") for name in ROLE_NAMES)
            for vocabulary, (constants, role_indices) in vocabularies.items():
                if row[f"{vocabulary}_A"]:
                    exact, _resolved = roundtrip_slots(
                        slots,
                        roles,
                        role_indices_by_slot=role_indices,
                        constants_by_slot=constants,
                    )
                    if not exact:
                        raise AssertionError(f"{vocabulary} A claim failed exact round-trip")
                    counters[f"{vocabulary}_A_roundtrips"] += 1
                for input_profile in ("passive", "factorized"):
                    b_column = f"{vocabulary}_B_{input_profile}"
                    if row[f"{vocabulary}_A"] and not row[b_column]:
                        raise AssertionError(f"{vocabulary} A is not a subset of {b_column}")
                    if not row[b_column]:
                        continue
                    input_flags = row[f"slot_in_{input_profile}_input"]
                    copies = [
                        {term} if term and bool(input_flags[index]) else set()
                        for index, term in enumerate(slots)
                    ]
                    exact, _resolved = roundtrip_slots(
                        slots,
                        roles,
                        role_indices_by_slot=role_indices,
                        constants_by_slot=constants,
                        copies_by_slot=copies,
                    )
                    if not exact:
                        raise AssertionError(f"{b_column} claim failed exact round-trip")
                    counters[f"{b_column}_roundtrips"] += 1
            counters["rows"] += 1
    expected = pq.ParquetFile(args.output / "row_level_audit.parquet").metadata.num_rows
    if counters["rows"] != expected:
        raise AssertionError("row validation did not cover the complete sidecar")
    manifest = {
        "schema_version": 1,
        "full_scan": True,
        "rows": int(counters["rows"]),
        "checks": {
            "all_claimed_edits_roundtrip_exactly": True,
            "no_copy_is_subset_of_copy": True,
        },
        "counts": dict(counters),
        "runtime_seconds": time.monotonic() - started,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    atomic_json(args.output / "validation_manifest.json", manifest)
    return manifest


def _group_summaries(frame: pd.DataFrame) -> Iterator[tuple[str, str, pd.DataFrame]]:
    yield "overall", "overall", frame
    for split, group in frame.groupby("split", sort=True, observed=True):
        yield str(split), "overall", group
    for family, group in frame.groupby("family", sort=True, observed=True):
        yield "overall", str(family), group
    for (split, family), group in frame.groupby(["split", "family"], sort=True, observed=True):
        yield str(split), str(family), group


def _group_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return every overall/split/family cell supported by ``frame``."""

    keys: list[tuple[str, str]] = [("overall", "overall")]
    keys.extend((str(value), "overall") for value in sorted(frame["split"].unique()))
    keys.extend(("overall", str(value)) for value in sorted(frame["family"].unique()))
    keys.extend(
        (str(split), str(family))
        for split, family in sorted(
            set(zip(frame["split"].astype(str), frame["family"].astype(str)))
        )
    )
    return keys


def _select_group(frame: pd.DataFrame, split: str, family: str) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    if split != "overall":
        mask &= frame["split"].astype(str).to_numpy() == split
    if family != "overall":
        mask &= frame["family"].astype(str).to_numpy() == family
    return frame.loc[mask]


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _aggregate_decisions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    populations = {
        "parent_pool": np.ones(len(frame), dtype=bool),
        "existing_sample": frame["existing_sample"].to_numpy(dtype=bool),
        "previously_discarded": ~frame["existing_sample"].to_numpy(dtype=bool),
    }
    boolean_columns = (
        "R",
        "existing_A",
        "clean_A",
        "existing_B_passive",
        "existing_B_factorized",
        "clean_B_passive",
        "clean_B_factorized",
        "checked_both_checkable",
        "checked_E",
        "checked_F",
        "checked_F_fully_applied",
        "v3_both_checkable",
        "v3_E",
        "v3_F",
        "v3_F_fully_applied",
    )
    flags = {column: frame[column].to_numpy(dtype=bool) for column in boolean_columns}
    for population, population_mask in populations.items():
        population_frame = frame.loc[population_mask]
        groups = [
            (split, family, group.index.to_numpy(dtype=np.int64, copy=False))
            for split, family, group in _group_summaries(population_frame)
        ]
        for profile, prefix in (
            ("checked_out_bounded", "checked"),
            ("audited_v3_bounded", "v3"),
        ):
            for vocabulary in ("existing_train_val_minocc100", "clean_parent_train_minocc100"):
                a_column = "existing_A" if vocabulary.startswith("existing") else "clean_A"
                for input_profile in ("passive", "factorized"):
                    b_column = (
                        f"existing_B_{input_profile}"
                        if vocabulary.startswith("existing")
                        else f"clean_B_{input_profile}"
                    )
                    for split, family, indices in groups:
                        r = flags["R"][indices]
                        a = flags[a_column][indices]
                        b = flags[b_column][indices]
                        both = flags[f"{prefix}_both_checkable"][indices]
                        e = flags[f"{prefix}_E"][indices]
                        f = flags[f"{prefix}_F"][indices]
                        fully = flags[f"{prefix}_F_fully_applied"][indices]
                        counts = {
                            "source_records": int(len(indices)),
                            "R": int(np.count_nonzero(r)),
                            "A": int(np.count_nonzero(a)),
                            "B": int(np.count_nonzero(b)),
                            "pre_post_both_checkable": int(np.count_nonzero(both)),
                            "E": int(np.count_nonzero(e)),
                            "F": int(np.count_nonzero(f)),
                            "F_and_A": int(np.count_nonzero(f & a)),
                            "F_and_B_minus_A": int(np.count_nonzero(f & b & ~a)),
                            "F_and_B": int(np.count_nonzero(f & b)),
                            "F_not_B": int(np.count_nonzero(f & ~b)),
                            "F_fully_applied": int(np.count_nonzero(fully)),
                            "F_fully_applied_and_A": int(np.count_nonzero(fully & a)),
                            "F_fully_applied_and_B": int(np.count_nonzero(fully & b)),
                        }
                        if counts["A"] > counts["B"]:
                            raise AssertionError("aggregate A exceeds B")
                        if counts["F_and_A"] + counts["F_and_B_minus_A"] != counts["F_and_B"]:
                            raise AssertionError("verified coverage partition does not reconcile")
                        if counts["F_and_B"] + counts["F_not_B"] != counts["F"]:
                            raise AssertionError("verified expressibility partition does not reconcile")
                        rows.append(
                            {
                                "population": population,
                                "validator_profile": profile,
                                "vocabulary_profile": vocabulary,
                                "input_profile": input_profile,
                                "split": split,
                                "family": family,
                                **counts,
                                "A_over_R": _safe_fraction(counts["A"], counts["R"]),
                                "B_over_R": _safe_fraction(counts["B"], counts["R"]),
                                "copy_only_gain": counts["B"] - counts["A"],
                                "copy_only_percentage_points_on_R": (
                                    100.0 * (counts["B"] - counts["A"]) / counts["R"]
                                    if counts["R"]
                                    else None
                                ),
                                "copy_relative_gain_over_A": (
                                    (counts["B"] - counts["A"]) / counts["A"]
                                    if counts["A"]
                                    else None
                                ),
                                "still_unexpressible": counts["R"] - counts["B"],
                                "F_and_A_over_F": _safe_fraction(counts["F_and_A"], counts["F"]),
                                "F_and_B_over_F": _safe_fraction(counts["F_and_B"], counts["F"]),
                            }
                        )
    return rows


def _aggregate_transitions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    populations = {
        "parent_pool": np.ones(len(frame), dtype=bool),
        "existing_sample": frame["existing_sample"].to_numpy(dtype=bool),
        "previously_discarded": ~frame["existing_sample"].to_numpy(dtype=bool),
    }
    outcomes = ("satisfied", "violated", "unknown")
    outcome_arrays = {
        column: frame[column].to_numpy()
        for column in (
            "checked_pre_outcome",
            "checked_post_outcome",
            "v3_pre_outcome",
            "v3_post_outcome",
        )
    }
    for population, population_mask in populations.items():
        population_frame = frame.loc[population_mask & frame["R"].to_numpy(dtype=bool)]
        groups = [
            (split, family, group.index.to_numpy(dtype=np.int64, copy=False))
            for split, family, group in _group_summaries(population_frame)
        ]
        for profile, prefix in (
            ("checked_out_bounded", "checked"),
            ("audited_v3_bounded", "v3"),
        ):
            for split, family, indices in groups:
                pre = outcome_arrays[f"{prefix}_pre_outcome"][indices]
                post = outcome_arrays[f"{prefix}_post_outcome"][indices]
                n = len(indices)
                e_count = int((pre == OUTCOME_CODE["violated"]).sum())
                for pre_name in outcomes:
                    for post_name in outcomes:
                        count = int(
                            (
                                (pre == OUTCOME_CODE[pre_name])
                                & (post == OUTCOME_CODE[post_name])
                            ).sum()
                        )
                        rows.append(
                            {
                                "population": population,
                                "validator_profile": profile,
                                "split": split,
                                "family": family,
                                "pre": pre_name,
                                "post": post_name,
                                "count": count,
                                "row_fraction": _safe_fraction(count, n),
                                "fixed_pre_violated_fraction": (
                                    _safe_fraction(count, e_count)
                                    if pre_name == "violated"
                                    else None
                                ),
                                "R": n,
                                "E": e_count,
                            }
                        )
                if sum(row["count"] for row in rows[-9:]) != n:
                    raise AssertionError("transition cells do not reconcile")
    return rows


def _aggregate_coverage(
    frame: pd.DataFrame, *, decisions: Sequence[Mapping[str, Any]] | None = None
) -> list[dict[str, Any]]:
    # Decoder coverage is validator-independent.  Deduplicate it from the
    # decision table rather than re-running a separate marginal calculation.
    decision_rows = list(decisions) if decisions is not None else _aggregate_decisions(frame)
    seen: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    for decision in decision_rows:
        key = tuple(
            decision[name]
            for name in (
                "population",
                "vocabulary_profile",
                "input_profile",
                "split",
                "family",
            )
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                name: decision[name]
                for name in (
                    "population",
                    "vocabulary_profile",
                    "input_profile",
                    "split",
                    "family",
                    "source_records",
                    "R",
                    "A",
                    "B",
                    "A_over_R",
                    "B_over_R",
                    "copy_only_gain",
                    "copy_only_percentage_points_on_R",
                    "copy_relative_gain_over_A",
                    "still_unexpressible",
                )
            }
        )
    return rows


def _aggregate_slot_coverage(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slot_names = (
        "add_subject",
        "add_predicate",
        "add_object",
        "del_subject",
        "del_predicate",
        "del_object",
    )
    populations = {
        "parent_pool": np.ones(len(frame), dtype=bool),
        "existing_sample": frame["existing_sample"].to_numpy(dtype=bool),
        "previously_discarded": ~frame["existing_sample"].to_numpy(dtype=bool),
    }
    flag_columns = (
        "existing_slot_fixed",
        "clean_slot_fixed",
        "existing_slot_passive",
        "existing_slot_factorized",
        "clean_slot_passive",
        "clean_slot_factorized",
    )
    flag_matrices = {
        column: np.asarray(frame[column].tolist(), dtype=np.bool_) for column in flag_columns
    }
    present_matrices = np.column_stack(
        [frame[column].fillna("").astype(str).to_numpy() != "" for column in slot_names]
    )
    for population, population_mask in populations.items():
        population_frame = frame.loc[population_mask & frame["R"].to_numpy(dtype=bool)]
        groups = [
            (split, family, group.index.to_numpy(dtype=np.int64, copy=False))
            for split, family, group in _group_summaries(population_frame)
        ]
        for vocabulary in ("existing_train_val_minocc100", "clean_parent_train_minocc100"):
            fixed_column = "existing_slot_fixed" if vocabulary.startswith("existing") else "clean_slot_fixed"
            for input_profile in ("passive", "factorized"):
                copy_column = (
                    f"existing_slot_{input_profile}"
                    if vocabulary.startswith("existing")
                    else f"clean_slot_{input_profile}"
                )
                fixed_matrix = flag_matrices[fixed_column]
                copy_matrix = flag_matrices[copy_column]
                for split, family, row_indices in groups:
                    for index, slot in enumerate(slot_names):
                        present = present_matrices[row_indices, index]
                        denominator = int(np.count_nonzero(present))
                        fixed_count = int(
                            np.count_nonzero(fixed_matrix[row_indices, index] & present)
                        )
                        copy_count = int(
                            np.count_nonzero(copy_matrix[row_indices, index] & present)
                        )
                        rows.append(
                            {
                                "population": population,
                                "vocabulary_profile": vocabulary,
                                "input_profile": input_profile,
                                "split": split,
                                "family": family,
                                "diagnostic": "populated_slot",
                                "slot_or_operation": slot,
                                "present": denominator,
                                "fixed_covered": fixed_count,
                                "fixed_or_copy_covered": copy_count,
                                "fixed_fraction": _safe_fraction(fixed_count, denominator),
                                "fixed_or_copy_fraction": _safe_fraction(copy_count, denominator),
                            }
                        )
                    for operation, slot_indices in (
                        ("addition", (0, 1, 2)),
                        ("deletion", (3, 4, 5)),
                    ):
                        # The operation exists iff its subject slot exists;
                        # compatible R rows have either three present slots or
                        # three genuinely absent slots.
                        subject_slot = slot_indices[0]
                        present = present_matrices[row_indices, subject_slot]
                        denominator = int(np.count_nonzero(present))
                        fixed_operation = fixed_matrix[
                            np.ix_(row_indices, slot_indices)
                        ].all(axis=1)
                        copied_operation = copy_matrix[
                            np.ix_(row_indices, slot_indices)
                        ].all(axis=1)
                        fixed_count = int(np.count_nonzero(fixed_operation & present))
                        copy_count = int(np.count_nonzero(copied_operation & present))
                        rows.append(
                            {
                                "population": population,
                                "vocabulary_profile": vocabulary,
                                "input_profile": input_profile,
                                "split": split,
                                "family": family,
                                "diagnostic": "present_operation",
                                "slot_or_operation": operation,
                                "present": denominator,
                                "fixed_covered": fixed_count,
                                "fixed_or_copy_covered": copy_count,
                                "fixed_fraction": _safe_fraction(fixed_count, denominator),
                                "fixed_or_copy_fraction": _safe_fraction(copy_count, denominator),
                            }
                        )
    return rows


def _aggregate_grounding_reasons(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    populations = {
        "parent_pool": np.ones(len(frame), dtype=bool),
        "existing_sample": frame["existing_sample"].to_numpy(dtype=bool),
        "previously_discarded": ~frame["existing_sample"].to_numpy(dtype=bool),
    }
    reason_columns = (
        "existing_reason_passive",
        "existing_reason_factorized",
        "clean_reason_passive",
        "clean_reason_factorized",
    )
    reason_matrices = {
        column: np.asarray(frame[column].tolist(), dtype=object) for column in reason_columns
    }
    for population, mask in populations.items():
        population_frame = frame.loc[mask & frame["R"].to_numpy(dtype=bool)]
        groups = [
            (split, family, group.index.to_numpy(dtype=np.int64, copy=False))
            for split, family, group in _group_summaries(population_frame)
        ]
        for vocabulary in ("existing_train_val_minocc100", "clean_parent_train_minocc100"):
            for input_profile in ("passive", "factorized"):
                reason_column = (
                    f"existing_reason_{input_profile}"
                    if vocabulary.startswith("existing")
                    else f"clean_reason_{input_profile}"
                )
                reason_matrix = reason_matrices[reason_column]
                for split, family, row_indices in groups:
                    values, counts = np.unique(
                        reason_matrix[row_indices].reshape(-1), return_counts=True
                    )
                    for reason, count in zip(values.tolist(), counts.tolist()):
                        if reason == "none":
                            continue
                        rows.append(
                            {
                                "population": population,
                                "vocabulary_profile": vocabulary,
                                "input_profile": input_profile,
                                "split": split,
                                "family": family,
                                "reason": str(reason),
                                "populated_slots": int(count),
                            }
                        )
    return rows


def _resource_value_summary(values: pd.Series, prefix: str) -> dict[str, Any]:
    if values.empty:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
            f"{prefix}_sum": 0,
        }
    numeric = values.astype(np.float64)
    return {
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_p95": float(numeric.quantile(0.95, interpolation="nearest")),
        f"{prefix}_max": int(numeric.max()),
        f"{prefix}_sum": int(numeric.sum()),
    }


def _aggregate_resources(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    populations = {
        "parent_pool": np.ones(len(frame), dtype=bool),
        "existing_sample": frame["existing_sample"].to_numpy(dtype=bool),
        "previously_discarded": ~frame["existing_sample"].to_numpy(dtype=bool),
    }
    metric_names = (
        "pre_triples",
        "distinct_semantic_terms",
        "nodes",
        "edges",
        "unknown_feature_nodes",
    )
    metric_matrices = {
        input_profile: frame[
            [f"{input_profile}_{metric}" for metric in metric_names]
        ].to_numpy(dtype=np.float64)
        for input_profile in ("passive", "factorized")
    }
    for population, population_mask in populations.items():
        population_frame = frame.loc[population_mask]
        r_frame = population_frame.loc[population_frame["R"].astype(bool)]
        group_keys = _group_keys(r_frame)
        r_counts = {
            (split, family): len(group)
            for split, family, group in _group_summaries(r_frame)
        }
        empty_frame = population_frame.iloc[:0]
        for profile, prefix in (
            ("checked_out_bounded", "checked"),
            ("audited_v3_bounded", "v3"),
        ):
            f_mask = population_frame[f"{prefix}_F"].astype(bool)
            for vocabulary in ("existing_train_val_minocc100", "clean_parent_train_minocc100"):
                a_column = "existing_A" if vocabulary.startswith("existing") else "clean_A"
                a_mask = population_frame[a_column].astype(bool)
                for input_profile in ("passive", "factorized"):
                    b_column = (
                        f"existing_B_{input_profile}"
                        if vocabulary.startswith("existing")
                        else f"clean_B_{input_profile}"
                    )
                    b_mask = population_frame[b_column].astype(bool)
                    subset_masks = {
                        "R": population_frame["R"].astype(bool),
                        "F": f_mask,
                        "F_and_A": f_mask & a_mask,
                        "F_and_B": f_mask & b_mask,
                        "F_and_B_minus_A": f_mask & b_mask & ~a_mask,
                    }
                    for subset, subset_mask in subset_masks.items():
                        subset_frame = population_frame.loc[subset_mask]
                        subset_groups = {
                            (split, family): group
                            for split, family, group in _group_summaries(subset_frame)
                        }
                        for split, family in group_keys:
                            group = subset_groups.get((split, family), empty_frame)
                            count = len(group)
                            r_count = r_counts[(split, family)]
                            record: dict[str, Any] = {
                                "population": population,
                                "validator_profile": profile,
                                "vocabulary_profile": vocabulary,
                                "input_profile": input_profile,
                                "subset": subset,
                                "split": split,
                                "family": family,
                                "records": count,
                                "distinct_definitions": int(group["constraint_id"].nunique()),
                                "loss_from_R": r_count - count,
                                "retention_from_R": _safe_fraction(count, r_count),
                                "addition_only": int((group["edit_kind"] == "addition_only").sum()),
                                "deletion_only": int((group["edit_kind"] == "deletion_only").sum()),
                                "paired": int((group["edit_kind"] == "paired").sum()),
                                "noop": int((group["edit_kind"] == "noop").sum()),
                                "same_subject_property_replacement": int(
                                    group["same_subject_property_replacement"].sum()
                                ),
                                "base_removed": int(group["base_removed"].sum()),
                                "base_preserved": int(group["base_preserved_source_intent"].sum()),
                                "unresolved_operations": int(
                                    (~group[f"{prefix}_edit_applicable"].astype(bool)).sum()
                                ),
                                "primary_pre_satisfied": int(
                                    (group[f"{prefix}_pre_outcome"] == OUTCOME_CODE["satisfied"]).sum()
                                ),
                                "primary_pre_violated": int(
                                    (group[f"{prefix}_pre_outcome"] == OUTCOME_CODE["violated"]).sum()
                                ),
                                "primary_pre_unknown": int(
                                    (group[f"{prefix}_pre_outcome"] == OUTCOME_CODE["unknown"]).sum()
                                ),
                                "primary_post_satisfied": int(
                                    (group[f"{prefix}_post_outcome"] == OUTCOME_CODE["satisfied"]).sum()
                                ),
                                "primary_post_violated": int(
                                    (group[f"{prefix}_post_outcome"] == OUTCOME_CODE["violated"]).sum()
                                ),
                                "primary_post_unknown": int(
                                    (group[f"{prefix}_post_outcome"] == OUTCOME_CODE["unknown"]).sum()
                                ),
                            }
                            if group.empty:
                                for metric in metric_names:
                                    record.update(_resource_value_summary(pd.Series(dtype=float), metric))
                            else:
                                indices = group.index.to_numpy(dtype=np.int64, copy=False)
                                numeric = metric_matrices[input_profile][indices]
                                means = numeric.mean(axis=0)
                                medians = np.median(numeric, axis=0)
                                p95s = np.quantile(numeric, 0.95, axis=0, method="nearest")
                                maxima = numeric.max(axis=0)
                                sums = numeric.sum(axis=0)
                                for index, metric in enumerate(metric_names):
                                    record.update(
                                        {
                                            f"{metric}_mean": float(means[index]),
                                            f"{metric}_median": float(medians[index]),
                                            f"{metric}_p95": float(p95s[index]),
                                            f"{metric}_max": int(maxima[index]),
                                            f"{metric}_sum": int(sums[index]),
                                        }
                                    )
                            rows.append(record)
    return rows


def _parameter_estimates(args: argparse.Namespace) -> list[dict[str, Any]]:
    with args.encoder.open("rb") as handle:
        encoding, _decoding, _filtered, _mapping = pickle.load(handle)
    clean = json.loads((args.output / "clean_parent_train_vocab.json").read_text(encoding="utf-8"))
    existing_payload = json.loads(args.target_vocabs.read_text(encoding="utf-8"))["per_split"]
    existing_entity = len(
        set(existing_payload["train"]["entity_class_ids"])
        | set(existing_payload["val"]["entity_class_ids"])
    )
    existing_predicate = len(
        set(existing_payload["train"]["predicate_class_ids"])
        | set(existing_payload["val"]["predicate_class_ids"])
    )
    vocabulary_sizes = {
        "existing_train_val_minocc100": (existing_entity, existing_predicate),
        "clean_parent_train_minocc100": (
            len(clean["entity_class_ids"]),
            len(clean["predicate_class_ids"]),
        ),
    }
    model_paths = {
        "Direct-Factor": REPO_ROOT
        / "models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id/config.json",
        "Direct-Passive": REPO_ROOT
        / "models/b0_eswc_reproduction__full_strat1m_minocc100__node_id/config.json",
    }
    rows: list[dict[str, Any]] = []
    for model, path in model_paths.items():
        config = json.loads(path.read_text(encoding="utf-8"))["model_config"]
        embedding_dim = int(config["num_embedding_size"])
        branch_hidden = int(config.get("branch_hidden") or config.get("head_hidden") or config["hidden_channels"])
        for vocabulary, (entity_size, predicate_size) in vocabulary_sizes.items():
            input_parameters = len(encoding) * embedding_dim
            output_parameters = (4 * entity_size + 2 * predicate_size) * (branch_hidden + 1)
            rows.append(
                {
                    "model": model,
                    "constraint_representation": config["constraint_representation"],
                    "threshold": 100,
                    "vocabulary_profile": vocabulary,
                    "input_vocabulary_size": len(encoding),
                    "entity_output_classes": entity_size,
                    "predicate_output_classes": predicate_size,
                    "embedding_dimension": embedding_dim,
                    "output_branch_hidden": branch_hidden,
                    "input_embedding_parameters": input_parameters,
                    "six_output_head_parameters": output_parameters,
                    "accounted_parameters": input_parameters + output_parameters,
                }
            )
    return rows


def _diagnostic_examples(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_columns = [
        "source_key",
        "split",
        "population",
        "family",
        "constraint_id",
        "add_subject",
        "add_predicate",
        "add_object",
        "del_subject",
        "del_predicate",
        "del_object",
    ]
    categories = (
        "missing_role_present_local_neighbor",
        "local_constraint_parameter",
        "known_constant_blocked_by_output_mask",
        "outside_fixed_vocabulary_and_input",
        "unsupported_term_or_operation",
    )
    for category in categories:
        if category == "unsupported_term_or_operation":
            candidates = frame.loc[~frame["R"].astype(bool)]
        else:
            candidates = frame.loc[
                frame["clean_reason_factorized"].map(
                    lambda reasons: reasons is not None and category in reasons
                )
            ]
        support = len(candidates)
        selected = candidates.nsmallest(5, "diagnostic_rank")
        if selected.empty:
            rows.append(
                {
                    "diagnostic": "uncovered_category",
                    "profile": "clean_parent_train_minocc100/factorized",
                    "category": category,
                    "support": support,
                    **{column: None for column in base_columns},
                }
            )
        for _, row in selected.iterrows():
            rows.append(
                {
                    "diagnostic": "uncovered_category",
                    "profile": "clean_parent_train_minocc100/factorized",
                    "category": category,
                    "support": support,
                    **{column: row[column] for column in base_columns},
                }
            )
    for profile, prefix in (
        ("checked_out_bounded", "checked"),
        ("audited_v3_bounded", "v3"),
    ):
        for pre_name in ("satisfied", "violated", "unknown"):
            for post_name in ("satisfied", "violated", "unknown"):
                if pre_name == "violated" and post_name == "satisfied":
                    continue
                candidates = frame.loc[
                    frame["R"].astype(bool)
                    & (frame[f"{prefix}_pre_outcome"] == OUTCOME_CODE[pre_name])
                    & (frame[f"{prefix}_post_outcome"] == OUTCOME_CODE[post_name])
                ]
                support = len(candidates)
                selected = candidates.nsmallest(5, "diagnostic_rank")
                if selected.empty:
                    rows.append(
                        {
                            "diagnostic": "non_F_transition",
                            "profile": profile,
                            "category": f"{pre_name}->{post_name}",
                            "support": support,
                            **{column: None for column in base_columns},
                        }
                    )
                for _, row in selected.iterrows():
                    rows.append(
                        {
                            "diagnostic": "non_F_transition",
                            "profile": profile,
                            "category": f"{pre_name}->{post_name}",
                            "support": support,
                            **{column: row[column] for column in base_columns},
                        }
                    )
    return rows


def _json_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def convert(value: Any) -> Any:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    return [{key: convert(value) for key, value in row.items()} for row in rows]


def _write_json_and_csv(output: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> None:
    normalized = _json_records(rows)
    atomic_json(output / f"{stem}.json", {"rows": normalized})
    atomic_csv(output / f"{stem}.csv", normalized)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    row_path = args.output / "row_level_audit.parquet"
    frame = pd.read_parquet(row_path)
    print(f"aggregate loaded rows={len(frame):,}", file=sys.stderr, flush=True)

    def materialize(stem: str, build: Any) -> list[dict[str, Any]]:
        path = args.output / f"{stem}.json"
        if args.resume and path.exists() and (args.output / f"{stem}.csv").exists():
            print(f"aggregate resume: {stem}", file=sys.stderr, flush=True)
            return json.loads(path.read_text(encoding="utf-8"))["rows"]
        result = build()
        _write_json_and_csv(args.output, stem, result)
        return result

    decisions = materialize("subset_decisions", lambda: _aggregate_decisions(frame))
    print("aggregate decisions complete", file=sys.stderr, flush=True)
    transitions = materialize("primary_transitions", lambda: _aggregate_transitions(frame))
    print("aggregate transitions complete", file=sys.stderr, flush=True)
    coverage = materialize(
        "decoder_coverage", lambda: _aggregate_coverage(frame, decisions=decisions)
    )
    resources = materialize("resource_estimates", lambda: _aggregate_resources(frame))
    print("aggregate resources complete", file=sys.stderr, flush=True)
    slot_coverage = materialize("slot_coverage", lambda: _aggregate_slot_coverage(frame))
    print("aggregate slot coverage complete", file=sys.stderr, flush=True)
    grounding_reasons = materialize(
        "grounding_reasons", lambda: _aggregate_grounding_reasons(frame)
    )
    print("aggregate grounding reasons complete", file=sys.stderr, flush=True)
    parameters = materialize("parameter_estimates", lambda: _parameter_estimates(args))
    examples = materialize(
        "diagnostic_examples_seed1729", lambda: _diagnostic_examples(frame)
    )
    print("aggregate diagnostics complete", file=sys.stderr, flush=True)

    # Parent = sample + complement must hold for every additive decision count.
    additive = (
        "source_records",
        "R",
        "A",
        "B",
        "pre_post_both_checkable",
        "E",
        "F",
        "F_and_A",
        "F_and_B_minus_A",
        "F_and_B",
        "F_not_B",
        "F_fully_applied",
        "F_fully_applied_and_A",
        "F_fully_applied_and_B",
    )
    keyed: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in decisions:
        key = tuple(
            row[name]
            for name in (
                "validator_profile",
                "vocabulary_profile",
                "input_profile",
                "split",
                "family",
            )
        )
        keyed[key][str(row["population"])] = row
    for key, populations in keyed.items():
        if set(populations) != {"parent_pool", "existing_sample", "previously_discarded"}:
            raise AssertionError(f"population partition missing for {key}")
        for name in additive:
            parent = int(populations["parent_pool"][name])
            children = int(populations["existing_sample"][name]) + int(
                populations["previously_discarded"][name]
            )
            if parent != children:
                raise AssertionError(f"parent/sample/complement mismatch for {key} {name}")

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "full_scan": True,
        "rows": len(frame),
        "runtime_seconds": time.monotonic() - started,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "tables": {
            stem: {
                "json": file_identity(args.output / f"{stem}.json"),
                "csv": file_identity(args.output / f"{stem}.csv"),
            }
            for stem in (
                "subset_decisions",
                "primary_transitions",
                "decoder_coverage",
                "resource_estimates",
                "slot_coverage",
                "grounding_reasons",
                "parameter_estimates",
                "diagnostic_examples_seed1729",
            )
        },
    }
    atomic_json(args.output / "aggregate_manifest.json", manifest)
    return manifest


def _fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def _fmt_pct(numerator: Any, denominator: Any) -> str:
    denominator = int(denominator)
    return "undefined" if denominator == 0 else f"{100.0 * int(numerator) / denominator:.2f}%"


def _markdown_table(headers: Sequence[str], records: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if index else "---" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in record) + " |" for record in records)
    return "\n".join(lines)


def generate_report(args: argparse.Namespace) -> Path:
    decisions = json.loads((args.output / "subset_decisions.json").read_text(encoding="utf-8"))["rows"]
    transitions = json.loads((args.output / "primary_transitions.json").read_text(encoding="utf-8"))["rows"]
    resources = json.loads((args.output / "resource_estimates.json").read_text(encoding="utf-8"))["rows"]
    grounding = json.loads((args.output / "grounding_reasons.json").read_text(encoding="utf-8"))["rows"]
    parameters = json.loads((args.output / "parameter_estimates.json").read_text(encoding="utf-8"))["rows"]
    lineage = json.loads((args.output / "lineage.json").read_text(encoding="utf-8"))
    clean_vocab = json.loads((args.output / "clean_parent_train_vocab.json").read_text(encoding="utf-8"))
    operation_patterns: Counter[str] = Counter()
    for source_file in lineage["files"]:
        operation_patterns.update(source_file.get("operation_patterns") or {})

    headline = [
        row
        for row in decisions
        if row["vocabulary_profile"] == "clean_parent_train_minocc100"
        and row["input_profile"] == "factorized"
        and row["family"] == "overall"
    ]
    headline.sort(
        key=lambda row: (
            row["validator_profile"],
            ("parent_pool", "existing_sample", "previously_discarded").index(row["population"]),
            ("overall", "train", "val", "test").index(row["split"]),
        )
    )
    headline_table = _markdown_table(
        (
            "Population/profile",
            "Split",
            "N source",
            "N R",
            "No-copy A",
            "With-copy B",
            "Verified F",
            "F & A",
            "F & (B-A)",
            "F & B",
            "F & not B",
        ),
        [
            (
                f"{row['population']} / {row['validator_profile']}",
                row["split"],
                _fmt_int(row["source_records"]),
                _fmt_int(row["R"]),
                _fmt_int(row["A"]),
                _fmt_int(row["B"]),
                _fmt_int(row["F"]),
                _fmt_int(row["F_and_A"]),
                _fmt_int(row["F_and_B_minus_A"]),
                _fmt_int(row["F_and_B"]),
                _fmt_int(row["F_not_B"]),
            )
            for row in headline
        ],
    )

    family_rows = [
        row
        for row in decisions
        if row["population"] == "parent_pool"
        and row["validator_profile"] == "audited_v3_bounded"
        and row["vocabulary_profile"] == "clean_parent_train_minocc100"
        and row["input_profile"] == "factorized"
        and row["split"] == "overall"
        and row["family"] != "overall"
    ]
    family_rows.sort(key=lambda row: row["family"])
    family_table = _markdown_table(
        ("Family", "N R", "A", "B", "F", "F&A", "F&(B-A)", "F&B", "F-not-B", "F&A/F", "F&B/F"),
        [
            (
                row["family"],
                _fmt_int(row["R"]),
                _fmt_int(row["A"]),
                _fmt_int(row["B"]),
                _fmt_int(row["F"]),
                _fmt_int(row["F_and_A"]),
                _fmt_int(row["F_and_B_minus_A"]),
                _fmt_int(row["F_and_B"]),
                _fmt_int(row["F_not_B"]),
                _fmt_pct(row["F_and_A"], row["F"]),
                _fmt_pct(row["F_and_B"], row["F"]),
            )
            for row in family_rows
        ],
    )

    comparison = [
        row
        for row in decisions
        if row["population"] == "parent_pool"
        and row["validator_profile"] == "audited_v3_bounded"
        and row["input_profile"] == "factorized"
        and row["split"] == "overall"
        and row["family"] == "overall"
    ]
    comparison.sort(key=lambda row: row["vocabulary_profile"])
    vocabulary_table = _markdown_table(
        ("Vocabulary", "A/R", "B/R", "copy-only", "gain pp", "F&A/F", "F&B/F"),
        [
            (
                row["vocabulary_profile"],
                _fmt_pct(row["A"], row["R"]),
                _fmt_pct(row["B"], row["R"]),
                _fmt_int(row["copy_only_gain"]),
                f"{row['copy_only_percentage_points_on_R']:.2f}",
                _fmt_pct(row["F_and_A"], row["F"]),
                _fmt_pct(row["F_and_B"], row["F"]),
            )
            for row in comparison
        ],
    )

    def transition_matrix(profile: str) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
        lookup = {
            (row["pre"], row["post"]): row
            for row in transitions
            if row["population"] == "parent_pool"
            and row["validator_profile"] == profile
            and row["split"] == "overall"
            and row["family"] == "overall"
        }
        table = _markdown_table(
            ("Pre \\ Post", "satisfied", "violated", "unknown"),
            [
                (
                    pre,
                    *(
                        _fmt_int(lookup[(pre, post)]["count"])
                        for post in ("satisfied", "violated", "unknown")
                    ),
                )
                for pre in ("satisfied", "violated", "unknown")
            ],
        )
        return lookup, table

    transition_lookup, transition_table = transition_matrix("audited_v3_bounded")
    checked_transition_lookup, checked_transition_table = transition_matrix(
        "checked_out_bounded"
    )
    v3_overall = next(
        row
        for row in comparison
        if row["vocabulary_profile"] == "clean_parent_train_minocc100"
    )
    e_outcomes = {
        post: transition_lookup[("violated", post)]["count"]
        for post in ("satisfied", "violated", "unknown")
    }
    checked_overall = next(
        row
        for row in decisions
        if row["population"] == "parent_pool"
        and row["validator_profile"] == "checked_out_bounded"
        and row["vocabulary_profile"] == "clean_parent_train_minocc100"
        and row["input_profile"] == "factorized"
        and row["split"] == "overall"
        and row["family"] == "overall"
    )
    checked_e_outcomes = {
        post: checked_transition_lookup[("violated", post)]["count"]
        for post in ("satisfied", "violated", "unknown")
    }

    copy_profile_rows = [
        row
        for row in decisions
        if row["population"] == "parent_pool"
        and row["validator_profile"] == "audited_v3_bounded"
        and row["vocabulary_profile"] == "clean_parent_train_minocc100"
        and row["split"] == "overall"
        and row["family"] == "overall"
    ]
    copy_profile_rows.sort(key=lambda row: row["input_profile"])
    copy_profile_table = _markdown_table(
        ("Input/copy scope", "A", "B", "B-A", "F&A", "F&(B-A)", "F&B", "F-not-B"),
        [
            (
                row["input_profile"],
                _fmt_int(row["A"]),
                _fmt_int(row["B"]),
                _fmt_int(row["copy_only_gain"]),
                _fmt_int(row["F_and_A"]),
                _fmt_int(row["F_and_B_minus_A"]),
                _fmt_int(row["F_and_B"]),
                _fmt_int(row["F_not_B"]),
            )
            for row in copy_profile_rows
        ],
    )
    grounding_counts = {
        row["reason"]: row["populated_slots"]
        for row in grounding
        if row["population"] == "parent_pool"
        and row["vocabulary_profile"] == "clean_parent_train_minocc100"
        and row["input_profile"] == "factorized"
        and row["split"] == "overall"
        and row["family"] == "overall"
    }
    grounding_table = _markdown_table(
        ("Grounding category", "Populated slots"),
        [
            (reason, _fmt_int(grounding_counts.get(reason, 0)))
            for reason in (
                "missing_role_present_local_neighbor",
                "local_constraint_parameter",
                "known_constant_blocked_by_output_mask",
                "outside_fixed_vocabulary_and_input",
                "unsupported_term_or_operation",
            )
        ],
    )

    resource_rows = [
        row
        for row in resources
        if row["population"] == "parent_pool"
        and row["validator_profile"] == "audited_v3_bounded"
        and row["vocabulary_profile"] == "clean_parent_train_minocc100"
        and row["input_profile"] == "factorized"
        and row["split"] == "overall"
        and row["family"] == "overall"
    ]
    resource_rows.sort(key=lambda row: ("R", "F", "F_and_A", "F_and_B", "F_and_B_minus_A").index(row["subset"]))
    resource_table = _markdown_table(
        ("Subset", "N", "definitions", "add", "del", "paired", "noop", "same-S/P replacement", "base removed", "unresolved", "nodes mean/p95/max", "UNK nodes mean", "edge workload"),
        [
            (
                row["subset"],
                _fmt_int(row["records"]),
                _fmt_int(row["distinct_definitions"]),
                _fmt_int(row["addition_only"]),
                _fmt_int(row["deletion_only"]),
                _fmt_int(row["paired"]),
                _fmt_int(row["noop"]),
                _fmt_int(row["same_subject_property_replacement"]),
                _fmt_int(row["base_removed"]),
                _fmt_int(row["unresolved_operations"]),
                f"{row['nodes_mean']:.1f}/{row['nodes_p95']:.0f}/{row['nodes_max']}",
                f"{row['unknown_feature_nodes_mean']:.1f}",
                _fmt_int(row["edges_sum"]),
            )
            for row in resource_rows
        ],
    )
    parameter_table = _markdown_table(
        ("Model", "Vocabulary", "input vocab", "E/P outputs", "input embedding params", "six-head params"),
        [
            (
                row["model"],
                row["vocabulary_profile"],
                _fmt_int(row["input_vocabulary_size"]),
                f"{_fmt_int(row['entity_output_classes'])}/{_fmt_int(row['predicate_output_classes'])}",
                _fmt_int(row["input_embedding_parameters"]),
                _fmt_int(row["six_output_head_parameters"]),
            )
            for row in parameters
        ],
    )

    train_row = next(
        row
        for row in headline
        if row["population"] == "parent_pool"
        and row["validator_profile"] == "audited_v3_bounded"
        and row["split"] == "train"
    )
    old_budget = lineage["sample_split_counts"]["train"]
    largest_family_gain = max(
        family_rows,
        key=lambda row: (
            (row["F_and_B"] - row["F_and_A"]) / row["F"] if row["F"] else -1
        ),
    )
    lowest_family_retention = min(
        (row for row in family_rows if row["F"]),
        key=lambda row: row["F_and_A"] / row["F"],
    )
    conclusion = (
        f"Under audited v3 plus the clean parent-train vocabulary, no-copy retains "
        f"{_fmt_int(v3_overall['F_and_A'])}/{_fmt_int(v3_overall['F'])} "
        f"({_fmt_pct(v3_overall['F_and_A'], v3_overall['F'])}) verified fixes. "
        f"Hypothetical factorized-input copying raises this to {_fmt_int(v3_overall['F_and_B'])} "
        f"({_fmt_pct(v3_overall['F_and_B'], v3_overall['F'])}), an exact copy-only verified gain of "
        f"{_fmt_int(v3_overall['F_and_B_minus_A'])}. The largest proportional family gain is "
        f"{largest_family_gain['family']} ({_fmt_int(largest_family_gain['F_and_B_minus_A'])} rows). "
        f"The weakest no-copy family is {lowest_family_retention['family']} at "
        f"{_fmt_pct(lowest_family_retention['F_and_A'], lowest_family_retention['F'])}; "
        f"copying reaches {_fmt_pct(lowest_family_retention['F_and_B'], lowest_family_retention['F'])}. "
        f"The parent training split contains {_fmt_int(train_row['F_and_A'])} verified no-copy rows, "
        f"compared with the old sampled training-row budget of {_fmt_int(old_budget)}. These are "
        "measurements, not an architecture or final-dataset selection rule."
    )

    report = f"""# Decoder Coverage and Verified-Fix Feasibility Audit

**Scan status:** complete full CPU scan of all **{lineage['parent_rows']:,}** parent records; the
**{lineage['sample_split_counts']['train'] + lineage['sample_split_counts']['val'] + lineage['sample_split_counts']['test']:,}** existing-sample rows and their exact complement are identified. Both
`checked_out_bounded` (`6c9f1818…`) and `audited_v3_bounded` (`018e2845…`) are complete.

Source accounting: {lineage['source_rows']:,} raw rows, {lineage['parent_rows']:,} R rows,
{lineage['unknown_constraint_rows']:,} unknown-constraint exclusions, and
{lineage['malformed_rows']:,} malformed rows. Operations comprise
{operation_patterns['addition']:,} addition-only, {operation_patterns['deletion']:,} deletion-only,
{operation_patterns['addition+deletion'] + operation_patterns['deletion+addition']:,} paired, and
{operation_patterns['noop']:,} no-op records; no source row has an incomplete or multi-add/multi-delete edit.

The headline table uses the clean threshold-100 parent-training vocabulary and the hypothetical
factorized-input copy domain. No copy head was implemented or trained.

## Joint decision table

{headline_table}

## Audited-v3 family intersections (parent population)

{family_table}

## Frozen-vocabulary comparison

{vocabulary_table}

The existing vocabulary is the actual training-time union of sampled train and validation target
classes; it is not train-only. The clean vocabulary uses only the established parent training split,
with the existing threshold-100 encoder and independently supplied registry metadata frozen before
evaluation. Threshold 50 is not reported because exact pre-pruning training feature frequencies were
not retained; no threshold-50 production configuration was synthesized in this measurement task.

## Copy-domain comparison (audited v3, clean vocabulary, parent population)

{copy_profile_table}

Uncovered/populated-slot grounding diagnostics (the first two categories are copy-addressable):

{grounding_table}

## Audited-v3 primary transitions (parent population)

{transition_table}

For the fixed pre-violated set E={v3_overall['E']:,}: post satisfied
{e_outcomes['satisfied']:,}/{v3_overall['E']:,} ({_fmt_pct(e_outcomes['satisfied'], v3_overall['E'])}),
post unknown {e_outcomes['unknown']:,}/{v3_overall['E']:,} ({_fmt_pct(e_outcomes['unknown'], v3_overall['E'])}),
and post violated {e_outcomes['violated']:,}/{v3_overall['E']:,} ({_fmt_pct(e_outcomes['violated'], v3_overall['E'])}).
Distinct-values results are **bounded-local**, not global uniqueness certificates.

## Checked-out comparator primary transitions (parent population)

{checked_transition_table}

For its fixed pre-violated set E={checked_overall['E']:,}: post satisfied
{checked_e_outcomes['satisfied']:,}/{checked_overall['E']:,} ({_fmt_pct(checked_e_outcomes['satisfied'], checked_overall['E'])}),
post unknown {checked_e_outcomes['unknown']:,}/{checked_overall['E']:,} ({_fmt_pct(checked_e_outcomes['unknown'], checked_overall['E'])}),
and post violated {checked_e_outcomes['violated']:,}/{checked_overall['E']:,} ({_fmt_pct(checked_e_outcomes['violated'], checked_overall['E'])}).
Only {checked_overall['F_fully_applied']:,}/{checked_overall['F']:,} checked-profile F rows have every historical operation locally applicable; audited v3 has
{v3_overall['F_fully_applied']:,}/{v3_overall['F']:,}.

## Subset and workload summary

{resource_table}

Factorized node/edge figures are identity-preserving structural estimates from the same pre-edit
facts and retained factor definitions; no graph tensors or training shards were materialized.

## Parameter-bearing vocabularies

{parameter_table}

Input UNK remains a trainable feature, but it is never used as semantic identity. Threshold changes
would alter feature/constant availability, not the already-computed semantic validator outcomes.

## Measured conclusion

{conclusion}

## Scope and limitations

- `checked_out_bounded` is the restored implementation comparator, not validator v3 and not a new
  correctness certification.
- `audited_v3_bounded` uses its parser, delete-then-add replay, partial/unresolved edit treatment,
  union-of-role relevance, and archived hierarchy-v2 artifact. Missing ancestry remains unknown.
- Both profiles retain their declared whole-subject/property bounded scope and assume represented
  entity fact maps are complete. No focal-only, global-uniqueness, separator-recovery, or new rule
  was introduced.
- Historical success means “verified under the named bounded profile,” not complete Wikidata or
  exact historical correctness. Copy figures are addressability upper bounds, not accuracy or
  candidate-generation recall.
- Primary counts are exact full scans. Secondary constraints were not re-evaluated because this
  audit prioritizes the requested exact primary intersections.

Machine-readable tables, the row audit, semantic chunks, diagnostic examples (seed 1729), hashes,
and execution metadata are in this directory.
"""
    report_path = args.output / "decision_report.md"
    temporary = report_path.with_suffix(".tmp.md")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    return report_path


def build_run_manifest(args: argparse.Namespace, *, started: float, commands: Sequence[str]) -> dict[str, Any]:
    artifacts = [
        "lineage.json",
        "semantic_manifest.json",
        "semantic_constraint_definitions.parquet",
        "clean_parent_train_vocab.json",
        "profile_checked_out_bounded.json",
        "profile_audited_v3_bounded.json",
        "row_level_manifest.json",
        "validation_manifest.json",
        "aggregate_manifest.json",
        "decision_report.md",
        "execution_log.md",
    ]
    usage = resource.getrusage(resource.RUSAGE_SELF)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "full_scan": True,
        "row_limit": None,
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "repository_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        ),
        "runtime_seconds_this_invocation": time.monotonic() - started,
        "max_rss_kib_this_process": int(usage.ru_maxrss),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "configuration": {
            "workers": args.workers,
            "batch_size": args.batch_size,
            "threshold": 100,
            "diagnostic_seed": 1729,
            "raw_root": str(args.raw_root.resolve()),
            "parent_root": str(args.parent_root.resolve()),
            "sample_root": str(args.sample_root.resolve()),
        },
        "inputs": {
            "registry": file_identity(args.registry),
            "encoder": file_identity(args.encoder),
            "target_vocabs": file_identity(args.target_vocabs),
            "hierarchy_v2": file_identity(args.hierarchy),
        },
        "commands": list(commands),
        "reproduction_command": " ".join([*_base_command(args), "--stage", "all", "--resume"]),
        "artifacts": {
            name: file_identity(args.output / name) for name in artifacts
        },
        "limitations": [
            "bounded-local distinct-values semantics",
            "fixed 2018-07-01 hierarchy evidence for v3",
            "no threshold-50 count because exact pre-pruning feature frequencies are unavailable",
            "no secondary full evaluation",
        ],
    }
    atomic_json(args.output / "run_manifest.json", manifest)
    return manifest


def _base_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(args.output),
        "--raw-root",
        str(args.raw_root),
        "--parent-root",
        str(args.parent_root),
        "--sample-root",
        str(args.sample_root),
        "--registry",
        str(args.registry),
        "--encoder",
        str(args.encoder),
        "--target-vocabs",
        str(args.target_vocabs),
        "--checked-worktree",
        str(args.checked_worktree),
        "--v3-worktree",
        str(args.v3_worktree),
        "--hierarchy",
        str(args.hierarchy),
        "--workers",
        str(args.workers),
        "--batch-size",
        str(args.batch_size),
    ]


def run_all(args: argparse.Namespace) -> None:
    started = time.monotonic()
    commands: list[str] = []
    stages: list[tuple[str, Path, Any]] = [
        ("lineage", args.output / "lineage.json", prepare_lineage),
        ("semantic", args.output / "semantic_manifest.json", build_semantic_sidecar),
        ("clean-vocab", args.output / "clean_parent_train_vocab.json", derive_clean_vocab),
    ]
    for name, marker, function in stages:
        if args.resume and marker.exists():
            print(f"resume: skipping completed stage {name}", file=sys.stderr, flush=True)
        else:
            stage_started = time.monotonic()
            function(args)
            print(f"stage={name} seconds={time.monotonic() - stage_started:.1f}", file=sys.stderr, flush=True)

    for profile in ("checked_out_bounded", "audited_v3_bounded"):
        marker = args.output / f"profile_{profile}.json"
        if args.resume and marker.exists():
            print(f"resume: skipping completed profile {profile}", file=sys.stderr, flush=True)
            continue
        command = [*_base_command(args), "--stage", "profile", "--profile-name", profile]
        commands.append(" ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    for name, marker, function in (
        ("finalize", args.output / "row_level_manifest.json", finalize_row_sidecar),
        ("validate", args.output / "validation_manifest.json", validate_row_sidecar),
        ("aggregate", args.output / "aggregate_manifest.json", aggregate),
    ):
        if args.resume and marker.exists():
            print(f"resume: skipping completed stage {name}", file=sys.stderr, flush=True)
        else:
            stage_started = time.monotonic()
            function(args)
            print(f"stage={name} seconds={time.monotonic() - stage_started:.1f}", file=sys.stderr, flush=True)
    generate_report(args)
    build_run_manifest(args, started=started, commands=commands)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "lineage", "semantic", "clean-vocab", "profile", "finalize", "validate", "aggregate", "report"),
        default="all",
    )
    parser.add_argument("--profile-name", choices=("checked_out_bounded", "audited_v3_bounded"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data/raw/full")
    parser.add_argument("--parent-root", type=Path, default=REPO_ROOT / "data/interim/full_minocc100")
    parser.add_argument("--sample-root", type=Path, default=REPO_ROOT / "data/interim/full_strat1m_minocc100")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "data/interim/constraint_registry_full.parquet")
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "data/interim/full_strat1m_minocc100/globalintencoder.txt")
    parser.add_argument("--target-vocabs", type=Path, default=REPO_ROOT / "data/processed/full_strat1m_minocc100/target_vocabs.json")
    parser.add_argument("--checked-worktree", type=Path, default=DEFAULT_CHECKED)
    parser.add_argument("--v3-worktree", type=Path, default=DEFAULT_V3)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.batch_size <= 0:
        parser.error("--workers and --batch-size must be positive")
    if args.stage == "profile" and not args.profile_name:
        parser.error("--profile-name is required for --stage profile")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output = args.output.resolve()
    dispatch = {
        "lineage": prepare_lineage,
        "semantic": build_semantic_sidecar,
        "clean-vocab": derive_clean_vocab,
        "profile": run_profile,
        "finalize": finalize_row_sidecar,
        "validate": validate_row_sidecar,
        "aggregate": aggregate,
        "report": generate_report,
    }
    if args.stage == "all":
        run_all(args)
    else:
        result = dispatch[args.stage](args)
        if isinstance(result, Mapping):
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(result)


if __name__ == "__main__":
    main()
