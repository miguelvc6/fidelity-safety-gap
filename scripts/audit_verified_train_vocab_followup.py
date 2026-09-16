#!/usr/bin/env python3
"""Derive verified-training vocabulary and action-retention measurements.

This command consumes the completed identity-preserving decoder audit.  It
never invokes a symbolic validator, changes a production artifact, selects a
training row, or materializes a graph.  The only row-level output is a compact
ignored audit derivative containing V1 coverage diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.identity_audit import ROLE_NAMES, sha256_file  # noqa: E402
from modules.verified_vocab_followup import (  # noqa: E402
    ACTION_CATEGORIES,
    SLOT_NAMES,
    SUPPORT_BUCKETS,
    VerifiedVocabularyBuilder,
    action_features,
    assert_v1_subset_v0,
    capped_target_allocation,
    compile_vocabulary,
    edit_coverage,
    failure_tags,
    js_divergence,
    largest_remainder_allocation,
    literal_datatype,
    nearest_rank,
    permitted_role_indices,
    primary_exclusion_reason,
    slot_type,
    slots_from_row,
    support_bucket,
    term_kind,
    validation_support_bucket,
)


DEFAULT_INPUT = REPO_ROOT / "audits" / "decoder_verified_fix_2026-09-16"
DEFAULT_OUTPUT = REPO_ROOT / "audits" / "verified_train_vocab_followup_2026-09-16"
EXPECTED_BASE_COMMIT = "8582c5418965690eb8cfdeb454ca43cef00c662a"
OLD_TRAIN_BUDGET = 668_779
FAMILIES = (
    "conflictWith",
    "distinct",
    "inverse",
    "itemRequiresStatement",
    "oneOf",
    "single",
    "symmetric",
    "type",
    "valueRequiresStatement",
    "valueType",
)
SPLITS = ("train", "val", "test")
POPULATIONS = ("parent_pool", "existing_sample", "previously_discarded")
FACTOR_BINS: tuple[tuple[int, int | None, str], ...] = (
    (1, 32, "1-32"),
    (33, 64, "33-64"),
    (65, 83, "65-83"),
    (84, 107, "84-107"),
    (108, 108, "108"),
    (109, 160, "109-160"),
    (161, 267, "161-267"),
    (268, None, "268+"),
)
GRAPH_NODE_BINS: tuple[tuple[int, int | None, str], ...] = (
    (1, 32, "1-32"),
    (33, 64, "33-64"),
    (65, 128, "65-128"),
    (129, 256, "129-256"),
    (257, 512, "257-512"),
    (513, 1024, "513-1024"),
    (1025, None, "1025+"),
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


def read_csv_typed(path: Path) -> list[dict[str, Any]]:
    categorical = {
        "axis",
        "bucket",
        "category",
        "cohort",
        "detail",
        "family",
        "level",
        "metric",
        "population",
        "reason",
        "reference_kind",
        "role_name",
        "section",
        "semantic_term",
        "slot",
        "split",
        "strategy",
        "stratum",
        "term_kind",
        "term_type",
        "vocabulary",
    }

    def convert(key: str, value: str) -> Any:
        if value == "":
            return None
        if key in categorical:
            return value
        if value == "True":
            return True
        if value == "False":
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: convert(key, value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _check_file(
    records: list[dict[str, Any]],
    *,
    label: str,
    path: Path,
    expected_size: int,
    expected_sha256: str,
    required: bool,
) -> None:
    exists = path.exists()
    actual_size = path.stat().st_size if exists else None
    actual_sha = sha256_file(path) if exists and actual_size == int(expected_size) else None
    ok = bool(exists and actual_size == int(expected_size) and actual_sha == expected_sha256)
    records.append(
        {
            "label": label,
            "path": str(path.resolve()),
            "required": required,
            "exists": exists,
            "expected_size_bytes": int(expected_size),
            "actual_size_bytes": actual_size,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha,
            "ok": ok,
        }
    )
    if required and not ok:
        raise RuntimeError(f"required audit input failed fingerprint verification: {label}")


def verify_inputs(audit_root: Path) -> dict[str, Any]:
    """Verify every authoritative core input and report non-core drift."""

    records: list[dict[str, Any]] = []
    run = json.loads((audit_root / "run_manifest.json").read_text(encoding="utf-8"))
    required_compact = {
        "clean_parent_train_vocab.json",
        "lineage.json",
        "profile_audited_v3_bounded.json",
        "row_level_manifest.json",
        "semantic_manifest.json",
        "semantic_constraint_definitions.parquet",
        "validation_manifest.json",
    }
    for name, expected in run["artifacts"].items():
        _check_file(
            records,
            label=f"run_artifact:{name}",
            path=audit_root / name,
            expected_size=expected["size_bytes"],
            expected_sha256=expected["sha256"],
            required=name in required_compact,
        )

    row_manifest = json.loads((audit_root / "row_level_manifest.json").read_text(encoding="utf-8"))
    _check_file(
        records,
        label="row_level_audit",
        path=audit_root / "row_level_audit.parquet",
        expected_size=row_manifest["size_bytes"],
        expected_sha256=row_manifest["sha256"],
        required=True,
    )
    if int(row_manifest["rows"]) != 1_910_794 or not row_manifest["full_scan"]:
        raise RuntimeError("row-level audit is not the expected completed full scan")

    semantic = json.loads((audit_root / "semantic_manifest.json").read_text(encoding="utf-8"))
    if int(semantic["rows"]) != int(row_manifest["rows"]) or not semantic["full_scan"]:
        raise RuntimeError("semantic manifest does not reconcile with the row-level audit")
    for chunk in semantic["chunks"]:
        _check_file(
            records,
            label=f"semantic_chunk:{Path(chunk['path']).name}",
            path=Path(chunk["path"]),
            expected_size=chunk["size_bytes"],
            expected_sha256=chunk["sha256"],
            required=True,
        )

    profile = json.loads(
        (audit_root / "profile_audited_v3_bounded.json").read_text(encoding="utf-8")
    )
    if profile["profile"] != "audited_v3_bounded" or profile["revision"] != "018e2845de43bdb9480a1aa9a7791befb8b317ee":
        raise RuntimeError("audited-v3 profile identity changed")
    if int(profile["rows"]) != int(row_manifest["rows"]) or not profile["full_scan"]:
        raise RuntimeError("audited-v3 profile does not cover the completed audit")
    for chunk in profile["chunks"]:
        _check_file(
            records,
            label=f"audited_v3_chunk:{Path(chunk['path']).name}",
            path=Path(chunk["path"]),
            expected_size=chunk["size_bytes"],
            expected_sha256=chunk["sha256"],
            required=True,
        )

    lineage = json.loads((audit_root / "lineage.json").read_text(encoding="utf-8"))
    if int(lineage["source_rows"]) != int(row_manifest["rows"]):
        raise RuntimeError("source lineage does not reconcile with row count")
    if int(lineage["parent_rows"]) != int(row_manifest["rows"]):
        raise RuntimeError("parent lineage does not reconcile with row count")
    if sum(lineage["sample_split_counts"].values()) + sum(
        lineage["discarded_split_counts"].values()
    ) != int(lineage["parent_rows"]):
        raise RuntimeError("sample plus discarded lineage does not equal parent")
    for source in lineage["files"]:
        _check_file(
            records,
            label=f"raw_source:{Path(source['path']).name}",
            path=Path(source["path"]),
            expected_size=source["size_bytes"],
            expected_sha256=source["sha256"],
            required=True,
        )
    for name, expected in run["inputs"].items():
        _check_file(
            records,
            label=f"run_input:{name}",
            path=Path(expected["path"]),
            expected_size=expected["size_bytes"],
            expected_sha256=expected["sha256"],
            required=True,
        )

    v0 = json.loads((audit_root / "clean_parent_train_vocab.json").read_text(encoding="utf-8"))
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            {"entity": sorted(v0["entity_class_ids"]), "predicate": sorted(v0["predicate_class_ids"])},
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if expected_fingerprint != v0["fingerprint"]:
        raise RuntimeError("clean parent-train vocabulary fingerprint is internally inconsistent")
    required_ok = all(record["ok"] for record in records if record["required"])
    return {
        "required_inputs_ok": required_ok,
        "records": records,
        "non_core_mismatches": [record for record in records if not record["required"] and not record["ok"]],
        "row_count": int(row_manifest["rows"]),
        "v0_fingerprint": v0["fingerprint"],
        "profile_revision": profile["revision"],
    }


def load_encoder_metadata(path: Path) -> tuple[dict[str, int], int]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    encoding = payload[0]
    return {name: int(encoding.get(name, 0)) for name in ROLE_NAMES}, len(encoding)


def fit_verified_vocab(
    row_path: Path,
    *,
    role_encoder_ids: Mapping[str, int],
    input_encoder_size: int,
    batch_size: int,
) -> tuple[dict[str, Any], VerifiedVocabularyBuilder]:
    builder = VerifiedVocabularyBuilder(role_encoder_ids, input_encoder_size)
    columns = ["split", "v3_F", *SLOT_NAMES, "target_encoder_ids", "target_role_masks"]
    parquet = pq.ParquetFile(row_path)
    seen = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            builder.add_row(row)
            seen += 1
        if seen % 250_000 < len(batch):
            print(f"fit pass rows={seen:,}", file=sys.stderr, flush=True)
    return builder.build(), builder


def _dims(split: str, family: str) -> Iterable[tuple[str, str]]:
    yield split, family
    yield split, "overall"
    yield "overall", family
    yield "overall", "overall"


def _population_names(existing_sample: bool) -> tuple[str, str]:
    return ("parent_pool", "existing_sample" if existing_sample else "previously_discarded")


def _bin(value: int, bins: Sequence[tuple[int, int | None, str]]) -> str:
    for lower, upper, label in bins:
        if value >= lower and (upper is None or value <= upper):
            return label
    return "0_or_missing"


def _cohort_names(f_flag: bool, a0: bool, a1: bool) -> list[str]:
    result: list[str] = []
    if f_flag:
        result.append("F")
        if a0:
            result.append("F_and_A_V0")
        if a1:
            result.append("F_and_A_V1")
    return result


def aggregate_rows(
    row_path: Path,
    derived_path: Path,
    *,
    v0: Mapping[str, Any],
    v1: Mapping[str, Any],
    fit_builder: VerifiedVocabularyBuilder,
    batch_size: int,
) -> dict[str, Any]:
    columns = [
        "source_index",
        "split",
        "existing_sample",
        "family",
        "constraint_id",
        *ROLE_NAMES,
        *SLOT_NAMES,
        "target_encoder_ids",
        "target_role_masks",
        "R",
        "clean_A",
        "v3_F",
        "v3_E",
        "slot_in_factorized_input",
        "local_constraint_ids",
        "factorized_nodes",
    ]
    coverage: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    action_stats: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    strata: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    distribution: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    definition_sets: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    support_f_train: Counter[tuple[int, str, str]] = Counter()
    support_fa_train: Counter[tuple[int, str, str]] = Counter()
    eval_components: Counter[tuple[str, str, int, str, str]] = Counter()
    exclusion_rows: Counter[tuple[str, str]] = Counter()
    exclusion_slots: Counter[tuple[str, str, str, str, bool, bool]] = Counter()
    exclusion_totals: Counter[str] = Counter()
    requires: defaultdict[str, Counter[Any]] = defaultdict(Counter)
    old_sample_train: Counter[str] = Counter()
    available_train: Counter[str] = Counter()
    available_actions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    row_counts = Counter()

    temporary = derived_path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None

    def write_derived(rows: list[dict[str, Any]]) -> None:
        nonlocal writer
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", use_dictionary=True)
        writer.write_table(table)

    parquet = pq.ParquetFile(row_path)
    processed = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        derived_rows: list[dict[str, Any]] = []
        for row in batch.to_pylist():
            processed += 1
            split = str(row["split"])
            family = str(row["family"])
            r_flag = bool(row["R"])
            f_flag = bool(row["v3_F"])
            e_flag = bool(row["v3_E"])
            a0_details = edit_coverage(row, v0)
            a1_details = edit_coverage(row, v1)
            a0 = bool(a0_details["covered"])
            a1 = bool(a1_details["covered"])
            if a0 != bool(row["clean_A"]):
                raise AssertionError(f"V0 recomputation disagrees at source_index={row['source_index']}")
            if a1 and not a0:
                raise AssertionError(f"A(V1) is not a subset of A(V0) at {row['source_index']}")
            features = action_features(row) if r_flag else None
            references = a1_details["references"]

            for population in _population_names(bool(row["existing_sample"])):
                for dim_split, dim_family in _dims(split, family):
                    for vocab_name, a_flag in (("V0_clean_parent_train_minocc100", a0), ("V1_verified_train_minocc100", a1)):
                        counts = coverage[(vocab_name, population, dim_split, dim_family)]
                        counts["source_records"] += 1
                        counts["R"] += int(r_flag)
                        counts["A"] += int(a_flag)
                        counts["F"] += int(f_flag)
                        counts["F_and_A"] += int(f_flag and a_flag)
                        counts["F_not_A"] += int(f_flag and not a_flag)

            if r_flag and features is not None:
                memberships = [("parent_R", True), ("F", f_flag), ("F_and_A_V0", f_flag and a0), ("F_and_A_V1", f_flag and a1)]
                for population, include in memberships:
                    if not include:
                        continue
                    for dim_family in (family, "overall"):
                        counts = action_stats[(population, dim_family)]
                        counts["N"] += 1
                        counts[features["category"]] += 1
                        counts["paired"] += int(features["paired"])
                        counts["same_sp_replacement"] += int(features["same_subject_property_replacement"])
                        counts["base_deleted_operation"] += int(features["base_deleted_operation"])
                        counts["base_reinserted"] += int(features["base_reinserted"])
                        counts["base_deleting"] += int(features["base_deleting"])
                        counts["base_preserving"] += int(features["base_ultimately_preserved"])
                        counts["changed_subject"] += int(features["changed_subject"])
                        counts["changed_property"] += int(features["changed_property"])

            if split == "train" and bool(row["existing_sample"]):
                old_sample_train[family] += 1
            if f_flag and a1 and split == "train" and features is not None:
                available_train[family] += 1
                available_actions[family][features["category"]] += 1

            if features is not None:
                candidates: list[tuple[str, bool]] = [
                    ("core_F_and_A_V1", f_flag and a1),
                    ("verified_unexpressible_F_not_A_V1", f_flag and not a1),
                    ("all_pre_violated_E", e_flag),
                    ("verified_base_preserving", f_flag and features["base_ultimately_preserved"]),
                    (
                        "verified_no_copy_base_preserving",
                        f_flag and a1 and features["base_ultimately_preserved"],
                    ),
                ]
                for category in ACTION_CATEGORIES:
                    candidates.extend(
                        [
                            (f"verified_{category}", f_flag and features["category"] == category),
                            (
                                f"verified_no_copy_{category}",
                                f_flag and a1 and features["category"] == category,
                            ),
                        ]
                    )
                for name, include in candidates:
                    if include:
                        for dim_split, dim_family in _dims(split, family):
                            strata[(name, dim_split, dim_family)] += 1

            if features is not None:
                for cohort in _cohort_names(f_flag, a0, a1):
                    distribution[(cohort, "family")][family] += 1
                    distribution[(cohort, "edit_action")][features["category"]] += 1
                    distribution[(cohort, "base_status")][
                        "base_preserving" if features["base_ultimately_preserved"] else "base_deleting"
                    ] += 1
                    distribution[(cohort, "same_sp_status")][
                        "same_sp_replacement" if features["same_subject_property_replacement"] else "other"
                    ] += 1
                    distribution[(cohort, "factor_count_bin")][
                        _bin(len(row["local_constraint_ids"] or ()), FACTOR_BINS)
                    ] += 1
                    distribution[(cohort, "factorized_node_bin")][
                        _bin(int(row["factorized_nodes"] or 0), GRAPH_NODE_BINS)
                    ] += 1
                    definition_sets[(cohort, family)].add(str(row["constraint_id"]))
                    definition_sets[(cohort, "overall")].add(str(row["constraint_id"]))

            slots = slots_from_row(row)
            masks = [int(value) for value in row["target_role_masks"]]
            encoder_ids = [int(value) for value in row["target_encoder_ids"]]
            if f_flag and split == "train":
                for index, reference in enumerate(references):
                    kind = str(reference["kind"])
                    if kind == "unresolved":
                        key = (index, "unrepresentable", slots[index])
                    else:
                        key = (index, kind, str(reference["reference"]))
                    support_f_train[key] += 1
                    if a1:
                        support_fa_train[key] += 1

            if f_flag and split in {"val", "test"}:
                for index, reference in enumerate(references):
                    if not slots[index]:
                        continue
                    kind = str(reference["kind"])
                    if kind == "unresolved":
                        kind = "constant"
                        value = slots[index]
                    else:
                        value = str(reference["reference"])
                    eval_components[("F", split, index, kind, value)] += 1
                    if a1:
                        eval_components[("F_and_A_V1", split, index, kind, value)] += 1

            failures: list[dict[str, Any]] = []
            if f_flag and not a1:
                local_flags = [bool(value) for value in row["slot_in_factorized_input"]]
                for index, reference in enumerate(references):
                    if reference["covered"]:
                        continue
                    tags = failure_tags(
                        term=slots[index],
                        role_mask=masks[index],
                        slot_index=index,
                        vocabulary=v1,
                        encoder_id=encoder_ids[index],
                    )
                    failure = {
                        "slot_index": index,
                        "slot": SLOT_NAMES[index],
                        "term": slots[index],
                        "term_kind": term_kind(slots[index]),
                        "encoder_known": encoder_ids[index] > 0,
                        "in_local_input": local_flags[index],
                        "tags": tags,
                    }
                    failures.append(failure)
                    for tag in tags:
                        for dim_family in (family, "overall"):
                            exclusion_slots[(dim_family, SLOT_NAMES[index], tag, failure["term_kind"], failure["encoder_known"], failure["in_local_input"])] += 1
                primary = primary_exclusion_reason(failures)
                for dim_family in (family, "overall"):
                    exclusion_rows[(dim_family, primary)] += 1
                    exclusion_totals[dim_family] += 1

                if family in {"itemRequiresStatement", "valueRequiresStatement"} and features is not None:
                    detail = requires[family]
                    detail[("excluded_action", features["category"])] += 1
                    detail[("constraint", str(row["constraint_id"]))] += 1
                    for index in (1, 4):
                        if slots[index]:
                            detail[("target_predicate", slots[index])] += 1
                    for failure in failures:
                        index = int(failure["slot_index"])
                        term = str(failure["term"])
                        detail[("failing_slot", SLOT_NAMES[index])] += 1
                        detail[("term_kind", failure["term_kind"])] += 1
                        detail[("train_seen", term in fit_builder.train_seen_terms)] += 1
                        detail[("in_local_input", bool(failure["in_local_input"]))] += 1
                        if failure["term_kind"] == "literal":
                            detail[("literal_datatype", literal_datatype(term))] += 1

            derived_rows.append(
                {
                    "source_index": int(row["source_index"]),
                    "A_V0": a0,
                    "A_V1": a1,
                    "F": f_flag,
                    "E": e_flag,
                    "action_category": features["category"] if features else "ineligible",
                    "base_ultimately_preserved": features["base_ultimately_preserved"] if features else None,
                    "v1_slot_covered": [bool(reference["covered"]) for reference in references],
                    "v1_slot_reference_kind": [str(reference["kind"]) for reference in references],
                    "v1_primary_exclusion_reason": primary_exclusion_reason(failures) if failures else "",
                }
            )
            row_counts["rows"] += 1
            row_counts["R"] += int(r_flag)
            row_counts["F"] += int(f_flag)
            row_counts["A0"] += int(a0)
            row_counts["A1"] += int(a1)
            row_counts["F_A0"] += int(f_flag and a0)
            row_counts["F_A1"] += int(f_flag and a1)
        write_derived(derived_rows)
        if processed % 250_000 < len(batch):
            print(f"aggregate pass rows={processed:,}", file=sys.stderr, flush=True)
    if writer is not None:
        writer.close()
    temporary.replace(derived_path)

    if support_f_train != fit_builder.raw_support:
        delta = support_f_train - fit_builder.raw_support
        reverse = fit_builder.raw_support - support_f_train
        raise AssertionError(f"V1 support counts do not equal raw fitting counts: {delta}, {reverse}")
    if row_counts["rows"] != 1_910_794 or row_counts["R"] != 1_910_794:
        raise AssertionError("follow-up did not scan the full compatible parent population")
    if row_counts["F"] != 1_323_129 or row_counts["F_A0"] != 1_220_925:
        raise AssertionError("authoritative audited-v3/V0 headline counts changed")

    return {
        "coverage": coverage,
        "action_stats": action_stats,
        "strata": strata,
        "distribution": distribution,
        "definition_sets": definition_sets,
        "support_f_train": support_f_train,
        "support_fa_train": support_fa_train,
        "eval_components": eval_components,
        "exclusion_rows": exclusion_rows,
        "exclusion_slots": exclusion_slots,
        "exclusion_totals": exclusion_totals,
        "requires": requires,
        "old_sample_train": old_sample_train,
        "available_train": available_train,
        "available_actions": available_actions,
        "row_counts": row_counts,
    }


def coverage_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (vocab, population, split, family), counts in sorted(aggregate["coverage"].items()):
        f_count = counts["F"]
        r_count = counts["R"]
        rows.append(
            {
                "vocabulary": vocab,
                "population": population,
                "split": split,
                "family": family,
                **{key: int(counts[key]) for key in ("source_records", "R", "A", "F", "F_and_A", "F_not_A")},
                "A_over_R": counts["A"] / r_count if r_count else None,
                "F_and_A_over_F": counts["F_and_A"] / f_count if f_count else None,
            }
        )
    return rows


def vocab_comparison_rows(v0: Mapping[str, Any], v1: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("entity", "predicate"):
        v0_terms = set(v0[f"{kind}_terms"])
        v1_terms = set(v1[f"{kind}_terms"])
        v0_roles = set(v0[f"{kind}_role_indices"])
        v1_roles = set(v1[f"{kind}_role_indices"])
        rows.extend(
            [
                {
                    "term_type": kind,
                    "metric": "concrete_output_terms",
                    "V0": len(v0_terms),
                    "V1": len(v1_terms),
                    "difference_V1_minus_V0": len(v1_terms) - len(v0_terms),
                    "V1_subset_V0": v1_terms <= v0_terms,
                },
                {
                    "term_type": kind,
                    "metric": "role_tokens",
                    "V0": len(v0_roles),
                    "V1": len(v1_roles),
                    "difference_V1_minus_V0": len(v1_roles) - len(v0_roles),
                    "V1_subset_V0": v1_roles <= v0_roles,
                },
                {
                    "term_type": kind,
                    "metric": "all_output_class_ids_including_NONE_roles",
                    "V0": len(v0[f"{kind}_class_ids"]),
                    "V1": len(v1[f"{kind}_class_ids"]),
                    "difference_V1_minus_V0": len(v1[f"{kind}_class_ids"])
                    - len(v0[f"{kind}_class_ids"]),
                    "V1_subset_V0": set(v1[f"{kind}_class_ids"])
                    <= set(v0[f"{kind}_class_ids"]),
                },
            ]
        )
    return rows


def target_support_rows(
    v1: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    f_support: Counter[Any] = aggregate["support_f_train"]
    fa_support: Counter[Any] = aggregate["support_fa_train"]
    rows: list[dict[str, Any]] = []
    histogram: list[dict[str, Any]] = []
    unique_support: Counter[str] = Counter()
    unique_classes = set(v1["entity_terms"]) | set(v1["predicate_terms"])
    class_ids = {
        str(item["semantic_term"]): int(item["encoder_id"])
        for kind in ("entity", "predicate")
        for item in v1[f"{kind}_constant_classes"]
    }
    role_ids = {
        str(item["role_name"]): int(item["encoder_id"]) for item in v1["role_tokens"]
    }
    for index, slot in enumerate(SLOT_NAMES):
        kind = slot_type(index)
        constants = list(v1[f"{kind}_terms"])
        for term in constants:
            support_f = int(f_support[(index, "constant", term)])
            support_fa = int(fa_support[(index, "constant", term)])
            rows.append(
                {
                    "slot": slot,
                    "slot_index": index,
                    "reference_kind": "constant",
                    "semantic_term": term,
                    "role_name": "",
                    "encoder_class_id": class_ids[term],
                    "support_F_train": support_f,
                    "support_F_and_A_V1_train": support_fa,
                }
            )
            unique_support[term] += support_fa
        for role_index in v1[f"{kind}_role_indices"]:
            role = ROLE_NAMES[int(role_index)]
            rows.append(
                {
                    "slot": slot,
                    "slot_index": index,
                    "reference_kind": "role",
                    "semantic_term": "",
                    "role_name": role,
                    "encoder_class_id": role_ids[role],
                    "support_F_train": int(f_support[(index, "role", role)]),
                    "support_F_and_A_V1_train": int(fa_support[(index, "role", role)]),
                }
            )
        rows.append(
            {
                "slot": slot,
                "slot_index": index,
                "reference_kind": "none",
                "semantic_term": "",
                "role_name": "NONE",
                "encoder_class_id": 0,
                "support_F_train": int(f_support[(index, "none", "NONE")]),
                "support_F_and_A_V1_train": int(fa_support[(index, "none", "NONE")]),
            }
        )
        unrepresentable_f = sum(
            int(value)
            for (slot_index, reference_kind, _term), value in f_support.items()
            if slot_index == index and reference_kind == "unrepresentable"
        )
        unrepresentable_fa = sum(
            int(value)
            for (slot_index, reference_kind, _term), value in fa_support.items()
            if slot_index == index and reference_kind == "unrepresentable"
        )
        rows.append(
            {
                "slot": slot,
                "slot_index": index,
                "reference_kind": "unrepresentable_summary_not_a_V1_class",
                "semantic_term": "",
                "role_name": "",
                "encoder_class_id": "",
                "support_F_train": unrepresentable_f,
                "support_F_and_A_V1_train": unrepresentable_fa,
            }
        )
        values = [int(fa_support[(index, "constant", term)]) for term in constants]
        counts = Counter(support_bucket(value) for value in values)
        for bucket in SUPPORT_BUCKETS:
            histogram.append(
                {
                    "section": "concrete_class_support_by_slot",
                    "cohort": "F_and_A_V1_train",
                    "split": "train",
                    "slot": slot,
                    "metric": "class_count",
                    "bucket": bucket,
                    "value": int(counts[bucket]),
                    "denominator": len(values),
                    "share": counts[bucket] / len(values) if values else None,
                }
            )
        nonzero = [value for value in values if value > 0]
        for metric, quantile in (("min", 0.0), ("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("max", 1.0)):
            histogram.append(
                {
                    "section": "concrete_class_nonzero_support_statistics",
                    "cohort": "F_and_A_V1_train",
                    "split": "train",
                    "slot": slot,
                    "metric": metric,
                    "bucket": "nonzero_classes",
                    "value": nearest_rank(nonzero, quantile),
                    "denominator": len(nonzero),
                    "share": None,
                }
            )

    unique_values = [int(unique_support[term]) for term in sorted(unique_classes)]
    unique_hist = Counter(support_bucket(value) for value in unique_values)
    for bucket in SUPPORT_BUCKETS:
        histogram.append(
            {
                "section": "unique_concrete_class_support",
                "cohort": "F_and_A_V1_train",
                "split": "train",
                "slot": "all_legal_slots",
                "metric": "class_count",
                "bucket": bucket,
                "value": int(unique_hist[bucket]),
                "denominator": len(unique_values),
                "share": unique_hist[bucket] / len(unique_values) if unique_values else None,
            }
        )
    nonzero = [value for value in unique_values if value > 0]
    for metric, quantile in (("min", 0.0), ("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("max", 1.0)):
        histogram.append(
            {
                "section": "unique_concrete_class_nonzero_support_statistics",
                "cohort": "F_and_A_V1_train",
                "split": "train",
                "slot": "all_legal_slots",
                "metric": metric,
                "bucket": "nonzero_classes",
                "value": nearest_rank(nonzero, quantile),
                "denominator": len(nonzero),
                "share": None,
            }
        )

    component_counts: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
    component_denoms: Counter[tuple[str, str, str]] = Counter()
    for (cohort, split, index, kind, reference), occurrences in aggregate["eval_components"].items():
        slot = SLOT_NAMES[index]
        if kind == "constant":
            support = int(fa_support[(index, "constant", reference)])
            bucket = validation_support_bucket(support)
            component_counts[(cohort, split, slot, bucket)] += int(occurrences)
            component_denoms[(cohort, split, slot)] += int(occurrences)
            component_counts[(cohort, split, "overall", bucket)] += int(occurrences)
            component_denoms[(cohort, split, "overall")] += int(occurrences)
        elif kind == "role":
            support = int(fa_support[(index, "role", reference)])
            component_counts[(cohort, split, slot, "role_reference")] += int(occurrences)
            component_denoms[(cohort, split, slot)] += int(occurrences)
            component_counts[(cohort, split, "overall", "role_reference")] += int(occurrences)
            component_denoms[(cohort, split, "overall")] += int(occurrences)
            histogram.append(
                {
                    "section": "heldout_role_reference_support_detail",
                    "cohort": cohort,
                    "split": split,
                    "slot": slot,
                    "metric": reference,
                    "bucket": validation_support_bucket(support),
                    "value": int(occurrences),
                    "denominator": support,
                    "share": None,
                }
            )
    for (cohort, split, slot), denominator in sorted(component_denoms.items()):
        for bucket in ("0", "1-9", "10-99", ">=100", "role_reference"):
            value = component_counts[(cohort, split, slot, bucket)]
            histogram.append(
                {
                    "section": "heldout_target_component_training_support",
                    "cohort": cohort,
                    "split": split,
                    "slot": slot,
                    "metric": "component_count",
                    "bucket": bucket,
                    "value": int(value),
                    "denominator": int(denominator),
                    "share": value / denominator if denominator else None,
                }
            )
    histogram = normalize_heldout_support_denominators(histogram)
    return rows, histogram, {bucket: int(unique_hist[bucket]) for bucket in SUPPORT_BUCKETS}


def normalize_heldout_support_denominators(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep concrete-class support denominators separate from role references."""

    groups: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("section") != "heldout_target_component_training_support":
            continue
        key = (str(row["cohort"]), str(row["split"]), str(row["slot"]))
        groups[key][str(row["bucket"])] += int(row["value"])
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if row.get("section") == "heldout_target_component_training_support":
            key = (str(row["cohort"]), str(row["split"]), str(row["slot"]))
            counts = groups[key]
            concrete_total = sum(counts[bucket] for bucket in ("0", "1-9", "10-99", ">=100"))
            all_total = concrete_total + counts["role_reference"]
            if str(row["bucket"]) == "role_reference":
                denominator = all_total
                row["denominator_kind"] = "all_populated_target_components"
            else:
                denominator = concrete_total
                row["denominator_kind"] = "concrete_target_components_only"
            row["denominator"] = denominator
            row["share"] = int(row["value"]) / denominator if denominator else None
        result.append(row)
    return result


def family_action_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    baseline = {
        family: aggregate["action_stats"].get(("F", family), Counter())
        for family in (*FAMILIES, "overall")
    }
    metrics = (
        "addition_only",
        "deletion_only",
        "paired",
        "same_sp_replacement",
        "base_deleting",
        "base_preserving",
        "changed_subject",
        "changed_property",
        "base_deleted_operation",
        "base_reinserted",
    )
    for (population, family), counts in sorted(aggregate["action_stats"].items()):
        n = int(counts["N"])
        row: dict[str, Any] = {"population": population, "family": family, "N": n}
        for metric in metrics:
            value = int(counts[metric])
            row[f"{metric}_N"] = value
            row[f"{metric}_share"] = value / n if n else None
            denominator = int(baseline[family][metric])
            row[f"retention_{metric}"] = (
                value / denominator
                if population in {"F_and_A_V0", "F_and_A_V1"} and denominator
                else None
            )
        row["paired_other_N"] = int(counts["paired_other"])
        row["paired_other_share"] = counts["paired_other"] / n if n else None
        row["paired_replacement_same_subject_property_N"] = int(
            counts["paired_replacement_same_subject_property"]
        )
        row["paired_replacement_same_subject_property_share"] = (
            counts["paired_replacement_same_subject_property"] / n if n else None
        )
        result.append(row)
    f_deletions = baseline["overall"]["deletion_only"]
    v0_deletions = aggregate["action_stats"][("F_and_A_V0", "overall")]["deletion_only"]
    if int(f_deletions) != int(v0_deletions):
        raise AssertionError("prior observation failed: V0 did not retain every verified deletion-only edit")
    for family in (*FAMILIES, "overall"):
        for population in ("parent_R", "F", "F_and_A_V0", "F_and_A_V1"):
            counts = aggregate["action_stats"].get((population, family), Counter())
            if counts["N"] != sum(counts[category] for category in ACTION_CATEGORIES):
                raise AssertionError(f"action categories do not reconcile for {population}/{family}")
    return result


def exclusion_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (family, reason), count in sorted(aggregate["exclusion_rows"].items()):
        denominator = aggregate["exclusion_totals"][family]
        rows.append(
            {
                "level": "row_primary_mutually_exclusive",
                "family": family,
                "slot": "",
                "reason": reason,
                "term_kind": "",
                "encoder_known": "",
                "present_in_pre_edit_factorized_input": "",
                "count": int(count),
                "denominator": int(denominator),
                "share": count / denominator if denominator else None,
            }
        )
    for (family, slot, reason, kind, encoder_known, in_input), count in sorted(
        aggregate["exclusion_slots"].items()
    ):
        rows.append(
            {
                "level": "slot_failure_tag_overlapping_across_tags_and_slots",
                "family": family,
                "slot": slot,
                "reason": reason,
                "term_kind": kind,
                "encoder_known": encoder_known,
                "present_in_pre_edit_factorized_input": in_input,
                "count": int(count),
                "denominator": int(aggregate["exclusion_totals"][family]),
                "share": None,
            }
        )
    return rows


def requires_rows(
    aggregate: Mapping[str, Any], coverage: Sequence[Mapping[str, Any]], definitions_path: Path
) -> list[dict[str, Any]]:
    definitions = {
        str(row["constraint_id"]): row
        for row in pq.read_table(definitions_path).to_pylist()
    }
    coverage_index = {
        (row["vocabulary"], row["population"], row["split"], row["family"]): row
        for row in coverage
    }
    rows: list[dict[str, Any]] = []
    for family in ("itemRequiresStatement", "valueRequiresStatement"):
        summary = coverage_index[("V1_verified_train_minocc100", "parent_pool", "overall", family)]
        rows.extend(
            [
                {"family": family, "section": "summary", "category": "F", "rank": "", "count": summary["F"], "detail": ""},
                {"family": family, "section": "summary", "category": "F_and_A_V1", "rank": "", "count": summary["F_and_A"], "detail": ""},
                {"family": family, "section": "summary", "category": "excluded", "rank": "", "count": summary["F_not_A"], "detail": ""},
            ]
        )
        detail: Counter[Any] = aggregate["requires"][family]
        for section in ("excluded_action", "failing_slot", "term_kind", "train_seen", "in_local_input"):
            for (_prefix, category), count in sorted(
                ((key, value) for key, value in detail.items() if key[0] == section),
                key=lambda item: str(item[0][1]),
            ):
                rows.append(
                    {
                        "family": family,
                        "section": section,
                        "category": str(category),
                        "rank": "",
                        "count": int(count),
                        "detail": "",
                    }
                )
        for section, limit in (("constraint", 20), ("target_predicate", 20), ("literal_datatype", 20)):
            ranked = sorted(
                ((key[1], value) for key, value in detail.items() if key[0] == section),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
            for rank, (category, count) in enumerate(ranked, 1):
                extra = ""
                if section == "constraint":
                    definition = definitions.get(category, {})
                    extra = json.dumps(
                        {
                            "constrained_property": definition.get("constrained_property"),
                            "parameter_predicates": definition.get("parameter_predicates"),
                            "parameter_objects": definition.get("parameter_objects"),
                        },
                        sort_keys=True,
                    )
                rows.append(
                    {
                        "family": family,
                        "section": f"top_{section}",
                        "category": category,
                        "rank": rank,
                        "count": int(count),
                        "detail": extra,
                    }
                )
    return rows


def distribution_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_name = "F"
    for candidate in ("F_and_A_V0", "F_and_A_V1"):
        vocab = "V0_clean_parent_train_minocc100" if candidate.endswith("V0") else "V1_verified_train_minocc100"
        for axis in ("family", "edit_action", "base_status", "same_sp_status", "factor_count_bin", "factorized_node_bin"):
            base = aggregate["distribution"][(baseline_name, axis)]
            selected = aggregate["distribution"][(candidate, axis)]
            base_total = sum(base.values())
            selected_total = sum(selected.values())
            for category in sorted(set(base) | set(selected)):
                b = int(base[category])
                s = int(selected[category])
                bp = b / base_total if base_total else None
                sp = s / selected_total if selected_total else None
                rows.append(
                    {
                        "vocabulary": vocab,
                        "axis": axis,
                        "category": category,
                        "F_count": b,
                        "F_share": bp,
                        "F_and_A_count": s,
                        "F_and_A_share": sp,
                        "percentage_point_change": (sp - bp) * 100 if bp is not None and sp is not None else None,
                        "relative_retention": s / b if b else None,
                        "js_divergence_bits": None,
                    }
                )
            rows.append(
                {
                    "vocabulary": vocab,
                    "axis": axis,
                    "category": "__distribution_summary__",
                    "F_count": base_total,
                    "F_share": 1.0 if base_total else None,
                    "F_and_A_count": selected_total,
                    "F_and_A_share": 1.0 if selected_total else None,
                    "percentage_point_change": 0.0 if base_total and selected_total else None,
                    "relative_retention": selected_total / base_total if base_total else None,
                    "js_divergence_bits": js_divergence(base, selected),
                }
            )
        for family in (*FAMILIES, "overall"):
            rows.append(
                {
                    "vocabulary": vocab,
                    "axis": "distinct_constraint_definitions",
                    "category": family,
                    "F_count": len(aggregate["definition_sets"][("F", family)]),
                    "F_share": None,
                    "F_and_A_count": len(aggregate["definition_sets"][(candidate, family)]),
                    "F_and_A_share": None,
                    "percentage_point_change": None,
                    "relative_retention": (
                        len(aggregate["definition_sets"][(candidate, family)])
                        / len(aggregate["definition_sets"][("F", family)])
                        if aggregate["definition_sets"][("F", family)]
                        else None
                    ),
                    "js_divergence_bits": None,
                }
            )
    return rows


def evaluation_strata_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"stratum": stratum, "split": split, "family": family, "count": int(count)}
        for (stratum, split, family), count in sorted(aggregate["strata"].items())
    ]


def validate_compact_outputs(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    v0: Mapping[str, Any],
    v1: Mapping[str, Any],
    derived_path: Path,
) -> dict[str, Any]:
    """Assert the required aggregate partitions against the compact outputs."""

    checks: list[str] = []
    assert_v1_subset_v0(v1, v0)
    checks.append("V1 output references are subsets of V0")
    coverage = _coverage_lookup(tables["coverage_comparison"])
    metrics = ("source_records", "R", "A", "F", "F_and_A", "F_not_A")
    for vocab in ("V0_clean_parent_train_minocc100", "V1_verified_train_minocc100"):
        for split in (*SPLITS, "overall"):
            for family in (*FAMILIES, "overall"):
                parent = coverage[(vocab, "parent_pool", split, family)]
                sample = coverage[(vocab, "existing_sample", split, family)]
                discarded = coverage[(vocab, "previously_discarded", split, family)]
                for metric in metrics:
                    if int(parent[metric]) != int(sample[metric]) + int(discarded[metric]):
                        raise AssertionError(
                            f"parent/sample/discarded mismatch for {vocab}/{split}/{family}/{metric}"
                        )
                if int(parent["F"]) != int(parent["F_and_A"]) + int(parent["F_not_A"]):
                    raise AssertionError(f"verified coverage partition mismatch for {vocab}/{split}/{family}")
    checks.append("parent = existing sample + discarded for every coverage cell")
    checks.append("F = (F and A) + (F not A) for every coverage cell")
    for population in POPULATIONS:
        for split in (*SPLITS, "overall"):
            for family in (*FAMILIES, "overall"):
                a0 = coverage[("V0_clean_parent_train_minocc100", population, split, family)]
                a1 = coverage[("V1_verified_train_minocc100", population, split, family)]
                if int(a1["A"]) > int(a0["A"]) or int(a1["F_and_A"]) > int(a0["F_and_A"]):
                    raise AssertionError(f"A(V1) count exceeds A(V0) for {population}/{split}/{family}")
    checks.append("A(V1) count never exceeds A(V0) in any reported cell")

    action = {
        (str(row["population"]), str(row["family"])): row
        for row in tables["family_action_retention"]
    }
    for population in ("parent_R", "F", "F_and_A_V0", "F_and_A_V1"):
        overall = action[(population, "overall")]
        if int(overall["N"]) != sum(int(action[(population, family)]["N"]) for family in FAMILIES):
            raise AssertionError(f"family action totals do not reconcile for {population}")
        for family in (*FAMILIES, "overall"):
            row = action[(population, family)]
            if int(row["N"]) != sum(int(row[f"{category}_N"]) for category in ACTION_CATEGORIES):
                raise AssertionError(f"action categories do not reconcile for {population}/{family}")
    checks.append("family and mutually exclusive action counts reconcile")
    for population, vocab in (
        ("F_and_A_V0", "V0_clean_parent_train_minocc100"),
        ("F_and_A_V1", "V1_verified_train_minocc100"),
    ):
        for family in (*FAMILIES, "overall"):
            expected = coverage[(vocab, "parent_pool", "overall", family)]["F_and_A"]
            if int(action[(population, family)]["N"]) != int(expected):
                raise AssertionError(f"action/coverage mismatch for {population}/{family}")
    checks.append("family-action F-and-A counts equal coverage counts")

    exclusion_primary = Counter()
    for row in tables["exclusion_reasons"]:
        if row["level"] == "row_primary_mutually_exclusive":
            exclusion_primary[str(row["family"])] += int(row["count"])
    for family in (*FAMILIES, "overall"):
        expected = coverage[
            ("V1_verified_train_minocc100", "parent_pool", "overall", family)
        ]["F_not_A"]
        if exclusion_primary[family] != int(expected):
            raise AssertionError(f"primary exclusion reasons do not reconcile for {family}")
    checks.append("mutually exclusive primary exclusion reasons sum to F not A(V1)")

    strata = {
        (str(row["stratum"]), str(row["split"]), str(row["family"])): int(row["count"])
        for row in tables["evaluation_strata"]
    }
    for split in (*SPLITS, "overall"):
        for family in (*FAMILIES, "overall"):
            cov = coverage[
                ("V1_verified_train_minocc100", "parent_pool", split, family)
            ]
            if strata.get(("core_F_and_A_V1", split, family), 0) != int(cov["F_and_A"]):
                raise AssertionError(f"core evaluation stratum mismatch for {split}/{family}")
            if strata.get(("verified_unexpressible_F_not_A_V1", split, family), 0) != int(
                cov["F_not_A"]
            ):
                raise AssertionError(f"unexpressible evaluation stratum mismatch for {split}/{family}")
    checks.append("core and unexpressible evaluation strata equal V1 coverage partitions")

    support = tables["target_support_by_slot"]
    f_train = int(
        coverage[("V1_verified_train_minocc100", "parent_pool", "train", "overall")]["F"]
    )
    fa_train = int(
        coverage[("V1_verified_train_minocc100", "parent_pool", "train", "overall")]["F_and_A"]
    )
    for slot in SLOT_NAMES:
        slot_rows = [row for row in support if row["slot"] == slot]
        if sum(int(row["support_F_train"]) for row in slot_rows) != f_train:
            raise AssertionError(f"F training target support does not reconcile for {slot}")
        if sum(int(row["support_F_and_A_V1_train"]) for row in slot_rows) != fa_train:
            raise AssertionError(f"F-and-A training target support does not reconcile for {slot}")
    checks.append("per-slot support counts equal raw F and F-and-A training occurrences")

    for strategy in sorted({str(row["strategy"]) for row in tables["budget_feasibility"]}):
        family_rows = [
            row
            for row in tables["budget_feasibility"]
            if row["strategy"] == strategy and row["family"] != "overall"
        ]
        if sum(int(row["hypothetical_allocation"]) for row in family_rows) != OLD_TRAIN_BUDGET:
            raise AssertionError(f"budget allocation does not reconcile for {strategy}")
        if any(
            int(row["hypothetical_allocation"]) > int(row["maximum_without_oversampling"])
            for row in family_rows
        ):
            raise AssertionError(f"budget strategy oversamples a family: {strategy}")
    checks.append("all hypothetical budgets sum exactly and do not oversample")
    if pq.ParquetFile(derived_path).metadata.num_rows != 1_910_794:
        raise AssertionError("row-level follow-up derivative is incomplete")
    checks.append("row-level follow-up derivative contains all 1,910,794 rows")
    return {"passed": True, "checks": checks, "check_count": len(checks)}


def budget_rows(aggregate: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    available = {family: int(aggregate["available_train"][family]) for family in FAMILIES}
    old = {family: int(aggregate["old_sample_train"][family]) for family in FAMILIES}
    if sum(old.values()) != OLD_TRAIN_BUDGET:
        raise AssertionError("old sampled training budget does not reconcile")
    if sum(available.values()) < OLD_TRAIN_BUDGET:
        raise AssertionError("V1 eligible training pool is below the old training budget")
    proportional = largest_remainder_allocation(available, OLD_TRAIN_BUDGET)
    allocations = {
        "proportional_sampling": proportional,
        "preserve_F_and_A_V1_family_distribution": dict(proportional),
        "preserve_old_benchmark_family_proportions_where_feasible": capped_target_allocation(
            old, available, OLD_TRAIN_BUDGET
        ),
    }
    rows: list[dict[str, Any]] = []
    for strategy, allocation in allocations.items():
        expected_totals = Counter()
        for family in FAMILIES:
            actions = aggregate["available_actions"][family]
            family_total = sum(actions.values())
            record: dict[str, Any] = {
                "strategy": strategy,
                "family": family,
                "available_F_and_A_V1_train": available[family],
                "old_sample_train_count": old[family],
                "maximum_without_oversampling": available[family],
                "excess_available_minus_old_sample_train": available[family] - old[family],
                "hypothetical_allocation": allocation[family],
                "fraction_of_available_retained": allocation[family] / available[family]
                if available[family]
                else None,
                "is_projection_not_sample": True,
            }
            for action in ACTION_CATEGORIES:
                expected = allocation[family] * actions[action] / family_total if family_total else 0.0
                record[f"expected_{action}_N"] = expected
                record[f"expected_{action}_share"] = actions[action] / family_total if family_total else None
                expected_totals[action] += expected
            rows.append(record)
        overall: dict[str, Any] = {
            "strategy": strategy,
            "family": "overall",
            "available_F_and_A_V1_train": sum(available.values()),
            "old_sample_train_count": sum(old.values()),
            "maximum_without_oversampling": sum(available.values()),
            "excess_available_minus_old_sample_train": sum(available.values()) - sum(old.values()),
            "hypothetical_allocation": sum(allocation.values()),
            "fraction_of_available_retained": sum(allocation.values()) / sum(available.values()),
            "is_projection_not_sample": True,
        }
        for action in ACTION_CATEGORIES:
            overall[f"expected_{action}_N"] = float(expected_totals[action])
            overall[f"expected_{action}_share"] = expected_totals[action] / OLD_TRAIN_BUDGET
        rows.append(overall)
    return rows, allocations


def _coverage_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    return {
        (row["vocabulary"], row["population"], row["split"], row["family"]): row
        for row in rows
    }


def _fmt_pct(value: Any, digits: int = 2) -> str:
    return "undefined" if value is None else f"{float(value):.{digits}%}"


def render_report(
    *,
    coverage: Sequence[Mapping[str, Any]],
    action: Sequence[Mapping[str, Any]],
    support_histogram: Sequence[Mapping[str, Any]],
    support_headline: Mapping[str, int],
    requires: Sequence[Mapping[str, Any]],
    budget: Sequence[Mapping[str, Any]],
    v1: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    cov = _coverage_lookup(coverage)
    v0_name = "V0_clean_parent_train_minocc100"
    v1_name = "V1_verified_train_minocc100"
    v0_splits = [cov[(v0_name, "parent_pool", split, "overall")]["F_and_A"] for split in SPLITS]
    v1_splits = [cov[(v1_name, "parent_pool", split, "overall")]["F_and_A"] for split in SPLITS]
    v0_overall = cov[(v0_name, "parent_pool", "overall", "overall")]
    v1_overall = cov[(v1_name, "parent_pool", "overall", "overall")]
    family_exclusions = sorted(
        (
            (family, int(cov[(v1_name, "parent_pool", "overall", family)]["F_not_A"]))
            for family in FAMILIES
        ),
        key=lambda item: (-item[1], item[0]),
    )
    action_index = {(row["population"], row["family"]): row for row in action}
    f_action = action_index[("F", "overall")]
    v1_action = action_index[("F_and_A_V1", "overall")]
    family_lines = []
    for family in FAMILIES:
        f = action_index[("F", family)]
        selected = action_index[("F_and_A_V1", family)]
        family_lines.append(
            f"| {family} | {f['N']:,} | {selected['N']:,} | "
            f"{selected['addition_only_N']:,} ({selected['addition_only_share']:.2%}) | "
            f"{selected['deletion_only_N']:,} ({selected['deletion_only_share']:.2%}) | "
            f"{selected['paired_N']:,} ({selected['paired_share']:.2%}) | "
            f"{_fmt_pct(selected['retention_addition_only'])} | "
            f"{_fmt_pct(selected['retention_deletion_only'])} | "
            f"{_fmt_pct(selected['retention_paired'])} |"
        )
    req_summary = {
        (row["family"], row["category"]): int(row["count"])
        for row in requires
        if row["section"] == "summary"
    }
    unique_1_9 = support_headline["1"] + support_headline["2-4"] + support_headline["5-9"]
    unique_10_99 = support_headline["10-49"] + support_headline["50-99"]
    heldout: dict[tuple[str, str, str], int] = {}
    for row in support_histogram:
        if (
            row["section"] == "heldout_target_component_training_support"
            and row["slot"] == "overall"
            and row["bucket"] in {"0", "1-9"}
        ):
            heldout[(row["cohort"], row["split"], row["bucket"])] = int(row["value"])
    budget_overall = next(
        row
        for row in budget
        if row["strategy"] == "proportional_sampling" and row["family"] == "overall"
    )
    summary = {
        "V0_F_and_A_by_split": dict(zip(SPLITS, v0_splits)),
        "V1_F_and_A_by_split": dict(zip(SPLITS, v1_splits)),
        "difference_V1_minus_V0_by_split": {
            split: v1_splits[index] - v0_splits[index] for index, split in enumerate(SPLITS)
        },
        "V0_F_and_A": int(v0_overall["F_and_A"]),
        "V1_F_and_A": int(v1_overall["F_and_A"]),
        "verified_F": int(v1_overall["F"]),
        "V0_coverage_among_F": v0_overall["F_and_A_over_F"],
        "V1_coverage_among_F": v1_overall["F_and_A_over_F"],
        "coverage_loss_rows": int(v0_overall["F_and_A"] - v1_overall["F_and_A"]),
        "coverage_loss_percentage_points_on_F": (
            v0_overall["F_and_A_over_F"] - v1_overall["F_and_A_over_F"]
        )
        * 100,
        "V1_action_retention": {
            "addition_only": v1_action["retention_addition_only"],
            "deletion_only": v1_action["retention_deletion_only"],
            "paired": v1_action["retention_paired"],
            "same_sp_replacement": v1_action["retention_same_sp_replacement"],
            "base_deleting": v1_action["retention_base_deleting"],
            "base_preserving": v1_action["retention_base_preserving"],
        },
        "action_composition": {
            "F_deletion_only_share": f_action["deletion_only_share"],
            "F_and_A_V1_deletion_only_share": v1_action["deletion_only_share"],
            "deletion_only_percentage_point_change": (
                v1_action["deletion_only_share"] - f_action["deletion_only_share"]
            )
            * 100,
            "F_addition_only_share": f_action["addition_only_share"],
            "F_and_A_V1_addition_only_share": v1_action["addition_only_share"],
            "F_paired_share": f_action["paired_share"],
            "F_and_A_V1_paired_share": v1_action["paired_share"],
        },
        "largest_F_not_A_V1_families": [
            {"family": family, "count": count} for family, count in family_exclusions
        ],
        "V1_unique_concrete_class_support": dict(support_headline),
        "heldout_concrete_target_support_from_F_and_A_training": {
            cohort: {
                split: {
                    bucket: heldout.get((cohort, split, bucket), 0)
                    for bucket in ("0", "1-9")
                }
                for split in ("val", "test")
            }
            for cohort in ("F", "F_and_A_V1")
        },
        "available_V1_train": int(budget_overall["available_F_and_A_V1_train"]),
        "old_train_budget": OLD_TRAIN_BUDGET,
        "excess_over_old_train_budget": int(
            budget_overall["available_F_and_A_V1_train"] - OLD_TRAIN_BUDGET
        ),
        "v1_fingerprint": v1["fingerprint"],
    }
    report = f"""# Verified-training vocabulary and action-retention follow-up

## Decision table

| Measure | Train | Validation | Test | Overall |
|---|---:|---:|---:|---:|
| V0 F & A | {v0_splits[0]:,} | {v0_splits[1]:,} | {v0_splits[2]:,} | {v0_overall['F_and_A']:,} |
| V1 F & A | {v1_splits[0]:,} | {v1_splits[1]:,} | {v1_splits[2]:,} | {v1_overall['F_and_A']:,} |
| Difference V1 - V0 | {v1_splits[0]-v0_splits[0]:,} | {v1_splits[1]-v0_splits[1]:,} | {v1_splits[2]-v0_splits[2]:,} | {v1_overall['F_and_A']-v0_overall['F_and_A']:,} |

| Headline | Count/rate |
|---|---:|
| V0 no-copy coverage among F | {v0_overall['F_and_A']:,} / {v0_overall['F']:,} ({v0_overall['F_and_A_over_F']:.3%}) |
| V1 no-copy coverage among F | {v1_overall['F_and_A']:,} / {v1_overall['F']:,} ({v1_overall['F_and_A_over_F']:.3%}) |
| V0 A over all parent R | {v0_overall['A']:,} / {v0_overall['R']:,} ({v0_overall['A_over_R']:.3%}) |
| V1 A over all parent R | {v1_overall['A']:,} / {v1_overall['R']:,} ({v1_overall['A_over_R']:.3%}) |
| V1 verified additions retained | {v1_action['addition_only_N']:,} / {f_action['addition_only_N']:,} ({v1_action['retention_addition_only']:.3%}) |
| V1 verified deletions retained | {v1_action['deletion_only_N']:,} / {f_action['deletion_only_N']:,} ({v1_action['retention_deletion_only']:.3%}) |
| V1 verified paired edits retained | {v1_action['paired_N']:,} / {f_action['paired_N']:,} ({v1_action['retention_paired']:.3%}) |
| V1 itemRequiresStatement coverage | {req_summary[('itemRequiresStatement','F_and_A_V1')]:,} / {req_summary[('itemRequiresStatement','F')]:,} ({req_summary[('itemRequiresStatement','F_and_A_V1')]/req_summary[('itemRequiresStatement','F')]:.3%}) |
| V1 valueRequiresStatement coverage | {req_summary[('valueRequiresStatement','F_and_A_V1')]:,} / {req_summary[('valueRequiresStatement','F')]:,} ({req_summary[('valueRequiresStatement','F_and_A_V1')]/req_summary[('valueRequiresStatement','F')]:.3%}) |
| V1 concrete classes with zero F&A train support | {support_headline['0']:,} |
| V1 concrete classes with support 1-9 | {unique_1_9:,} |
| V1 concrete classes with support 10-99 | {unique_10_99:,} |
| V1 concrete classes with support >=100 | {support_headline['>=100']:,} |
| V1 entity/predicate concrete terms | {len(v1['entity_terms']):,} / {len(v1['predicate_terms']):,} |

V1 uses the unchanged threshold-100 input encoder. Its output constants and role
tokens were fitted once from `parent_pool ∩ train ∩ F`; validation and test
targets did not participate. No vocabulary refit was performed after A(V1)
filtering, and no rows were sampled or selected.
Every V1 reference set is a subset of V0; there are no independently reserved
exceptions. Consequently, every A(V1) row is also A(V0).

## Family-by-action retention under V1

| Family | F | F&A(V1) | Additions retained | Deletions retained | Paired retained | Addition retention | Deletion retention | Paired retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(family_lines)}

The complete structural table, including same-subject/property replacements,
base deletion/preservation, changed subjects/properties, and both V0 and V1,
is in `family_action_retention.csv`. Percentages above are compositions within
F&A(V1); retention columns use the corresponding F action count as denominator.

## Requires-statement exclusions

`requires_statement_exclusions.csv` records exact F, F&A(V1), and exclusion
counts for both requires-statement families, followed by excluded action types,
failing slots, term kinds, train-seen status, local-input availability, and the
top 20 definitions, predicates, and literal datatypes. `exclusion_reasons.csv`
contains a mutually exclusive deterministic row-primary reason plus overlapping
slot-level diagnostic tags.

## Distribution and fixed-budget feasibility

`distribution_shift.csv` reports percentage-point changes, relative retention,
and base-2 Jensen-Shannon divergence for family and action distributions. It
also covers base status, same-S/P replacement status, the original local-factor
bins, descriptive factorized-node bins, and distinct constraint definitions.

The V1 train pool contains {budget_overall['available_F_and_A_V1_train']:,}
eligible rows, {summary['excess_over_old_train_budget']:,} more than the old
{OLD_TRAIN_BUDGET:,}-row training budget. `budget_feasibility.csv` contains only
mathematical allocations and projected within-family action mixtures; it does
not contain or freeze row identities.

## Factual answers

1. Fitting output references on verified training rows removes {summary['coverage_loss_rows']:,} V0-covered verified fixes ({summary['coverage_loss_percentage_points_on_F']:.4f} percentage points of F).
2. V1 changes train/validation/test counts by {v1_splits[0]-v0_splits[0]:,} / {v1_splits[1]-v0_splits[1]:,} / {v1_splits[2]-v0_splits[2]:,} rows, respectively.
3. V1 retains {v1_action['retention_deletion_only']:.3%} of deletion-only fixes versus {v1_action['retention_addition_only']:.3%} of addition-only and {v1_action['retention_paired']:.3%} of paired fixes. Deletion-only rows rise from {f_action['deletion_only_share']:.3%} of F to {v1_action['deletion_only_share']:.3%} of F&A(V1), a {(v1_action['deletion_only_share']-f_action['deletion_only_share'])*100:.3f}-percentage-point compositional shift.
4. The largest family exclusions are {family_exclusions[0][0]} ({family_exclusions[0][1]:,}), {family_exclusions[1][0]} ({family_exclusions[1][1]:,}), and {family_exclusions[2][0]} ({family_exclusions[2][1]:,}). By action, exclusions are {f_action['addition_only_N']-v1_action['addition_only_N']:,} addition-only, {f_action['paired_N']-v1_action['paired_N']:,} paired, and {f_action['deletion_only_N']-v1_action['deletion_only_N']:,} deletion-only rows.
5. Across all verified held-out concrete targets, validation has {heldout.get(('F','val','0'),0):,} zero-support and {heldout.get(('F','val','1-9'),0):,} support-1-9 components; test has {heldout.get(('F','test','0'),0):,} and {heldout.get(('F','test','1-9'),0):,}. Conditional on F&A(V1), validation has {heldout.get(('F_and_A_V1','val','0'),0):,} and {heldout.get(('F_and_A_V1','val','1-9'),0):,}; test has {heldout.get(('F_and_A_V1','test','0'),0):,} and {heldout.get(('F_and_A_V1','test','1-9'),0):,}. Concrete-only denominators, shares, and slot splits are in `target_support_histogram.csv`; role references are reported separately.
6. Yes numerically: {budget_overall['available_F_and_A_V1_train']:,} eligible training rows exceed {OLD_TRAIN_BUDGET:,} by {summary['excess_over_old_train_budget']:,}, without oversampling.
7. The author still must choose the final cohort, sampling/allocation rule, evaluation population(s), treatment of unexpressible verified fixes, any minimum-support policy, and whether architecture or production preprocessing changes are warranted. This audit makes none of those choices.

## Scope and limitations

All validator outcomes are reused from `audited_v3_bounded`; no validator was
rerun. They remain conditional on its bounded-local scope, completeness policy,
fixed hierarchy, registry, and evidence. The prior run's authoritative row,
semantic, profile, lineage, input, and V0 vocabulary hashes verified. Its
non-core `aggregate_manifest.json` byte fingerprint differs from the enclosing
run manifest after later CSV reserialization; this follow-up does not use that
file and records the mismatch in `manifest.json`.
"""
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audit", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--stage", choices=("all", "report"), default="all")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain=v1")
    base_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT, head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode == 0
    if not base_is_ancestor:
        raise RuntimeError(f"follow-up must descend from {EXPECTED_BASE_COMMIT}, got {head}")

    print("verifying completed audit inputs", file=sys.stderr, flush=True)
    verification = verify_inputs(args.input_audit.resolve())
    run_manifest = json.loads((args.input_audit / "run_manifest.json").read_text(encoding="utf-8"))
    encoder_path = Path(run_manifest["inputs"]["encoder"]["path"])
    role_ids, encoder_size = load_encoder_metadata(encoder_path)
    v0 = json.loads((args.input_audit / "clean_parent_train_vocab.json").read_text(encoding="utf-8"))
    v0 = {
        **v0,
        "profile": "clean_parent_train_minocc100",
    }
    row_path = args.input_audit / "row_level_audit.parquet"

    table_names = (
        "vocab_comparison",
        "coverage_comparison",
        "target_support_by_slot",
        "target_support_histogram",
        "family_action_retention",
        "exclusion_reasons",
        "requires_statement_exclusions",
        "distribution_shift",
        "evaluation_strata",
        "budget_feasibility",
    )
    derived_path = args.output / "row_level_followup.parquet"
    if args.stage == "all":
        print("fitting verified TRAIN output vocabulary", file=sys.stderr, flush=True)
        v1, fit_builder = fit_verified_vocab(
            row_path,
            role_encoder_ids=role_ids,
            input_encoder_size=encoder_size,
            batch_size=args.batch_size,
        )
        assert_v1_subset_v0(v1, v0)
        v1["source_audit"] = {
            "row_level_sha256": verification["records"][
                next(
                    i
                    for i, record in enumerate(verification["records"])
                    if record["label"] == "row_level_audit"
                )
            ]["actual_sha256"],
            "v0_fingerprint": v0["fingerprint"],
            "audited_v3_revision": verification["profile_revision"],
        }
        atomic_json(args.output / "verified_train_vocab.json", v1)

        print("scanning row sidecar for coverage/support/action aggregates", file=sys.stderr, flush=True)
        aggregate = aggregate_rows(
            row_path,
            derived_path,
            v0=compile_vocabulary(v0),
            v1=compile_vocabulary(v1),
            fit_builder=fit_builder,
            batch_size=args.batch_size,
        )
        coverage = coverage_rows(aggregate)
        support, support_hist, support_headline = target_support_rows(v1, aggregate)
        action = family_action_rows(aggregate)
        budget, allocations = budget_rows(aggregate)
        tables = {
            "vocab_comparison": vocab_comparison_rows(v0, v1),
            "coverage_comparison": coverage,
            "target_support_by_slot": support,
            "target_support_histogram": support_hist,
            "family_action_retention": action,
            "exclusion_reasons": exclusion_rows(aggregate),
            "requires_statement_exclusions": requires_rows(
                aggregate,
                coverage,
                args.input_audit / "semantic_constraint_definitions.parquet",
            ),
            "distribution_shift": distribution_rows(aggregate),
            "evaluation_strata": evaluation_strata_rows(aggregate),
            "budget_feasibility": budget,
        }
        for name, rows in tables.items():
            atomic_csv(args.output / f"{name}.csv", rows)
        source_rows_scanned = int(aggregate["row_counts"]["rows"])
        scan_note = (
            f"scanned all {source_rows_scanned:,} completed row-sidecar records twice "
            "(vocabulary fit, then aggregate derivation)"
        )
    else:
        print("report stage: reusing completed aggregate CSV files", file=sys.stderr, flush=True)
        v1 = json.loads((args.output / "verified_train_vocab.json").read_text(encoding="utf-8"))
        assert_v1_subset_v0(v1, v0)
        tables = {name: read_csv_typed(args.output / f"{name}.csv") for name in table_names}
        tables["target_support_histogram"] = normalize_heldout_support_denominators(
            tables["target_support_histogram"]
        )
        atomic_csv(
            args.output / "target_support_histogram.csv",
            tables["target_support_histogram"],
        )
        coverage = tables["coverage_comparison"]
        support_hist = tables["target_support_histogram"]
        action = tables["family_action_retention"]
        budget = tables["budget_feasibility"]
        support_headline = {
            str(row["bucket"]): int(row["value"])
            for row in support_hist
            if row["section"] == "unique_concrete_class_support"
        }
        allocations = {
            str(strategy): {
                str(row["family"]): int(row["hypothetical_allocation"])
                for row in budget
                if row["strategy"] == strategy and row["family"] != "overall"
            }
            for strategy in sorted({row["strategy"] for row in budget})
        }
        source_rows_scanned = int(
            _coverage_lookup(coverage)[
                ("V1_verified_train_minocc100", "parent_pool", "overall", "overall")
            ]["source_records"]
        )
        scan_note = (
            f"reused compact aggregates and the row derivative from the completed "
            f"{source_rows_scanned:,}-row stage-all scan"
        )

    validation = validate_compact_outputs(
        tables,
        v0=v0,
        v1=v1,
        derived_path=derived_path,
    )

    report, summary = render_report(
        coverage=coverage,
        action=action,
        support_histogram=support_hist,
        support_headline=support_headline,
        requires=tables["requires_statement_exclusions"],
        budget=budget,
        v1=v1,
    )
    atomic_json(args.output / "summary.json", summary)
    (args.output / "decision_report.md").write_text(report, encoding="utf-8")
    command = " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    execution_log = f"""# Execution log

Executed from `{REPO_ROOT}` at {datetime.now(UTC).isoformat()}.

```bash
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_verified_train_vocab_followup.py \\
  --stage {args.stage} \\
  --input-audit audits/decoder_verified_fix_2026-09-16 \\
  --output audits/verified_train_vocab_followup_2026-09-16
```

Resolved command: `{command}`

The command verified all authoritative source/audit fingerprints, {scan_note},
and invoked no symbolic validator. It wrote no
selected-row list and did not modify data/interim, data/processed, model, or
manuscript artifacts.
"""
    (args.output / "execution_log.md").write_text(execution_log, encoding="utf-8")

    output_names = [
        "verified_train_vocab.json",
        *[f"{name}.csv" for name in tables],
        "summary.json",
        "decision_report.md",
        "execution_log.md",
        "row_level_followup.parquet",
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repository": str(REPO_ROOT),
        "repository_head": head,
        "expected_base_commit": EXPECTED_BASE_COMMIT,
        "repository_dirty_state_at_start": dirty.splitlines(),
        "input_audit": str(args.input_audit.resolve()),
        "input_verification": verification,
        "vocabulary_profiles": {
            "V0": {"name": "clean_parent_train_minocc100", "fingerprint": v0["fingerprint"]},
            "V1": {"name": "verified_train_minocc100", "fingerprint": v1["fingerprint"]},
        },
        "audited_profile": "audited_v3_bounded",
        "validator_rescan_performed": False,
        "source_rows_scanned": source_rows_scanned,
        "stage": args.stage,
        "final_dataset_selected": False,
        "sampling_performed": False,
        "output_validation": validation,
        "old_train_budget": OLD_TRAIN_BUDGET,
        "hypothetical_allocations": allocations,
        "runtime_seconds": time.monotonic() - started,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "pyarrow": pa.__version__,
        },
        "outputs": {name: file_record(args.output / name) for name in output_names},
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
