#!/usr/bin/env python3
"""
Smoke test for local-closure + factor-wiring pipeline changes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print("Command failed:", " ".join(cmd))
        if result.stdout:
            print("stdout:\n", result.stdout)
        if result.stderr:
            print("stderr:\n", result.stderr)
        raise SystemExit(1)


def _ensure_sample_data(repo_root: Path) -> Path:
    raw_sample = repo_root / "data" / "raw" / "sample"
    constraints_path = raw_sample / "constraints.tsv"
    corrections_path = raw_sample / "constraint-corrections"
    train_files = list(raw_sample.glob("constraint-corrections-*.tsv.gz.full.train.tsv.gz"))

    if constraints_path.exists() and corrections_path.exists() and train_files:
        return raw_sample

    print("Sample data missing; running 01_data_downloader.py in sample mode...")
    _run(
        [sys.executable, str(repo_root / "src" / "01_data_downloader.py"), "--dataset", "sample"],
        cwd=repo_root,
    )

    train_files = list(raw_sample.glob("constraint-corrections-*.tsv.gz.full.train.tsv.gz"))
    if not constraints_path.exists():
        print("Missing constraints.tsv after sample download.")
        raise SystemExit(1)
    if not train_files:
        print("Missing constraint correction files after sample download.")
        raise SystemExit(1)
    return raw_sample


def _prepare_workdir(repo_root: Path, workdir: Path, sample_raw: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    raw_root = workdir / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    sample_link = raw_root / "sample"
    if sample_link.exists():
        sample_link.unlink()
    try:
        os.symlink(sample_raw, sample_link, target_is_directory=True)
    except OSError as exc:
        print(f"Failed to symlink sample data into workdir: {exc}")
        raise SystemExit(1)


def _load_parquet(interim_dir: Path) -> pd.DataFrame:
    train_path = interim_dir / "df_train.parquet"
    if train_path.exists():
        return pd.read_parquet(train_path)
    candidates = list(interim_dir.glob("df_*.parquet"))
    if not candidates:
        print(f"No parquet files found in {interim_dir}")
        raise SystemExit(1)
    return pd.read_parquet(candidates[0])


def _load_registry(
    repo_root: Path,
    workdir: Path,
    dataset: str,
) -> dict:
    registry_path = workdir / "data" / "interim" / f"constraint_registry_{dataset}.parquet"
    if registry_path.exists():
        registry_df = pd.read_parquet(registry_path)
        registry_json = registry_df["registry_json"].iloc[0]
        return json.loads(registry_json) if isinstance(registry_json, str) else registry_json

    print("Constraint registry missing; building via 02a_constraint_registry.py...")
    _run(
        [sys.executable, str(repo_root / "src" / "02a_constraint_registry.py"), "--dataset", dataset],
        cwd=workdir,
    )
    if not registry_path.exists():
        print("Constraint registry still missing after rebuild.")
        raise SystemExit(1)
    registry_df = pd.read_parquet(registry_path)
    registry_json = registry_df["registry_json"].iloc[0]
    return json.loads(registry_json) if isinstance(registry_json, str) else registry_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for local closure pipeline changes.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("tmp/test_local_closure_smoke"),
        help="Temporary work directory for pipeline outputs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from modules.data_encoders import GlobalIntEncoder, dataset_variant_name

    dataset = "sample"
    min_occurrence = 100
    max_rows = 300
    max_instances = 50

    sample_raw = _ensure_sample_data(repo_root)
    workdir = args.workdir
    _prepare_workdir(repo_root, workdir, sample_raw)

    print("Running 02_dataframe_builder.py...")
    _run(
        [
            sys.executable,
            str(repo_root / "src" / "02_dataframe_builder.py"),
            "--dataset",
            dataset,
            "--min-occurrence",
            str(min_occurrence),
            "--max_rows",
            str(max_rows),
        ],
        cwd=workdir,
    )

    dataset_variant = dataset_variant_name(dataset, min_occurrence)
    interim_dir = workdir / "data" / "interim" / dataset_variant
    if not interim_dir.exists():
        print(f"Interim directory not found: {interim_dir}")
        raise SystemExit(1)

    df = _load_parquet(interim_dir)
    if "local_constraint_ids" not in df.columns:
        print("Test 1 failed: missing local_constraint_ids column.")
        raise SystemExit(1)
    if len(df) < 10:
        print("Test 1 failed: need at least 10 rows to validate local_constraint_ids.")
        raise SystemExit(1)

    checked_rows = 0
    multi_constraint_row = None
    for _, row in df.iterrows():
        local_ids = row["local_constraint_ids"]
        if local_ids is None:
            continue
        local_list = list(map(int, list(local_ids)))
        constraint_id = int(row["constraint_id"])
        if constraint_id not in local_list:
            print("Test 1 failed: constraint_id not found in local_constraint_ids.")
            raise SystemExit(1)
        if local_list != sorted(set(local_list)):
            print("Test 1 failed: local_constraint_ids not unique + sorted.")
            raise SystemExit(1)
        checked_rows += 1
        if len(local_list) > 1 and multi_constraint_row is None:
            multi_constraint_row = row
        if checked_rows >= 10:
            break

    if checked_rows < 10:
        print("Test 1 failed: insufficient rows to validate.")
        raise SystemExit(1)

    if multi_constraint_row is None:
        for _, row in df.iterrows():
            local_ids = row["local_constraint_ids"]
            if local_ids is None:
                continue
            local_list = list(map(int, list(local_ids)))
            if len(local_list) > 1:
                multi_constraint_row = row
                break

    if multi_constraint_row is None:
        print("Test 1 failed: no row found with len(local_constraint_ids) > 1.")
        print("Hint: increase --max-rows or check P_local extraction.")
        raise SystemExit(1)

    encoder = GlobalIntEncoder()
    encoder_path = interim_dir / "globalintencoder.txt"
    if not encoder_path.exists():
        print(f"Test 2 failed: encoder artifact missing at {encoder_path}")
        raise SystemExit(1)
    encoder.load(encoder_path)

    registry = _load_registry(repo_root, workdir, dataset)
    for entry in registry.values():
        family = entry.get("constraint_family")
        if not family:
            print("Test 2 failed: registry missing constraint_family.")
            raise SystemExit(1)
        if family.startswith("Q") and not family.startswith("unsupported"):
            print(f"Test 2 failed: registry has bare Q-id constraint_family: {family}")
            raise SystemExit(1)

    primary_constraint_id = int(multi_constraint_row["constraint_id"])
    local_ids = list(map(int, list(multi_constraint_row["local_constraint_ids"])))
    non_primary_ids = [cid for cid in local_ids if cid != primary_constraint_id]
    if not non_primary_ids:
        print("Test 2 failed: no non-primary constraint ids found in local_constraint_ids.")
        raise SystemExit(1)
    non_primary_id = non_primary_ids[0]
    constraint_token = encoder._decoding.get(non_primary_id)
    if constraint_token is None:
        print(f"Test 2 failed: missing token for constraint id {non_primary_id}")
        raise SystemExit(1)

    entry = registry.get(constraint_token)
    if entry is None:
        entry = registry.get(constraint_token.strip("<>"))
    if entry is None:
        print(f"Test 2 failed: missing registry entry for constraint token {constraint_token}")
        raise SystemExit(1)

    tokens_to_check = []
    constrained_property = entry.get("constrained_property")
    if constrained_property:
        tokens_to_check.append(("constrained_property", constrained_property))
    param_predicates = entry.get("param_predicates") or []
    param_objects = entry.get("param_objects") or []
    if param_predicates and param_objects:
        tokens_to_check.append(("param_predicate", param_predicates[0]))
        tokens_to_check.append(("param_object", param_objects[0]))

    if not tokens_to_check:
        print("Test 2 failed: registry entry missing constrained_property/params.")
        raise SystemExit(1)

    for label, token in tokens_to_check:
        token_id = encoder.encode(token, add_new=False)
        if token_id == 0 or token_id in encoder._filtered_ids:
            print(
                "Test 2 failed: encoder missing reserved token "
                f"{label}='{token}' for constraint id {constraint_token}"
            )
            raise SystemExit(1)

    print("Running 03_graph.py...")
    _run(
        [
            sys.executable,
            str(repo_root / "src" / "03_graph.py"),
            "--dataset",
            dataset,
            "--encoding",
            "node_id",
            "--min-occurrence",
            str(min_occurrence),
            "--max_instances",
            str(max_instances),
            "--debug_factor_wiring",
        ],
        cwd=workdir,
    )

    processed_dir = workdir / "data" / "processed" / dataset_variant
    debug_path = processed_dir / "factor_wiring_debug.json"
    if not debug_path.exists():
        print("Test 3 failed: factor_wiring_debug.json not found.")
        raise SystemExit(1)
    with debug_path.open("r", encoding="utf-8") as fh:
        debug_payload = json.load(fh)

    factors = debug_payload.get("factors") or []
    wiring_edges_total = sum(int(entry.get("wiring_edges_created", 0)) for entry in factors)
    if wiring_edges_total <= 0:
        print("Test 3 failed: no factor wiring edges created.")
        raise SystemExit(1)

    primary_constraint_id = debug_payload.get("primary_constraint_id")
    primary_entry = None
    for entry in factors:
        if entry.get("constraint_id") == primary_constraint_id:
            primary_entry = entry
            break
    if primary_entry is None:
        print("Test 3 failed: primary factor entry missing from debug payload.")
        raise SystemExit(1)
    if not primary_entry.get("matched_focus_predicate"):
        print("Test 3 failed: primary factor not wired to violating predicate.")
        raise SystemExit(1)

    if primary_entry.get("constraint_type") == "conflictWith" and debug_payload.get("other_predicate_global_id"):
        scope_counts = primary_entry.get("scope_predicate_counts") or {}
        focus_pred_gid = str(int(debug_payload.get("focus_predicate_global_id") or 0))
        other_pred_gid = str(int(debug_payload.get("other_predicate_global_id") or 0))
        if int(scope_counts.get(focus_pred_gid, 0)) <= 0:
            print("Test 3 failed: conflictWith primary factor missing focus predicate scope wiring.")
            raise SystemExit(1)
        if int(scope_counts.get(other_pred_gid, 0)) <= 0:
            print("Test 3 failed: conflictWith primary factor missing other_predicate scope wiring.")
            raise SystemExit(1)

    print("Running 04_constraint_labeler.py...")
    label_output_dir = workdir / "data" / "interim" / f"{dataset_variant}_labeled"
    _run(
        [
            sys.executable,
            str(repo_root / "src" / "04_constraint_labeler.py"),
            "--dataset",
            dataset,
            "--min-occurrence",
            str(min_occurrence),
            "--max-rows",
            str(max_rows),
        ],
        cwd=workdir,
    )

    labeled_df = _load_parquet(label_output_dir)
    required_columns = [
        "factor_checkable_pre",
        "factor_satisfied_pre",
        "factor_checkable_post_gold",
        "factor_satisfied_post_gold",
        "factor_types",
    ]
    for col in required_columns:
        if col not in labeled_df.columns:
            print(f"Test 4 failed: missing column {col} in labeled parquet.")
            raise SystemExit(1)

    if len(labeled_df) < 10:
        print("Test 4 failed: need at least 10 rows to validate label columns.")
        raise SystemExit(1)

    checked_rows = 0
    for _, row in labeled_df.head(10).iterrows():
        local_ids = row["local_constraint_ids"]
        if local_ids is None:
            continue
        local_ids = list(map(int, list(local_ids)))
        if not local_ids:
            continue
        pre_checkable = list(row["factor_checkable_pre"])
        pre_satisfied = list(row["factor_satisfied_pre"])
        post_checkable = list(row["factor_checkable_post_gold"])
        post_satisfied = list(row["factor_satisfied_post_gold"])

        if not (
            len(local_ids) == len(pre_checkable) == len(pre_satisfied) == len(post_checkable) == len(post_satisfied)
        ):
            print("Test 4 failed: label list lengths do not match local_constraint_ids.")
            raise SystemExit(1)

        checked_rows += 1

    if checked_rows < 10:
        print("Test 4 failed: insufficient rows to validate label arrays.")
        raise SystemExit(1)

    from modules.constraint_checkers import EvidenceState
    import importlib.util

    labeler_path = repo_root / "src" / "04_constraint_labeler.py"
    spec = importlib.util.spec_from_file_location("constraint_labeler", labeler_path)
    if spec is None or spec.loader is None:
        print(f"Test 4 failed: unable to load {labeler_path}")
        raise SystemExit(1)
    labeler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labeler)
    _compute_p_local = getattr(labeler, "_compute_p_local")
    _build_facts_state = getattr(labeler, "_build_facts_state")
    _build_placeholder_map = getattr(labeler, "_build_placeholder_map")
    _apply_edit = getattr(labeler, "_apply_edit")
    _resolve_registry_id = getattr(labeler, "_resolve_registry_id")

    assume_complete = True
    use_encoded_ids = pd.api.types.is_integer_dtype(labeled_df["constraint_id"])

    def _registry_entry_for_row(row: pd.Series) -> dict | None:
        cid = int(row["constraint_id"])
        token = encoder._decoding.get(cid)
        if token is None:
            return None
        entry = registry.get(token)
        if entry is None:
            entry = registry.get(token.strip("<>"))
        return entry

    def _build_states(row: pd.Series) -> tuple[EvidenceState, EvidenceState, set[int]]:
        p_local = _compute_p_local(row, cast_int=use_encoded_ids)
        facts_by_entity, predicates_present = _build_facts_state(
            row, p_local=p_local, assume_complete=assume_complete, cast_int=use_encoded_ids
        )
        subject = int(getattr(row, "subject", 0) or 0)
        predicate = int(getattr(row, "predicate", 0) or 0)
        obj = int(getattr(row, "object", 0) or 0)
        other_subject = int(getattr(row, "other_subject", 0) or 0)
        other_predicate = int(getattr(row, "other_predicate", 0) or 0)
        other_object = int(getattr(row, "other_object", 0) or 0)
        pre_state = EvidenceState(
            facts_by_entity=facts_by_entity,
            predicates_present=predicates_present,
            assume_complete=assume_complete,
            missing_edits=set(),
            focus_subject=subject,
            focus_predicate=predicate,
            focus_object=obj,
            other_subject=other_subject,
            other_predicate=other_predicate,
            other_object=other_object,
        )
        post_facts = {ent: {pred: set(values) for pred, values in facts.items()} for ent, facts in facts_by_entity.items()}
        post_predicates = {ent: set(preds) for ent, preds in predicates_present.items()}
        placeholder_map = _build_placeholder_map(encoder, row)
        missing_edits = _apply_edit(
            post_facts,
            post_predicates,
            p_local,
            row,
            placeholder_map=placeholder_map,
            assume_complete=assume_complete,
            cast_int=use_encoded_ids,
        )
        post_state = EvidenceState(
            facts_by_entity=post_facts,
            predicates_present=post_predicates,
            assume_complete=assume_complete,
            missing_edits=missing_edits,
            focus_subject=subject,
            focus_predicate=predicate,
            focus_object=obj,
            other_subject=other_subject,
            other_predicate=other_predicate,
            other_object=other_object,
        )
        return pre_state, post_state, p_local

    pre_violation_found = False
    post_fix_found = False
    post_checkable_found = False
    diagnostic_rows: list[str] = []

    for _, row in labeled_df.iterrows():
        local_ids = row["local_constraint_ids"]
        if local_ids is None:
            continue
        local_ids = list(map(int, list(local_ids)))
        if not local_ids:
            continue
        primary_id = int(row["constraint_id"])
        if primary_id not in local_ids:
            continue

        idx = local_ids.index(primary_id)
        pre_checkable = list(row["factor_checkable_pre"])
        pre_satisfied = list(row["factor_satisfied_pre"])
        post_checkable = list(row["factor_checkable_post_gold"])
        post_satisfied = list(row["factor_satisfied_post_gold"])

        if pre_checkable[idx] and pre_satisfied[idx] == 0:
            pre_violation_found = True
        if post_checkable[idx]:
            post_checkable_found = True
        if post_checkable[idx] and post_satisfied[idx] == 1:
            post_fix_found = True

        if not post_fix_found and len(diagnostic_rows) < 5 and pre_checkable[idx] and pre_satisfied[idx] == 0:
            entry = _registry_entry_for_row(row) or {}
            constrained_property_token = entry.get("constrained_property") or ""
            constrained_property_id = _resolve_registry_id(constrained_property_token, encoder) if encoder else 0
            pre_state, post_state, _ = _build_states(row)
            pre_focus_present = pre_state.focus_statement_present()
            post_focus_present = post_state.focus_statement_present()
            pre_has_prop = pre_state.has_property(pre_state.focus_subject, constrained_property_id)
            post_has_prop = post_state.has_property(post_state.focus_subject, constrained_property_id)
            constraint_type = entry.get("constraint_family") or entry.get("constraint_type_name") or "unknown"
            del_trip = (
                int(getattr(row, "del_subject", 0) or 0),
                int(getattr(row, "del_predicate", 0) or 0),
                int(getattr(row, "del_object", 0) or 0),
            )
            add_trip = (
                int(getattr(row, "add_subject", 0) or 0),
                int(getattr(row, "add_predicate", 0) or 0),
                int(getattr(row, "add_object", 0) or 0),
            )
            diagnostic_rows.append(
                "constraint_type={} del={} add={} focus_present_pre={} focus_present_post={} has_P_pre={} has_P_post={} assume_complete={}".format(
                    constraint_type,
                    del_trip,
                    add_trip,
                    pre_focus_present,
                    post_focus_present,
                    pre_has_prop,
                    post_has_prop,
                    assume_complete,
                )
            )

    if not pre_violation_found:
        print("Test 4 failed: no row found with primary constraint violated pre.")
        raise SystemExit(1)

    if not post_fix_found:
        print("Test 4 failed: no row found with primary constraint fixed post-gold.")
        coverage_pre = labeled_df["num_checkable_factors_pre"].sum()
        coverage_post = labeled_df["num_checkable_factors_post_gold"].sum()
        print(
            "Coverage summary: pre_checkable_total={}, post_checkable_total={}".format(
                int(coverage_pre), int(coverage_post)
            )
        )
        if diagnostic_rows:
            print("Diagnostics (up to 5):")
            for line in diagnostic_rows:
                print(" -", line)
        if not post_checkable_found:
            print("Test 4 failed: zero checkable post-gold constraints.")
        raise SystemExit(1)

    print(
        "PASS: checked_rows={}, multi_constraint_instance=yes, encoder_tokens_ok={}, wiring_edges_ok={}, labels_ok={}".format(
            checked_rows,
            len(tokens_to_check),
            wiring_edges_total,
            pre_violation_found,
        )
    )


if __name__ == "__main__":
    main()
