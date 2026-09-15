#!/usr/bin/env python3
"""
03_constraint_registry.py
==========================
Build a standalone constraint registry from constraints.tsv.

Usage
-----
python src/03_constraint_registry.py --dataset {sample,full}
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

from modules.constraint_checkers import CHECKERS
from modules.constraint_type_map import canonicalize_constraint_type

CONSTRAINT_TYPE_PREDICATE = "<http://www.wikidata.org/entity/P2302>"
REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "data" / "static" / "constraint_type_catalog.json"


def _load_dataframe_builder() -> Any:
    module_path = Path(__file__).with_name("02_dataframe_builder.py")
    spec = importlib.util.spec_from_file_location("dataframe_builder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_catalog_builder() -> Any:
    module_path = REPO_ROOT / "scripts" / "build_constraint_type_catalog.py"
    spec = importlib.util.spec_from_file_location("build_constraint_type_catalog", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_entity_id(value: str) -> str:
    raw = value.strip().strip("<>").strip()
    if raw.startswith("http://www.wikidata.org/entity/"):
        raw = raw.rsplit("/", 1)[-1]
    return raw


def _normalize_token(value: str) -> str:
    """Normalize tokens to match the encoder's string form."""
    raw = value.strip()
    if raw.startswith("http://www.wikidata.org/prop/direct/"):
        raw = raw.replace("http://www.wikidata.org/prop/direct/", "http://www.wikidata.org/entity/")
    return raw


def build_registry(
    constraints_def: dict[str, dict[str, list[str]]],
    constraints_by_property: dict[str, list[str]],
    constraint_catalog: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    constraint_to_property: dict[str, str] = {}
    for prop, constraint_ids in constraints_by_property.items():
        for constraint_id in constraint_ids:
            if constraint_id in constraint_to_property:
                raise ValueError(f"Constraint id {constraint_id} appears in multiple properties.")
            constraint_to_property[constraint_id] = prop

    registry: dict[str, dict[str, Any]] = {}
    for constraint_id in sorted(constraints_def):
        definition = constraints_def[constraint_id]
        predicates = definition["predicates"]
        objects = definition["objects"]
        if len(predicates) != len(objects):
            raise ValueError(f"Predicate/object mismatch for {constraint_id}.")

        type_objects = [obj for pred, obj in zip(predicates, objects) if pred == CONSTRAINT_TYPE_PREDICATE]
        if len(type_objects) != 1:
            raise ValueError(f"Constraint {constraint_id} has {len(type_objects)} type objects; expected 1.")
        constraint_type_raw = _normalize_token(type_objects[0])
        constraint_type_item = _normalize_entity_id(type_objects[0])
        if constraint_type_item is None and constraint_type_raw:
            constraint_type_item = _normalize_entity_id(constraint_type_raw)
        if constraint_type_item is None:
            constraint_type_item = ""
        catalog_entry = constraint_catalog.get(constraint_type_item or "")
        constraint_label = ""
        if catalog_entry:
            constraint_label = str(catalog_entry.get("label") or "")

        constraint_family, constraint_supported = canonicalize_constraint_type(constraint_type_item or "")
        if not constraint_family:
            constraint_family = "unsupported"
        if constraint_family in CHECKERS:
            constraint_supported = True

        constrained_property = constraint_to_property.get(constraint_id)
        if constrained_property is None:
            raise ValueError(f"Constraint {constraint_id} missing constrained property.")

        param_predicates: list[str] = []
        param_objects: list[str] = []
        for pred, obj in zip(predicates, objects):
            if pred == CONSTRAINT_TYPE_PREDICATE:
                continue
            if pred.startswith("^") and _normalize_entity_id(obj) == constrained_property:
                continue
            param_predicates.append(_normalize_token(pred))
            param_objects.append(_normalize_token(obj))

        registry[constraint_id] = {
            "constraint_type": constraint_type_raw,
            "constraint_type_item": constraint_type_item,
            "constraint_family": constraint_family,
            "constraint_label": constraint_label,
            "constraint_family_supported": constraint_supported,
            "constrained_property": constrained_property,
            "param_predicates": param_predicates,
            "param_objects": param_objects,
        }

    return registry


def assign_constraint_type_indices(registry: dict[str, dict[str, Any]]) -> dict[str, int]:
    type_items = sorted(
        {
            str(entry.get("constraint_type_item") or "").strip()
            for entry in registry.values()
            if str(entry.get("constraint_type_item") or "").strip()
        }
    )
    type_index_by_item = {type_item: idx for idx, type_item in enumerate(type_items)}
    for entry in registry.values():
        type_item = str(entry.get("constraint_type_item") or "").strip()
        entry["constraint_type_index"] = int(type_index_by_item.get(type_item, -1))
    return type_index_by_item


def validate_registry(
    registry: dict[str, dict[str, Any]],
    constraints_def: dict[str, dict[str, list[str]]],
    constraints_by_property: dict[str, list[str]],
) -> None:
    if len(registry) != len(constraints_def):
        raise ValueError("Registry entry count does not match parsed constraint instances.")

    if len(registry) != len(set(registry.keys())):
        raise ValueError("Constraint ids appear more than once in the registry.")

    constraint_to_property = {}
    for prop, constraint_ids in constraints_by_property.items():
        for constraint_id in constraint_ids:
            if constraint_id in constraint_to_property:
                raise ValueError(f"Constraint {constraint_id} has multiple constrained properties.")
            constraint_to_property[constraint_id] = prop
    if set(constraint_to_property.keys()) != set(registry.keys()):
        raise ValueError("Registry constraint ids differ from constrained properties index.")

    for constraint_id, entry in registry.items():
        if entry.get("constrained_property") != constraint_to_property.get(constraint_id):
            raise ValueError(f"Constraint {constraint_id} has inconsistent constrained property.")
        if not isinstance(entry.get("constraint_type_item"), str):
            raise TypeError(f"{constraint_id} field constraint_type_item must be a string.")
        if not isinstance(entry.get("constraint_family"), str):
            raise TypeError(f"{constraint_id} field constraint_family must be a string.")
        if not isinstance(entry.get("constraint_label"), str):
            raise TypeError(f"{constraint_id} field constraint_label must be a string.")
        if not isinstance(entry.get("constraint_family_supported"), bool):
            raise TypeError(f"{constraint_id} field constraint_family_supported must be a bool.")
        if not isinstance(entry.get("constraint_type_index"), int):
            raise TypeError(f"{constraint_id} field constraint_type_index must be an int.")
        for field in ("param_predicates", "param_objects"):
            values = entry.get(field)
            if not isinstance(values, list):
                raise TypeError(f"{constraint_id} field {field} must be a list.")
            if any(value in (None, "") for value in values):
                raise ValueError(f"{constraint_id} field {field} contains null/empty values.")
            if not all(isinstance(value, str) for value in values):
                raise TypeError(f"{constraint_id} field {field} must contain strings.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a constraint registry artifact.")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        required=True,
        help="Which dataset to read constraints.tsv from.",
    )
    args = parser.parse_args()

    raw_data_path = Path("data/raw") / args.dataset
    interim_root = Path("data/interim")
    interim_root.mkdir(parents=True, exist_ok=True)

    builder = _load_dataframe_builder()
    builder.RAW_DATA_PATH = raw_data_path
    constraints_def, constraints_by_property = builder.load_constraint_data()

    if not CATALOG_PATH.exists():
        catalog_builder = _load_catalog_builder()
        print(
            "Constraint type catalog missing at "
            f"{CATALOG_PATH}. Bootstrapping it from data/raw/{args.dataset}/constraints.tsv via Wikidata."
        )
        try:
            catalog_builder.write_catalog(dataset=args.dataset, output=CATALOG_PATH)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Constraint type catalog not found at {CATALOG_PATH}, and automatic bootstrap failed: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Constraint type catalog bootstrap failed. "
                f"Try `uv run scripts/build_constraint_type_catalog.py --dataset {args.dataset}`. "
                "This step requires outbound network access to Wikidata."
            ) from exc
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    registry = build_registry(constraints_def, constraints_by_property, catalog)
    type_index_by_item = assign_constraint_type_indices(registry)
    validate_registry(registry, constraints_def, constraints_by_property)

    registry_json = json.dumps(registry, sort_keys=True)
    output_path = interim_root / f"constraint_registry_{args.dataset}.parquet"
    pd.DataFrame({"registry_json": [registry_json]}).to_parquet(output_path)

    family_counts: dict[str, int] = {}
    supported_counts: dict[str, int] = {}
    missing_catalog: dict[str, int] = {}
    for entry in registry.values():
        type_item = entry.get("constraint_type_item", "")
        if type_item and type_item not in catalog:
            missing_catalog[type_item] = missing_catalog.get(type_item, 0) + 1
        family = entry.get("constraint_family", "")
        family_counts[family] = family_counts.get(family, 0) + 1
        if entry.get("constraint_family_supported"):
            supported_counts[family] = supported_counts.get(family, 0) + 1
    supported_total = sum(supported_counts.values())
    unsupported_total = len(registry) - supported_total
    print(f"Unique constraint_type_item values: {len(type_index_by_item)}")
    unsupported_sorted = sorted(
        ((family, count) for family, count in family_counts.items() if family not in supported_counts),
        key=lambda item: item[1],
        reverse=True,
    )
    print(f"Unique constraint_family values: {len(family_counts)}")
    print(f"Supported constraints: {supported_total}, Unsupported constraints: {unsupported_total}")
    if unsupported_sorted:
        print("Top unsupported constraint families:")
        for family, count in unsupported_sorted[:20]:
            print(f"  {family}: {count}")
    if missing_catalog:
        missing_sorted = sorted(missing_catalog.items(), key=lambda item: item[1], reverse=True)
        print(f"Constraint types missing from catalog: {len(missing_sorted)}")
        for qid, count in missing_sorted[:20]:
            print(f"  {qid}: {count}")

    print(f"Wrote constraint registry with {len(registry)} entries to {output_path}")


if __name__ == "__main__":
    main()
