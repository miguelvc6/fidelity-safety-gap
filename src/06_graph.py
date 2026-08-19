#!/usr/bin/env python3
"""
06_graph.py
====================
Transforms the interim dataframes into a NetworkX-backed, PyTorch Geometric
`Data` object and stores the resulting list of graphs as pickle files.

Datasets
--------
* **sample** - the sample corpus shipped in the GitHub repository
  `Tpt/bass-materials` ≈ 1 .80 GB
* **full**   - the full dump hosted on Figshare
  (article ID 13338743) ≈ 10 .50 GB

Encoding Options
----------------
* **node_id**     - represent each URI by its global integer node_id
* **text_embedding** - represent each URI by a sentence-transformer text_embedding

Usage
-----
python src/06_graph.py --dataset {sample,full} --encoding {node_id,text_embedding} \
    [--min-occurrence N] [--shard-size N] [--use-torch-save] \
    [--persistence-profile {research_safe,full}] [--overwrite {atomic,unsafe,skip}]
"""

import os

# Tell HF Transformers to skip TensorFlow entirely
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import gc
import itertools
import json
import logging
import pickle
import re
import shutil
from datetime import datetime, timezone
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from modules.data_encoders import (
    ArtifactWriteResult,
    ROLE_NONE,
    ROLE_OBJECT,
    ROLE_PREDICATE,
    ROLE_SUBJECT,
    GlobalIntEncoder,
    GlobalToLocalNodeMap,
    PrecomputedWikidataCache,
    base_dataset_name,
    dataset_variant_name,
    discover_graph_artifacts,
    graph_dataset_filename,
    dump_in_shards,
    dump_stream,
    iter_stream,
)


LITERAL_ID = 0
PERSISTENCE_PROFILE_FULL = "full"
PERSISTENCE_PROFILE_RESEARCH_SAFE = "research_safe"
PERSISTENCE_PROFILE_CHOICES = (PERSISTENCE_PROFILE_FULL, PERSISTENCE_PROFILE_RESEARCH_SAFE)
OVERWRITE_MODE_ATOMIC = "atomic"
OVERWRITE_MODE_UNSAFE = "unsafe"
OVERWRITE_MODE_SKIP = "skip"
OVERWRITE_MODE_CHOICES = (OVERWRITE_MODE_ATOMIC, OVERWRITE_MODE_UNSAFE, OVERWRITE_MODE_SKIP)


def _torch_load_trusted(path: Path) -> Any:
    """
    Load a trusted local torch artifact with backwards-compatible semantics.

    PyTorch 2.6 changed torch.load default `weights_only` from False to True,
    which breaks loading generic pickled objects like PyG Data shards.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def _normalize_target_id(value: Any, encoder: GlobalIntEncoder, unknown_id: int) -> int:
    """Return a sanitized class id compatible with the filtered encoder."""
    if value is None:
        return 0
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return 0
    if idx <= 0:
        return 0
    if idx in getattr(encoder, "_filtered_ids", set()):
        return unknown_id
    return idx


def _is_id_iterable(value: Any) -> bool:
    """True when value should be treated as a list of ids."""
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _normalize_id_sequence(value: Any) -> list[Any]:
    """Normalize singletons and iterables into a concrete list."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _is_id_iterable(value):
        return list(value)
    return [value]


def _factor_ids_for_graph(graph: dict[str, Any], constraint_scope: str) -> list[int]:
    """Return the factor IDs to instantiate, preferring labeler-filtered IDs."""
    constraint_ids_raw = graph.get("factor_constraint_ids")
    if constraint_ids_raw is None:
        if constraint_scope == "focus":
            constraint_ids_raw = graph.get("local_constraint_ids_focus")
            if constraint_ids_raw is None:
                constraint_ids_raw = graph.get("local_constraint_ids")
        else:
            constraint_ids_raw = graph.get("local_constraint_ids")

    if isinstance(constraint_ids_raw, Iterable) and not isinstance(constraint_ids_raw, (str, bytes)):
        factor_ids = [int(cid) for cid in constraint_ids_raw if cid is not None]
    else:
        factor_ids = []
    if not factor_ids:
        factor_ids = [int(graph["constraint_id"])]
    return factor_ids


def _constraint_family_from_registry_entry(registry_entry: dict[str, Any]) -> str:
    """Return canonical factor family used for graph wiring and debug metadata."""
    return str(
        registry_entry.get("constraint_family")
        or registry_entry.get("constraint_type_name")
        or registry_entry.get("constraint_type", "")
    )


def _is_literal_node(graph: dict[str, Any], key: str) -> bool:
    text_value = graph.get(f"{key}_text")
    if isinstance(text_value, str) and text_value != "":
        return True
    if LITERAL_ID and graph.get(key) == LITERAL_ID:
        return True
    return False


def create_graph(
    graph: dict[str, Any],
    wikidata_cache: PrecomputedWikidataCache | None,
    global_int_encoder: GlobalIntEncoder,
    constraint_registry: Dict[str, Dict[str, Any]],
    encoding: Literal["node_id", "text_embedding"] = "text_embedding",
    constraint_scope: Literal["local", "focus"] = "local",
    constraint_representation: Literal["factorized", "eswc_passive"] = "factorized",
    store_node_names: bool = False,
    include_debug_fields: bool = True,
    embedding_dtype: np.dtype | None = None,
    debug_factor_wiring: bool = False,
) -> Data:
    """
    Convert a single Wikidata conflict record into a homogeneous multigraph
    represented as a `torch_geometric.data.Data` object.

    Each triple in the record is rewritten as a length-2 path
    *(subject → predicate → object)*.  All triples that appear in the local
    neighbourhood of the conflict (main statement, neighbour statements,
    constraint statements, …) are merged into one multirelational graph.
    Optionally, the graph can be visualised with NetworkX/Matplotlib.

    Parameters
    ----------
    graph:
        A dictionary obtained from a row in the intermediate parquet file
        whose keys correspond to the columns described in the BASS paper
        (e.g. "subject", "predicate", "object", …).
    wikidata_cache:
        Cache that provides pre-computed labels and embeddings for all
        identifiers encountered in the dataset.
    global_int_encoder:
        An instance of `GlobalIntEncoder` used to encode URIs to global integer IDs.
    encode_predicates_as_nodes:
        Must be *True*: the predicate URI is represented as a proper node,
        leading to a uniform first-order graph. Other encodings are unsupported.
    used_attribute:
        Controls the node feature:
        - "node_id"    → one-hot integer label
        - "text_embedding" → 768-d sentence-transformer embedding
    embedding_dtype:
        Optional numpy dtype used when `encoding` is ``"text_embedding"`` to
        control the precision (e.g., ``np.float16`` or ``np.float32``).

    Returns
    -------
    torch_geometric.data.Data
        A directed graph with
        1. x -- holding node features,
        2. edge_index -- holding source/target indices,
        3. y -- holding the 6-way multi-label classification target.
    """

    global_to_local_id_encoder = GlobalToLocalNodeMap()
    unknown_global_id = global_int_encoder.encode("unknown", add_new=False)
    if unknown_global_id == 0:
        unknown_global_id = global_int_encoder.encode("unknown", add_new=True)
    EDGE_SUBJECT_TO_PREDICATE = 0
    EDGE_PREDICATE_TO_OBJECT = 1
    EDGE_FACTOR_TO_PARAM_PREDICATE = 2
    EDGE_PARAM_PREDICATE_TO_OBJECT = 3
    EDGE_FACTOR_TO_LOCAL_PREDICATE = 4
    EDGE_FACTOR_TO_LOCAL_SUBJECT = 5
    EDGE_FACTOR_TO_LOCAL_OBJECT = 6
    EDGE_LOCAL_PREDICATE_TO_FACTOR = 7
    EDGE_LOCAL_SUBJECT_TO_FACTOR = 8
    EDGE_LOCAL_OBJECT_TO_FACTOR = 9

    edges: List[tuple[int, int]] = []
    edge_types: List[int] = []
    edges_non_flattened: List[tuple[int, int]] = []
    non_flattened_edge_attributes: List[Any] = []
    resolved_embedding_dtype: np.dtype | None = None
    if encoding == "text_embedding":
        resolved_embedding_dtype = np.dtype(embedding_dtype or np.float16)
    focus_local_nodes: dict[str, int] = {}
    factor_constraint_ids: list[int] = []
    factor_constraint_types: list[str] = []
    factor_local_ids: list[int] = []
    primary_factor_index: int = -1
    pred_global_to_local_triples: dict[int, list[tuple[int, int, int]]] = {}
    pred_global_to_pred_local_ids: dict[int, list[int]] = {}
    subj_local_to_pred_global_to_pred_local_ids: dict[int, dict[int, list[int]]] = {}
    debug_entries: list[dict[str, Any]] = []
    primary_factor_focus_scope_ok: bool | None = None
    factor_checkable_pre_tensor: torch.Tensor | None = None
    factor_satisfied_pre_tensor: torch.Tensor | None = None
    factor_checkable_post_gold_tensor: torch.Tensor | None = None
    factor_satisfied_post_gold_tensor: torch.Tensor | None = None
    factor_types_tensor: torch.Tensor | None = None
    factorized_representation = constraint_representation == "factorized"

    def _resolve_registry_id(raw_id: str) -> int | None:
        raw = raw_id.strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1].strip()
        if raw.startswith("http://www.wikidata.org/prop/direct/"):
            raw = raw.replace(
                "http://www.wikidata.org/prop/direct/",
                "http://www.wikidata.org/entity/",
            )
        candidates: list[str] = []
        seen: set[str] = set()
        if raw.startswith("http://") or raw.startswith("https://"):
            candidates.extend([raw, f"<{raw}>"])
            last = raw.rsplit("/", 1)[-1]
            if last and last[0] in ("P", "Q") and last[1:].isdigit():
                candidates.append(last)
        else:
            if raw and raw[0] in ("P", "Q") and raw[1:].isdigit():
                entity_uri = f"http://www.wikidata.org/entity/{raw}"
                candidates.extend([entity_uri, f"<{entity_uri}>"])
            candidates.append(raw)
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            gid = global_int_encoder.encode(candidate, add_new=False)
            if gid:
                return gid
        return None

    def _maybe_encode_registry_token(raw_token: str | None) -> int:
        if not raw_token:
            return 0
        try:
            return global_int_encoder.encode(raw_token, add_new=False)
        except Exception:
            return 0

    def _is_property_token(raw_value: str | None) -> bool:
        if not raw_value:
            return False
        raw = raw_value.strip().strip("<>").strip()
        if raw.startswith("http://www.wikidata.org/entity/"):
            raw = raw.rsplit("/", 1)[-1]
        if raw.startswith("http://www.wikidata.org/prop/direct/"):
            raw = raw.rsplit("/", 1)[-1]
        return raw.startswith("P") and raw[1:].isdigit()

    if encoding == "text_embedding" and wikidata_cache is None:
        raise ValueError("wikidata_cache is required when encoding='text_embedding'")
    if store_node_names and wikidata_cache is None:
        raise ValueError("wikidata_cache is required to store node names")

    def get_node_attribute(global_node_id: int, node_text: str | None) -> Any:
        """Get the node attribute for a global node ID."""
        if encoding == "text_embedding":
            assert wikidata_cache is not None
            if node_text is not None and node_text != "":
                return wikidata_cache.get_embedding_for_literal(
                    node_text,
                    dtype=resolved_embedding_dtype,
                    fallback_id=unknown_global_id,
                )
            return wikidata_cache.get_embedding_for_id(
                global_node_id,
                dtype=resolved_embedding_dtype,
                fallback_id=unknown_global_id,
            )
        else:
            if global_node_id in global_int_encoder._filtered_ids:
                global_node_id = unknown_global_id
            return global_int_encoder.get_unfiltered_global_id(global_node_id)

    def add_edge_from_ids(
        subject_global_id: int,
        predicate_global_id: int,
        object_global_id: int,
        subject_text: str | None = None,
        predicate_text: str | None = None,
        object_text: str | None = None,
        assign_focus: bool = False,
    ):
        if subject_global_id == 0:
            if not subject_text:
                subject_global_id = unknown_global_id
            else:
                # Fall back to the literal placeholder when a text literal lacks a global id.
                subject_global_id = LITERAL_ID
        if predicate_global_id == 0:
            if not predicate_text:
                predicate_global_id = unknown_global_id
            else:
                predicate_global_id = LITERAL_ID
        if object_global_id == 0:
            if not object_text:
                object_global_id = unknown_global_id
            else:
                object_global_id = LITERAL_ID

        subject_name = None
        predicate_name = None
        object_name = None
        if store_node_names:
            assert wikidata_cache is not None
            subject_name = subject_text or wikidata_cache.get_text_for_id(
                subject_global_id,
                fallback_id=unknown_global_id,
            )
            predicate_name = predicate_text or wikidata_cache.get_text_for_id(
                predicate_global_id,
                fallback_id=unknown_global_id,
            )
            if object_text:
                object_name = object_text
            else:
                object_name = wikidata_cache.get_text_for_id(
                    object_global_id,
                    fallback_id=unknown_global_id,
                )

        subject_id = global_to_local_id_encoder.store(
            subject_global_id,
            get_node_attribute(subject_global_id, subject_text),
            name=subject_name,
        )
        predicate_id = global_to_local_id_encoder.store(
            predicate_global_id,
            get_node_attribute(predicate_global_id, predicate_text),
            # because we encode predicates as nodes, we need to duplicate them to not change the graph structure
            force_create=True,
            name=predicate_name,
        )

        object_id = global_to_local_id_encoder.store(
            object_global_id,
            get_node_attribute(object_global_id, object_text),
            name=object_name,
        )

        if assign_focus:
            focus_local_nodes["subject"] = subject_id
            focus_local_nodes["predicate"] = predicate_id
            focus_local_nodes["object"] = object_id

        edges.append((subject_id, predicate_id))
        edge_types.append(EDGE_SUBJECT_TO_PREDICATE)
        edges.append((predicate_id, object_id))
        edge_types.append(EDGE_PREDICATE_TO_OBJECT)

        pred_global_to_pred_local_ids.setdefault(predicate_global_id, []).append(predicate_id)
        pred_global_to_local_triples.setdefault(predicate_global_id, []).append((subject_id, predicate_id, object_id))
        subj_local_to_pred_global_to_pred_local_ids.setdefault(subject_id, {}).setdefault(
            predicate_global_id, []
        ).append(predicate_id)

        edges_non_flattened.append((subject_id, object_id))
        non_flattened_edge_attributes.append(predicate_id)

    def add_edge(
        graph: dict[str, Any],
        subject_key: str,
        predicate_key: str,
        object_key: str,
        assign_focus: bool = False,
    ) -> None:
        subject_global_id = graph[subject_key]
        predicate_global_id = graph[predicate_key]
        object_global_id = graph[object_key]

        # handle iterable ids
        if (
            _is_id_iterable(subject_global_id)
            or _is_id_iterable(object_global_id)
            or _is_id_iterable(predicate_global_id)
        ):
            subject_ids = _normalize_id_sequence(subject_global_id)
            object_ids = _normalize_id_sequence(object_global_id)
            predicate_ids = _normalize_id_sequence(predicate_global_id)
            if len(subject_ids) == 0 or len(object_ids) == 0 or len(predicate_ids) == 0:
                # skip creation, triple doesn't exist
                return
            length = max(len(subject_ids), len(object_ids), len(predicate_ids))
            if length > 1:
                # broadcast singletons
                if len(subject_ids) == 1:
                    subject_ids = subject_ids * length
                if len(object_ids) == 1:
                    object_ids = object_ids * length
                if len(predicate_ids) == 1:
                    predicate_ids = predicate_ids * length
            assert len(subject_ids) == len(object_ids) == len(predicate_ids), (
                f"Length mismatch when adding edge: {len(subject_ids)} subjects, {len(predicate_ids)} predicates, {len(object_ids)} objects."
            )

            for subject_global, predicate_global, object_global in zip(subject_ids, predicate_ids, object_ids):
                add_edge_from_ids(
                    subject_global,
                    predicate_global,
                    object_global,
                    subject_text=graph.get(f"{subject_key}_text"),
                    predicate_text=graph.get(f"{predicate_key}_text"),
                    object_text=graph.get(f"{object_key}_text"),
                    assign_focus=assign_focus,
                )
        else:
            if object_global_id == 0 and graph.get(f"{object_key}_text") in ["", None]:
                assert predicate_global_id == 0, f"Predicate {predicate_global_id} without object {object_global_id}"
                # skip creation, triple doesn't exist
                return
            add_edge_from_ids(
                subject_global_id,
                predicate_global_id,
                object_global_id,
                subject_text=graph.get(f"{subject_key}_text"),
                predicate_text=graph.get(f"{predicate_key}_text"),
                object_text=graph.get(f"{object_key}_text"),
                assign_focus=assign_focus,
            )

    def add_factor_definition_edge(
        factor_local_id: int,
        predicate_global_id: int,
        object_global_id: int,
    ) -> None:
        if predicate_global_id == 0:
            predicate_global_id = unknown_global_id
        if object_global_id == 0:
            object_global_id = unknown_global_id

        predicate_name = None
        object_name = None
        if store_node_names:
            assert wikidata_cache is not None
            cache = wikidata_cache
            predicate_name = cache.get_text_for_id(
                predicate_global_id,
                fallback_id=unknown_global_id,
            )
            object_name = cache.get_text_for_id(
                object_global_id,
                fallback_id=unknown_global_id,
            )

        predicate_id = global_to_local_id_encoder.store(
            predicate_global_id,
            get_node_attribute(predicate_global_id, None),
            force_create=True,
            name=predicate_name,
        )
        object_id = global_to_local_id_encoder.store(
            object_global_id,
            get_node_attribute(object_global_id, None),
            name=object_name,
        )

        edges.append((factor_local_id, predicate_id))
        edge_types.append(EDGE_FACTOR_TO_PARAM_PREDICATE)
        edges.append((predicate_id, object_id))
        edge_types.append(EDGE_PARAM_PREDICATE_TO_OBJECT)

        edges_non_flattened.append((factor_local_id, object_id))
        non_flattened_edge_attributes.append(predicate_id)

    # --- Node construction -------------------------------------------------
    add_edge(graph, "subject", "predicate", "object", assign_focus=True)
    add_edge(graph, "other_subject", "other_predicate", "other_object")

    # add subject neighbours
    add_edge(graph, "subject", "subject_predicates", "subject_objects")
    add_edge(graph, "object", "object_predicates", "object_objects")
    # Neighbour statements use the same observed
    # subject -> predicate occurrence -> object direction as the focus statement.

    # pick the "other_entity"
    if graph["other_subject"] == graph["subject"]:
        other_entity_name = "other_object"
    elif graph["other_object"] == graph["object"]:
        other_entity_name = "other_subject"
    else:
        assert graph["other_subject"] == graph["other_predicate"] == graph["other_object"] == 0
        other_entity_name = None

    if other_entity_name is not None:
        add_edge(graph, other_entity_name, "other_entity_predicates", "other_entity_objects")

    param_property_gid = _maybe_encode_registry_token("<http://www.wikidata.org/entity/P2306>")
    param_relation_gid = _maybe_encode_registry_token("<http://www.wikidata.org/entity/P2309>")
    param_inverse_gid = _maybe_encode_registry_token("<http://www.wikidata.org/entity/P1696>")
    default_relation_gids = [
        gid
        for gid in (
            _resolve_registry_id("P31"),
            _resolve_registry_id("P279"),
        )
        if gid is not None
    ]

    # add constraint factor branches
    if factorized_representation:
        factor_ids = _factor_ids_for_graph(graph, constraint_scope)
    else:
        factor_ids = [int(graph["constraint_id"])]

    required_factor_fields = (
        "factor_checkable_pre",
        "factor_satisfied_pre",
        "factor_checkable_post_gold",
        "factor_satisfied_post_gold",
        "factor_types",
        "factor_constraint_ids",
    )
    if factorized_representation and all(graph.get(field) is not None for field in required_factor_fields):
        expected_len = len(factor_ids)

        def _normalize_factor_list(value: Any, name: str) -> list[Any]:
            if isinstance(value, list):
                items = value
            elif isinstance(value, tuple):
                items = list(value)
            elif _is_id_iterable(value):
                items = list(value)
            else:
                items = [value]
            if len(items) != expected_len:
                raise AssertionError(
                    f"Factor label length mismatch for {name}: expected {expected_len}, got {len(items)}."
                )
            return items

        factor_ids_from_row = _normalize_factor_list(graph.get("factor_constraint_ids"), "factor_constraint_ids")
        if factor_ids_from_row and [int(cid) for cid in factor_ids_from_row] != factor_ids:
            raise AssertionError("Factor constraint id order mismatch between labeled data and graph builder.")

        factor_checkable_pre_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_checkable_pre"), "factor_checkable_pre"),
            dtype=torch.bool,
        )
        factor_satisfied_pre_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_satisfied_pre"), "factor_satisfied_pre"),
            dtype=torch.long,
        )
        factor_checkable_post_gold_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_checkable_post_gold"), "factor_checkable_post_gold"),
            dtype=torch.bool,
        )
        factor_satisfied_post_gold_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_satisfied_post_gold"), "factor_satisfied_post_gold"),
            dtype=torch.long,
        )
        factor_types_tensor = torch.tensor(
            _normalize_factor_list(graph.get("factor_types"), "factor_types"),
            dtype=torch.long,
        )

    for idx, constraint_id in enumerate(factor_ids):
        constraint_token = global_int_encoder._decoding.get(constraint_id)
        if constraint_token in (None, "", "unknown") or constraint_id == unknown_global_id:
            raise AssertionError(
                "Constraint id token is unknown; rebuild interim data with constraint ids preserved."
            )
        registry_entry = constraint_registry.get(constraint_token)
        if registry_entry is None:
            registry_entry = constraint_registry.get(constraint_token.strip("<>"))
        assert registry_entry is not None, f"Missing constraint_id={constraint_token} in constraint registry."

        try:
            factor_gid = global_int_encoder.encode(f"constraint_factor::{constraint_token}", add_new=False)
        except Exception as exc:
            raise AssertionError(f"Missing factor token for constraint_id={constraint_token} in encoder.") from exc
        assert factor_gid != 0, f"Invalid factor global id for constraint_id={constraint_id}."

        factor_local_id = global_to_local_id_encoder.store(
            factor_gid,
            get_node_attribute(factor_gid, None),
            name=f"constraint_factor::{constraint_token}" if store_node_names else None,
            force_create=True,
        )

        factor_constraint_ids.append(constraint_id)
        factor_local_ids.append(factor_local_id)
        constraint_type = _constraint_family_from_registry_entry(registry_entry)
        factor_constraint_types.append(constraint_type)
        if constraint_id == int(graph["constraint_id"]):
            primary_factor_index = idx

        constrained_property = registry_entry.get("constrained_property")

        param_predicates = registry_entry.get("param_predicates") or []
        param_objects = registry_entry.get("param_objects") or []
        assert len(param_predicates) == len(param_objects), (
            f"Constraint registry param list length mismatch for constraint_id={constraint_id}."
        )

        observed_predicates: list[int] = []
        observed_predicates_set: set[int] = set()

        def _add_observed(predicate_gid: int | None) -> None:
            if predicate_gid is None:
                return
            if predicate_gid in observed_predicates_set:
                return
            observed_predicates_set.add(predicate_gid)
            observed_predicates.append(predicate_gid)

        def _collect_param_object_gids(param_predicate_gid: int) -> list[int]:
            if not param_predicate_gid:
                return []
            matches: list[int] = []
            for pred_raw, obj_raw in zip(param_predicates, param_objects):
                pred_gid = _maybe_encode_registry_token(pred_raw)
                if pred_gid != param_predicate_gid:
                    continue
                obj_gid = _resolve_registry_id(obj_raw)
                if obj_gid is not None:
                    matches.append(obj_gid)
            return matches

        matched_predicate_local_ids = 0
        wiring_edges_created = 0
        matched_focus_predicate = False
        scope_predicate_counts: dict[int, int] = {}
        constrained_gid = None
        if factorized_representation:
            if constrained_property:
                constrained_gid = _resolve_registry_id(constrained_property)
                if constrained_gid is not None:
                    _add_observed(constrained_gid)

            if constraint_type == "conflictWith":
                for obj_raw in param_objects:
                    if _is_property_token(obj_raw):
                        obj_gid = _resolve_registry_id(obj_raw)
                        if obj_gid is not None:
                            _add_observed(obj_gid)
                other_predicate_gid = int(graph.get("other_predicate") or 0)
                if other_predicate_gid:
                    _add_observed(other_predicate_gid)
            elif constraint_type in {"inverse", "symmetric"}:
                inverse_gids = _collect_param_object_gids(param_inverse_gid)
                if not inverse_gids and constrained_gid is not None:
                    inverse_gids = [constrained_gid]
                for obj_gid in inverse_gids:
                    _add_observed(obj_gid)
            elif constraint_type == "itemRequiresStatement":
                for obj_gid in _collect_param_object_gids(param_property_gid):
                    _add_observed(obj_gid)
            elif constraint_type == "valueRequiresStatement":
                for obj_gid in _collect_param_object_gids(param_property_gid):
                    _add_observed(obj_gid)
            elif constraint_type == "type":
                relation_gids = _collect_param_object_gids(param_relation_gid)
                if not relation_gids:
                    relation_gids = list(default_relation_gids)
                for obj_gid in relation_gids:
                    _add_observed(obj_gid)
            elif constraint_type == "valueType":
                relation_gids = _collect_param_object_gids(param_relation_gid)
                if not relation_gids:
                    relation_gids = list(default_relation_gids)
                for obj_gid in relation_gids:
                    _add_observed(obj_gid)

            subject_scope_ids: set[int] | None
            if constraint_type in {"single", "conflictWith", "itemRequiresStatement"}:
                subject_id = focus_local_nodes.get("subject")
                subject_scope_ids = {subject_id} if subject_id is not None else set()
            elif constraint_type in {"valueRequiresStatement", "valueType"}:
                object_id = focus_local_nodes.get("object")
                if object_id is not None and not _is_literal_node(graph, "object"):
                    subject_scope_ids = {object_id}
                else:
                    subject_scope_ids = set()
            elif constraint_type == "distinct":
                subject_scope_ids = set()
                focus_subject_id = focus_local_nodes.get("subject")
                if focus_subject_id is not None:
                    subject_scope_ids.add(focus_subject_id)
                other_subject_gid = int(graph.get("other_subject") or 0)
                if other_subject_gid:
                    other_subject_id = global_to_local_id_encoder.global_to_local.get(other_subject_gid)
                    if other_subject_id is not None:
                        subject_scope_ids.add(other_subject_id)
            else:
                subject_scope_ids = None

            def _collect_pred_local_ids(predicate_gid: int, subject_ids: set[int] | None) -> list[int]:
                if subject_ids is None:
                    return list(pred_global_to_pred_local_ids.get(predicate_gid, []))
                pred_ids: list[int] = []
                for subj_id in sorted(subject_ids):
                    pred_ids.extend(
                        subj_local_to_pred_global_to_pred_local_ids.get(subj_id, {}).get(predicate_gid, [])
                    )
                return pred_ids

            scope_pred_local_ids: list[int] = []
            scope_pred_local_ids_seen: set[int] = set()
            for predicate_gid in observed_predicates:
                local_pred_ids = _collect_pred_local_ids(predicate_gid, subject_scope_ids)
                if local_pred_ids:
                    scope_predicate_counts[predicate_gid] = len(local_pred_ids)
                for pred_local_id in local_pred_ids:
                    if pred_local_id in scope_pred_local_ids_seen:
                        continue
                    scope_pred_local_ids_seen.add(pred_local_id)
                    scope_pred_local_ids.append(pred_local_id)

            matched_predicate_local_ids = len(scope_pred_local_ids)
            for pred_local_id in scope_pred_local_ids:
                edges.append((factor_local_id, pred_local_id))
                edge_types.append(EDGE_FACTOR_TO_LOCAL_PREDICATE)
                edges.append((pred_local_id, factor_local_id))
                edge_types.append(EDGE_LOCAL_PREDICATE_TO_FACTOR)
                wiring_edges_created += 1
                if focus_local_nodes.get("predicate") == pred_local_id:
                    matched_focus_predicate = True

            if constrained_gid is not None:
                local_triples = pred_global_to_local_triples.get(constrained_gid, [])
                for subject_id, predicate_id, object_id in local_triples:
                    edges.append((factor_local_id, subject_id))
                    edge_types.append(EDGE_FACTOR_TO_LOCAL_SUBJECT)
                    edges.append((subject_id, factor_local_id))
                    edge_types.append(EDGE_LOCAL_SUBJECT_TO_FACTOR)
                    edges.append((factor_local_id, object_id))
                    edge_types.append(EDGE_FACTOR_TO_LOCAL_OBJECT)
                    edges.append((object_id, factor_local_id))
                    edge_types.append(EDGE_LOCAL_OBJECT_TO_FACTOR)
                    wiring_edges_created += 2
                    if focus_local_nodes.get("predicate") == predicate_id:
                        matched_focus_predicate = True

        for predicate_id_raw, object_id_raw in zip(param_predicates, param_objects):
            try:
                predicate_gid = global_int_encoder.encode(predicate_id_raw, add_new=False)
            except Exception as exc:
                raise AssertionError(f"Missing predicate token '{predicate_id_raw}' in encoder.") from exc
            try:
                object_gid = global_int_encoder.encode(object_id_raw, add_new=False)
            except Exception as exc:
                raise AssertionError(f"Missing object token '{object_id_raw}' in encoder.") from exc

            add_factor_definition_edge(
                factor_local_id=factor_local_id,
                predicate_global_id=predicate_gid,
                object_global_id=object_gid,
            )
        if debug_factor_wiring:
            debug_entries.append(
                {
                    "constraint_id": int(constraint_id),
                    "constraint_token": str(constraint_token),
                    "constraint_type": constraint_type,
                    "constrained_property": constrained_property,
                    "observed_predicates": [int(gid) for gid in observed_predicates],
                    "matched_pred_local_ids": int(matched_predicate_local_ids),
                    "wiring_edges_created": int(wiring_edges_created),
                    "matched_focus_predicate": bool(matched_focus_predicate),
                    "scope_predicate_counts": {
                        str(int(gid)): int(count) for gid, count in scope_predicate_counts.items()
                    },
                }
            )
            if constraint_id == int(graph["constraint_id"]):
                primary_factor_focus_scope_ok = matched_focus_predicate

    assert primary_factor_index >= 0, "Primary constraint_id missing from factor list."
    if factorized_representation and debug_factor_wiring and primary_factor_focus_scope_ok is not None:
        assert primary_factor_focus_scope_ok, "Primary factor missing scope edge to focus predicate."
        if factor_local_ids:
            factor_set = set(factor_local_ids)
            reverse_edge_types = {
                EDGE_LOCAL_PREDICATE_TO_FACTOR,
                EDGE_LOCAL_SUBJECT_TO_FACTOR,
                EDGE_LOCAL_OBJECT_TO_FACTOR,
            }
            has_incoming = False
            for (src, dst), etype in zip(edges, edge_types):
                if dst in factor_set and etype in reverse_edge_types:
                    has_incoming = True
                    break
            assert has_incoming, "Factor nodes missing incoming variable edges."

    # --- Finalise -----------------------------------------------------------
    num_nodes = len(global_to_local_id_encoder.local_attributes)
    role_flags = torch.full((num_nodes,), ROLE_NONE, dtype=torch.long)
    for key, role_value in (
        ("object", ROLE_OBJECT),
        ("predicate", ROLE_PREDICATE),
        ("subject", ROLE_SUBJECT),
    ):
        local_id = focus_local_nodes.get(key)
        if local_id is not None:
            role_flags[local_id] |= role_value

    if encoding == "text_embedding":
        x_np = np.asarray(global_to_local_id_encoder.local_attributes, dtype=resolved_embedding_dtype)
        if not x_np.flags["C_CONTIGUOUS"]:
            x_np = np.ascontiguousarray(x_np)
        x_tensor = torch.from_numpy(x_np)
    else:
        # integer node ids case
        x_tensor = torch.tensor(global_to_local_id_encoder.local_attributes, dtype=torch.long)

    data_graph = Data(
        x=x_tensor,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        edge_index_non_flattened=torch.tensor(edges_non_flattened, dtype=torch.long).t().contiguous(),
        edge_attr_non_flattened=torch.tensor(non_flattened_edge_attributes, dtype=torch.long),
        y=torch.tensor(
            [
                [
                    _normalize_target_id(graph["add_subject"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["add_predicate"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["add_object"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["del_subject"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["del_predicate"], global_int_encoder, unknown_global_id),
                    _normalize_target_id(graph["del_object"], global_int_encoder, unknown_global_id),
                ]
            ],
            dtype=torch.long,
        ),
        role_flags=role_flags,
    )
    if include_debug_fields:
        data_graph.x_names = global_to_local_id_encoder.local_names

    # Baseline metadata
    # Violating triple in global ID space
    data_graph.focus_triple = torch.tensor([graph["subject"], graph["predicate"], graph["object"]], dtype=torch.long)
    data_graph.shape_id = int(graph["constraint_id"])
    # Standardize on `constraint_type` across the pipeline
    data_graph.constraint_type = str(graph["constraint_type"])
    data_graph.constraint_representation = constraint_representation
    data_graph.factor_constraint_ids = torch.tensor(factor_constraint_ids, dtype=torch.long)
    data_graph.factor_node_index = torch.tensor(factor_local_ids, dtype=torch.long)
    data_graph.primary_factor_index = int(primary_factor_index)
    if include_debug_fields:
        data_graph.factor_constraint_types = factor_constraint_types
    if factor_checkable_pre_tensor is not None:
        data_graph.factor_checkable_pre = factor_checkable_pre_tensor
        data_graph.factor_satisfied_pre = factor_satisfied_pre_tensor
        data_graph.factor_checkable_post_gold = factor_checkable_post_gold_tensor
        data_graph.factor_satisfied_post_gold = factor_satisfied_post_gold_tensor
        data_graph.factor_types = factor_types_tensor
    if include_debug_fields and debug_factor_wiring:
        data_graph.factor_wiring_debug = {
            "constraint_id": int(graph["constraint_id"]),
            "primary_constraint_id": int(graph["constraint_id"]),
            "primary_factor_index": int(primary_factor_index),
            "local_constraint_count": int(len(factor_ids)),
            "focus_predicate_local_id": int(focus_local_nodes.get("predicate", -1)),
            "focus_predicate_global_id": int(graph.get("predicate") or 0),
            "other_predicate_global_id": int(graph.get("other_predicate") or 0),
            "factors": debug_entries,
        }

    if factor_local_ids:
        is_factor_node = torch.zeros(num_nodes, dtype=torch.bool)
        is_factor_node[data_graph.factor_node_index] = True
        data_graph.is_factor_node = is_factor_node

    return data_graph


def _parquet_num_rows(path: Path) -> int:
    """Return the total number of rows in a parquet file."""
    return int(pq.ParquetFile(path).metadata.num_rows)


def iter_parquet_rows(
    path: Path,
    batch_size: int = 4096,
    max_rows: int = 0,
    skip_rows: int = 0,
) -> Iterator[dict[str, Any]]:
    """
    Stream parquet rows as Python dictionaries in bounded batches.
    """
    parquet_file = pq.ParquetFile(path)
    emitted = 0
    seen = 0
    skip_limit = max(0, int(skip_rows))
    if max_rows and max_rows > 0:
        remaining = max(0, int(max_rows) - skip_limit)
        limit = remaining
    else:
        limit = None
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            seen += 1
            if seen <= skip_limit:
                continue
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _maybe_add_target(value: Any, store: set[int]) -> None:
    """Add a scalar integer target to ``store`` if possible."""
    if value is None:
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        # Target slots should be scalar but guard against accidental iterables.
        for item in value:
            _maybe_add_target(item, store)
        return

    try:
        idx = int(value)
    except (TypeError, ValueError):
        return
    if idx >= 0:
        store.add(idx)


def _update_target_sets(graph: dict[str, Any], entity_store: set[int], predicate_store: set[int]) -> None:
    """Collect target class ids from a raw graph row."""
    for key in ("add_subject", "add_object", "del_subject", "del_object"):
        _maybe_add_target(graph.get(key), entity_store)
    for key in ("add_predicate", "del_predicate"):
        _maybe_add_target(graph.get(key), predicate_store)


def compute_torch_geometric_objects(
    rows: Iterable[dict[str, Any]],
    wikidata_cache: PrecomputedWikidataCache | None,
    global_int_encoder: GlobalIntEncoder,
    constraint_registry: Dict[str, Dict[str, Any]],
    encoding: Literal["node_id", "text_embedding"],
    constraint_scope: Literal["local", "focus"] = "local",
    constraint_representation: Literal["factorized", "eswc_passive"] = "factorized",
    store_node_names: bool = False,
    include_debug_fields: bool = True,
    embedding_dtype: np.dtype | None = None,
    on_sample: Optional[Callable[[Data], None]] = None,
    debug_state: dict[str, Any] | None = None,
    total_rows: int | None = None,
) -> Iterator[Data]:
    """
    Build a stream of torch_geometric.data.Data graphs from row dictionaries.

    Yields
    -----
    torch_geometric.data.Data
        Individual graphs emitted one-by-one to avoid materialising the full
        dataset in memory.
    """
    for row_idx, row in enumerate(tqdm(rows, desc="Processing rows", total=total_rows)):
        graph = create_graph(
            row,
            wikidata_cache=wikidata_cache,
            global_int_encoder=global_int_encoder,
            constraint_registry=constraint_registry,
            encoding=encoding,
            constraint_scope=constraint_scope,
            constraint_representation=constraint_representation,
            store_node_names=store_node_names,
            include_debug_fields=include_debug_fields,
            embedding_dtype=embedding_dtype,
            debug_factor_wiring=bool(debug_state and debug_state.get("enabled")),
        )
        if on_sample is not None:
            on_sample(graph)
        if include_debug_fields and debug_state and debug_state.get("enabled") and not debug_state.get("written"):
            debug_payload = getattr(graph, "factor_wiring_debug", None)
            if debug_payload and debug_payload.get("local_constraint_count", 0) > 1:
                debug_payload = dict(debug_payload)
                debug_payload["row_index"] = int(row_idx)
                for entry in debug_payload.get("factors", []):
                    logging.info(
                        "Factor wiring debug: constraint_id=%s constraint_type=%s constrained_property=%s observed_predicates=%s matched_pred_local_ids=%s wiring_edges_created=%s",
                        entry.get("constraint_id"),
                        entry.get("constraint_type"),
                        entry.get("constrained_property"),
                        entry.get("observed_predicates"),
                        entry.get("matched_pred_local_ids"),
                        entry.get("wiring_edges_created"),
                    )
                debug_path = Path(debug_state["path"])
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with debug_path.open("w", encoding="utf-8") as fh:
                    json.dump(debug_payload, fh)
                debug_state["written"] = True
        yield graph


def check_data_graph(geometric_objects: Iterable[Data], batch_size: int = 32) -> None:
    """
    Sanity-check that the graphs can be batched without shape mismatches.
    """
    objects = list(geometric_objects)
    if not objects:
        logging.warning("No graph objects available for sanity check.")
        return

    loader = DataLoader(objects, batch_size=batch_size)

    for batch in loader:
        logging.info("Sample batch size: %s", batch.num_graphs)
        logging.info("Sample edge index shape: %s", batch.edge_index.shape)
        logging.info("Sample node features shape: %s", batch.x.shape)
        logging.info("Sample label shape: %s", batch.y.shape)
        break  # Just to check the first batch


def collect_sample_for_check(
    output_path: Path,
    shard_size: int,
    use_torch_save: bool,
    sample_size: int,
) -> list[Data]:
    """Collect up to ``sample_size`` graphs from the written artifacts."""
    if sample_size <= 0:
        return []

    output_path = Path(output_path)

    if shard_size <= 0:
        return list(itertools.islice(iter_stream(output_path), sample_size))

    suffix = ".pt" if use_torch_save else output_path.suffix
    sample: list[Data] = []
    for shard_path in sorted(output_path.parent.glob(f"{output_path.stem}-shard*{suffix}")):
        if use_torch_save:
            shard_objects = _torch_load_trusted(shard_path)
        else:
            with open(shard_path, "rb") as f:
                shard_objects = pickle.load(f)
        for obj in shard_objects:
            sample.append(obj)
            if len(sample) >= sample_size:
                del shard_objects
                return sample

        del shard_objects

    return sample


def _profile_debug_fields_enabled(persistence_profile: str) -> bool:
    return persistence_profile == PERSISTENCE_PROFILE_FULL


def _profile_kept_fields(persistence_profile: str) -> list[str]:
    core_fields = [
        "x",
        "edge_index",
        "edge_type",
        "edge_index_non_flattened",
        "edge_attr_non_flattened",
        "y",
        "role_flags",
        "focus_triple",
        "shape_id",
        "constraint_type",
        "factor_constraint_ids",
        "factor_node_index",
        "primary_factor_index",
        "factor_checkable_pre",
        "factor_satisfied_pre",
        "factor_checkable_post_gold",
        "factor_satisfied_post_gold",
        "factor_types",
        "is_factor_node",
    ]
    if _profile_debug_fields_enabled(persistence_profile):
        core_fields.extend(["x_names", "factor_constraint_types", "factor_wiring_debug"])
    return core_fields


def _profile_dropped_fields(persistence_profile: str) -> list[str]:
    if _profile_debug_fields_enabled(persistence_profile):
        return []
    return ["x_names", "factor_constraint_types", "factor_wiring_debug"]


def _manifest_path_for_split(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def _load_existing_vocab_targets(vocab_path: Path, split: str) -> dict[str, list[int]] | None:
    if not vocab_path.exists():
        return None
    try:
        with vocab_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        per_split = payload.get("per_split") if isinstance(payload, dict) else None
        split_payload = per_split.get(split) if isinstance(per_split, dict) else None
        if not isinstance(split_payload, dict):
            return None
        entity_ids = split_payload.get("entity_class_ids")
        predicate_ids = split_payload.get("predicate_class_ids")
        if not isinstance(entity_ids, list) or not isinstance(predicate_ids, list):
            return None
        return {"entity_class_ids": entity_ids, "predicate_class_ids": predicate_ids}
    except Exception:
        logging.exception("Failed to load existing target vocab payload from %s", vocab_path)
        return None


def _clear_existing_shards(output_path: Path) -> None:
    for shard_path in sorted(output_path.parent.glob(f"{output_path.stem}-shard*.pkl")):
        shard_path.unlink(missing_ok=True)
    for shard_path in sorted(output_path.parent.glob(f"{output_path.stem}-shard*.pt")):
        shard_path.unlink(missing_ok=True)


def _discover_split_shards(output_path: Path, use_torch_save: bool) -> list[Path]:
    suffix = ".pt" if use_torch_save else output_path.suffix
    return sorted(output_path.parent.glob(f"{output_path.stem}-shard*{suffix}"))


def _extract_shard_index(shard_path: Path) -> int | None:
    match = re.search(r"-shard(\d+)$", shard_path.stem)
    if not match:
        return None
    return int(match.group(1))


def _load_shard_payload(shard_path: Path) -> list[Data]:
    if shard_path.suffix == ".pt":
        payload = _torch_load_trusted(shard_path)
    else:
        with shard_path.open("rb") as f:
            payload = pickle.load(f)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {shard_path}, got {type(payload)!r}")
    return payload


def _load_resume_state(
    shard_paths: list[Path],
    split_entity_targets: set[int],
    split_predicate_targets: set[int],
    total_entity_targets: set[int],
    total_predicate_targets: set[int],
) -> tuple[int, int]:
    if not shard_paths:
        return 0, 0
    indexed_paths: list[tuple[int, Path]] = []
    for shard_path in shard_paths:
        index = _extract_shard_index(shard_path)
        if index is None:
            raise ValueError(f"Unexpected shard naming format: {shard_path.name}")
        indexed_paths.append((index, shard_path))
    indexed_paths = sorted(indexed_paths, key=lambda item: item[0])
    indices = [idx for idx, _ in indexed_paths]
    expected = list(range(min(indices), max(indices) + 1))
    if indices != expected or (indices and indices[0] != 0):
        raise ValueError(
            "Resume requires contiguous shard indices starting at 000; found "
            f"indices={indices[:5]}...{indices[-5:] if len(indices) > 5 else indices}"
        )

    existing_graphs = 0
    next_shard = indices[-1] + 1
    for pos, (index, shard_path) in enumerate(indexed_paths):
        try:
            shard_objects = _load_shard_payload(shard_path)
        except Exception as exc:
            is_last = pos == len(indexed_paths) - 1
            if not is_last:
                raise
            logging.warning(
                "Ignoring unreadable trailing shard %s (%s); resume will overwrite it.",
                shard_path,
                exc,
            )
            next_shard = index
            break
        existing_graphs += len(shard_objects)
        for graph in shard_objects:
            y = getattr(graph, "y", None)
            if y is None:
                continue
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            if y.ndim == 1:
                y = y.unsqueeze(0)
            if y.numel() < 6:
                continue
            add_subject, add_predicate, add_object, del_subject, del_predicate, del_object = y[0].tolist()
            for idx in (add_subject, add_object, del_subject, del_object):
                split_entity_targets.add(int(idx))
                total_entity_targets.add(int(idx))
            for idx in (add_predicate, del_predicate):
                split_predicate_targets.add(int(idx))
                total_predicate_targets.add(int(idx))
        del shard_objects
    return existing_graphs, next_shard


def _log_free_space_estimate(
    processed_path: Path,
    split: str,
    shard_paths: list[Path],
    shard_size: int,
    remaining_rows: int,
) -> None:
    if shard_size <= 0 or remaining_rows <= 0:
        return
    if not shard_paths:
        return
    total_shard_bytes = sum(path.stat().st_size for path in shard_paths)
    avg_shard_bytes = total_shard_bytes / max(len(shard_paths), 1)
    estimated_remaining_shards = int(np.ceil(remaining_rows / shard_size))
    estimated_remaining_bytes = int(avg_shard_bytes * estimated_remaining_shards)
    free_bytes = shutil.disk_usage(processed_path).free
    logging.info(
        "Disk estimate for %s: remaining_rows=%s estimated_remaining_shards=%s estimated_bytes=%.2f GB free=%.2f GB",
        split,
        remaining_rows,
        estimated_remaining_shards,
        estimated_remaining_bytes / (1024**3),
        free_bytes / (1024**3),
    )
    if estimated_remaining_bytes > free_bytes:
        logging.warning(
            "Estimated remaining disk required for %s (~%.2f GB) exceeds free disk (~%.2f GB).",
            split,
            estimated_remaining_bytes / (1024**3),
            free_bytes / (1024**3),
        )


def _write_split_manifest(
    split: str,
    output_path: Path,
    artifact_writes: list[ArtifactWriteResult],
    graph_count: int,
    shard_count: int,
    shard_size: int,
    use_torch_save: bool,
    persistence_profile: str,
    overwrite_mode: str,
    encoding: str,
    dataset_variant: str,
    constraint_scope: str,
    constraint_representation: str,
) -> None:
    manifest = {
        "split": split,
        "dataset_variant": dataset_variant,
        "encoding": encoding,
        "constraint_scope": constraint_scope,
        "constraint_representation": constraint_representation,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph_count": int(graph_count),
        "shard_count": int(shard_count),
        "shard_size": int(shard_size),
        "sharded": bool(shard_size > 0),
        "use_torch_save": bool(use_torch_save),
        "persistence_profile": persistence_profile,
        "overwrite_mode": overwrite_mode,
        "kept_fields": _profile_kept_fields(persistence_profile),
        "dropped_fields": _profile_dropped_fields(persistence_profile),
        "artifacts": [
            {
                "path": str(artifact.path),
                "bytes": int(artifact.bytes_written),
                "checksum": artifact.checksum,
                "checksum_mode": "sha256_prefix_16mb",
            }
            for artifact in artifact_writes
        ],
    }
    manifest_path = _manifest_path_for_split(output_path)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logging.info("Wrote split manifest to %s", manifest_path)


def main(
    wikidata_cache: PrecomputedWikidataCache | None,
    global_int_encoder: GlobalIntEncoder,
    constraint_registry: Dict[str, Dict[str, Any]],
    encoding: Literal["node_id", "text_embedding"],
    dataset_variant: str,
    constraint_scope: Literal["local", "focus"] = "local",
    constraint_representation: Literal["factorized", "eswc_passive"] = "factorized",
    store_node_names: bool = False,
    persistence_profile: str = PERSISTENCE_PROFILE_RESEARCH_SAFE,
    shard_size: int = 0,
    use_torch_save: bool = False,
    overwrite_mode: str = OVERWRITE_MODE_ATOMIC,
    resume_partial_shards: bool = False,
    embedding_dtype: np.dtype | None = None,
    check_sample_size: int = 32,
    max_instances: int = 0,
    debug_factor_wiring: bool = False,
) -> dict[str, Path]:
    """Sequentially build graphs per split and persist them to disk."""

    split_to_parquet = {
        "train": "df_train.parquet",
        "val": "df_val.parquet",
        "test": "df_test.parquet",
    }

    outputs: dict[str, Path] = {}

    # Record target vocabularies seen in the labels (`add_*` / `del_*`)
    # so training can precompute class vocabularies.
    total_entity_targets: set[int] = {0}
    total_predicate_targets: set[int] = {0}
    split_targets: dict[str, dict[str, list[int]]] = {}

    debug_state = {
        "enabled": debug_factor_wiring and _profile_debug_fields_enabled(persistence_profile),
        "written": False,
        "path": PROCESSED_DATA_PATH / "factor_wiring_debug.json",
    }
    include_debug_fields = _profile_debug_fields_enabled(persistence_profile)
    vocab_path = PROCESSED_DATA_PATH / "target_vocabs.json"

    for split, parquet_name in split_to_parquet.items():
        logging.info("Processing %s split", split)

        parquet_path = INTERIM_DATA_PATH / parquet_name
        split_total_rows = _parquet_num_rows(parquet_path)
        if max_instances and max_instances > 0:
            limit = min(max_instances, split_total_rows)
            logging.info("Limiting %s split to first %s instances", split, limit)
            split_total_rows = limit

        split_entity_targets: set[int] = {0}
        split_predicate_targets: set[int] = {0}

        def collect_targets(graph: Data):
            y = graph.y
            if y is None:
                raise ValueError("Missing graph.y targets for sanity check.")
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            if y.ndim == 1:
                y = y.unsqueeze(0)
            if y.shape != (1, 6):
                raise ValueError(f"Expected y to be of shape (1, 6), got {tuple(y.shape)}")
            add_subject, add_predicate, add_object, del_subject, del_predicate, del_object = y[0].tolist()
            for idx in (add_subject, add_object, del_subject, del_object):
                split_entity_targets.add(idx)
                total_entity_targets.add(idx)
            for idx in (add_predicate, del_predicate):
                split_predicate_targets.add(idx)
                total_predicate_targets.add(idx)

        output_path = PROCESSED_DATA_PATH / graph_dataset_filename(
            split,
            encoding,
            constraint_representation=constraint_representation,
        )
        existing_artifacts = discover_graph_artifacts(output_path)
        if overwrite_mode == OVERWRITE_MODE_SKIP and existing_artifacts:
            logging.info(
                "Skipping %s split because artifacts already exist and overwrite mode is 'skip'.",
                split,
            )
            existing_targets = _load_existing_vocab_targets(vocab_path, split)
            if existing_targets:
                split_entity_targets.update(existing_targets["entity_class_ids"])
                split_predicate_targets.update(existing_targets["predicate_class_ids"])
                total_entity_targets.update(existing_targets["entity_class_ids"])
                total_predicate_targets.update(existing_targets["predicate_class_ids"])
            else:
                logging.warning(
                    "No reusable target vocab entries found for skipped split %s; "
                    "vocab output may be incomplete until regenerated.",
                    split,
                )
            outputs[split] = output_path
            split_targets[split] = {
                "entity_class_ids": sorted(split_entity_targets),
                "predicate_class_ids": sorted(split_predicate_targets),
            }
            continue

        resume_skip_rows = 0
        start_shard = 0
        existing_shard_count = 0
        existing_split_shards: list[Path] = []
        if resume_partial_shards and shard_size > 0:
            existing_split_shards = _discover_split_shards(output_path, use_torch_save=use_torch_save)
            if existing_split_shards:
                resume_skip_rows, start_shard = _load_resume_state(
                    existing_split_shards,
                    split_entity_targets=split_entity_targets,
                    split_predicate_targets=split_predicate_targets,
                    total_entity_targets=total_entity_targets,
                    total_predicate_targets=total_predicate_targets,
                )
                existing_shard_count = start_shard
                split_total_rows = max(0, split_total_rows - resume_skip_rows)
                _log_free_space_estimate(
                    PROCESSED_DATA_PATH,
                    split=split,
                    shard_paths=existing_split_shards[:existing_shard_count],
                    shard_size=shard_size,
                    remaining_rows=split_total_rows,
                )
                logging.info(
                    "Resuming %s split from existing shards: shard_count=%s resume_skip_rows=%s next_shard=%s",
                    split,
                    existing_shard_count,
                    resume_skip_rows,
                    start_shard,
                )

        # Build graphs (the heavy lifting happens here)
        row_iter = iter_parquet_rows(
            parquet_path,
            max_rows=max_instances,
            skip_rows=resume_skip_rows,
        )
        generator = compute_torch_geometric_objects(
            row_iter,
            wikidata_cache,
            global_int_encoder,
            constraint_registry,
            encoding=encoding,
            constraint_scope=constraint_scope,
            constraint_representation=constraint_representation,
            store_node_names=store_node_names,
            include_debug_fields=include_debug_fields,
            embedding_dtype=embedding_dtype,
            on_sample=collect_targets,
            debug_state=debug_state,
            total_rows=split_total_rows,
        )
        if shard_size > 0:
            if not (resume_partial_shards and existing_split_shards):
                output_path.unlink(missing_ok=True)
                _clear_existing_shards(output_path)
        else:
            _clear_existing_shards(output_path)

        # Persist graphs
        total_objects = 0
        shard_count = 0
        artifact_writes: list[ArtifactWriteResult] = []
        atomic_write = overwrite_mode == OVERWRITE_MODE_ATOMIC
        if shard_size > 0:
            total_objects, shard_count, artifact_writes = dump_in_shards(
                generator,
                output_path,
                shard_size=shard_size,
                use_torch_save=use_torch_save,
                atomic_write=atomic_write,
                start_shard=start_shard,
            )
            shard_count += existing_shard_count
            total_objects += resume_skip_rows
        else:
            total_objects, stream_artifact = dump_stream(
                generator,
                output_path,
                atomic_write=atomic_write,
            )
            artifact_writes = [stream_artifact]
            shard_count = 1 if total_objects else 0

        logging.info(
            "Finished writing %s split (%s graphs across %s shard%s)",
            split,
            total_objects,
            shard_count,
            "s" if shard_count != 1 else "",
        )
        _write_split_manifest(
            split=split,
            output_path=output_path,
            artifact_writes=artifact_writes,
            graph_count=total_objects,
            shard_count=shard_count,
            shard_size=shard_size,
            use_torch_save=use_torch_save,
            persistence_profile=persistence_profile,
            overwrite_mode=overwrite_mode,
            encoding=encoding,
            dataset_variant=dataset_variant,
            constraint_scope=constraint_scope,
            constraint_representation=constraint_representation,
        )

        del generator

        # Sanity check
        sample_size = min(check_sample_size, shard_size) if shard_size > 0 else check_sample_size
        sample_graphs = collect_sample_for_check(
            output_path,
            shard_size=shard_size,
            use_torch_save=use_torch_save,
            sample_size=sample_size,
        )
        if sample_graphs:
            check_data_graph(sample_graphs)
        else:
            logging.warning("No data available for sanity checking %s split", split)
        del sample_graphs

        outputs[split] = output_path
        gc.collect()

        split_targets[split] = {
            "entity_class_ids": sorted(split_entity_targets),
            "predicate_class_ids": sorted(split_predicate_targets),
        }

    vocab_payload = {
        "entity_class_ids": sorted(total_entity_targets),
        "predicate_class_ids": sorted(total_predicate_targets),
        "per_split": split_targets,
    }
    with vocab_path.open("w", encoding="utf-8") as fh:
        json.dump(vocab_payload, fh)
    logging.info(
        "Stored target vocabularies in %s | entities=%s predicates=%s",
        vocab_path,
        len(vocab_payload["entity_class_ids"]),
        len(vocab_payload["predicate_class_ids"]),
    )

    return outputs


def display_graph(encoder: GlobalIntEncoder) -> None:
    train_output = outputs.get("train")
    graph_to_show: Optional[Data] = None

    if train_output is None:
        logging.warning("Train split output not found; skipping graph visualisation.")
    elif args.shard_size > 0:
        suffix = ".pt" if args.use_torch_save else train_output.suffix
        for shard_path in sorted(train_output.parent.glob(f"{train_output.stem}-shard*{suffix}")):
            if args.use_torch_save:
                shard_objects = _torch_load_trusted(shard_path)
            else:
                with open(shard_path, "rb") as f:
                    shard_objects = pickle.load(f)
            if shard_objects:
                graph_to_show = shard_objects[0]
                del shard_objects
                break
            del shard_objects
    else:
        graph_to_show = next(iter_stream(train_output), None)

    if graph_to_show is None:
        logging.warning("No graph available to display.")
    else:
        graph_to_show.validate()

        print("Graph to show:")
        print(graph_to_show)

        x_names = getattr(graph_to_show, "x_names", None)
        node_attrs = ["x_names"] if x_names is not None else []
        g = utils.to_networkx(
            graph_to_show,
            to_undirected=False,
            node_attrs=node_attrs,
        )
        labels = dict(enumerate(x_names)) if x_names is not None else None
        nx.draw(
            g,
            pos=nx.spring_layout(g, k=7 / g.order() ** (1 / 2)),
            with_labels=True,
            node_size=500,
            font_size=8,
            labels=labels,
        )
        plt.savefig(PROCESSED_DATA_PATH / "graph_visualization.png")
        plt.show()

        # show non-flattened graph
        if graph_to_show.edge_index_non_flattened is None:
            raise ValueError("Missing edge_index_non_flattened for non-flattened display.")
        graph_to_show.edge_index = graph_to_show.edge_index_non_flattened
        graph_to_show.validate()
        g = utils.to_networkx(
            graph_to_show,
            to_undirected=False,
            node_attrs=node_attrs,
            edge_attrs=["edge_attr_non_flattened"],
        )
        pos = nx.spring_layout(g, k=7 / g.order() ** (1 / 2))
        x = graph_to_show.x
        if x is None:
            raise ValueError("Missing node features for graph visualization.")
        edge_attr = graph_to_show.edge_attr_non_flattened
        if edge_attr is None:
            raise ValueError("Missing edge_attr_non_flattened for edge label display.")
        nx.draw(
            g,
            pos=pos,
            with_labels=True,
            node_size=500,
            font_size=8,
            labels={
                i: encoder.decode(int(global_id.item()), use_filtered_id_mapping=True)
                for i, global_id in enumerate(x)
            },
        )
        nx.draw_networkx_edge_labels(
            g,
            pos=pos,
            edge_labels={
                (u, v): encoder.decode(
                    int(x[int(edge_attr[i].item())].item()),
                    use_filtered_id_mapping=True,
                )
                for i, (u, v) in enumerate(g.edges())
            },
        )
        plt.savefig(PROCESSED_DATA_PATH / "graph_visualization-non_flattened.png")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Build PyG graphs from interim dataframes")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset variant to graph, e.g. full or full_strat1m.",
    )
    parser.add_argument(
        "--registry-dataset",
        default=None,
        help="Raw dataset name for constraint_registry_<dataset>.parquet. Defaults to --dataset.",
    )
    parser.add_argument(
        "--encoding",
        choices=["node_id", "text_embedding"],
        required=True,
        help="Encoding options for the Wikidata URIs",
    )
    parser.add_argument(
        "--min-occurrence",
        type=int,
        default=100,
        help="Dataset variant produced with the given training frequency threshold.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="Number of graphs per shard when streaming to disk (0 disables sharding).",
    )
    parser.add_argument(
        "--max_instances",
        "--max-instances",
        type=int,
        default=0,
        help="Limit the number of rows converted per split (0 uses the full split).",
    )
    parser.add_argument(
        "--use-torch-save",
        action="store_true",
        help="Persist shards with torch.save instead of pickle for reduced overhead.",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Visualise the first graph using NetworkX and Matplotlib.",
    )
    parser.add_argument(
        "--wikidata-cache-path",
        type=str,
        default="data/interim/wikidata_text.parquet",
        help="Path to the precomputed Wikidata cache parquet file.",
    )
    parser.add_argument(
        "--debug_factor_wiring",
        "--debug-factor-wiring",
        action="store_true",
        help="Write factor wiring diagnostics for the first multi-constraint instance.",
    )
    parser.add_argument(
        "--constraint-scope",
        choices=["local", "focus"],
        default="local",
        help="Which constraint neighborhood to use for factor nodes (default: local).",
    )
    parser.add_argument(
        "--constraint-representation",
        choices=["factorized", "eswc_passive"],
        default="factorized",
        help="Graph representation regime: factorized (default) or eswc_passive.",
    )
    parser.add_argument(
        "--use-unlabeled-interim",
        action="store_true",
        help="Force using data/interim/<variant> even if a labeled interim dataset exists.",
    )
    parser.add_argument(
        "--persistence-profile",
        choices=PERSISTENCE_PROFILE_CHOICES,
        default=PERSISTENCE_PROFILE_RESEARCH_SAFE,
        help=(
            "Field persistence profile. 'research_safe' drops debug-only fields "
            "(x_names, factor_constraint_types, factor_wiring_debug) to reduce disk usage."
        ),
    )
    parser.add_argument(
        "--overwrite",
        choices=OVERWRITE_MODE_CHOICES,
        default=OVERWRITE_MODE_ATOMIC,
        help=(
            "Write mode: 'atomic' writes to *.tmp then renames (crash-safe), "
            "'unsafe' writes directly to final paths, 'skip' reuses existing split artifacts."
        ),
    )
    parser.add_argument(
        "--resume-partial-shards",
        action="store_true",
        help=(
            "Resume shard generation when shard files already exist: counts existing shards, "
            "skips already-processed rows, and continues from the next shard index."
        ),
    )

    args = parser.parse_args()

    if args.shard_size < 0:
        parser.error("--shard-size must be non-negative")
    if args.max_instances < 0:
        parser.error("--max-instances must be non-negative")
    if args.debug_factor_wiring and args.persistence_profile != PERSISTENCE_PROFILE_FULL:
        parser.error("--debug-factor-wiring requires --persistence-profile full")
    if args.resume_partial_shards and args.shard_size <= 0:
        parser.error("--resume-partial-shards requires --shard-size > 0")
    if args.resume_partial_shards and args.overwrite == OVERWRITE_MODE_SKIP:
        parser.error("--resume-partial-shards is incompatible with --overwrite skip")
    if args.constraint_representation == "eswc_passive" and args.constraint_scope != "local":
        parser.error("--constraint-scope is only meaningful for factorized graphs")

    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    text_embedding_dtype: np.dtype = np.dtype("float16")

    # Parse command-line arguments
    args = parse_args()
    DATASET: str = args.dataset
    MIN_OCCURRENCE = max(1, args.min_occurrence)
    dataset_variant = dataset_variant_name(DATASET, MIN_OCCURRENCE)

    # .gitignore the data files
    gitignore = Path("data/.gitignore")
    gitignore.write_text("*\n!.gitignore\n")

    # Define data paths
    base_interim_path = Path("data/interim/") / dataset_variant  # Load encoder + base data
    labeled_interim_path = Path("data/interim/") / f"{dataset_variant}_labeled"
    use_labeled_interim = (
        not args.use_unlabeled_interim
        and labeled_interim_path.exists()
        and (labeled_interim_path / "df_train.parquet").exists()
    )
    INTERIM_DATA_PATH = labeled_interim_path if use_labeled_interim else base_interim_path
    PROCESSED_DATA_PATH = Path("data/processed/") / dataset_variant  # Save data
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    if use_labeled_interim:
        logging.info("Using labeled interim dataframes from %s", INTERIM_DATA_PATH)
    else:
        logging.info("Using interim dataframes from %s", INTERIM_DATA_PATH)

    logging.info(
        "Building graphs for dataset=%s (variant=%s, min_occurrence=%s)",
        DATASET,
        dataset_variant,
        MIN_OCCURRENCE,
    )
    if args.overwrite == OVERWRITE_MODE_UNSAFE:
        logging.warning(
            "Unsafe overwrite mode enabled: writes go directly to destination files and may leave partial outputs on interruption."
        )

    # Load and freeze int encoder
    encoder = GlobalIntEncoder()
    encoder.load(base_interim_path / "globalintencoder.txt")
    encoder.freeze()

    registry_candidates = []
    if args.registry_dataset:
        registry_candidates.append(args.registry_dataset)
    registry_candidates.extend([DATASET, base_dataset_name(DATASET)])
    if "_strat" in base_dataset_name(DATASET):
        registry_candidates.append(base_dataset_name(DATASET).split("_strat", 1)[0])
    registry_path = None
    for candidate in dict.fromkeys(registry_candidates):
        candidate_path = Path("data/interim") / f"constraint_registry_{candidate}.parquet"
        if candidate_path.exists():
            registry_path = candidate_path
            break
    if registry_path is None:
        raise FileNotFoundError(f"No constraint registry found for candidates: {', '.join(dict.fromkeys(registry_candidates))}")
    registry_df = pd.read_parquet(registry_path)
    if "registry_json" not in registry_df.columns:
        raise KeyError(f"Missing registry_json column in {registry_path}")
    constraint_registry: Dict[str, Dict[str, Any]] = {}
    for registry_payload in registry_df["registry_json"]:
        if isinstance(registry_payload, str):
            registry_payload = json.loads(registry_payload)
        if not isinstance(registry_payload, dict):
            raise TypeError(f"Unexpected registry_json payload type: {type(registry_payload)}")
        for key, value in registry_payload.items():
            constraint_registry[str(key)] = value

    # Graph
    LITERAL_ID = encoder.encode("LITERAL_OBJECT", add_new=False)
    if args.encoding == "node_id":
        logging.info("Using node ID encoding for graph nodes.")
        wikidata_cache = None
    else:
        logging.info("Using text embedding encoding for graph nodes.")
        wikidata_cache = PrecomputedWikidataCache(Path(args.wikidata_cache_path))

    outputs = main(
        wikidata_cache=wikidata_cache,
        global_int_encoder=encoder,
        constraint_registry=constraint_registry,
        encoding=args.encoding,
        dataset_variant=dataset_variant,
        constraint_scope=args.constraint_scope,
        constraint_representation=args.constraint_representation,
        persistence_profile=args.persistence_profile,
        shard_size=args.shard_size,
        use_torch_save=args.use_torch_save,
        overwrite_mode=args.overwrite,
        resume_partial_shards=args.resume_partial_shards,
        embedding_dtype=text_embedding_dtype,
        max_instances=args.max_instances,
        debug_factor_wiring=args.debug_factor_wiring,
    )

    if args.show_graph:
        display_graph(encoder)
