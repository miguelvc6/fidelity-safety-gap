#!/usr/bin/env python3
"""Audit that retained predictions exist without injecting historical test labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.candidates import (  # noqa: E402
    CandidateConfig,
    batch_topk_candidate_triples,
    build_candidates,
)
from modules.config import ModelConfig  # noqa: E402
from modules.data_encoders import (  # noqa: E402
    GlobalIntEncoder,
    dataset_variant_name,
    graph_dataset_filename,
)
from modules.evaluation_artifacts import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    PREDICTION_COLUMNS,
    atomic_write_json,
    repository_relative_path,
    sha256_file,
)
from modules.constraint_checkers import VALIDATOR_SEMANTICS_VERSION  # noqa: E402
from modules.repair_eval import (  # noqa: E402
    ConstraintRepairHeuristics,
    load_violation_contexts,
)
from modules.training_utils import load_graph_dataset, placeholder_ids_from_encoder  # noqa: E402


def _load_eval_module():
    module_path = SRC / "09_eval.py"
    spec = importlib.util.spec_from_file_location("eval_09_for_membership_audit", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluation helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--proposal-run-directory", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--examples-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_directory = args.run_directory.resolve()
    config_path = run_directory / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_config = config["model_config"]
    training_config = config.get("training_config") or {}
    variant = dataset_variant_name(
        str(model_config["dataset_variant"]), int(model_config.get("min_occurrence", 100))
    )
    interim_directory = ROOT / "data" / "interim" / variant
    encoder_path = interim_directory / "globalintencoder.txt"

    encoder = GlobalIntEncoder()
    encoder.load(encoder_path)
    encoder.freeze()
    placeholder_ids = placeholder_ids_from_encoder(encoder)
    heuristics = ConstraintRepairHeuristics(
        encoder=encoder,
        placeholder_ids=placeholder_ids,
        none_class=0,
    )
    contexts = load_violation_contexts(interim_directory, "test", none_class=0)

    predictions_path = args.predictions.resolve()
    if predictions_path.suffix.lower() != ".parquet":
        raise ValueError("Candidate membership requires a schema-v3 prediction Parquet artifact.")
    predictions_manifest_path = predictions_path.with_suffix(".manifest.json")
    with predictions_manifest_path.open("r", encoding="utf-8") as handle:
        predictions_manifest = json.load(handle)
    if int(predictions_manifest.get("schema_version", -1)) != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Candidate membership rejects non-schema-v3 predictions.")
    if int(predictions_manifest.get("validator_semantics_version", -1)) != VALIDATOR_SEMANTICS_VERSION:
        raise ValueError("Candidate membership rejects cross-validator predictions.")
    hierarchy_identity = predictions_manifest.get("hierarchy") or {}
    if not hierarchy_identity.get("content_sha256"):
        raise ValueError("Prediction manifest lacks a hierarchy identity.")
    if (predictions_manifest.get("predictions") or {}).get("sha256") != sha256_file(predictions_path):
        raise ValueError("Prediction Parquet checksum does not match its manifest.")
    prediction_frame = pd.read_parquet(predictions_path)
    missing_columns = set(PREDICTION_COLUMNS) - set(prediction_frame.columns)
    if missing_columns:
        raise ValueError(f"Predictions are missing columns: {sorted(missing_columns)}")
    predictions = [
        tuple(int(value) for value in row)
        for row in prediction_frame.loc[:, list(PREDICTION_COLUMNS)].itertuples(index=False, name=None)
    ]
    if len(predictions) != len(contexts):
        raise ValueError(
            f"Prediction/context count mismatch: {len(predictions)} versus {len(contexts)}."
        )

    candidate_config = CandidateConfig(
        topk_candidates=int(training_config.get("topk_candidates", 20)),
        topk_per_slot=int(training_config.get("topk_per_slot", 5)),
        heuristic_max_candidates=int(training_config.get("heuristic_max_candidates", 30)),
        heuristic_max_values=int(training_config.get("heuristic_max_values", 3)),
        include_gold=False,
        max_candidates_total=int(training_config.get("max_candidates_total", 80)),
    )
    maximum_class = max(encoder._decoding, default=0)
    proposal_run_directory = args.proposal_run_directory.resolve()
    proposal_config_path = proposal_run_directory / "config.json"
    with proposal_config_path.open("r", encoding="utf-8") as handle:
        proposal_config = json.load(handle)
    proposal_model_config = ModelConfig.from_mapping(proposal_config["model_config"])
    proposal_variant = dataset_variant_name(
        proposal_model_config.dataset_variant, proposal_model_config.min_occurrence
    )
    if proposal_variant != variant:
        raise ValueError(
            f"Proposal dataset {proposal_variant} does not match prediction dataset {variant}."
        )
    missing_examples: list[dict[str, Any]] = []
    membership_count = 0
    candidate_count_sum = 0
    processed = 0
    graph_path = (
        ROOT
        / "data"
        / "processed"
        / proposal_variant
        / graph_dataset_filename(
            "test",
            proposal_model_config.encoding,
            constraint_representation=proposal_model_config.constraint_representation,
        )
    )
    test_data = load_graph_dataset(graph_path)
    device = EVAL.get_device()
    proposal_model = EVAL.load_trained_model_for_eval(
        run_directory=proposal_run_directory,
        model_cfg=proposal_model_config,
        device=device,
        chooser_support=None,
    )
    proposal_model.eval()
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Candidate membership"):
            graphs = batch.to_data_list()
            outputs = proposal_model.forward_for_evaluation(batch.to(device))
            logits = outputs["edit_logits"] if isinstance(outputs, dict) else outputs
            logits_cpu = logits.detach().cpu()
            batch_add_topk, batch_del_topk = batch_topk_candidate_triples(
                logits,
                topk_triples=candidate_config.topk_candidates,
                topk_per_slot=candidate_config.topk_per_slot,
            )
            for local_index, graph in enumerate(graphs):
                row_index = int(getattr(graph, "context_index", processed))
                context = contexts[row_index]
                prediction = predictions[row_index]
                candidates, gold_index = build_candidates(
                    context=context,
                    heuristics=heuristics,
                    proposal_logits=logits_cpu[local_index],
                    cfg=candidate_config,
                    placeholder_ids=set(placeholder_ids.values()),
                    num_target_ids=maximum_class + 1,
                    precomputed_add_topk=batch_add_topk[local_index],
                    precomputed_del_topk=batch_del_topk[local_index],
                )
                if gold_index is not None:
                    raise RuntimeError(
                        "Gold-excluded membership audit unexpectedly received a gold index."
                    )
                candidate_count_sum += len(candidates)
                if prediction in candidates:
                    membership_count += 1
                elif len(missing_examples) < args.examples_limit:
                    missing_examples.append(
                        {
                            "row_index": row_index,
                            "constraint_type": context.constraint_type,
                            "prediction": list(prediction),
                            "candidate_count": len(candidates),
                        }
                    )
                processed += 1
    if processed != len(predictions):
        raise RuntimeError(f"Audited {processed} rows but expected {len(predictions)}.")

    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "validator_semantics_version": VALIDATOR_SEMANTICS_VERSION,
        "hierarchy": hierarchy_identity,
        "status": "ok" if membership_count == len(predictions) else "failed",
        "run_directory": repository_relative_path(run_directory),
        "config": {
            "path": repository_relative_path(config_path),
            "sha256": sha256_file(config_path),
        },
        "predictions": {
            "path": repository_relative_path(predictions_path),
            "sha256": sha256_file(predictions_path),
            "manifest_path": repository_relative_path(predictions_manifest_path),
            "manifest_sha256": sha256_file(predictions_manifest_path),
        },
        "proposal": {
            "run_directory": repository_relative_path(proposal_run_directory),
            "config": {
                "path": repository_relative_path(proposal_config_path),
                "sha256": sha256_file(proposal_config_path),
            },
            "checkpoint": {
                "path": repository_relative_path(proposal_run_directory / "checkpoint.pth"),
                "sha256": sha256_file(proposal_run_directory / "checkpoint.pth"),
            },
        },
        "dataset_variant": variant,
        "row_count": len(predictions),
        "candidate_protocol": {
            "include_gold": False,
            "source": "heuristic_plus_proposal_topk",
            "topk_proposal_candidates_used": True,
        },
        "membership_count": membership_count,
        "membership_rate": membership_count / len(predictions) if predictions else 0.0,
        "mean_candidate_count": candidate_count_sum / len(predictions)
        if predictions
        else 0.0,
        "missing_examples": missing_examples,
        "inference": (
            "A prediction absent from the reconstructed gold-excluded candidate set "
            "cannot be certified as an output of that candidate protocol."
        ),
    }
    atomic_write_json(args.output.resolve(), report)
    if report["status"] != "ok":
        raise RuntimeError(
            f"Only {membership_count}/{len(predictions)} predictions belong to the label-blind "
            "candidate set."
        )


if __name__ == "__main__":
    main()
