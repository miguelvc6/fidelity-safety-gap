#!/usr/bin/env python3
"""
Generate the paper-facing experiment bundle under ``models/<exp_name>/config.json``.

Default output:
- ``b0_eswc_reproduction``
- ``a1_factorized_imitation_compact_grouped``
- ``m1c_safe_factor_chooser_compact_grouped``
- ``m1d_safe_factor_direct_compact_grouped``
- ``g0_globalfix_reference_v2``

Optional appendix / ablation configs are only emitted with ``--include-experimental``
or ``--include-h2-ablations``.
"""

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import pandas as pd

from modules.data_encoders import base_dataset_name, graph_dataset_filename
from modules.factor_types import dataset_factor_type_ids

VARIANT_MINOCC_RE = re.compile(r"minocc(\d+)", re.IGNORECASE)
FACTORIZED_RE = re.compile(r"^train_graph-(?P<encoding>.+)\.pkl$")
FACTORIZED_SHARD_RE = re.compile(r"^train_graph-(?P<encoding>.+)-shard\d+\.(?:pkl|pt)$")
PASSIVE_RE = re.compile(r"^train_graph_repr-eswc_passive-(?P<encoding>.+)\.pkl$")
PASSIVE_SHARD_RE = re.compile(r"^train_graph_repr-eswc_passive-(?P<encoding>.+)-shard\d+\.(?:pkl|pt)$")
SAFE_STREAMING_NUM_WORKERS = 2
SAFE_STREAMING_PIN_MEMORY = False
VALIDATION_SUBSET_SIZE = 25_000
CHEAPER_NUM_EPOCHS = 10
CHEAPER_EARLY_STOPPING_ROUNDS = 2
CHEAPER_LEARNING_RATE = 1e-4
CHEAPER_GRAD_CLIP = 0.5
CHEAPER_SCHEDULER_FACTOR = 0.5
CHEAPER_SCHEDULER_PATIENCE = 0
LOCKED_NUM_LAYERS = 4
LOCKED_HIDDEN_CHANNELS = 400
LOCKED_HEAD_HIDDEN = 400
LOCKED_DROPOUT = 0.17
LOCKED_WEIGHT_DECAY = 1.1e-4
LOCKED_PRESSURE_RESIDUAL_SCALE = 0.1
LOCKED_FACTOR_LOSS_WEIGHT_PRE = 0.1
LOCKED_FACTOR_LOSS_WEIGHT_POST_GOLD = 0.1
LOCKED_CHOOSER_TOPK_CANDIDATES = 20
LOCKED_CHOOSER_MAX_CANDIDATES_TOTAL = 80
ORIGINAL_B0_TRAINABLE_PARAMETERS = 27_920_000
MATCHED_B0_TRAINABLE_PARAMETERS = 50_110_102
COMPACT_A1_TRAINABLE_PARAMETERS = 50_072_465
MAX_PARAMETER_RELATIVE_DIFFERENCE = 0.001


def assert_parameter_match(
    matched_b0: int = MATCHED_B0_TRAINABLE_PARAMETERS,
    compact_a1: int = COMPACT_A1_TRAINABLE_PARAMETERS,
) -> None:
    relative_difference = abs(int(matched_b0) - int(compact_a1)) / float(compact_a1)
    if relative_difference > MAX_PARAMETER_RELATIVE_DIFFERENCE:
        raise AssertionError(
            "Parameter-matched B0 differs from Compact A1 by "
            f"{relative_difference:.4%}, above the 0.1% limit."
        )


def _parse_min_occurrence(variant: str) -> int:
    match = VARIANT_MINOCC_RE.search(variant)
    return int(match.group(1)) if match else 1


def _torch_load_trusted(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _discover_artifacts(path: Path) -> list[Path]:
    if path.exists():
        return [path]
    artifacts: list[Path] = []
    artifacts.extend(sorted(path.parent.glob(f"{path.stem}-shard*.pkl")))
    artifacts.extend(sorted(path.parent.glob(f"{path.stem}-shard*.pt")))
    return artifacts


def _load_first_data_obj(path: Path) -> Any | None:
    try:
        artifacts = _discover_artifacts(path)
        if not artifacts:
            return None
        first_path = artifacts[0]
        if first_path.suffix == ".pt":
            payload = _torch_load_trusted(first_path)
        else:
            with first_path.open("rb") as fh:
                payload = pickle.Unpickler(fh).load()
        if isinstance(payload, list):
            return payload[0] if payload else None
        return payload
    except Exception:
        return None


def _infer_num_factor_types(sample_data: Any) -> int:
    if sample_data is None:
        return 0
    for attr in ("factor_types", "factor_type_id", "factor_type_ids"):
        if hasattr(sample_data, attr):
            value = getattr(sample_data, attr)
            try:
                return int(value.max().item()) + 1
            except Exception:
                pass
    return 0


def _infer_num_factor_types_from_registry(dataset_variant: str, interim_root: Path = Path("data/interim")) -> int:
    registry_bases = [dataset_variant, base_dataset_name(dataset_variant)]
    if "_strat" in dataset_variant:
        registry_bases.append(dataset_variant.split("_strat", 1)[0])
    candidates = [
        interim_root / f"constraint_registry_{name}.parquet"
        for name in dict.fromkeys(registry_bases)
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "registry_json" in df.columns and len(df) > 0:
            payload = json.loads(df.iloc[0]["registry_json"])
            indices = [
                int(item["constraint_type_index"])
                for item in payload.values()
                if isinstance(item, dict) and item.get("constraint_type_index") is not None
            ]
            if indices:
                return max(indices) + 1
        for column in ("constraint_type_index", "constraint_type_id"):
            if column in df.columns and len(df[column].dropna()) > 0:
                return int(df[column].max()) + 1
    return 0


def _iter_variant_encodings(processed_root: Path) -> Iterable[tuple[str, str]]:
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")

    for variant_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        encodings: set[str] = set()
        for candidate in sorted(variant_dir.iterdir()):
            if not candidate.is_file():
                continue
            for pattern in (FACTORIZED_RE, FACTORIZED_SHARD_RE, PASSIVE_RE, PASSIVE_SHARD_RE):
                match = pattern.match(candidate.name)
                if match:
                    encodings.add(match.group("encoding"))
                    break
        for encoding in sorted(encodings):
            yield variant_dir.name, encoding


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = True) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return True


@dataclass(frozen=True)
class ProposalExperiment:
    name: str
    model_name: str
    constraint_representation: str
    pressure_enabled: bool
    pressure_type_conditioning: str
    chooser_enabled: bool = False
    chooser_loss_mode: str = "fix1"
    chooser_loss_weight: float = 0.25
    chooser_beta_no_regression: float = 0.5
    chooser_gamma_primary: float = 0.0
    direct_safety_enabled: bool = False
    direct_safety_alpha_primary: float = 1.0
    direct_safety_beta_secondary: float = 0.5
    direct_safety_loss_weight: float = 1.0
    direct_safety_score_temperature: float = 1.0
    direct_safety_focus_deletion_weight: float = 0.0
    initialization_proposal_name: str | None = None
    learning_rate: float | None = None
    save_last_checkpoint: bool = False
    max_valid_edit_logit_abs: float | None = None
    validate_factor_labels: bool = False
    include_gold_candidates: bool = True
    enable_policy_choice: bool = False
    locked_backbone: bool = True
    dynamic_reweighting_enabled: bool = False
    pressure_module_sharing: str = "per_type"
    factor_executor_impl: str = "per_type_v1"
    factor_loss_enabled: bool | None = None
    compact_grouped: bool = False
    num_layers: int | None = None
    hidden_channels: int | None = None
    head_hidden: int | None = None
    dropout: float | None = None
    expected_trainable_parameters: int | None = None


@dataclass(frozen=True)
class RerankerExperiment:
    name: str
    objective: str
    proposal_name: str
    constraint_scope: str = "local"
    focus_deletion_weight: float = 0.0
    save_last_checkpoint: bool = False
    seed: int | None = None


def _proposal_config_payload(
    *,
    exp: ProposalExperiment,
    variant: str,
    encoding: str,
    min_occurrence: int,
    num_factor_types: int,
    active_factor_type_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    model_config = {
        "dataset_variant": variant,
        "encoding": encoding,
        "min_occurrence": min_occurrence,
        "model": exp.model_name,
        "constraint_representation": exp.constraint_representation,
        "factor_executor_impl": exp.factor_executor_impl,
        "use_edge_attributes": True,
        "use_edge_subtraction": False,
        "use_role_embeddings": True,
        "role_embedding_dim": 16,
        "pressure_enabled": exp.pressure_enabled,
        "pressure_type_conditioning": exp.pressure_type_conditioning,
        "pressure_module_sharing": exp.pressure_module_sharing,
        "pressure_residual_scale": LOCKED_PRESSURE_RESIDUAL_SCALE,
        "num_factor_types": int(num_factor_types),
        "enable_policy_choice": exp.enable_policy_choice,
        "policy_num_classes": 6,
    }
    if exp.locked_backbone:
        model_config.update(
            {
                "num_layers": LOCKED_NUM_LAYERS,
                "hidden_channels": LOCKED_HIDDEN_CHANNELS,
                "head_hidden": LOCKED_HEAD_HIDDEN,
                "dropout": LOCKED_DROPOUT,
            }
        )
    explicit_backbone = {
        "num_layers": exp.num_layers,
        "hidden_channels": exp.hidden_channels,
        "head_hidden": exp.head_hidden,
        "dropout": exp.dropout,
    }
    model_config.update({key: value for key, value in explicit_backbone.items() if value is not None})
    if exp.compact_grouped:
        if not active_factor_type_ids:
            raise ValueError(f"{exp.name} requires train/validation active factor ids.")
        model_config.update(
            {
                "factor_executor_impl": "per_type_grouped_v2",
                "gold_edit_embedding_mode": "compact",
                "pressure_module_sharing": exp.pressure_module_sharing,
                "active_factor_type_ids": list(active_factor_type_ids),
            }
        )
    payload = {
        "model_config": model_config,
        "training_config": {
            "batch_size": 256,
            "num_epochs": CHEAPER_NUM_EPOCHS,
            "early_stopping_rounds": CHEAPER_EARLY_STOPPING_ROUNDS,
            "learning_rate": (
                exp.learning_rate if exp.learning_rate is not None else CHEAPER_LEARNING_RATE
            ),
            "weight_decay": LOCKED_WEIGHT_DECAY,
            "grad_clip": CHEAPER_GRAD_CLIP,
            "scheduler_factor": CHEAPER_SCHEDULER_FACTOR,
            "scheduler_patience": CHEAPER_SCHEDULER_PATIENCE,
            "num_workers": SAFE_STREAMING_NUM_WORKERS,
            "pin_memory": SAFE_STREAMING_PIN_MEMORY,
            "validate_factor_labels": exp.validate_factor_labels,
            "validation_subset_size": VALIDATION_SUBSET_SIZE,
            "constraint_loss": {
                "dynamic_reweighting": {
                    "enabled": exp.dynamic_reweighting_enabled,
                }
            },
            "fix_probability_loss": {
                "enabled": False,
            },
            "factor_loss": {
                "enabled": (
                    exp.factor_loss_enabled
                    if exp.factor_loss_enabled is not None
                    else exp.constraint_representation == "factorized"
                ),
                "weight_pre": LOCKED_FACTOR_LOSS_WEIGHT_PRE,
                "weight_post_gold": LOCKED_FACTOR_LOSS_WEIGHT_POST_GOLD,
            },
            "chooser": {
                "enabled": exp.chooser_enabled,
                "loss_mode": exp.chooser_loss_mode,
                "loss_weight": exp.chooser_loss_weight,
                "beta_no_regression": exp.chooser_beta_no_regression,
                "gamma_primary": exp.chooser_gamma_primary,
                "topk_candidates": LOCKED_CHOOSER_TOPK_CANDIDATES,
                "max_candidates_total": LOCKED_CHOOSER_MAX_CANDIDATES_TOTAL,
            },
            "direct_safety": {
                "enabled": exp.direct_safety_enabled,
                "alpha_primary": exp.direct_safety_alpha_primary,
                "beta_secondary": exp.direct_safety_beta_secondary,
                "topk_candidates": LOCKED_CHOOSER_TOPK_CANDIDATES,
                "max_candidates_total": LOCKED_CHOOSER_MAX_CANDIDATES_TOTAL,
            },
            "policy_filter_strict": True,
            "seed": 42,
        },
    }
    if exp.direct_safety_loss_weight != 1.0:
        payload["training_config"]["direct_safety"]["loss_weight"] = exp.direct_safety_loss_weight
    if exp.direct_safety_score_temperature != 1.0:
        payload["training_config"]["direct_safety"]["score_temperature"] = (
            exp.direct_safety_score_temperature
        )
    if exp.direct_safety_focus_deletion_weight != 0.0:
        payload["training_config"]["direct_safety"]["focus_deletion_weight"] = (
            exp.direct_safety_focus_deletion_weight
        )
    if exp.initialization_proposal_name is not None:
        initialization_tag = f"{exp.initialization_proposal_name}__{variant}__{encoding}"
        payload["training_config"]["initialization_checkpoint"] = (
            f"../{initialization_tag}/checkpoint.pth"
        )
    if exp.save_last_checkpoint:
        payload["training_config"]["save_last_checkpoint"] = True
    if exp.max_valid_edit_logit_abs is not None:
        payload["training_config"]["max_valid_edit_logit_abs"] = exp.max_valid_edit_logit_abs
    if exp.expected_trainable_parameters is not None:
        payload["expected_trainable_parameters"] = int(exp.expected_trainable_parameters)
    return payload


def _reranker_config_payload(
    *,
    exp: RerankerExperiment,
    variant: str,
    encoding: str,
    min_occurrence: int,
    num_factor_types: int,
    proposal_config_tag: str,
) -> dict[str, Any]:
    payload = {
        "model_config": {
            "dataset_variant": variant,
            "encoding": encoding,
            "model": "RERANKER",
            "min_occurrence": min_occurrence,
            "constraint_representation": "factorized",
            "num_factor_types": int(num_factor_types),
        },
        "reranker_config": {},
        "training_config": {
            "batch_size": 64,
            "num_epochs": CHEAPER_NUM_EPOCHS,
            "early_stopping_rounds": CHEAPER_EARLY_STOPPING_ROUNDS,
            "learning_rate": CHEAPER_LEARNING_RATE,
            "weight_decay": 1e-4,
            "grad_clip": CHEAPER_GRAD_CLIP,
            "scheduler_factor": CHEAPER_SCHEDULER_FACTOR,
            "scheduler_patience": CHEAPER_SCHEDULER_PATIENCE,
            "validation_subset_size": VALIDATION_SUBSET_SIZE,
            "objective": exp.objective,
            "regression_weight": 0.5,
            "topk_candidates": 20,
            "topk_per_slot": 5,
            "heuristic_max_candidates": 30,
            "heuristic_max_values": 3,
            "include_gold": True,
            "prediction_include_gold": False,
            "max_candidates_total": 80,
            "assume_complete_entity_facts": True,
            "constraint_scope": exp.constraint_scope,
        },
        "proposal_config": {
            "model": "GIN_PRESSURE",
            "config_tag": proposal_config_tag,
        },
    }
    if exp.seed is not None:
        payload["training_config"]["seed"] = exp.seed
    if exp.focus_deletion_weight != 0.0:
        payload["training_config"]["focus_deletion_weight"] = exp.focus_deletion_weight
    if exp.save_last_checkpoint:
        payload["training_config"]["save_last_checkpoint"] = True
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--variant", default=None, help="Optional exact dataset-variant filter.")
    parser.add_argument("--encoding", default=None, help="Optional exact encoding filter.")
    parser.add_argument("--limit", type=int, default=0, help="Limit variant/encoding pairs (0 = no limit).")
    parser.add_argument("--include-experimental", action="store_true")
    parser.add_argument(
        "--study",
        choices=("canonical", "deletion-shortcut-v2"),
        default="canonical",
        help="Emit the canonical suite or only the isolated deletion-shortcut study configs.",
    )
    parser.add_argument(
        "--include-h2-ablations",
        action="store_true",
        help="Emit the three H2 supporting ablation configs. Existing config files are left untouched.",
    )
    args = parser.parse_args()
    if args.study != "canonical" and (args.include_experimental or args.include_h2_ablations):
        parser.error("--study deletion-shortcut-v2 cannot be combined with other config bundles")
    if args.include_experimental:
        assert_parameter_match()

    pairs = list(_iter_variant_encodings(args.processed_root))
    if args.variant is not None:
        pairs = [pair for pair in pairs if pair[0] == args.variant]
    if args.encoding is not None:
        pairs = [pair for pair in pairs if pair[1] == args.encoding]
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(
            "No graph artifacts found under "
            f"{args.processed_root}.\n"
            "Restore the paper graph artifacts, or build graphs for a new labeled dataset "
            "as described in docs-technical/00_training_and_evaluation_execution_plan.md. "
            "Do not relabel the released paper benchmark: its training labels are part of "
            "the recorded experimental provenance."
        )

    canonical_proposals: list[ProposalExperiment] = [
        ProposalExperiment(
            name="b0_eswc_reproduction",
            model_name="GIN",
            constraint_representation="eswc_passive",
            pressure_enabled=False,
            pressure_type_conditioning="none",
            validate_factor_labels=False,
            locked_backbone=False,
            num_layers=2,
            hidden_channels=128,
            head_hidden=128,
            dropout=0.5,
            expected_trainable_parameters=ORIGINAL_B0_TRAINABLE_PARAMETERS,
        ),
        ProposalExperiment(
            name="a1_factorized_imitation_compact_grouped",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            validate_factor_labels=True,
            compact_grouped=True,
            expected_trainable_parameters=COMPACT_A1_TRAINABLE_PARAMETERS,
        ),
        ProposalExperiment(
            name="m1c_safe_factor_chooser_compact_grouped",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            chooser_enabled=True,
            chooser_loss_mode="fix1",
            chooser_loss_weight=0.25,
            chooser_beta_no_regression=0.25,
            chooser_gamma_primary=0.2,
            validate_factor_labels=True,
            compact_grouped=True,
        ),
        ProposalExperiment(
            name="m1d_safe_factor_direct_compact_grouped",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            direct_safety_enabled=True,
            direct_safety_alpha_primary=1.0,
            direct_safety_beta_secondary=0.5,
            validate_factor_labels=True,
            compact_grouped=True,
        ),
    ]
    canonical_rerankers: list[RerankerExperiment] = [
        RerankerExperiment(
            name="g0_globalfix_reference_v2",
            objective="global_fix",
            proposal_name="a1_factorized_imitation_compact_grouped",
            constraint_scope="local",
            seed=42,
            save_last_checkpoint=True,
        )
    ]
    deletion_study_proposals: list[ProposalExperiment] = [
        ProposalExperiment(
            name="m1d_safe_factor_direct_v2",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            direct_safety_enabled=True,
            direct_safety_alpha_primary=1.0,
            direct_safety_beta_secondary=0.5,
            direct_safety_loss_weight=0.25,
            direct_safety_score_temperature=6.0,
            direct_safety_focus_deletion_weight=0.0,
            initialization_proposal_name="a1_factorized_imitation_compact_grouped",
            learning_rate=1e-5,
            save_last_checkpoint=True,
            max_valid_edit_logit_abs=10_000.0,
            validate_factor_labels=True,
            compact_grouped=True,
        ),
        ProposalExperiment(
            name="m1d_safe_factor_direct_base_preserving_v2",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            direct_safety_enabled=True,
            direct_safety_alpha_primary=1.0,
            direct_safety_beta_secondary=0.5,
            direct_safety_loss_weight=0.25,
            direct_safety_score_temperature=6.0,
            direct_safety_focus_deletion_weight=1.0,
            initialization_proposal_name="a1_factorized_imitation_compact_grouped",
            learning_rate=1e-5,
            save_last_checkpoint=True,
            max_valid_edit_logit_abs=10_000.0,
            validate_factor_labels=True,
            compact_grouped=True,
        ),
    ]
    deletion_study_rerankers: list[RerankerExperiment] = [
        RerankerExperiment(
            name="g0_globalfix_reference_v2",
            objective="global_fix",
            proposal_name="a1_factorized_imitation_compact_grouped",
            constraint_scope="local",
            seed=42,
            save_last_checkpoint=True,
        ),
        RerankerExperiment(
            name="g0_globalfix_base_preserving_v2",
            objective="global_fix",
            proposal_name="a1_factorized_imitation_compact_grouped",
            constraint_scope="local",
            focus_deletion_weight=1.0,
            seed=42,
            save_last_checkpoint=True,
        ),
    ]
    h2_ablation_proposals: list[ProposalExperiment] = [
        ProposalExperiment(
            name="h2_a1_no_factor_loss",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            validate_factor_labels=True,
            factor_loss_enabled=False,
            compact_grouped=True,
        ),
        ProposalExperiment(
            name="h2_a1_shared_pressure",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            pressure_module_sharing="shared",
            validate_factor_labels=True,
            compact_grouped=True,
        ),
        ProposalExperiment(
            name="h2_a1_legacy_shared_executor",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            factor_executor_impl="legacy_shared",
            validate_factor_labels=True,
        ),
    ]

    experimental_proposals: list[ProposalExperiment] = [
        ProposalExperiment(
            name="b0_parameter_matched",
            model_name="GIN",
            constraint_representation="eswc_passive",
            pressure_enabled=False,
            pressure_type_conditioning="none",
            validate_factor_labels=False,
            locked_backbone=False,
            num_layers=4,
            hidden_channels=304,
            head_hidden=304,
            dropout=0.17,
            expected_trainable_parameters=MATCHED_B0_TRAINABLE_PARAMETERS,
        ),
        ProposalExperiment(
            name="x1_policy_choice_appendix",
            model_name="GIN_PRESSURE",
            constraint_representation="factorized",
            pressure_enabled=True,
            pressure_type_conditioning="concat",
            validate_factor_labels=True,
            enable_policy_choice=True,
        ),
        ProposalExperiment(
            name="x2_factor_loss_only_appendix",
            model_name="GIN",
            constraint_representation="factorized",
            pressure_enabled=False,
            pressure_type_conditioning="none",
            validate_factor_labels=True,
        ),
    ]
    experimental_rerankers: list[RerankerExperiment] = [
        RerankerExperiment(
            name="x3_fix1_reranker_appendix",
            objective="main",
            proposal_name="a1_factorized_imitation",
            constraint_scope="local",
        )
    ]

    created = 0
    for variant, encoding in pairs:
        min_occurrence = _parse_min_occurrence(variant)
        factorized_path = args.processed_root / variant / graph_dataset_filename("train", encoding)
        passive_path = args.processed_root / variant / graph_dataset_filename(
            "train",
            encoding,
            constraint_representation="eswc_passive",
        )
        num_factor_types = _infer_num_factor_types_from_registry(variant)
        if num_factor_types <= 0:
            sample = _load_first_data_obj(factorized_path) or _load_first_data_obj(passive_path)
            num_factor_types = _infer_num_factor_types(sample)

        labeled_interim = args.interim_root / f"{variant}_labeled"
        if not labeled_interim.exists():
            labeled_interim = args.interim_root / variant
        active_factor_type_ids = dataset_factor_type_ids(
            labeled_interim,
            splits=("train", "val"),
        )
        test_factor_type_ids = dataset_factor_type_ids(
            labeled_interim,
            splits=("test",),
        )
        unseen_test_ids = sorted(set(test_factor_type_ids) - set(active_factor_type_ids))
        if unseen_test_ids:
            raise ValueError(
                f"{variant} test contains factor types absent from train/validation: "
                f"{unseen_test_ids}"
            )

        if args.study == "deletion-shortcut-v2":
            proposal_experiments = list(deletion_study_proposals)
            reranker_experiments = list(deletion_study_rerankers)
        else:
            proposal_experiments = list(canonical_proposals)
            reranker_experiments = list(canonical_rerankers)
            if args.include_h2_ablations:
                proposal_experiments.extend(h2_ablation_proposals)
            if args.include_experimental:
                proposal_experiments.extend(experimental_proposals)
                reranker_experiments.extend(experimental_rerankers)

        for exp in proposal_experiments:
            exp_dir_name = f"{exp.name}__{variant}__{encoding}"
            cfg_path = args.models_root / exp_dir_name / "config.json"
            payload = _proposal_config_payload(
                exp=exp,
                variant=variant,
                encoding=encoding,
                min_occurrence=min_occurrence,
                num_factor_types=num_factor_types,
                active_factor_type_ids=active_factor_type_ids,
            )
            if exp.name == "x2_factor_loss_only_appendix":
                payload["training_config"]["factor_loss"]["enabled"] = True
            overwrite = args.study == "canonical" and not args.include_h2_ablations
            if _write_json(cfg_path, payload, overwrite=overwrite):
                created += 1

        for exp in reranker_experiments:
            exp_dir_name = f"{exp.name}__{variant}__{encoding}"
            proposal_config_tag = f"{exp.proposal_name}__{variant}__{encoding}"
            cfg_path = args.models_root / exp_dir_name / "config.json"
            payload = _reranker_config_payload(
                exp=exp,
                variant=variant,
                encoding=encoding,
                min_occurrence=min_occurrence,
                num_factor_types=num_factor_types,
                proposal_config_tag=proposal_config_tag,
            )
            overwrite = args.study == "canonical" and not args.include_h2_ablations
            if _write_json(cfg_path, payload, overwrite=overwrite):
                created += 1

    print(f"[ok] wrote {created} configs under {args.models_root}")


if __name__ == "__main__":
    main()
