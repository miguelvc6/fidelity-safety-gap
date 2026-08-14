# Model Config Reference

Date: 2026-08-03

This document lists the paper-relevant config fields accepted by `src/07_train.py` and `src/08_train_reranker.py`.

## Top-level keys

- `model_config`
- `training_config`
- `reranker_config` for reranker runs
- `proposal_config` for reranker runs

## `model_config`

Core fields:

- `dataset_variant`
- `encoding`
- `model`
- `min_occurrence`
- `num_layers`
- `hidden_channels`
- `head_hidden`
- `dropout`
- `use_edge_attributes`
- `use_edge_subtraction`
- `use_role_embeddings`
- `role_embedding_dim`
- `num_role_types`
- `entity_class_ids`
- `predicate_class_ids`
- `num_factor_types`
- `active_factor_type_ids`
- `factor_type_embedding_dim`
- `factor_executor_impl`
- `gold_edit_embedding_mode`
- `pressure_enabled`
- `pressure_type_conditioning`
- `pressure_module_sharing`
- `pressure_residual_scale`
- `enable_policy_choice`
- `policy_num_classes`

Paper-facing additions:

- `constraint_representation`
  - allowed values: `factorized`, `eswc_passive`
  - `B0` should use `eswc_passive`
  - `A1`, `M1C`, `M1D`, and proposal sources for `G0` should use `factorized`

- `pressure_module_sharing`
  - allowed values: `per_type`, `shared`
  - default: `per_type`
  - `per_type` preserves the current typed-pressure behavior
  - `shared` keeps factor pressure enabled but shares the role pressure modules across factor types; use this for the H2 untyped-pressure ablation only

- `factor_executor_impl`
  - allowed values: `per_type_v1`, `per_type_grouped_v2`, `legacy_shared`
  - default: `per_type_v1`, retaining the original module layout and checkpoint keys
  - `per_type_grouped_v2` packs the same independent per-type MLP weights into tensor banks and groups rows by compact type index; it requires `active_factor_type_ids`
  - `legacy_shared` is retained for the H2 shared-executor ablation and is not part of the compact A1 change

- `active_factor_type_ids`
  - sorted, unique stable factor-type ids that are reachable in the selected dataset
  - `num_factor_types` remains the upper bound of the stable registry address space; it is not replaced by the active count
  - the compact A1 generator derives this list from train and validation only, then verifies that test has no unseen types

- `gold_edit_embedding_mode`
  - allowed values: `full`, `compact`
  - default: `full`, retaining the original target-id-sized factor gold-edit table
  - `compact` stores rows only for the union of reachable entity/predicate target ids (plus id `0`) and uses a stable-to-compact lookup

## `training_config` for proposal runs

Core optimization fields:

- `seed` (canonical corrected-suite value: `42`)
- `batch_size`
- `num_epochs`
- `early_stopping_rounds`
- `grad_clip`
- `learning_rate`
- `weight_decay`
- `scheduler_factor`
- `scheduler_patience`
- `num_workers`
- `pin_memory`
- `validate_factor_labels`
- `validation_subset_size`

Generator defaults for the paper-facing proposal configs:

- `batch_size: 256`
- `num_epochs: 10`
- `early_stopping_rounds: 2`
- `grad_clip: 0.5`
- `learning_rate: 1e-4`
- `scheduler_factor: 0.5`
- `scheduler_patience: 0`
- `num_workers: 2`
- `pin_memory: false`

These defaults are intentionally conservative for the large streamed graph artifacts under `data/processed/`. The shorter schedule is meant to stop soon after the best validation checkpoint on runs that otherwise diverge after several good epochs.

The paper-facing reranker generator uses the same cheaper schedule (`num_epochs: 10`, `early_stopping_rounds: 2`, `learning_rate: 1e-4`, `grad_clip: 0.5`, `scheduler_patience: 0`) with its own reranker batch size.

Set `validation_subset_size` to a positive integer for development runs that should validate on only the first N validation graphs each epoch. Leave it unset or `null` for full validation. For streamed graph artifacts, subset validation uses a single validation worker so the stream produces one global prefix rather than one prefix per worker.

For `num_factor_types`, the paper-facing generators prefer the compact factor-type count derived from the constraint registry rather than inferring from a single graph sample.

Nested blocks:

- `constraint_loss.dynamic_reweighting`
- `fix_probability_loss`
- `factor_loss`
- `chooser`
- `direct_safety`

### `chooser`

- `enabled`
- `topk_candidates`
- `max_candidates_total`
- `beta_no_regression`
- `gamma_primary`
- `loss_weight`
- `loss_mode`

Paper use:

- `M1C`: enabled
- `A1`, `B0`, `M1D`: disabled

Generator defaults for chooser-enabled proposal runs are `loss_weight: 0.25`, `beta_no_regression <= 0.5`, and `gamma_primary: 0.0` unless a targeted stress-test config explicitly overrides `gamma_primary`.

### `direct_safety`

- `enabled`
- `alpha_primary`
- `beta_secondary`
- `topk_candidates`
- `max_candidates_total`

Paper use:

- `M1D`: enabled
- `A1`, `B0`, `M1C`: disabled

### `factor_loss`

This remains supported, but it is not part of the default paper-facing suite.

## `training_config` for reranker runs

Reranker configs use the schema in `src/08_train_reranker.py`.

Paper-relevant fields:

- `validation_subset_size`
- `objective`
  - `main`
  - `global_fix`
- `topk_candidates`
- `topk_per_slot`
- `max_candidates_total`
- `regression_weight`
- `constraint_scope`

Paper use:

- `G0`: `objective="global_fix"`

## Validation notes

- Config loading is strict: unknown keys raise an error.
- `pressure_type_conditioning` must be one of `none`, `concat`, `gate`.
- `pressure_module_sharing` must be one of `per_type`, `shared`.
- `factor_executor_impl` must be one of `per_type_v1`, `per_type_grouped_v2`, `legacy_shared`.
- `per_type_grouped_v2` requires a non-empty, sorted `active_factor_type_ids` list whose ids fit inside `num_factor_types`.
- `gold_edit_embedding_mode` must be one of `full`, `compact`.
- `constraint_representation` must be one of `factorized`, `eswc_passive`.
- `chooser` and `direct_safety` should not both be enabled in the same proposal config.
