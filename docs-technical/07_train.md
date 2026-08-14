# 07_train.py

## Objective
- Train a graph neural network (configured via JSON) on the graphs exported by `06_graph.py`, optimising the six action slots (`add/del × subject/predicate/object`) with per-slot cross-entropy while tracking per-constraint performance.
- Persist the best checkpoint, the resolved experiment configuration, and the full training history under the deterministic run slug `models/<variant>-<encoding>_<MODEL>_<config_tag>/`.

## Inputs & Outputs
- **Inputs:** Processed graph files in `data/processed/<variant>/` (either `train_graph-<encoding>.pkl` / `val_graph-<encoding>.pkl` or the `*_repr-eswc_passive-*` equivalents, plus shards when present), experiment config JSONs (model + training blocks), and the frozen encoder from `data/interim/<variant>/globalintencoder.txt`.
- **Outputs:** Run directory under `models/<variant>-<encoding>_<MODEL>_<config_tag>/` containing `checkpoint.pth`, `config.json`, `training_history.json`, plots, and evaluation artifacts when evaluated later.

## Workflow
1. **Configuration intake** – The script requires `--experiment-config path/to/config.json`. The file contains `model_config` (dataset variant, encoding, architecture name, hyperparameters) and `training_config` (batch size, epochs, scheduler knobs, constraint weighting).
2. **Run directory setup** – `ensure_run_dir()` creates or reuses a deterministic slugged folder based on dataset variant, encoding, model name, and config tag, while `config_copy_path()` determines where the config snapshot will live beside the checkpoint.
3. **Data loading** – `dataset_variant_name()` selects the processed root (`data/processed/<variant>/`), `load_graph_dataset()` discovers either monolithic files or shard collections (`*-shardNNN.{pkl,pt}`), returning an in-memory list or a lazy `GraphStreamDataset`/sharded stream as appropriate. `infer_node_feature_spec()` inspects samples to decide whether node features are embeddings or categorical IDs (including optional role flags).
4. **Target vocabularies** – The model predicts six categorical slots. `load_precomputed_target_vocabs()` reuses cached entity/predicate class IDs when available, otherwise `derive_target_class_ids()` scans the loaded graphs. These IDs are passed into the model so entity and predicate heads can be expanded/masked into a shared `num_target_ids` space.
5. **Factor type setup** – For `constraint_representation="factorized"`, if `model_config.num_factor_types > 0`, training uses that stable-id address-space bound directly unless the constraint registry reports a larger `constraint_type_index` range, in which case the registry value wins and the resolved config is updated. If the config leaves the field at `0`, the trainer prefers the constraint registry count and only falls back to a dataset scan when no registry-derived count is available. Compact execution separately uses `active_factor_type_ids` to decide which stable ids receive parameters. For `constraint_representation="eswc_passive"`, the trainer clears `num_factor_types` in the resolved config because passive graphs may include passive constraint nodes but do not execute per-type factor heads.
6. **Encoder + model build** – The frozen `GlobalIntEncoder` from `data/interim/<variant>/globalintencoder.txt` defines `num_graph_nodes`. `build_model()` instantiates the chosen architecture (e.g., message-passing network with dual branches). For `GIN_PRESSURE`, `model_config.pressure_module_sharing` controls whether factor-to-local pressure modules are per factor type (`per_type`, default) or shared across types (`shared`). `factor_executor_impl="per_type_grouped_v2"` keeps independent executor, post-edit, and pressure parameters per active type while storing them in packed banks. CPU uses a segmented linear fallback; compatible CUDA/BF16 execution uses grouped matrix multiplication. Device selection is automatic (CUDA if available) with memory logging hooks for debugging.
7. **Training loop (`train()`):**
   - Wrap datasets in split-specific `DataLoader`s, shuffling the in-memory train split while leaving streaming datasets ordered. For streamed datasets the trainer disables `pin_memory`, reduces `prefetch_factor` to `1`, and keeps `persistent_workers=False` so train/validation worker pools do not overlap and exhaust shared memory at epoch boundaries.
   - Forward pass returns logits of shape `(batch, 6, num_target_ids)` where entity/predicate slots are masked to the per-split vocabularies. Each slot is compared against the gold IDs via `CrossEntropyLoss(reduction="none")`, producing a `(batch, 6)` loss matrix.
   - Per-graph loss is computed as the mean over the six slots (`graph_loss = loss_matrix.mean(dim=1)`), then optionally:
     - `FixProbabilityScheduler` adds a repair-aware penalty when violation contexts are available.
     - `DynamicConstraintWeighter` rescales each per-graph loss based on constraint types (`extract_constraint_types()` reads `data.constraint_type`).
   - Accuracy is tracked both per-slot (percentage of correctly predicted IDs) and as “all-6 correct” (all slots match simultaneously).
   - `ConstraintMetricsAccumulator` aggregates loss/accuracy per constraint type so reports can highlight which shapes dominate or lag.
   - If chooser training is enabled, candidate sets are built per graph and scored by the chooser head. Training uses an optimized path:
     - candidate scoring is done in a packed/batched call (`score_candidates_packed`) rather than one scorer call per graph,
     - `fix1`-style chooser losses use `evaluate_candidates_loss_terms()` to compute only the required terms (no full diagnostic payload),
     - top-k candidate extraction can be restricted to valid entity/predicate class IDs per slot.
   - `torch.optim.Adam` drives the updates, `ReduceLROnPlateau` reduces LR when validation loss stalls, gradient clipping is optional, and early stopping is triggered after `training_config.early_stopping_rounds` epochs without improvement.
   - The trainer records stability diagnostics every epoch: learning rate, unclipped gradient norm mean/max, parameter norm/max absolute parameter value, edit-logit max magnitude, factor-logit max magnitude, and chooser-score max magnitude.
8. **Validation** – Mirrors the training pass sans gradient steps, feeding results into the same metric accumulators for apples-to-apples comparisons. If `training_config.validation_subset_size` is set, each epoch validates on only the first N validation graphs. Streamed validation subsets force the validation loader to `num_workers=0` so the subset is one global prefix, not one prefix per worker.
9. **Artifacts** – Once training finishes (or early stopping fires), the best-performing weights are saved via `torch.save()` alongside the effective `model_cfg`/`training_cfg`. `history_path()` stores the scalar curves, and `plot_training_history()` renders PNG charts for quick inspection.

## Common Pitfalls / Gotchas
- The `model_config.dataset_variant` and `model_config.encoding` must match the graphs on disk; mismatches surface as missing-file errors or shape mismatches deep in PyG.
- When using `GraphStreamDataset`, `len(dataset)` is undefined, so progress bars may look odd—this is expected and doesn’t mean data is missing.
- Early stopping patience is enforced even if validation batches fail intermittently; run with a stable validation split and monitor logs before trusting the saved checkpoint.
- Non-finite weighted loss, validation loss, gradient norm, or parameter norm now raises a `FloatingPointError` with epoch/batch context rather than silently producing a corrupt checkpoint.
- Subset validation changes model selection because scheduler and early stopping use the subset loss. Use full validation for final paper-facing runs.
- Generated paper configs now default to `learning_rate=1e-4`, `grad_clip=0.5`, `num_epochs=10`, `early_stopping_rounds=2`, `scheduler_patience=0`, `num_workers=2`, and `pin_memory=false`. The data-loader defaults avoid shared-memory pressure; the optimization defaults reduce the late-epoch divergence observed in long M1C runs.
- If CUDA is available but `num_workers` is high, pin-memory can still amplify host-memory pressure on in-memory datasets; tune `pin_memory` in the config if throughput does not justify the footprint.
- Fix-probability loss requires in-memory datasets (lists) so the script can attach `context_index` and look up contexts; streamed datasets will disable that term automatically.
- Chooser training supports streamed datasets via per-graph `context_index` assignment; contexts/parquet sidecars must align with graph ordering/counts.
- CUDA batch prefetch (`TRAIN_CUDA_PREFETCH`) is available and enabled by default; on some hardware/data combinations it may not improve throughput, so treat it as a tunable runtime flag.
- H2 ablation configs are appendix runs. Train them only into their generated `h2_a1_*` run directories; they should not replace the current canonical or hyperparameter-search checkpoints.
- Compact execution validates every graph's stable factor ids against `active_factor_type_ids` and fails rather than silently routing an unknown type. Regenerate the compact config after regenerating labeled splits.

## Compact/grouped A1 experiment

After the `full_strat1m_minocc100` labeled parquet and graph artifacts exist, create the opt-in config:

```bash
uv run python scripts/make_compact_a1_config.py
```

The script reads the existing canonical A1 config, preserves its training block and all unrelated model fields, and writes:

```text
models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id/config.json
```

It derives active factor types from `df_train.parquet` and `df_val.parquet`, checks that `df_test.parquet` introduces none, and refuses to overwrite an existing config. It changes only:

- `factor_executor_impl` to `per_type_grouped_v2`;
- `active_factor_type_ids` to the discovered stable-id vocabulary;
- `gold_edit_embedding_mode` to `compact`; and
- `pressure_module_sharing` explicitly to `per_type`.

Train and evaluate through the existing entry points:

```bash
uv run python src/07_train.py \
  --experiment-config models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id/config.json

uv run python src/09_eval.py \
  --run-directory models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id \
  --batch-size 256
```

Standard evaluation uses the inference forward path, which omits the training-only post-gold factor branch. This allows test-only target IDs to be counted normally without attempting to look them up in the compact train/validation gold-edit table. `--batch-size` controls standard evaluation throughput; lower it if GPU memory is insufficient.

Do not reuse a config generated from different labeled splits: the compact mapping is persisted in the checkpoint and is part of the architecture. Existing `per_type_v1` configs and checkpoints remain on the legacy dense path.

## Profiling & Throughput Controls

The trainer exposes runtime environment switches for profiling and data movement overlap:

- `TRAIN_TIMING_PROFILE=1` enables per-phase timing logs in both train and validation loops.
- `TRAIN_TIMING_WARMUP_BATCHES=<int>` excludes early warmup batches from timing summaries.
- `TRAIN_TIMING_LOG_EVERY=<int>` controls timing window size/frequency.
- `TRAIN_CUDA_PREFETCH=0|1` disables/enables asynchronous batch prefetch to GPU using a side CUDA stream.

Timing logs break the batch into phases such as:
- `data`, `forward`, `chooser`, `factor`, `backward`, `optim`, `metrics`, and `total`.

This makes bottlenecks explicit (for example, chooser-heavy runs where `chooser` dominates `total`).

## Implementation Details
- The script intentionally supports streamed graphs (via `GraphStreamDataset`) so very large runs never exceed RAM even when the serialized graphs are sharded.
- Per-slot histories are nested under `history["per_slot"][slot_index]`, enabling later analysis of which action (e.g., `del_predicate`) converged slower.
- GPU monitoring hooks (`log_cuda_memory`) fire at strategic checkpoints (epoch boundaries, first batch) to simplify diagnosing OOMs or fragmentation.
- Model checkpoints store both the state dict and the resolved configuration, allowing `09_eval.py` to rebuild the architecture without guessing hyperparameters.
- If `training_config.validate_factor_labels` is enabled, training asserts that factor label tensors exist and align with `factor_constraint_ids` (useful for upcoming factor supervision).
- Models receive `model_config.constraint_representation` at construction time. Passive models skip factor-head and pressure execution even if the passive graph contains constraint/factor nodes; factorized models require stable `factor_types` whenever per-type factor execution is reached. The legacy path expects the dense registry address space; the grouped path maps the configured active stable ids to compact parameter-bank indices.

## Dynamic Weighting per constraint type

`DynamicConstraintWeighter` keeps per‑constraint weights so the trainer can emphasize underperforming constraint types. Its behaviour can be specified from the configs json files: you can toggle it on/off, choose update_frequency (epoch uses validation metrics, batch reacts after every batch), decide which metrics drive “difficulty” (target_metrics defaults to loss but can include accuracies).

The weights are updated every batch or every epoch (can choose from model's configuration).

- Per batch: averages the current batch losses per constraint and treats them as “difficulty” scores.
- Per epoch: after validation it collects per-constraint metrics (loss/acc), converts the configured metrics into difficulty (loss directly, accuracies as 1 - acc/100), and updates weights once per epoch.

To calculate the weights from the difficulties it rescales difficulties relative to their mean, blends with prior weights using smoothing, clamps between min_weight/max_weight, and renormalizes so the mean weight stays ~1.

During training each batch multiplies the per-constraint loss rows by these weights before averaging/backpropagating; if the feature is disabled, it reduces to the standard uniform mean.
