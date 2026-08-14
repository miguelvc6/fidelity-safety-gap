# Training and Evaluation Execution Plan

Date: 2026-03-11

This document defines the recommended execution order for training and evaluating the baselines and learned models in [docs-technical/00_models_and_evaluation_matrix.md](/home/mvazquez/constraint_factors/docs-technical/00_models_and_evaluation_matrix.md).

The plan is optimized for:

- paper-facing reproducibility
- short hyperparameter search
- minimal wasted long-running training
- one consistent dataset/encoding/backbone policy

Conceptual-to-technical mapping for this paper line:

- conceptual `M0 ESWC model` -> technical `B0`
- conceptual `M1 Main model` -> technical `A1`, `M1C`, and `M1D`
- conceptual `M2 Global Fix model` -> technical `G0`
- heuristic results -> `DFB`, `AMB`, `CFM`, `CDM`

`A1` is the representation-only step inside the broader conceptual `M1` story, while `M1C` and `M1D` are the two decision-level safe-factor realizations of that same model family.

## 1. Fixed run policy

Before running anything, freeze these decisions for the entire paper run:

- dataset variant: one paper dataset only, `full_strat1m_minocc100`
- `min_occurrence`: one value only, typically `100`
- encoding: one paper encoding only
- proposal random seed: `42`
- reranker random seed: `42`
- constraint neighborhood for the paper line: `local`

Reproducibility notes:

- `src/07_train.py` currently hardcodes `set_seed(42)`, so all proposal runs are already locked to seed `42`.
- `src/08_train_reranker.py` accepts `--seed`; use `--seed 42` for all reranker runs.
- heuristic baselines are deterministic.
- do not mix encodings or dataset variants inside the same paper table.
- keep `training_config.validation_subset_size: 25000` for the paper proposal runs. This is the frozen validation policy for the current paper line, not a development-only shortcut.
- do not edit configs once training starts; generate configs once, then only make one explicit “locked” hyperparameter update after the short search.

Recommended run ledger:

- record `git rev-parse HEAD`
- keep the generated config JSONs under `models/`
- keep scheduler logs under `logs/`

Strict-metrics precondition:

- proposal and reranker evaluations in `--paper-suite` mode require graph artifacts that already contain factor-label fields
- in the current pipeline, that means running `05_constraint_labeler.py` before `06_graph.py`; once `data/interim/<variant>_labeled/` exists, `06_graph.py` will use it automatically unless `--use-unlabeled-interim` is passed

## 2. Overall execution order

Run in this order:

1. Prepare paper artifacts.
2. Generate the canonical paper configs.
3. Run heuristic baselines first.
4. Run a brief `M1C` hyperparameter search only.
5. Select one winning `M1C` configuration.
6. Propagate the winning factorized-model settings to `A1`, `M1C`, and `M1D`, and reuse only the compatible training schedule for `B0`.
7. Train and evaluate the canonical learned suite in order:
   - `B0`
   - `A1`
   - final `M1C`
   - `M1D`
   - `G0`
8. Freeze tables and figures from those final runs only.

This order avoids spending time on secondary model families before the main model has a stable configuration.

Result coverage relative to the conceptual docs:

- heuristic references: `DFB`, `AMB`, `CFM`, `CDM`
- prior passive-context baseline: `B0`
- representation effect inside the main executable-factor story: `A1`
- main safe-factor results: `M1C`, `M1D`
- global-fix upper-bound reference: `G0`

The policy-choice model is intentionally out of scope for this paper line and is therefore not scheduled here.
`G0` is the repository's global-fix reference implementation for conceptual `M2`.

## 3. Step-by-step plan

### Step 0. Prepare paper artifacts

The paper line must materialize the artifact stack before config generation:

1. optional text cache for `text_embedding`
2. stratified benchmark parquet (`full_strat1m_minocc100`)
3. labeled interim parquet with `constraint-scope=local`
4. factorized processed graphs
5. passive processed graphs

**Build the stratified paper benchmark**

Run this after `02_dataframe_builder.py` has produced `data/interim/full_minocc100/`:

```bash
uv run src/02b_stratified_benchmark_sampler.py \
  --source-dataset full \
  --output-dataset full_strat1m \
  --min-occurrence 100 \
  --sample-fraction 0.5 \
  --seed 42 \
  --scope local
```

**Optional: build the text cache**

Only needed when the paper encoding is `text_embedding`.

```bash
uv run src/04_wikidata_retriever.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --registry-dataset full
```

**Build labeled interim parquet for the paper scope**

```bash
uv run src/05_constraint_labeler.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --constraint-scope local \
  --registry-dataset full \
  --factor-family-policy supported_only
```

**Build factorized executable-factor graphs**

For `node_id`:

```bash
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-scope local \
  --constraint-representation factorized \
  --registry-dataset full
```

If monolithic graph writes run out of memory, add:

```bash
  --shard-size 200000 \
  --use-torch-save
```

For `text_embedding`:

```bash
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding text_embedding \
  --constraint-scope local \
  --constraint-representation factorized \
  --registry-dataset full
```

The training, reranker, config-generation, and evaluation paths ingest these shard artifacts transparently.

**Build passive ESWC-style graphs**

For `node_id`:

```bash
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-representation eswc_passive \
  --registry-dataset full
```

The same optional shard flags may be used here if needed:

```bash
  --shard-size 200000 \
  --use-torch-save
```

For `text_embedding`:

```bash
uv run src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding text_embedding \
  --constraint-representation eswc_passive \
  --registry-dataset full
```

Paper readiness check:

- stratified parquet exists under `data/interim/full_strat1m_minocc100/`
- factorized train/test artifacts exist under `data/processed/full_strat1m_minocc100/`
- passive train/test artifacts exist under `data/processed/full_strat1m_minocc100/`
- labeled parquet exists under `data/interim/full_strat1m_minocc100_labeled/`
- coverage reports exist:
  - `coverage_local.csv`
  - `coverage_local.md`
- baseline evaluation emits the corrected schema-v2 paper metrics with explicit
  numerator and denominator support

### Step 1. Generate canonical configs

Use the canonical paper config generator:

```bash
uv run scripts/make_experiment_configs.py --models-root models
```

This emits only:

- `b0_eswc_reproduction`
- `a1_factorized_imitation`
- `m1c_safe_factor_chooser`
- `m1d_safe_factor_direct`
- `g0_globalfix_reference`

If appendix runs are needed later, generate them separately with `--include-experimental`.

H2 supporting ablations are generated with a separate opt-in flag:

```bash
uv run scripts/make_experiment_configs.py \
  --models-root models \
  --include-h2-ablations
```

When run after canonical config generation, existing config files are left untouched and this adds only the three H2 appendix proposal configs:

- `h2_a1_no_factor_loss__<variant>__<encoding>`
- `h2_a1_shared_pressure__<variant>__<encoding>`
- `h2_a1_legacy_shared_executor__<variant>__<encoding>`

The flag does not modify existing checkpoints or existing config files. Generated H2 ablations should be trained later as separate runs.

### Step 2. Run heuristic baselines first

Run baselines before any neural training so the paper already has stable reference numbers:

```bash
uv run src/09_eval.py \
  --run-baselines \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --registry-dataset full \
  --per-constraint-csv
```

Outputs are written under:

- `models/baselines/full_strat1m/parquet/`

This gives the reference results for:

- `DFB`
- `AMB`
- `CFM`
- `CDM`

If `AMB` is not used in the final main table, keep it as appendix support.

Code note:

- `src/09_eval.py --run-baselines` now prefers `data/interim/<variant>_labeled/` when it exists and falls back to `data/interim/<variant>/` otherwise.
- This is the paper-safe path for recomputing PFR, pooled Local Satisfaction,
  ΔLocalSat, pooled SIR/SRR, disruption, and evidence-preservation metrics from
  the interim rows. Stored factor tensors are not evaluation truth.

### Step 3. Run a brief `M1C` hyperparameter search

Search only on the paper’s main practical model: `M1C`.

Reason:

- `M1C` is the main paper model.
- `A1` and `M1D` should inherit its backbone/optimizer settings.
- `B0` should reuse the same training schedule where possible, but not get its own expensive search.
- `G0` should not trigger a separate reranker search in phase 1.

Generate the short search set:

```bash
uv run scripts/make_hparam_search_configs_m1.py \
  --processed-root data/processed \
  --models-root models \
  --dataset-variant full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --num-configs 5 \
  --seed 42
```

Recommended search budget:

- default: `5` configs max
- one seed only
- no repeated sweeps
- keep `training_config.validation_subset_size: 25000` in the generated configs and in the final paper-facing proposal configs. This keeps search and final runs on the same validation policy and bounds validation cost consistently.
- generated `M1C` configs use the conservative stability schedule: `learning_rate=1e-4`, `grad_clip=0.5`, `num_epochs=10`, `early_stopping_rounds=2`, `scheduler_patience=0`, and `chooser.loss_weight=0.25`

Run the short search with the scheduler:

```bash
uv run src/10_scheduler.py \
  --only hp_m1c_ \
  --paper-suite \
  --keep-going
```

Why this is acceptable even if training is long:

- the search is capped at a very small number of configs
- early stopping is enabled in the generated configs
- evaluation is automatic and strict

### Step 4. Select one winning `M1C` config

Choose the winner using the evaluation JSONs from the search runs.

Primary selection criteria:

1. higher corrected `PFR`
2. lower pooled `SRR`
3. higher historical fidelity
4. lower disruption

Practical rule:

- use the documented weighted score in the `model_selection` block of each
  run's `evaluations/model.json` as a ranking aid
- do not accept a config that improves fidelity by noticeably worsening `SRR`

Search outputs to inspect:

- `models/hp_m1c_*/evaluations/model.json`

If you need the per-constraint breakdown, inspect the resolved run directory under `models/`; the scheduler does not copy `per_constraint.csv` back into the config directory.

### Step 5. Lock the paper hyperparameters

Once one `M1C` run wins, copy these settings into the canonical proposal configs:

Backbone and optimizer:

- `num_layers`
- `hidden_channels`
- `head_hidden`
- `dropout`
- `learning_rate`
- `weight_decay`
- `batch_size`
- `grad_clip`
- `scheduler_factor`
- `scheduler_patience`
- `early_stopping_rounds`

Factorized-model settings to lock alongside the backbone:

- `pressure_type_conditioning`
- `pressure_residual_scale`
- `factor_executor_impl`
- `factor_loss.weight_pre`
- `factor_loss.weight_post_gold`
- chooser settings selected by the sweep:
  - `chooser.topk_candidates`
  - `chooser.max_candidates_total`
  - `chooser.beta_no_regression`
  - `chooser.gamma_primary`
  - `chooser.loss_weight`

Apply the full locked factorized setting bundle to:

- `a1_factorized_imitation`
- `m1c_safe_factor_chooser`
- `m1d_safe_factor_direct`

For `b0_eswc_reproduction`, reuse only the compatible shared training schedule:

- `learning_rate`
- `weight_decay`
- `batch_size`
- `scheduler_factor`
- `scheduler_patience`
- `early_stopping_rounds`

Do not copy factorized-only settings such as pressure or factor-loss weights into `B0`, because `B0` uses `constraint_representation="eswc_passive"`.

Do not run a second search on:

- `A1`
- `M1D`
- `B0`
- `G0`

### Step 6. Train and evaluate the final learned suite

Run the canonical learned models after the hyperparameters are locked.

Recommended order:

1. `B0`
2. `A1`
3. final `M1C`
4. `M1D`
5. `G0`

Reason:

- `B0` establishes the prior-work baseline
- `A1` establishes the representation-only step
- `M1C` is the main practical model
- `M1D` is the direct-loss counterpart
- `G0` depends on the factorized proposal stack and is the most downstream model

Recommended execution via scheduler:

```bash
uv run src/10_scheduler.py \
  --paper-suite \
  --keep-going
```

`--paper-suite` does not enforce the exact `B0 -> A1 -> M1C -> M1D -> G0` order by itself; the scheduler runs proposal configs before rerankers and otherwise follows directory-name ordering. Use the manual `--only` commands below when order matters.

If you want to enforce the order manually, use substring filters:

```bash
uv run src/10_scheduler.py --only b0_eswc_reproduction --paper-suite
uv run src/10_scheduler.py --only a1_factorized_imitation --paper-suite
uv run src/10_scheduler.py --only m1c_safe_factor_chooser --paper-suite
uv run src/10_scheduler.py --only m1d_safe_factor_direct --paper-suite
uv run src/10_scheduler.py --only g0_globalfix_reference --paper-suite
```

The scheduler will:

- train the run
- leave `checkpoint.pth`, `training_history.json`, and evaluation artifacts in the run directory
- evaluate with strict global metrics
- for passive `B0` graphs, strict global metrics use the interim parquet/registry state and do not require factor-label tensors on the passive test graphs
- automatically add `--use-chooser` for chooser runs
- automatically evaluate reranker runs from `reranker_predictions.json`

## 4. Manual evaluation commands

Use these only if you do not use the scheduler.

### Proposal models

`B0`, `A1`, and `M1D`:

```bash
uv run src/09_eval.py \
  --run-directory models/<run_dir> \
  --strict-global-metrics \
  --per-constraint-csv
```

`M1C`:

```bash
uv run src/09_eval.py \
  --run-directory models/<run_dir> \
  --use-chooser \
  --strict-global-metrics \
  --per-constraint-csv
```

### H2 diagnostics for factorized proposal models

Run H2 as a read-only sidecar evaluation on an existing factorized checkpoint:

```bash
uv run src/09_eval.py \
  --run-directory models/<run_dir> \
  --strict-global-metrics \
  --h2-eval
```

For chooser checkpoints, pass the chooser flag so every H2 variant uses the
same candidate construction and chooser scoring as the main evaluation:

```bash
uv run src/09_eval.py \
  --run-directory models/<run_dir> \
  --use-chooser \
  --strict-global-metrics \
  --h2-eval
```

Outputs are written under:

- `models/<run_dir>/evaluations/h2/h2_report.json`
- `models/<run_dir>/evaluations/h2/factor_semantics.csv`
- `models/<run_dir>/evaluations/h2/transfer_slices.csv`
- `models/<run_dir>/evaluations/h2/density_slices.csv`
- `models/<run_dir>/evaluations/h2/density_factor_semantics.csv`
- `models/<run_dir>/evaluations/h2/counterfactual_masking.csv`
- `models/<run_dir>/evaluations/h2/counterfactual_deltas.csv`
- `models/<run_dir>/evaluations/h2/counterfactual_overall_deltas.csv`
- `models/<run_dir>/evaluations/h2/graph_density.csv`

The command does not write `models/<run_dir>/evaluations/model.json` and does not mutate train/test graph artifacts. It requires factorized processed graphs with factor-label fields and reuses the labeled train Parquet only to count train exposure buckets. H2 records `selection_mode` in its report: raw A1-style proposals use `slot_argmax`, M1C uses `chooser`, and direct-safety M1D uses `direct_safety`. A normal row backed by validated schema-v2 replay must exactly match the corrected main evaluation. Independently recomputed selector outputs may differ on tied CUDA scores; the readiness gate permits at most `0.001` drift in metric values and normalized event counts, which rejects selector bypass while tolerating tie-breaking noise.
When the run already has validated schema-v2 predictions, H2 replays them for
the normal prediction row while still forwarding the checkpoint to collect
factor semantics. Counterfactual pressure variants always recompute predictions
through the configured selector.

### Reranker model

Train with:

```bash
uv run src/08_train_reranker.py \
  --experiment-config models/g0_globalfix_reference__<variant>__<encoding>/config.json \
  --seed 42
```

Then evaluate with:

```bash
uv run src/09_eval.py \
  --run-directory models/<g0_run_dir> \
  --reranker-predictions models/<g0_run_dir>/reranker_predictions.json \
  --strict-global-metrics \
  --per-constraint-csv
```

## 5. Final reporting set

Only the following runs should feed the paper tables:

- heuristic baselines: `DFB`, `CFM`, `CDM`, optional `AMB`
- learned models:
  - `B0`
  - `A1`
  - final `M1C`
  - `M1D`
  - `G0`

The hyperparameter search runs must not appear in the final paper tables.

These runs are sufficient to support the conceptual paper claims:

- `B0` gives the passive-context ESWC-style comparison point
- `A1` tests whether executable-factor structure helps before safety-aware decision logic
- `M1C` and `M1D` are the main safe-factor results
- `G0` serves as the global-satisfaction reference / upper-bound style comparison
- `DFB`, `AMB`, `CFM`, and `CDM` provide heuristic anchors

## 6. Minimal reproducibility checklist

Before declaring the suite complete, verify:

- the same dataset variant, `min_occurrence`, and encoding were used everywhere
- all proposal runs used the fixed seed behavior in `src/07_train.py`
- all reranker runs used `--seed 42`
- each generated config directory contains:
  - `config.json`
  - `checkpoint.pth`
  - `training_history.json`
  - `evaluations/model.json`
  - `evaluations/per_constraint.csv`
- baselines were written under `models/baselines/full_strat1m/parquet/`
- the resolved runtime directories under `models/` contain `evaluations/model.json` and `evaluations/per_constraint.csv` for the final paper runs

## 7. Recommended stop rule

Stop after:

- one brief `M1C` sweep
- one locked final training pass for the canonical suite

Do not expand into:

- per-model searches
- multi-seed sweeps
- separate reranker searches
- broad appendix-model training beyond the planned H2 ablations

unless the final paper suite fails in a way that blocks the main claims.

## 8. H2 ablation appendix runs

The H2 ablations are generated from the current processed factorized graph artifacts by adding `--include-h2-ablations` to the canonical config generator:

```bash
uv run scripts/make_experiment_configs.py \
  --models-root models \
  --include-h2-ablations
```

For the current paper artifact stack, this emits:

- `models/h2_a1_no_factor_loss__full_strat1m_minocc100__node_id/config.json`
- `models/h2_a1_shared_pressure__full_strat1m_minocc100__node_id/config.json`
- `models/h2_a1_legacy_shared_executor__full_strat1m_minocc100__node_id/config.json`

These are supporting ablations only. They use the existing processed graphs, the locked A1-style proposal setup, no chooser, and no direct-safety objective:

- `h2_a1_no_factor_loss`: tests whether the auxiliary factor satisfaction loss contributes to H2 factor semantics.
- `h2_a1_shared_pressure`: keeps pressure enabled but shares role pressure modules across factor types, testing typed pressure specifically.
- `h2_a1_legacy_shared_executor`: uses the legacy shared factor executor, testing the newer per-type executor path.

Train and evaluate the three runs with:

```bash
uv run src/10_scheduler.py \
  --only h2_a1_ \
  --paper-suite \
  --keep-going
```

After the standard evaluation has completed, run the H2 sidecar report for each ablation:

```bash
uv run src/09_eval.py \
  --run-directory models/h2_a1_no_factor_loss__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --h2-eval

uv run src/09_eval.py \
  --run-directory models/h2_a1_shared_pressure__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --h2-eval

uv run src/09_eval.py \
  --run-directory models/h2_a1_legacy_shared_executor__full_strat1m_minocc100__node_id \
  --strict-global-metrics \
  --h2-eval
```

The H2 outputs are written under each run's `evaluations/h2/` directory. These ablations should be reported as appendix diagnostics for H2, not as replacements for the canonical `B0`, `A1`, `M1C`, `M1D`, or `G0` results.

## 9. Candidate-oracle analysis

Candidate-oracle analysis is the next non-training diagnostic after the H2 ablations. It answers whether the candidate set already contains safe repairs that the model fails to select, or whether the candidate generator rarely proposes safe repairs in the first place.
It uses the evaluation-safe model forward, so compact checkpoints do not try to
embed test-only gold edit IDs when producing proposal logits.

This should be implemented as a read-only analysis script that reuses the existing candidate and symbolic-evaluation code paths:

- candidate generation: `modules.candidates.CandidateConfig` and `modules.candidates.build_candidates`
- symbolic candidate scoring: `modules.reranker_eval.CandidateConstraintEvaluator`
- model logits/proposals: the trained `A1`, `M1C`, and `M1D` checkpoints

Recommended script interface:

The repository provides this diagnostic as `scripts/analyze_candidate_oracle.py`.

```bash
uv run python scripts/analyze_candidate_oracle.py \
  --run-directory models/m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id \
  --strict-global-metrics --batch-size 256
```

Run the same analysis for the canonical compact `A1`, `M1C`, and `M1D`
checkpoints. Historical non-compact and hyperparameter-search oracle outputs are
not inputs to the corrected main table.

For each test instance, the script should:

1. Build the candidate set using the run's configured candidate policy.
2. Evaluate every candidate against the local symbolic constraint state.
3. Mark whether at least one candidate:
   - has a corrected PFR event;
   - introduces no common-support secondary regression;
   - has non-negative common-support Local Satisfaction change;
   - stays within the accepted disruption budget; and
   - for the evidence-preserving variant, retains the base statement.
4. Compare the oracle candidate against the model-selected candidate and, where relevant, the `G0` reranker-selected candidate.

Required outputs:

- `oracle_summary.json`: aggregate oracle rates, selected-candidate rates, and gaps.
- `oracle_by_density.csv`: oracle and selected-candidate rates by factor-density bucket.
- `oracle_by_constraint_type.csv`: oracle and selected-candidate rates by primary constraint family.
- `oracle_examples.csv`: representative rows where a safe oracle candidate exists but the model selected an unsafe candidate.

Decision rule:

- if the oracle is strong but `M1C`/`M1D` are weak, the bottleneck is selection learning or objective design;
- if the oracle is weak, the bottleneck is candidate generation;
- if oracle strength is concentrated in dense contexts or specific families, scope any safety claim to those regimes.
