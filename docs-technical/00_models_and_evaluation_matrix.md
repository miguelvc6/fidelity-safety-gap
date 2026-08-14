# Paper-Facing Models and Evaluation Matrix

Date: 2026-08-06

This document defines the canonical paper-facing suite implemented in the repository. The default experiment surface should describe only these models.

## Canonical learned suite

| ID | Name | Representation | Training path | Objective | Inference role |
| --- | --- | --- | --- | --- | --- |
| `B0-R` | `Original B0 ESWC-Reproduction` | `eswc_passive` | `src/07_train.py` | `L_edit` | slot argmax |
| `B0-M` | `Parameter-matched B0` | `eswc_passive` | `src/07_train.py` | `L_edit` | slot argmax |
| `A1` | `Compact A1 Factorized Imitation` | `factorized` | existing checkpoint | `L_edit` | slot argmax |
| `M1C` | `M1C Safe Factor Chooser` | `factorized` | `src/07_train.py` | `L_edit + L_chooser` | chooser over symbolic candidates |
| `M1D` | `M1D Safe Factor Direct` | `factorized` | `src/07_train.py` | `L_edit + alpha L_primary + beta L_secondary` | candidate argmax from proposal logits |
| `G0` | `G0 GlobalFix Reference` | `factorized` proposal + reranker | `src/08_train_reranker.py` | `L_global` | reranker over symbolic candidates |

## Definitions

### Original and parameter-matched B0
- Uses `constraint_representation="eswc_passive"`.
- Disables local executable factor expansion, local factor-scope wiring, typed pressure, chooser loss, and direct safety loss.
- Original B0 is the 128×2 replication baseline. Parameter-matched B0 is the
  304×4 capacity control and differs from Compact A1 by at most 0.1% trainable
  parameters. Both consume the same passive graph suite.

### A1 Factorized Imitation
- Uses `constraint_representation="factorized"`.
- Keeps the factorized graph, per-type factor executors, and per-role pressure, but trains only with edit imitation plus auxiliary factor supervision.
- Answers whether factorized local constraint context helps before any safety-aware decision objective.
- The compact/grouped 400×4 checkpoint is canonical. Original A1 is prior
  compression-equivalence evidence only and is not retrained.

### M1C Safe Factor Chooser
- Builds on `A1`.
- Uses chooser scoring over the same symbolic candidate set used by reranking.
- Retains the same per-type factor executor backbone and auxiliary factor supervision as `A1`.
- Default paper configs should keep a non-zero primary term (`gamma_primary > 0`) so primary-fix preference is explicit.

### M1D Safe Factor Direct
- Builds on `A1`.
- Reuses the same candidate builder and symbolic evaluator contract as `M1C` and `G0`.
- Retains the same per-type factor executor backbone and auxiliary factor supervision as `A1`.
- Computes candidate scores directly from proposal slot logits, then optimizes expected primary-failure and secondary-regression penalties.

### G0 GlobalFix Reference
- Uses the factorized proposal regime as candidate source.
- Trains a reranker for expected global satisfaction.
- It is a frontier reference, not the default practical system.

## Fixed paper defaults

The default paper-facing generator should enforce these settings for `A1`, `M1C`, and `M1D`:

- backbone: `GIN_PRESSURE`
- graph path: multi-relational / edge-attribute regime
- encoding: whichever dataset artifact is selected, typically frozen text embeddings when available
- role embeddings: enabled
- dynamic per-type reweighting: fixed by the locked winning `M1C` configuration
- fix-probability loss: disabled
- factor-loss-only training: disabled
- factor executor: `per_type_grouped_v2`
- gold edit embedding: `compact`
- pressure modules: `per_type`
- active factor IDs: train/validation union, with test-only IDs rejected

`B0` is the exception: it keeps the passive representation and disables typed pressure.

## Default experiment bundle

The default config generator should emit only:

- `b0_eswc_reproduction`
- `b0_parameter_matched`
- `a1_factorized_imitation_compact_grouped`
- `m1c_safe_factor_chooser_compact_grouped`
- `m1d_safe_factor_direct_compact_grouped`
- `g0_globalfix_reference`

Appendix or exploratory variants such as policy-choice, factor-loss-only, untyped-pressure, and non-paper rerankers should be gated behind `--include-experimental` or a more specific appendix flag.

## Main evaluation suites

### Main results
- `DFB`
- `CFM`
- `CDM`
- `B0`
- `A1`
- `M1C`
- `M1D`
- `G0`

### Minimal ablation view
- `B0 -> A1`: representation effect
- `A1 -> M1C`: chooser-based safe selection
- `A1 -> M1D`: direct-loss safe selection

### Required metrics
- historical fidelity: precision / recall / micro-F1
- PFR
- pooled Local Satisfaction and common-support ΔLocalSat
- pooled secondary regression and improvement rates (`SRR`, `SIR`)
- disruption
- Base-deletion Rate and Deletes-base-action Rate
- EPPF and Vacuous Improvement

Each paper metric contains `value`, `numerator`, and `denominator`. Stored graph
pre-label tensors are never paper-metric truth; pre/post states are recomputed
from interim rows.

### Evaluation rule
- `M1C`, `M1D`, and `G0` must share the same candidate-level symbolic evaluator contract so reported safety metrics are definitionally aligned.

## H2 appendix diagnostics

H2 is evaluated as an opt-in diagnostic layer over existing factorized train/test graph artifacts and trained checkpoints. It is not part of the canonical main-suite score and writes under `models/<run>/evaluations/h2/` so the normal `evaluations/model.json` remains unchanged.

The H2 report includes:

- factor semantic metrics for `factor_logits_pre` and `factor_logits_post_gold` against existing factor satisfaction labels, grouped by factor state, family, and compact type
- transfer slices by train-set factor exposure bucket (`unseen`, `low_1_10`, `medium_11_100`, `high_gt100`), primary vs secondary role, and family
- density/composition slices by local factor count bucket (`1`, `2_4`, `5_16`, `17_64`, `65_plus`) plus shared pressure-overlap summaries
- inference-time pressure masking variants: `normal`, `no_factor_pressure`, `primary_only_pressure`, and `secondary_only_pressure`
- counterfactual prediction-change rates and repair/global metric deltas relative to `normal`

Factor semantic rows report support, positive rate, accuracy, precision, recall, F1, AUROC, AUPRC, and ECE. If a model checkpoint does not emit factor logits, the H2 report is marked partial and records the unsupported section instead of failing the whole evaluation.

## H2 supporting ablations

The H2 ablations are appendix/supporting runs. They are not canonical main-suite models and should be trained only into new run directories:

- `h2_a1_no_factor_loss__<variant>__<encoding>`: A1-style factorized graph and pressure, but disables auxiliary factor satisfaction loss.
- `h2_a1_shared_pressure__<variant>__<encoding>`: keeps factor pressure enabled but shares role pressure modules across factor types through `pressure_module_sharing="shared"`.
- `h2_a1_legacy_shared_executor__<variant>__<encoding>`: uses the older shared factor executor path through `factor_executor_impl="legacy_shared"`.

All three use current processed factorized graphs, A1-style slot inference, no chooser, and no direct safety objective.
