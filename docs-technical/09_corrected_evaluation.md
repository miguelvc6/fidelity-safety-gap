# Corrected Schema-v2 Evaluation

This document describes the implementation and operations for the corrected
paper suite. The research rationale is in
[the corrected evaluation protocol](../docs-conceptual/08_corrected_evaluation_protocol.md).

## State and label policy

`src/modules/evidence_state.py` is shared by `src/05_constraint_labeler.py` and
`src/modules/reranker_eval.py`. It merges aliased role evidence, inserts the base
statement before checking constraints, and applies delete then add.

Do not run `src/05_constraint_labeler.py` against the current
`full_strat1m_minocc100` experiment artifacts. Existing labeled Parquet and
factorized graph shards remain the training source for Compact A1, M1C, and M1D.
Only evaluation labels are recomputed. Future datasets receive the corrected
semantics when the labeler is run normally.

The read-only drift audit is:

```bash
uv run python scripts/audit_label_semantics.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --registry-dataset full \
  --labeled-dir data/interim/full_strat1m_minocc100_labeled \
  --output models/paper_diagnostics/label_semantics_audit.json
```

The audit compares stored pre/post factor arrays with corrected recomputation by
split and constraint family. It does not modify Parquet, graph, checkpoint, or
configuration artifacts and does not invoke training.

## Output schema

Every standard evaluation writes the following under `<run>/evaluations/`:

- `model.json`, with `schema_version: 2`, fidelity metrics, `paper_metrics`,
  corrected model selection, and prediction-artifact references;
- `per_constraint.csv`, whose paper metric columns each have `_value`,
  `_numerator`, and `_denominator` forms;
- `predictions.parquet`, with ordered row identity, constraint metadata, six
  predicted slots, resolved add/delete operations, and per-instance metric
  numerator/denominator columns; and
- `predictions.manifest.json`, with row count, producer checkpoint/config
  identity, interim dataset identity, ordered graph identity, and SHA-256
  checksums.

`disruption` counts resolved add/delete operations, not merely non-`NONE`
components in a three-slot group. A partially populated add or delete group is
not executable, is serialized with no resolved operation, and contributes zero
for that action. Standard evaluation, H2, and candidate-oracle diagnostics all
use this definition through the shared per-instance event builder.

Writes use a temporary file in the destination directory followed by atomic
replacement. The first schema-v2 write preserves existing outputs once as
`model.pre-schema-v2.json` and `per_constraint.pre-schema-v2.csv`.

Replay uses `--predictions`; `--reranker-predictions` is an alias. Replay rejects
missing manifests, non-v2 schemas, Parquet checksum changes, row count/order
changes, dataset identity/checksum changes, and graph artifact order/checksum
changes.

Legacy G0 JSON is accepted only by the explicit one-time
`--legacy-predictions-json` migration path. The migration requires exact row
count, six valid integer slots per row, and encoder-range validity, then records
the legacy source checksum alongside the canonical dataset and graph identity.
Because the legacy format has no row identifiers, the migration assumes the
canonical test-row order in which the file was originally generated. All later
replay and diagnostics use the resulting schema-v2 Parquet artifact.

Candidate construction has two explicit modes. Training may set
`include_gold=true` so a supervised ranking loss always has its target and
receives the returned gold-candidate index. Evaluation, candidate-oracle
analysis, H2 selection, and test prediction set `include_gold=false`; in that
mode the builder does not read `graph.y` or `gold_slots` and returns no gold
index. A label edit may still appear if it is independently proposed by the
heuristic or top-k logits, but it is never injected from the target. G0 keeps
gold inclusion for its historical training objective and records
`prediction_include_gold=false` for future checkpoint-only test generation.

## Canonical configurations

Generate the bundle after the existing factorized graph suite and labeled
Parquet are present:

```bash
uv run python scripts/make_experiment_configs.py
```

The canonical learned runs are:

- Original B0: `b0_eswc_reproduction` (`128×2`, dropout `0.5`);
- Compact A1: `a1_factorized_imitation_compact_grouped`;
- compact M1C: `m1c_safe_factor_chooser_compact_grouped`; and
- compact M1D: `m1d_safe_factor_direct_compact_grouped`.

Compact factor runs use `factor_executor_impl="per_type_grouped_v2"`,
`gold_edit_embedding_mode="compact"`, and per-type pressure modules. Active
factor IDs are the train/validation union; any test-only factor type aborts
generation. M1C uses `gamma_primary=0.2`. Parameter-matched B0 remains
available as a generator capability but is
excluded from the current paper run by scope. G0 links to Compact A1 proposals.

The two passive B0 configurations intentionally reference the same passive
graph filenames. Generate that graph suite once if absent:

```bash
uv run python src/06_graph.py \
  --dataset full_strat1m \
  --min-occurrence 100 \
  --encoding node_id \
  --constraint-representation eswc_passive \
  --registry-dataset full \
  --shard-size 10000 \
  --use-torch-save \
  --persistence-profile research_safe \
  --overwrite atomic
```

The bounded shards avoid retaining the complete graph split in memory during
serialization, and atomic replacement prevents a partial shard from appearing
as a completed artifact. Do not regenerate the existing factorized graph
shards.

## Exact suite commands

Compact A1 evaluation reuses its existing checkpoint:

```bash
uv run python src/09_eval.py \
  --run-directory models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id \
  --strict-global-metrics --per-constraint-csv --batch-size 256
```

Train the new single-seed systems, then evaluate each with the same flags:

```bash
uv run python src/07_train.py --experiment-config models/b0_eswc_reproduction__full_strat1m_minocc100__node_id/config.json
uv run python src/07_train.py --experiment-config models/m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id/config.json
uv run python src/07_train.py --experiment-config models/m1d_safe_factor_direct_compact_grouped__full_strat1m_minocc100__node_id/config.json

uv run python src/09_eval.py --run-directory models/<run-directory> \
  --strict-global-metrics --per-constraint-csv --batch-size 256
```

Replay a schema-v2 proposal or reranker artifact with:

```bash
uv run python src/09_eval.py --run-directory models/<run-directory> \
  --predictions models/<producer>/evaluations/predictions.parquet \
  --strict-global-metrics --per-constraint-csv
```

The retained G0 directory does not contain its reranker checkpoint. Its legacy
predictions also fail the complete gold-excluded candidate-membership audit:
909 of 143,316 selected edits are absent after forced gold insertion is
disabled. Therefore the retained G0 prediction/evaluation files are excluded
from paper-ready results. The read-only audit command is:

```bash
uv run python scripts/audit_prediction_candidate_membership.py \
  --run-directory models/g0_globalfix_reference__full_strat1m_minocc100__node_id \
  --proposal-run-directory models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id \
  --predictions models/g0_globalfix_reference__full_strat1m_minocc100__node_id/reranker_predictions.json \
  --output models/g0_globalfix_reference__full_strat1m_minocc100__node_id/evaluations/candidate_membership_audit.json \
  --batch-size 256
```

To restore G0 to the suite, retrain it and let the same invocation generate
test predictions with `prediction_include_gold=false`; `--prediction-batch-size`
does not change training:

```bash
uv run python src/08_train_reranker.py \
  --experiment-config models/g0_globalfix_reference__full_strat1m_minocc100__node_id/config.json \
  --prediction-batch-size 256 --seed 42

uv run python src/09_eval.py \
  --run-directory models/g0_globalfix_reference__full_strat1m_minocc100__node_id \
  --legacy-predictions-json models/g0_globalfix_reference__full_strat1m_minocc100__node_id/reranker_predictions.json \
  --strict-global-metrics --per-constraint-csv --batch-size 256

uv run python scripts/analyze_deletion_degeneracy.py \
  --g0-run-directory models/g0_globalfix_reference__full_strat1m_minocc100__node_id \
  --predictions models/g0_globalfix_reference__full_strat1m_minocc100__node_id/evaluations/predictions.parquet \
  --strict-global-metrics
```

After a fresh G0 checkpoint and prediction artifact exist, the deletion
diagnostic validates the Parquet manifest and reports all ten corrected metrics
for both G0 and DFB. It rejects truncated or reordered input; it no longer
derives EPPF or Vacuous Improvement from legacy scalar fields. The retained G0
files do not satisfy this prerequisite and remain excluded.

For canonical compact factor diagnostics, run `--h2-eval` on each existing
checkpoint and run `scripts/analyze_candidate_oracle.py`. The candidate oracle
uses the same per-instance event builder as standard evaluation, so PFR/EPPF,
pooled Local Satisfaction and SIR/SRR, common-support delta, disruption, and
evidence-preservation denominators are identical to the main schema.
Its selected-candidate columns replay the validated schema-v2 paper predictions
exactly; the script rebuilds only the label-blind candidate set and oracle
choices. This prevents independent CUDA tie resolution from changing the
reported selected metrics.
It also uses the evaluation-safe compact forward and therefore never requires
test-only gold edit IDs to exist in the train/validation compact vocabulary.
H2 also reuses the main prediction selector: A1 uses slot argmax, M1C requires
`--use-chooser`, and M1D automatically applies its configured direct-safety
candidate selection. The report records this as `selection_mode`. Validated
normal replay must match `evaluations/model.json` exactly; independent selector
reruns may differ on tied CUDA scores, so the gate allows at most `0.001` drift
in metric values and normalized event counts.
H2 uses the validated `evaluations/predictions.parquet` for its normal
prediction row when available, while still running the checkpoint for factor
logits. Masked-pressure variants are always inferred and selected directly.

Run the deletion-focused and other baselines with:

```bash
uv run python src/09_eval.py --run-baselines --dataset full_strat1m \
  --min-occurrence 100 --strict-global-metrics --per-constraint-csv \
  --batch-size 256
```

Original A1 is not retrained and its old safety outputs are excluded from the
corrected main table. Its retained aggregate Micro-F1 is used only as prior
compact-path consistency evidence.

After the compact H2/oracle sidecars and the retained-G0 exclusion audit
complete, enforce the paper gate with:

```bash
uv run python scripts/check_corrected_paper_readiness.py \
  --paper latex_paper/main_compact.tex \
  --verify-graph-checksums
```

The gate covers the four learned paper systems and all four deterministic
baselines. It verifies the exact canonical run paths and architecture/training
settings, schema versions, complete numerator/denominator metrics, prediction,
dataset, configuration, and checkpoint checksums, common row identity, and
absence of legacy metric fields. It independently reaggregates every symbolic
metric event and strict operation-level fidelity count from
`predictions.parquet`, checks every `per_constraint.csv`, validates the H2 and
candidate-oracle selection modes, enforces DFB's expected deletion-degeneracy
rates, and checks the evidence supporting G0's exclusion. With `--paper`, it
also requires every row in the aggregate, symbolic, transition, and pressure
masking tables to equal the canonical artifacts at the displayed precision and
records the TeX checksum. `--verify-graph-checksums` hashes each unique graph
artifact once, including the shared factorized graph, rather than validating
only its recorded path and size. It writes
`models/paper_diagnostics/corrected_paper_readiness.json` only when every check
passes. A newly trained G0 checkpoint intentionally invalidates that exclusion
gate until the run is added back to the paper-ready system set.
