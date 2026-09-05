# Symbolic Evaluation and Prediction Artifacts

Validator version 3 uses the same constraint parser, evidence-state builder,
and three-valued evaluator in label generation, candidate training, reranking,
diagnostics, and evaluation. The full semantics and hierarchy construction are
specified in [the validator-v3 guide](12_validator_semantics_revision.md); the
evaluation-population rationale is in
[Validator-v3 evaluation and objective decisions](../docs-conceptual/11_validator_v3_evaluation_and_objective_decisions.md).

## State contract

`src/modules/evidence_state.py` merges aliased role projections, inserts the
focus and explicit auxiliary statements, and applies deletion before addition.
Deleting and re-adding the same statement preserves it. The primary definition
is bound to the focus subject/property, while secondary definitions are bound
to represented occurrences of their own constrained property.

The validator returns `satisfied`, `violated`, or `unknown`. Boolean
`factor_checkable_*` columns are only the storage projection of this result:
`unknown` is not checkable; the other two states are checkable.

The label audit must report zero drift after regeneration:

```bash
uv run python scripts/audit_label_semantics.py \
  --dataset full_strat1m --min-occurrence 100 --registry-dataset full \
  --labeled-dir data/interim/full_strat1m_minocc100_labeled \
  --output models/paper_diagnostics/label_semantics_audit.json
```

## Schema-v3 outputs

Every evaluation writes under `<run>/evaluations/`:

- `model.json`: operation-level fidelity, symbolic metrics, validator and
  semantic-contract versions, and hierarchy identity;
- `per_constraint.csv`: family metrics with the same provenance columns;
- `historical_strata.csv`: historical fix, non-fix, and uncheckable outcomes;
- `predictions.parquet`: ordered identity, six predicted slots, resolved
  operations, strata, and per-row metric events; and
- `predictions.manifest.json`: schema, producer, row count, dataset, graph,
  validator, hierarchy/parser/cache/effective-policy/objective contracts, and
  SHA-256 identities.

Writes are atomic. If an older artifact exists, it is preserved once with the
suffix `.pre-schema-v3`. Prediction replay recomputes metric events and rejects
a missing manifest, schema versions other than 3, row/order changes, checksum
changes, validator/semantic-contract mismatches, or hierarchy mismatches.

```bash
uv run python src/09_eval.py --run-directory models/<run-directory> \
  --predictions models/<run-directory>/evaluations/predictions.parquet \
  --strict-global-metrics --per-constraint-csv
```

`--legacy-predictions-json` is limited to the first conversion of
Candidate--SR output. Paper diagnostics and replay consume only the validated
Parquet artifact.

## Baselines and diagnostics

The deterministic/statistical baseline command refits family and definition
majorities on the unchanged training rows and reruns the deterministic
baselines:

```bash
uv run python src/09_eval.py --run-baselines --dataset full_strat1m \
  --min-occurrence 100 --strict-global-metrics --per-constraint-csv --batch-size 256
```

H2 and candidate-oracle sidecars apply to Direct--Factor, Candidate--C, and
Candidate--DP. Candidate--C H2 uses `--use-chooser`. Candidate--SR additionally
requires the schema-v3 candidate-membership and deletion-degeneracy audits. The
paper scheduler runs all of these automatically.

The final readiness and acceptance commands are documented in the
[execution plan](00_training_and_evaluation_execution_plan.md).
