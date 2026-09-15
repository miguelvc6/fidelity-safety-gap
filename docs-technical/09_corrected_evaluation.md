# Symbolic Evaluation and Prediction Artifacts

The research rationale is described in the
[evaluation protocol](../docs-conceptual/08_corrected_evaluation_protocol.md).
This document defines the implementation, artifact contract, and commands.

## State and label policy

`src/modules/evidence_state.py` is shared by evaluation and
`src/05_constraint_labeler.py`. It merges aliased role evidence, guarantees that
the base statement exists in the pre-edit state, and applies deletion before
addition.

Do not run the labeler against the recorded `full_strat1m_minocc100` paper
benchmark. Its labeled Parquet and factorized graph shards are the common
training source for the reported factor models. Evaluation independently
reconstructs symbolic states from the benchmark rows. The labeler applies the
same state semantics when preparing a future dataset.

The read-only comparison between stored labels and recomputed semantics is:

```bash
uv run scripts/audit_label_semantics.py --dataset full_strat1m --min-occurrence 100 --registry-dataset full --labeled-dir data/interim/full_strat1m_minocc100_labeled --output models/paper_diagnostics/label_semantics_audit.json
```

It reports differences by split and constraint family without modifying data,
graphs, configurations, or checkpoints.

## Evaluation outputs

Every evaluation writes under `<run>/evaluations/`:

- `model.json`: operation-level fidelity, symbolic metrics, and artifact links;
- `per_constraint.csv`: the same metrics by constraint family;
- `predictions.parquet`: ordered row identity, six predicted slots, resolved
  operations, and per-row metric events; and
- `predictions.manifest.json`: schema version, row count, producer config and
  checkpoint, dataset and graph identities, and SHA-256 checksums.

Repository-owned paths in JSON provenance are repository-relative, so a clone
can be moved without invalidating identity checks. External paths remain
absolute. Writes use a temporary file followed by atomic replacement. When an
older `model.json` or `per_constraint.csv` exists, the first version-2 write
preserves it once as `*.pre-schema-v2.*`.

Replay uses `--predictions`. It rejects a missing manifest, the wrong schema,
count or row-order changes, dataset-identity changes, and prediction, dataset,
or graph checksum changes. `--reranker-predictions` remains a deprecated alias.
`--legacy-predictions-json` exists only to migrate the reranker's ordered JSON
output into the validated Parquet format.

## Canonical configurations

After restoring the recorded graph and labeled-Parquet inputs, generate the
five paper configurations with:

```bash
uv run scripts/make_experiment_configs.py --variant full_strat1m_minocc100 --encoding node_id
```

The default set comprises Direct--Passive GNN, Direct--Factor GNN,
Candidate--C, Candidate--DP, and Candidate--SR. The factor proposal systems use
grouped family execution, compact gold-edit encoding, per-family pressure
modules, and factor IDs derived only from training and validation. Candidate--SR
references the Direct--Factor proposal configuration. Experimental variants are
opt-in and are not scheduled by `--paper-suite`.

The passive graph suite is shared and can be generated once if it is absent:

```bash
uv run src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation eswc_passive --registry-dataset full --shard-size 10000 --use-torch-save --persistence-profile research_safe --overwrite atomic
```

Do not regenerate the recorded factorized graph suite.

## Running the learned suite

Inspect the exact order and next action without creating logs:

```bash
uv run src/10_scheduler.py --paper-suite --dry-run
```

Run the suite:

```bash
uv run src/10_scheduler.py --paper-suite
```

The scheduler evaluates an existing checkpoint and otherwise trains then
evaluates. Direct--Factor GNN is the retained exception and must already have
its checkpoint. The exact order is Direct--Passive GNN, Direct--Factor GNN,
Candidate--C, Candidate--DP, and Candidate--SR. Candidate--SR training uses
`src/08_train_reranker.py`; its test predictions exclude injected gold edits,
then evaluation migrates the generated ordered JSON to the standard Parquet
artifact.

Individual evaluation and replay are:

```bash
uv run src/09_eval.py --run-directory models/<run-directory> --strict-global-metrics --per-constraint-csv --batch-size 256
uv run src/09_eval.py --run-directory models/<run-directory> --predictions models/<run-directory>/evaluations/predictions.parquet --strict-global-metrics --per-constraint-csv
```

Run deterministic baselines with:

```bash
uv run src/09_eval.py --run-baselines --dataset full_strat1m --min-occurrence 100 --strict-global-metrics --per-constraint-csv --batch-size 256
```

## Diagnostics and readiness

H2 pressure masking and the candidate oracle apply to Direct--Factor GNN,
Candidate--C, and Candidate--DP. They replay each run's validated predictions
for the unmodified condition, while masked or oracle choices are recomputed
through the same symbolic event builder. The deletion-degeneracy analysis can
be regenerated for Candidate--SR from its standard Parquet predictions.

The final gate is:

```bash
uv run scripts/check_corrected_paper_readiness.py --paper latex_paper/main.tex --verify-graph-checksums
```

It validates the five learned systems and four deterministic baselines,
reaggregates operation fidelity and every symbolic metric from predictions,
checks per-family tables and provenance, validates supporting diagnostics, and
requires the displayed LaTeX tables to equal the allowlisted artifacts. It
writes `models/paper_diagnostics/corrected_paper_readiness.json` only after all
checks pass.
