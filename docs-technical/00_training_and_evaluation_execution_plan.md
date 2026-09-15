# Training and Evaluation Execution Plan

This is the operational sequence for the reported single-seed experiment suite.
The system-to-directory mapping is in the
[models and evaluation matrix](00_models_and_evaluation_matrix.md).

## Fixed policy

- benchmark: `full_strat1m_minocc100`
- encoding: `node_id`
- constraint scope: local
- seed: 42 for every learned system
- proposal validation subset: 25,000 rows
- one recorded factorized graph suite shared by all factor proposal systems
- one passive graph suite used by Direct--Passive GNN

The existing labeled Parquet and factorized graph artifacts define the training
target for the reported runs. Do not relabel or regenerate them. The current
labeler semantics apply to future datasets, while paper evaluation reconstructs
symbolic states directly from benchmark rows.

## Required local artifacts

Before execution, restore:

- `data/interim/full_strat1m_minocc100/`;
- `data/interim/full_strat1m_minocc100_labeled/`;
- factorized shards under `data/processed/full_strat1m_minocc100/`; and
- the retained Direct--Factor checkpoint at its canonical model path.

Data, graphs, and checkpoints are ignored by Git. Checked-in prediction
manifests contain the expected sizes and checksums.

If the passive graph suite is absent, generate it once:

```bash
uv run src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation eswc_passive --registry-dataset full --shard-size 10000 --use-torch-save --persistence-profile research_safe --overwrite atomic
```

## Generate configurations

```bash
uv run scripts/make_experiment_configs.py --variant full_strat1m_minocc100 --encoding node_id --models-root models
```

This emits exactly the five learned paper configurations. Use
`--include-experimental` only for work outside the reported suite. The generator
derives active factor-family IDs from training and validation, and stops if an
unseen family occurs only in the test split.

## Evaluate deterministic baselines

```bash
uv run src/09_eval.py --run-baselines --dataset full_strat1m --min-occurrence 100 --registry-dataset full --strict-global-metrics --per-constraint-csv --batch-size 256
```

Outputs are stored under `models/baselines/full_strat1m/parquet/`.

## Train and evaluate learned systems

Review the exact work list first:

```bash
uv run src/10_scheduler.py --paper-suite --dry-run
```

Then execute it:

```bash
uv run src/10_scheduler.py --paper-suite
```

The scheduler uses this fixed order:

1. Direct--Passive GNN;
2. Direct--Factor GNN;
3. Candidate--C;
4. Candidate--DP; and
5. Candidate--SR.

For an existing checkpoint it runs evaluation without retraining. Otherwise it
uses `src/07_train.py` for proposal models or `src/08_train_reranker.py` for the
satisfaction reranker, then evaluates the result. The retained Direct--Factor
checkpoint is never retrained; a missing copy stops the suite. `--only
<substring>` filters this exact list without adding other model directories.

Manual proposal training and evaluation use:

```bash
uv run src/07_train.py --experiment-config models/<run-directory>/config.json
uv run src/09_eval.py --run-directory models/<run-directory> --strict-global-metrics --per-constraint-csv --batch-size 256
```

Manual reranker training and evaluation use:

```bash
uv run src/08_train_reranker.py --experiment-config models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/config.json
uv run src/09_eval.py --run-directory models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id --legacy-predictions-json models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/reranker_predictions.json --strict-global-metrics --per-constraint-csv --batch-size 256
```

The legacy-JSON option is a one-time conversion of the reranker's ordered output
to the standard Parquet artifact; subsequent evaluation should use
`--predictions`.

## Supporting diagnostics

H2 pressure masking and candidate-oracle analysis apply to Direct--Factor GNN,
Candidate--C, and Candidate--DP. They are read-only with respect to the main
evaluation outputs.

```bash
uv run src/09_eval.py --run-directory models/<factor-run-directory> --strict-global-metrics --h2-eval
uv run scripts/analyze_candidate_oracle.py --run-directory models/<factor-run-directory> --strict-global-metrics --batch-size 256
```

For Candidate--C, pass `--use-chooser` to the H2 command. The normal diagnostic
condition replays the validated predictions; counterfactual pressure variants
are inferred from the checkpoint. Candidate-oracle analysis uses the same
label-blind candidate builder and symbolic event definitions as evaluation.

Regenerate the satisfaction-reranker deletion diagnostic with:

```bash
uv run scripts/analyze_deletion_degeneracy.py --g0-run-directory models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id --predictions models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/evaluations/predictions.parquet --strict-global-metrics
```

The read-only label audit is documented in
[symbolic evaluation and prediction artifacts](09_corrected_evaluation.md).

## Final verification

```bash
uv run scripts/check_corrected_paper_readiness.py --paper latex_paper/main.tex --verify-graph-checksums
uv run pytest
uv run python -m compileall -q src scripts
```

The readiness command checks exact configs and checkpoints, artifact provenance,
prediction order, all aggregate and per-family metrics, supporting diagnostics,
and the values displayed in the paper. Multiple seeds remain outside the scope
of the reported experiment.
