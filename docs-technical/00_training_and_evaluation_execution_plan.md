# Training and Evaluation Execution Plan

This is the operational sequence for the validator-v3, single-seed paper
suite. The implementation and artifact contract are detailed in
[Validator semantics v3 and regeneration](12_validator_semantics_revision.md).

## Fixed policy

- benchmark: `full_strat1m_minocc100`;
- encoding: `node_id`;
- constraint scope: `local`;
- validator semantics: version 3;
- hierarchy/parser/cache contracts: version 2;
- effective-hierarchy policy: version 1;
- candidate objective: version 2;
- class hierarchy cutoff: `2018-07-01T00:00:00Z`;
- training seed: 42;
- training population: every sampled training row; and
- validator-dependent losses: masked only for `unknown` instances.

The immutable unlabelled Parquet rows, constraint registry, and encoder remain
in place. Validator-derived labels, factorized graphs, learned checkpoints,
predictions, metrics, and diagnostics must all be generated under one matching
validator/hierarchy identity.

## Generation order

First build and verify the fixed hierarchy, then generate labels and both graph
representations using the commands in the validator-v3 guide. Generate the
five canonical configurations only after the new labels and graph manifests
exist:

```bash
uv run python scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 --encoding node_id --models-root models
```

The generator derives active factor-family IDs from training and validation
and fails if a type is present only in test. Candidate--C is fixed at
`gamma_primary=0.2`; every learned configuration uses seed 42.

## Learned and deterministic systems

Inspect the work list without changing state:

```bash
uv run python src/10_scheduler.py --paper-suite --dry-run
```

Run the complete suite:

```bash
uv run python src/10_scheduler.py --paper-suite
```

The scheduler enforces this order:

1. Direct--Passive GNN;
2. Direct--Factor GNN;
3. Candidate--C;
4. Candidate--DP; and
5. Candidate--SR.

Only the archived Direct--Passive checkpoint is reused, after checksum and
architecture validation. The other four systems are trained from scratch;
Candidate--SR cannot start before the new Direct--Factor checkpoint exists.
Existing non-passive checkpoints are accepted on restart only when their
training provenance has validator version 3, every current semantic-contract
version, and the current hierarchy identity.

After learned evaluation the scheduler refits and evaluates the four
deterministic/statistical baselines from the unchanged rows. It then regenerates
H2, candidate-oracle, Candidate--SR candidate-membership and deletion
diagnostics, the label audit, the combined factor summary, and benchmark
statistics. Finally it runs readiness and full-suite acceptance. Any failing
step stops the paper suite.

## Individual execution

Proposal training and strict evaluation:

```bash
uv run python src/07_train.py --experiment-config models/<run-directory>/config.json
uv run python src/09_eval.py --run-directory models/<run-directory> \
  --strict-global-metrics --per-constraint-csv --batch-size 256
```

Candidate--SR training creates an ordered internal prediction JSON, which the
scheduler immediately converts to the schema-v3 Parquet artifact:

```bash
uv run python src/08_train_reranker.py \
  --experiment-config models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/config.json
uv run python src/09_eval.py \
  --run-directory models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id \
  --legacy-predictions-json models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/reranker_predictions.json \
  --strict-global-metrics --per-constraint-csv --batch-size 256
```

Subsequent replay must use `--predictions .../evaluations/predictions.parquet`;
schema-v2 and cross-semantics replay are rejected.

## Manuscript gate

Do not edit paper values until the scheduler's acceptance step succeeds. After
updating the manuscript from the machine-readable results, bind it to those
results with:

```bash
uv run python scripts/check_corrected_paper_readiness.py \
  --paper latex_paper/main.tex --verify-graph-checksums
uv run python scripts/validate_regenerated_suite.py --require-paper
uv run pytest -q
uv run python -m compileall -q src scripts tests
```

The readiness check recomputes metric events from predictions and requires the
LaTeX tables to match. `--require-paper` makes final acceptance fail unless that
paper-bound readiness report is present. Multiple seeds and external release
publication remain outside this experiment.
