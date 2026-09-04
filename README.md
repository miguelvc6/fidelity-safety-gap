# Constraint Factors

Executable constraint factors for neuro-symbolic knowledge-graph repair.

This repository accompanies *The Fidelity--Safety Gap in Neural Wikidata
Constraint Repair*. It studies whether explicitly representing local constraints
helps a repair model reproduce curator edits, and whether that historical
fidelity translates into safe symbolic outcomes.

## Paper systems

The reported learned systems are:

| Paper name | Internal run prefix | Role |
| --- | --- | --- |
| Direct--Passive GNN | `b0_eswc_reproduction` | neural GNN with passive constraint context |
| Direct--Factor GNN | `a1_factorized_imitation_compact_grouped` | direct prediction with executable factors |
| Candidate--C | `m1c_safe_factor_chooser_compact_grouped` | learned candidate chooser |
| Candidate--DP | `m1d_safe_factor_direct_compact_grouped` | direct candidate scoring with safety losses |
| Candidate--SR | `g0_globalfix_reference_v2` | learned satisfaction reranker |

The paper also evaluates four deterministic baselines. The internal prefixes
are kept for artifact compatibility; the paper uses only the reader-facing
names above.

## Installation

Requirements are Python 3.12 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is
recommended for training the reported models.

```bash
uv sync --group dev --python 3.12
```

## Validator-v2 smoke run

The self-contained smoke fixture exercises label generation and factorized
graph construction without consulting live Wikidata:

```bash
uv run python tests/smoke_validator_v2_pipeline.py
```

## Reproducing the reported experiment suite

The paper uses `full_strat1m_minocc100`, `node_id` encoding, seed 42, validator
semantics version 2, and a fixed Wikidata class hierarchy cut off at
2018-07-01. Build that artifact first, then regenerate labels and both graph
suites. Large generated inputs and checkpoints are intentionally not stored in
Git.

```bash
uv run python scripts/build_historical_class_hierarchy.py --seeds-only
uv run python scripts/build_historical_class_hierarchy.py
uv run python src/05_constraint_labeler.py --dataset full_strat1m --min-occurrence 100 --registry-dataset full --constraint-scope local --hierarchy data/static/wikidata-p279-2018-07-01.v1.json
uv run python src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation factorized --registry-dataset full --constraint-scope local --shard-size 200000 --use-torch-save --persistence-profile research_safe --overwrite atomic --hierarchy data/static/wikidata-p279-2018-07-01.v1.json
uv run python src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation eswc_passive --registry-dataset full --constraint-scope local --shard-size 10000 --use-torch-save --persistence-profile research_safe --overwrite atomic --use-unlabeled-interim --hierarchy data/static/wikidata-p279-2018-07-01.v1.json
uv run python scripts/make_experiment_configs.py --variant full_strat1m_minocc100 --encoding node_id
uv run python src/10_scheduler.py --paper-suite --dry-run
uv run python src/10_scheduler.py --paper-suite
```

The dry run prints exactly the five learned experiments and whether each will be
trained or evaluated. Only the checksummed Direct--Passive checkpoint is
retained; all factor-dependent systems are retrained. The scheduler also runs
the four baselines, diagnostics, and acceptance checks. Paper values are updated
only after that suite passes.

For individual training, evaluation, replay, and diagnostic commands, see the
[execution guide](docs-technical/00_training_and_evaluation_execution_plan.md)
and [evaluation artifact guide](docs-technical/09_corrected_evaluation.md).

## Repository layout

- `src/`: pipeline, model training, and evaluation entry points
- `src/modules/`: reusable graph, model, and symbolic-evaluation components
- `scripts/`: configuration, audit, and readiness utilities
- `tests/`: unit, regression, and paper-surface checks
- `latex_paper/`: submission source and rendered paper
- `models/`: allowlisted configurations and evaluation results
- `docs-conceptual/`: research framing and evaluation rationale
- `docs-technical/`: implementation and operating procedures

Raw/interim data, processed graphs, checkpoints, logs, notebooks, and local run
commands are ignored by Git.

## Validation

```bash
uv run pytest -q
uv run python -m compileall -q src scripts
```

The readiness gate additionally reaggregates reported metrics from prediction
artifacts, checks provenance, and verifies that the LaTeX result tables match
the allowlisted outputs.

## Paper and citation

The paper source is [latex_paper/main.tex](latex_paper/main.tex), with the
rendered manuscript at [latex_paper/main.pdf](latex_paper/main.pdf). Citation
metadata is provided in [CITATION.cff](CITATION.cff).

## Licensing

Software is licensed under the MIT License. Author-created paper text and
figures, evaluation results, processed annotations, prediction artifacts, and
model weights are licensed under CC BY 4.0. Upstream data and third-party files
retain their own terms. See [LICENSES/README.md](LICENSES/README.md).
