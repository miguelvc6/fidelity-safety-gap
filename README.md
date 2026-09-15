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

## Sample smoke run

The sample pipeline is intended for development, not reproduction of the paper
table.

```bash
uv run src/01_data_downloader.py --dataset sample
uv run src/02_dataframe_builder.py --dataset sample --min-occurrence 100 --max-rows 300
uv run src/03_constraint_registry.py --dataset sample
uv run src/05_constraint_labeler.py --dataset sample --min-occurrence 100 --constraint-scope local --max-rows 300
uv run src/06_graph.py --dataset sample --min-occurrence 100 --encoding node_id --constraint-scope local --constraint-representation factorized
uv run src/09_eval.py --run-baselines --dataset sample --min-occurrence 100
```

## Reproducing the reported experiment suite

The paper uses `full_strat1m_minocc100`, `node_id` encoding, seed 42, and the
recorded labeled Parquet and graph artifacts. These large inputs and learned
checkpoints are intentionally not stored in Git. Place them under `data/` and
the corresponding `models/<run>/checkpoint.pth` paths before running the suite.
The checked-in configurations, aggregate reports, prediction manifests, and
paper tables record their identities and checksums.

Do not rerun `src/05_constraint_labeler.py` over the released paper benchmark:
the reported models were trained on its recorded labels. Evaluation reconstructs
pre- and post-edit symbolic states from the benchmark rows using the current
semantics. The labeler contains the same semantics for future datasets.

Generate or restore the passive graph suite once; all factor-based systems use
the recorded factorized graph suite:

```bash
uv run src/06_graph.py --dataset full_strat1m --min-occurrence 100 --encoding node_id --constraint-representation eswc_passive --registry-dataset full --shard-size 10000 --use-torch-save --persistence-profile research_safe --overwrite atomic
uv run scripts/make_experiment_configs.py --variant full_strat1m_minocc100 --encoding node_id
uv run src/09_eval.py --run-baselines --dataset full_strat1m --min-occurrence 100 --strict-global-metrics --per-constraint-csv --batch-size 256
uv run src/10_scheduler.py --paper-suite --dry-run
uv run src/10_scheduler.py --paper-suite
uv run scripts/check_corrected_paper_readiness.py --paper latex_paper/main.tex --verify-graph-checksums
```

The dry run prints exactly the five learned experiments and whether each will be
trained or evaluated. An existing checkpoint is evaluated; otherwise its model
is trained first. Direct--Factor GNN is the exception: its retained checkpoint
must be restored and is never retrained by the paper scheduler.

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
uv run pytest
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
