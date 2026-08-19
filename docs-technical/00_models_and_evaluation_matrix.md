# Paper Models and Evaluation Matrix

This document maps the reader-facing systems in the paper to repository
configurations. Internal run names are retained only to make commands and
artifacts unambiguous.

## Learned systems

| Paper name | Internal prefix | Representation | Training entry point | Decision rule |
| --- | --- | --- | --- | --- |
| Direct--Passive GNN | `b0_eswc_reproduction` | passive constraint context | `src/07_train.py` | independent slot argmax |
| Direct--Factor GNN | `a1_factorized_imitation_compact_grouped` | executable factors | `src/07_train.py` | independent slot argmax |
| Candidate--C | `m1c_safe_factor_chooser_compact_grouped` | executable factors | `src/07_train.py` | learned chooser over candidates |
| Candidate--DP | `m1d_safe_factor_direct_compact_grouped` | executable factors | `src/07_train.py` | proposal-score candidate selection |
| Candidate--SR | `g0_globalfix_reference_v2` | factor proposals plus reranker | `src/08_train_reranker.py` | learned satisfaction reranking |

All reported training uses seed 42. Direct--Passive GNN has a 128-wide,
two-layer backbone and dropout 0.5. The three factor-based proposal models use
the same 400-wide, four-layer backbone and the recorded factorized graph suite.
Candidate--SR draws proposals from Direct--Factor GNN.

The default generator emits exactly these five configurations. Experimental
and appendix configurations, including the unused 304-wide passive capacity
control, require `--include-experimental` or a study-specific option and are not
part of the paper suite.

## Factor-model defaults

Direct--Factor GNN, Candidate--C, and Candidate--DP use:

- `factor_executor_impl="per_type_grouped_v2"`;
- `gold_edit_embedding_mode="compact"`;
- `pressure_module_sharing="per_type"`;
- active factor IDs derived from training and validation only; and
- rejection of any factor family that appears only at test time.

Candidate--C uses `gamma_primary=0.2`. Candidate--DP adds direct primary and
secondary candidate losses. Candidate--SR trains a separate reranker for local
constraint satisfaction and saves its final checkpoint.

## Deterministic baselines

The paper reports:

| Paper name | Implementation |
| --- | --- |
| Baseline--DB | `DeleteFocusBaseline` |
| Baseline--AM | `AddMirrorBaseline` |
| Baseline--FM | `ConstraintFamilyMajorityBaseline` |
| Baseline--DM | `ConstraintDefinitionMajorityBaseline` |

These systems have no learned checkpoint. Their outputs are written under
`models/baselines/full_strat1m/parquet/`.

## Required evaluation outputs

Every system reports operation-level precision, recall, and micro-F1; mean
additions and deletions; and the symbolic metrics defined in the paper:
Primary-Fix Rate, Local Satisfaction, change in Local Satisfaction, Secondary
Improvement and Regression Rates, disruption, both base-deletion measures,
Evidence-Preserving Primary Fix, and Vacuous Improvement.

Each symbolic aggregate stores `value`, `numerator`, and `denominator`.
Evaluation reconstructs pre- and post-edit states from benchmark rows rather
than treating stored graph factor-label tensors as evaluation truth.

## Diagnostics

H2 pressure masking and the candidate oracle are supporting analyses for
Direct--Factor GNN, Candidate--C, and Candidate--DP. They write below each run's
`evaluations/` directory and never overwrite the main result. The final
readiness check validates all five learned systems, all four deterministic
baselines, diagnostic selection modes, prediction provenance, and paper tables.
