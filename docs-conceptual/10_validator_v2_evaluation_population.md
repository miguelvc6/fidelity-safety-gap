# Validator-v2 Evaluation Populations

The principal experiment continues to train on every sampled historical
correction. Validator-dependent losses are masked only where the attached
constraint instance is not checkable. We do not filter the training population
according to whether the recorded historical edit satisfies the new validator.
Doing so would condition the benchmark on the outcome being measured and would
turn the main experiment into a different, easier imitation task.

Historical fidelity is evaluated on the complete test split. Symbolic repair
metrics use instances whose primary constraint is definitely checkable and
violated before the edit. Within that fixed population, reports distinguish
historical fixes, historical non-fixes, and historical edits whose result is
uncheckable. These strata diagnose supervision quality; they neither select
training examples nor tune validator behavior. The post-edit primary-
satisfaction percentage is reported by split and family without an acceptance
threshold.

Symbolic evaluation combines the bounded instance evidence with one fixed
Wikidata `P279` hierarchy whose cutoff is 1 July 2018. It never consults live
Wikidata. Missing local evidence or incomplete historical ancestry produces
`unknown` rather than a guessed Boolean result.

The engineering contract and regeneration commands are documented in
[Validator semantics v2 and experiment regeneration](../docs-technical/12_validator_semantics_revision.md).
