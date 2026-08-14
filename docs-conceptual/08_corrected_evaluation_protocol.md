# Corrected Evaluation Protocol

The corrected paper protocol separates training-label continuity from symbolic
evaluation correctness. The current factorized graphs retain their historical
factor labels so Compact A1 can be evaluated without checkpoint migration and
M1C/M1D remain directly comparable to it. At evaluation time, however, both the
pre-edit and post-edit evidence states are reconstructed from the interim row
under the corrected semantics. Stored graph satisfaction tensors are not
evaluation truth.

## Canonical comparison

Compact A1 is the canonical factorized imitation system. Original A1 is kept
only as prior compression-equivalence evidence; its older safety diagnostics do
not belong in the corrected main comparison.

Original B0 is the passive comparison in the current paper suite. It is the
128-wide, two-layer replication system and answers whether the earlier passive
setup can be reproduced. The proposed 304-wide, four-layer parameter-matched
B0 remains a future capacity control but is excluded from the current run and
paper comparison.

Compact M1C and M1D use the same stored factorized train/validation/test graphs
as Compact A1. The G0 design uses Compact A1 proposals, but the retained G0 run
is excluded because its reranker checkpoint is missing and its saved selections
cannot all be reconstructed from label-blind candidates. Restoring G0 requires
single-seed retraining with seed 42. All new training uses that seed;
multiple-seed confirmation remains deferred.

## Symbolic-state contract

One evidence-state definition is used for corrected evaluation and future label
generation:

- evidence projections are merged when graph roles resolve to the same entity;
- the base statement is inserted into every reconstructed pre-state;
- deletion is applied before addition; and
- deleting and then adding the same statement preserves reinsertion.

The current labeled Parquet and graph artifacts are intentionally not rebuilt.
A read-only label-semantics audit measures their drift from this corrected
contract.

## Paper metrics

Every aggregate reports a value together with its numerator and denominator.

- PFR: a pre-checkable, pre-violated primary constraint that is checkable and
  satisfied after the edit.
- Local Satisfaction: pooled satisfied post-edit constraints over post-edit
  checkable constraints.
- ΔLocalSat: the signed satisfaction change over common pre/post checkable
  support.
- SIR and SRR: pooled secondary improvements and regressions on common support.
- Disruption: complete non-`none` predicted add/delete slot groups per instance;
  partial groups do not count, while a complete group still counts if bounded
  state reconstruction cannot apply it.
- Base-deletion Rate: reconstructed base statements removed by the edit.
- Deletes-base-action Rate: predictions whose resolved delete action names the
  base statement.
- EPPF: eligible primary fixes that preserve the base evidence.
- Vacuous Improvement: base-deleting edits with positive common-support
  ΔLocalSat.

This schema replaces the historical GFR label and action-history-derived
primary-fix diagnostic. The deletion-focused baseline is an explicit
non-vacuity control: it must delete the base on every row and cannot receive
EPPF credit.
