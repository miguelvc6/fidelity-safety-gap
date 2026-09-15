# Evaluation Protocol

The paper separates continuity of the training target from correctness of the
symbolic evaluation. The reported factor-based models use the same recorded
labels and factorized graphs, preserving a common historical-imitation target.
At evaluation time, pre- and post-edit evidence states are reconstructed from
the benchmark rows. Stored graph satisfaction tensors are therefore training
metadata, not evaluation truth.

## Comparison scope

The learned comparison contains a passive-context GNN, a direct factor-based
GNN, two candidate-selection variants, and a learned satisfaction reranker.
The factor-based proposal systems share the same graph artifacts and backbone;
the reranker uses proposals from the direct factor-based model. All training
uses seed 42. Multiple-seed confirmation remains future work.

Older architecture variants and an unused passive capacity control are outside
the reported comparison. They are not needed to interpret the paper's claims.

## Symbolic-state contract

One evidence-state definition is used for evaluation and for labeling future
datasets:

- evidence projections are merged when graph roles resolve to the same entity;
- the base statement is inserted into every reconstructed pre-edit state;
- deletion is applied before addition; and
- deleting and adding the same statement preserves its reinsertion.

The recorded benchmark labels and graphs are not rebuilt. A read-only audit
quantifies differences between their stored factor labels and recomputation
under this contract without changing training artifacts.

## Metrics

Every aggregate reports a value, numerator, and denominator.

- Primary-Fix Rate measures eligible primary violations that become satisfied.
- Local Satisfaction pools satisfied post-edit constraints over post-edit
  checkable constraints.
- Change in Local Satisfaction measures signed change on constraints checkable
  both before and after the edit.
- Secondary Improvement and Regression Rates pool secondary transitions on that
  common support.
- Disruption counts complete predicted addition and deletion operations.
- Base-deletion Rate measures whether the reconstructed base statement is lost.
- Deletes-base-action Rate measures whether the predicted deletion explicitly
  names the base statement.
- Evidence-Preserving Primary Fix credits an eligible primary fix only when the
  base evidence remains.
- Vacuous Improvement records a positive common-support satisfaction change
  accompanied by base deletion.

The delete-base baseline is the non-vacuity control: it deletes the base on
every row and cannot receive evidence-preserving primary-fix credit.
