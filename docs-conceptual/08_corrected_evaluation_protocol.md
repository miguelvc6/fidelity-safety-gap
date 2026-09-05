# Evaluation Protocol

The reported comparison uses one regenerated validator-v3 generation. All
factor-derived labels, factorized graph wiring, factor-dependent checkpoints,
predictions, and metrics share the same fixed semantics and 1 July 2018 class
hierarchy. Stored graph tensors are training metadata; evaluation reconstructs
symbolic states and metric events from benchmark rows.

## Comparison scope

The learned comparison contains Direct--Passive GNN, Direct--Factor GNN,
Candidate--C, Candidate--DP, and Candidate--SR, plus four symbolic/statistical
baselines. All training uses seed 42 and all sampled training rows. Only
validator-dependent losses are masked for `unknown` instances. The passive
checkpoint is reused because its input payload is unchanged; every
factor-dependent system is retrained.

The rationale for full-population training and the historical-fix diagnostic
strata is specified in
[Validator-v3 evaluation and objective decisions](11_validator_v3_evaluation_and_objective_decisions.md).

## Symbolic evidence

One state definition is shared by labeling, training objectives, reranking,
diagnostics, and evaluation:

- role projections resolving to the same entity are merged;
- the focus and explicit auxiliary statements are represented;
- deletion precedes addition;
- delete-and-reinsert preserves the statement;
- primary definitions bind to the focus subject/property; and
- secondary definitions bind to every represented occurrence of their
  constrained property.

Constraint results are three-valued. Incomplete local evidence, unavailable
historical hierarchy revisions, malformed definitions, unsupported scopes,
and unrepresentable mandatory parameters produce `unknown`, not an assumed
Boolean. Type traversal uses the immutable fixed historical hierarchy with the
current state's local P279 additions and deletion tombstones; live Wikidata is
never a fallback.

## Metrics

Every aggregate reports a value, numerator, and denominator.

- Primary-Fix Rate measures eligible primary violations that become satisfied.
- Local Satisfaction pools satisfied post-edit constraints over post-edit
  checkable constraints.
- Change in Local Satisfaction measures signed change on constraints checkable
  both before and after the edit.
- Secondary Improvement and Regression Rates pool secondary transitions on
  common support.
- Disruption counts complete predicted additions and deletions.
- Base-deletion Rate measures whether the reconstructed focus statement is
  lost.
- Deletes-base-action Rate measures whether the predicted deletion explicitly
  names it.
- Evidence-Preserving Primary Fix requires both a primary fix and retained base
  evidence.
- Vacuous Improvement records positive common-support change accompanied by
  base deletion.

Historical fidelity is always reported on the full test split. Symbolic repair
metrics use the definitely checkable, pre-edit violated primary population,
with historical fix/non-fix/uncheckable status retained only as diagnostic
strata.
