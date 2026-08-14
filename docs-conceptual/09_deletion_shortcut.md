# Satisfaction objectives and evidence deletion

A repair objective can report improved constraint satisfaction while destroying the statement that made the constraint relevant. This is not merely an implementation failure: when the objective rewards post-edit satisfaction without rewarding evidence preservation, deleting the base statement is a valid shortcut under that objective.

The M1D and G0 experiments test this failure mode at two levels. M1D applies the satisfaction objective directly to the proposal model. G0 learns a candidate reranker whose objective is local post-edit satisfaction. Their shared tendency toward deletion is therefore evidence about the objective, rather than evidence about one particular architecture.

The matched base-preserving variants add the symbolic event “the base statement is absent after the edit” as a soft training penalty. They do not remove deletion candidates or hard-code a repair policy. A delete-and-reinsert operation remains available and is recorded as a delete action, but it is not counted as base deletion because the final evidence state preserves the statement.

The comparison is interpreted through two separate gates:

- Correctness requires stable training, label-blind G0 inference, complete provenance, and corrected full-test evaluation.
- Successful mitigation additionally requires lower Base-deletion Rate and Deletes-base-action Rate, together with higher evidence-preserving primary fixes, than the matched satisfaction-only control.

Only one registered seed, 42, is used. The mitigation settings are fixed before test evaluation; test results are not used for hyperparameter selection.
