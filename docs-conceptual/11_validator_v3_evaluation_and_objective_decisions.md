# Validator-v3 Evaluation and Objective Decisions

The principal experiment continues to train on every sampled historical
correction. We do not filter training rows according to whether the recorded
edit satisfies the symbolic validator. Such a filter would condition the task
on an evaluated outcome and turn it into an easier, different imitation
problem.

Historical fidelity is evaluated on the complete test split. Primary symbolic
repair metrics use the fixed population whose pre-edit primary constraint is
definitely violated. Historical-fix, historical-nonfix, and post-edit-unknown
subsets are diagnostic strata only. The post-edit satisfaction percentage is
reported by split and family without an acceptance threshold. Validator
behavior is not tuned to reproduce the former 97.16% conditional pass rate or
any old model ranking.

The external type evidence is the Wikidata `P279` hierarchy at the fixed
1 July 2018 cutoff. It is never replaced by live Wikidata. Because the repair
language permits `P279` changes, the semantic state combines that immutable
background with the candidate's bounded local additions and deletions. A known
path proves satisfaction; incomplete ancestry prevents a definite negative.
The fixed cutoff provides a reproducible evidence boundary, not an exact
per-edit temporal reconstruction.

Candidate--C and Candidate--DP deliberately optimize proven primary fixes. An
instance is eligible only when its pre-edit primary result is violated. Within
that fixed population, probability assigned to a candidate whose post-edit
result is unknown or violated incurs the same primary failure cost; unknown is
not relabeled as a violation in stored validator outcomes. Candidate--SR remains
satisfaction-only so the systems retain their intended conceptual distinction.
Auxiliary factor-label learning continues to mask unknown outcomes.

The implementation and regeneration contract are in
[Validator semantics v3 and regeneration](../docs-technical/12_validator_semantics_revision.md).
