# Executable Constraint Factors and the Fidelity-Safety Gap

This document is the canonical conceptual framing for the constraint-factors
project. It records the current paper narrative: executable constraint factors
are valuable because they improve historical repair imitation and reveal where
imitation, symbolic safety, and non-vacuous repair quality diverge.

Implementation details, code-level boundaries, and run status belong in the
[technical documentation](../docs-technical/README.md). The paper-facing system
mapping is maintained in the
[models and evaluation matrix](../docs-technical/00_models_and_evaluation_matrix.md).

## 1) Research Idea, Main Thesis, Scientific Contribution

### Core idea

Extend structure-aware neural repair for collaborative knowledge graphs by
representing Wikidata constraints as first-class executable factor nodes inside
each violation-centered subgraph.

These factors are not passive context labels. Each local constraint instance is
represented as a typed factor with:

- a constraint family, such as `single`, `conflictWith`, or `valueType`;
- a scope over subject, predicate, object, or value nodes;
- a learned satisfaction head;
- role-conditioned pressure messages back to the variables in its scope.

The factors are learned neural approximations of local constraint behavior. The
grounded post-edit evaluation still uses symbolic validators.

### Current thesis

**Historical repair fidelity and symbolic repair safety are distinct objectives.**
Executable constraint factors improve historical edit imitation, and diagnostic
experiments show that their pressure signals affect predictions. However, better
imitation does not automatically produce safer post-edit graphs. Conversely,
directly optimizing local post-edit satisfaction can collapse into deleting the
violating statement.

The paper should therefore argue for a multi-axis evaluation framework for KG
repair rather than claim that the current architecture has solved safe repair.

### Scientific contribution

1. **Representation result:** executable constraint factors improve historical
   repair prediction over a passive constraint-context baseline.
2. **Evaluation result:** higher historical fidelity does not imply better
   local symbolic safety.
3. **Degeneracy result:** local satisfaction alone is an unsafe repair objective
   because it can reward deleting the evidence that made the violation visible.
4. **Diagnostic result:** factor satisfaction heads learn useful signals, and
   pressure masking shows that factor messages are causally used, but mostly for
   imitation and primary repair behavior rather than secondary no-regression.
5. **Benchmark contribution:** the project separates historical fidelity,
   primary repair, local satisfaction, secondary improvement/regression, edit
   disruption, and non-vacuity.

We do not assume that every applicable constraint should be satisfied after a
local repair. Wikidata constraints are soft, evolving, and sometimes incomplete.
The goal is to understand when a local edit imitates historical repair behavior,
when it improves the symbolic graph state, and when it merely removes evidence.

---

## 2) Paper Narrative Summary

### Problem

Collaborative KGs such as Wikidata use property constraints to flag suspicious
statements. Historical curator edits are useful supervision, but they are not a
complete proxy for repair quality. A curator-like edit can preserve or introduce
secondary violations, while a highly satisfying symbolic edit can be vacuous if
it deletes the statement under repair.

### Limitation of prior framing

The original project framing expected executable factors to directly reduce
collateral damage while preserving curator fidelity. The evaluated results are
more interesting and more cautious:

- factorized models improve historical edit-slot prediction;
- aggregate symbolic safety does not automatically improve;
- direct safety objectives are promising but not yet decisive;
- global/local satisfaction optimization is vulnerable to a delete-focus
  degeneracy.

### Revised solution framing

The method is best presented as a constraint-aware representation and diagnostic
instrument:

- it makes local constraint state available to the neural repair model;
- it tests whether learned constraint pressure improves edit decisions;
- it exposes a measurable fidelity-safety gap;
- it motivates future decision-level repair objectives with explicit
  non-vacuity and secondary no-regression controls.

### Key claims to validate

- Executable factors improve historical repair modeling.
- Better historical repair modeling does not imply safer post-edit local graphs.
- Local satisfaction must be evaluated separately from repair usefulness.
- Secondary no-regression requires decision-level control beyond representation
  learning.

---

## 3) Dataset Construction

### Starting point

Use the historical Wikidata constraint repair corpus: violation instances paired
with human repair edits.

Each instance includes:

- a violating focus triple `(s, p, o)`;
- an optional conflicting triple `(s', p', o')`;
- a constraint identifier, family, and definition metadata;
- a local neighborhood around the involved entities;
- the historical repair edit `Delta*` as add/delete operations.

### Paper benchmark slice

The preprocessing pipeline can produce roughly 1.91M local repair instances for
the full min-occurrence-100 corpus. The paper-facing benchmark is a deterministic
stratified slice of roughly 1M instances rather than the full produced corpus.

The strata combine:

- train/validation/test split;
- primary constraint family;
- local executable-constraint density.

This keeps the benchmark computationally tractable while preserving the factor
most directly tied to the paper's hypothesis: how local constraint density
changes repair behavior.

### Locally applicable constraint set

For each instance, define the locally applicable constraint set as the primary
violated constraint plus the constraints attached to predicates visible in the
instance's bounded local neighborhood.

Let `N(i)` be the local neighborhood materialized for repair instance `i`. It
contains the focus triple, the optional conflicting triple, and cached
one-hop statements for the focus subject, focus object, and conflicting entity
when available. Let:

$$
P_{\mathrm{local}}(i)
$$

be the set of predicates appearing in that materialized neighborhood. If
`A(p)` denotes the constraint instances attached to property `p`, then:

$$
C_{\mathrm{local}}(i)
= \{c^*(i)\}
  \cup
  \bigcup_{p \in P_{\mathrm{local}}(i)} A(p)
$$

Equivalently:

$$
C_{\mathrm{local}} = \{c^*, c_1, \ldots, c_m\}
$$

where `c*` is the primary violated constraint and the remaining constraints are
locally applicable secondary constraints.

This is a bounded local-closure definition. It is broader than using only the
property of the focus triple, because historical repairs may add or delete a
different nearby statement to fix a violation. For example, type-, value-type-,
inverse-, required-statement-, or conflict-style repairs can involve predicates
on the same local entities rather than only the originally flagged predicate.
Including those attached constraints gives the model and evaluator visibility
into secondary constraints that the candidate edit may improve or regress.

The definition is still local. It does not attempt to enumerate all Wikidata
constraints that could become relevant after arbitrary graph expansion, and it
does not assert that every secondary constraint should be satisfied by the
repair. Secondary constraints are included because they are locally visible and
potentially affected, not because they are all repair objectives.

The current paper-facing pipeline uses the bounded local-closure version through
`constraint_scope=local`. A narrower focus-scoped alternative is exposed through
the technical `focus` scope; that alternative restricts the constraint set to
the primary constraint and constraints attached to the focus/conflict predicates
plus the constrained property of the violated constraint. Thus, in the paper
line, `local` means bounded local closure, not an unbounded KG neighborhood and
not the predicate-only focus scope.

### Constraint labels

For every local constraint `c`, symbolic validators compute satisfaction on:

- the pre-repair local graph `G^-`;
- the post-gold graph after applying the historical edit `Delta*`;
- the post-predicted graph after applying a model edit `Delta`.

These labels support both auxiliary factor supervision and post-edit symbolic
evaluation.

---

## 4) Per-Instance Subgraph Model

Each repair instance becomes a heterogeneous graph.

### Variable nodes

- entity nodes;
- predicate/property nodes;
- literal/value nodes;
- optional role-specific nodes for focus triple elements.

### Factor nodes

For each `c in C_local`, add a typed constraint factor node:

- type or family `t(c)`;
- scope `Scope(c)` over variable nodes;
- learned satisfaction estimator;
- learned role-conditioned pressure messages.

All factors of the same family share parameters. This is what makes the factor
representation reusable across concrete constraint instances.

### Edges

Variable-to-variable edges encode the local KG context. Variable-to-factor edges
connect scoped variables to each constraint factor. Factor-to-variable messages
send constraint pressure back to subjects, predicates, objects, or values.

---

## 5) Prediction Task and Factor Semantics

### Edit prediction

The primary task predicts a local six-slot add/delete edit:

$$
\hat{\Delta} =
(\hat{y}_{s+}, \hat{y}_{p+}, \hat{y}_{o+},
 \hat{y}_{s-}, \hat{y}_{p-}, \hat{y}_{o-})
$$

The supervision target is the historical curator edit `Delta*`.

### Factor satisfaction prediction

Each factor predicts satisfaction:

$$
\hat{s}_c(G) \in [0, 1]
$$

Labels come from symbolic validation on pre-repair and post-gold local graph
states. These heads test whether the model learns meaningful local constraint
signals, but they are not themselves proof of safe repair decisions.

### Important interpretation

The term "executable" should be used precisely. The factor is executable in the
sense that it computes a learned, typed constraint-satisfaction signal and emits
pressure messages during neural inference. It is not a replacement for the
symbolic Wikidata validator used in evaluation.

---

## 6) Message Passing and Constraint Pressure

### Variable-to-variable message passing

Use a standard GNN backbone over the local KG graph:

$$
h_v^{(k+1)}
= \mathrm{GNN}\left(h_v^{(k)}, \mathcal{N}(v)\right)
$$

### Factor execution

For each factor node `c`:

$$
z_c^{(k)}
= \operatorname{AGG}_{t(c)}
\left(\{h_v^{(k)} : v \in \operatorname{Scope}(c)\}\right)
$$

$$
\hat{s}_c^{(k)}
= \sigma\left(f_{t(c)}(z_c^{(k)})\right)
$$

$$
m_{c \to v}^{(k)}
=
g_{t(c), r(v,c)}
\left(z_c^{(k)}, h_v^{(k)}, 1 - \hat{s}_c^{(k)}\right)
$$

where `r(v,c)` is the role of variable `v` in constraint `c`.

### Constraint pressure integration

Variables incorporate pressure messages through a residual update:

$$
h_v^{(k+1)}
\leftarrow
h_v^{(k+1)}
+
\lambda
\sum_{c: v \in \operatorname{Scope}(c)}
m_{c \to v}^{(k)}
$$

Pressure-masking diagnostics test whether these messages are actually used by
the model and whether secondary constraints influence decisions.

---

## 7) Model Suite and Baseline Roles

### Learned models

#### B0) Passive constraint-context baseline

`B0` is the prior-work-style learned baseline. It uses passive constraint
context and historical edit imitation. It is weaker on Micro-F1 than the
factorized model but surprisingly strong on some symbolic safety metrics.

#### A1) Factorized imitation model

`A1` isolates the representation effect. It uses executable factors and
auxiliary factor supervision, but no explicit candidate-level safety objective.

Current role:

- main positive representation result;
- improves historical edit imitation;
- does not improve aggregate symbolic safety over `B0`.

#### M1C) Chooser-style safety attempt

`M1C` tests whether a symbolic candidate chooser can improve the safety side of
the trade-off while retaining the factorized proposal model.

Current role:

- useful safety-objective attempt;
- not the main positive result;
- current aggregate results do not show a broad safety win.

#### M1D) Direct safety-loss attempt

`M1D` adds direct primary and secondary safety pressure to the factorized model.

Current role:

- closest practical compromise among factorized variants;
- improves some safety trade-offs relative to `A1`;
- not a decisive aggregate symbolic-safety win over `B0`.

#### G0) Global/local satisfaction endpoint

`G0` optimizes local post-edit satisfaction over candidate repairs.

Current role:

- intended diagnostic endpoint, not a practical repair system;
- included as the learned satisfaction-reranking endpoint; and
- demonstrates that a satisfaction objective can favor evidence deletion even
  when candidates are produced without injecting the historical edit.

### Heuristic baselines

#### H1) DeleteFocusBaseline

Always delete the violating focus triple. This is the key control for detecting
whether a satisfaction objective is rewarding evidence removal rather than
meaningful repair.

#### H2) Conservative add heuristic

Adds mirrored triples for inverse/symmetric cases. It is conservative, low
disruption, and weak on broad repair intent.

#### H3) ConstraintFamilyMajority

Memorizes the most common historical repair pattern by coarse constraint family.
It tests how much behavior is explained by repair priors without learned
factorized reasoning.

---

## 8) Evaluation Metrics

The evaluation applies each predicted edit to the pre-repair local graph and
then rechecks symbolic constraints.

The canonical definitions are specified in
[the evaluation protocol](08_corrected_evaluation_protocol.md). In
particular, pre/post states are reconstructed from interim rows, all aggregates
carry explicit numerators and denominators, PFR replaces the historical
action-derived primary diagnostic, and EPPF/base-deletion metrics make
non-vacuity explicit. The GFR wording below is retained only to interpret prior
outputs; current reports use `local_satisfaction` instead.

### Historical fidelity

Micro-F1 compares predicted add/delete edit slots against the historical curator
edit. It measures imitation, not repair quality.

### Primary Fix

Primary Fix checks whether the originally violated constraint becomes satisfied:

$$
\operatorname{PrimaryFix}
= S_{c^*}(G^\Delta)
$$

### GFR / Local Satisfaction

The old label `GFR` should be read as local post-edit satisfaction over
`C_local`, not as global Wikidata consistency:

$$
\operatorname{LocalSatisfaction}
=
\frac{1}{|C_{\mathrm{local}}|}
\sum_{c \in C_{\mathrm{local}}}
S_c(G^\Delta)
$$

This metric is useful but dangerous as a standalone objective because deleting
the focus statement can raise satisfaction without producing a semantically
useful repair.

### Secondary Improvement Rate

`SIR` is higher when the edit fixes secondary constraints that were violated
before the edit:

$$
\operatorname{SIR}
=
\frac{
|\{c \ne c^*: S_c(G^-) = 0 \land S_c(G^\Delta) = 1\}|
}{
|\{c \ne c^*: S_c(G^-) = 0\}|
}
$$

### Secondary Regression Rate

`SRR` is lower when the edit avoids breaking secondary constraints that were
satisfied before the edit:

$$
\operatorname{SRR}
=
\frac{
|\{c \ne c^*: S_c(G^-) = 1 \land S_c(G^\Delta) = 0\}|
}{
|\{c \ne c^*: S_c(G^-) = 1\}|
}
$$

### Disruption

Disruption approximates edit invasiveness. It is useful for detecting
over-editing, but it does not by itself measure semantic usefulness.

### Deferred non-vacuity metric

The current results reveal that evidence preservation needs an explicit metric.
Future reporting should distinguish repairs that improve constraint state from
edits that improve metrics by deleting the focus evidence.

---

## 9) Prior schema-v1 empirical pattern

The values in this section predate the current state reconstruction. They are
retained only as historical project notes and must not be copied into the paper.
The current comparison and reader-facing system names are defined in the
[models and evaluation matrix](../docs-technical/00_models_and_evaluation_matrix.md).

The current paper-facing result table should be narrated as evidence for a
fidelity-safety gap, not as a single leaderboard.

| Model | Primary Fix | Micro-F1 | GFR / Local Satisfaction | SIR | SRR | Disruption | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B0` | 0.8603 | 0.5434 | 0.8705 | 0.1925 | 0.0097 | 1.0036 | Lower learned fidelity, surprisingly strong symbolic-safety baseline. |
| `A1` | 0.8589 | 0.6785 | 0.8638 | 0.1397 | 0.0128 | 1.0330 | Main representation win: better imitation, not safer symbolic repair. |
| `M1C` | 0.8155 | 0.6198 | 0.8629 | 0.1158 | 0.0132 | 1.0663 | Chooser-style safety objective does not improve the aggregate trade-off. |
| `M1D` | 0.8588 | 0.6772 | 0.8661 | 0.1396 | 0.0103 | 1.0274 | Closest practical compromise, but not a decisive aggregate safety win over `B0`. |
| `G0` | 0.7174 | 0.2804 | 0.9098 | 0.2441 | 0.0091 | 1.0000 | Best local satisfaction, but degenerates toward delete-focus behavior. |
| `H1` | 0.7174 | 0.2804 | 0.9098 | 0.2441 | 0.0091 | 1.0000 | Delete-focus control; aggregate match with `G0` exposes the degeneracy. |
| `H2` | 0.5157 | 0.0503 | 0.8544 | 0.0001 | 0.0043 | 0.1047 | Conservative and low disruption, but weak repair intent and fidelity. |
| `H3` | 0.7731 | 0.2654 | 0.8689 | 0.1917 | 0.0103 | 1.0072 | Coarse repair-prior baseline; much weaker fidelity than learned factorized models. |

### Main findings

1. `A1` improves Micro-F1 from `0.5434` to `0.6785`, validating executable
   factors as a representation for historical repair imitation.
2. `A1` is worse than `B0` on local satisfaction, SIR, SRR, and disruption,
   showing that imitation is not safety.
3. `M1D` is the strongest practical factorized compromise, but the aggregate
   evidence is not strong enough to claim broad safety dominance.
4. The historical schema-v1 `G0`/`H1` match motivated the
   satisfaction-by-deletion analysis, but the learned `G0` claim is not part of
   the corrected evidence. Corrected `H1` remains a deterministic counterexample:
   local satisfaction can be high after removing the focus evidence.

---

## 10) H2 Diagnostics: What the Factors Actually Do

### Factor semantics are learned

Factor heads achieve high weighted F1 and strong AUROC on pre-repair and
post-gold satisfaction labels. This supports the claim that executable factors
learn meaningful local constraint signals.

However, many factor labels are imbalanced. The paper should report AUROC,
AUPRC, and calibration where possible, not only weighted F1.

### Factor pressure is causally used

Inference-time masking shows large drops in Micro-F1 and Primary Fix when factor
pressure is removed. The factor pathway is not decorative; it materially affects
predictions.

### Secondary pressure is weak

Secondary-only pressure masking changes few predictions and produces small
metric shifts. This explains the main result: the model uses constraint pressure,
but the learned pressure is mostly aligned with historical imitation and primary
repair behavior rather than secondary no-regression.

---

## 11) Revised Research Questions

### RQ1: Do executable factors improve historical repair modeling?

Expected answer from current results: yes.

Evidence:

- `A1` improves Micro-F1 over `B0`.
- factor heads learn satisfaction signals;
- pressure masking shows factor messages affect decisions.

### RQ2: Does better imitation imply safer post-edit graphs?

Expected answer from current results: no.

Evidence:

- `A1` improves fidelity but worsens aggregate symbolic safety metrics relative
  to `B0`.

### RQ3: Can local safety objectives move the trade-off?

Expected answer from current results: partially, but not conclusively.

Evidence:

- `M1D` is the best practical factorized compromise;
- `M1C` does not recover a broad aggregate safety win;
- sliced results and candidate-oracle analysis are needed to determine where
  the bottleneck lies.

### RQ4: What happens when local satisfaction is optimized directly?

Expected answer from corrected results: the metrics admit a degenerate endpoint,
but the current repository cannot establish that a learned objective converges
to it.

Evidence:

- corrected `H1` deletes every base statement, has EPPF zero, and nevertheless
  receives high Primary Fix Rate and Local Satisfaction;
- the retained `G0` checkpoint is missing and its predictions fail label-blind
  candidate-membership validation; and
- high local satisfaction is therefore not equivalent to meaningful repair,
  while a learned-collapse claim remains deferred until `G0` is retrained.

---

## 12) Recommended Paper Presentation

### Introduction

Frame the project around the fidelity-safety gap:

- historical edit logs are useful but incomplete supervision;
- symbolic satisfaction is useful but can be vacuous;
- repair systems need multi-axis evaluation.

### Method

Present executable constraint factors as a representation that makes local
constraint state available to the model and supports diagnostic pressure tests.

### Evaluation

Separate:

- historical fidelity;
- primary fix;
- local satisfaction;
- secondary improvement;
- secondary regression;
- disruption;
- non-vacuous evidence preservation.

### Results

Lead with the trade-off:

- factors improve imitation;
- imitation is not safety;
- local satisfaction can be degenerate;
- secondary no-regression remains an open decision-level problem.

### Discussion

The project is not a failed safe-repair system. It is evidence that KG repair
needs separate objectives and metrics for imitation, symbolic safety, and
non-vacuity.

---

## 13) Future Work and Success Criteria

### Near-term analyses

1. Candidate-oracle analysis: determine whether safe candidates exist but are
   not selected, or whether candidate generation is the bottleneck.
2. Density and family slices: identify regimes where factorized safety pressure
   helps or fails.
3. Calibration of factor semantics: report AUROC, AUPRC, and calibration for
   imbalanced constraint families.
4. Non-vacuity metrics: explicitly distinguish meaningful repair from deleting
   the problem away.

### Revised success criteria

The project succeeds if it establishes:

1. executable factors improve historical repair imitation;
2. factor semantics and pressure are learned and causally used;
3. historical fidelity and symbolic safety are empirically separable;
4. local satisfaction alone is shown to be an unsafe optimization target;
5. future safe-repair objectives are motivated by concrete failure modes rather
   than by aggregate leaderboard claims.
