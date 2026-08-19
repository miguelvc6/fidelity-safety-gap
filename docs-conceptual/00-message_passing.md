# Executable Constraint Factors: Message Passing and Constraint Execution

This document specifies the **conceptual message-passing equations** and computational semantics of **constraint factors as executable subprograms**, expanding the informal description into a precise, paper-ready formulation.

It is the architecture-specification companion to
[00-constraint_factors.md](00-constraint_factors.md). Implementation status and
configuration details belong in the
[technical documentation](../docs-technical/README.md).

The goal is to make explicit:
- how constraint semantics are computed,
- how constraint violations generate *pressure*,
- how this pressure interacts with GNN message passing,
- and how repair decisions emerge without hardcoded heuristics.

This conceptual specification does not assume that every decision-time safety quantity must be produced by a fully neural post-edit rollout. The main research plan remains compatible with symbolic candidate evaluation at repair-selection time, as long as executable factors shape the proposal model itself.

---

## 1. Graph Structure and Notation

We work with a **heterogeneous factor graph** per violation instance.

### Node sets
- **Variable nodes** $v \in \mathcal{V}$
  - entities, predicates, literals, role-specific nodes
- **Constraint factor nodes** $c \in \mathcal{C}$
  - each corresponds to a *constraint instance*
  - each has a constraint type $t(c) \in \mathcal{T}$

### Scope of a constraint
Each factor node $c$ has a scope:
$$
\mathrm{scope}(c) \subseteq \mathcal{V}
$$
representing the variables involved in that constraint.

Examples:
- `conflictWith(p,q)` → scope = {predicate node $p$, predicate node $q$, optional subject node $s$}
- `single(p)` → scope = {subject $s$, predicate $p$, all value nodes $o_i$}

---

## 2. Variable-to-Variable Message Passing (Base GNN)

Let $h_v^{(k)} \in \mathbb{R}^d$ be the embedding of variable node $v$ at layer $k$.

We use a standard GNN backbone (e.g., GIN over a multi-relational graph):

$$
\tilde{h}_v^{(k+1)} =
\mathrm{GNN}\Big(
h_v^{(k)},
\big\{ (h_u^{(k)}, e_{u\to v}) : u \in \mathcal{N}(v) \big\}
\Big)
$$

This step captures **structural and semantic context** from the data graph alone, without constraints.

---

## 3. Constraint Execution: Violation / Satisfaction Computation

Constraint factors are **executable operators**, not passive nodes.

### 3.1 Type-specific constraint function

Each constraint type $t \in \mathcal{T}$ has a shared neural function:
$$
f_t : \mathbb{R}^{|\mathrm{scope}(c)| \cdot d} \rightarrow \mathbb{R}
$$

All constraint instances of the same type share parameters.

---

### 3.2 Example: `conflictWith` constraint

For a `conflictWith(p,q)` constraint with optional subject $s$, define:

$$
z_c^{(k)} = 
\big[
h_p^{(k)} \;\Vert\; h_q^{(k)} \;\Vert\; h_s^{(k)}
\big]
$$

Violation score:
$$
\mathrm{viol}_c^{(k)} =
\sigma\big( W_{conflict} \cdot z_c^{(k)} \big)
$$

where:
- $\sigma$ is a sigmoid,
- $\mathrm{viol}_c \in [0,1]$.

**Interpretation**
- $\mathrm{viol}_c \approx 0$: configuration compatible
- $\mathrm{viol}_c \approx 1$: strong violation

This learned function **replaces**:
- heuristic rule matching,
- handcrafted fix-probability losses.

---

### 3.3 Satisfaction score

We define satisfaction as:
$$
\hat{s}_c^{(k)} = 1 - \mathrm{viol}_c^{(k)}
$$

This is the quantity used in:
- auxiliary satisfaction losses,
- secondary no-regression constraints,
- global fix evaluation.

---

## 4. Constraint-to-Variable Feedback (Repair Pressure)

Constraint execution alone is insufficient; constraints must **influence representations**.

Each factor emits **directed, role-conditioned pressure messages** to its scoped variables.

---

### 4.1 Pressure generation

For each $v \in \mathrm{scope}(c)$:

$$
m_{c \rightarrow v}^{(k)} =
g_{t(c), r(v,c)}
\big(
z_c^{(k)},\;
h_v^{(k)},\;
\mathrm{viol}_c^{(k)}
\big)
$$

where:
- $r(v,c)$ is the *role* of variable $v$ in constraint $c$,
- $g_{t,r}$ is a small MLP, shared per (constraint type, role).

---

### 4.2 Simplified example (conflictWith)

For predicates $p, q$:

$$
m_{c \rightarrow p}^{(k)} =
- \alpha \cdot \mathrm{viol}_c^{(k)} \cdot h_p^{(k)}
$$

$$
m_{c \rightarrow q}^{(k)} =
- \alpha \cdot \mathrm{viol}_c^{(k)} \cdot h_q^{(k)}
$$

Optionally, for subject $s$:

$$
m_{c \rightarrow s}^{(k)} =
- \alpha_s \cdot \mathrm{viol}_c^{(k)} \cdot h_s^{(k)}
$$

---

### 4.3 Interpretation of pressure

- High violation → strong negative feedback
- Pressure pushes embeddings **away from a locally stable manifold**
- Variables involved in violations become *less compatible* with the current configuration

This is **not** symbolic deletion:
- no rule says “delete focus triple”
- instead, representations encode *tension*

---

## 5. Variable State Update with Constraint Pressure

The final variable update at layer $k+1$ is:

$$
h_v^{(k+1)} =
\tilde{h}_v^{(k+1)}
+
\sum_{c : v \in \mathrm{scope}(c)}
m_{c \rightarrow v}^{(k)}
$$

Thus:
- standard GNN aggregation builds context,
- constraint pressure reshapes representations toward consistency.

Multiple constraints can simultaneously act on the same variable, allowing:
- constraint interaction,
- trade-offs,
- partial satisfaction.

---

## 6. Emergent Repair Behavior

### 6.1 No hardcoded repair logic

There is:
- no explicit “delete focus” rule,
- no enumerated fix patterns inside the model.

Instead:
- constraint pressure accumulates in embeddings,
- the decoder reads this tension.

---

### 6.2 Decoder intuition

Let $h_G$ be the pooled graph embedding.

High constraint pressure on:
- a predicate embedding → higher probability of deletion
- missing required structures → higher probability of addition

The decoder learns mappings like:
- “when pressure concentrates on this slot, delete”
- “when pressure indicates absence, add”

This is **learned**, not prescribed.

---

## 7. Multi-Constraint Interaction

Because multiple factors emit pressure:
- constraints can reinforce each other,
- or partially cancel out.

Example:
- primary constraint pushes for deletion,
- secondary constraint resists deletion due to value requirements,
- final decision balances both.

This interaction is impossible in:
- flattened graphs with passive constraint nodes,
- post-hoc constraint losses.

---

## 8. Why This Is Fundamentally Different from Flattened Graphs

| Aspect | Flattened graph | Executable constraint factors |
|---|---|---|
| Constraint node | Passive context | Active operator |
| Semantics | Implicit correlations | Explicit learned function |
| Execution | Outside GNN | Inside message passing |
| Interaction | Single constraint | Multi-constraint superposition |
| Repair logic | Pattern imitation | Pressure-driven emergence |

---

## 9. Summary

Executable constraint factors are intended to turn constraints from passive context into active local operators:

- each factor reads the variables in its scope,
- computes a type-specific compatibility or violation signal,
- emits role-conditioned pressure back into those variables,
- and thereby influences repair decisions from inside the representation-learning process.

This is the conceptual mechanism behind the broader executable-factor repair direction. The research claim is not merely that constraints are present in the graph, but that they are executed as typed local programs whose feedback can shape the repair trajectory.

- Constraints are modeled as **typed neural operators**.
- Each constraint instance executes a validation subprogram.
- Violations generate **continuous pressure**, not discrete rules.
- Repair decisions emerge from representation dynamics.
- This enables principled modeling of soft, interacting constraints under curator intent.
- A symbolic candidate evaluator can still remain the paper's decision-time safety layer without weakening the core executable-factor contribution.

This formulation makes constraints **first-class computational objects**, not annotations.
