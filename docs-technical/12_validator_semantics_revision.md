# Validator Semantics v3 and Regeneration

Validator semantics v3 corrects five defects found in the v2 audit and changes
the Candidate--C and Candidate--DP primary objective. One parser, evidence-state
builder, three-valued evaluator, and primary-identity contract are shared by
label generation, candidate training, reranking, diagnostics, and evaluation.
The outcomes remain `satisfied`, `violated`, and `unknown`.

The neural architectures, six-slot edit language, splits, encoder vocabulary,
candidate budgets, and configured coefficients are unchanged. Historical
post-edit satisfaction remains a diagnostic; it is neither ground truth nor an
acceptance threshold.

## Versioned contracts

Every new validator-dependent artifact must carry all of these identities:

| Contract | Version |
| --- | ---: |
| validator semantics | 3 |
| hierarchy artifact schema | 2 |
| hierarchy parser | 2 |
| hierarchy cache schema | 2 |
| effective hierarchy policy | 1 |
| candidate objective | 2 |
| prediction/evaluation schema | 3 |

`src/modules/semantic_versions.py` and
`src/modules/semantics_provenance.py` are the canonical sources. Labels,
factorized graph build contracts, validator-dependent checkpoints, prediction
manifests, replay, diagnostics, readiness, and release preparation reject a
missing or unequal contract. Changing a legacy artifact's version field does
not make it compatible.

## Shared constraint and anchor semantics

`src/modules/constraint_checkers.py` parses Wikidata parameters as follows:

| Parameter | Meaning |
| --- | --- |
| `P2306` | conflicting, required, or inverse property |
| `P2305` | optional restricted value, or permitted one-of value |
| `P2308` | permitted class |
| `P2309=Q21503252` | instance: `P31/P279*` |
| `P2309=Q21514624` | subclass: `P279*` |
| `P2309=Q30208840` | union of instance and subclass |
| `P2303` | exempt checked anchor |

The constrained property must have property-ID syntax before it is resolved;
a resolvable item such as `Q1` is invalid. Missing, malformed, repeated-singular,
unsupported-scope, or unrepresentable mandatory parameters produce `unknown`.
An unrepresentable `P2305` value invalidates the definition rather than silently
broadening it. `P4155` makes single- and distinct-values definitions unknown
because the benchmark does not retain the required qualifier separators.

Primary definitions bind to the focus subject and constrained property in the
edited state. Secondary definitions bind to every represented occurrence of
their constrained property, including a possible occurrence introduced by a
relevant unresolved addition. Multiple anchors aggregate as violated if any is
definitely violated, satisfied only if every applicable anchor is definitely
satisfied, and unknown otherwise. A definitely absent non-exempt primary
trigger remains vacuously satisfied; an exempt primary is removed before that
shortcut and therefore remains unknown. The existing secondary no-anchor
contract is also unknown.

For distinct-values, `P2303` exempts a subject as an anchor being checked but
does not remove its statements from the competing-owner evidence. This matches
the inspected WikibaseQualityConstraints behavior at commit
[`e58668f5d76db19681c1f38e6a2f6d818da35faa`](https://github.com/wikimedia/mediawiki-extensions-WikibaseQualityConstraints/tree/e58668f5d76db19681c1f38e6a2f6d818da35faa):
[`DelegatingConstraintChecker.php`](https://github.com/wikimedia/mediawiki-extensions-WikibaseQualityConstraints/blob/e58668f5d76db19681c1f38e6a2f6d818da35faa/src/ConstraintCheck/DelegatingConstraintChecker.php)
checks the current entity's exception before
[`UniqueValueChecker.php`](https://github.com/wikimedia/mediawiki-extensions-WikibaseQualityConstraints/blob/e58668f5d76db19681c1f38e6a2f6d818da35faa/src/ConstraintCheck/Checker/UniqueValueChecker.php)
performs the cross-entity lookup. This records current reference behavior, not a
claim that the same source revision was deployed in July 2018.

## Effective P279 hierarchy

The downloaded 1 July 2018 hierarchy stays immutable. Each evaluation traverses
an effective adjacency made from:

1. known direct edges in the fixed artifact;
2. represented local `P279` statements in the current evidence state; and
3. successful deletions as tombstones, which take precedence over an identical
   background edge. A delete followed by the same addition clears the tombstone.

Other background edges remain available, so deleting one proof does not remove
an alternate path. A known surviving path is enough for satisfaction. A failed
search is violated only if every visited node's direct adjacency is complete;
otherwise it is unknown. Explicitly complete empty local adjacency is therefore
different from an entity that is absent from local evidence. Traversal handles
cycles and raw, unencoded intermediate classes without extending the neural
vocabulary. The background object is never mutated or copied per candidate,
and no state-specific closure cache is shared across candidates. Hierarchy
nodes remain symbolic evidence and are not neural graph inputs.

## Unresolved edit contract

`src/modules/evidence_state.py` still applies edits within bounded evidence, in
delete-then-add order. It now records each unapplied requested operation with
its kind, subject, property, object, and reason, separately from an intentionally
absent all-zero operation. Successful additions and deletions are also retained
for effective-hierarchy precedence.

The validator considers only unresolved edits relevant to the selected
definition. For the benchmark's at-most-one-add/one-delete language it evaluates
the projected state and every success/failure combination of relevant complete
operations. It returns a Boolean outcome only when all worlds agree. This makes
an unresolved competing distinct owner or new secondary one-of anchor unknown,
while preserving an unaffected definite violation and ignoring unrelated
failed edits. Legacy pair-only `missing_edits` metadata fails closed unless an
unaffected violation can be established without the uncertain pair. Detailed
results expose applicability and unknown reasons; labeled rows persist edit
applicability and unresolved-operation JSON.

## Primary identity contract

`src/modules/constraint_identity.py` defines the only list-selection and
primary-resolution policy. An explicitly supplied list wins; otherwise the
filtered `factor_constraint_ids` vector wins; only then may a configured raw
local vector be used. `row.constraint_id` must occur exactly once in that exact
vector. A supplied `primary_factor_index` must point to the same ID. Missing,
duplicate, out-of-range, and mismatched identities raise actionable errors.

The labeler resolves the primary both before and after supported-family
filtering. Graph construction, PyG batching, training row loaders, single-edit
evaluation, candidate-batch evaluation, retained loss helpers, reranking, and
final metrics preserve the filtered factor IDs and use the same resolver.
Secondary definitions exclude the primary by ID, not by an index borrowed from
another vector.

## Candidate objectives

Candidate--C and Candidate--DP now use the deliberate proven-fix objective. For
one instance, pre-edit eligibility `e` is fixed across candidates:

```text
e = 1 iff the pre-edit primary outcome is violated
q_j = 1 iff candidate j makes the post-edit primary satisfied and checkable
L_primary = e * sum_j softmax(score)_j * (1 - q_j)
```

A post-edit unknown costs one on an eligible instance because it is not a
proven fix; this does not relabel the validator outcome as violated. A pre-edit
unknown or satisfied primary contributes zero primary-repair loss while the row
still contributes imitation and eligible auxiliary losses. The same production
helper is used in training and validation. Candidate--SR remains deliberately
satisfaction-only.

Auxiliary satisfaction BCE continues to mask unknown labels. Secondary
common-support penalties and Candidate--C's historical-relative SRR budget are
unchanged. Conditional expectations that remain are computed with float32
softmax directly over eligible logits; an empty eligible set returns a finite,
differentiable zero. This removes the former epsilon-denominator collapse at
tiny globally normalized mass.

## Hierarchy construction and migration

`scripts/build_historical_class_hierarchy.py` starts from the fixed 3,941 seed
classes and recursively fetches the latest revision at or before
`2018-07-01T00:00:00Z`. It uses `rvstart`, `rvdir=older`, bounded retry/backoff,
`maxlag`, a descriptive user agent, and per-entity resumable caching. Deprecated
claims are ignored; preferred claims replace normal claims when present. There
is no present-day fallback.

Parser v2 accepts a validated legacy item value containing positive integer
`numeric-id`, rejects conflicting explicit/numeric IDs and wrong entity types,
and treats snaks distinctly. A valid `value` parent is retained; malformed
values and `somevalue` make direct adjacency incomplete; `novalue` contributes
no concrete parent under the truthy projection without inventing uncertainty.
Valid parents survive alongside unknown parents. Retrieval success, direct
adjacency completeness, and transitive closure completeness are separate.

The new cache is `data/interim/hierarchy_cache_2018-07-01.v2/` and the artifact
is `data/static/wikidata-p279-2018-07-01.v2.json`. A stale cache entry is reparsed
only from its retained raw revision page; otherwise it is refetched. The old
entry is preserved as `*.pre-parser-v2.json`. Schema-v1 artifacts are rejected.
The cutoff is fixed external evidence, not a reconstruction at each benchmark
edit's individual timestamp.

## Non-destructive regeneration procedure

The v2 labels, factorized graphs, validator-derived evaluations, and all
factor-dependent checkpoints are stale. The currently running factor model was
stopped, its run directory was removed, and stale Direct--Passive evaluation
files were removed; the Direct--Passive checkpoint itself is validator-
independent and retained. Do not restart training until the hierarchy-v2 artifact below
has been rebuilt and every prerequisite passes its contract checks.

The full procedure was prepared but not run as part of this correction:

```bash
# 1. Inventory/archive the stale generation without overwriting released files.
uv run python scripts/archive_generation.py

# 2. Verify seeds, then build parser-v2/cache-v2 hierarchy evidence.
uv run python scripts/build_historical_class_hierarchy.py --seeds-only
uv run python scripts/build_historical_class_hierarchy.py

# 3. Regenerate labels.
uv run python src/05_constraint_labeler.py \
  --dataset full_strat1m --registry-dataset full --min-occurrence 100 \
  --constraint-scope local --factor-family-policy supported_only \
  --hierarchy data/static/wikidata-p279-2018-07-01.v2.json

# 4. Rebuild factorized graphs and rebuild/passively checksum the passive suite.
uv run python src/06_graph.py \
  --dataset full_strat1m --registry-dataset full --min-occurrence 100 \
  --encoding node_id --constraint-representation factorized \
  --constraint-scope local --shard-size 200000 --use-torch-save \
  --persistence-profile research_safe --overwrite atomic \
  --hierarchy data/static/wikidata-p279-2018-07-01.v2.json
uv run python src/06_graph.py \
  --dataset full_strat1m --registry-dataset full --min-occurrence 100 \
  --encoding node_id --constraint-representation eswc_passive \
  --constraint-scope local --shard-size 10000 --use-torch-save \
  --persistence-profile research_safe --overwrite atomic --use-unlabeled-interim \
  --hierarchy data/static/wikidata-p279-2018-07-01.v2.json

# 5. Recreate canonical configs; inspect the dependency plan before execution.
uv run python scripts/make_experiment_configs.py \
  --variant full_strat1m_minocc100 --encoding node_id --models-root models
uv run python src/10_scheduler.py --paper-suite --dry-run

# 6. Separate authorized run: retrain/re-evaluate in dependency order.
uv run python src/10_scheduler.py --paper-suite
```

The scheduler reuses the checksummed Direct--Passive checkpoint, retrains
Direct--Factor, Candidate--C, Candidate--DP, then Candidate--SR, and finally
refits/evaluates the four baselines. It regenerates predictions, metrics,
historical strata, factor probes, masking, candidate, deletion, and readiness
diagnostics. Any v2 or cross-contract resume is rejected.

After new labels exist, generate the transition diagnostics first on a bounded
sample and then explicitly on the complete suite:

```bash
uv run python scripts/diagnose_validator_transitions.py \
  --max-rows 1000 \
  --output models/paper_diagnostics/validator_transitions_bounded.json
uv run python scripts/diagnose_validator_transitions.py \
  --full-scan \
  --output models/paper_diagnostics/validator_transitions.json
```

The report contains the 3-by-3 pre/post primary transitions overall and by
family, post-edit unknown reasons, primary-index mismatches, and single-value
failure witnesses. No transition or historical pass-rate threshold is imposed.

Final artifact gates, after the complete generation and paper update, are:

```bash
uv run python scripts/check_corrected_paper_readiness.py
uv run python scripts/validate_regenerated_suite.py
uv run python scripts/check_corrected_paper_readiness.py \
  --paper latex_paper/main.tex --verify-graph-checksums
uv run python scripts/validate_regenerated_suite.py --require-paper
```

The packager remains downstream of these gates and does not publish or
overwrite the existing Zenodo record automatically.

## Verification scope

The production suite, extracted audit suites, and bounded smoke procedure are
recorded in [Validator-v3 verification](13_validator_v3_verification.md). Full
hierarchy retrieval, full labels/graphs, GPU training, paper result replacement,
and release publication were intentionally not executed here.

Research decisions and manuscript obligations are separated into the
[validator-v3 evaluation and objective decisions](../docs-conceptual/11_validator_v3_evaluation_and_objective_decisions.md)
and [manuscript change notes](../docs-conceptual/12_validator_v3_manuscript_change_notes.md).
