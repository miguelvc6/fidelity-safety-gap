# Validator-v3 Verification Record

This record separates reproduced audit evidence, corrected production tests,
and work intentionally deferred to full regeneration.

## Findings

| Finding | Status | Production location | Regression evidence |
| --- | --- | --- | --- |
| Local P279 additions/deletions ignored | reproduced and fixed | `class_hierarchy.py`, `_type_relation` | state overlay, tombstone, alternate path, instance/value/either, cycle, raw-intermediate tests |
| Filtered factor index selects raw definition | reproduced and fixed | `constraint_identity.py` and every row/graph/evaluator consumer | audit integration plus filtered/reordered/batched and pipeline smoke tests |
| Relevant unresolved edits disappear | reproduced and fixed | `evidence_state.py`, detailed validator | distinct competitor, new secondary anchor, unrelated edit, definite-violation tests |
| Parser certifies incomplete ancestry | reproduced and fixed | hierarchy downloader/loader | legacy numeric IDs, malformed IDs, `somevalue`, `novalue`, mixed parent, cache migration tests |
| Exempt primary becomes satisfied after deletion | reproduced and fixed | shared validator | every occurrence family, mixed anchors, deletion, reinsertion tests |
| Distinct exemption removes competitor evidence | audit expectation corrected | distinct owner map | anchor-only reference contract test |
| Item accepted as constrained property | reproduced and fixed | shared parser | resolvable `Q`-ID rejection test |
| Conditional loss collapses below epsilon | reproduced and fixed | production loss helpers | extreme logits, BF16, finite gradient, eligible/ineligible tests |
| Candidate primary term masks post unknowns | deliberate objective change | Candidate--C/DP train and validation paths | proven-fix monotonicity and pre-eligibility tests |
| Final metric adapter erases partial operations | reproduced and fixed | `09_eval.py`, `RepairSample`, paper/H2 evaluation | raw-slot callback, artifact round-trip, supplied full-entry-point test |
| Unresolved deletion replayed after successful addition | reproduced and fixed | canonical evidence events and detailed validator | five-family ordered-world regressions and labeler/candidate parity |
| Partial-edit relevance uses only the property | reproduced and fixed | `_edit_may_affect` | primary single/one-of controls plus conservative distinct/secondary controls |
| Trigger/support roles treated as exclusive | reproduced and fixed | `_edit_may_affect` role union | unknown-predicate inverse/value-requires and self-inverse callback regressions |

No applicable finding was left unresolved. The distinct-values disagreement is
not implemented as the original audit expectation: the checked anchor is
exempted, while another exempt subject's statement remains competing evidence,
matching the separately inspected reference implementation revision cited in
the semantics guide.

The independent review of commit `e5cd092` subsequently reproduced ten cases
across the final three rows above, plus one full metric-callback failure. The
fix retains raw prediction slots for symbolic evaluation, records requested
operations in source order, and replays uncertain operations from the pre-edit
state. These are corrections to the not-yet-generated v3 artifact contract;
no v3 labels, graphs, checkpoints, or results existed to migrate.

The follow-up review of commit `0db57f5` confirmed all 100 carried-forward
cases, then found three role-dependency counterexamples. An unknown predicate
could reintroduce a deleted primary occurrence for inverse or value-requires,
and a self-inverse predicate was classified only as a trigger rather than also
as reciprocal support. The corrected dependency check takes the union of all
compatible semantic roles. Repository regressions cover the primitive
validator and the real candidate/final-metric callback, including the rule that
an unresolved result receives no primary-fix credit.

## Current primary historical diagnostic

The resumable downloader rebuilt all 6,382 historical nodes by retrieving raw
content for the retained exact revision identities. Every parser-v2 cache entry
retains its source page; revision ID, timestamp, and content checksum were
verified during migration. The direct edge topology remains equal to v1 (8,732
edges), while parser v2 marks 55 direct
adjacencies and 434 transitive closures incomplete instead of certifying them.

The primary-only streaming diagnostic evaluated all 955,405 immutable sampled
rows with the corrected working tree and parser-v2 hierarchy. `Pass / all`
uses every primary instance as its denominator; `pass / checkable` excludes
three-valued `unknown` outcomes. These are descriptive historical-edit
diagnostics, not validator acceptance targets.

| Constraint family | Primary rows | Pre pass / all | Pre pass / checkable | Post pass / all | Post pass / checkable |
|---|---:|---:|---:|---:|---:|
| conflictWith | 100,612 | 1.23% | 1.23% | 93.58% | 93.59% |
| distinct | 228,208 | 4.64% | 4.64% | 99.51% | 99.52% |
| inverse | 70,361 | 25.78% | 25.82% | 97.51% | 98.34% |
| itemRequiresStatement | 99,999 | 19.77% | 19.77% | 99.48% | 99.49% |
| oneOf | 6,855 | 0.00% | 0.00% | 23.15% | 97.00% |
| single | 119,723 | 54.33% | 71.65% | 65.29% | 86.10% |
| symmetric | 29,642 | 88.93% | 93.62% | 90.94% | 99.94% |
| type | 100,003 | 33.83% | 34.43% | 95.86% | 96.20% |
| valueRequiresStatement | 100,002 | 37.21% | 45.12% | 77.75% | 99.10% |
| valueType | 100,000 | 48.11% | 51.39% | 89.15% | 98.17% |

The machine-readable report is
`models/paper_diagnostics/primary_historical_satisfaction_v3.json`; it includes
exact counts, split-level results, all nine pre/post transitions, and input/code
identities. The report SHA-256 is
`39bffd7cad9005bf45a1277dcb69e302f3942fb028e75b565a986d9cdb73370f`.
The hierarchy is
`data/static/wikidata-p279-2018-07-01.v2.json`, SHA-256
`2c331f662799715d5a2fda042061fa88c0922801c5dfa29601c029244ce1b1f4`.

## Executed commands

Before correction, the audit suites against live production code reproduced:

- first audit core/parser/integration selection: 17 failed, 11 passed;
- independent audit review: 14 failed, 27 passed.

After correction:

```bash
AUDIT_DIR=audit/extracted/fidelity_safety_gap_validator_audit/validator_audit
PYTHONPATH="$PWD/src" uv run python -m pytest -q \
  "$AUDIT_DIR/test_core_regressions.py" \
  "$AUDIT_DIR/test_hierarchy_parser_regressions.py" \
  "$AUDIT_DIR/integration_only/test_candidate_factor_identity.py"
# 28 passed

AUDIT_DIR=audit/extracted/fidelity_safety_gap_validator_audit_2d6f212/validator_audit
PYTHONPATH="$PWD/src" uv run python -m pytest -q "$AUDIT_DIR/tests"
# 41 passed

PYTHONPATH="$PWD/src" uv run python tests/smoke_validator_v2_pipeline.py
# validator-v3 smoke pipeline: PASS (legacy filename retained for automation)

AUDIT_DIR=audit/extracted/fidelity_safety_gap_v3_review_e5cd092/fsg_v3_audit
FSG_REPO="$PWD" PYTHONPATH="$PWD/src" uv run python -m pytest -q \
  "$AUDIT_DIR/tests" --tb=short
# before: 10 failed, 90 passed; after: 100 passed

FSG_REPO="$PWD" PYTHONPATH="$PWD/src" uv run python -m pytest -q \
  "$AUDIT_DIR/integration_only" --tb=short
# before: 1 failed; after: 1 passed

uv run python -m pytest -q -ra
# 191 passed, 3 third-party deprecation warnings, 0 skipped

uv run python -m compileall -q src scripts tests
# exit 0
```

The smoke uses 12 copied rows in a temporary directory. It runs the production
labeler and graph builder, round-trips graphs through PyG batching, and compares
stored labels, IDs, primary positions, single-candidate evaluation,
candidate-batch evaluation, and the fields consumed by the primary loss. It
asserts that at least one row's primary position shifts when unsupported raw
definitions are filtered.

Ruff is not installed in the locked environment (`uv run ruff` fails to
spawn), so no Ruff result is claimed. The three pytest warnings come from
PyTorch Geometric's deprecated distributed import and PyTorch's deprecated
`torch.jit.script`; they are not validator failures. The zero-skip result is
expected: the approved test environment exposed one NVIDIA A30 and ran the
three CUDA-only compact-factor tests. A separate sandboxed `nvidia-smi` failure
was not evidence that those tests were skipped.

The immutable-input verification recomputed size and SHA-256 for all 38 paths
recorded in
`deprecated/pre-validator-rewrite-2026-09-04/ARCHIVE_MANIFEST.json`; all 38
matched. `sha256sum --check release/zenodo/v1.0.0/SHA256SUMS` also passed for
all eight release entries. The downloader's read-only `--seeds-only` check
reported the expected 3,941 fixed hierarchy seeds.

Before the hierarchy rebuild, both final readiness commands were invoked as
negative pre-regeneration checks. Each exited 1 as required: paper readiness
stopped at the deliberately removed stale B0 evaluation, and suite validation
stopped because the v2 hierarchy artifact was absent. Neither accepted the old
generation.

## Bounded real-row identity observation

A read-only prefix scan of the existing stale v2 labeled files inspected 1,000
rows in each of train, validation, and test. The primary ID was never missing or
duplicated in either vector. Its raw `local_constraint_ids` position differed
from its filtered `factor_constraint_ids` position in 2,924 of 3,000 rows
(train 975, validation 976, test 973). This quantifies exposure to positional
transfer, not the number of wrong historical predictions and not a v3 label
result. The v3 full-scan diagnostic must be run only after labels regenerate.

## Bounded edit-edge-case observation

A read-only prefix scan inspected 1,000 unchanged historical edits from each
split. None contained a partial operation or an unresolved deletion followed
by a successful addition. This is only a 3,000-row gold-edit observation.

A separate scan inspected the first 1,000 predictions from each of ten archived
paper-system artifacts (10,000 rows total). It found 56 partial additions and
four partial deletions: A1 had 19/3, B0 had 28/1, the definition-majority
baseline had 6/0, and M1D had 3/0; the other six artifacts had none in their
prefixes. One A1 prediction contained an unresolved partial deletion followed
by a successful addition on a different triple; no identical-triple overlap
occurred in the bounded prefixes. These stale v2 predictions are prevalence
evidence only, not v3 metrics. Their partial-operation frequency confirms that
retaining raw slots in final symbolic evaluation is operationally relevant.

## Deferred checks

The following require the intentionally deferred full run and are not claimed
as completed: full label and graph regeneration, passive-payload checksum
equality, GPU training/evaluation, schema-v3 result completion for all nine
paper systems, paper/PDF rebuild, and paper-to-artifact numerical equality. The
readiness scripts now fail closed until those artifacts share the exact v3
contract identities.
