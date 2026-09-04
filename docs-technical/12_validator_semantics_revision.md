# Validator Semantics v2 and Experiment Regeneration

Validator semantics v2 replaces the family-specific symbolic implementations
with one parser and one three-valued evaluator shared by label generation,
candidate objectives, reranking, diagnostics, and evaluation. The possible
outcomes are `satisfied`, `violated`, and `unknown`.

## Preserved generation

Tracked state is preserved by the annotated tag
`pre-validator-rewrite-2026-09-04`, which points to commit
`a9724319f2cac42ed495b1c4a8ead601edd81c25`. The existing Zenodo v1.0.0 files
remain unchanged and are verified against `release/zenodo/v1.0.0/SHA256SUMS`.
Generated models, labelled rows, graphs, and diagnostics from that generation
are stored locally below
`deprecated/pre-validator-rewrite-2026-09-04/ARCHIVE_MANIFEST.json`. The
manifest records retained-system mappings, sizes, and SHA-256 checksums.
`deprecated/` and `notebooks/` are intentionally ignored.

Create the archive once, before changing the validator implementation:

```bash
uv run python scripts/archive_generation.py
```

It does not move the raw corpus, constraint registry, encoder, or sampled
unlabelled benchmark rows.

## Shared semantic contract

The parser in `src/modules/constraint_checkers.py` assigns Wikidata parameter
roles as follows:

| Parameter | Meaning |
|---|---|
| `P2306` | conflicting, required, or inverse property |
| `P2305` | optional restricted value, or permitted value for one-of |
| `P2308` | permitted class |
| `P2309=Q21503252` | instance selector: `P31/P279*` |
| `P2309=Q21514624` | subclass selector: `P279*` |
| `P2309=Q30208840` | union of instance and subclass selectors |
| `P2303` | exempt anchor |

Mandatory missing, repeated-singular, malformed, unsupported-scope, and
unrepresentable parameters produce `unknown`. `P4155` separators also make
single-value and distinct-values definitions unknown because qualifiers are
not retained by the benchmark's triple abstraction. If exceptions remove all
applicable anchors, the result is unknown.

Primary definitions bind to the focus subject and constrained property after
the proposed edit, so replacements use the replacement value. Secondary
definitions bind to every represented occurrence of their own constrained
property. Multiple anchors aggregate to violated if any definitely violates,
satisfied if all definitely satisfy, and unknown otherwise. Edits are applied
as deletion followed by addition; deleting and reinserting the same statement
therefore preserves it.

Candidate construction and factor wiring use these same roles. Type repairs
emit `P31` or `P279`, inverse repairs use the `P2306` property, required-
statement repairs retain optional `P2305` values, and secondary factor anchors
come from actual constrained-property occurrences.

## Historical hierarchy

`scripts/build_historical_class_hierarchy.py` recursively downloads direct
`P279` parents starting from exactly 3,941 benchmark class IDs. That fixed seed
population is the union of registry `P2308` targets, `P31`/`P279` values in
the focus subject and object descriptions, and children of focus `P279`
statements. Repair-output fields and auxiliary-neighbour descriptions do not
expand the fixed seed population. Each request asks the
MediaWiki revisions API for the latest entity revision at or before
`2018-07-01T00:00:00Z`, records revision IDs and timestamps, and uses bounded
retry/backoff (including explicit `maxlag` and HTTP 429 delays) plus a per-entity
cache. Deprecated claims are ignored; preferred
claims replace normal claims when any preferred claim exists. Construction
stops on a network failure and never falls back to present-day data.

```bash
uv run python scripts/build_historical_class_hierarchy.py --seeds-only
uv run python scripts/build_historical_class_hierarchy.py
```

The versioned artifact is
`data/static/wikidata-p279-2018-07-01.v1.json`. It contains seed IDs, direct
edges, unavailable entities, revision provenance, closure-completeness flags,
cycles, the cutoff, source checksums, and a canonical content checksum.
Hierarchy nodes are symbolic evidence only and are not added to neural graphs.

## Regeneration sequence

Run the fixture and test gates before generating large artifacts:

```bash
uv run pytest -q tests/test_validator_semantics_v2.py tests/test_class_hierarchy_v2.py
uv run pytest -q
```

Then regenerate labels and both graph representations:

```bash
uv run python src/05_constraint_labeler.py \
  --dataset full_strat1m --registry-dataset full --min-occurrence 100 \
  --hierarchy data/static/wikidata-p279-2018-07-01.v1.json
uv run python src/06_graph.py \
  --dataset full_strat1m --registry-dataset full --min-occurrence 100 \
  --encoding node_id --constraint-representation factorized \
  --constraint-scope local --shard-size 200000 --use-torch-save \
  --persistence-profile research_safe --overwrite atomic \
  --hierarchy data/static/wikidata-p279-2018-07-01.v1.json
uv run python src/06_graph.py \
  --dataset full_strat1m --registry-dataset full --min-occurrence 100 \
  --encoding node_id --constraint-representation eswc_passive \
  --constraint-scope local --shard-size 10000 --use-torch-save \
  --persistence-profile research_safe --overwrite atomic --use-unlabeled-interim \
  --hierarchy data/static/wikidata-p279-2018-07-01.v1.json
```

The label manifest identifies validator version 2, the registry, encoder,
source rows, hierarchy, code revision, row counts, and per-family outcomes.
Graph manifests carry the same validator and hierarchy identity. Passive graph
payload checksums must match the archived generation exactly.

The scheduler validates dependencies and semantic provenance before work. It
reuses the archived Direct–Passive checkpoint after checksum and architecture
validation, then trains Direct–Factor, Candidate–C (`gamma_primary=0.2`),
Candidate–DP, and Candidate–SR in dependency order with seed 42. It evaluates
all five learned systems and refits/evaluates the four baselines:

```bash
uv run python src/10_scheduler.py --paper-suite
```

Prediction and evaluation artifacts use schema v3. Replay recomputes metric
events from rows and rejects schema-v2 artifacts, validator mismatches, and
hierarchy mismatches. Historical-fix/non-fix reports are diagnostic strata;
the principal training population is not filtered.

After generation and diagnostics complete, the result suite must pass before
manuscript values are edited:

```bash
uv run python scripts/check_corrected_paper_readiness.py
uv run python scripts/validate_regenerated_suite.py
```

After updating `latex_paper/main.tex` from those accepted artifacts, bind the
paper tables back to the machine-readable results and run final acceptance:

```bash
uv run python scripts/check_corrected_paper_readiness.py \
  --paper latex_paper/main.tex --verify-graph-checksums
uv run python scripts/validate_regenerated_suite.py --require-paper
```

Only artifacts passing these checks may supply manuscript values. The release
packager requires the paper-bound acceptance report and is updated for the new
provenance, but it does not publish or overwrite the existing Zenodo record.
