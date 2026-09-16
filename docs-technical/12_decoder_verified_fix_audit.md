# Decoder Coverage and Verified-Fix Audit

The identity-preserving audit is implemented by
`scripts/audit_decoder_verified_fixes.py` with semantic parsing and grounding
helpers in `src/modules/identity_audit.py`. It is measurement-only: it does not
change decoder architecture, candidate generation, training data, model
configuration, or production validation.

## Identity and lineage

Raw RDF/Wikidata terms are retained as canonical strings. The semantic integer
adapter used by historical validators is a reversible encoding of those
strings, not the frequency-pruned neural ID. Typed literals remain distinct,
blank nodes are source-scoped, and neural UNK is only a feature annotation.

Each row is keyed by raw filename and one-based line number. The script
replays the original stratified train/validation/test construction and the
50-percent benchmark sampler, then checks exact Arrow equality for the
retained scalar role and edit columns. It never joins rows by compressed
triples.

## Profiles

- `checked_out_bounded` loads revision `6c9f1818e2436e5bdfdc2023d01b3ec022af7596`
  from an isolated worktree. It is a restored-profile comparator.
- `audited_v3_bounded` loads revision
  `018e2845de43bdb9480a1aa9a7791befb8b317ee` from a separate worktree and
  requires the archived 2018-07-01 hierarchy-v2 artifact.

Both preserve local whole-subject/property scope and the configured complete
represented-fact-map policy. The v3 path retains delete-then-add replay,
partial/unresolved operation handling, and union-of-role relevance. The two
profiles are never imported in one Python process or pooled in an aggregate.

## Outputs

The completed run is in `audits/decoder_verified_fix_2026-09-16/`:

- `decision_report.md`: compact numerical decision report;
- `row_level_audit.parquet`: row-level lineage, coverage, validator outcomes,
  applicability, uncertainty, and workload sidecar;
- `semantic_chunks/`: lossless pre-edit semantic evidence by source shard;
- `semantic_constraint_definitions.parquet`: normalized registry definitions;
- `decoder_coverage`, `primary_transitions`, `subset_decisions`,
  `resource_estimates`, `slot_coverage`, and `grounding_reasons` as JSON/CSV;
- profile, aggregate, row, semantic, and run manifests with hashes and runtime
  metadata;
- fixed-seed diagnostic examples in
  `diagnostic_examples_seed1729.{json,csv}`.

The 865 MiB full-scan data products remain on the audit machine. In keeping
with the repository's local-data policy, Git excludes the row-level Parquet
sidecar and the resumable semantic/profile chunks; the compact numerical
report, aggregate JSON/CSV tables, manifests, semantic definitions, and exact
lineage maps are tracked. The excluded files are reproducible with the command
below and are fingerprinted by the tracked manifests.

Run all stages with:

```bash
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python \
  scripts/audit_decoder_verified_fixes.py \
  --stage all \
  --output audits/decoder_verified_fix_2026-09-16 \
  --checked-worktree /tmp/fsg-audit-checked \
  --v3-worktree /tmp/fsg-audit-v3 \
  --resume
```

`--resume` treats completed manifest-backed stages as immutable. The aggregate
stage also checkpoints each JSON/CSV pair atomically.

Threshold 50 is deliberately absent: the exact pre-pruning training-frequency
table needed to reconstruct it was not retained. Validator outcomes do not
depend on the neural threshold and must not be recomputed under guessed
frequencies.
