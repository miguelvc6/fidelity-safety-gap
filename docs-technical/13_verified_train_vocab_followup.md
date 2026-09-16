# Verified-training output-vocabulary follow-up

The focused follow-up is implemented by
`scripts/audit_verified_train_vocab_followup.py`, with pure grounding,
action-classification, support, and allocation helpers in
`src/modules/verified_vocab_followup.py`.

It consumes the completed identity-preserving audit under
`audits/decoder_verified_fix_2026-09-16/`. The command verifies the raw-source,
row-sidecar, semantic-chunk, audited-v3 profile-chunk, registry, encoder,
hierarchy, and V0 vocabulary fingerprints before use. It does not invoke a
validator: `v3_F` and `v3_E` are immutable inputs from
`audited_v3_bounded`.

## Vocabulary profiles

- `clean_parent_train_minocc100` (V0) is the existing output vocabulary fitted
  before verified-fix filtering.
- `verified_train_minocc100` (V1) retains the same threshold-100 input encoder
  and fits output constants and role tokens only from parent training rows for
  which the recorded audited-v3 `F` flag is true.

Role aliases take precedence over concrete constants. A non-role target enters
V1 only if it already has a nonzero class in the fixed input encoder. The
command does not apply a second target-frequency threshold, use held-out
targets, add copy references, or refit after A(V1) filtering.

## Execution

Run the full derivative scan with:

```bash
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python \
  scripts/audit_verified_train_vocab_followup.py \
  --stage all \
  --input-audit audits/decoder_verified_fix_2026-09-16 \
  --output audits/verified_train_vocab_followup_2026-09-16
```

If table generation completed but report rendering was interrupted, the
report-only stage revalidates all inputs and compact-table partitions without
rescanning the row sidecar:

```bash
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python \
  scripts/audit_verified_train_vocab_followup.py \
  --stage report \
  --input-audit audits/decoder_verified_fix_2026-09-16 \
  --output audits/verified_train_vocab_followup_2026-09-16
```

## Outputs and artifact policy

The compact report, vocabulary, manifest, summary, and CSV tables live under
`audits/verified_train_vocab_followup_2026-09-16/`. They contain vocabulary
comparison, V0/V1 coverage, per-slot target support, family/action retention,
exclusion diagnoses, distribution shift, evaluation strata, and hypothetical
fixed-budget feasibility.

`row_level_followup.parquet` is a local, ignored derivative. It contains only
source indices and derived V0/V1/action flags; it is not a selected training
dataset. Its checksum is recorded in the compact manifest. No selected row
list, graph, model, production configuration, or manuscript change is emitted.

The input verifier records one non-authoritative mismatch inherited from the
prior run: `aggregate_manifest.json` was reserialized after its enclosing run
manifest. The authoritative row sidecar, semantic chunks, audited-v3 profile
chunks, source lineage, and V0 vocabulary all match their recorded hashes, and
the follow-up does not consume the mismatched aggregate manifest.
