# Execution Log

All commands ran from `/home/mvazquez/fidelity-safety-gap` on 2026-09-16 UTC.
No GPU training, model fitting, live Wikidata query, hierarchy download, or
copy-head implementation was performed.

## Isolated revisions

```bash
git worktree add --detach /tmp/fsg-audit-checked 6c9f1818e2436e5bdfdc2023d01b3ec022af7596
git worktree add --detach /tmp/fsg-audit-v3 018e2845de43bdb9480a1aa9a7791befb8b317ee
```

The worktrees remained detached and the primary checkout was not reset or
populated with historical files.

## Full data stages

```bash
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage lineage --output audits/decoder_verified_fix_2026-09-16
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage semantic --output audits/decoder_verified_fix_2026-09-16
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage clean-vocab --output audits/decoder_verified_fix_2026-09-16
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage profile --profile-name checked_out_bounded --output audits/decoder_verified_fix_2026-09-16
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage profile --profile-name audited_v3_bounded --output audits/decoder_verified_fix_2026-09-16
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage finalize --output audits/decoder_verified_fix_2026-09-16
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage validate --output audits/decoder_verified_fix_2026-09-16
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage aggregate --output audits/decoder_verified_fix_2026-09-16 --resume
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage report --output audits/decoder_verified_fix_2026-09-16
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_decoder_verified_fixes.py --stage all --output audits/decoder_verified_fix_2026-09-16 --resume
```

The full semantic pass covered 27/27 raw shards and 1,910,794/1,910,794 parent
rows. Profile runtimes were 128.23 seconds for `checked_out_bounded` and 137.09
seconds for `audited_v3_bounded`. The final row join took 596.96 seconds with
934,208 KiB peak RSS. The full exact-round-trip validation took 136.30 seconds
with 426,364 KiB peak RSS. The successful exact aggregate took 1,184.75 seconds
with 21,459,440 KiB peak RSS. The original semantic run took 1,809 seconds by
its start/end artifact timestamps; its worker peak RSS was not captured.

Two aggregate attempts were interrupted while optimizing repeated group
filtering. A subsequent attempt reached diagnostics and failed on Arrow list
values represented as NumPy arrays. The fixed diagnostic was exercised on a
real Parquet row group, per-table atomic checkpoints were added, and the full
aggregate command above completed. These attempts did not change the immutable
row-level outcomes.

## Tests and validation

```bash
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python -m py_compile scripts/audit_decoder_verified_fixes.py src/modules/identity_audit.py
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run pytest -q tests/test_identity_audit.py
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run pytest -q tests/test_corrected_evaluation.py
PYTHONPATH=/tmp/fsg-audit-v3/src UV_CACHE_DIR=/tmp/fsg-uv-cache uv run pytest -q /tmp/fsg-audit-v3/tests/test_validator_semantics_v3_regressions.py /tmp/fsg-audit-v3/tests/test_constraint_identity_v3.py /tmp/fsg-audit-v3/tests/test_class_hierarchy_v2.py
```

Results: 8 identity-audit tests passed, 6 checked-out evaluation tests passed,
and 56 v3 regression/identity/hierarchy tests passed. A bounded smoke run over
the first 100 real `conflictWith` test rows completed under each isolated
profile before the full scans. `ruff` was requested through `uv run ruff` but
is not installed in this environment; compilation and tests succeeded.

The aggregate assertions verified `A` is a subset of `B`, every covered edit
round-trips exactly during the semantic pass, all transition and verified
coverage partitions reconcile, and parent counts equal existing sample plus
the exact discarded complement for every additive decision cell.
