# Execution log

Executed from `/home/mvazquez/fidelity-safety-gap` at 2026-09-16T12:13:26.212436+00:00.

```bash
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python scripts/audit_verified_train_vocab_followup.py \
  --stage all \
  --input-audit audits/decoder_verified_fix_2026-09-16 \
  --output audits/verified_train_vocab_followup_2026-09-16
```

Resolved command: `/home/mvazquez/fidelity-safety-gap/.venv/bin/python3 /home/mvazquez/fidelity-safety-gap/scripts/audit_verified_train_vocab_followup.py --stage all --input-audit audits/decoder_verified_fix_2026-09-16 --output audits/verified_train_vocab_followup_2026-09-16`

The command verified all authoritative source/audit fingerprints, scanned all 1,910,794 completed row-sidecar records twice (vocabulary fit, then aggregate derivation),
and invoked no symbolic validator. It wrote no
selected-row list and did not modify data/interim, data/processed, model, or
manuscript artifacts.

Validation commands:

```bash
UV_CACHE_DIR=/tmp/fsg-uv-cache uv run python -m py_compile \
  scripts/audit_verified_train_vocab_followup.py \
  src/modules/verified_vocab_followup.py \
  tests/test_verified_vocab_followup.py

UV_CACHE_DIR=/tmp/fsg-uv-cache uv run pytest -q \
  tests/test_identity_audit.py \
  tests/test_corrected_evaluation.py \
  tests/test_verified_vocab_followup.py
```

The current-checkout suite passed: `24 passed`.

The pinned audited-v3 regressions were run from the existing isolated detached
worktree at revision `018e2845de43bdb9480a1aa9a7791befb8b317ee`:

```bash
PYTHONPATH=/tmp/fsg-audit-v3/src \
  UV_CACHE_DIR=/tmp/fsg-uv-cache \
  uv run pytest -q \
  /tmp/fsg-audit-v3/tests/test_corrected_evaluation.py \
  /tmp/fsg-audit-v3/tests/test_evaluation_artifacts_v2.py \
  /tmp/fsg-audit-v3/tests/test_validator_semantics_v2.py \
  /tmp/fsg-audit-v3/tests/test_validator_semantics_v3_regressions.py \
  /tmp/fsg-audit-v3/tests/test_validator_transition_diagnostics.py
```

The audited-v3 suite passed: `66 passed`. Both pytest runs emitted only three
upstream PyTorch/PyG deprecation warnings. `ruff` was not available in the
project environment; Python compilation and both pytest suites completed.
