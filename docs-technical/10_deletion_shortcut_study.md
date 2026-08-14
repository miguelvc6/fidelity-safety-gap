# M1D/G0 deletion-shortcut study

## Isolation and rollback

The pre-study repository state is tagged `pre-m1d-g0-stability-v2`. Historical M1D and G0 directories are read-only evidence and are never reused. The study writes four additive versioned directories listed in `commands.md`.

The implementation is opt-in. Legacy direct-safety behavior remains `loss_weight=1`, `score_temperature=1`, and `focus_deletion_weight=0`; last-checkpoint persistence and stability thresholds are disabled by default. The config generator's `--study deletion-shortcut-v2` mode emits only the four study configs and does not overwrite existing configs.

To restore the pre-study state, revert commits after the tag and restore `latex_paper/main_compact.pre-m1d-g0-stability-v2.tex` if the paper has been updated. The four additive model directories can be moved aside without touching any historical artifact.

## Registered training configuration

Both M1D variants strictly load the existing A1 model state, validate the complete architecture and output vocabularies, create a fresh Adam optimizer, and fine-tune at learning rate `1e-5`. Candidate scores are divided by six before softmax because each candidate score sums six slot logits. The direct objective has outer weight `0.25`. The base-preserving variant alone sets `focus_deletion_weight=1`.

Both G0 variants use the frozen A1 proposal and fresh reranker parameters. They retain learning rate `1e-4` and label-blind test candidate generation. The base-preserving variant alone adds the unit focus-deletion penalty.

All runs use seed 42, a ten-epoch maximum, validation-only checkpoint selection, and the existing 25,000-row validation prefix. Best and last checkpoints are written atomically. Checkpoints and histories record the effective config, seed, A1 source checksum, graph paths, best epoch, and stability telemetry.

## Evaluation and gate

Each command performs corrected full-test evaluation, saves the direct result, replays `predictions.parquet`, and verifies equality. G0 additionally reconstructs every label-blind candidate set and requires 100% prediction membership. The study gate is:

```bash
uv run python scripts/check_deletion_shortcut_study.py
```

For unattended execution, the resumable sequential scheduler runs the same four registered commands, all replay and sidecar checks, and finally the gate:

```bash
uv run python scripts/run_deletion_shortcut_study.py
```

It never discovers or runs other model directories. It skips training only when the corresponding versioned best checkpoint already exists, writes one log per step under `logs/deletion_shortcut_v2/`, and atomically updates `status.json`. `--only M1D`, `--only M1D-BP`, `--only G0`, and `--only G0-BP` select one experiment; `--force-train` is required to overwrite a versioned checkpoint.

The gate rejects non-finite histories, total loss at or above 100, valid M1D logit magnitude at or above 10,000, incomplete schema-v2 metrics, legacy metric fields, checksum or row-count failures, replay disagreement, or incorrect A1 provenance. It promotes a base-preserving variant as mitigation only when both deletion rates fall and EPPF rises relative to its matched control.

The existing factorized graphs and labeled Parquet files remain unchanged. Corrected symbolic states are reconstructed only for training-objective events and evaluation metrics.
