# Corrected Experiment Commands

Run these commands from the repository root. Each model experiment has one unwrapped Bash command. Training, corrected evaluation, replay verification, and applicable sidecar diagnostics are chained with `&&`, so each later stage runs only after the previous stage succeeds. The versioned M1D/G0 study directories leave the historical runs untouched. All new training uses seed 42.

## Compact A1

Reuse the existing Compact A1 checkpoint; do not retrain it.

```bash
uv run python src/09_eval.py --run-directory models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id --strict-global-metrics --per-constraint-csv --batch-size 256 && uv run python src/09_eval.py --run-directory models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id --strict-global-metrics --h2-eval --h2-batch-size 256 && uv run python scripts/analyze_candidate_oracle.py --run-directory models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id --strict-global-metrics --batch-size 256
```

## Original B0

```bash
uv run python src/07_train.py --experiment-config models/b0_eswc_reproduction__full_strat1m_minocc100__node_id/config.json && uv run python src/09_eval.py --run-directory models/b0_eswc_reproduction__full_strat1m_minocc100__node_id --strict-global-metrics --per-constraint-csv --batch-size 256
```

## Compact M1C

```bash
uv run python src/07_train.py --experiment-config models/m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id/config.json && uv run python src/09_eval.py --run-directory models/m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id --strict-global-metrics --per-constraint-csv --batch-size 256 --use-chooser && uv run python src/09_eval.py --run-directory models/m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id --use-chooser --strict-global-metrics --h2-eval --h2-batch-size 256 && uv run python scripts/analyze_candidate_oracle.py --run-directory models/m1c_safe_factor_chooser_compact_grouped__full_strat1m_minocc100__node_id --strict-global-metrics --batch-size 256
```

## M1D

```bash
uv run python src/07_train.py --experiment-config models/m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id/config.json && uv run python src/09_eval.py --run-directory models/m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id --strict-global-metrics --per-constraint-csv --batch-size 256 && cp models/m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id/evaluations/model.json models/m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id/evaluations/model.direct.json && uv run python src/09_eval.py --run-directory models/m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id --predictions models/m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id/evaluations/predictions.parquet --strict-global-metrics --per-constraint-csv --batch-size 256 && uv run python src/09_eval.py --run-directory models/m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id --strict-global-metrics --h2-eval --h2-batch-size 256 && uv run python scripts/analyze_candidate_oracle.py --run-directory models/m1d_safe_factor_direct_v2__full_strat1m_minocc100__node_id --strict-global-metrics --batch-size 256
```

## M1D with base preservation

```bash
uv run python src/07_train.py --experiment-config models/m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id/config.json && uv run python src/09_eval.py --run-directory models/m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id --strict-global-metrics --per-constraint-csv --batch-size 256 && cp models/m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id/evaluations/model.json models/m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id/evaluations/model.direct.json && uv run python src/09_eval.py --run-directory models/m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id --predictions models/m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id/evaluations/predictions.parquet --strict-global-metrics --per-constraint-csv --batch-size 256 && uv run python src/09_eval.py --run-directory models/m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id --strict-global-metrics --h2-eval --h2-batch-size 256 && uv run python scripts/analyze_candidate_oracle.py --run-directory models/m1d_safe_factor_direct_base_preserving_v2__full_strat1m_minocc100__node_id --strict-global-metrics --batch-size 256
```

## G0

Retrain G0 with gold available only to its training objective, then generate label-blind test predictions, migrate them in canonical test-row order, and write the corrected evaluation and deletion diagnostic.

```bash
uv run python src/08_train_reranker.py --experiment-config models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/config.json --prediction-batch-size 256 --seed 42 && uv run python src/09_eval.py --run-directory models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id --legacy-predictions-json models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/reranker_predictions.json --strict-global-metrics --per-constraint-csv --batch-size 256 && cp models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/evaluations/model.json models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/evaluations/model.direct.json && uv run python src/09_eval.py --run-directory models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id --predictions models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/evaluations/predictions.parquet --strict-global-metrics --per-constraint-csv --batch-size 256 && uv run python scripts/audit_prediction_candidate_membership.py --run-directory models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id --proposal-run-directory models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id --predictions models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/reranker_predictions.json --output models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/evaluations/candidate_membership_audit.json --batch-size 256 && uv run python scripts/analyze_deletion_degeneracy.py --g0-run-directory models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id --predictions models/g0_globalfix_reference_v2__full_strat1m_minocc100__node_id/evaluations/predictions.parquet --strict-global-metrics
```

## G0 with base preservation

```bash
uv run python src/08_train_reranker.py --experiment-config models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id/config.json --prediction-batch-size 256 --seed 42 && uv run python src/09_eval.py --run-directory models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id --legacy-predictions-json models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id/reranker_predictions.json --strict-global-metrics --per-constraint-csv --batch-size 256 && cp models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id/evaluations/model.json models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id/evaluations/model.direct.json && uv run python src/09_eval.py --run-directory models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id --predictions models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id/evaluations/predictions.parquet --strict-global-metrics --per-constraint-csv --batch-size 256 && uv run python scripts/audit_prediction_candidate_membership.py --run-directory models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id --proposal-run-directory models/a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id --predictions models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id/reranker_predictions.json --output models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id/evaluations/candidate_membership_audit.json --batch-size 256 && uv run python scripts/analyze_deletion_degeneracy.py --g0-run-directory models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id --predictions models/g0_globalfix_base_preserving_v2__full_strat1m_minocc100__node_id/evaluations/predictions.parquet --strict-global-metrics
```

## DFB

The baseline evaluator runs DFB together with the other deterministic baselines and writes separate schema-v2 outputs for each.

```bash
uv run python src/09_eval.py --run-baselines --dataset full_strat1m --min-occurrence 100 --strict-global-metrics --per-constraint-csv --batch-size 256
```

Original A1 is intentionally omitted: it is retained only as prior compression-equivalence evidence and must not be retrained or included in the corrected safety table.

## Study readiness gate

Run this only after all four M1D/G0 study commands finish:

```bash
uv run python scripts/check_deletion_shortcut_study.py
```
