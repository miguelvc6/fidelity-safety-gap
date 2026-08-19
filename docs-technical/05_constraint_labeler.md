# 05_constraint_labeler.py

## Objective
- Generate per-factor constraint labels (checkable + satisfied) for the local or focus constraint neighborhood.
- Produce both **pre-edit** and **post-gold-edit** labels without rebuilding graphs.
- Track coverage and emit per-type summaries and coverage reports.

For the paper-facing run, `--constraint-scope local` is the canonical setting. `focus` remains supported for exploratory or appendix work only.

## Inputs & Outputs
**Inputs**
- Parquet split file(s) produced by `02_dataframe_builder.py` (from `data/interim/<dataset_variant>`).
- Constraint registry from `03_constraint_registry.py` (`data/interim/constraint_registry_<dataset>.parquet`).
- Encoder (`data/interim/<dataset_variant>/globalintencoder.txt`) for encoded parquet IDs.

**Outputs**
- Labeled parquet files under `data/interim/<dataset_variant>_labeled/` with additional columns:
  - `factor_checkable_pre`, `factor_satisfied_pre`
  - `factor_checkable_post_gold`, `factor_satisfied_post_gold`
  - `factor_types` (constraint type ids, aligned with `factor_constraint_ids`)
  - `factor_constraint_ids` (the constraint ids evaluated for the row)
  - `num_checkable_factors_pre`, `coverage_pre`
  - `num_checkable_factors_post_gold`, `coverage_post_gold`
- Coverage reports in the same output folder:
  - `coverage_<scope>.csv`
  - `coverage_<scope>.md`
- Filtered factor reports in the same output folder:
  - `filtered_factors_<scope>.csv`
  - `filtered_factors_<scope>.md`
  - `filtered_factor_families_<scope>.csv`

By default, `--factor-family-policy supported_only` writes only executable
supported constraints into `factor_constraint_ids` and aligned label arrays.
Unsupported secondary constraints remain in `local_constraint_ids` for
auditability but are not emitted as supervised factor nodes. If an unsupported
primary constraint is encountered, it is retained and marked not checkable so
graph construction can still identify the primary factor.

## Evidence Model
The labeler builds a normalized evidence structure per row:
```
facts_by_entity: Dict[entity_id, Dict[predicate_id, Set[object_id]]]
```
where entity/predicate/object IDs match the representation found in the parquet.

### P_local
`P_local` is the union of:
- `predicate`, `other_predicate`
- all predicate IDs appearing in `subject_predicates`, `object_predicates`, `other_entity_predicates`

Facts are restricted to `P_local` to ensure local-closure compatibility.

### Completeness Assumptions
We cannot directly observe whether all statements for an entity-property pair are present, so we use a conservative proxy:
- If `--assume-complete-entity-facts` (default), treat the entity facts blob as complete for all properties in scope.
- If `--no-assume-complete-entity-facts`, only treat properties explicitly present in the facts blob as complete.

For **single**, we additionally require:
- at least one statement for `(subject, P, *)`, and
- completeness for `(subject, P, *)`.

## Gold Edit Application
Two states are evaluated:
- **PRE**: facts as stored in the parquet.
- **POST_GOLD**: apply `add_*` and `del_*` edits to the facts representation.

Edits are resolved through placeholder tokens (`subject`, `predicate`, `object`, `other_*`) when present.
If an edit references an entity/property/value outside the local evidence structure, the corresponding
constraint checks are marked **not checkable** (conservative).

## Constraint Types Implemented (v1)
Per-type checkability and satisfaction are implemented in `src/modules/constraint_checkers.py`.
Canonical constraint-family names come from the registry (`constraint_family`), generated via the
static catalog in `data/static/constraint_type_catalog.json`. On a fresh clone,
`03_constraint_registry.py` bootstraps that catalog automatically if it is missing.
- `conflictWith`
- `inverse`
- `symmetric`
- `itemRequiresStatement`
- `valueRequiresStatement`
- `oneOf`
- `single`
- `type`
- `valueType`
- `distinct`

Semantics follow the short descriptions in
[constraint_types.md](../docs-conceptual/constraint_types.md). When evidence is
insufficient, the factor is marked **not checkable** to prioritize correctness
over coverage.

## Coverage Summary
At the end of a run, the script prints a per-type summary including:
- checkable rate (pre / post)
- satisfied rate (pre / post)

Use this report to tune completeness assumptions and identify constraint types with weak coverage.

## CLI
Example usage:
```bash
python src/05_constraint_labeler.py \
  --dataset sample \
  --min-occurrence 100 \
  --constraint-scope local \
  --factor-family-policy supported_only
```

Key flags:
- `--constraint-scope {local,focus}` selects `local_constraint_ids` vs `local_constraint_ids_focus`.
  The paper default is `local`.
- `--factor-family-policy {supported_only,all}` controls factor supervision.
  The paper default is `supported_only`; use `all` only to reproduce the older
  all-attached-factor behavior.
- `--registry-dataset` selects the raw dataset registry to use for derived variants such as `full_strat1m`; use `--registry-dataset full` for the paper benchmark.
- `--assume-complete-entity-facts/--no-assume-complete-entity-facts` toggles completeness assumptions.
- `--max-rows` caps rows per parquet for debugging.
