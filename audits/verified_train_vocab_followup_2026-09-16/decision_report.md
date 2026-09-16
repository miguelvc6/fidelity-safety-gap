# Verified-training vocabulary and action-retention follow-up

## Decision table

| Measure | Train | Validation | Test | Overall |
|---|---:|---:|---:|---:|
| V0 F & A | 854,411 | 183,221 | 183,293 | 1,220,925 |
| V1 F & A | 854,411 | 183,114 | 183,181 | 1,220,706 |
| Difference V1 - V0 | 0 | -107 | -112 | -219 |

| Headline | Count/rate |
|---|---:|
| V0 no-copy coverage among F | 1,220,925 / 1,323,129 (92.276%) |
| V1 no-copy coverage among F | 1,220,706 / 1,323,129 (92.259%) |
| V0 A over all parent R | 1,710,505 / 1,910,794 (89.518%) |
| V1 A over all parent R | 1,627,468 / 1,910,794 (85.172%) |
| V1 verified additions retained | 499,197 / 587,650 (84.948%) |
| V1 verified deletions retained | 677,259 / 677,259 (100.000%) |
| V1 verified paired edits retained | 44,250 / 58,220 (76.005%) |
| V1 itemRequiresStatement coverage | 98,502 / 166,379 (59.203%) |
| V1 valueRequiresStatement coverage | 70,949 / 91,155 (77.833%) |
| V1 concrete classes with zero F&A train support | 50 |
| V1 concrete classes with support 1-9 | 2,584 |
| V1 concrete classes with support 10-99 | 610 |
| V1 concrete classes with support >=100 | 274 |
| V1 entity/predicate concrete terms | 3,039 / 479 |

V1 uses the unchanged threshold-100 input encoder. Its output constants and role
tokens were fitted once from `parent_pool ∩ train ∩ F`; validation and test
targets did not participate. No vocabulary refit was performed after A(V1)
filtering, and no rows were sampled or selected.
Every V1 reference set is a subset of V0; there are no independently reserved
exceptions. Consequently, every A(V1) row is also A(V0).

## Family-by-action retention under V1

| Family | F | F&A(V1) | Additions retained | Deletions retained | Paired retained | Addition retention | Deletion retention | Paired retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conflictWith | 153,675 | 152,970 | 0 (0.00%) | 126,107 (82.44%) | 26,863 (17.56%) | undefined | 100.00% | 97.44% |
| distinct | 454,400 | 441,214 | 0 (0.00%) | 428,221 (97.06%) | 12,993 (2.94%) | undefined | 100.00% | 49.63% |
| inverse | 124,787 | 124,782 | 114,478 (91.74%) | 9,429 (7.56%) | 875 (0.70%) | 100.00% | 100.00% | 99.43% |
| itemRequiresStatement | 166,379 | 98,502 | 95,732 (97.19%) | 2,655 (2.70%) | 115 (0.12%) | 58.51% | 100.00% | 97.46% |
| oneOf | 3,016 | 2,979 | 0 (0.00%) | 1,179 (39.58%) | 1,800 (60.42%) | undefined | 100.00% | 97.99% |
| single | 105,434 | 105,410 | 0 (0.00%) | 104,317 (98.96%) | 1,093 (1.04%) | undefined | 100.00% | 97.85% |
| symmetric | 20,527 | 20,525 | 19,462 (94.82%) | 967 (4.71%) | 96 (0.47%) | 100.00% | 100.00% | 97.96% |
| type | 128,443 | 128,153 | 125,063 (97.59%) | 2,863 (2.23%) | 227 (0.18%) | 99.77% | 100.00% | 98.27% |
| valueRequiresStatement | 91,155 | 70,949 | 70,196 (98.94%) | 680 (0.96%) | 73 (0.10%) | 77.65% | 100.00% | 98.65% |
| valueType | 75,313 | 75,222 | 74,266 (98.73%) | 841 (1.12%) | 115 (0.15%) | 99.88% | 100.00% | 97.46% |

The complete structural table, including same-subject/property replacements,
base deletion/preservation, changed subjects/properties, and both V0 and V1,
is in `family_action_retention.csv`. Percentages above are compositions within
F&A(V1); retention columns use the corresponding F action count as denominator.

## Requires-statement exclusions

`requires_statement_exclusions.csv` records exact F, F&A(V1), and exclusion
counts for both requires-statement families, followed by excluded action types,
failing slots, term kinds, train-seen status, local-input availability, and the
top 20 definitions, predicates, and literal datatypes. `exclusion_reasons.csv`
contains a mutually exclusive deterministic row-primary reason plus overlapping
slot-level diagnostic tags.

## Distribution and fixed-budget feasibility

`distribution_shift.csv` reports percentage-point changes, relative retention,
and base-2 Jensen-Shannon divergence for family and action distributions. It
also covers base status, same-S/P replacement status, the original local-factor
bins, descriptive factorized-node bins, and distinct constraint definitions.

The V1 train pool contains 854,411
eligible rows, 185,632 more than the old
668,779-row training budget. `budget_feasibility.csv` contains only
mathematical allocations and projected within-family action mixtures; it does
not contain or freeze row identities.

## Factual answers

1. Fitting output references on verified training rows removes 219 V0-covered verified fixes (0.0166 percentage points of F).
2. V1 changes train/validation/test counts by 0 / -107 / -112 rows, respectively.
3. V1 retains 100.000% of deletion-only fixes versus 84.948% of addition-only and 76.005% of paired fixes. Deletion-only rows rise from 51.186% of F to 55.481% of F&A(V1), a 4.295-percentage-point compositional shift.
4. The largest family exclusions are itemRequiresStatement (67,877), valueRequiresStatement (20,206), and distinct (13,186). By action, exclusions are 88,453 addition-only, 13,970 paired, and 0 deletion-only rows.
5. Across all verified held-out concrete targets, validation has 17,149 zero-support and 2,157 support-1-9 components; test has 17,092 and 2,217. Conditional on F&A(V1), validation has 2 and 1,082; test has 1 and 1,113. Concrete-only denominators, shares, and slot splits are in `target_support_histogram.csv`; role references are reported separately.
6. Yes numerically: 854,411 eligible training rows exceed 668,779 by 185,632, without oversampling.
7. The author still must choose the final cohort, sampling/allocation rule, evaluation population(s), treatment of unexpressible verified fixes, any minimum-support policy, and whether architecture or production preprocessing changes are warranted. This audit makes none of those choices.

## Scope and limitations

All validator outcomes are reused from `audited_v3_bounded`; no validator was
rerun. They remain conditional on its bounded-local scope, completeness policy,
fixed hierarchy, registry, and evidence. The prior run's authoritative row,
semantic, profile, lineage, input, and V0 vocabulary hashes verified. Its
non-core `aggregate_manifest.json` byte fingerprint differs from the enclosing
run manifest after later CSV reserialization; this follow-up does not use that
file and records the mismatch in `manifest.json`.
