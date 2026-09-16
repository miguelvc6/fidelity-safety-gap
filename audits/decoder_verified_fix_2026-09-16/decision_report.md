# Decoder Coverage and Verified-Fix Feasibility Audit

**Scan status:** complete full CPU scan of all **1,910,794** parent records; the
**955,405** existing-sample rows and their exact complement are identified. Both
`checked_out_bounded` (`6c9f1818…`) and `audited_v3_bounded` (`018e2845…`) are complete.

Source accounting: 1,910,794 raw rows, 1,910,794 R rows,
0 unknown-constraint exclusions, and
0 malformed rows. Operations comprise
951,751 addition-only, 833,145 deletion-only,
125,898 paired, and
0 no-op records; no source row has an incomplete or multi-add/multi-delete edit.

The headline table uses the clean threshold-100 parent-training vocabulary and the hypothetical
factorized-input copy domain. No copy head was implemented or trained.

## Joint decision table

| Population/profile | Split | N source | N R | No-copy A | With-copy B | Verified F | F & A | F & (B-A) | F & B | F & not B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent_pool / audited_v3_bounded | overall | 1,910,794 | 1,910,794 | 1,710,505 | 1,719,936 | 1,323,129 | 1,220,925 | 5,150 | 1,226,075 | 97,054 |
| parent_pool / audited_v3_bounded | train | 1,337,555 | 1,337,555 | 1,197,671 | 1,204,142 | 925,727 | 854,411 | 3,546 | 857,957 | 67,770 |
| parent_pool / audited_v3_bounded | val | 286,619 | 286,619 | 256,417 | 257,902 | 198,677 | 183,221 | 791 | 184,012 | 14,665 |
| parent_pool / audited_v3_bounded | test | 286,620 | 286,620 | 256,417 | 257,892 | 198,725 | 183,293 | 813 | 184,106 | 14,619 |
| existing_sample / audited_v3_bounded | overall | 955,405 | 955,405 | 855,320 | 860,036 | 661,595 | 610,564 | 2,588 | 613,152 | 48,443 |
| existing_sample / audited_v3_bounded | train | 668,779 | 668,779 | 598,857 | 602,110 | 462,849 | 427,174 | 1,790 | 428,964 | 33,885 |
| existing_sample / audited_v3_bounded | val | 143,310 | 143,310 | 128,322 | 129,055 | 99,474 | 91,833 | 387 | 92,220 | 7,254 |
| existing_sample / audited_v3_bounded | test | 143,316 | 143,316 | 128,141 | 128,871 | 99,272 | 91,557 | 411 | 91,968 | 7,304 |
| previously_discarded / audited_v3_bounded | overall | 955,389 | 955,389 | 855,185 | 859,900 | 661,534 | 610,361 | 2,562 | 612,923 | 48,611 |
| previously_discarded / audited_v3_bounded | train | 668,776 | 668,776 | 598,814 | 602,032 | 462,878 | 427,237 | 1,756 | 428,993 | 33,885 |
| previously_discarded / audited_v3_bounded | val | 143,309 | 143,309 | 128,095 | 128,847 | 99,203 | 91,388 | 404 | 91,792 | 7,411 |
| previously_discarded / audited_v3_bounded | test | 143,304 | 143,304 | 128,276 | 129,021 | 99,453 | 91,736 | 402 | 92,138 | 7,315 |
| parent_pool / checked_out_bounded | overall | 1,910,794 | 1,910,794 | 1,710,505 | 1,719,936 | 643,000 | 625,695 | 2,632 | 628,327 | 14,673 |
| parent_pool / checked_out_bounded | train | 1,337,555 | 1,337,555 | 1,197,671 | 1,204,142 | 450,157 | 438,027 | 1,820 | 439,847 | 10,310 |
| parent_pool / checked_out_bounded | val | 286,619 | 286,619 | 256,417 | 257,902 | 96,470 | 93,838 | 426 | 94,264 | 2,206 |
| parent_pool / checked_out_bounded | test | 286,620 | 286,620 | 256,417 | 257,892 | 96,373 | 93,830 | 386 | 94,216 | 2,157 |
| existing_sample / checked_out_bounded | overall | 955,405 | 955,405 | 855,320 | 860,036 | 321,199 | 312,447 | 1,336 | 313,783 | 7,416 |
| existing_sample / checked_out_bounded | train | 668,779 | 668,779 | 598,857 | 602,110 | 224,774 | 218,664 | 942 | 219,606 | 5,168 |
| existing_sample / checked_out_bounded | val | 143,310 | 143,310 | 128,322 | 129,055 | 48,308 | 46,941 | 208 | 47,149 | 1,159 |
| existing_sample / checked_out_bounded | test | 143,316 | 143,316 | 128,141 | 128,871 | 48,117 | 46,842 | 186 | 47,028 | 1,089 |
| previously_discarded / checked_out_bounded | overall | 955,389 | 955,389 | 855,185 | 859,900 | 321,801 | 313,248 | 1,296 | 314,544 | 7,257 |
| previously_discarded / checked_out_bounded | train | 668,776 | 668,776 | 598,814 | 602,032 | 225,383 | 219,363 | 878 | 220,241 | 5,142 |
| previously_discarded / checked_out_bounded | val | 143,309 | 143,309 | 128,095 | 128,847 | 48,162 | 46,897 | 218 | 47,115 | 1,047 |
| previously_discarded / checked_out_bounded | test | 143,304 | 143,304 | 128,276 | 129,021 | 48,256 | 46,988 | 200 | 47,188 | 1,068 |

## Audited-v3 family intersections (parent population)

| Family | N R | A | B | F | F&A | F&(B-A) | F&B | F-not-B | F&A/F | F&B/F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conflictWith | 201,225 | 189,271 | 189,699 | 153,675 | 152,990 | 161 | 153,151 | 524 | 99.55% | 99.66% |
| distinct | 456,409 | 443,201 | 443,222 | 454,400 | 441,215 | 16 | 441,231 | 13,169 | 97.10% | 97.10% |
| inverse | 140,349 | 137,350 | 137,634 | 124,787 | 124,782 | 0 | 124,782 | 5 | 100.00% | 100.00% |
| itemRequiresStatement | 200,000 | 109,973 | 114,112 | 166,379 | 98,630 | 3,766 | 102,396 | 63,983 | 59.28% | 61.54% |
| oneOf | 13,709 | 13,572 | 13,609 | 3,016 | 2,983 | 31 | 3,014 | 2 | 98.91% | 99.93% |
| single | 239,446 | 208,936 | 210,331 | 105,434 | 105,412 | 1 | 105,413 | 21 | 99.98% | 99.98% |
| symmetric | 59,652 | 58,753 | 58,851 | 20,527 | 20,525 | 0 | 20,525 | 2 | 99.99% | 99.99% |
| type | 200,002 | 190,735 | 191,450 | 128,443 | 128,185 | 31 | 128,216 | 227 | 99.80% | 99.82% |
| valueRequiresStatement | 200,001 | 166,601 | 168,140 | 91,155 | 70,969 | 1,142 | 72,111 | 19,044 | 77.86% | 79.11% |
| valueType | 200,001 | 192,113 | 192,888 | 75,313 | 75,234 | 2 | 75,236 | 77 | 99.90% | 99.90% |

## Frozen-vocabulary comparison

| Vocabulary | A/R | B/R | copy-only | gain pp | F&A/F | F&B/F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean_parent_train_minocc100 | 89.52% | 90.01% | 9,431 | 0.49 | 92.28% | 92.66% |
| existing_train_val_minocc100 | 89.44% | 89.95% | 9,776 | 0.51 | 92.22% | 92.63% |

The existing vocabulary is the actual training-time union of sampled train and validation target
classes; it is not train-only. The clean vocabulary uses only the established parent training split,
with the existing threshold-100 encoder and independently supplied registry metadata frozen before
evaluation. Threshold 50 is not reported because exact pre-pruning training feature frequencies were
not retained; no threshold-50 production configuration was synthesized in this measurement task.

## Copy-domain comparison (audited v3, clean vocabulary, parent population)

| Input/copy scope | A | B | B-A | F&A | F&(B-A) | F&B | F-not-B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factorized | 1,710,505 | 1,719,936 | 9,431 | 1,220,925 | 5,150 | 1,226,075 | 97,054 |
| passive | 1,710,505 | 1,719,857 | 9,352 | 1,220,925 | 5,093 | 1,226,018 | 97,111 |

Uncovered/populated-slot grounding diagnostics (the first two categories are copy-addressable):

| Grounding category | Populated slots |
| --- | ---: |
| missing_role_present_local_neighbor | 9,679 |
| local_constraint_parameter | 162 |
| known_constant_blocked_by_output_mask | 625 |
| outside_fixed_vocabulary_and_input | 190,473 |
| unsupported_term_or_operation | 0 |

## Audited-v3 primary transitions (parent population)

| Pre \ Post | satisfied | violated | unknown |
| --- | ---: | ---: | ---: |
| satisfied | 257,572 | 6 | 207 |
| violated | 1,323,129 | 115,233 | 93,412 |
| unknown | 8,347 | 0 | 112,888 |

For the fixed pre-violated set E=1,531,774: post satisfied
1,323,129/1,531,774 (86.38%),
post unknown 93,412/1,531,774 (6.10%),
and post violated 115,233/1,531,774 (7.52%).
Distinct-values results are **bounded-local**, not global uniqueness certificates.

## Checked-out comparator primary transitions (parent population)

| Pre \ Post | satisfied | violated | unknown |
| --- | ---: | ---: | ---: |
| satisfied | 188,786 | 284 | 0 |
| violated | 643,000 | 546,131 | 0 |
| unknown | 14,850 | 5,042 | 512,701 |

For its fixed pre-violated set E=1,189,131: post satisfied
643,000/1,189,131 (54.07%),
post unknown 0/1,189,131 (0.00%),
and post violated 546,131/1,189,131 (45.93%).
Only 625,732/643,000 checked-profile F rows have every historical operation locally applicable; audited v3 has
1,323,129/1,323,129.

## Subset and workload summary

| Subset | N | definitions | add | del | paired | noop | same-S/P replacement | base removed | unresolved | nodes mean/p95/max | UNK nodes mean | edge workload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 1,910,794 | 5,667 | 951,751 | 833,145 | 125,898 | 0 | 99,080 | 556,080 | 0 | 1068.1/4344/8532 | 10.9 | 3,459,971,798 |
| F | 1,323,129 | 4,845 | 587,650 | 677,259 | 58,220 | 0 | 37,092 | 398,222 | 0 | 1034.4/4333/8232 | 7.9 | 2,087,915,220 |
| F_and_A | 1,220,925 | 4,445 | 499,388 | 677,259 | 44,278 | 0 | 23,198 | 394,641 | 0 | 1021.6/4332/8232 | 8.0 | 1,925,225,694 |
| F_and_B | 1,226,075 | 4,473 | 504,329 | 677,259 | 44,487 | 0 | 23,405 | 394,676 | 0 | 1031.5/4332/8232 | 7.9 | 1,944,193,518 |
| F_and_B_minus_A | 5,150 | 143 | 4,941 | 0 | 209 | 0 | 207 | 35 | 0 | 3381.9/4374/5891 | 6.2 | 18,967,824 |

Factorized node/edge figures are identity-preserving structural estimates from the same pre-edit
facts and retained factor definitions; no graph tensors or training shards were materialized.

## Parameter-bearing vocabularies

| Model | Vocabulary | input vocab | E/P outputs | input embedding params | six-head params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct-Factor | existing_train_val_minocc100 | 98,873 | 4,255/500 | 12,655,744 | 7,226,020 |
| Direct-Factor | clean_parent_train_minocc100 | 98,873 | 5,054/575 | 12,655,744 | 8,567,766 |
| Direct-Passive | existing_train_val_minocc100 | 98,873 | 4,255/500 | 12,655,744 | 2,324,580 |
| Direct-Passive | clean_parent_train_minocc100 | 98,873 | 5,054/575 | 12,655,744 | 2,756,214 |

Input UNK remains a trainable feature, but it is never used as semantic identity. Threshold changes
would alter feature/constant availability, not the already-computed semantic validator outcomes.

## Measured conclusion

Under audited v3 plus the clean parent-train vocabulary, no-copy retains 1,220,925/1,323,129 (92.28%) verified fixes. Hypothetical factorized-input copying raises this to 1,226,075 (92.66%), an exact copy-only verified gain of 5,150. The largest proportional family gain is itemRequiresStatement (3,766 rows). The weakest no-copy family is itemRequiresStatement at 59.28%; copying reaches 61.54%. The parent training split contains 854,411 verified no-copy rows, compared with the old sampled training-row budget of 668,779. These are measurements, not an architecture or final-dataset selection rule.

## Scope and limitations

- `checked_out_bounded` is the restored implementation comparator, not validator v3 and not a new
  correctness certification.
- `audited_v3_bounded` uses its parser, delete-then-add replay, partial/unresolved edit treatment,
  union-of-role relevance, and archived hierarchy-v2 artifact. Missing ancestry remains unknown.
- Both profiles retain their declared whole-subject/property bounded scope and assume represented
  entity fact maps are complete. No focal-only, global-uniqueness, separator-recovery, or new rule
  was introduced.
- Historical success means “verified under the named bounded profile,” not complete Wikidata or
  exact historical correctness. Copy figures are addressability upper bounds, not accuracy or
  candidate-generation recall.
- Primary counts are exact full scans. Secondary constraints were not re-evaluated because this
  audit prioritizes the requested exact primary intersections.

Machine-readable tables, the row audit, semantic chunks, diagnostic examples (seed 1729), hashes,
and execution metadata are in this directory.
