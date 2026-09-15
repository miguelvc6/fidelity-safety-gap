# Model Architecture Efficiency Changes: `fidelity-safety-gap` vs. `constraint_factors`

Date: 2026-08-03

## Executive summary

The GNN backbone and six edit-prediction heads are essentially unchanged between the two repositories. The parameter- and compute-oriented work in `constraint_factors` is concentrated in the factor subsystem of `GIN_PRESSURE` and consists of three successive design changes:

1. sharing the three role-pressure MLPs across factor types;
2. allocating executor, post-edit, and embedding parameters only for active/reachable ids and packing the per-type linear layers for grouped dispatch; and
3. replacing independent per-type factor MLPs with shared trunks plus rank-16 type adapters.

All three changes are successful as parameter-storage reductions. On a common, data-normalized model configuration, the reconstructed original architecture has 137,391,887 parameters, the compact per-type variant has 36,380,649 (-73.5%), and the shared-adapter variant has 26,543,369 (-80.7%). The largest individual reductions come from shared pressure (-96.6% in that block) and the compact gold-edit embedding (-95.9% in that table).

The computational result is much weaker. Compacting inactive modules and unused embedding rows does not reduce the matrix work performed for an active factor. The shared-adapter path actually adds rank-16 residual calculations to otherwise unchanged `1603 -> 400 -> 400` and `800 -> 400` computations. Grouped BF16 dispatch can improve hardware utilization, but it also adds sorting, permutation, padding, and backend-dispatch overhead.

This distinction is visible in the stored runs. Relative to compact per-type execution, the shared adapter removes another 9,837,280 parameters (-27.0% of the whole model), but its mean epoch time improves by only 0.98%, its median epoch time by 0.25%, and its peak CUDA allocation by 1.32%. The adapter is therefore a strong model-size optimization, not evidence of a meaningful speed optimization.

## Scope and method

This report compares:

- `fidelity-safety-gap` at revision `ea2701a`;
- `constraint_factors` at revision `bf4f3d9`; and
- the efficiency changes introduced principally by `2423bfa` (shared pressure promoted to the canonical A1 configuration), `3a2801c` (compact and vectorized execution), and `7d51560` (shared adapters).

The inspection covered the model definitions, model configuration, parameter manifests, factor-dispatch implementation, architecture tests, and timing fields stored with the two executor-comparison runs. Parameter totals were also reproduced by instantiating the four architecture configurations through `constraint_factors/src/modules/models.py` on one fixed vocabulary.

The following are deliberately out of scope:

- graph/data schema changes and factor-type discovery as a data-processing concern;
- loss functions, optimization schedules, batching policy, training safeguards, and early stopping;
- evaluation behavior or model-quality comparisons; and
- primary-query, oracle-input, candidate-generation, and scientific-integrity changes that were not introduced to reduce parameters or architecture compute.

Several nearby model-code features were inspected but are not counted as efficiency changes. The passive/factorized allocation boundary, `legacy_shared` executor, and packed chooser scoring already exist in the inspected `fidelity-safety-gap` revision. Conversely, the new primary-query embeddings/MLPs and gold-scalar oracle input in `constraint_factors` serve different experimental questions and can add parameters or work; they are not optimizations of the original architecture.

## Common architecture that did not change

Both versions use the same main A1 shape for the comparison in this report:

- node-id embedding width: 128;
- optional role embedding width: 16;
- a two-linear initialization block projecting to hidden width 400;
- a separate initialization stage followed by three GIN/GINE message-passing layers;
- mean graph pooling followed by a shared `400 -> 400` projection;
- separate subject, object, and predicate branches; and
- six edit heads: add/delete for subject, predicate, and object.

The efficiency changes do not shrink the 400-wide GNN, reduce its layer count, factorize the edit heads, or reduce their output vocabularies. They target the executable-factor path attached to this backbone.

For hidden width `H=400`, an executable factor is represented by the factor-node state, three role-specific scope summaries, and three counts. The factor executor input is therefore `4H + 3 = 1603` dimensions.

## Original factor architecture

The original `fidelity-safety-gap` A1 configuration uses:

- `factor_executor_impl="per_type_v1"`;
- `num_factor_types=29`;
- `pressure_module_sharing="per_type"`; and
- the implicit full gold-edit embedding table.

Each of the 29 stable factor-type slots owns an independent precondition executor:

```text
1603 -> 400 -> 400 -> scalar
```

It also owns an independent post-edit head:

```text
[400 factor state ; 400 edit representation] -> 400 -> scalar
```

Finally, each of the three graph roles (predicate, subject, and object) owns an independent `800 -> 400 -> 400` pressure MLP for every factor type. This produces `3 x 29 = 87` large role-pressure MLPs.

At the widths above, the original factor blocks have the following exact sizes:

| Block                    |                             Per-unit formula |              Units |     Parameters |
| ------------------------ | -------------------------------------------: | -----------------: | -------------: |
| Factor executor          | `(1603x400+400) + (400x400+400) + (400x1+1)` |           29 types |     23,269,629 |
| Post-edit head           |                  `(800x400+400) + (400x1+1)` |           29 types |      9,303,229 |
| Role-pressure MLP        |              `(800x400+400) + (400x400+400)` | 3 roles x 29 types |     41,829,600 |
| Full gold-edit embedding |              `target_id_address_space x 400` |          one table | data-dependent |

The original repository's stored A1 vocabulary gives the full embedding 98,873 rows, or 39,549,200 parameters. Because raw id maxima differ between repository artifacts, the cross-version totals below use one common `constraint_factors` vocabulary instead of counting that data difference as an architecture change.

## Change 1: shared role-pressure modules

`constraint_factors` promotes `pressure_module_sharing="shared"` from an H2 ablation to the canonical A1-family configuration. The model code already exposed the option in the original repository; the architectural change is its adoption as the default/canonical parameterization.

The change replaces 29 MLPs per role with one MLP per role:

```text
before: 3 roles x 29 types x (800 -> 400 -> 400)
after:  3 roles x  1 shared x (800 -> 400 -> 400)
```

Consequences:

- role-pressure parameters fall from 41,829,600 to 1,442,400, a reduction of 40,387,200 (96.55%);
- the normalized whole model falls from 137,391,887 to 97,004,687 parameters (-29.40%);
- the shared parameterization permits all pressure edges for one role to be processed by one MLP call; the later packed-execution refactor implements this batching and avoids a Python/type loop and small per-type kernel launches; and
- the arithmetic per pressure edge is not reduced: each edge still passes through an `800 -> 400 -> 400` MLP.

This change also modifies the inductive bias. Type-specific behavior must be encoded in the factor state supplied to the shared role MLP rather than in separate role-MLP weights. It is not merely a storage-layout refactor.

## Change 2: compact active factor vocabulary

The `per_type_grouped_v2` path separates the stable factor-type address space from the model-local module count:

- stable address space: 29 ids;
- active model ids: `[0, 2, 3, 4, 5, 9, 12, 14, 15, 16]`;
- allocated type modules: 10 rather than 29; and
- checkpointed lookup buffers map stable ids to compact indices and back.

The per-type MLP architecture remains exactly the same. Only unused stable-id slots are removed, so the executor and post-edit blocks each retain `10/29` of their former parameters:

| Block            | Dense 29-type layout | Compact 10-type layout | Reduction |
| ---------------- | -------------------: | ---------------------: | --------: |
| Factor executors |           23,269,629 |              8,024,010 |    65.52% |
| Post-edit heads  |            9,303,229 |              3,208,010 |    65.52% |

This is an architecture-equivalent optimization for the ten active types. It reduces checkpoint size and optimizer state, but it does not reduce the MLP work for any factor that is actually processed. The original path already selected only the relevant type module for a factor row; the other 28 modules consumed storage, not per-row FLOPs.

The explicit mapping is safer than treating stable ids as dense array offsets. An inactive or out-of-range id fails rather than being clamped or accidentally routed to another module.

## Change 3: compact gold-edit embedding

The original post-gold auxiliary path allocates a `400`-wide embedding for every raw target id up to the maximum id. On the normalized vocabulary this is:

```text
102,380 rows x 400 = 40,952,000 parameters
```

With `gold_edit_embedding_mode="compact"`, the table contains only the sorted union of reachable entity and predicate target ids (including the no-op id). The stored compact runs have 4,172 rows:

```text
4,172 rows x 400 = 1,668,800 parameters
```

This removes 39,283,200 parameters (95.92% of the table). It is the largest storage reduction in the compact executor commit.

The lookup still retrieves and averages the same number of 400-dimensional vectors for each example, so lookup FLOPs are effectively unchanged. The benefit is parameter, gradient, optimizer-state, and checkpoint memory. The full-length stable-to-compact lookup is a non-parameter integer buffer and is small relative to the removed FP32 table.

The compact table is used by the gold-conditioned post-edit auxiliary head, not by ordinary edit inference. Unknown or unreachable ids now fail explicitly when that auxiliary path is requested.

## Change 4: packed banks and grouped dispatch

`per_type_grouped_v2` stores independent type weights in packed three-dimensional `GroupedLinearBank` tensors instead of a `ModuleList` of separate `nn.Linear` objects. A dispatch object:

1. stable-sorts factor rows by compact type;
2. records per-type counts and cumulative offsets;
3. runs the packed linear banks on the sorted rows; and
4. restores the original row order.

This storage change does not itself alter parameter count beyond active-type compaction. Its purpose is execution efficiency.

The implementation has three execution modes:

- BF16 CUDA on SM80 or newer uses `torch.nn.functional.grouped_mm` when the output width is a multiple of eight;
- scalar heads use a vectorized selected-weight dot product; and
- CPU, unsupported CUDA, and full-precision execution use a segmented `F.linear` fallback.

The grouped path can replace repeated boolean masks, scatters, and small per-type launches with grouped matrix operations. The same dispatch is reused by the precondition executor and post-edit head. Per-type pressure banks use the same mechanism when pressure is not shared.

Important limitations:

- dispatch construction adds a stable sort plus forward and inverse permutations;
- grouped BF16 execution pads input widths to an eight-element alignment and explicitly casts packed inputs and weights;
- the fallback still loops over non-empty type segments; and
- on a CUDA tensor the fallback converts offsets to a CPU list, which introduces a device synchronization. Full-precision GPU evaluation can therefore lose much of the intended dispatch benefit.

The run manifests confirm that both stored compact executor variants reached `grouped_mm_bf16`, so the direct comparison later in this report is not measuring the slow fallback.

## Change 5: shared trunks with low-rank type adapters

`shared_adapter_v1` goes beyond storage compaction and changes the factor model's parameterization.

The precondition executor becomes:

```text
shared:        1603 -> 400 -> 400
per type:      400 -> 16 -> 400 residual adapter
per type:      400 -> scalar
```

The post-edit head becomes:

```text
shared:        800 -> 400
per type:      400 -> 16 -> 400 residual adapter
per type:      400 -> scalar
```

The adapter-up matrices and biases are zero-initialized, so every type begins on the shared trunk and learns a type-specific residual.

For ten active types, the exact parameter change is:

| Block            | Compact independent types | Shared rank-16 adapters | Reduction |
| ---------------- | ------------------------: | ----------------------: | --------: |
| Factor executors |                 8,024,010 |                 938,170 |    88.31% |
| Post-edit heads  |                 3,208,010 |                 456,570 |    85.77% |
| Both blocks      |                11,232,020 |               1,394,740 |    87.58% |

This is a capacity-sharing change, not an architecture-equivalent packing change. All types share the large transformations and differ through a rank-16 residual and scalar head.

It also does not reduce arithmetic per factor row. Ignoring activations and biases:

| Path                  | Compact independent MACs per row | Shared-adapter MACs per row | Change |
| --------------------- | -------------------------------: | --------------------------: | -----: |
| Precondition executor |                          801,600 |                     814,400 | +1.60% |
| Post-edit head        |                          320,400 |                     333,200 | +4.00% |

The shared dense trunks may run more efficiently than multiple grouped matrices, but the rank-16 down/up projections add work. Any speed gain must come from better batching and kernel efficiency, not fewer mathematical operations.

## Normalized parameter comparison

To isolate architecture from changing class-id vocabularies, all four variants below were instantiated with the same current vocabulary (`102,381` input nodes, `3,679` entity output classes, `496` predicate output classes, and `102,380` raw target-id rows), the same 400-wide backbone, and the same ten active factor types where compact execution applies.

| Architecture                                                                   | Total parameters | Reduction vs. original | FP32 parameter bytes | Approx. FP32 parameter + gradient + Adam state |
| ------------------------------------------------------------------------------ | ---------------: | ---------------------: | -------------------: | ---------------------------------------------: |
| Original-equivalent: 29 per-type executors, full gold table, per-type pressure |      137,391,887 |                      - |            524.1 MiB |                                    2,096.4 MiB |
| Low-risk port: compact 10-type grouped executor + compact gold table + per-type pressure | 49,362,249 | 64.07% | 188.3 MiB | 753.2 MiB |
| Canonical shared pressure only                                                 |       97,004,687 |                 29.40% |            370.0 MiB |                                    1,480.2 MiB |
| Compact 10-type grouped executor + compact gold table + shared pressure        |       36,380,649 |                 73.52% |            138.8 MiB |                                      555.1 MiB |
| Shared rank-16 adapter + compact gold table + shared pressure                  |       26,543,369 |                 80.68% |            101.3 MiB |                                      405.0 MiB |

The final column is a planning estimate of 16 bytes per parameter: FP32 parameter, FP32 gradient, and two FP32 Adam moments. It excludes activations, temporary tensors, allocator reservation, mixed-precision implementation details, and framework overhead.

The efficiency-targeted blocks account for the following parameters; the remaining 22,037,429 parameters are common across all four normalized variants.

| Component                    | Original-equivalent | Shared pressure | Compact per-type | Shared adapter |
| ---------------------------- | ------------------: | --------------: | ---------------: | -------------: |
| Factor executors             |          23,269,629 |      23,269,629 |        8,024,010 |        938,170 |
| Post-edit heads              |           9,303,229 |       9,303,229 |        3,208,010 |        456,570 |
| Gold-edit embeddings         |          40,952,000 |      40,952,000 |        1,668,800 |      1,668,800 |
| Role-pressure modules        |          41,829,600 |       1,442,400 |        1,442,400 |      1,442,400 |
| Efficiency-targeted subtotal |         115,354,458 |      74,967,258 |       14,343,220 |      4,505,940 |

## What the stored runtime artifacts show

The closest available architecture-only comparison is between:

- `a1_factorized_imitation_per_type_compact__full_strat1m_minocc100__node_id`; and
- `a1_factorized_imitation_shared_adapter__full_strat1m_minocc100__node_id`.

Their manifests record the same seed (`42`), training-config hash, dataset-manifest hash, train/validation graph-manifest hashes, ten-type mapping, 4,172-row compact gold table, and `grouped_mm_bf16` backend. The runs were not produced from identical clean source states: their recorded commits differ and both manifests mark the source dirty. The figures below are therefore strong observational evidence, not a controlled microbenchmark.

| Recorded quantity over 15 epochs | Compact per-type | Shared adapter | Adapter change |
| -------------------------------- | ---------------: | -------------: | -------------: |
| Mean train graphs/s              |            90.14 |          91.01 |         +0.97% |
| Median train graphs/s            |            90.08 |          89.89 |         -0.21% |
| Mean total epoch seconds         |          9,867.7 |        9,771.2 |         -0.98% |
| Median total epoch seconds       |          9,895.9 |        9,871.2 |         -0.25% |
| Peak CUDA allocated              |         9.45 GiB |       9.33 GiB |         -1.32% |
| Peak CUDA reserved               |        17.01 GiB |      16.84 GiB |         -0.96% |

These results are consistent with the static analysis:

- 9.84 million fewer parameters materially reduce model and optimizer storage;
- factor-row arithmetic is not reduced and is slightly increased by adapters;
- both variants already use the same grouped BF16 dispatch; and
- whole-run memory and time are dominated by the unchanged GNN, edit heads, activations, graph tensors, and other work outside the factor parameter banks.

The shared-adapter experiment therefore achieved parameter efficiency but did not convert that reduction into a practically significant end-to-end compute improvement.

## Status of the variants in `constraint_factors`

The current canonical `a1_factorized_imitation` configuration adopts shared pressure but still uses the legacy dense 29-type executor and full gold table. The compact and adapter designs exist as separate experiment configurations:

- canonical A1: `per_type_v1` + full gold table + shared pressure;
- compact comparison: `per_type_grouped_v2` + ten active types + compact gold table + shared pressure; and
- adapter comparison: `shared_adapter_v1` + rank 16 + ten active types + compact gold table + shared pressure.

Consequently, it would be inaccurate to describe the current canonical A1 as the smallest architecture implemented in `constraint_factors`. Only the shared-pressure saving has been promoted to that configuration.

## Conclusions by change

| Change                  | Parameter/storage result                                | Compute result                                                        | Architectural risk                                     |
| ----------------------- | ------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------ |
| Shared pressure         | Excellent: -96.55% in the role-pressure block           | Fewer type loops/launches, but the same per-edge MLP work             | Removes type-specific role-MLP weights                 |
| Active-type compaction  | Excellent: -65.52% in executor and post-head banks      | Same per-active-factor FLOPs                                          | Low; equivalent for the declared active types          |
| Compact gold embedding  | Excellent: -95.92% in the table                         | Same lookup work                                                      | Low if the reachable-id contract is complete           |
| Grouped packed dispatch | Parameter-neutral beyond compaction                     | Potentially faster on BF16 SM80+; conditional and overhead-sensitive  | Low numerically, but backend/fallback behavior matters |
| Shared rank-16 adapter  | Excellent: -87.58% across executor and post-head blocks | No meaningful observed end-to-end speedup; slightly more per-row MACs | Material; imposes a shared low-rank parameterization   |

For a parameter-memory objective, the compact active vocabulary and compact gold table are the cleanest changes because they remove parameters that cannot be reached by the declared model vocabulary. Shared pressure is also highly effective but changes model capacity. The adapter should be understood primarily as a stronger capacity-sharing/model-compression experiment. It is not, in its present form, a convincing computational-efficiency optimization.

For a speed objective, the next architecture investigation should begin with profiling the unchanged GNN, edit heads, and activation memory rather than further reducing dormant factor parameters. The grouped factor path should be benchmarked separately at realistic factor counts on the exact target hardware, including BF16 training and full-precision evaluation, because its fast and fallback paths have materially different execution behavior.

## Low-risk implementation in `fidelity-safety-gap`

The implemented opt-in variant combines active-type compaction, the compact gold-edit table, and grouped packed dispatch while deliberately retaining `pressure_module_sharing="per_type"`. Thus it does not take the shared-pressure or shared-adapter capacity changes. `scripts/make_compact_a1_config.py` derives the active mapping from train/validation, verifies test coverage, copies the canonical A1 configuration, and writes the separate non-overwriting run directory `a1_factorized_imitation_compact_grouped__full_strat1m_minocc100__node_id`.

The normalized low-risk row above assumes the same ten active types observed in the advanced repository. The actual regenerated experiment size depends on the active mapping discovered from the newly produced labeled artifacts and should be recorded after config generation.

## Evidence locations

- Original-compatible and opt-in compact architecture: [`src/modules/models.py`](../src/modules/models.py) and [`src/modules/config.py`](../src/modules/config.py) in `fidelity-safety-gap`.
- Local packed dispatch: [`src/modules/factor_dispatch.py`](../src/modules/factor_dispatch.py).
- Local compact vocabulary helpers: [`src/modules/factor_types.py`](../src/modules/factor_types.py).
- Local architecture regression tests: [`tests/test_compact_factor_execution.py`](../tests/test_compact_factor_execution.py).
- Advanced architecture: `constraint_factors/src/modules/models.py`.
- Packed dispatch: `constraint_factors/src/modules/factor_dispatch.py`.
- Compact vocabulary helpers: `constraint_factors/src/modules/factor_types.py`.
- Architecture regression tests: `constraint_factors/tests/test_compact_factor_execution.py`.
- Advanced configuration reference: `constraint_factors/docs-technical/00_model_config_reference.md`.
- Exact component counts and backend records: the two `run_manifest.json` files under the compact-per-type and shared-adapter model directories.
- Runtime observations: the corresponding `training_history.json` files.
