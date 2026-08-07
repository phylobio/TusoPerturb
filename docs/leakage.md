# Split integrity of the sealed run

The values in [`champion/params.json`](../champion/params.json) come from a
single read of a sealed held-out test set. This page records what was checked
before that read, what the checks found, and the two places where a prediction
legitimately depends on other rows in the same batch.

**Verdict: clean. No test perturbation appears in any training set.**

## What was checked

131 recorded checks, 123 of them direct set intersections between a training
perturbation list and the corresponding test list, across all five benchmarks
and every dataset and seed.

| Scope | Result |
|---|---|
| Sealed metric rows | 1,444; leaked rows 0; error rows 0 |
| Comparable rows across arms | 1,336; mismatches 0 |
| Per-arm grid | 361 rows, 171 cells, 0 leaked, 0 errors, grid complete |
| Two-head conformance during the run | passed for every arm |

Splits used, as train / validation / test perturbation counts:

| Benchmark | HepG2 | Jurkat | K562 | RPE1 |
|---|---|---|---|---|
| CellSimBench and scPerturBench (fold 0) | 1602 / 255 / 465 | 1619 / 257 / 470 | 1411 / 224 / 409 | 1607 / 256 / 466 |
| PerturbHD regression | 1447 / 161 / 536 | 1519 / 169 / 563 | 1354 / 151 / 502 | 1476 / 164 / 547 |
| PerturbHD hit prediction | 1446 / 161 / 536 | 1518 / 169 / 563 | 1353 / 151 / 502 | 1475 / 164 / 547 |

Systema standard-split train / test counts:

| Panel | Train / test |
|---|---|
| adamson | 60 / 21 |
| norman (seeds 1, 2, 3) | 169 / 107, 159 / 117, 167 / 109 |
| replogle_k562_gwps | 1366 / 456 |
| replogle_rpe1 | 1061 / 354 |
| tian_crispra | 73 / 25 |
| tian_crispri | 136 / 46 |
| xu | 148 / 50 |

In all 21 Systema dataset-seed combinations, the standard training set equals
the union of the development training and validation sets, so the development
split is a partition of the standard training set and never borrows a test row.

## Code-level guards

Five guards in the harness raise `RuntimeError` rather than returning a value
when a split contract is violated. The test split is reachable only through an
explicitly named unsealing flag on the sealed-run entry point, so a test read
cannot happen by default or by accident.

## Honest disclosures

Three things are worth stating plainly rather than leaving for a reader to find.

### 1. Systema perturbations can share genes across the split

Systema panels contain combinatorial labels, so a test label such as `TP53+MYC`
can involve a gene that also appears in a training label. The *label* never
crosses the split, but the *gene* can. For the norman panel this affects 36 of
62 test labels at seed 1, 40 of 66 at seed 2, and 38 of 64 at seed 3. The other
18 of 21 dataset-seed combinations share no genes at all across the split.

This is a property of the benchmark's own split definition, not of TusoPerturb.
Every method scored on these panels inherits it. It is disclosed because a
gene-level reader could otherwise interpret combinatorial Systema scores as
fully gene-disjoint generalisation, which they are not.

### 2. Two prediction paths pool information across rows in a batch

Everything that is *fitted* — the scaler, the ridge models, the kNN feature
weights, the binary-arm thresholds, the target mean — uses training rows only.
Two operations nonetheless read more than one prediction row at a time.

**Adaptive kNN bandwidth (`knn_adaptive_bw='mean'`, shared head).** Each row's
neighbour distances are divided by that row's own mean distance, and the
resulting per-row scales are renormalised by their median across the batch. The
median is a batch statistic. For Systema the batch is the test rows, because the
adapter passes `test_idx`; for the other four benchmarks `test_idx` is `None`, so
the batch is every row and is dominated by training rows.

**Per-arm z-scoring (`y_z_score_arms=True`, hit head).** Each arm's output is
z-scored per gene column across the prediction batch before blending. The
PerturbHD-hit batch is all valid rows and is train-dominated.

Both are transductive in the weak sense that a prediction depends on the
composition of its batch. Neither reads a test *label*. This is why
[`REPRODUCE.md`](../REPRODUCE.md) insists the Systema test set be submitted in
one call: a different batching gives a different median and therefore different
numbers.

The shared head sets `y_z_score_arms=False` and `target_shape='residual_pert_mean'`,
so its final step applies only a scalar amplitude factor and adds back a
training-derived mean. It is not transductive at that step.

### 3. The co-essentiality basis is fitted on external data, and the control

The 96 co-essentiality columns come from a truncated SVD of the DepMap CRISPR
gene-effect matrix. No benchmark target, split index, or perturbation label
enters that fit. It is unsupervised with respect to every benchmark.

The DepMap matrix does, however, contain the cell lines the benchmarks are run
in. To check that the gain was not coming from those lines, the basis was
rebuilt with K562 (ACH-000551), HEPG2 (ACH-000739), and JURKAT (ACH-000995)
dropped before any statistic was computed, leaving 1,183 of 1,186 lines. The
holdout basis reproduced the champion's headline result. The control is
reproducible with the `--holdout-lines` flag documented in
[`scripts/gen_embeddings/README.md`](../scripts/gen_embeddings/README.md); the
holdout tables themselves are not shipped, because the champion uses the full
basis.

## Seed behaviour

Across-seed agreement of the predictions was 0.5162 / 0.5138 / 0.5188 for
PerturbHD regression and 0.5241 / 0.5217 / 0.5266 for hit prediction. Seeds
change the split, not the model: no shipped code path draws a random number that
affects a prediction.

## Determinism

The sealed run set `OPENBLAS_CORETYPE=Haswell` before the first NumPy import and
verified bit-identical outputs across two machines: 19 of 19 checked arrays
identical, maximum absolute difference 0. The kNN tie-expansion epsilon exists
for the same reason — it keeps the neighbour set, the only discrete decision in
the pipeline, stable when distances differ by floating-point reduction order.
