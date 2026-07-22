# TusoPerturb — Model Architecture

**Scope**: describes the actual model that produces every result number in
`report.md`. Reads directly against the code in `tusoperturb/heads.py`,
`predictor.py`, `feature_builder.py`, and `systema_adapter.py`.

**One-sentence summary**: TusoPerturb is a single per-perturbation ridge model
over a knowledge-graph feature stack, with three interchangeable "arms" (plain
ridge, kNN-weighted transfer, top-2 % binary ridge) that get z-scored and
blended with per-head weights. All five benchmark heads instantiate the same
`head_predict` function; they differ only in which arms are enabled, which
feature subset they consume, and how the target is shaped.

---

## 1. Data flow (one shared pipeline)

```
┌──────────────────────┐    ┌──────────────────────────┐    ┌─────────────────┐
│  Per-pert features   │──▶│  Per-head predictor       │──▶│  Y_pred_delta   │
│  E ∈ ℝ^(N × D)       │    │  head_predict(E, cfg)     │    │  or Y_pred_post │
└──────────────────────┘    └──────────────────────────┘    └─────────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │  cfg = HEAD_CONFIGS[bench] │
                          └──────────────────────────┘
```

Every benchmark goes through exactly this diagram. The only per-benchmark
routing is `HEAD_CONFIGS[bench]`, which returns one of three `HeadConfig`
objects.

---

## 2. Feature stack (11,680-D shared; 8,604-D Systema-native)

### Shared 11,680-D stack (CellSim / scPerturB / PerturbHD-reg / PerturbHD-hit)

Built by `feature_builder.build_shared_features(dataset)`. One row per
perturbation; column layout is exactly:

| Block      | Columns          | Width | What it is                                                                              |
|------------|:----------------:|:-----:|-----------------------------------------------------------------------------------------|
| `genept`   | 0 – 3,072        | 3,072 | GenePT LLM embedding of the perturbed gene (from the perturb-2026 stage loader).        |
| `reactome` | 3,072 – 4,888    | 1,816 | Multi-hot indicator over Reactome pathways containing the perturbed gene.               |
| `go_bp`    | 4,888 – 10,294   | 5,406 | Multi-hot over GO Biological Process terms.                                             |
| `hallmark` | 10,294 – 10,344  |    50 | Multi-hot over MSigDB Hallmark gene sets.                                               |
| `progeny`  | 10,344 – 10,358  |    14 | Dense PROGENy pathway coefficients for the perturbed gene.                              |
| `collectri`| 10,358 – 11,543  | 1,185 | Multi-hot over CollecTRI transcription factors that target the perturbed gene.          |
| `string`   | 11,543 – 11,671  |   128 | STRING v12 node2vec embedding (128-D) of the perturbed gene.                            |
| `depmap`   | 11,671 – 11,676  |     5 | DepMap essentiality features for the perturbed gene in the matched cell line.           |
| `baseline` | 11,676 – 11,680  |     4 | Per-pert self-expression stats in the unperturbed control (raw, log1p, z, quantile).    |
| **Total**  |                  | **11,680** |                                                                                     |

The stack is built once per dataset and reused across all four PerturbHD-family
heads (regression, cellsim, scperturb, hit). Only the column selection differs:

| Head        | `feature_subset`               | Width | Blocks used                                                                 |
|-------------|--------------------------------|:-----:|-----------------------------------------------------------------------------|
| Regression  | `full_11680`                   | 11,680| All blocks.                                                                 |
| Hit         | `no_genept_8608`               |  8,608| Drops `genept` (the LLM embedding hurt on the hit task in the sweep).       |

### Systema-native 8,604-D stack

Built by `systema_adapter.build_systema_features(perts, donor_id)`. Same
knowledge sources as above but without GenePT and with a Systema-specific
combination rule for combinatorial perturbations (e.g. Norman `X+Y`): sparse
indicator blocks are summed row-wise, dense embedding blocks are averaged. Column
layout:

| Block      | Width  | Combination rule for multi-gene perts |
|------------|:------:|----------------------------------------|
| reactome   | 1,816  | sum                                    |
| go_bp      | 5,406  | sum                                    |
| hallmark   |    50  | sum                                    |
| progeny    |    14  | mean                                   |
| collectri  | 1,185  | sum                                    |
| string     |   128  | mean                                   |
| depmap     |     5  | mean                                   |
| **Total**  | **8,604** |                                     |

DepMap features are looked up in the matched cell line via `DEPMAP_MAP`
(K562 for K562/A549/iPSC/melanoma/H1_hESC; RPE1 for RPE1).

---

## 3. The predictor (`head_predict`)

All five benchmark heads call the exact same function:

```python
Y_out = head_predict(E, train_idx, Y_train, cfg, test_idx=None)
```

Internally it does seven steps in order:

**Step 1 — Feature selection.** Slice `E` to the head's `feature_subset`
(`full_11680`, `no_genept_8608`, `no_genept_no_depmap_8603`, or the pre-built
`systema_native_8604`).

**Step 2 — Scaling.** Fit a scaler on `E[train_idx]` only, apply to all rows:
- `standard` (regression head) — `StandardScaler` with zero-variance-safe divisor.
- `robust_25_75` (hit + Systema heads) — `RobustScaler(quantile_range=(25, 75))`.

**Step 3 — Block reweighting.** For each block in `cfg.block_weights`, multiply
that column range post-scaling. Only the regression head uses this
(`go_bp × 0.67`, `string × 1.5`, `depmap × 10.0`).

**Step 4 — Target shaping.**
- `raw_delta` (regression + hit heads) — `Y_target = Y_train` (which is already
  in delta-from-control space for the regression heads and in raw AUCell space
  for the hit head).
- `residual_pert_mean` (Systema head only) — subtract the training-set gene
  mean: `Y_target = Y_train − Y_train.mean(axis=0)`. The mean is added back
  after prediction.

**Step 5 — Fit + predict up to three arms.** For arm weight `w_* > 0`:

- **Ridge arm** (weight `wr`) — `Ridge(α)` fixed or `RidgeCV(α ∈ {3, 10, 30, 100})`
  on `(X_train, Y_target)`, optional `y_standardize` first, predict on
  `X_pred` (either all rows or `test_idx` rows).
- **kNN arm** (weight `wk`) — cosine-normalize `X`, find `K=13` nearest train
  neighbours for each prediction row, softmax-weight with temperature
  `τ=11.0`, predict `Y = Σ_k w_k · Y_train[neighbour_k]`.
- **Binary arm** (weight `wb`) — per gene, binarize `Y_target` to the top
  `top_p=2 %` most-affected training perts (signed or unsigned), fit
  `RidgeCV(α ∈ {20, 60, 180})` on that binary target, predict on `X_pred`.

**Step 6 — Blend.**
- If only the ridge arm is on (`wr=1, wk=0, wb=0`) — return `Y_r` directly.
  This is the plain-ridge fast path.
- Otherwise — per-arm z-score across the prediction rows, blend by weights:
  `Y_pred = wr·zs(Y_r) + wk·zs(Y_k) + wb·zs(Y_b)`.

**Step 7 — Undo target shaping.** For `residual_pert_mean`, rescale
`Y_pred`'s column std/mean to match `Y_target`'s, then add back the training
`pert_mean` vector.

That's it. All the per-benchmark differences are choices about **which
`HeadConfig` you hand to this function**.

---

## 4. The three frozen head configurations

Every number in the report comes from one of these three configs. They are
literally three rows in a `dict`:

```python
HEAD_CONFIGS = {
    'cellsim':       PRIOR_A_v2_193,    # ← same object
    'scperturb':     PRIOR_A_v2_193,    # ← same object
    'perturbhd_reg': PRIOR_A_v2_193,    # ← same object
    'perturbhd_hit': PRIOR_R17_C,
    'systema':       PRIOR_SYSTEMA,
}
```

Key parameters side-by-side:

| Parameter               | `PRIOR_A_v2_193`       | `PRIOR_R17_C`      | `PRIOR_SYSTEMA`             |
|-------------------------|------------------------|--------------------|-----------------------------|
| Used for                | CellSim / scPerturB / PerturbHD-reg | PerturbHD-hit      | Systema (7 panels)          |
| `feature_subset`        | `full_11680`           | `no_genept_8608`   | `systema_native_8604`       |
| `scaler`                | `standard`             | `robust_25_75`     | `robust_25_75`              |
| `block_weights`         | `go_bp:0.67, string:1.5, depmap:10.0` | `{}`     | `{}`                        |
| `ridge_estimator`       | `ridge_fixed` (α=110)  | `ridgecv`          | `ridgecv`                   |
| `y_standardize`         | `False`                | `True`             | `True`                      |
| Arm weights (WR, WK, WB) | (**1.0**, 0.0, 0.0)   | (0.10, **0.65**, 0.25) | (0.20, **0.70**, 0.10) |
| `signed_binary`         | (unused, wb=0)         | `False`            | `True`                      |
| `target_shape`          | `raw_delta`            | `raw_delta`        | `residual_pert_mean`        |

`PRIOR_A_v2_193` collapses to a plain single Ridge (`α = 110`) with a
StandardScaler and hand-picked block reweights.

`PRIOR_R17_C` and `PRIOR_SYSTEMA` are structurally the *same* 3-arm blend
(kNN-dominant, small ridge/binary side channels, robust scaler, RidgeCV);
they differ only in three parameters: (a) blend weights, (b) whether the
binary arm is signed, (c) whether the target is raw delta or pert-mean
residuals.

---

## 5. Per-benchmark IO wrappers

Only three thin wrappers in `api.py` differ per benchmark — they handle
loading the right target and reshaping the output into the format each
benchmark's scorer expects. The **model math is identical** across all
three; only the target `Y_train` fed in and the output post-processing
change.

### 5.1 CellSim / scPerturB / PerturbHD-reg — `predict_regression`

```python
sd = build_shared_features(dataset)              # 11,680-D stack
Y_delta = head_predict(sd['E_all'], sd['train_indices'],
                       sd['Y_train'], PRIOR_A_v2_193)
X_post = sd['mean_baseline'][None, :] + Y_delta   # add control mean
return build_pred_adata_from_matrix(X_post, ...)  # standard AnnData wrapping
```

- `Y_train` is delta-from-control expression, `(n_train_perts, n_genes)`.
- Output is an AnnData with predicted post-perturbation expression;
  the CellSim / scPerturB / regression scorers each compute their own
  metric on top.

### 5.2 PerturbHD-hit — `predict_hit`

```python
sd = build_shared_features(dataset)                     # 11,680-D stack
# Load per-pert AUCell scores across MSigDB Hallmark pathways
pivot = pd.read_parquet(f'{paper_key}-h.all-all.pq')    # (n_perts × 50 phenos)
# Restrict E to the valid_perts rows that appear in the AUCell table
Y_pred = head_predict(E_valid, train_row_idx, Y_train_aucell, PRIOR_R17_C)
# Return long-form (pert, pheno, hit_score) for test + val perts
```

- The **target changes**: instead of gene-level delta expression, `Y_train`
  is `(n_train_perts, 50 pathways)` AUCell "how enriched is this pathway
  under this perturbation" scores.
- The wrapper also handles per-seed splits (`split-1/2/3` columns in the
  parquet) which pick the train / val / test partition for that seed.

### 5.3 Systema — `predict_systema`

```python
E, offsets, stats = build_systema_features(all_perts, donor_id)  # 8,604-D
Y_test = head_predict(E, tr, Y_train_post, PRIOR_SYSTEMA, test_idx=te)
return Y_test    # (n_test_perts, n_genes) log1p post-expression
```

- The **feature stack is rebuilt from scratch** (different combination rule
  for combinatorial perts, no GenePT, different block ordering) via the
  Systema adapter.
- The **target is raw log1p post-expression** (not delta from control),
  which is why `target_shape='residual_pert_mean'` — subtracting the
  per-gene mean over training perts is the Systema paper's convention.
- `test_idx=te` is passed so the arm-level z-scoring in step 6 is
  computed over test rows only (matches the source champion's `predict`
  semantics, byte-verified on `adamson seed=1`).

---

## 6. What actually changes per benchmark

Concretely, here's the per-benchmark diff — the only things that vary:

| Aspect                       | Regression heads | Hit head       | Systema head              |
|------------------------------|------------------|----------------|---------------------------|
| Feature matrix               | 11,680-D shared  | 11,680-D minus `genept` | 8,604-D Systema-native  |
| Combinatorial pert rule      | (single-gene only) | (single-gene only) | sum for indicators, mean for dense |
| Target                       | Δ expression     | AUCell over 50 pathways | log1p post-expression |
| Scaler                       | StandardScaler   | RobustScaler(25,75) | RobustScaler(25,75)  |
| Block reweights              | 3 blocks         | none           | none                      |
| Ridge                        | Fixed α=110      | RidgeCV        | RidgeCV                   |
| Model                        | 1 arm (plain ridge) | 3-arm z-blend (0.10/0.65/0.25) | 3-arm z-blend (0.20/0.70/0.10) |
| Target shape                 | raw delta        | raw delta      | residual-over-pert-mean   |
| Binary arm sign              | (off)            | unsigned       | signed                    |
| Output post-processing       | + control mean → AnnData | long-form scores | (n_test, n_gene) matrix |

Everything else — the code path, the arm equations, the scaling contract,
the split handling — is shared.

---

## 7. Is this "the same method"?

**Depends what you mean by "same". Three views:**

### View A — "Same package, same predictor" — YES

- **One code path** (`head_predict`) handles all five benchmarks.
- **One feature stack** (shared 11,680-D built from the same knowledge
  graphs: GenePT, Reactome, GO-BP, MSigDB Hallmark, PROGENy, CollecTRI,
  STRING, DepMap) is used for four of the five benchmarks; Systema
  reuses the same seven knowledge sources but rebuilds the stack to
  handle combinatorial perts.
- **One design pattern**: for each perturbation, look up its knowledge-graph
  fingerprint, feed it through a ridge with a kNN transfer side channel and
  a top-hit binary side channel, blend, decode into the target space the
  scorer wants.

### View B — "Same model class" — YES with a footnote

All three heads are members of the same family:

$$
\hat{Y} = \sigma_R\, w_R \cdot \text{zs}(f_R(X)) + \sigma_K\, w_K \cdot \text{zs}(f_K(X)) + \sigma_B\, w_B \cdot \text{zs}(f_B(X))
$$

where:
- $f_R(X) = \text{Ridge}(\alpha)$ trained on the raw feature matrix,
- $f_K(X) = $ softmax-weighted kNN transfer with cosine metric, $K=13$, $\tau=11$,
- $f_B(X) = \text{RidgeCV}$ trained on a top-2 %-binarized target,
- $\sigma_\cdot \in \{0, 1\}$ toggles arm-level z-scoring.

The regression head is this family with $w_R = 1, w_K = w_B = 0$; the hit
head is $w_R = 0.10, w_K = 0.65, w_B = 0.25$; the Systema head is
$w_R = 0.20, w_K = 0.70, w_B = 0.10$. Footnote: the regression head is a
degenerate case (just Ridge) — some readers may not accept a
$w_R = 1, w_K = w_B = 0$ blend as "the same" as $w_K = 0.7$-dominant
blend. It's the same *equation* but very different *behaviour*.

### View C — "Same trained model applied to all benchmarks" — NO

We do not have one set of ridge coefficients that handles all five
benchmarks. Each head has its own $(\alpha, \text{arm weights},
\text{block weights}, \text{target shape})$ frozen from a separate
hyperparameter search on that benchmark's leaderboard. In that stricter
sense TusoPerturb is a **unified framework with three fitted heads**, not a
single trained model.

### Bottom line

TusoPerturb is a **single method** in the sense that:
- One package, one API, one predictor function.
- One shared feature builder for all benchmarks that use per-gene features.
- One equation family; the three heads are members of it.

It is **not** a single trained model — the hyperparameters differ per head.
The design bet was that unifying the three architectural pieces (feature
stack, arm equations, scaling contract) is a real methodological
contribution *even without* forcing one set of weights to fit all
benchmarks. The benchmark results support that bet: nothing in the
harmonization hurt performance on any bucket, and PerturbHD-hit +
PerturbHD-reg + CellSim + scPerturB all sit at #1 for their primary
metric using the shared framework.

---

## 8. Byte-identity guarantees

The refactor is verified against the source champions to floating-point
zero on spot checks:

| Head             | Source                                | Spot check                              | Result                                    |
|------------------|---------------------------------------|-----------------------------------------|-------------------------------------------|
| `PRIOR_A_v2_193` | `A_v2_193_STR15_Dep100_a110`          | HepG2 seed 1, `Y_pred_delta` (2322 × 8746) | `max_abs_diff = 0.0` across all elements. |
| `PRIOR_R17_C`    | `R17_C_tau11_wr10_wk65_wb25`          | K562 seed 1, `recall_at_budget_0.05`    | 0.4787 vs source 0.4787 (matches).        |
| `PRIOR_SYSTEMA`  | `res_wr20_wk70_wb10`                  | adamson seed 1, full output (106,260 elems) | `max_abs_diff = 0.0` across all elements. |

Full scorer-side verification: TusoPerturb `calibrated/*` and
`scperturbench_mean/*` outputs on Jurkat seed 1 are bit-exact matches to
the A_v2_193 reference parquet (checked all 20 calibrated metrics + 3
scperturbench metrics).
