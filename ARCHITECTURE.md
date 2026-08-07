# Architecture

TusoPerturb represents each perturbation with biological reference features and
maps those features to an expression or phenotype target. Version 2 is a
**two-head** method: all five benchmark keys read the same 8,700-dimensional
feature matrix, and the two heads differ only in head parameters.

The implementation is split across five modules:

- [`tusoperturb/systema_adapter.py`](tusoperturb/systema_adapter.py) builds the
  8,604-column annotation stack from the bundled references, including support
  for combinatorial perturbations.
- [`tusoperturb/coessentiality.py`](tusoperturb/coessentiality.py) looks up the
  96 co-essentiality columns.
- [`tusoperturb/feature_builder.py`](tusoperturb/feature_builder.py) joins the
  two into the 8,700-column matrix and loads staged benchmark data.
- [`tusoperturb/heads.py`](tusoperturb/heads.py) defines the two head
  configurations and the five-key map.
- [`tusoperturb/predictor.py`](tusoperturb/predictor.py) implements feature
  processing, model fitting, and prediction.

## Data flow

```text
perturbation labels
        │
        ▼
annotation stack (8604) ──┐
                          ├──► feature matrix E  (n_perts × 8700)
co-essentiality (96) ─────┘             │
                                        │
training rows + targets  ───────────────┤
                                        ▼
                             head_predict(E, ...)
                                        │
                                        ▼
                          benchmark-specific output
```

The benchmark adapters in [`tusoperturb/api.py`](tusoperturb/api.py) load
targets, choose a head from `HEAD_CONFIGS`, and format predictions for an
external scorer.

## Feature representation

`build_features(perts, donor_id)` creates one row per perturbation. Column
ranges use Python's half-open notation, `[start:end)`.

| Block | Columns | Width | Multi-gene aggregation | Contents |
|---|---:|---:|---|---|
| `reactome` | `[0:1816)` | 1,816 | Sum | Reactome pathway membership indicators. |
| `go_bp` | `[1816:7222)` | 5,406 | Sum | Gene Ontology Biological Process indicators. |
| `hallmark` | `[7222:7272)` | 50 | Sum | MSigDB Hallmark gene-set indicators. |
| `progeny` | `[7272:7286)` | 14 | Mean | PROGENy pathway coefficients. |
| `collectri` | `[7286:8471)` | 1,185 | Sum | CollecTRI target-to-regulator indicators. |
| `string` | `[8471:8599)` | 128 | Mean | STRING network embedding. |
| `depmap` | `[8599:8604)` | 5 | Mean | Cell-line DepMap summary features. |
| `coess64` | `[8604:8668)` | 64 | Mean | Rank-64 DepMap co-essentiality coordinates. |
| `coess32` | `[8668:8700)` | 32 | Mean | Rank-32 DepMap co-essentiality coordinates. |
| **Total** |  | **8,700** |  |  |

Every head sees every column. There is no feature subsetting, no block
weighting, and no language-model embedding block.

Perturbation labels are split on `+`, and the literal component `ctrl` is
ignored: `TP53+MYC` combines two genes, `TP53+ctrl` uses only `TP53`. If none of
a perturbation's component genes are present in a reference, that block stays
zero; when only some components are present, aggregation uses the matched
components.

### The co-essentiality blocks

The 96 trailing columns are the only non-annotation signal in the model. They
come from the DepMap CRISPR gene-effect matrix: genes are scored across 1,186
cell lines, the matrix is centred per gene, and a truncated SVD gives each gene
a rank-64 and a rank-32 coordinate vector. Two genes are close in this space
when knocking them out has correlated fitness consequences across cell lines,
which is a functional-similarity signal that pathway membership does not carry.

Both blocks are included; the rank-32 block is not a subspace of the rank-64
block, because each is fitted independently. Provenance and rebuild
instructions are in
[`scripts/gen_embeddings/README.md`](scripts/gen_embeddings/README.md).

### DepMap cell-line selection

The `depmap` block is cell-line specific. `resolve_depmap_line(donor_id)`
accepts either a cell-line name or a staged-dataset identifier:

| `donor_id` | DepMap table |
|---|---|
| `K562`, `replogle22k562` | `K562.pq` |
| `RPE1`, `replogle22rpe1` | `RPE1.pq` |
| `HepG2`, `nadig25hepg2` | `HepG2.pq` |
| `Jurkat`, `nadig25jurkat` | `Jurkat.pq` |
| `A549`, `iPSC`, `melanoma`, `H1_hESC` | `K562.pq` |
| Missing or unrecognized value | `K562.pq` |

**This is a behavioural change from v1.** The v1 map had no HepG2 or Jurkat
entry and no dataset identifiers, so those datasets silently received K562
essentiality values. The sealed v2 run used the dataset-correct line, and this
package does the same. Predictions for HepG2 and Jurkat therefore differ from v1
in five columns of the feature matrix even before any head change.

## Prediction pipeline

```python
Y_pred = head_predict(
    E_full,
    train_idx,
    Y_train,
    cfg,
    test_idx=None,
)
```

`head_predict` is the whole model. Nothing in it is fitted on anything but
`train_idx` rows.

### 1. Scale from training rows

The scaler is fitted on `E_full[train_idx]` and applied to both training and
prediction rows, so test-row values never enter the scaler. Both shipped heads
use `robust_25_75` (`RobustScaler` on the 25th–75th percentiles). `standard` and
`none` are implemented; any other value raises.

### 2. Weight the co-essentiality columns

The trailing 96 columns are multiplied by `cfg.emb_weight` **after** scaling, so
the weight is a pure amplitude knob that the column-equivariant scaler cannot
absorb. The shared head uses 1.15; the hit head uses 1.0, which is a no-op.

### 3. Shape the target

- `raw_delta` passes `Y_train` through unchanged. Used by the hit head.
- `residual_pert_mean` subtracts the training-row mean of every target column
  and restores it in step 7. Used by the shared head.

### 4. Ridge arm

Either a fixed-alpha `Ridge` (shared head, alpha 10) or `RidgeCV` over
`ridge_alphas` (hit head). With `y_standardize=True`, target columns are
standardized for fitting and converted back afterwards. Both shipped heads
standardize.

### 5. Nearest-neighbour arm

This is the dominant arm in both heads and carries the v2 refinements.

**Supervised feature weighting** (shared head only). Most of the 8,700 columns
say nothing about the response. The training target block is reduced to
`knn_feat_rank` SVD components (512), each feature column's correlation with
that reduced block is computed, the per-column norm of those correlations is
normalised to mean 1, and the result is raised to `knn_feat_pow` (6.0). Features
are multiplied by these weights before the metric is evaluated, which makes
neighbour selection target-aware. The SVD is fitted on training targets only.
The hit head sets `knn_feat_weight='none'` and skips this entirely.

**Neighbour selection.** Rows are L2-normalised and cosine `NearestNeighbors`
returns the `knn_K` nearest training perturbations — 80 for the shared head, 13
for the hit head.

**Tie expansion** (shared head only, `knn_tie_break='expand'`). The query asks
for 32 extra neighbours, and every training row within the K-th neighbour's
distance is kept, using a float32-scale epsilon. Distances that differ only by
SIMD reduction order therefore produce the same neighbour set on every CPU. The
hit head uses `'none'`: exactly the first K rows.

**Kernel weights.** Weights are `exp(-knn_tau * d)`. With
`knn_adaptive_bw='mean'` (shared head), each row's distances are first divided
by that row's own mean retained-neighbour distance, and the resulting scales are
renormalised by their median across rows, so `knn_tau` keeps its global meaning
and a typical row is unchanged. Rows in unusually sparse regions get a wider
kernel rather than collapsing onto their single closest neighbour. The hit head
uses `'none'`.

The retained weights are normalised to sum to 1 and applied to the neighbours'
targets.

### 6. Binary-response arm

Each target column is binarised at the top `top_p` percent and `RidgeCV` is fit
to the result. The signed variant (shared head) ranks by absolute magnitude and
keeps the sign; the unsigned variant (hit head) marks only the largest positive
values.

### 7. Blend and restore

Arms are combined with the fixed weights in `cfg.weights` — `0.10 / 0.85 / 0.05`
for the shared head, `0.10 / 0.65 / 0.25` for the hit head. With
`y_z_score_arms=True` (hit head) each arm is z-scored across the prediction rows
first. Arms with zero weight are never fitted.

For `residual_pert_mean`, the blended output is multiplied by the amplitude
factor `shrink` (1.25) and the removed per-column mean is added back. The factor
is above 1 because the kNN-dominant blend regresses toward the mean; 1.25
restores the response amplitude. For `raw_delta` with `shrink=1.0` the blend is
returned unchanged.

When `test_idx` is supplied, prediction and every cross-row operation — per-arm
z-scoring and the adaptive bandwidth's median normalisation — are restricted to
those rows. The Systema adapter uses this mode; the other adapters predict every
row and let their output wrapper select the records they need. This is the one
place where the shipped model's output for a row depends on which other rows are
in the same batch; it is documented in
[`docs/leakage.md`](docs/leakage.md).

## The two heads

| Parameter | `SHARED_HEAD` | `HIT_HEAD` |
|---|---|---|
| Slots | `cellsim`, `scperturb`, `perturbhd_reg`, `systema` | `perturbhd_hit` |
| `emb_weight` | 1.15 | 1.0 |
| `ridge_estimator` | `ridge_fixed`, alpha 10 | `ridgecv` over `(3, 10, 30, 100)` |
| `knn_K` | 80 | 13 |
| `knn_feat_weight` | `target_corr`, power 6.0, rank 512 | `none` |
| `knn_tie_break` | `expand` | `none` |
| `knn_adaptive_bw` | `mean` | `none` |
| `signed_binary` | true | false |
| `weights` (ridge/kNN/binary) | `0.10 / 0.85 / 0.05` | `0.10 / 0.65 / 0.25` |
| `y_z_score_arms` | false | true |
| `target_shape` | `residual_pert_mean` | `raw_delta` |
| `shrink` | 1.25 | 1.0 |

Every other field is identical, including the scaler, the feature space, the
kNN metric and temperature, `top_p`, the binary-arm alphas, and target
standardization. `HIT_HEAD.ridge_fixed_alpha` is recorded as 110.0 but is inert
under `ridge_estimator='ridgecv'`.

The four shared slots are the *same object*, not four equal copies, so they
cannot drift apart. `heads.assert_two_head()` is the executable form of the
claim and returns an empty dict on the shipped table:

```python
from tusoperturb import assert_two_head, head_deviation

assert assert_two_head() == {}
print(sorted(head_deviation()))   # the 14 fields above
```

## Public adapters

### `predict_regression`

Builds the feature matrix, predicts expression deltas for every staged
perturbation, adds the staged control mean, and delegates `AnnData` construction
to `perturb_2026.loop.helpers.build_pred_adata_from_matrix`. The `seed` argument
is accepted for API consistency but unused.

### `predict_regression_gene`

Runs the same head and returns a long-form `DataFrame` with `pert`, `gene`, and
`effect` columns. It does not use the external AnnData helper, but feature
construction still requires the staged loader. `master` and `seed` are retained
in the signature for compatibility.

### `predict_hit`

Loads staged perturbation features and a per-dataset phenotype table, trains on
rows labeled `train` in `split-{seed}`, and returns scores for rows labeled
`test` or `val`, as `pert`, `pheno`, and `hit_score`.

### `predict_systema`

Builds features for `panel_master.all_perts`, trains on
`panel_master.train_perts`, and predicts `panel_master.test_perts` with
`test_idx` set. Output rows follow `test_perts`; columns follow the target order
in `panel_master.Y_train_post`.

## Custom configurations

```python
from dataclasses import replace
from tusoperturb import HEAD_CONFIGS, head_predict

cfg = replace(HEAD_CONFIGS["cellsim"], knn_K=40)
Y_pred = head_predict(E, train_idx, Y_train, cfg)
```

`HeadConfig` is frozen and validates on construction: blend weights must sum to
1, and every enumerated field must hold a supported value. A custom
configuration changes the method and will not be comparable to the values in
[`champion/params.json`](champion/params.json).

## Numerical behavior

No shipped prediction path performs stochastic sampling; results are
deterministic for fixed inputs and dependency versions. The sealed run pinned
`OPENBLAS_CORETYPE=Haswell` before the first NumPy import and reproduced
bit-identical outputs across machines. Without that pin, small floating-point
differences can appear across BLAS builds, CPU instruction sets, and
scikit-learn versions. The kNN tie-expansion epsilon exists to keep the
neighbour set — a discrete choice, and the one place where a floating-point
difference could produce a visibly different prediction — stable regardless.
