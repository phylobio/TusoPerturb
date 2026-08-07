# CellSimBench adapter

TusoPerturb uses `predict_regression` for CellSimBench. The adapter predicts an
expression delta for every staged perturbation, adds the staged control mean,
and delegates construction of the benchmark-compatible `AnnData` object to the
external `perturb_2026` package.

## API

```python
from tusoperturb.api import predict_regression

pred_adata = predict_regression(
    master,
    dataset="nadig25hepg2",
    seed=1,
    head="cellsim",
)
```

Supported dataset identifiers are:

```text
nadig25hepg2
nadig25jurkat
replogle22k562
replogle22rpe1
```

## Model configuration

CellSimBench uses the shared head on the full 8,700-dimensional feature matrix:
robust scaling, the co-essentiality columns at amplitude 1.15, and a
`0.10 / 0.85 / 0.05` blend of fixed-alpha ridge, target-weighted kNN at K=80,
and a signed top-2% binary ridge, predicting the residual over the training
perturbation mean and restoring amplitude with a factor of 1.25.

The `scperturb`, `perturbhd_reg`, and `systema` keys resolve to the same object.
See [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for the complete feature layout
and prediction pipeline.

## Recorded results

| Metric | Direction | hepg2 | jurkat | k562 | rpe1 | v2 better |
|---|---|---|---|---|---|---|
| `mae` | lower | 0.06346 / 0.06679 | 0.06234 / 0.06569 | 0.04569 / 0.04803 | 0.06313 / 0.06684 | 4/4 |
| `mae_degs` | lower | 0.1472 / 0.1463 | 0.1499 / 0.1467 | 0.1104 / 0.1101 | 0.1794 / 0.1813 | 1/4 |
| `mse` | lower | 0.008076 / 0.008916 | 0.007705 / 0.008485 | 0.004196 / 0.004584 | 0.008563 / 0.009512 | 4/4 |
| `mse_degs` | lower | 0.04686 / 0.04645 | 0.04756 / 0.04615 | 0.02905 / 0.02847 | 0.07311 / 0.07419 | 1/4 |
| `pearson_deltactrl` | higher | 0.313 / 0.2796 | 0.2807 / 0.2453 | 0.2943 / 0.2687 | 0.4373 / 0.403 | 4/4 |
| `pearson_deltactrl_degs` | higher | 0.4074 / 0.3502 | 0.3944 / 0.3536 | 0.3814 / 0.3506 | 0.4646 / 0.4344 | 4/4 |
| `pearson_deltapert` | higher | 0.2977 / 0.2685 | 0.232 / 0.2135 | 0.2808 / 0.2587 | 0.3718 / 0.3179 | 4/4 |
| `pearson_deltapert_degs` | higher | 0.3533 / 0.3223 | 0.3053 / 0.2967 | 0.348 / 0.3307 | 0.4534 / 0.3996 | 4/4 |
| `r2_deltactrl` | higher | 0.0421 / -0.06928 | 0.02917 / -0.0953 | 8.204e-04 / -0.1164 | 0.08404 / -0.02454 | 4/4 |
| `r2_deltactrl_degs` | higher | -0.2002 / -0.2385 | -0.1627 / -0.1897 | -0.09536 / -0.1326 | -0.0126 / -0.06796 | 4/4 |
| `r2_deltapert` | higher | 0.03031 / -0.08301 | 0.009615 / -0.1177 | -0.001492 / -0.1214 | 0.08093 / -0.04047 | 4/4 |
| `r2_deltapert_degs` | higher | -0.2047 / -0.2212 | -0.175 / -0.1888 | -0.1059 / -0.1255 | -0.005747 / -0.05807 | 4/4 |
| `weighted_r2_deltactrl` | higher | 0.09414 / 0.07271 | 0.09227 / 0.06823 | 0.1029 / 0.08457 | 0.1228 / 0.09241 | 4/4 |
| `weighted_r2_deltapert` | higher | 0.1642 / 0.142 | 0.1158 / 0.09346 | 0.143 / 0.1268 | 0.2324 / 0.1855 | 4/4 |
| `wmae` | lower | 0.1265 / 0.1253 | 0.1303 / 0.1284 | 0.09664 / 0.09568 | 0.1384 / 0.1392 | 1/4 |
| `wmse` | lower | 0.03535 / 0.03553 | 0.03866 / 0.03868 | 0.02568 / 0.02541 | 0.04991 / 0.05074 | 3/4 |

Each cell is **TusoPerturb v2 / old TusoPerturb**, mean over 1 seed. v2 is better in 54 of 64 scored cells.

Values are the sealed-test scores; `direction` says whether higher or lower is
better. The full set including standard deviations is in
[`champion/params.json`](../../champion/params.json).

## Required inputs

### Benchmark master

`master` is passed to
`perturb_2026.loop.helpers.build_pred_adata_from_matrix`. TusoPerturb does not
inspect it directly. Use the `AnnData` schema and preprocessing expected by the
CellSimBench scorer and the external output helper.

### Staged dataset

`build_shared_features(dataset)` calls
`perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`. The returned mapping
must provide the training targets, perturbation ordering, control mean, and gene
names documented in
[`REPRODUCE.md`](../../REPRODUCE.md#perturb_2026-stage-loader). No embedding
block is read from the stage.

The staged dataset and `perturb_2026` package are not included in this
repository.

### Bundled references

Reactome, GO Biological Process, Hallmark, PROGENy, CollecTRI, STRING, DepMap,
and co-essentiality features are read from `data/embeddings/` by default. Set
`TUSOPERTURB_REF_DIR`, `TUSOPERTURB_DEPMAP_DIR`, or `TUSOPERTURB_COESS_DIR`
before starting Python when using another location.

## Example

```python
import anndata as ad
from tusoperturb.api import predict_regression

master = ad.read_h5ad("/path/to/nadig25hepg2.h5ad")

pred_adata = predict_regression(
    master,
    "nadig25hepg2",
    seed=1,
    head="cellsim",
)

pred_adata.write_h5ad("cellsim-nadig25hepg2.h5ad")
```

The regression model does not use `seed`; it is retained so benchmark call sites
can share a common signature. Split selection and final AnnData formatting are
handled by the external staging and output utilities.

## Scoring

Score `pred_adata` with the CellSimBench evaluation code that corresponds to
your dataset release. Before scoring, verify that:

- perturbation labels match the observed data;
- gene order and identifiers match;
- prediction values are finite; and
- the same preprocessing and split definitions are used for prediction and
  evaluation.
