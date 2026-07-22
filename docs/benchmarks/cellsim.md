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

The CellSimBench key uses the full 11,680-dimensional shared feature matrix. It
fits a standard-scaled ridge model with alpha 110 after applying these feature
block weights:

```text
go_bp × 0.67
string × 1.5
depmap × 10.0
```

The same numerical configuration is used by the `scperturb` and
`perturbhd_reg` keys. See [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for the
complete feature layout and prediction pipeline.

## Required inputs

### Benchmark master

`master` is passed to
`perturb_2026.loop.helpers.build_pred_adata_from_matrix`. TusoPerturb does not
inspect it directly. Use the `AnnData` schema and preprocessing expected by the
CellSimBench scorer and the external output helper.

### Staged dataset

`build_shared_features(dataset)` calls
`perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`. The returned mapping
must provide the GenePT matrix, training targets, perturbation ordering,
control mean, and gene names documented in
[`REPRODUCE.md`](../../REPRODUCE.md#perturb_2026-stage-loader).

The staged dataset and `perturb_2026` package are not included in this
repository.

### Bundled references

Reactome, GO Biological Process, Hallmark, PROGENy, CollecTRI, STRING, and
DepMap features are read from `data/embeddings/` by default. Set
`TUSOPERTURB_REF_DIR` or `TUSOPERTURB_DEPMAP_DIR` before starting Python when
using another location.

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

The current regression model does not use `seed`; it is retained so benchmark
call sites can share a common signature. Split selection and final AnnData
formatting are handled by the external staging and output utilities.

## Scoring

Score `pred_adata` with the CellSimBench evaluation code that corresponds to
your dataset release. Before scoring, verify that:

- perturbation labels match the observed data;
- gene order and identifiers match;
- prediction values are finite; and
- the same preprocessing and split definitions are used for prediction and
  evaluation.

The project-recorded CellSimBench values are summarized in
[`report.md`](../../report.md#cellsimbench). They are not bundled as executable
scorer tests.
