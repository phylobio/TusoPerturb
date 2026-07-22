# scPerturBench adapter

TusoPerturb uses `predict_regression` for scPerturBench. The model path is the
same as the CellSimBench adapter; the difference is the external scorer used to
evaluate the resulting `AnnData` object.

## API

```python
from tusoperturb.api import predict_regression

pred_adata = predict_regression(
    master,
    dataset="replogle22k562",
    seed=1,
    head="scperturb",
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

`scperturb`, `cellsim`, and `perturbhd_reg` point to the same fixed ridge
configuration:

- full 11,680-dimensional shared feature matrix;
- standard feature scaling;
- alpha 110;
- block weights for GO Biological Process, STRING, and DepMap; and
- no nearest-neighbor or binary-response blend.

Given the same staged inputs, the `cellsim` and `scperturb` calls produce the
same numerical prediction matrix before benchmark-specific scoring.

## Required inputs

This adapter requires:

1. a benchmark `master` object accepted by
   `perturb_2026.loop.helpers.build_pred_adata_from_matrix`;
2. the external `perturb_2026` package;
3. a staged dataset loadable through
   `perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`; and
4. the bundled reference features or equivalent path overrides.

The required stage keys and ordering constraints are documented in
[`REPRODUCE.md`](../../REPRODUCE.md#perturb_2026-stage-loader).

## Example

```python
import anndata as ad
from tusoperturb.api import predict_regression

master = ad.read_h5ad("/path/to/replogle22k562.h5ad")

pred_adata = predict_regression(
    master,
    "replogle22k562",
    seed=1,
    head="scperturb",
)

pred_adata.write_h5ad("scperturb-replogle22k562.h5ad")
```

The current regression implementation does not use `seed`. Ensure that the
staged targets, AnnData output helper, and scorer all use the intended
scPerturBench split and preprocessing.

## Scoring

Run the scPerturBench scorer on the generated `AnnData` object. Metric names,
DE-gene selection, and aggregation are owned by that external scorer and may
vary across revisions, so record the scorer version with any reported result.

The project-recorded values are listed in
[`report.md`](../../report.md#scperturbench).
