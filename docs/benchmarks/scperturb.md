# scPerturBench adapter

TusoPerturb uses `predict_regression` with `head="scperturb"` for scPerturBench.
The adapter follows the same path as CellSimBench: predict an expression delta
for every staged perturbation, add the staged control mean, and delegate
`AnnData` construction to the external `perturb_2026` package.

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

The `scperturb` key resolves to the shared head — the same object as `cellsim`,
`perturbhd_reg`, and `systema`. With identical staged inputs, `head="cellsim"`
and `head="scperturb"` produce the same prediction matrix; the keys exist so
call sites and downstream scorers stay distinguishable.

See [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for the feature layout and the
prediction pipeline.

## Recorded results

| Metric | Direction | hepg2 | jurkat | k562 | rpe1 | v2 better |
|---|---|---|---|---|---|---|
| `common_degs` | higher | 0.06467 / 0.06576 | 0.0747 / 0.07396 | 0.09166 / 0.08892 | 0.1043 / 0.1078 | 2/4 |
| `mse` | lower | 0.03047 / 0.02984 | 0.03102 / 0.02966 | 0.02192 / 0.02126 | 0.06093 / 0.06172 | 1/4 |
| `pearson_distance` | lower | 0.666 / 0.698 | 0.6611 / 0.6948 | 0.66 / 0.6774 | 0.5778 / 0.6009 | 4/4 |

Each cell is **TusoPerturb v2 / old TusoPerturb**, mean over 1 seed. v2 is better in 7 of 12 scored cells.

`pearson_distance` is a distance, so lower is better; `common_degs` counts
recovered differentially expressed genes, so higher is better. Standard
deviations are in [`champion/params.json`](../../champion/params.json).

## Required inputs

Identical to [CellSimBench](cellsim.md#required-inputs): a benchmark `master`
object accepted by the external output helper, a staged dataset from
`perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`, and the bundled
reference features under `data/embeddings/`.

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

The regression model does not use `seed`.

## Scoring

Score `pred_adata` with the scPerturBench evaluation code matching your dataset
release. Confirm perturbation labels, gene order, and split definitions agree
between prediction and evaluation, and that predictions are finite.
