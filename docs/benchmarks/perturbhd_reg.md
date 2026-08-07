# PerturbHD regression adapter

`predict_regression_gene` runs the same head as the expression-response
adapters, but returns a long-form table instead of an `AnnData` object.

## API

```python
from tusoperturb.api import predict_regression_gene

pred_df = predict_regression_gene(
    master,
    dataset="nadig25hepg2",
    seed=1,
    head="perturbhd_reg",
)
```

The result contains:

| Column | Description |
|---|---|
| `pert` | Perturbation label. |
| `gene` | Gene symbol, from the staged `gene_names`. |
| `effect` | Predicted gene-level effect. |

Supported dataset identifiers are:

```text
nadig25hepg2
nadig25jurkat
replogle22k562
replogle22rpe1
```

## Model configuration

`perturbhd_reg` resolves to the shared head, the same object used by `cellsim`,
`scperturb`, and `systema`. The adapter differs only in output shape: it does
not build an `AnnData` object and does not call the external output helper.

See [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for the pipeline.

## Recorded results

| Metric | Direction | hepg2 | jurkat | k562 | rpe1 | v2 better |
|---|---|---|---|---|---|---|
| `corr` | higher | 0.3901 / 0.3471 | 0.3507 / 0.3049 | 0.3704 / 0.3399 | 0.544 / 0.4918 | 4/4 |
| `corr_top100` | higher | 0.538 / 0.4909 | 0.5296 / 0.477 | 0.5299 / 0.4959 | 0.6687 / 0.6247 | 4/4 |
| `mae` | lower | 0.04861 / 0.05279 | 0.04428 / 0.04862 | 0.03581 / 0.03846 | 0.04889 / 0.05371 | 4/4 |
| `mae_top100` | lower | 0.1965 / 0.2023 | 0.1724 / 0.1733 | 0.1336 / 0.1353 | 0.1941 / 0.2093 | 4/4 |
| `mse` | lower | 0.005314 / 0.006208 | 0.004256 / 0.005025 | 0.002872 / 0.003233 | 0.00601 / 0.007137 | 4/4 |
| `mse_top100` | lower | 0.05976 / 0.0643 | 0.04683 / 0.04851 | 0.03054 / 0.03139 | 0.06804 / 0.07765 | 4/4 |

Each cell is **TusoPerturb v2 / old TusoPerturb**, mean over 3 seeds. v2 is better in 24 of 24 scored cells.

`corr` and `corr_top100` are correlations against the measured effects, the
latter restricted to the 100 largest-magnitude genes per perturbation. `mae`,
`mse`, and their `_top100` variants are error metrics. Standard deviations are
in [`champion/params.json`](../../champion/params.json).

## Required inputs

### Staged dataset

The adapter uses the same shared feature builder as the expression workflows and
therefore requires a supported stage from
`perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`, plus the bundled
reference features. It does not require the AnnData output helper.

### Benchmark master

`master` is retained in the signature for consistency with the other adapters
but is not inspected.

## Example

```python
from tusoperturb.api import predict_regression_gene

pred_df = predict_regression_gene(
    None,
    "nadig25hepg2",
    seed=1,
    head="perturbhd_reg",
)

assert list(pred_df.columns) == ["pert", "gene", "effect"]
pred_df.to_parquet("perturbhd-reg-nadig25hepg2.parquet", index=False)
```

The regression model does not use `seed`.

## Scoring

Pass the long-form table to the PerturbHD regression scorer for the same dataset
release and split definition. Confirm that `gene` values match the scorer's gene
universe; genes present in the prediction but absent from the reference, or the
reverse, silently change the evaluated set.
