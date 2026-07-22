# PerturbHD regression adapter

`predict_regression_gene` exposes the shared expression-regression model in the
long-form table expected by the PerturbHD regression workflow.

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

The result has three columns:

| Column | Description |
|---|---|
| `pert` | Perturbation label. |
| `gene` | Target gene label. |
| `effect` | Predicted expression delta from the staged control baseline. |

There is one row for every combination of staged perturbation and target gene,
so the table can be large.

## Model configuration

The adapter uses the same full-feature, standard-scaled ridge configuration as
CellSimBench and scPerturBench. Only output formatting differs.

## Required inputs

Feature construction requires a supported staged dataset and the external
`perturb_2026.loop.gpu_stage_loader.load_stage` function. The required stage
keys are documented in
[`REPRODUCE.md`](../../REPRODUCE.md#perturb_2026-stage-loader).

Unlike `predict_regression`, this adapter does not call the external AnnData
output helper. The current implementation retains `master` and `seed` in its
signature for API compatibility but does not otherwise use them.

Supported dataset identifiers are:

```text
nadig25hepg2
nadig25jurkat
replogle22k562
replogle22rpe1
```

## Example

```python
import anndata as ad
from tusoperturb.api import predict_regression_gene

master = ad.read_h5ad("/path/to/nadig25hepg2.h5ad")

pred_df = predict_regression_gene(
    master,
    "nadig25hepg2",
    seed=1,
)

assert list(pred_df.columns) == ["pert", "gene", "effect"]
pred_df.to_parquet("perturbhd-reg-nadig25hepg2.parquet", index=False)
```

For large datasets, write the result to Parquet promptly rather than retaining
additional copies in memory.

## Scoring

Pass the table to the PerturbHD regression scorer associated with your benchmark
release. Confirm that the scorer expects expression deltas rather than
post-perturbation expression and that its perturbation and gene identifiers
match the staged data.

The project-recorded values are summarized in
[`report.md`](../../report.md#perturbhd-regression).
