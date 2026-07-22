# PerturbHD hit-prediction adapter

`predict_hit` predicts a score for each perturbation and phenotype represented
in a PerturbHD mean-effect table. It trains on the rows assigned to the selected
split's training set and returns predictions for validation and test rows.

## API

```python
from tusoperturb.api import predict_hit

pred_df = predict_hit(
    master,
    dataset="nadig25hepg2",
    seed=1,
)
```

The result contains:

| Column | Description |
|---|---|
| `pert` | Perturbation label. |
| `pheno` | Phenotype label from the input table. |
| `hit_score` | Predicted phenotype score. |

## Model configuration

The hit-prediction configuration removes the 3,072-dimensional GenePT block and
uses the remaining 8,608 features. After robust scaling, it combines:

- a ridge model with weight 0.10;
- cosine nearest-neighbor target transfer with weight 0.65; and
- an unsigned top-2% binary-response ridge model with weight 0.25.

The configuration is defined in
[`tusoperturb/heads.py`](../../tusoperturb/heads.py) and described in
[`ARCHITECTURE.md`](../../ARCHITECTURE.md).

## Required inputs

### Staged dataset

The adapter uses the same shared feature builder as the expression workflows.
It therefore requires a supported stage from
`perturb_2026.loop.gpu_stage_loader.load_stage(dataset)` and the bundled
reference features.

### Phenotype Parquet file

Set `TUSOPERTURB_MEAN_EFFECT_DIR` to a directory containing one file per
dataset:

```text
<paper_key>-h.all-all.pq
```

| Dataset | `paper_key` |
|---|---|
| `nadig25hepg2` | `nadig_hepg2_essential_full` |
| `nadig25jurkat` | `nadig_jurkat_essential_full` |
| `replogle22k562` | `replogle_k562_essential_full` |
| `replogle22rpe1` | `replogle_rpe1_essential_full` |

Set the variable before starting Python:

```bash
export TUSOPERTURB_MEAN_EFFECT_DIR=/path/to/mean_effects_aucell
```

Each file must contain:

| Column | Description |
|---|---|
| `pert` | Perturbation label. |
| `pheno` | Phenotype label. |
| `mean_diff` | Training target value. |
| `split-1`, `split-2`, `split-3` | Split labels containing `train`, `val`, or `test`. |

Each `(pert, pheno)` pair should be unique. The current implementation removes
all rows belonging to duplicated pairs before pivoting the table.

Use `seed=1`, `seed=2`, or `seed=3`. Other values currently fall back to
`split-1`.

## Example

```python
import anndata as ad
from tusoperturb.api import predict_hit

master = ad.read_h5ad("/path/to/nadig25hepg2.h5ad")

pred_df = predict_hit(
    master,
    "nadig25hepg2",
    seed=1,
)

assert list(pred_df.columns) == ["pert", "pheno", "hit_score"]
pred_df.to_parquet("perturbhd-hit-nadig25hepg2-seed1.parquet", index=False)
```

The current implementation does not inspect `master`, but the argument remains
part of the public signature for consistency with the other benchmark adapters.

Only perturbations present in both the phenotype table and the staged feature
matrix are used. Returned rows are restricted to the selected split's `val` and
`test` labels.

## Scoring

Pass the long-form table to the PerturbHD hit-prediction scorer for the same
phenotype-table and split release. Validate the intersection of perturbations
before interpreting a score; silently missing labels reduce the evaluated set.

The project-recorded values are summarized in
[`report.md`](../../report.md#perturbhd-hit-prediction).
