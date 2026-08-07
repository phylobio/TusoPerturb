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

This is the only key that uses the second head. It reads the same 8,700-column
feature matrix as every other key, with the co-essentiality columns at unit
amplitude. After robust scaling it combines:

- a `RidgeCV` model with weight 0.10;
- unweighted cosine nearest-neighbour target transfer at K=13, weight 0.65; and
- an unsigned top-2% binary-response ridge model, weight 0.25;

with each arm z-scored across the prediction rows before blending, on the raw
target and with no amplitude factor.

The hit head differs from the shared head in exactly 14 fields, all of them head
parameters; the feature space is identical.
[`heads.head_deviation()`](../../tusoperturb/heads.py) returns that list, and
`heads.assert_two_head()` fails if the two heads ever diverge in anything else.

## Recorded results

| Metric | Direction | hepg2 | jurkat | k562 | rpe1 | v2 better |
|---|---|---|---|---|---|---|
| `recall_at_budget_0.05` | higher | 0.4953 / 0.468 | 0.3906 / 0.3772 | 0.4833 / 0.4787 | 0.484 / 0.45 | 4/4 |
| `recall_at_fdr_0.20` | higher | 0.1333 / 0.1173 | 0.0549 / 0.0551 | 0.094 / 0.08867 | 0.1527 / 0.1527 | 3/4 |

Each cell is **TusoPerturb v2 / old TusoPerturb**, mean over 3 seeds. v2 is better in 7 of 8 scored cells.

`recall_at_budget_0.05` is recall among the top 5% of ranked candidates;
`recall_at_fdr_0.20` is recall at a 20% false-discovery threshold. Both are
higher-better. Standard deviations are in
[`champion/params.json`](../../champion/params.json).

## Required inputs

### Staged dataset

The adapter uses the same shared feature builder as the expression workflows. It
therefore requires a supported stage from
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

If it is unset, `predict_hit` raises a `FileNotFoundError` naming the expected
file. Every other workflow runs without it.

Each file must contain:

| Column | Description |
|---|---|
| `pert` | Perturbation label. |
| `pheno` | Phenotype label. |
| `mean_diff` | Training target value. |
| `split-1`, `split-2`, `split-3` | Split labels containing `train`, `val`, or `test`. |

Each `(pert, pheno)` pair should be unique. The implementation removes all rows
belonging to duplicated pairs before pivoting the table.

Use `seed=1`, `seed=2`, or `seed=3`. Other values fall back to `split-1`.

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

The implementation does not inspect `master`, but the argument remains part of
the public signature for consistency with the other adapters.

Only perturbations present in both the phenotype table and the staged feature
matrix are used. Returned rows are restricted to the selected split's `val` and
`test` labels.

## Scoring

Pass the long-form table to the PerturbHD hit-prediction scorer for the same
phenotype-table and split release. Validate the intersection of perturbations
before interpreting a score; silently missing labels reduce the evaluated set.
