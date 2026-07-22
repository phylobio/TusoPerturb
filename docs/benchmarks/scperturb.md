# scPerturBench (Wei 2024)

## What it measures

scPerturBench evaluates per-perturbation delta predictions with three
core metrics: `pcc_delta` (Pearson correlation of predicted vs. observed
delta on top DEGs — the primary), `mse` (mean squared error), and
`common_degs` (overlap of predicted and observed top-DEG sets).

**Head**: `predict_regression(master, dataset, seed, head='scperturb')`
**HeadConfig**: `PRIOR_A_v2_193` — the same config as CellSim. The output
numerical values are identical to CellSim's `pred_adata`; scPerturBench
differs only in how it scores them.
**Champion metric**: `pcc_delta = 0.682` on `replogle22k562`.

## Data you need to obtain

### Master h5ad

The same 4 masters as CellSim (`nadig25hepg2`, `nadig25jurkat`,
`replogle22k562`, `replogle22rpe1`), same sizes, same schema requirements.
scPerturBench's split code may differ from CellSim's — check the
scPerturBench paper for its train/test partition.

### The `gpu_stage/<dataset>/` bake

Same schema as documented in
[cellsim.md § The gpu_stage bake](cellsim.md#the-gpu_stagedataset-bake).
If you built the bake for CellSim, it's reusable for scPerturB provided the
train/test split you baked matches scPerturBench's expected partition.
(In the champion run, the same bake was used for both.)

### The `perturb_2026` module

Same requirement as CellSim. `predict_regression` calls
`build_pred_adata_from_matrix` regardless of `head=`.

## Running the head

```python
import anndata as ad
from tusoperturb.api import predict_regression

master = ad.read_h5ad("/path/to/masters/replogle22k562.h5ad")
pred_adata = predict_regression(master, "replogle22k562", seed=1, head="scperturb")
```

Run over all 4 datasets × 3 seeds. Runtime is the same as CellSim
(2-5 min per prediction, dominated by the shared feature build).

## Scoring your prediction

Feed `pred_adata` into scPerturBench's scorer, which returns `pcc_delta`,
`mse`, and `common_degs` per perturbation and aggregated.

scPerturBench code + scorer: see the scPerturBench GitHub repo referenced
in Wei et al. 2024.

## Champion parity check

```python
assert abs(pcc_delta_replogle22k562 - 0.682) < 5e-3
```

Exact value from the champion run: `0.6817649548329194`.

## Why the same config wins on both

`PRIOR_A_v2_193` is a plain Ridge (α=110) on RobustScaler-normalized
features with per-block weights that favor the annotation blocks
(Reactome/GO/CollecTRI) over the noisier DepMap/baseline blocks. That
combination is optimal for delta-from-control regression regardless of
which downstream metric-set (CellSim vs. scPerturB) you score against —
the underlying prediction problem is the same. This is why the
`predict_regression(head=...)` argument mostly serves as a documentation
alias: `head='cellsim'`, `head='scperturb'`, and `head='perturbhd_reg'`
all resolve to `PRIOR_A_v2_193`.
