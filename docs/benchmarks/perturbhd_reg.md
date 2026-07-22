# PerturbHD-regression (Vinas 2025)

## What it measures

PerturbHD-regression scores per-perturbation × per-gene delta predictions
via Pearson correlation on the training-perturbation set. Unlike CellSim /
scPerturB, the scorer consumes a **long-format `(pert, gene, effect)`
DataFrame** rather than a paired pred-AnnData.

**Head**: `predict_regression_gene(master, dataset, seed, head='perturbhd_reg')`
**HeadConfig**: `PRIOR_A_v2_193` — same numerical config as CellSim /
scPerturB. Only the output shaping differs.
**Champion metric**: `pearson` on train_pert set, all 4 datasets rank #1
against the published baselines.

## Data you need to obtain

### Master h5ad + gpu_stage bake

Same 4 datasets and same gpu_stage schema as
[cellsim.md](cellsim.md#the-gpu_stagedataset-bake). If you've already
built the bake for CellSim or scPerturB, it's directly reusable here.

### The `perturb_2026` module

The `_gene` variant does not call `build_pred_adata_from_matrix` (it emits
a plain long-format DataFrame), but the shared feature builder still needs
`perturb_2026.loop.gpu_stage_loader.load_stage`. So the module must be on
`PYTHONPATH`.

## Running the head

```python
import anndata as ad
from tusoperturb.api import predict_regression_gene

master = ad.read_h5ad("/path/to/masters/nadig25hepg2.h5ad")
pred_df = predict_regression_gene(master, "nadig25hepg2", seed=1, head="perturbhd_reg")

# pred_df columns: ['pert', 'gene', 'effect']
# One row per (pert, gene) with `effect` = predicted delta from control.
# Length: n_all_perts × n_genes.
print(pred_df.head())
```

Run over all 4 datasets × 3 seeds.

## Scoring your prediction

Feed `pred_df` into the PerturbHD `SEval` scorer (`run_full_benchmark_v2.py`
in the perturb-hd repo). The scorer joins on `(pert, gene)`, computes
per-perturbation Pearson correlations against the ground-truth delta, and
aggregates over the train_pert set.

perturb-hd code + scorer: see the perturb-hd GitHub repo referenced in
Vinas 2025.

## Why the long-format output

CellSim / scPerturB's paired-AnnData scoring assumes the prediction was
generated for the specific train/test split that produced the observed
counts. PerturbHD-regression instead pools all perturbations and asks:
"across every pert in the training set, how well does your predicted
delta correlate with the observed delta at the gene level?" The
long-format `(pert, gene, effect)` shape is the direct input the perturb-hd
scorer expects.

## Champion parity check

Per-dataset expected Pearson values are in the published PerturbHD leaderboard
(rank #1 in all 4 datasets under `PRIOR_A_v2_193`). No single-number assert
is documented for the champion run because PerturbHD-reg is scored as a
distribution over train_perts rather than a single scalar.
