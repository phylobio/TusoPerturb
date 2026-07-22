# CellSimBench (Miller 2025)

## What it measures

CellSimBench evaluates per-perturbation delta predictions against three
similarity-flavored metrics: `pearson_deltactrl_degs` (Pearson correlation
of predicted vs. observed delta-from-control on the top DEGs),
`knn_jaccard_deltapert`, `nir` (normalized improvement ratio), and
`pds` (pathway distance score). TusoPerturb reports only
`pearson_deltactrl_degs` as its primary metric — see `report.md §8` for the
rationale on which metrics are computed vs. inherited from the published
baselines.

**Head**: `predict_regression(master, dataset, seed, head='cellsim')`
**HeadConfig**: `PRIOR_A_v2_193` (Ridge α=110 on RobustScaler + block-weighted features, plain ridge arm, no kNN/binary blend, full 11,680-D feature stack).
**Champion metric**: `pearson_deltactrl_degs = 0.6423` on `nadig25hepg2`.

## Data you need to obtain

### Master h5ad

| Dataset | Source paper | Approx size | What it is |
|---|---|---|---|
| `nadig25hepg2` | Nadig et al. 2025 | ~14 GB | scRNA-seq perturb-seq master for HepG2 with essential gene knockouts. |
| `nadig25jurkat` | Nadig et al. 2025 | ~15 GB | Same, Jurkat. |
| `replogle22k562` | Replogle et al. 2022 | ~18 GB | Genome-wide perturb-seq master for K562. |
| `replogle22rpe1` | Replogle et al. 2022 | ~20 GB | Genome-wide perturb-seq master for RPE1. |

Each master `h5ad` must expose the columns that CellSimBench's own scorer
depends on (perturbation labels in `.obs`, gene symbols in `.var`, log-norm
counts in `.X`). See the CellSimBench paper + code for the schema.

### The `gpu_stage/<dataset>/` bake

`feature_builder.build_shared_features()` calls
`perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`. That function
reads a pre-staged feature bake with this schema:

```
gpu_stage/<dataset>/
├── E_all_genept.npy      (n_all_perts, 3072) float32 GenePT embedding per pert
├── Y_train.npy           (n_train_perts, n_genes) float32 target delta from control
├── A_train.npy           (n_train_perts, n_all_perts) float32 (unused by the champion; kept for API parity)
├── all_perts.json        list[str] of pert names for E_all rows
├── train_perts.json      list[str] of pert names for Y_train rows
├── donor_id.txt          'HepG2' / 'Jurkat' / 'K562' / 'RPE1'
├── mean_baseline.npy     (n_genes,) float32 control-mean log-norm expression
├── gene_names.json       list[str] gene symbols for Y_train / mean_baseline columns
```

Building this bake yourself:

1. Read the master h5ad with `anndata.read_h5ad`.
2. Split perturbations into `train_perts` and `test_perts` per the paper's
   split (see benchmark-specific split code).
3. Compute per-perturbation delta from control: for each pert `p`, take the
   mean of `.X[obs.pert == p]` minus the mean of the control cells; that's
   your `Y_train` row.
4. Load or regenerate the GenePT embedding per pert (see
   [`../../scripts/gen_embeddings/README.md`](../../scripts/gen_embeddings/README.md);
   or use the cached values from GenePT's public release keyed by gene symbol).
5. Save each file at the paths above.

Alternative: get the bake directly from the perturb-2026 monorepo staging pipeline
(not distributed with this zip).

### The `perturb_2026` module

`feature_builder.py` and `api.py` do lazy `from perturb_2026.…` imports.
You need this module on `PYTHONPATH`:

- `perturb_2026.loop.gpu_stage_loader.load_stage(dataset)` — reads the bake above.
- `perturb_2026.loop.helpers.build_pred_adata_from_matrix(X, all_perts, donor_id, master, fold=…)` — shapes prediction into a pred-AnnData for CellSimBench's scorer.
- `perturb_2026.loop.paths.FOLD` — a constant tag identifying which fold the pred-AnnData belongs to.

## Running the head

```python
import anndata as ad
from tusoperturb.api import predict_regression

master = ad.read_h5ad("/path/to/masters/nadig25hepg2.h5ad")
pred_adata = predict_regression(master, "nadig25hepg2", seed=1, head="cellsim")

# pred_adata is an AnnData with post-expression predictions for the test perts.
# Shape: (n_test_perts, n_genes). obs.pert holds perturbation labels;
# var_names holds gene symbols.
```

Run over all 4 datasets × 3 seeds (12 predictions total). Each takes 2-5
minutes once the gpu_stage bake is warm on local disk.

## Scoring your prediction

Feed `pred_adata` into CellSimBench's scorer. The scorer expects a
paired (predicted, observed) AnnData with matching pert/gene indices and
returns the CellSim metric set including `pearson_deltactrl_degs`.

CellSimBench code + scorer: see the CellSimBench GitHub repo referenced in
Miller et al. 2025.

## Champion parity check

If your reproduction is faithful:

```python
assert abs(pearson_deltactrl_degs_hepg2 - 0.6423) < 5e-4
```

Exact value from the champion run: `0.6423267403536010`.
