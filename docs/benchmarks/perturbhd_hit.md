# PerturbHD-hit (Vinas 2025)

## What it measures

PerturbHD-hit reframes perturbation prediction as a **phenotype hit-scoring
task**. Instead of predicting a full expression delta, the model outputs
a per-perturbation × per-phenotype score (AUCell-derived), and the scorer
evaluates recall at a fixed budget (`recall_at_budget_0.05` = fraction of
true hits recovered in the top 5% of predictions).

**Head**: `predict_hit(master, dataset, seed)`
**HeadConfig**: `PRIOR_R17_C` — a 3-arm blend (weights `WR=0.10 / WK=0.65 / WB=0.25`)
of plain Ridge / kNN-weighted / top-2% binary arms on the `no_genept_8608`
subset (drops the GenePT block; a sweep showed it hurt this task).
**Champion metric**: `recall_at_budget_0.05` #1 on 4/4 PerturbHD-hit datasets.

## Data you need to obtain

### Master h5ad + gpu_stage bake

Same 4 datasets and same gpu_stage schema as
[cellsim.md](cellsim.md#the-gpu_stagedataset-bake).

### AUCell phenotype parquets

`predict_hit` reads a per-dataset AUCell parquet:

```
<MEAN_EFFECT_DIR>/<paper_key>-h.all-all.pq
```

where `<paper_key>` maps from dataset name to the perturb-hd naming:

| Dataset | `paper_key` |
|---|---|
| `nadig25hepg2` | `nadig_hepg2_essential_full` |
| `nadig25jurkat` | `nadig_jurkat_essential_full` |
| `replogle22k562` | `replogle_k562_essential_full` |
| `replogle22rpe1` | `replogle_rpe1_essential_full` |

Each parquet has columns:

| Column | Type | What it is |
|---|---|---|
| `pert` | str | Perturbation label. |
| `pheno` | str | Phenotype identifier (from MSigDB Hallmark H set). |
| `mean_diff` | float | Mean AUCell score difference vs. control. |
| `split-1`, `split-2`, `split-3` | str | Per-seed train/test/val assignment (one column per seed; values 'train', 'test', 'val'). |

By default, `MEAN_EFFECT_DIR` points at the original method-development
sandbox path. Outside that sandbox, set `TUSOPERTURB_MEAN_EFFECT_DIR` before
starting Python:

```bash
export TUSOPERTURB_MEAN_EFFECT_DIR=/your/local/mean_effects_aucell
```

If Python is already running, you can also override the feature-builder
constant directly:

```python
from pathlib import Path
import tusoperturb.feature_builder as fb
fb.MEAN_EFFECT_DIR = Path("/your/local/mean_effects_aucell")
```

Source: the PerturbHD supplementary data release accompanying Vinas 2025.
These are precomputed AUCell scores per (perturbation, Hallmark phenotype)
across each dataset.

### The `perturb_2026` module

Same requirement as CellSim. Feature builder still needs `load_stage`.

## Running the head

```python
import anndata as ad
from tusoperturb.api import predict_hit

# Before launching Python, set:
# export TUSOPERTURB_MEAN_EFFECT_DIR=/path/to/your/mean_effects_aucell
master = ad.read_h5ad("/path/to/masters/nadig25hepg2.h5ad")
pred_df = predict_hit(master, "nadig25hepg2", seed=1)

# pred_df columns: ['pert', 'pheno', 'hit_score']
# Restricted to test + val perts. One row per (pert, pheno).
print(pred_df.head())
```

Run over all 4 datasets × 3 seeds. Runtime: 30-60 s per prediction (fewer
targets than the regression heads: ~50 Hallmark phenotypes vs. ~8k genes).

## Scoring your prediction

Feed `pred_df` into the PerturbHD-hit scorer from `run_full_benchmark_v2.py`.
The scorer computes `recall_at_budget_{0.01, 0.05, 0.10}` per phenotype and
aggregates.

perturb-hd code + scorer: see the perturb-hd GitHub repo referenced in
Vinas 2025.

## Why the config differs from the regression heads

The hit task is a ranking problem, not a regression problem. Two things
change:

1. **Drop GenePT.** The LLM embedding hurts on hit — probably because the
   3072-D GenePT block overwhelms the per-phenotype signal in the other
   annotation blocks. The `no_genept_8608` subset uses the same feature
   layout as CellSim minus columns `[0, 3072)`.
2. **3-arm blend with a binary arm.** The binary arm z-scores each feature
   and thresholds at the top 2%, encoding "is this pert an outlier on this
   feature?" That kind of coarse categorical signal is more informative for
   hit ranking than the continuous ridge output alone.

See [`ARCHITECTURE.md §3`](../../ARCHITECTURE.md) for the full 3-arm math.
