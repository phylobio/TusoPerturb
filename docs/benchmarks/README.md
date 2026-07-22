# Per-benchmark run instructions

TusoPerturb evaluates on 5 published benchmark buckets covering 11 unique
datasets. Each has its own scorer, its own data format, and its own external
dependencies. This directory has one page per benchmark.

## Benchmark index

| Benchmark | Head function | HeadConfig | External data | Instructions |
|---|---|---|---|---|
| **CellSimBench** (Miller 2025) | [`predict_regression`](../../tusoperturb/api.py) `head='cellsim'` | `PRIOR_A_v2_193` | master h5ad + gpu_stage bake + `perturb_2026` | [cellsim.md](cellsim.md) |
| **scPerturBench** (Wei 2024) | [`predict_regression`](../../tusoperturb/api.py) `head='scperturb'` | `PRIOR_A_v2_193` | master h5ad + gpu_stage bake + `perturb_2026` | [scperturb.md](scperturb.md) |
| **PerturbHD-regression** (Vinas 2025) | [`predict_regression_gene`](../../tusoperturb/api.py) `head='perturbhd_reg'` | `PRIOR_A_v2_193` | master h5ad + gpu_stage bake + `perturb_2026` | [perturbhd_reg.md](perturbhd_reg.md) |
| **PerturbHD-hit** (Vinas 2025) | [`predict_hit`](../../tusoperturb/api.py) | `PRIOR_R17_C` | master h5ad + gpu_stage bake + AUCell parquets + `perturb_2026` | [perturbhd_hit.md](perturbhd_hit.md) |
| **Systema** (Vinas 2025) | [`predict_systema`](../../tusoperturb/api.py) | `PRIOR_SYSTEMA` | `PanelMaster` from `systema_r1` harness | [systema.md](systema.md) |

The three regression heads (`cellsim`, `scperturb`, `perturbhd_reg`) share
the same numerical output — a per-perturbation × per-gene delta matrix
produced by `PRIOR_A_v2_193`. They differ only in how the output is scored:
CellSim and scPerturB feed it through a pred-AnnData shaping step and hand
it to their published scorer; PerturbHD-regression consumes the raw
`(pert, gene, effect)` long-format.

## What the four PerturbHD-family heads have in common

They all call `feature_builder.build_shared_features(dataset)`, which:

1. Loads a pre-staged feature bake via
   `perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`. The bake
   contains the GenePT embedding per perturbation (3072-D), the training
   target `Y_train`, the pert lists, donor id, control-mean baseline, and
   gene names.
2. Loads the 6 static annotation blocks (Reactome, GO BP, Hallmark, PROGENy,
   CollecTRI, STRING) from `data/embeddings/ref/` via the vendored
   `orth_features_v2` loader.
3. Loads the DepMap essentiality features for the matched cell line from
   `data/embeddings/depmap_essentiality/`.
4. Computes per-perturbation control-baseline stats (4-D).
5. Concatenates blocks into the 11,680-D matrix in the fixed layout from
   [`ARCHITECTURE.md §2`](../../ARCHITECTURE.md).

Steps 2-5 run against the vendored data in this repo and require no
external staging. **Step 1 is where you need the external data.** The pages
below all describe how to get / build the gpu_stage bake.

## What the Systema head does differently

Systema does not use the GenePT block. It builds its own 8,604-D block
stack via `systema_adapter.build_systema_features(perts, donor_id)`, which
reads the same vendored refs as the PerturbHD family but sums (sparse
blocks) or averages (dense blocks) across genes for combinatorial
perturbations like Norman `X+Y`. See [systema.md](systema.md) and
[`ARCHITECTURE.md §2`](../../ARCHITECTURE.md).

## Runtime budget (per (dataset, seed) with the bake on local disk)

| Head | Runtime | Bottleneck |
|---|---|---|
| CellSim / scPerturB regression | 2-5 min | Feature builder + ridge fit on 11,680-D × ~2k perts. |
| PerturbHD-regression | 2-5 min | Same as above. |
| PerturbHD-hit | 30-60 s | Fewer targets (AUCell phenotype count vs. gene count). |
| Systema | 30-180 s | Slower on `replogle_rpe1` / `replogle_k562_gwps` (larger panels). |
