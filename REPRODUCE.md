# Reproducing TusoPerturb results

This page tells you how to reproduce the champion numerical result from
`report.md`. The vendored reference data + `champion/params.json` pin the
static side; you need to supply the benchmark data yourself. Every benchmark
page in [`docs/benchmarks/`](docs/benchmarks/) restates the specific data
you'd need to bring.

## What "reproduce" means here

TusoPerturb has three levels of reproducibility, in decreasing strictness:

1. **Byte-identity of the feature stack.** With the vendored refs in
   `data/embeddings/` and matching Python + NumPy versions, the intermediate
   feature matrix `E_all` is bit-identical to the values produced during
   method development. This was verified after the 2026-07 embedding sync
   inside the original sandbox: `E_all` (2322 × 11680) and `Y_delta`
   (2322 × 8746) `max_abs_diff = 0.0` between the vendored-refs run and
   the method-development-refs run. Numerical determinism across different
   BLAS / OpenMP thread counts / NumPy versions is likely but not
   guaranteed; if you see small (<1e-6) drift on a different toolchain,
   that's expected and does not indicate a real difference.
2. **Head-config parity.** With `champion/params.json` unchanged, the three
   `HeadConfig` objects (`PRIOR_A_v2_193`, `PRIOR_R17_C`, `PRIOR_SYSTEMA`)
   produce byte-identical output given the same input feature matrix.
3. **Metric-level parity.** Given the same benchmark master `h5ad` and the
   same scoring pipeline (each benchmark's published scorer), you should
   recover the champion metric numbers to within numerical noise. Assert
   snippets for three headline numbers are at the bottom of this page.

Levels 1 and 2 are settled by this zip alone. Level 3 requires the external
data listed below.

## Environment

- Python 3.11+
- Runtime deps installed via `pip install -e .` from the repo root (see
  [pyproject.toml](pyproject.toml)).
- No GPU required for prediction. Training the champion used CPU-only ridge
  regression; there is no neural component to reproduce.

Optional environment variables:

| Var | When to set it | What it does |
|---|---|---|
| `TUSOPERTURB_REF_DIR` | You've moved `data/embeddings/ref/` elsewhere, or installed the wheel without the source tree. | Overrides where `orth_features_v2` reads Reactome / GO BP / Hallmark / CollecTRI / PROGENy / STRING refs from. |
| `TUSOPERTURB_DEPMAP_DIR` | You've moved `data/embeddings/depmap_essentiality/` elsewhere. | Overrides where `feature_builder` reads DepMap essentiality parquets from. |
| `TUSOPERTURB_MEAN_EFFECT_DIR` | You're running PerturbHD-hit outside the original method-development sandbox. | Overrides where `predict_hit` reads AUCell mean-effect parquets. |

These env vars are consulted before the vendored/default paths, so you can
force local data directories without editing code.

## External data you need to bring

Reproducing any of the four PerturbHD-family heads (CellSim, scPerturB,
PerturbHD-reg, PerturbHD-hit) requires all of:

- **Master `h5ad`** for each of `nadig25hepg2`, `nadig25jurkat`,
  `replogle22k562`, `replogle22rpe1`. Sizes ~14-20 GB each. Published with
  the original datasets — see individual benchmark pages for URLs.
- **`gpu_stage/<dataset>/` bake** containing `E_all_genept.npy`, `Y_train.npy`,
  `A_train.npy`, and `meta_master.h5ad`. Built from the master via the
  perturb-2026 staging pipeline; schema is documented in
  [`docs/benchmarks/cellsim.md`](docs/benchmarks/cellsim.md) so a determined
  reader can build the bake independently.
- **`perturb_2026` Python module** on `PYTHONPATH`. Both `feature_builder.py`
  and `api.py` do lazy `from perturb_2026.…` imports for the AnnData shaping
  step and the staged-feature loader.
- **AUCell phenotype parquets** (`<paper_key>-h.all-all.pq`) for the
  PerturbHD-hit head specifically. These are per-dataset from the perturb-hd
  supplementary data.

Reproducing the Systema head only requires:

- **A `PanelMaster` object** from the `systema_r1` benchmarking harness for
  each of the 7 Vinas panels. See
  [`docs/benchmarks/systema.md`](docs/benchmarks/systema.md).

## The three reproduction paths

### 1. PerturbHD family (CellSim + scPerturB + PerturbHD-reg + PerturbHD-hit)

For each of the 4 datasets and each of 3 seeds:

```python
import anndata as ad
from tusoperturb.api import predict_regression, predict_regression_gene, predict_hit

master = ad.read_h5ad("/path/to/masters/nadig25hepg2.h5ad")

# CellSim / scPerturB heads (both use PRIOR_A_v2_193; output = pred AnnData)
pred_cellsim = predict_regression(master, "nadig25hepg2", seed=1, head="cellsim")
pred_scperturb = predict_regression(master, "nadig25hepg2", seed=1, head="scperturb")

# PerturbHD-reg head (same config; output = pert/gene/effect long-format DataFrame)
pred_reg = predict_regression_gene(master, "nadig25hepg2", seed=1, head="perturbhd_reg")

# PerturbHD-hit head (uses PRIOR_R17_C; needs AUCell parquets)
pred_hit = predict_hit(master, "nadig25hepg2", seed=1)
```

Note: the regression heads' `seed` argument is a formality — `PRIOR_A_v2_193`
is deterministic and doesn't consume it. The hit head genuinely uses it to
select the `split-{seed}` train/test column.

Score each prediction against the corresponding published scorer. Runtime
per (dataset, seed): 2-5 minutes for the regression heads, 30-60 seconds
for the hit head, assuming the gpu_stage bake is on local disk.

### 2. Systema (7 Vinas panels)

For each of the 7 panels (`adamson`, `norman`, `tian_crispra`,
`tian_crispri`, `xu`, `replogle_rpe1`, `replogle_k562_gwps`) and each of
3 seeds:

```python
from tusoperturb import predict_systema

# panel_master is produced by the systema_r1 harness (Vinas 2025).
Y_test = predict_systema(panel_master, seed=1)  # (n_test_perts, n_genes) log1p post-expression
```

Score `Y_test` against the ground truth for `panel_master.test_perts` using
the systema_r1 `SEval` pipeline. Runtime per (panel, seed): 30-180 seconds
depending on panel size (`replogle_k562_gwps` and `replogle_rpe1` are the
slow ones).

### 3. Aggregate + verify

After running all 5 heads across all datasets × 3 seeds, aggregate to
long-format CSV and check the three headline numbers below. The
aggregation glue is benchmark-specific; the champion assertions are the
canonical parity check.

## Verifying reproduction

After reproducing the metrics, these three asserts confirm you have
byte-identical (or near-identical) champion parity:

```python
# Systema: adamson corr_20de
assert abs(systema_adamson_corr_20de - 0.795273) < 1e-4, "Systema champion parity broken"

# CellSim: nadig25hepg2 pearson_deltactrl_degs
assert abs(cellsim_hepg2_pearson_deltactrl_degs - 0.6423) < 5e-4, "CellSim champion parity broken"

# scPerturB: replogle22k562 pcc_delta
assert abs(scperturb_k562_pcc_delta - 0.682) < 5e-3, "scPerturB champion parity broken"

print("OK — champion parity verified.")
```

Absolute numbers from the champion run:

| Benchmark  | Dataset         | Metric                    | Value               |
|------------|-----------------|---------------------------|---------------------|
| Systema    | adamson         | `corr_20de`               | 0.7952732835851964  |
| CellSim    | nadig25hepg2    | `pearson_deltactrl_degs`  | 0.6423267403536010  |
| scPerturB  | replogle22k562  | `pcc_delta`               | 0.6817649548329194  |

## Sanity check without external data

Even without the benchmark data, you can verify the vendored refs load
correctly and produce the right feature-block widths:

```python
from tusoperturb._deps.orth_features_v2 import load_orth_features_v2

# Toy pert list — just needs valid HGNC symbols
perts = ["TP53", "MYC", "KRAS", "BRCA1", "EGFR"]
feats, _ = load_orth_features_v2(perts, features=('reactome', 'go_bp', 'hallmark', 'progeny', 'collectri', 'string'))

# Expected widths from ARCHITECTURE.md §2 feature-block table:
assert feats['reactome'].shape == (5, 1816)
assert feats['go_bp'].shape    == (5, 5406)
assert feats['hallmark'].shape == (5, 50)
assert feats['progeny'].shape  == (5, 14)
assert feats['collectri'].shape == (5, 1185)
assert feats['string'].shape   == (5, 128)
print("OK — vendored refs load and feature widths match architecture doc.")
```

If this passes, you have a working feature stack. What you can't do without
the external data is exercise the full `build_shared_features(dataset)`
pipeline (which pulls the GenePT block from the `gpu_stage` bake) or the
`predict_regression` / `predict_hit` heads that consume it.
