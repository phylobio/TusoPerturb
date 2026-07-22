# Systema (Vinas 2025, 7-panel)

## What it measures

Systema evaluates predicted log1p post-expression against ground-truth
observed post-expression across 7 panels covering CRISPRi (Adamson, Xu,
Replogle-RPE1, Replogle-K562-GWPS), CRISPRa (Tian), and combinatorial
CRISPRi (Norman). Primary metric: `corr_20de` (Pearson correlation on the
top 20 DEGs per pert), aggregated across perturbations.

**Head**: `predict_systema(panel_master, seed)`
**HeadConfig**: `PRIOR_SYSTEMA` = `res_wr20_wk70_wb10` — a 3-arm blend
(weights `WR=0.20 / WK=0.70 / WB=0.10`) with residual-over-pert-mean
target framing.
**Champion metric**: `corr_20de = 0.7953` on Adamson.

## Data you need to obtain

Systema does **not** need the `perturb_2026` gpu_stage bake. It runs off
a `PanelMaster` object produced by the `systema_r1` benchmarking harness
(Vinas 2025).

### PanelMaster

`predict_systema` expects a `panel_master` with these attributes:

| Attribute | Type | What it is |
|---|---|---|
| `.all_perts` | list[str] | All perturbations in the panel (train ∪ test). For combinatorial panels like Norman these look like `'AHR+FEV'`. |
| `.train_perts` | list[str] | Subset of `.all_perts` used for training. |
| `.test_perts` | list[str] | Subset of `.all_perts` used for evaluation. |
| `.Y_train_post` | (n_train, n_genes) np.ndarray float32 | log1p post-expression for the train perts. |
| `.donor_id` | str | 'K562', 'RPE1', 'HepG2', or 'Jurkat' (falls back to 'K562' if missing). |

The `systema_r1` harness produces one `PanelMaster` per (panel, seed). See
the Vinas 2025 supplementary methods + the harness code for how it's built
from raw dataset h5ads.

### The 7 panels

| Panel | Source | Note |
|---|---|---|
| `adamson` | Adamson et al. 2016 | Small (~90 perts). |
| `norman` | Norman et al. 2019 | Combinatorial: some perts are `'GENE1+GENE2'`. |
| `tian_crispra` | Tian et al. 2021 CRISPRa arm | |
| `tian_crispri` | Tian et al. 2021 CRISPRi arm | |
| `xu` | Xu et al. 2024 | |
| `replogle_rpe1` | Replogle et al. 2022 | Larger (~1.5k perts). |
| `replogle_k562_gwps` | Replogle et al. 2022 GWPS arm | Largest (~2.5k perts). |

### No gpu_stage bake

Systema's feature stack is built in-process by
`systema_adapter.build_systema_features(perts, donor_id)`, which reads the
vendored refs in `data/embeddings/ref/` and the DepMap parquets in
`data/embeddings/depmap_essentiality/`. Everything it needs is in this repo.

### Combinatorial perts

`build_systema_features` handles multi-gene perts (e.g. Norman `X+Y`) by
combining per-gene features using a per-block rule from
[`ARCHITECTURE.md §2`](../../ARCHITECTURE.md):

- **Sum** for sparse indicator blocks (Reactome, GO BP, Hallmark, CollecTRI).
- **Mean** for dense embedding blocks (PROGENy, STRING, DepMap).

This is what `res_wr20_wk70_wb10` used during the champion run and is the
rule that lets a single model handle both single-gene and combinatorial
panels without any panel-specific code.

## Running the head

```python
from tusoperturb import predict_systema

# panel_master comes from the systema_r1 harness for the given (panel, seed).
# See Vinas 2025 for the harness.
Y_test = predict_systema(panel_master, seed=1)

# Y_test shape: (len(panel_master.test_perts), n_genes) log1p post-expression.
# Row order matches panel_master.test_perts; column order matches the harness's
# gene ordering (typically panel_master.gene_names if present).
```

Run over all 7 panels × 3 seeds (21 predictions total). Runtime per
prediction: 30 s to 3 min depending on panel size. The blend evaluates
Ridge + kNN + binary arms only on the test rows (`test_idx=te` inside
`head_predict`), matching the `res_wr20_wk70_wb10` semantics from the
champion.

## Scoring your prediction

Feed `Y_test` into the `systema_r1` `SEval` scorer alongside the observed
post-expression for `panel_master.test_perts`. The scorer computes
`corr_20de` per perturbation and aggregates across the panel.

`systema_r1` code + scorer: see the systema_r1 GitHub repo referenced in
Vinas 2025.

## Champion parity check

```python
assert abs(adamson_corr_20de - 0.795273) < 1e-4
```

Exact value from the champion run: `0.7952732835851964`.

## Where TusoPerturb wins (and doesn't)

Systema is the one benchmark bucket where TusoPerturb does not sweep every
panel. The champion #1 count is 3/7 primary panels (`adamson`, `norman`,
`replogle_rpe1` under the `corr_20de` metric — see `report.md` for the
per-metric breakdown). The `tian_crispra`, `tian_crispri`, and `xu`
panels are weaker; this is an inherited limitation from the underlying
`res_wr20_wk70_wb10` config that TusoPerturb didn't attempt to fix during
harmonization (the goal was single-package unification, not per-panel
tuning). See `report.md §4` for interpretation.
