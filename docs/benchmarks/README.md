# Benchmark guides

TusoPerturb provides adapters for five perturbation-prediction benchmark
interfaces. The pages in this directory describe the inputs expected by each
adapter, how to run it, the format of its predictions, and the values it
recorded on the sealed test set.

Raw benchmark datasets, staged dataset artifacts, and third-party scoring code
are not included in this repository.

The wrapper implementations live in `tusoperturb.api`. `predict_regression`,
`predict_regression_gene`, `predict_hit`, and `predict_systema` are also
re-exported from the package root.

## Guide index

| Benchmark | API | Prediction format | Guide |
|---|---|---|---|
| CellSimBench | `predict_regression(..., head="cellsim")` | `AnnData` containing predicted post-perturbation expression | [cellsim.md](cellsim.md) |
| scPerturBench | `predict_regression(..., head="scperturb")` | `AnnData` containing predicted post-perturbation expression | [scperturb.md](scperturb.md) |
| PerturbHD regression | `predict_regression_gene(...)` | Long-form `DataFrame` with `pert`, `gene`, and `effect` columns | [perturbhd_reg.md](perturbhd_reg.md) |
| PerturbHD hit prediction | `predict_hit(...)` | Long-form `DataFrame` with `pert`, `pheno`, and `hit_score` columns | [perturbhd_hit.md](perturbhd_hit.md) |
| Systema | `predict_systema(...)` | NumPy array for the test perturbations | [systema.md](systema.md) |

## Recorded results

Each benchmark was scored once against a sealed held-out test set, and every
cell is compared with the old TusoPerturb.

| Benchmark | Datasets | Metrics | Scored cells | v2 better | old better |
|---|---|---|---|---|---|
| `cellsim` | 4 | 16 | 64 | 54 | 10 |
| `scperturb` | 4 | 3 | 12 | 7 | 5 |
| `perturbhd_reg` | 4 | 6 | 24 | 24 | 0 |
| `perturbhd_hit` | 4 | 2 | 8 | 7 | 1 |
| `systema` | 7 | 8 (+1 not scored) | 56 | 38 | 18 |
| **total** |  |  | **164** | **130** | **34** |

The comparison baseline is the frozen TusoPerturb head table as it stood when
the sealed run was executed. For `perturbhd_hit` that is the v1.0.0
configuration unchanged; for the other four keys it is the stronger
post-transfer table, not the weaker v1.0.0 table. The `comparison_baseline`
block of [`champion/params.json`](../../champion/params.json) states exactly
what it is.

The seven Systema `composite_dev` cells are a harness bookkeeping field rather
than a model-discriminating score. They are recorded with `is_placeholder: true`
and excluded from every count above.

Split integrity for this run is documented in [`leakage.md`](../leakage.md).

## Shared expression and hit-prediction inputs

The CellSimBench, scPerturBench, and PerturbHD adapters build features through
`tusoperturb.feature_builder.build_shared_features()`. They support these
dataset identifiers:

| Dataset identifier | Cell line | DepMap table |
|---|---|---|
| `nadig25hepg2` | HepG2 | `HepG2.pq` |
| `nadig25jurkat` | Jurkat | `Jurkat.pq` |
| `replogle22k562` | K562 | `K562.pq` |
| `replogle22rpe1` | RPE1 | `RPE1.pq` |

The DepMap column of this table is a behavioural change from v1, which had no
HepG2 or Jurkat entry and used K562 values for both. See
[`ARCHITECTURE.md`](../../ARCHITECTURE.md#depmap-cell-line-selection).

These workflows require a staged dataset loadable through
`perturb_2026.loop.gpu_stage_loader.load_stage()`. The staged data supplies the
training targets, perturbation split, baseline expression, and gene names.
TusoPerturb combines those values with the reference annotations, DepMap
summaries, and co-essentiality basis included under `data/embeddings/`. Unlike
v1, no embedding block comes from the stage.

Additional requirements depend on the adapter:

- `predict_regression()` also uses the external `perturb_2026` helpers to shape
  its output as an `AnnData` object.
- `predict_regression_gene()` returns a plain `DataFrame`, but still requires
  the staged dataset loader for feature construction.
- `predict_hit()` additionally requires the PerturbHD AUCell mean-effect
  Parquet files. Set `TUSOPERTURB_MEAN_EFFECT_DIR` to the directory holding
  them.

## Systema inputs

The Systema adapter follows a separate path. It accepts a `PanelMaster`-like
object containing the perturbation lists and training expression matrix, then
builds the feature matrix from the bundled reference data alone. It does not use
the `perturb_2026` staged dataset loader.

Single-gene and combinatorial perturbations are both supported. For
combinatorial labels, the adapter aggregates the component-gene features before
prediction. See [systema.md](systema.md) for the required object attributes and
output ordering.

## Model configurations

All five keys read the same 8,700-dimensional feature matrix. Four of them
resolve to one shared head; PerturbHD hit prediction uses a second head that
differs only in head parameters. Exact values are in
[`tusoperturb/heads.py`](../../tusoperturb/heads.py) and
[`champion/params.json`](../../champion/params.json), and the pipeline is
described in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

For the complete reproduction workflow and external-data checklist, see
[`REPRODUCE.md`](../../REPRODUCE.md).
