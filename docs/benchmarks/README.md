# Benchmark guides

TusoPerturb provides adapters for five perturbation-prediction benchmark
interfaces. The pages in this directory describe the inputs expected by each
adapter, how to run it, and the format of its predictions.

Raw benchmark datasets, staged dataset artifacts, and third-party scoring code
are not included in this repository.

The wrapper implementations live in `tusoperturb.api`. `predict_regression`,
`predict_hit`, and `predict_systema` are also re-exported from the package root.

## Guide index

| Benchmark | API | Prediction format | Guide |
|---|---|---|---|
| CellSimBench | `predict_regression(..., head="cellsim")` | `AnnData` containing predicted post-perturbation expression | [cellsim.md](cellsim.md) |
| scPerturBench | `predict_regression(..., head="scperturb")` | `AnnData` containing predicted post-perturbation expression | [scperturb.md](scperturb.md) |
| PerturbHD regression | `predict_regression_gene(...)` | Long-form `DataFrame` with `pert`, `gene`, and `effect` columns | [perturbhd_reg.md](perturbhd_reg.md) |
| PerturbHD hit prediction | `predict_hit(...)` | Long-form `DataFrame` with `pert`, `pheno`, and `hit_score` columns | [perturbhd_hit.md](perturbhd_hit.md) |
| Systema | `predict_systema(...)` | NumPy array for the test perturbations | [systema.md](systema.md) |

## Shared expression and hit-prediction inputs

The CellSimBench, scPerturBench, and PerturbHD adapters build features through
`tusoperturb.feature_builder.build_shared_features()`. They support these
dataset identifiers:

| Dataset identifier | Cell line |
|---|---|
| `nadig25hepg2` | HepG2 |
| `nadig25jurkat` | Jurkat |
| `replogle22k562` | K562 |
| `replogle22rpe1` | RPE1 |

These workflows require a staged dataset loadable through
`perturb_2026.loop.gpu_stage_loader.load_stage()`. The staged data supplies the
GenePT features, training targets, perturbation split, baseline expression, and
gene names. TusoPerturb combines those values with the reference annotations
and DepMap summaries included under `data/embeddings/`.

Additional requirements depend on the adapter:

- `predict_regression()` also uses the external `perturb_2026` helpers to shape
  its output as an `AnnData` object.
- `predict_regression_gene()` returns a plain `DataFrame`, but still requires
  the staged dataset loader for feature construction.
- `predict_hit()` additionally requires the PerturbHD AUCell mean-effect
  Parquet files. Set `TUSOPERTURB_MEAN_EFFECT_DIR` when they are not stored at
  the default location expected by the code.

Each benchmark page documents the required staged-data schema and scoring
workflow in more detail.

## Systema inputs

The Systema adapter follows a separate path. It accepts a `PanelMaster`-like
object containing the perturbation lists and training expression matrix, then
builds an 8,604-dimensional feature matrix from the bundled reference data. It
does not use the GenePT block or the `perturb_2026` staged dataset loader.

Single-gene and combinatorial perturbations are both supported. For
combinatorial labels, the adapter aggregates the component-gene features
before prediction. See [systema.md](systema.md) for the required object
attributes and output ordering.

## Model configurations

The three expression-regression adapters use the same fixed ridge
configuration. PerturbHD hit prediction and Systema use task-specific blended
predictors. Exact feature subsets and model parameters are defined in
[`tusoperturb/heads.py`](../../tusoperturb/heads.py) and described in
[`ARCHITECTURE.md`](../../ARCHITECTURE.md).

For the complete reproduction workflow and external-data checklist, see
[`REPRODUCE.md`](../../REPRODUCE.md).
