# Architecture

TusoPerturb represents each perturbation with biological reference features and
uses a configurable predictor to map those features to an expression or
phenotype target. The public benchmark adapters share the same low-level
prediction function but use different feature subsets, targets, and model
parameters.

The implementation is split across four modules:

- [`tusoperturb/feature_builder.py`](tusoperturb/feature_builder.py) builds the
  shared feature matrix used by the CellSimBench, scPerturBench, and PerturbHD
  adapters.
- [`tusoperturb/systema_adapter.py`](tusoperturb/systema_adapter.py) builds the
  Systema-specific feature matrix, including support for combinatorial
  perturbations.
- [`tusoperturb/heads.py`](tusoperturb/heads.py) defines the available model
  configurations.
- [`tusoperturb/predictor.py`](tusoperturb/predictor.py) implements feature
  processing, model fitting, and prediction.

## Data flow

```text
perturbation labels
        │
        ▼
feature builder ──► feature matrix E
                         │
training rows + targets  │
             └───────────┤
                         ▼
              head_predict(E, ...)
                         │
                         ▼
            benchmark-specific output
```

The benchmark adapters in [`tusoperturb/api.py`](tusoperturb/api.py) are
responsible for loading targets, choosing a configuration from `HEAD_CONFIGS`,
and formatting predictions for an external scorer.

## Feature representations

### Shared 11,680-dimensional representation

`build_shared_features(dataset)` creates one row per perturbation. Column ranges
below use Python's half-open notation, `[start:end)`.

| Block | Columns | Width | Contents |
|---|---:|---:|---|
| `genept` | `[0:3072)` | 3,072 | GenePT embedding supplied by the staged dataset loader. |
| `reactome` | `[3072:4888)` | 1,816 | Reactome pathway membership indicators. |
| `go_bp` | `[4888:10294)` | 5,406 | Gene Ontology Biological Process indicators. |
| `hallmark` | `[10294:10344)` | 50 | MSigDB Hallmark gene-set indicators. |
| `progeny` | `[10344:10358)` | 14 | PROGENy pathway coefficients. |
| `collectri` | `[10358:11543)` | 1,185 | CollecTRI target-to-regulator indicators. |
| `string` | `[11543:11671)` | 128 | STRING network embedding. |
| `depmap` | `[11671:11676)` | 5 | Cell-line-specific DepMap summary features. |
| `baseline` | `[11676:11680)` | 4 | Control-expression value, `log1p` value, z-score, and empirical rank fraction for the perturbed gene. |
| **Total** |  | **11,680** |  |

The GenePT block and training targets come from the external staged dataset.
The remaining static annotation files are read from `data/embeddings/` unless
an environment-variable override is configured.

The shipped configurations use two views of this matrix:

- `full_11680` keeps every block and is used by the three expression-regression
  adapters.
- `no_genept_8608` removes `[0:3072)` and is used by PerturbHD hit prediction.

The predictor also implements `no_genept_no_depmap_8603`, but none of the
public configurations currently select it.

### Systema 8,604-dimensional representation

`build_systema_features(perts, donor_id)` builds a separate matrix directly
from the bundled annotations and DepMap tables. It does not use GenePT or the
baseline-expression block.

| Block | Width | Multi-gene aggregation |
|---|---:|---|
| `reactome` | 1,816 | Sum |
| `go_bp` | 5,406 | Sum |
| `hallmark` | 50 | Sum |
| `progeny` | 14 | Mean |
| `collectri` | 1,185 | Sum |
| `string` | 128 | Mean |
| `depmap` | 5 | Mean |
| **Total** | **8,604** |  |

Perturbation labels are split on `+`, and the literal component `ctrl` is
ignored. For example, `TP53+MYC` combines two genes, while `TP53+ctrl` uses only
`TP53`. If none of a perturbation's component genes are present in a reference,
that block remains zero. When only some components are present, aggregation uses
the matched components.

DepMap lookup uses the following mapping from `donor_id`:

| `donor_id` | DepMap table |
|---|---|
| `RPE1` | `RPE1.pq` |
| `K562` | `K562.pq` |
| `A549`, `iPSC`, `melanoma`, `H1_hESC` | `K562.pq` |
| Missing or unrecognized value | `K562.pq` |

## Prediction pipeline

The low-level interface is:

```python
Y_pred = head_predict(
    E_full,
    train_idx,
    Y_train,
    cfg,
    test_idx=None,
)
```

`head_predict` performs the following operations.

### 1. Select features

The feature matrix is sliced according to `cfg.feature_subset`. The Systema
adapter passes an already constructed 8,604-dimensional matrix, so no further
slicing is required for `systema_native_8604`.

### 2. Scale from training rows

The scaler is fitted on `E_full[train_idx]` and then applied to both training
and prediction rows.

- `standard` uses standard mean and variance scaling.
- `robust_25_75` uses `RobustScaler` with the 25th and 75th percentiles.
- `none` leaves the features unchanged apart from conversion to `float32`.

This keeps test-row feature values out of scaler fitting.

### 3. Apply block weights

Configured block multipliers are applied after scaling. Only the shared
expression-regression configuration currently uses them:

```text
go_bp × 0.67
string × 1.5
depmap × 10.0
```

### 4. Transform the target

Two target modes are implemented:

- `raw_delta` passes `Y_train` through unchanged. Despite the historical name,
  this mode is also used for the PerturbHD phenotype target.
- `residual_pert_mean` subtracts the training-row mean for every target column.
  The mean is restored after prediction.

### 5. Fit enabled prediction arms

A configuration can enable up to three arms.

**Ridge arm.** Fits either a fixed-alpha `Ridge` model or `RidgeCV`. When
`y_standardize=True`, target columns are standardized for fitting and converted
back to their original scale after prediction.

**Nearest-neighbor arm.** Normalizes feature rows for cosine distance, finds up
to 13 nearest training perturbations, and transfers their target values using
softmax weights based on distance. The shipped blended configurations use a
temperature of 11.

**Binary-response arm.** Converts each target column to a top-2% response
indicator and fits `RidgeCV` to the resulting matrix. The signed variant ranks
by absolute target magnitude and preserves the sign; the unsigned variant marks
only the largest positive values.

All three arms are fitted against the target produced in step 4.

### 6. Combine arm predictions

A ridge-only configuration returns the ridge output directly. Otherwise, each
enabled arm is z-scored independently across the prediction rows and combined
using the configured weights.

When `test_idx` is supplied, prediction and arm-level z-scoring are restricted
to those rows. The Systema adapter uses this mode. Other adapters predict every
row and let their output wrapper select the required records.

### 7. Restore a residual target

For `residual_pert_mean`, the blended output is adjusted to the training
target's per-column mean and standard deviation before the mean removed in step
4 is added back.

## Shipped configurations

The public keys below are the stable interface. The constant names in
`heads.py` are implementation details retained for compatibility.

| Public key | Feature view | Scaling | Predictor | Target mode |
|---|---|---|---|---|
| `cellsim` | `full_11680` | Standard | Fixed ridge, alpha 110 | Unchanged target |
| `scperturb` | `full_11680` | Standard | Fixed ridge, alpha 110 | Unchanged target |
| `perturbhd_reg` | `full_11680` | Standard | Fixed ridge, alpha 110 | Unchanged target |
| `perturbhd_hit` | `no_genept_8608` | Robust | Ridge/kNN/binary blend, weights `0.10/0.65/0.25` | Unchanged target |
| `systema` | `systema_native_8604` | Robust | Ridge/kNN/binary blend, weights `0.20/0.70/0.10` | Residual over training-column mean |

The first three keys point to the same `HeadConfig` object and therefore
produce the same numerical prediction matrix when given the same inputs. Their
adapters differ only in output format and downstream scorer.

## Public adapters

### `predict_regression`

Builds the shared feature matrix, predicts expression deltas for every staged
perturbation, adds the staged control mean, and delegates `AnnData` construction
to `perturb_2026.loop.helpers.build_pred_adata_from_matrix`.

The `seed` argument is accepted for API consistency but is not used by the
current regression implementation.

### `predict_regression_gene`

Runs the same regression model and returns a long-form `DataFrame` with
`pert`, `gene`, and `effect` columns. It does not use the external AnnData
output helper, although feature construction still requires the staged loader.
The current implementation retains `master` and `seed` in the signature for
compatibility but does not otherwise use them.

### `predict_hit`

Loads staged perturbation features and a per-dataset phenotype table, trains on
rows labeled `train` in `split-{seed}`, and returns scores for rows labeled
`test` or `val`. The result contains `pert`, `pheno`, and `hit_score` columns.

### `predict_systema`

Builds Systema features for `panel_master.all_perts`, trains on
`panel_master.train_perts`, and predicts only `panel_master.test_perts`. Output
rows follow `test_perts`; columns follow the target order in
`panel_master.Y_train_post`.

## Custom configurations

Advanced users can pass a custom `HeadConfig` to any adapter or call
`head_predict` directly:

```python
from dataclasses import replace
from tusoperturb import HEAD_CONFIGS, head_predict

cfg = replace(
    HEAD_CONFIGS["cellsim"],
    ridge_fixed_alpha=50.0,
)

Y_pred = head_predict(E, train_idx, Y_train, cfg)
```

A custom configuration changes the method and will not be directly comparable
to the recorded benchmark values in [`report.md`](report.md).

## Numerical behavior

The shipped prediction paths do not perform stochastic sampling. Results should
be deterministic for fixed inputs and dependency versions, although small
floating-point differences can occur across operating systems, BLAS libraries,
or scikit-learn versions.
