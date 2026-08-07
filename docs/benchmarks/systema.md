# Systema adapter

`predict_systema` builds its feature matrix entirely from bundled reference
data. It does not use the `perturb_2026` stage loader, and it is the only
adapter that predicts a specified subset of rows rather than all of them.

## API

```python
from tusoperturb import predict_systema

Y_test = predict_systema(panel_master, seed=1)
```

The returned array has one row per entry of `panel_master.test_perts`, in that
order, and one column per gene, in the column order of
`panel_master.Y_train_post`.

## Required panel attributes

| Attribute | Requirement |
|---|---|
| `all_perts` | Sequence containing every training and test perturbation exactly once. |
| `train_perts` | Sequence whose order matches the rows of `Y_train_post`. |
| `test_perts` | Sequence defining prediction row order. |
| `Y_train_post` | Numeric array with shape `(len(train_perts), n_genes)`. |
| `donor_id` | Optional identifier used to select a DepMap table. Missing or unrecognized values fall back to K562. |

## Feature construction

`build_features(all_perts, donor_id)` produces the 8,700-column matrix: the
seven annotation blocks (8,604 columns) followed by the rank-64 and rank-32
co-essentiality blocks (96 columns).

Perturbation labels are split on `+` and the literal component `ctrl` is
ignored, so `TP53+MYC` combines two genes and `TP53+ctrl` uses only `TP53`.
Indicator blocks union their component genes; dense blocks average them. If none
of a label's component genes appear in a reference, that block stays zero.

DepMap lookup uses `resolve_depmap_line(donor_id)`:

| `donor_id` | DepMap table |
|---|---|
| `K562`, `replogle22k562` | K562 |
| `RPE1`, `replogle22rpe1` | RPE1 |
| `HepG2`, `nadig25hepg2` | HepG2 |
| `Jurkat`, `nadig25jurkat` | Jurkat |
| `A549`, `iPSC`, `melanoma`, `H1_hESC` | K562 |
| Missing or any other value | K562 |

## Model configuration

Systema uses the shared head, the same object as `cellsim`, `scperturb`, and
`perturbhd_reg`. The target is `log1p` post-expression, shaped as the residual
over the training perturbation mean.

The adapter passes `test_idx`, so prediction and every cross-row operation are
restricted to the test rows. This is the one path where a row's prediction
depends on which other rows are in the same batch, through the adaptive kNN
bandwidth's median normalisation. Submit the panel's full test set in a single
call, as the harness does; splitting it changes the numbers. The property is
described in [`leakage.md`](../leakage.md).

## Recorded results

| Metric | Direction | adamson | norman | k562_gwps | rpe1 | crispra | crispri | xu | v2 better |
|---|---|---|---|---|---|---|---|---|---|
| `centroid_accuracy` | higher | 0.7238 / 0.7548 | 0.7518 / 0.7768 | 0.7474 / 0.7775 | 0.7766 / 0.7853 | 0.5078 / 0.5311 | 0.4908 / 0.4805 | 0.6363 / 0.6747 | 1/7 |
| `composite_dev` | higher | 0.6912 / 0.7002 | 0.6681 / 0.6724 | 0.6135 / 0.5835 | 0.6699 / 0.6321 | 0.4914 / 0.4374 | 0.4841 / 0.4369 | 0.507 / 0.5393 | not scored |
| `corr_20de` | higher | 0.764 / 0.7953 | 0.7569 / 0.7147 | 0.593 / 0.5185 | 0.7128 / 0.651 | 0.6633 / 0.4046 | 0.675 / 0.4811 | 0.4992 / 0.4167 | 6/7 |
| `corr_20de_allpert` | higher | 0.4113 / 0.4587 | 0.5352 / 0.5227 | 0.4638 / 0.4253 | 0.4401 / 0.4046 | 0.02407 / -0.03276 | 0.001906 / -0.04601 | 0.3215 / 0.3054 | 6/7 |
| `corr_all` | higher | 0.7535 / 0.718 | 0.5865 / 0.5958 | 0.4147 / 0.3363 | 0.6016 / 0.5133 | 0.3835 / 0.2871 | 0.3354 / 0.2678 | 0.1088 / 0.2493 | 5/7 |
| `corr_all_allpert` | higher | 0.4792 / 0.481 | 0.4017 / 0.4495 | 0.3265 / 0.29 | 0.4688 / 0.4027 | 0.02414 / 0.05522 | -0.002742 / -0.004092 | 0.07569 / 0.2113 | 3/7 |
| `jaccard_top20` | higher | 0.2271 / 0.1875 | 0.24 / 0.2236 | 0.09273 / 0.0832 | 0.06642 / 0.06337 | 0.08265 / 0.04518 | 0.01771 / 0.01004 | 0.03973 / 0.05737 | 6/7 |
| `rmse_20de` | lower | 0.3148 / 0.3234 | 0.4285 / 0.4248 | 0.2419 / 0.2437 | 0.2979 / 0.3083 | 0.1537 / 0.1729 | 0.0943 / 0.1003 | 0.07161 / 0.07278 | 6/7 |
| `rmse_all` | lower | 0.05762 / 0.06661 | 0.05864 / 0.05761 | 0.05984 / 0.07194 | 0.08844 / 0.1073 | 0.03079 / 0.03765 | 0.02603 / 0.03048 | 0.02675 / 0.01486 | 5/7 |

Each cell is **TusoPerturb v2 / old TusoPerturb**, mean over 3 seeds. v2 is better in 38 of 56 scored cells.

`corr_*` metrics are correlations against measured post-expression, `rmse_*` are
errors, `jaccard_top20` and `centroid_accuracy` are set- and centroid-level
agreement. The `_allpert` variants pool across perturbations rather than
averaging per perturbation. `composite_dev` is a harness bookkeeping field, not
a model-discriminating score; it is recorded for completeness and excluded from
every count. Standard deviations are in
[`champion/params.json`](../../champion/params.json).

## Example

```python
import numpy as np
from tusoperturb import predict_systema

Y_test = predict_systema(panel_master, seed=1)

assert Y_test.shape == (len(panel_master.test_perts),
                        panel_master.Y_train_post.shape[1])
assert np.isfinite(Y_test).all()

np.save("systema-predictions.npy", Y_test)
```

`seed` is retained for harnesses that build a different `panel_master` per
split; the prediction function does not use it.

## Scoring

Score with the Systema evaluation code for your panel release. Column order is
not stored by TusoPerturb, so the caller is responsible for keeping the same
gene order in `Y_train_post`, the ground truth, and the scorer. Every entry of
`test_perts` and `train_perts` must occur in `all_perts`; a label that does not
raises a `KeyError`. A label whose component genes are absent from a reference
table does not raise — that block is simply zero for the row — so check the
coverage returned by `build_features` if predictions look degenerate.
