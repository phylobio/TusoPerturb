# Systema adapter

`predict_systema` predicts post-perturbation expression for the test rows of a
Systema-style panel object. This adapter builds features entirely from the
reference data bundled with TusoPerturb and does not use the external
`perturb_2026` stage loader.

## API

```python
from tusoperturb import predict_systema

Y_test = predict_systema(panel_master, seed=1)
```

The returned array has shape:

```text
(len(panel_master.test_perts), n_genes)
```

Rows follow `panel_master.test_perts`. Columns follow the order of
`panel_master.Y_train_post`; TusoPerturb does not attach gene names to the
returned NumPy array.

## Required panel attributes

| Attribute | Requirement |
|---|---|
| `all_perts` | Sequence containing every training and test perturbation exactly once. |
| `train_perts` | Sequence whose order matches the rows of `Y_train_post`. |
| `test_perts` | Sequence defining prediction row order. |
| `Y_train_post` | Numeric array with shape `(len(train_perts), n_genes)`. |
| `donor_id` | Optional identifier used to select a DepMap table. Missing or unrecognized values fall back to K562. |

Every label in `train_perts` and `test_perts` must occur in `all_perts`. The
caller is responsible for maintaining a consistent gene order between
`Y_train_post`, observed test expression, and the scorer.

The `seed` argument is retained for benchmark harnesses that construct a
different `panel_master` for each split. The prediction function itself does
not use the value.

## Feature construction

The Systema adapter builds an 8,604-dimensional representation from Reactome,
GO Biological Process, Hallmark, PROGENy, CollecTRI, STRING, and DepMap.

Combinatorial labels are split on `+`:

- sparse membership blocks are summed across component genes;
- PROGENy, STRING, and DepMap blocks are averaged; and
- the literal component `ctrl` is ignored.

Use exact gene symbols and lowercase `ctrl` in perturbation labels. A block remains zero when none of the component genes are present in its
reference. When only some components match, aggregation uses the matched genes.

DepMap selection follows the implementation:

| `donor_id` | DepMap table |
|---|---|
| `RPE1` | RPE1 |
| `K562` | K562 |
| `A549`, `iPSC`, `melanoma`, `H1_hESC` | K562 |
| Missing or any other value | K562 |

## Model configuration

After robust scaling, the Systema configuration combines:

- a ridge model with weight 0.20;
- cosine nearest-neighbor target transfer with weight 0.70; and
- a signed top-2% binary-response ridge model with weight 0.10.

Training targets are centered by their per-gene training mean. Predictions are
rescaled and the mean is restored before the array is returned.

## Example

```python
import numpy as np
from tusoperturb import predict_systema

Y_test = predict_systema(panel_master, seed=1)

assert Y_test.shape == (
    len(panel_master.test_perts),
    panel_master.Y_train_post.shape[1],
)
assert np.isfinite(Y_test).all()

np.save("systema-predictions.npy", Y_test)
```

## Scoring

Use the Systema scorer associated with the harness that created
`panel_master`. Preserve test-perturbation order and gene order when pairing
`Y_test` with observed expression.

The project-recorded panel values are listed in
[`report.md`](../../report.md#systema).
