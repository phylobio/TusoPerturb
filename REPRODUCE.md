# Reproducing benchmark evaluations

This repository contains the TusoPerturb model code, fixed configurations, and
static biological reference features. It does not contain the benchmark
`AnnData` files, staged GenePT features, phenotype tables, or third-party
scoring harnesses required to recreate the recorded benchmark values.

The reproduction workflow therefore has two parts:

1. verify that the package and bundled reference data load correctly; and
2. run the adapters inside the corresponding benchmark environment, using the
   same datasets, splits, preprocessing, and scorer versions.

Recorded three-seed means are stored in
[`champion/params.json`](champion/params.json) and summarized in
[`report.md`](report.md). They are reference metadata rather than executable
test fixtures.

## Install the package

TusoPerturb requires Python 3.11 or newer. An editable install is recommended
because the bundled reference data lives at the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Prediction uses NumPy and scikit-learn and does not require a GPU.

## Configure data paths

Set path overrides before importing `tusoperturb`:

| Variable | Purpose |
|---|---|
| `TUSOPERTURB_REF_DIR` | Directory containing the Reactome, GO, Hallmark, PROGENy, CollecTRI, and STRING reference files. |
| `TUSOPERTURB_DEPMAP_DIR` | Directory containing `HepG2.pq`, `Jurkat.pq`, `K562.pq`, and `RPE1.pq`. |
| `TUSOPERTURB_MEAN_EFFECT_DIR` | Directory containing the PerturbHD phenotype Parquet files used by `predict_hit`. |

The first two variables are optional when running from a source checkout with
`data/embeddings/` intact. The PerturbHD hit workflow normally requires the
third because those phenotype tables are not bundled.

Example:

```bash
export TUSOPERTURB_MEAN_EFFECT_DIR=/path/to/mean_effects_aucell
```

## Verify the local installation

The following check uses only bundled annotation files and does not require a
benchmark dataset:

```python
from tusoperturb._deps.orth_features_v2 import load_orth_features_v2

perts = ["TP53", "MYC", "KRAS"]
features, _ = load_orth_features_v2(
    perts,
    features=(
        "reactome",
        "go_bp",
        "hallmark",
        "progeny",
        "collectri",
        "string",
    ),
)

expected_widths = {
    "reactome": 1816,
    "go_bp": 5406,
    "hallmark": 50,
    "progeny": 14,
    "collectri": 1185,
    "string": 128,
}

for name, width in expected_widths.items():
    assert features[name].shape == (len(perts), width)
```

You can also exercise the predictor with synthetic data:

```python
import numpy as np
from tusoperturb import HEAD_CONFIGS, head_predict

rng = np.random.default_rng(0)
E = rng.normal(size=(8, 11680)).astype(np.float32)
Y_train = rng.normal(size=(5, 4)).astype(np.float32)
train_idx = np.arange(5)

Y_pred = head_predict(E, train_idx, Y_train, HEAD_CONFIGS["cellsim"])
assert Y_pred.shape == (8, 4)
```

These checks validate package loading and array contracts. They do not validate
benchmark preprocessing or scoring.

## External requirements for shared-feature workflows

CellSimBench, scPerturBench, PerturbHD regression, and PerturbHD hit prediction
all call `build_shared_features(dataset)`. The supported dataset identifiers
are:

```text
nadig25hepg2
nadig25jurkat
replogle22k562
replogle22rpe1
```

### `perturb_2026` stage loader

The external module
`perturb_2026.loop.gpu_stage_loader` must provide
`load_stage(dataset)`. The current TusoPerturb implementation expects the
returned mapping to contain these keys:

| Key | Expected contents |
|---|---|
| `E_train_genept` | Training-row GenePT matrix. The current builder reads this key for compatibility but does not otherwise use the value. |
| `E_all_genept` | `float32` array with shape `(n_all_perts, 3072)`. |
| `Y_train` | Training target array with shape `(n_train_perts, n_genes)`. |
| `all_perts` | Perturbation labels in the same order as `E_all_genept`. |
| `train_perts` | Training labels in the same row order as `Y_train`. |
| `donor_id` | Cell-line identifier used for DepMap lookup and output construction. |
| `mean_baseline` | Control mean-expression vector with length `n_genes`. |
| `gene_names` | Gene labels in the same order as `Y_train` columns. |

`train_perts` must be a subset of `all_perts`, and perturbation labels must
match the labels used by the reference and phenotype tables.

The on-disk layout behind `load_stage` is owned by the external
`perturb_2026` package and is not defined by this repository.

### AnnData output helper

`predict_regression` also imports:

```text
perturb_2026.loop.helpers.build_pred_adata_from_matrix
perturb_2026.loop.paths.FOLD
```

The supplied `master` object must satisfy that helper and the downstream
benchmark scorer. TusoPerturb does not inspect the `master` object directly
before passing it to the helper.

`predict_regression_gene` does not use the AnnData output helper, but it still
requires `load_stage` for feature construction.

### PerturbHD phenotype tables

`predict_hit` additionally expects one file per dataset:

```text
<TUSOPERTURB_MEAN_EFFECT_DIR>/<paper_key>-h.all-all.pq
```

The required `paper_key` values are documented in
[`docs/benchmarks/perturbhd_hit.md`](docs/benchmarks/perturbhd_hit.md).

## Generate shared-feature predictions

```python
import anndata as ad
from tusoperturb.api import (
    predict_hit,
    predict_regression,
    predict_regression_gene,
)

master = ad.read_h5ad("/path/to/nadig25hepg2.h5ad")
dataset = "nadig25hepg2"

pred_cellsim = predict_regression(
    master,
    dataset,
    seed=1,
    head="cellsim",
)
pred_cellsim.write_h5ad("cellsim-nadig25hepg2.h5ad")

pred_scperturb = predict_regression(
    master,
    dataset,
    seed=1,
    head="scperturb",
)
pred_scperturb.write_h5ad("scperturb-nadig25hepg2.h5ad")

pred_regression = predict_regression_gene(
    master,
    dataset,
    seed=1,
    head="perturbhd_reg",
)
pred_regression.to_parquet("perturbhd-reg-nadig25hepg2.parquet", index=False)

pred_hit = predict_hit(master, dataset, seed=1)
pred_hit.to_parquet("perturbhd-hit-nadig25hepg2-seed1.parquet", index=False)
```

The three expression-regression keys use the same model configuration. With
identical staged inputs, their underlying prediction matrix is the same; only
the output format or downstream scorer differs.

The current regression implementation does not use `seed`. PerturbHD hit
prediction does use it to select `split-1`, `split-2`, or `split-3` from the
phenotype table.

## External requirements for Systema

Systema does not use `perturb_2026`. It expects a `panel_master` object with:

- `all_perts`: all perturbation labels;
- `train_perts`: labels corresponding to the rows of `Y_train_post`;
- `test_perts`: labels to predict;
- `Y_train_post`: training expression array with shape
  `(len(train_perts), n_genes)`; and
- optionally, `donor_id` for DepMap selection.

Every training and test label must occur in `all_perts`. Column order is not
stored separately by TusoPerturb, so the caller is responsible for preserving
the same gene order in `Y_train_post`, ground truth, and scoring code.

```python
import numpy as np
from tusoperturb import predict_systema

Y_test = predict_systema(panel_master, seed=1)
assert Y_test.shape[0] == len(panel_master.test_perts)

np.save("systema-predictions.npy", Y_test)
```

The `seed` argument is retained for compatibility with harnesses that build a
different `panel_master` per split. The prediction function itself does not use
it.

## Score predictions

TusoPerturb does not bundle benchmark scorers. Use the scorer version associated
with each benchmark and preserve the expected perturbation and gene ordering.
See the pages under [`docs/benchmarks/`](docs/benchmarks/) for adapter-specific
output contracts.

For a valid comparison with the recorded results, keep all of the following
fixed:

- dataset release and preprocessing;
- train, validation, and test assignments;
- staged GenePT features;
- target construction;
- scorer implementation and metric aggregation;
- TusoPerturb configuration; and
- dependency versions where exact numerical agreement matters.

## Compare with the recorded values

The repository's reference values can be read directly rather than copied into
custom scripts:

```python
import json
from pathlib import Path

params = json.loads(Path("champion/params.json").read_text())
recorded = params["validated_results_3seed_mean"]

for benchmark, datasets in recorded.items():
    for dataset, result in datasets.items():
        print(
            benchmark,
            dataset,
            result["metric"],
            result["value"],
            f'rank={result["rank"]}/{result["total_methods"]}',
        )
```

Treat `rank` and `total_methods` as historical metadata from the recorded
benchmark table. This repository does not include the baseline predictions or
raw scorer outputs needed to independently reconstruct those ranks.

Small floating-point differences may appear across NumPy, scikit-learn, BLAS,
and operating-system combinations. Larger differences usually indicate a
change in input ordering, target preprocessing, split assignment, or scorer
version.

## Troubleshooting

### `ImportError: No module named perturb_2026`

Install or expose the external `perturb_2026` package before using the shared
feature builder. Systema-only workflows do not require it.

### Missing reference files

Use an editable installation from the source checkout, or set
`TUSOPERTURB_REF_DIR` and `TUSOPERTURB_DEPMAP_DIR` before starting Python.

### Missing PerturbHD Parquet file

Check `TUSOPERTURB_MEAN_EFFECT_DIR` and confirm the filename uses the expected
`paper_key`.

### Shape or ordering mismatch

Confirm that `train_perts` matches `Y_train` row order, `all_perts` matches
`E_all_genept` row order, and `gene_names` matches every expression-target
column.
