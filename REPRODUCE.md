# Reproducing benchmark evaluations

This repository contains the TusoPerturb v2 model code, the frozen
configuration, and the static biological reference features. It does not contain
the benchmark `AnnData` files, staged dataset caches, phenotype tables, or
third-party scoring harnesses required to recreate the recorded values.

The reproduction workflow therefore has two parts:

1. verify that the package and bundled reference data load correctly; and
2. run the adapters inside the corresponding benchmark environment, using the
   same datasets, splits, preprocessing, and scorer versions.

Recorded values are stored in [`champion/params.json`](champion/params.json) and
tabulated per benchmark under [`docs/benchmarks/`](docs/benchmarks/). They are
reference metadata, not executable test fixtures.

## Install the package

TusoPerturb requires Python 3.11 or newer. An editable install is recommended
because the bundled reference data lives at the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Prediction uses NumPy and scikit-learn and does not require a GPU. There is no
optional dependency group and no external API key.

## Configure data paths

Set path overrides before importing `tusoperturb`:

| Variable | Purpose |
|---|---|
| `TUSOPERTURB_REF_DIR` | Directory containing the Reactome, GO, Hallmark, PROGENy, CollecTRI, and STRING reference files. |
| `TUSOPERTURB_DEPMAP_DIR` | Directory containing `HepG2.pq`, `Jurkat.pq`, `K562.pq`, and `RPE1.pq`. |
| `TUSOPERTURB_COESS_DIR` | Directory containing `depmap_coess_64.npy`, `depmap_coess_32.npy`, `depmap_coess_gene_names.json`, and `depmap_coess_manifest.json`. |
| `TUSOPERTURB_MEAN_EFFECT_DIR` | Directory containing the PerturbHD phenotype Parquet files used by `predict_hit`. |

The first three are optional when running from a source checkout with
`data/embeddings/` intact. `predict_hit` requires the fourth, because those
phenotype tables are not bundled; it raises a `FileNotFoundError` naming the
expected file if the variable is unset.

```bash
export TUSOPERTURB_MEAN_EFFECT_DIR=/path/to/mean_effects_aucell
```

## Verify the local installation

These checks use only bundled data and no benchmark dataset.

### Feature matrix

```python
from tusoperturb.feature_builder import build_features

perts = ["TP53", "MYC", "KRAS", "TP53+MYC"]
E, offsets, coverage = build_features(perts, donor_id="K562")

assert E.shape == (4, 8700)
assert offsets["reactome"] == (0, 1816)
assert offsets["depmap"] == (8599, 8604)
assert offsets["coess64"] == (8604, 8668)
assert offsets["coess32"] == (8668, 8700)
print(coverage)
```

`donor_id` accepts a cell line (`K562`, `RPE1`, `HepG2`, `Jurkat`) or a staged
dataset identifier (`replogle22k562`, `replogle22rpe1`, `nadig25hepg2`,
`nadig25jurkat`). Anything else falls back to K562.

### Head table

```python
from tusoperturb import HEAD_CONFIGS, assert_two_head, head_deviation

assert assert_two_head() == {}
assert len({id(c) for c in HEAD_CONFIGS.values()}) == 2
print(sorted(head_deviation()))
```

`assert_two_head()` returns an empty dict only if the four shared slots resolve
to one field-identical config and the hit slot is distinct. A non-empty return
means the shipped head table no longer matches the architecture this repository
documents.

### Predictor

```python
import numpy as np
from tusoperturb import HEAD_CONFIGS, head_predict

rng = np.random.default_rng(0)
E = rng.normal(size=(40, 8700)).astype(np.float32)
Y_train = rng.normal(size=(30, 12)).astype(np.float32)
train_idx = np.arange(30)

Y_all = head_predict(E, train_idx, Y_train, HEAD_CONFIGS["cellsim"])
assert Y_all.shape == (40, 12)

Y_test = head_predict(E, train_idx, Y_train, HEAD_CONFIGS["systema"],
                      test_idx=np.arange(30, 40))
assert Y_test.shape == (10, 12)
```

These checks validate package loading and array contracts. They do not validate
benchmark preprocessing or scoring.

## External requirements for shared-feature workflows

CellSimBench, scPerturBench, PerturbHD regression, and PerturbHD hit prediction
all call `build_shared_features(dataset)`. The supported dataset identifiers are:

```text
nadig25hepg2
nadig25jurkat
replogle22k562
replogle22rpe1
```

### `perturb_2026` stage loader

The external module `perturb_2026.loop.gpu_stage_loader` must provide
`load_stage(dataset)`. The v2 builder expects the returned mapping to contain:

| Key | Expected contents |
|---|---|
| `Y_train` | Training target array with shape `(n_train_perts, n_genes)`. |
| `all_perts` | Every perturbation label, defining feature-matrix row order. |
| `train_perts` | Training labels in the same row order as `Y_train`. |
| `donor_id` | Cell-line identifier used for DepMap lookup and output construction. |
| `mean_baseline` | Control mean-expression vector with length `n_genes`. |
| `gene_names` | Gene labels in the same order as `Y_train` columns. |

`train_perts` must be a subset of `all_perts`, and perturbation labels must
match the labels used by the reference and phenotype tables.

v1 additionally required `E_all_genept` and `E_train_genept`. v2 does not read
either key; a stage that still carries them works unchanged.

`build_shared_features` asserts that `resolve_depmap_line(donor_id)` agrees with
the dataset's own DepMap line and raises if it does not. This is deliberate: v1
silently fell back to K562 for HepG2 and Jurkat, and the assertion makes that
failure mode loud rather than invisible.

The on-disk layout behind `load_stage` is owned by the external `perturb_2026`
package and is not defined by this repository.

### AnnData output helper

`predict_regression` also imports:

```text
perturb_2026.loop.helpers.build_pred_adata_from_matrix
perturb_2026.loop.paths.FOLD
```

The supplied `master` object must satisfy that helper and the downstream
benchmark scorer. TusoPerturb does not inspect it directly.
`predict_regression_gene` does not use the AnnData helper but still requires
`load_stage`.

### PerturbHD phenotype tables

`predict_hit` expects one file per dataset:

```text
<TUSOPERTURB_MEAN_EFFECT_DIR>/<paper_key>-h.all-all.pq
```

The `paper_key` values are documented in
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

pred_cellsim = predict_regression(master, dataset, seed=1, head="cellsim")
pred_cellsim.write_h5ad("cellsim-nadig25hepg2.h5ad")

pred_scperturb = predict_regression(master, dataset, seed=1, head="scperturb")
pred_scperturb.write_h5ad("scperturb-nadig25hepg2.h5ad")

pred_regression = predict_regression_gene(master, dataset, seed=1,
                                          head="perturbhd_reg")
pred_regression.to_parquet("perturbhd-reg-nadig25hepg2.parquet", index=False)

pred_hit = predict_hit(master, dataset, seed=1)
pred_hit.to_parquet("perturbhd-hit-nadig25hepg2-seed1.parquet", index=False)
```

`cellsim`, `scperturb`, and `perturbhd_reg` resolve to the same head object, so
with identical staged inputs their underlying prediction matrix is the same;
only the output format and downstream scorer differ.

The regression paths do not use `seed`. PerturbHD hit prediction does use it, to
select `split-1`, `split-2`, or `split-3` from the phenotype table.

## External requirements for Systema

Systema does not use `perturb_2026`. It expects a `panel_master` object with:

- `all_perts`: all perturbation labels;
- `train_perts`: labels corresponding to the rows of `Y_train_post`;
- `test_perts`: labels to predict;
- `Y_train_post`: training expression array with shape
  `(len(train_perts), n_genes)`; and
- optionally, `donor_id` for DepMap selection.

Every training and test label must occur in `all_perts`. Column order is not
stored separately by TusoPerturb, so the caller preserves the same gene order in
`Y_train_post`, ground truth, and scoring code.

```python
import numpy as np
from tusoperturb import predict_systema

Y_test = predict_systema(panel_master, seed=1)
assert Y_test.shape[0] == len(panel_master.test_perts)

np.save("systema-predictions.npy", Y_test)
```

`seed` is retained for harnesses that build a different `panel_master` per
split; the prediction function does not use it.

Systema is the one path that predicts with `test_idx` set, so its per-row
outputs depend on which other rows are in the same prediction batch. Submit the
panel's full test set in one call, exactly as the harness does, or the numbers
will not match. See [`docs/leakage.md`](docs/leakage.md).

## Score predictions

TusoPerturb does not bundle benchmark scorers. Use the scorer version associated
with each benchmark and preserve the expected perturbation and gene ordering.
See the pages under [`docs/benchmarks/`](docs/benchmarks/) for adapter-specific
output contracts.

For a valid comparison with the recorded values, keep all of the following
fixed:

- dataset release and preprocessing;
- train, validation, and test assignments;
- target construction;
- scorer implementation and metric aggregation;
- TusoPerturb configuration; and
- dependency versions where exact numerical agreement matters.

The recorded run additionally pinned the BLAS kernel by setting
`OPENBLAS_CORETYPE=Haswell` before the first NumPy import, which made outputs
bit-identical across machines. Reproducing to the last decimal place requires
the same pin; reproducing to within floating-point noise does not.

## Compare with the recorded values

```python
import json
from pathlib import Path

params = json.loads(Path("champion/params.json").read_text())
rec = params["recorded_results_sealed_test"]

print(rec["head_to_head"])

for bench, datasets in rec["cells"].items():
    for ds, metrics in datasets.items():
        for metric, cell in metrics.items():
            if cell.get("is_placeholder"):
                continue
            print(bench, ds, metric, cell["direction"],
                  cell["tusoperturb_v2"]["mean"],
                  cell["old_tusoperturb"]["mean"],
                  cell["v2_better"])
```

Each cell carries the v2 value, the old-TusoPerturb value, the metric direction,
the seed count, and the across-seed standard deviation. `head_to_head` counts
only the 164 scored cells; the seven Systema `composite_dev` cells are recorded
with `is_placeholder: true` and excluded.

What the comparison baseline is, and how it relates to the v1.0.0 release, is
recorded in the `comparison_baseline` block of the same file.

Small floating-point differences may appear across NumPy, scikit-learn, BLAS,
and operating-system combinations. Larger differences usually indicate a change
in input ordering, target preprocessing, split assignment, or scorer version.

## Rebuild the co-essentiality basis

The 96 co-essentiality columns are the only bundled asset derived by fitting
rather than by download. The script that produced them, including the
cell-line-holdout control, is documented in
[`scripts/gen_embeddings/README.md`](scripts/gen_embeddings/README.md).

## Troubleshooting

### `ImportError: No module named perturb_2026`

Install or expose the external `perturb_2026` package before using the shared
feature builder. Systema-only workflows do not require it.

### `FileNotFoundError: Annotation reference files not found`

Use an editable installation from the source checkout, or set
`TUSOPERTURB_REF_DIR`, `TUSOPERTURB_DEPMAP_DIR`, and `TUSOPERTURB_COESS_DIR`
before starting Python.

### `FileNotFoundError: PerturbHD hit prediction needs the AUCell mean-effect tables`

Set `TUSOPERTURB_MEAN_EFFECT_DIR` and confirm the filename uses the expected
`paper_key`.

### `AssertionError: ... resolves to DepMap line ...`

The staged `donor_id` does not match the dataset's cell line. Fix the stage
rather than the assertion; predictions built on the wrong DepMap line are not
comparable to the recorded values.

### Shape or ordering mismatch

Confirm that `train_perts` matches `Y_train` row order, `all_perts` defines the
feature-matrix row order, and `gene_names` matches every expression-target
column.
