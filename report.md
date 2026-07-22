# Recorded benchmark results

This document summarizes the values stored in
[`champion/params.json`](champion/params.json) under
`validated_results_3seed_mean`.

The repository does not include the raw prediction files, per-seed scorer
outputs, baseline predictions, or complete third-party evaluation pipelines.
The numbers and ranks below should therefore be treated as project-recorded
reference results, not as an independently reproducible leaderboard bundled
with the source code. See [`REPRODUCE.md`](REPRODUCE.md) for the inputs needed
to rerun the evaluations.

## Evaluation scope

| Benchmark interface | Dataset or panel count | Recorded primary metric |
|---|---:|---|
| CellSimBench | 4 | `pearson_deltactrl_degs` |
| scPerturBench | 4 | `pcc_delta` |
| PerturbHD regression | 4 | `corr` |
| PerturbHD hit prediction | 4 | `recall_at_budget_0.05` |
| Systema | 7 | `corr_20de` |
| **Total** | **23** |  |

Nineteen of the 23 entries are marked as rank 1 in the stored metadata. Four of
those entries have `total_methods=1`, which indicates that no comparative
baseline was represented for that dataset in the recorded table.

## Model configurations

TusoPerturb exposes five benchmark keys backed by three configurations:

| Benchmark keys | Feature representation | Predictor |
|---|---|---|
| `cellsim`, `scperturb`, `perturbhd_reg` | Shared 11,680-dimensional representation | Standard-scaled fixed ridge model with alpha 110 and block weights |
| `perturbhd_hit` | Shared representation without the 3,072-dimensional GenePT block | Robust-scaled ridge/kNN/binary blend |
| `systema` | Systema-native 8,604-dimensional representation | Robust-scaled ridge/kNN/binary blend with residual target centering |

The three expression-regression keys use the same numerical configuration.
Differences among their reported scores come from output shaping and the
external benchmark scorers, not from different model coefficients or
hyperparameters.

Full implementation details are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## CellSimBench

Recorded metric: `pearson_deltactrl_degs`.

| Dataset | Three-seed mean | Recorded rank |
|---|---:|---:|
| `nadig25hepg2` | 0.642327 | 1 / 11 |
| `nadig25jurkat` | 0.660456 | 1 / 1 |
| `replogle22k562` | 0.684758 | 1 / 11 |
| `replogle22rpe1` | 0.738199 | 1 / 1 |

The `1 / 1` entries should be read as recorded scores without a baseline
comparison.

## scPerturBench

Recorded metric: `pcc_delta`.

| Dataset | Three-seed mean | Recorded rank |
|---|---:|---:|
| `nadig25hepg2` | 0.644262 | 1 / 1 |
| `nadig25jurkat` | 0.664925 | 1 / 1 |
| `replogle22k562` | 0.681765 | 1 / 15 |
| `replogle22rpe1` | 0.727599 | 1 / 15 |

The first two entries have no comparative baseline represented in the stored
metadata.

## PerturbHD regression

Recorded metric: `corr`.

| Dataset | Three-seed mean | Recorded rank |
|---|---:|---:|
| `nadig25hepg2` | 0.727730 | 1 / 6 |
| `nadig25jurkat` | 0.714003 | 1 / 6 |
| `replogle22k562` | 0.725986 | 1 / 6 |
| `replogle22rpe1` | 0.774313 | 1 / 6 |

## PerturbHD hit prediction

Recorded metric: `recall_at_budget_0.05`.

| Dataset | Three-seed mean | Recorded rank |
|---|---:|---:|
| `nadig25hepg2` | 0.468000 | 1 / 9 |
| `nadig25jurkat` | 0.377172 | 1 / 9 |
| `replogle22k562` | 0.478667 | 1 / 9 |
| `replogle22rpe1` | 0.450000 | 1 / 9 |

## Systema

Recorded metric: `corr_20de`.

| Panel | Three-seed mean | Recorded rank |
|---|---:|---:|
| `adamson` | 0.795273 | 1 / 8 |
| `norman` | 0.714719 | 1 / 8 |
| `replogle_k562_gwps` | 0.518520 | 1 / 8 |
| `replogle_rpe1` | 0.650952 | 3 / 8 |
| `tian_crispra` | 0.404583 | 7 / 8 |
| `tian_crispri` | 0.481148 | 6 / 8 |
| `xu` | 0.416661 | 4 / 8 |

The Systema results vary substantially by panel. The stored metadata records
three first-place entries and four lower-ranked entries; a single aggregate
claim would obscure that variation.

## Interpreting the tables

Scores should be compared only within the same benchmark, dataset, metric,
preprocessing pipeline, and split definition. Raw values from different metric
families are not directly comparable.

The tables do not provide uncertainty estimates. Although the JSON labels the
values as three-seed means, it does not include the individual seed values or
standard deviations. It also does not identify exact scorer revisions or store
the baseline rows used to derive each rank.

For publication or a new benchmark submission, rerun the appropriate official
scorer and report:

- the dataset release and preprocessing;
- split definitions and seeds;
- TusoPerturb version and configuration;
- scorer repository revision;
- per-seed values and aggregation method; and
- the baseline source used for any rank comparison.

## Source of record

The machine-readable values remain in
[`champion/params.json`](champion/params.json). The JSON also stores the model
configuration snapshot used by this repository. `tusoperturb/heads.py` is the
runtime source of truth for the configurations actually executed by the
package.
