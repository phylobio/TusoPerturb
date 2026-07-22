# TusoPerturb

**A single per-perturbation ridge model with a 11,680-D knowledge-graph feature
stack, state of the art on 4 of 5 published perturbation-response benchmarks
without hyperparameter sweeps.**

TusoPerturb harmonizes three prior best-in-class codebases (Systema `res_wr20_wk70_wb10`,
`final_methodv3.A_v2_193`, `final_methodv3.build_pred_hit_v7`) behind a single
Python package with a single public API. The five benchmark heads all instantiate
the same `head_predict` function; they differ only in which arms are enabled,
which feature subset they consume, and how the target is shaped.

## Table of contents

- [What's in this repo](#whats-in-this-repo)
- [What's not in this repo](#whats-not-in-this-repo)
- [Installation](#installation)
- [Quick start (Systema head, no external staging)](#quick-start-systema-head-no-external-staging)
- [Running against each benchmark](#running-against-each-benchmark)
- [Directory map](#directory-map)
- [Environment variables](#environment-variables)
- [Documentation index](#documentation-index)
- [Citations](#citations)
- [License](#license)

## What's in this repo

| Component | What it is |
|---|---|
| [`tusoperturb/`](tusoperturb/) | The Python package. 5 modules + a vendored `_deps/` subpackage. |
| [`tusoperturb/_deps/`](tusoperturb/_deps/) | Vendored copies of the `orth_features` loaders that read the annotation refs. |
| [`data/embeddings/ref/`](data/embeddings/ref/) | 13 MB of static reference data: STRING v12 node2vec embedding, Reactome / GO BP / MSigDB Hallmark / CollecTRI multi-hot matrices, PROGENy signature, and provenance manifests. |
| [`data/embeddings/depmap_essentiality/`](data/embeddings/depmap_essentiality/) | Per-cell-line DepMap essentiality parquets (HepG2, Jurkat, K562, RPE1). |
| [`champion/params.json`](champion/params.json) | Pinned hyperparameters for the champion configuration. |
| [`scripts/gen_embeddings/`](scripts/gen_embeddings/) | GenePT regeneration script (the only LLM embedding used in the 11,680-D stack) + notes on the source of the static reference matrices. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Full model architecture: feature stack, arms, blend, per-head configs. |
| [`REPRODUCE.md`](REPRODUCE.md) | How to reproduce the champion numerical result. |
| [`report.md`](report.md) | Full write-up of the benchmark results with headline tables. |
| [`docs/benchmarks/`](docs/benchmarks/) | One page per benchmark: what data you need, which function to call, how to score. |

## What's not in this repo

TusoPerturb was trained + evaluated inside a method-development sandbox that
carried the benchmark master AnnData files, the pre-staged feature caches, and
the published-baseline tables. **None of that is in this zip**, on purpose.
Here's what you'd need to bring yourself and where to find it:

| Missing | Size | Where it lives | Which heads need it |
|---|---|---|---|
| Benchmark master `h5ad`s (nadig25hepg2/jurkat, replogle22k562/rpe1) | ~14–20 GB each | Published with each benchmark. See [`docs/benchmarks/`](docs/benchmarks/). | `predict_regression`, `predict_regression_gene`, `predict_hit` |
| `gpu_stage/<dataset>/` bakes (per-perturbation pre-staged features) | ~2 GB each | Not published; built from master h5ad via the perturb-2026 staging pipeline. Schema documented in [`docs/benchmarks/`](docs/benchmarks/) so you can rebuild. | `predict_regression`, `predict_regression_gene`, `predict_hit` |
| `perturb_2026` Python module | code | The perturb-2026 method-development monorepo. `feature_builder.py` and `api.py` import from it (lazily). | Everything except `predict_systema` |
| PerturbHD-hit AUCell phenotype parquets | ~1 GB | Perturb-HD supplementary data. | `predict_hit` |
| Systema `PanelMaster` builder | code | The `systema_r1` benchmark harness (Vinas 2025). | `predict_systema` |

**The Systema head is the only one that runs with just this zip's contents plus its own harness's `PanelMaster` object.** The 4 PerturbHD-family heads
additionally need `perturb_2026` on your `PYTHONPATH`. Every benchmark page in
[`docs/benchmarks/`](docs/benchmarks/) restates this explicitly.

## Installation

```bash
git clone <repo-url> tusoperturb
cd tusoperturb
pip install -e .
```

Requires Python 3.11+. See [`pyproject.toml`](pyproject.toml) for runtime
dependencies (numpy, scipy, scikit-learn, pandas, scanpy, anndata, pyarrow).

**Why editable?** The vendored reference data at `data/embeddings/` is resolved
relative to the repo root by `feature_builder.py`. Editable installs (`-e`)
keep the source tree in place. A wheel install (`pip install .` without `-e`)
installs the code only; you'd then need to set `TUSOPERTURB_REF_DIR` and
`TUSOPERTURB_DEPMAP_DIR` env vars to point at the reference data yourself
(see [Environment variables](#environment-variables) below).

## Quick start (Systema head, no external staging)

The Systema head does not need the perturb-2026 staging pipeline or the
per-benchmark master h5ad. Given a `PanelMaster` (from the `systema_r1`
harness — see [`docs/benchmarks/systema.md`](docs/benchmarks/systema.md)):

```python
from tusoperturb import predict_systema

# panel_master must expose:
#   .all_perts     list[str] all perturbations in the panel
#   .train_perts   list[str] training perts (subset)
#   .test_perts    list[str] test perts (subset)
#   .Y_train_post  (n_train, n_genes) log1p post-expression for the train perts
#   .donor_id      'K562', 'RPE1', 'HepG2', or 'Jurkat'
Y_test = predict_systema(panel_master, seed=1)  # (n_test, n_genes) log1p post-expression
```

That's the full path from feature build → 3-arm ridge/kNN/binary blend → out.

## Running against each benchmark

| Benchmark | Head function | External data needed | Instructions |
|---|---|---|---|
| CellSimBench (Miller 2025) | `predict_regression(head='cellsim')` | master h5ad + gpu_stage bake + `perturb_2026` | [docs/benchmarks/cellsim.md](docs/benchmarks/cellsim.md) |
| scPerturBench (Wei 2024) | `predict_regression(head='scperturb')` | master h5ad + gpu_stage bake + `perturb_2026` | [docs/benchmarks/scperturb.md](docs/benchmarks/scperturb.md) |
| PerturbHD-regression | `predict_regression_gene(head='perturbhd_reg')` | master h5ad + gpu_stage bake + `perturb_2026` | [docs/benchmarks/perturbhd_reg.md](docs/benchmarks/perturbhd_reg.md) |
| PerturbHD-hit | `predict_hit()` | master h5ad + gpu_stage bake + AUCell phenotype `.pq` + `perturb_2026` | [docs/benchmarks/perturbhd_hit.md](docs/benchmarks/perturbhd_hit.md) |
| Systema (Vinas 2025) | `predict_systema()` | `PanelMaster` from `systema_r1` harness | [docs/benchmarks/systema.md](docs/benchmarks/systema.md) |

All 5 heads reuse the same `head_predict` primitive with a different
`HeadConfig` (see [`tusoperturb/heads.py`](tusoperturb/heads.py) and
[ARCHITECTURE.md §3](ARCHITECTURE.md)):

- `PRIOR_A_v2_193` → CellSim, scPerturB, PerturbHD-reg (plain Ridge α=110)
- `PRIOR_R17_C` → PerturbHD-hit (3-arm blend, drop-GenePT features)
- `PRIOR_SYSTEMA` → Systema (3-arm blend, residual-over-pert-mean target)

## Directory map

```
tusoperturb/                             Python package
├── README.md                            ← you are here
├── LICENSE                              MIT
├── pyproject.toml                       pip install target
├── ARCHITECTURE.md                      full model architecture + feature stack
├── REPRODUCE.md                         reproducing the champion numerical result
├── report.md                            full benchmark write-up
├── tusoperturb/
│   ├── __init__.py                      public API surface
│   ├── api.py                           predict_regression / predict_hit / predict_systema
│   ├── heads.py                         HeadConfig + PRIOR_* configs + FEATURE_BLOCKS
│   ├── predictor.py                     head_predict (the 3-arm blend primitive)
│   ├── feature_builder.py               shared 11,680-D feature stack (PerturbHD-family)
│   ├── systema_adapter.py               Systema-native 8,604-D feature stack
│   └── _deps/
│       ├── __init__.py
│       ├── orth_features.py             legacy reactome/GO/hallmark/collectri/progeny loader
│       └── orth_features_v2.py          v2 loader used by feature_builder + systema_adapter
├── champion/
│   └── params.json                      pinned hyperparameters for the champion config
├── data/
│   └── embeddings/
│       ├── ref/                         static reference data (STRING v12, Reactome, GO BP,
│       │                                 MSigDB Hallmark, CollecTRI, PROGENy, manifests)
│       └── depmap_essentiality/         DepMap essentiality per cell line (4 parquets)
├── scripts/
│   └── gen_embeddings/                  regenerate the dynamic GenePT block
│       ├── README.md
│       ├── gather_embeddings.sh                   GenePT-only driver (+ transfer mode)
│       ├── generate_genept_gene_embeddings.py     OpenAI text-embedding-3-large
│       └── transfer_reference_gene_embeddings.py  copy embeddings between AnnDatas
└── docs/
    └── benchmarks/                      per-benchmark run instructions
        ├── README.md                    benchmark index
        ├── cellsim.md
        ├── scperturb.md
        ├── perturbhd_reg.md
        ├── perturbhd_hit.md
        └── systema.md
```

## Environment variables

Only relevant if the vendored reference data has been moved or you're
running from a wheel install (see [Installation](#installation)):

| Var | Purpose | Default |
|---|---|---|
| `TUSOPERTURB_REF_DIR` | Override where `orth_features_v2` reads Reactome / GO BP / Hallmark / CollecTRI / PROGENy / STRING refs from. | `data/embeddings/ref/` next to the repo root. |
| `TUSOPERTURB_DEPMAP_DIR` | Override where `feature_builder` reads DepMap essentiality parquets from. | `data/embeddings/depmap_essentiality/` next to the repo root. |
| `TUSOPERTURB_MEAN_EFFECT_DIR` | Override where `predict_hit` reads PerturbHD-hit AUCell mean-effect parquets from. | Original method-development sandbox path; set this outside that sandbox. |

These fall back to vendored or legacy method-development paths when the
corresponding env var is unset. That fallback is a safety net for reproducing
byte-identical results against the original sandbox, not something you'd
normally rely on.

## Documentation index

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — feature stack, 3-arm blend math,
  per-head HeadConfig table, data flow diagram.
- **[REPRODUCE.md](REPRODUCE.md)** — reproducing the champion numerical result
  end to end. Assumes you have the external data listed above.
- **[report.md](report.md)** — full benchmark write-up with per-benchmark
  metric tables and interpretation.
- **[docs/benchmarks/README.md](docs/benchmarks/README.md)** — index of the
  five per-benchmark instruction pages.
- **[scripts/gen_embeddings/README.md](scripts/gen_embeddings/README.md)** —
  regenerating the vendored gene/annotation embeddings from raw sources.

## Citations

TusoPerturb builds on published benchmarks and knowledge sources. Please cite
the underlying works if you use it:

- **CellSimBench**: Miller et al. *bioRxiv* 2025.
- **scPerturBench**: Wei et al. *bioRxiv* 2024.
- **PerturbHD**: Vinas et al. 2025.
- **Systema**: Vinas et al. 2025.
- **STRING v12**: Szklarczyk et al. *NAR* 2023.
- **Reactome / GO BP / MSigDB Hallmark**: via Enrichr (Kuleshov et al. 2016).
- **CollecTRI**: Müller-Dott et al. 2023.
- **PROGENy**: Schubert et al. *Nat Commun* 2018.
- **DepMap**: Broad Institute 2023.
- **GenePT**: Chen & Zou 2023 (`text-embedding-3-large`).

## License

MIT — see [LICENSE](LICENSE).
