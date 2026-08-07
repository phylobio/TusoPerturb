# Co-essentiality basis generation

This directory contains the script that rebuilds the DepMap co-essentiality
basis used by TusoPerturb's shared feature builder. The basis is the trailing
96 columns of the 8,700-dimensional feature space: a rank-64 block and a
rank-32 block, both indexed by gene symbol.

It is the only bundled asset produced by fitting rather than by download, so it
is the only one that needs a script. The seven annotation blocks — Reactome, GO
Biological Process, MSigDB Hallmark, PROGENy, CollecTRI, STRING, and DepMap
essentiality — are distributed under [`data/embeddings/`](../../data/embeddings/)
and are not regenerated here; their provenance is recorded in the manifests
under [`data/embeddings/ref/`](../../data/embeddings/ref/).

The shipped tables are already in the repository. Rebuilding is only necessary
to move to a newer DepMap release, to audit the pipeline, or to reproduce the
cell-line-holdout control.

## Files

| File | Purpose |
|---|---|
| `build_coessentiality_embeddings.py` | Truncated SVD of the DepMap gene-effect matrix; writes both shipped ranks, the gene-name index, and the manifest |

## Requirements

The script needs only `numpy`, `pandas`, and `scikit-learn`, which are already
dependencies of the package:

```bash
python -m pip install -e .
```

It also needs one input file: `CRISPRGeneEffect.csv` from a DepMap public
release, available from <https://depmap.org/portal/download/>. No API key, no
network access at run time, and no paid service are involved.

Rows of that file are cell-line model identifiers (`ACH-......`), columns are
gene labels of the form `A1BG (1)`, and values are Chronos gene effect scores.

## What the script computes

1. Strip the Entrez identifier from each column label and upper-case the symbol.
2. Transpose to genes × cell lines.
3. Collapse duplicate gene symbols by `nanmean`.
4. Impute residual missing values with the per-gene mean.
5. Drop genes that are non-finite or have zero variance across lines.
6. Centre each gene, so the SVD describes *differential* dependency profile
   rather than mean essentiality — the latter is already carried by the
   five-column `depmap` block.
7. `TruncatedSVD(n_components=k, random_state=0, algorithm="randomized")` for
   k = 64 and k = 32.

The fit is unsupervised on the external DepMap matrix. No benchmark target,
split index, or perturbation label enters it.

## Provenance of the shipped basis

| Quantity | Value |
|---|---|
| Input | DepMap `CRISPRGeneEffect.csv` (Chronos gene effect, public release) |
| Cell lines | 1,186 |
| Genes after QC | 18,435 unique symbols |
| Missing fraction before impute | 0.0352248 |
| Explained variance ratio sum, k = 64 | 0.3414 |
| Explained variance ratio sum, k = 32 | 0.2690 |

Both ranks are shipped and both are used. The rank-32 block is not a subspace
of the rank-64 block, because each is fitted independently.

## Rebuild

From the repository root:

```bash
python scripts/gen_embeddings/build_coessentiality_embeddings.py \
  path/to/CRISPRGeneEffect.csv
```

This overwrites the four files in
[`data/embeddings/coessentiality/`](../../data/embeddings/coessentiality/):

| File | Contents |
|---|---|
| `depmap_coess_64.npy` | `float32`, `(18435, 64)` |
| `depmap_coess_32.npy` | `float32`, `(18435, 32)` |
| `depmap_coess_gene_names.json` | `list[str]`, row order for both tables |
| `depmap_coess_manifest.json` | Provenance and explained-variance record |

Write elsewhere with `--out-dir`:

```bash
python scripts/gen_embeddings/build_coessentiality_embeddings.py \
  path/to/CRISPRGeneEffect.csv \
  --out-dir /tmp/coess_rebuild
```

The run prints the recovered gene count, line count, and explained-variance
sums next to the shipped values. A different DepMap release will shift all of
them; the shipped basis is the one the recorded results were produced with, so
a rebuild from a newer release is a new basis, not a reproduction.

The script pins `OPENBLAS_CORETYPE=Haswell` before importing NumPy. The
randomized SVD is deterministic given a fixed BLAS kernel and `random_state=0`;
letting OpenBLAS dispatch on the host CPU makes the trailing digits of the basis
machine-dependent.

Runtime is a few minutes, peaking around 6 GB of resident memory.

## Cell-line-holdout control

Three of the four benchmark cell lines are themselves among the screened DepMap
lines: K562 (`ACH-000551`), HEPG2 (`ACH-000739`), and JURKAT (`ACH-000995`).
RPE1 is not a cancer line and does not appear. The basis could in principle
encode those lines' own viability rather than a transferable functional prior.

`--holdout-lines` drops all three *before* any statistic is computed — before
the per-gene mean impute, before centring, before the SVD — so no benchmark line
can reach the basis through any path:

```bash
python scripts/gen_embeddings/build_coessentiality_embeddings.py \
  path/to/CRISPRGeneEffect.csv \
  --holdout-lines
```

Every output filename gains a `_ho` suffix, so the control never overwrites the
shipped basis, and the manifest records an `excluded_models` list.

| Quantity | Full basis | Holdout basis |
|---|---|---|
| Cell lines | 1,186 | 1,183 |
| Genes after QC | 18,435 | 18,435 |
| Explained variance ratio sum, k = 64 | 0.3414 | 0.3416 |
| Explained variance ratio sum, k = 32 | 0.2690 | 0.2694 |

The holdout basis reproduced the champion's headline result, which is the
evidence that the co-essentiality gain is a functional prior and not the
benchmark lines' own dependency profile. See
[`docs/leakage.md`](../../docs/leakage.md). The holdout tables themselves are
not shipped, because the champion uses the full basis.

## Use in TusoPerturb

[`tusoperturb/coessentiality.py`](../../tusoperturb/coessentiality.py) loads the
two `.npy` tables and the gene-name index, and maps each perturbation's gene
symbols onto rows. Multi-gene perturbations are averaged within each block;
symbols absent from the index contribute zero. The resulting 96 columns are
appended to the 8,604 annotation columns and scaled by the head's `emb_weight`
before ridge and k-NN fitting — see
[`ARCHITECTURE.md`](../../ARCHITECTURE.md).

Set `TUSOPERTURB_COESS_DIR` to point the loader at a directory other than the
vendored one.
