# Gene-embedding generation

TusoPerturb's 11,680-D feature stack combines a single LLM embedding block
(GenePT) with seven static reference blocks and one per-dataset baseline block:

| Block           | Dim  | Source                                             |
|-----------------|------|----------------------------------------------------|
| GenePT          | 3072 | OpenAI `text-embedding-3-large` on gene summaries  |
| Reactome        | 1816 | multi-hot pathway membership                       |
| GO Biological   | 5406 | multi-hot GO BP term membership                    |
| MSigDB Hallmark | 50   | multi-hot pathway membership                       |
| PROGENy         | 14   | signed pathway response signature                  |
| CollecTRI       | 1185 | TF-target multi-hot (per-target)                   |
| STRING v12      | 128  | node2vec embedding of the PPI graph                |
| DepMap essent.  | 5    | per-cell-line essentiality metrics                 |
| Baseline        | 4    | per-pert control-expression statistics             |

The first 8 blocks are static reference data (already vendored in this repo
under `data/embeddings/`). The last block is derived at build time from each
dataset's control expression. **This directory contains only the scripts
needed to (re)generate the *dynamic* per-dataset piece: the GenePT block.**

TusoPerturb does not use Geneformer, ESM-2, scGPT, or Presage embeddings —
those were evaluated during method development but do not appear in the
production feature stack, and their generators are not shipped here.

---

## Directory contents

```
gen_embeddings/
  README.md                              — this file
  gather_embeddings.sh                   — top-level driver (GenePT-only)
  generate_genept_gene_embeddings.py     — GenePT via OpenAI text-embedding-3-large
  transfer_reference_gene_embeddings.py  — copy precomputed embeddings from
                                           one AnnData to another (fast path
                                           for follow-up datasets that share
                                           the same gene panel)
```

GenePT calls the OpenAI embeddings endpoint and does not need a foundation-
model conda env — a standard Python env with `openai`, `scanpy`, `pybiomart`,
and `pandas` is enough. See `pyproject.toml` at the repo root for runtime
dependencies.

---

## Inputs

The generator consumes:

1. A dataset `.h5ad` (`AnnData`) whose `.var` gives the gene panel to embed.
2. External resources:
   - `OPENAI_API_KEY` env var — for `text-embedding-3-large`.
   - NCBI Entrez access — for gene-summary text (unauthenticated works but is
     rate-limited; set `NCBI_API_KEY` to raise the limit).

## Outputs

`generate_genept_gene_embeddings.py` writes its embedding matrix back into
the input `.h5ad`, keyed under `adata.uns["embeddings_genept"]`. Downstream
consumers (the perturb-2026 stage loader in the method-development repo)
extract this key and re-shape it into the per-pert `E_all_genept` matrix
that `perturb_2026.loop.gpu_stage_loader.load_stage()` returns.

## Where GenePT plugs in

The **primary consumer** for TusoPerturb is
`tusoperturb.feature_builder.build_shared_features()`, which stitches the
GenePT block together with the other 8 static/derived blocks:

```python
all_parts = [E_all_genept]                 # 3072
for f in ('reactome','go_bp','hallmark','progeny','collectri','string'):
    all_parts.append(feats_all[f])          # 1816+5406+50+14+1185+128 = 8599
all_parts.append(depmap_feats)              # 5
all_parts.append(pert_baseline)             # 4
E_all = np.concatenate(all_parts, axis=1)   # 11680
```

`feats_all` is produced by `tusoperturb._deps.orth_features_v2.load_orth_features_v2()`,
which reads from `data/embeddings/ref/` in this repo. `depmap_feats` is
read from `data/embeddings/depmap_essentiality/`. Both are byte-identical
to the internal method-development copies (sha256-verified).

## Reproducing GenePT

```bash
export OPENAI_API_KEY=sk-...
cd scripts/gen_embeddings
bash gather_embeddings.sh path/to/dataset.h5ad path/to/dataset_out.h5ad
```

To transfer already-computed GenePT embeddings from a reference `.h5ad`
(much faster than regenerating, and appropriate when your target dataset
shares the same gene panel as the reference):

```bash
bash gather_embeddings.sh \
    path/to/target.h5ad \
    path/to/target_out.h5ad \
    path/to/reference_with_embeddings.h5ad
```

Two caveats when reproducing exact numerical parity:

- OpenAI embedding models are versioned. TusoPerturb's cached GenePT values
  were generated against `text-embedding-3-large` at the model version pinned
  inside `generate_genept_gene_embeddings.py`. If OpenAI updates the model,
  regenerated embeddings may differ at the float level while remaining
  semantically equivalent.
- NCBI Entrez rate-limiting varies by API key. The script batches and retries
  but a fresh key may need a few reruns to fill the summary cache.

## Reproducing STRING / Reactome / GO / Hallmark / CollecTRI / PROGENy

These are **static reference files** with well-defined provenance:

| File                     | Source                                                             |
|--------------------------|--------------------------------------------------------------------|
| `string_v12_n2v_128.npy` | STRING v12 human PPI, node2vec (d=128), computed offline           |
| `reactome_multihot.npz`  | Reactome 2022 gene-set library from `Enrichr`                      |
| `go_bp_multihot.npz`     | GO Biological Process 2023 from `Enrichr`                          |
| `hallmark_multihot.npz`  | MSigDB Hallmark 2020 from `Enrichr`                                |
| `collectri_multihot.npz` | CollecTRI (decoupleR public regulon)                               |
| `progeny.npz`            | PROGENy 14-pathway signature (decoupleR)                           |

The provenance manifests (`annotations_manifest.json`, `string_v12_manifest.json`)
in the vendored `data/embeddings/ref/` directory record the exact source URLs,
dates, and SHA sums. Regeneration is scriptable but rarely needed — the
vendored copies are frozen and byte-verified.

## Reproducing DepMap essentiality

Not scripted here — the 4 parquets (`HepG2.pq`, `Jurkat.pq`, `K562.pq`,
`RPE1.pq`) are per-cell-line summaries of DepMap CRISPR essentiality
scores (columns: `own_effect`, `own_abs_effect`, `pop_mean`, `pop_std`,
`selectivity`). Source: DepMap Public 23Q4 CRISPRGeneEffect matrix,
aggregated per cell line. Regenerate via the DepMap CLI if a newer DepMap
release is desired.
