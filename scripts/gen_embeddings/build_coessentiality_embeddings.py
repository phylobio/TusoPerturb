#!/usr/bin/env python
"""Rebuild the DepMap co-essentiality basis shipped in ``data/embeddings/coessentiality``.

The basis is a truncated SVD of the DepMap CRISPR gene-effect matrix, transposed
to genes x cell lines and centred per gene.  Genes required by the same cell
lines end up near each other, so the embedding encodes *differential* dependency
structure -- functional co-membership -- rather than mean essentiality, which the
5-column DepMap block already carries.

The fit is unsupervised and touches the external DepMap matrix only.  No
benchmark target, split index, or perturbation label enters it.

Three of the four benchmark cell lines are themselves among the screened DepMap
lines (K562, HepG2, Jurkat; RPE1 is not a cancer line and is absent).  Pass
``--holdout-lines`` to drop them *before* any statistic is computed -- before the
per-gene mean impute, before centring, before the SVD -- so no benchmark line can
reach the basis through any path.  That is the control described in
``docs/leakage.md``.

Inputs
------
``CRISPRGeneEffect.csv`` from a DepMap public release (https://depmap.org/portal/download/).
Rows are cell-line model IDs (``ACH-......``), columns are gene labels of the form
``A1BG (1)``, values are Chronos gene effect scores.

Outputs (written to ``--out-dir``)
----------------------------------
``depmap_coess_64.npy``            float32, (n_genes, 64)
``depmap_coess_32.npy``            float32, (n_genes, 32)
``depmap_coess_gene_names.json``   list[str], row order for both tables
``depmap_coess_manifest.json``     provenance + explained-variance record

With ``--holdout-lines`` every filename gains a ``_ho`` suffix so the control
basis never overwrites the shipped one.

Usage
-----
    python scripts/gen_embeddings/build_coessentiality_embeddings.py CRISPRGeneEffect.csv
    python scripts/gen_embeddings/build_coessentiality_embeddings.py CRISPRGeneEffect.csv --holdout-lines

Runtime is a few minutes and peaks around 6 GB of RAM for a 1186 x 18443 matrix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Pin the BLAS kernel before numpy is imported.  The SVD is deterministic given a
# fixed kernel; letting OpenBLAS dispatch on the host CPU makes the last few
# digits of the basis machine-dependent.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

import numpy as np

# DepMap model IDs for the benchmark's own cell lines, resolved via
# StrippedCellLineName in the DepMap ``Model.csv`` table.
BENCH_MODELS = {
    "ACH-000551": "K562 (replogle22k562, replogle_k562_gwps)",
    "ACH-000739": "HEPG2 (nadig25hepg2)",
    "ACH-000995": "JURKAT (nadig25jurkat)",
}

# Widths shipped with the repo.  The head consumes both: a 64-column block and a
# 32-column block, concatenated (see ARCHITECTURE.md, FEATURE_BLOCKS).
SHIPPED_KS = (64, 32)

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "embeddings" / "coessentiality"


def build(csv_path: Path, out_dir: Path, ks=SHIPPED_KS, exclude_models=(), suffix="") -> dict:
    import pandas as pd

    print(f"[read] {csv_path.name}  ({csv_path.stat().st_size / 1e6:.0f} MB)")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"[read] raw matrix {df.shape}  (cell lines x genes)")

    if exclude_models:
        present = [m for m in exclude_models if m in df.index]
        df = df.drop(index=present)
        print(f"[excl] dropped {len(present)} benchmark cell lines -> {df.shape}")
        for m in present:
            print(f"         - {m}  {BENCH_MODELS.get(m, '')}")
        missing = [m for m in exclude_models if m not in present]
        if missing:
            print(f"[excl] not in matrix, nothing to drop: {missing}")

    # Column labels look like "A1BG (1)": strip the Entrez id, upper-case.
    genes = [c.split(" (")[0].strip().upper() for c in df.columns]
    X = df.to_numpy(dtype=np.float32).T  # genes x cell lines
    del df

    # Collapse duplicate symbols by nanmean.
    order: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        order.setdefault(g, []).append(i)
    uniq = sorted(order)
    M = np.empty((len(uniq), X.shape[1]), dtype=np.float32)
    for j, g in enumerate(uniq):
        idx = order[g]
        M[j] = X[idx[0]] if len(idx) == 1 else np.nanmean(X[idx], axis=0)
    del X
    print(f"[qc  ] unique symbols {M.shape}")

    # Per-gene mean impute, drop zero-variance / non-finite genes, per-gene centre.
    nan_frac = float(np.isnan(M).mean())
    gmean = np.nanmean(M, axis=1, keepdims=True)
    gmean = np.where(np.isnan(gmean), 0.0, gmean)
    M = np.where(np.isnan(M), np.broadcast_to(gmean, M.shape), M)
    keep = np.isfinite(M).all(1) & (M.std(1) > 0)
    M, uniq = M[keep], [g for g, k in zip(uniq, keep) if k]
    M = M - M.mean(1, keepdims=True)
    print(f"[qc  ] after QC {M.shape}  nan_frac_before={nan_frac:.6f}")

    from sklearn.decomposition import TruncatedSVD

    out_dir.mkdir(parents=True, exist_ok=True)
    components: dict[str, dict] = {}
    for k in ks:
        svd = TruncatedSVD(n_components=k, random_state=0, algorithm="randomized")
        Z = svd.fit_transform(M).astype(np.float32)
        evr = float(svd.explained_variance_ratio_.sum())
        fname = f"depmap_coess_{k}{suffix}.npy"
        np.save(out_dir / fname, Z)
        components[str(k)] = dict(file=fname, shape=list(Z.shape),
                                  explained_variance_ratio_sum=round(evr, 4))
        print(f"[svd ] k={k:>3}  shape={Z.shape}  evr_sum={evr:.4f}  -> {fname}")

    names_file = f"depmap_coess_gene_names{suffix}.json"
    (out_dir / names_file).write_text(json.dumps(uniq))

    manifest = dict(
        source="DepMap CRISPRGeneEffect.csv (Chronos gene effect, public release)",
        n_genes=len(uniq),
        n_cell_lines=int(M.shape[1]),
        nan_frac_before=nan_frac,
        preprocessing="strip Entrez ids from column labels; transpose to genes x cell "
                      "lines; collapse duplicate symbols by nanmean; per-gene mean "
                      "impute; drop zero-variance and non-finite genes; per-gene "
                      "centre; TruncatedSVD",
        leakage="unsupervised TruncatedSVD on the external DepMap matrix only; no "
                "benchmark target, split index or perturbation label enters the fit",
        shipped_components=components,
        gene_names=f"{names_file}, row order for both tables",
        rebuild="scripts/gen_embeddings/build_coessentiality_embeddings.py",
    )
    if exclude_models:
        manifest["excluded_models"] = list(exclude_models)
    man_file = f"depmap_coess_manifest{suffix}.json"
    (out_dir / man_file).write_text(json.dumps(manifest, indent=2))
    print(f"[out ] {out_dir}/{man_file}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild the DepMap co-essentiality SVD basis.")
    ap.add_argument("csv", type=Path,
                    help="path to DepMap CRISPRGeneEffect.csv")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="output directory (default: the vendored "
                         "data/embeddings/coessentiality)")
    ap.add_argument("--holdout-lines", action="store_true",
                    help="drop the three benchmark cell lines present in DepMap "
                         "(ACH-000551 K562, ACH-000739 HEPG2, ACH-000995 JURKAT) "
                         "before any statistic, and write with a _ho suffix")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"not found: {args.csv}")

    excl = tuple(BENCH_MODELS) if args.holdout_lines else ()
    suffix = "_ho" if args.holdout_lines else ""
    man = build(args.csv, args.out_dir, ks=SHIPPED_KS,
                exclude_models=excl, suffix=suffix)

    # Reference values from the shipped basis, for a quick sanity check against a
    # DepMap release other than the one used here.
    ref = {"": dict(n_genes=18435, n_cell_lines=1186, evr={"64": 0.3414, "32": 0.2690}),
           "_ho": dict(n_genes=18435, n_cell_lines=1183, evr={"64": 0.3416, "32": 0.2694})}[suffix]
    print(f"\n[chk ] n_genes      {man['n_genes']:>6}   shipped {ref['n_genes']}")
    print(f"[chk ] n_cell_lines {man['n_cell_lines']:>6}   shipped {ref['n_cell_lines']}")
    for k, v in ref["evr"].items():
        got = man["shipped_components"][k]["explained_variance_ratio_sum"]
        print(f"[chk ] evr k={k:<3}    {got:.4f}   shipped {v:.4f}")
    print("\n[note] a different DepMap release will shift these numbers; the shipped "
          "basis is the one the recorded results were produced with.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
