"""Annotation adapter: builds the 8604-D annotation stack for every benchmark.

Blocks (in order):
  reactome  (1816)
  go_bp     (5406)
  hallmark  (50)
  progeny   (14)
  collectri (1185)
  string    (128)
  depmap    (5)
Total: 8604 dims.

This block stack originated as the Systema-native feature space. In v2 it is
the *shared* feature space: all five benchmarks are built here, then the
96-column co-essentiality block is appended (see
`tusoperturb.feature_builder`). Nothing is sliced out of a wider matrix, so
each benchmark gets its own cell line's DepMap block rather than a fallback.

Donors are mapped to DepMap cell lines via `DEPMAP_MAP`. Combinatorial perts
(Norman `X+Y`) combine per-gene rows by SUM for multi-hot indicator blocks and
MEAN for dense-embedding blocks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


# --- reference-data resolution ------------------------------------------------
# Reference data resolution: environment override first, then the copies
# vendored under TusoPerturb/data/embeddings/.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent  # .../TusoPerturb


def _resolve_dir(env_var: str, vendored: Path, what: str) -> Path:
    env = os.environ.get(env_var)
    if env and Path(env).is_dir():
        return Path(env)
    if vendored.is_dir():
        return vendored
    raise FileNotFoundError(
        f"{what} not found. Expected {vendored} (source checkout) or "
        f"${env_var} pointing at a directory containing them.")


REF_DIR = _resolve_dir("TUSOPERTURB_REF_DIR",
                       _PKG_ROOT / "data" / "embeddings" / "ref",
                       "Annotation reference files")
DEPMAP_DIR = _resolve_dir("TUSOPERTURB_DEPMAP_DIR",
                          _PKG_ROOT / "data" / "embeddings" / "depmap_essentiality",
                          "DepMap essentiality tables")


BLOCK_NAMES = ("reactome", "go_bp", "hallmark", "progeny", "collectri", "string", "depmap")


def _extract_pert_genes(pert: str) -> List[str]:
    """`X+ctrl` -> ['X']; `X+Y` -> ['X','Y']; `ctrl+X` -> ['X']; `ctrl` -> []."""
    parts = str(pert).split("+")
    return [p for p in parts if p != "ctrl"]


def _combine_rows(rows: List[np.ndarray], mode: str) -> np.ndarray:
    if len(rows) == 0:
        raise ValueError("no rows to combine")
    if len(rows) == 1:
        return rows[0]
    stacked = np.stack(rows)
    if mode == "sum":  return stacked.sum(axis=0)
    if mode == "mean": return stacked.mean(axis=0)
    if mode == "max":  return stacked.max(axis=0)
    raise ValueError(f"unknown mode {mode}")


_CACHE: Dict[str, tuple] = {}


def _reactome_matrix():
    if "reactome" not in _CACHE:
        m = sparse.load_npz(REF_DIR / "reactome_multihot.npz")
        with open(REF_DIR / "annot_gene_names.json") as f:
            names = json.load(f)
        _CACHE["reactome"] = (m, {g: i for i, g in enumerate(names)})
    return _CACHE["reactome"]


def _gobp_matrix():
    if "go_bp" not in _CACHE:
        m = sparse.load_npz(REF_DIR / "go_bp_multihot.npz")
        with open(REF_DIR / "annot_gene_names.json") as f:
            names = json.load(f)
        _CACHE["go_bp"] = (m, {g: i for i, g in enumerate(names)})
    return _CACHE["go_bp"]


def _hallmark_matrix():
    if "hallmark" not in _CACHE:
        m = sparse.load_npz(REF_DIR / "hallmark_multihot.npz")
        with open(REF_DIR / "hallmark_genes_list.json") as f:
            names = json.load(f)
        _CACHE["hallmark"] = (m, {g: i for i, g in enumerate(names)})
    return _CACHE["hallmark"]


def _collectri_matrix():
    if "collectri" not in _CACHE:
        m = sparse.load_npz(REF_DIR / "collectri_multihot.npz")
        with open(REF_DIR / "collectri_targets.json") as f:
            names = json.load(f)
        _CACHE["collectri"] = (m, {g: i for i, g in enumerate(names)})
    return _CACHE["collectri"]


def _progeny_matrix():
    if "progeny" not in _CACHE:
        m = sparse.load_npz(REF_DIR / "progeny.npz").toarray().astype(np.float32)
        with open(REF_DIR / "progeny_targets.json") as f:
            names = json.load(f)
        _CACHE["progeny"] = (m, {g: i for i, g in enumerate(names)})
    return _CACHE["progeny"]


def _string_matrix():
    if "string" not in _CACHE:
        emb = np.load(REF_DIR / "string_v12_n2v_128.npy").astype(np.float32)
        with open(REF_DIR / "string_v12_gene_names.json") as f:
            names = json.load(f)
        with open(REF_DIR / "string_v12_pert_aliases.json") as f:
            aliases = json.load(f)
        _CACHE["string"] = (emb, {g: i for i, g in enumerate(names)}, aliases)
    return _CACHE["string"]


def _depmap_matrix(cell_line: str):
    key = f"depmap_{cell_line}"
    if key not in _CACHE:
        df = pd.read_parquet(DEPMAP_DIR / f"{cell_line}.pq").set_index("gene")
        cols = ["own_effect", "own_abs_effect", "pop_mean", "pop_std", "selectivity"]
        _CACHE[key] = (df[cols].astype(np.float32),
                       {g.upper(): i for i, g in enumerate(df.index)})
    return _CACHE[key]


# Donor -> vendored DepMap cell line.
#
# v1 had no HepG2 or Jurkat entry, so every PerturbHD-family dataset that went
# through this adapter silently took the K562 dependency block. The sealed v2
# run carried each dataset's own line, so the map does too.
DEPMAP_MAP = {
    # Lines vendored under data/embeddings/depmap_essentiality/.
    "K562": "K562",
    "RPE1": "RPE1",
    "HepG2": "HepG2",
    "Jurkat": "Jurkat",
    # Systema donors with no vendored DepMap line -> K562.
    "A549": "K562",
    "iPSC": "K562",
    "melanoma": "K562",
    "H1_hESC": "K562",
    # `gpu_stage_loader` reports `donor_id` as the dataset key rather than the
    # cell line, so the four PerturbHD-family keys resolve here too. Without
    # them `build_features(perts, stage['donor_id'])` would take the K562
    # fallback for HepG2, Jurkat and RPE1.
    "replogle22k562": "K562",
    "replogle22rpe1": "RPE1",
    "nadig25hepg2": "HepG2",
    "nadig25jurkat": "Jurkat",
}
_DEPMAP_MAP_UPPER = {k.upper(): v for k, v in DEPMAP_MAP.items()}


def resolve_depmap_line(donor_id: str) -> str:
    """Donor id or dataset key -> DepMap cell line, case-insensitively.

    Unknown donors fall back to K562, as in v1; the fallback now only fires for
    cell lines that genuinely have no vendored DepMap table.
    """
    return _DEPMAP_MAP_UPPER.get(str(donor_id).upper(), "K562")


def build_systema_features(
    perts: List[str],
    donor_id: str,
    blocks: Tuple[str, ...] = BLOCK_NAMES,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], Dict[str, str]]:
    """Build (n_perts, 8604) feature matrix (all 7 blocks by default)."""
    n = len(perts)
    parts, offsets, stats = [], {}, {}
    off = 0
    depmap_cl = resolve_depmap_line(donor_id)

    if "reactome" in blocks:
        m, idx = _reactome_matrix()
        A = np.zeros((n, m.shape[1]), dtype=np.float32); hits = 0
        for i, p in enumerate(perts):
            rows = []
            for g in _extract_pert_genes(p):
                if g in idx:
                    row = m[idx[g], :].toarray().flatten()
                    if row.sum() > 0: rows.append(row)
            if rows: A[i] = _combine_rows(rows, "sum"); hits += 1
        parts.append(A); offsets["reactome"] = (off, off + m.shape[1]); off += m.shape[1]
        stats["reactome"] = f"{hits}/{n} ({hits/n:.1%})"

    if "go_bp" in blocks:
        m, idx = _gobp_matrix()
        A = np.zeros((n, m.shape[1]), dtype=np.float32); hits = 0
        for i, p in enumerate(perts):
            rows = []
            for g in _extract_pert_genes(p):
                if g in idx:
                    row = m[idx[g], :].toarray().flatten()
                    if row.sum() > 0: rows.append(row)
            if rows: A[i] = _combine_rows(rows, "sum"); hits += 1
        parts.append(A); offsets["go_bp"] = (off, off + m.shape[1]); off += m.shape[1]
        stats["go_bp"] = f"{hits}/{n} ({hits/n:.1%})"

    if "hallmark" in blocks:
        m, idx = _hallmark_matrix()
        A = np.zeros((n, m.shape[1]), dtype=np.float32); hits = 0
        for i, p in enumerate(perts):
            rows = []
            for g in _extract_pert_genes(p):
                if g in idx:
                    row = m[idx[g], :].toarray().flatten()
                    if row.sum() > 0: rows.append(row)
            if rows: A[i] = _combine_rows(rows, "sum"); hits += 1
        parts.append(A); offsets["hallmark"] = (off, off + m.shape[1]); off += m.shape[1]
        stats["hallmark"] = f"{hits}/{n} ({hits/n:.1%})"

    if "progeny" in blocks:
        m, idx = _progeny_matrix()
        A = np.zeros((n, m.shape[1]), dtype=np.float32); hits = 0
        for i, p in enumerate(perts):
            rows = []
            for g in _extract_pert_genes(p):
                if g in idx:
                    row = m[idx[g], :]
                    if np.any(row != 0): rows.append(row)
            if rows: A[i] = _combine_rows(rows, "mean"); hits += 1
        parts.append(A); offsets["progeny"] = (off, off + m.shape[1]); off += m.shape[1]
        stats["progeny"] = f"{hits}/{n} ({hits/n:.1%})"

    if "collectri" in blocks:
        m, idx = _collectri_matrix()
        A = np.zeros((n, m.shape[1]), dtype=np.float32); hits = 0
        for i, p in enumerate(perts):
            rows = []
            for g in _extract_pert_genes(p):
                if g in idx:
                    row = m[idx[g], :].toarray().flatten()
                    if row.sum() > 0: rows.append(row)
            if rows: A[i] = _combine_rows(rows, "sum"); hits += 1
        parts.append(A); offsets["collectri"] = (off, off + m.shape[1]); off += m.shape[1]
        stats["collectri"] = f"{hits}/{n} ({hits/n:.1%})"

    if "string" in blocks:
        emb, idx, aliases = _string_matrix()
        A = np.zeros((n, emb.shape[1]), dtype=np.float32); hits = 0
        for i, p in enumerate(perts):
            rows = []
            for g in _extract_pert_genes(p):
                if g in idx:
                    rows.append(emb[idx[g]])
                elif g in aliases and aliases[g] in idx:
                    rows.append(emb[idx[aliases[g]]])
            if rows: A[i] = _combine_rows(rows, "mean"); hits += 1
        parts.append(A); offsets["string"] = (off, off + emb.shape[1]); off += emb.shape[1]
        stats["string"] = f"{hits}/{n} ({hits/n:.1%})"

    if "depmap" in blocks:
        df, idx = _depmap_matrix(depmap_cl)
        A = np.zeros((n, df.shape[1]), dtype=np.float32); hits = 0
        for i, p in enumerate(perts):
            rows = []
            for g in _extract_pert_genes(p):
                gu = g.upper()
                if gu in idx:
                    rows.append(df.iloc[idx[gu]].values.astype(np.float32))
            if rows: A[i] = _combine_rows(rows, "mean"); hits += 1
        parts.append(A); offsets["depmap"] = (off, off + df.shape[1]); off += df.shape[1]
        stats["depmap"] = f"{hits}/{n} ({hits/n:.1%}) [{depmap_cl}]"

    E = np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), dtype=np.float32)
    return E, offsets, stats
