"""Systema feature adapter: builds the 8604-D per-pert feature matrix.

Wraps `scripts/systema_features.py` from the systema_r1 codebase (vendored
here so TusoPerturb is self-contained).

Blocks (in champion order, same as systema_r1):
  reactome  (1816)
  go_bp     (5406)
  hallmark  (50)
  progeny   (14)
  collectri (1185)
  string    (128)
  depmap    (5)
Total: 8604 dims.

Systema donors are mapped to DepMap cell lines via `DEPMAP_MAP`.
Combinatorial perts (Norman `X+Y`) combine per-gene rows by SUM for multi-hot
indicator blocks and MEAN for dense-embedding blocks.
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
# Prefer vendored TusoPerturb/data/embeddings/{ref,depmap_essentiality}/; fall
# back to the shared-workspace copies used during method development.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent  # .../TusoPerturb

_VENDORED_REF = _PKG_ROOT / "data" / "embeddings" / "ref"
_LEGACY_REF   = Path("/mnt/shared-workspace/biomni-method/perturb-2026/data/ref")
_ENV_REF      = os.environ.get("TUSOPERTURB_REF_DIR")
if _ENV_REF and Path(_ENV_REF).is_dir():
    REF_DIR = Path(_ENV_REF)
elif _VENDORED_REF.is_dir():
    REF_DIR = _VENDORED_REF
else:
    REF_DIR = _LEGACY_REF

_VENDORED_DEPMAP = _PKG_ROOT / "data" / "embeddings" / "depmap_essentiality"
_LEGACY_DEPMAP   = Path("/mnt/shared-workspace/biomni-method/perturb-2026/data/orth_info/depmap_essentiality")
_ENV_DEPMAP      = os.environ.get("TUSOPERTURB_DEPMAP_DIR")
if _ENV_DEPMAP and Path(_ENV_DEPMAP).is_dir():
    DEPMAP_DIR = Path(_ENV_DEPMAP)
elif _VENDORED_DEPMAP.is_dir():
    DEPMAP_DIR = _VENDORED_DEPMAP
else:
    DEPMAP_DIR = _LEGACY_DEPMAP


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


# Best-guess DepMap cell-line mapping per Systema donor.
DEPMAP_MAP = {
    "K562": "K562",
    "A549": "K562",      # A549 not in DepMap set -> K562 fallback
    "RPE1": "RPE1",
    "iPSC": "K562",
    "melanoma": "K562",
    "H1_hESC": "K562",
}


def build_systema_features(
    perts: List[str],
    donor_id: str,
    blocks: Tuple[str, ...] = BLOCK_NAMES,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], Dict[str, str]]:
    """Build (n_perts, 8604) feature matrix (all 7 blocks by default)."""
    n = len(perts)
    parts, offsets, stats = [], {}, {}
    off = 0
    depmap_cl = DEPMAP_MAP.get(donor_id, "K562")

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
