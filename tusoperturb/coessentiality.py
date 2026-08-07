"""Co-essentiality block: the 96 trailing columns of the shared feature space.

Two truncated-SVD bases of the DepMap CRISPR gene-effect matrix (18435 genes x
1186 cell lines), per-gene centred so the components describe a gene's
*differential* dependency profile across cell lines rather than its mean
essentiality -- the latter is already carried by the 5-column `depmap` block.

    coess64   64 columns   k=64 basis
    coess32   32 columns   k=32 basis

Both bases are built by `scripts/gen_embeddings/build_coessentiality_embeddings.py`
and vendored under ``data/embeddings/coessentiality/``. Shipping both is not
redundancy: the two ranks resolve different scales of the dependency
correlation structure, and the shared head reads the concatenation.

A block row is the mean of its perturbed genes' rows (`X+Y` -> mean of X and
Y); `ctrl` contributes nothing, and a gene missing from the basis leaves a zero
row, exactly as the `string` block does.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# --- reference-data resolution ------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent  # .../TusoPerturb

_VENDORED_COESS = _PKG_ROOT / 'data' / 'embeddings' / 'coessentiality'
_ENV_COESS = os.environ.get('TUSOPERTURB_COESS_DIR')
COESS_DIR = (Path(_ENV_COESS) if _ENV_COESS and Path(_ENV_COESS).is_dir()
             else _VENDORED_COESS)

# (block name, array file, width) in the order they are concatenated.
BLOCKS: Tuple[Tuple[str, str, int], ...] = (
    ('coess64', 'depmap_coess_64.npy', 64),
    ('coess32', 'depmap_coess_32.npy', 32),
)
GENE_NAMES_FILE = 'depmap_coess_gene_names.json'
COESS_WIDTH = sum(w for _, _, w in BLOCKS)  # 96

_CACHE: Dict[str, Tuple[np.ndarray, Dict[str, int]]] = {}


def _table(block: str) -> Tuple[np.ndarray, Dict[str, int]]:
    if block not in _CACHE:
        spec = {b: (f, w) for b, f, w in BLOCKS}[block]
        arr_f, width = spec
        E = np.load(COESS_DIR / arr_f).astype(np.float32)
        names = json.loads((COESS_DIR / GENE_NAMES_FILE).read_text())
        if E.shape[0] != len(names):
            raise AssertionError(f"{block}: {E.shape} rows vs {len(names)} gene names")
        if E.shape[1] != width:
            raise AssertionError(f"{block}: width {E.shape[1]} != {width}")
        _CACHE[block] = (E, {g: i for i, g in enumerate(names)})
    return _CACHE[block]


def _extract_pert_genes(pert: str) -> List[str]:
    """`X+ctrl` -> ['X']; `X+Y` -> ['X','Y']; `ctrl` -> []."""
    return [p for p in str(pert).split('+') if p and p != 'ctrl']


def lookup_block(block: str, perts: Sequence[str]) -> Tuple[np.ndarray, str]:
    """(n_perts, width) block by gene-symbol lookup, plus its coverage string."""
    E, idx = _table(block)
    n = len(perts)
    A = np.zeros((n, E.shape[1]), dtype=np.float32)
    hits = 0
    for i, p in enumerate(perts):
        rows = []
        for g in _extract_pert_genes(p):
            j = idx.get(g)
            if j is None:
                j = idx.get(str(g).upper())
            if j is not None:
                rows.append(E[j])
        if rows:
            A[i] = rows[0] if len(rows) == 1 else np.stack(rows).mean(axis=0)
            hits += 1
    return A, f"{hits}/{n} ({hits / max(n, 1):.1%})"


def build_coessentiality_block(perts: Sequence[str]
                               ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]],
                                          Dict[str, str]]:
    """Build the (n_perts, 96) co-essentiality block.

    Returns (block, offsets, coverage) with offsets relative to the start of
    the block, matching the shape of `build_systema_features`'s return.
    """
    parts, offsets, stats = [], {}, {}
    off = 0
    for name, _, width in BLOCKS:
        A, cov = lookup_block(name, perts)
        parts.append(A)
        offsets[name] = (off, off + width)
        off += width
        stats[name] = cov
    return np.concatenate(parts, axis=1).astype(np.float32), offsets, stats
