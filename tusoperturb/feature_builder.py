"""Shared 8700-D feature builder for all five benchmark heads.

One feature space, built the same way for every benchmark::

    annotation stack (8604)   reactome 1816 | go_bp 5406 | hallmark 50
                              | progeny 14 | collectri 1185 | string 128
                              | depmap 5
    co-essentiality    (96)   coess64 64 | coess32 32
    -------------------------------------------------------------------
    total            8700

The annotation stack comes from :mod:`tusoperturb.systema_adapter` and the
trailing block from :mod:`tusoperturb.coessentiality`. Both are built by
gene-symbol lookup from the perturbation names, so a Systema panel and a
CellSim dataset go through exactly the same code with the same column
semantics. Nothing is sliced out of a wider matrix.

Reference data (STRING v12, Reactome, GO BP, Hallmark, PROGENy, CollecTRI,
DepMap essentiality, DepMap co-essentiality) is vendored under
``TusoPerturb/data/embeddings/``.

For the CellSim / scPerturB / PerturbHD datasets the perturbation panel and
targets are keyed by the essential-perturbation gene panel loaded via
`perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`; see
``build_shared_features``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from .coessentiality import COESS_WIDTH, build_coessentiality_block
from .heads import ANNOT_WIDTH, WIDTH
from .systema_adapter import build_systema_features, resolve_depmap_line

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent  # .../TusoPerturb


# `perturb_2026.loop.gpu_stage_loader.load_stage` is only needed by
# `build_shared_features()` (the PerturbHD-family entry point). It is imported
# lazily so that the rest of the package -- `build_features`, `head_predict`,
# the HeadConfig table, the Systema path -- stays importable on machines that
# don't have `perturb_2026` installed.
def _lazy_load_stage():
    try:
        from perturb_2026.loop.gpu_stage_loader import load_stage
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "tusoperturb.feature_builder.build_shared_features() requires the "
            "`perturb_2026` module for its `gpu_stage_loader.load_stage` "
            "function. This module is not distributed with TusoPerturb and "
            "must be installed separately (it lives in the perturb-2026 "
            "method-development monorepo). See docs/benchmarks/cellsim.md for "
            "what the gpu_stage bake contains and how to reproduce it."
        ) from e
    return load_stage


DS_TO_PAPER_KEY = {
    'replogle22k562':  'replogle_k562_essential_full',
    'replogle22rpe1':  'replogle_rpe1_essential_full',
    'nadig25hepg2':    'nadig_hepg2_essential_full',
    'nadig25jurkat':   'nadig_jurkat_essential_full',
}
DS_TO_DEPMAP_KEY = {
    'replogle22k562':  'K562',
    'replogle22rpe1':  'RPE1',
    'nadig25hepg2':    'HepG2',
    'nadig25jurkat':   'Jurkat',
}

# Mean-effect AUCell precomputes are not vendored (~GB). Only the PerturbHD-hit
# workflow consults them, so the variable is resolved lazily and every other
# workflow runs without it.
_ENV_MEAN_EFFECT = os.environ.get('TUSOPERTURB_MEAN_EFFECT_DIR')
MEAN_EFFECT_DIR = Path(_ENV_MEAN_EFFECT) if _ENV_MEAN_EFFECT else None


def mean_effect_path(paper_key: str) -> Path:
    """Locate the PerturbHD AUCell phenotype table for one dataset."""
    if MEAN_EFFECT_DIR is None:
        raise FileNotFoundError(
            "PerturbHD hit prediction needs the AUCell mean-effect tables. "
            "Set TUSOPERTURB_MEAN_EFFECT_DIR to the directory holding "
            f"'{paper_key}-h.all-all.pq' before importing tusoperturb. "
            "See docs/benchmarks/perturbhd_hit.md.")
    return MEAN_EFFECT_DIR / f'{paper_key}-h.all-all.pq'


def build_features(perts: Sequence[str], donor_id: str
                   ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], Dict[str, str]]:
    """Build the (n_perts, 8700) feature matrix for any benchmark.

    Args:
      perts:    perturbation names in row order. `X`, `X+ctrl` and `X+Y` are
                all accepted; a combination averages (dense blocks) or unions
                (indicator blocks) its genes.
      donor_id: cell line of the screen, e.g. 'K562', 'RPE1', 'HepG2',
                'Jurkat'. Selects the 5-column DepMap dependency block;
                unknown donors fall back to K562 (see
                `systema_adapter.resolve_depmap_line`).

    Returns:
      (E, offsets, coverage) where E is (n_perts, 8700) float32, `offsets`
      maps block name -> (start, stop) column bounds in E, and `coverage` maps
      block name -> the fraction of perturbations that hit that block.
    """
    E_annot, offsets, stats = build_systema_features(perts, donor_id)
    if E_annot.shape[1] != ANNOT_WIDTH:
        raise AssertionError(
            f"annotation stack is {E_annot.shape[1]} wide, expected {ANNOT_WIDTH}")

    E_coess, coess_offsets, coess_stats = build_coessentiality_block(perts)
    if E_coess.shape[1] != COESS_WIDTH:
        raise AssertionError(
            f"co-essentiality block is {E_coess.shape[1]} wide, expected {COESS_WIDTH}")

    for name, (lo, hi) in coess_offsets.items():
        offsets[name] = (ANNOT_WIDTH + lo, ANNOT_WIDTH + hi)
    stats.update(coess_stats)

    E = np.concatenate([E_annot, E_coess], axis=1).astype(np.float32)
    if E.shape[1] != WIDTH:
        raise AssertionError(f"feature width {E.shape[1]} != {WIDTH}")
    return E, offsets, stats


def build_shared_features(dataset: str) -> dict:
    """Build the 8700-D feature matrix and targets for a PerturbHD-family dataset.

    Args:
      dataset: one of 'nadig25hepg2', 'nadig25jurkat', 'replogle22k562',
               'replogle22rpe1'.

    Returns a dict with:
      E_all:         (n_all_perts, 8700) float32
      Y_train:       (n_train_perts, n_genes) target deltas from load_stage
      all_perts:     list[str] pert names in E_all row order
      train_perts:   list[str] pert names used for training
      train_indices: indices into E_all/all_perts of the training perts
      donor_id:      'K562', 'RPE1', 'HepG2', or 'Jurkat'
      mean_baseline: (n_genes,) control mean expression
      gene_names:    list[str] gene symbols in Y_train column order
    """
    load_stage = _lazy_load_stage()
    stage = load_stage(dataset)
    Y_train = stage['Y_train']
    all_perts = stage['all_perts']
    train_perts = stage['train_perts']
    donor_id = stage['donor_id']
    mean_baseline = stage['mean_baseline']
    gene_names = stage['gene_names']

    # v1 mapped every PerturbHD-family donor onto the K562 dependency block by
    # accident (no HepG2/Jurkat entry in DEPMAP_MAP). The sealed v2 run used
    # the dataset's own line, so assert the two routes agree rather than
    # trusting the fallback.
    expected = DS_TO_DEPMAP_KEY[dataset]
    resolved = resolve_depmap_line(donor_id)
    if resolved != expected:
        raise AssertionError(
            f"{dataset}: donor_id {donor_id!r} resolves to DepMap line "
            f"{resolved!r}, expected {expected!r}")

    E_all, _, _ = build_features(all_perts, donor_id)

    all_idx = {p: i for i, p in enumerate(all_perts)}
    train_indices = np.array([all_idx[p] for p in train_perts])

    return {
        'E_all': E_all,
        'Y_train': Y_train,
        'all_perts': all_perts,
        'train_perts': train_perts,
        'train_indices': train_indices,
        'donor_id': donor_id,
        'mean_baseline': mean_baseline,
        'gene_names': gene_names,
    }
