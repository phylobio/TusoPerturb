"""Shared 11680-D feature builder for CellSim / scPerturB / PerturbHD heads.

Builds the same feature stack A_v2_193 uses:
  GenePT (3072) + reactome (1816) + go_bp (5406) + hallmark (50) + progeny (14)
  + collectri (1185) + string (128) + depmap (5) + baseline (4) = 11680.

For CellSim / scPerturB / PerturbHD-hit / PerturbHD-reg the features are keyed
by the essential-perturbation gene panel loaded via
`perturb_2026.loop.gpu_stage_loader.load_stage(dataset)`.

Reference data (STRING v12, Reactome, GO BP, Hallmark, PROGENy, CollecTRI,
DepMap essentiality) is vendored under ``TusoPerturb/data/embeddings/``. The
vendored orth-feature loaders live in ``tusoperturb._deps`` and are used by
default; the module still falls back to the shared-workspace copies used
during method development if the vendored copies are missing.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# --- bootstrap ----------------------------------------------------------------
# Local package root (parent of the tusoperturb/ folder) is inserted first so
# `import tusoperturb._deps.orth_features_v2` resolves to the vendored copy.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent  # .../TusoPerturb
_BOOTSTRAP = [
    str(_PKG_ROOT),
    str(_HERE / '_deps'),
    # Legacy method-development paths — kept for byte-identity re-runs against
    # the sprint_004 / sprint_005 trees. Harmless if the paths don't exist.
    '/mnt/results/method_repo_76/code/sprint_deps',
    '/mnt/results/method_repo_76/code',
    '/mnt/shared-workspace/biomni-method/perturb-2026/experiments/sprints/sprint_004',
    '/mnt/shared-workspace/biomni-method/perturb-2026/experiments/sprints/sprint_005',
    '/mnt/shared-workspace/biomni-method/perturb-2026/benchmark/reference_repos/perturb-hd/packages/perturb-hd/src',
    '/mnt/shared-workspace/biomni-method/perturb-2026/benchmark/reference_repos/calibrated-metrics',
    '/workspace',
]
for p in _BOOTSTRAP:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

# `perturb_2026.loop.gpu_stage_loader.load_stage` is only needed by
# `build_shared_features()` (the PerturbHD-family entry point). It is
# imported lazily so that the rest of the module — `head_predict`, the
# HeadConfig table, the vendored orth-feature loaders, systema_adapter —
# remains importable on machines that don't have `perturb_2026` installed.
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


# Prefer the vendored loader; fall back to the flat legacy import.
try:
    from tusoperturb._deps.orth_features_v2 import load_orth_features_v2 as load_orth_features
except ImportError:
    from orth_features_v2 import load_orth_features_v2 as load_orth_features


FEATURES = ('reactome', 'go_bp', 'hallmark', 'progeny', 'collectri', 'string')

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

# --- reference-data resolution ------------------------------------------------
# DepMap parquets: prefer vendored TusoPerturb/data/embeddings/depmap_essentiality/,
# fall back to the shared-workspace method-dev copy.
_VENDORED_DEPMAP = _PKG_ROOT / 'data' / 'embeddings' / 'depmap_essentiality'
_LEGACY_DEPMAP   = Path('/mnt/shared-workspace/biomni-method/perturb-2026/data/orth_info/depmap_essentiality')
_ENV_DEPMAP      = os.environ.get('TUSOPERTURB_DEPMAP_DIR')
if _ENV_DEPMAP and Path(_ENV_DEPMAP).is_dir():
    DEPMAP_DIR = Path(_ENV_DEPMAP)
elif _VENDORED_DEPMAP.is_dir():
    DEPMAP_DIR = _VENDORED_DEPMAP
else:
    DEPMAP_DIR = _LEGACY_DEPMAP

# Mean-effect AUCell precomputes are not vendored (too large; ~GB). This path
# is only consulted by PerturbHD-hit workflows. Prefer an explicit environment
# override and retain the shared-workspace default for byte-identity re-runs.
_ENV_MEAN_EFFECT = os.environ.get('TUSOPERTURB_MEAN_EFFECT_DIR')
MEAN_EFFECT_DIR = (
    Path(_ENV_MEAN_EFFECT)
    if _ENV_MEAN_EFFECT
    else Path('/mnt/shared-workspace/biomni-method/perturb-2026/data/raw/perturbhd_precomputed/mean_effects_aucell')
)


def _load_depmap_features(dataset, all_perts):
    key = DS_TO_DEPMAP_KEY[dataset]
    df = pd.read_parquet(DEPMAP_DIR / f'{key}.pq').set_index('gene')
    cols = ['own_effect', 'own_abs_effect', 'pop_mean', 'pop_std', 'selectivity']
    features = np.zeros((len(all_perts), 5), dtype=np.float32)
    gene_lookup = {g.upper(): i for i, g in enumerate(df.index)}
    for i, pert in enumerate(all_perts):
        p_upper = pert.upper() if isinstance(pert, str) else str(pert).upper()
        if p_upper in gene_lookup:
            features[i] = df.iloc[gene_lookup[p_upper]][cols].values.astype(np.float32)
    return features


def _per_pert_baseline_features(all_perts, gene_names, mean_baseline):
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    n = len(all_perts)
    features = np.zeros((n, 4), dtype=np.float32)
    baseline_mean = mean_baseline.mean()
    baseline_std = max(mean_baseline.std(), 1e-6)
    sorted_baseline = np.sort(mean_baseline)
    for i, pert in enumerate(all_perts):
        p_upper = pert.upper() if isinstance(pert, str) else str(pert).upper()
        if p_upper in gene_to_idx:
            idx = gene_to_idx[p_upper]
            val = float(mean_baseline[idx])
            features[i, 0] = val
            features[i, 1] = np.log1p(max(val, 0.0))
            features[i, 2] = (val - baseline_mean) / baseline_std
            features[i, 3] = float(np.searchsorted(sorted_baseline, val)) / len(sorted_baseline)
    return features


def build_shared_features(dataset: str) -> dict:
    """Build the shared 11680-D feature matrix for a PerturbHD-family dataset.

    Returns a dict with:
      E_all:        (n_all_perts, 11680) float32
      Y_train:      (n_train_perts, n_genes) target deltas from load_stage
      all_perts:    list[str] pert names in E_all row order
      train_perts:  list[str] pert names used for training
      train_indices: indices into E_all/all_perts of the training perts
      donor_id:     'K562', 'RPE1', 'HepG2', or 'Jurkat'
      mean_baseline: (n_genes,) control mean expression
      gene_names:   list[str] gene symbols in Y_train column order
    """
    load_stage = _lazy_load_stage()
    stage = load_stage(dataset)
    E_train_genept = stage['E_train_genept']
    E_all_genept = stage['E_all_genept']
    Y_train = stage['Y_train']
    all_perts = stage['all_perts']
    train_perts = stage['train_perts']
    donor_id = stage['donor_id']
    mean_baseline = stage['mean_baseline']
    gene_names = stage['gene_names']

    feats_all, _ = load_orth_features(all_perts, features=FEATURES)
    depmap_feats = _load_depmap_features(dataset, all_perts)
    pert_baseline = _per_pert_baseline_features(all_perts, gene_names, mean_baseline)
    all_idx = {p: i for i, p in enumerate(all_perts)}
    train_indices = np.array([all_idx[p] for p in train_perts])

    all_parts = [E_all_genept]
    for f in FEATURES:
        all_parts.append(feats_all[f])
    all_parts.append(depmap_feats)
    all_parts.append(pert_baseline)

    E_all = np.concatenate(all_parts, axis=1).astype(np.float32)

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
