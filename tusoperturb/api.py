"""TusoPerturb public API: 3 predict_* functions.

Each function wraps the harmonized `head_predict` with the correct
per-benchmark HeadConfig and IO shaping. Signatures mirror the source
codebases (final_methodv3 `build_pred_adata` / `build_pred_hit` and
systema_r1 `predict`).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd

from .heads import HEAD_CONFIGS, HeadConfig
from .predictor import head_predict
from . import feature_builder as fb
from .feature_builder import build_shared_features, DS_TO_PAPER_KEY

# perturb_2026.loop.helpers.build_pred_adata_from_matrix expects a specific
# AnnData wrapper -- import lazily so this module is importable even if
# perturb-2026 isn't on sys.path (e.g. Systema-only workflow).
def _lazy_bpafm():
    try:
        from perturb_2026.loop.helpers import build_pred_adata_from_matrix
        from perturb_2026.loop.paths import FOLD
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "predict_regression() shapes its output into a pred-AnnData via "
            "`perturb_2026.loop.helpers.build_pred_adata_from_matrix`. This "
            "module is not distributed with TusoPerturb (see "
            "docs/benchmarks/cellsim.md). If you only need the raw prediction "
            "matrix, call `predict_regression_gene()` instead, which returns "
            "a plain (pert, gene, effect) DataFrame with no perturb_2026 "
            "dependency for its output shaping (it still needs perturb_2026 "
            "for the gpu_stage feature loader upstream)."
        ) from e
    return build_pred_adata_from_matrix, FOLD


# ---------- Regression heads (CellSim, scPerturB, PerturbHD-reg) ----------

def predict_regression(master, dataset: str, seed: int,
                       head: str = 'cellsim',
                       cfg: Optional[HeadConfig] = None):
    """Predict per-pert Y_delta for the CellSim / scPerturB / PerturbHD-reg benchmarks.

    Reproduces final_methodv3.A_v2_193.build_pred_adata verbatim when
    cfg = PRIOR_A_v2_193 (the default via head='cellsim' / 'scperturb' / 'perturbhd_reg').

    Args:
      master:  reference master AnnData (used for train/test split alignment
               via build_pred_adata_from_matrix; not touched by head_predict).
      dataset: one of 'nadig25hepg2', 'nadig25jurkat', 'replogle22k562', 'replogle22rpe1'.
      seed:    ignored by the regression path (A_v2_193 is deterministic, and
               regression heads don't depend on split-{seed}); kept in signature
               for API uniformity with the hit head.
      head:    which HeadConfig row to use. Defaults to 'cellsim' which is
               PRIOR_A_v2_193 (same as 'scperturb', 'perturbhd_reg').
      cfg:     override HeadConfig if provided.

    Returns:
      pred_adata: AnnData with the predicted post-expression per-pert per-gene.
    """
    if cfg is None:
        cfg = HEAD_CONFIGS[head]

    sd = build_shared_features(dataset)
    Y_delta = head_predict(sd['E_all'], sd['train_indices'], sd['Y_train'], cfg)
    X = sd['mean_baseline'][np.newaxis, :] + Y_delta

    build_pred_adata_from_matrix, FOLD = _lazy_bpafm()
    return build_pred_adata_from_matrix(X, sd['all_perts'], sd['donor_id'], master, fold=FOLD)


def predict_regression_gene(master, dataset: str, seed: int,
                            head: str = 'perturbhd_reg',
                            cfg: Optional[HeadConfig] = None) -> pd.DataFrame:
    """Emit pert/gene/effect DataFrame for the PerturbHD-regression scorer."""
    if cfg is None:
        cfg = HEAD_CONFIGS[head]

    sd = build_shared_features(dataset)
    Y_delta = head_predict(sd['E_all'], sd['train_indices'], sd['Y_train'], cfg)

    rows = []
    gene_names = sd['gene_names']
    all_perts = sd['all_perts']
    for i, pert in enumerate(all_perts):
        for j, gene in enumerate(gene_names):
            rows.append((pert, gene, float(Y_delta[i, j])))
    return pd.DataFrame(rows, columns=['pert', 'gene', 'effect'])


# ---------- Hit head (PerturbHD-hit) ----------

def predict_hit(master, dataset: str, seed: int,
                cfg: Optional[HeadConfig] = None) -> pd.DataFrame:
    """Predict AUCell phenotype scores for the PerturbHD-hit benchmark.

    Reproduces final_methodv3.build_pred_hit_v7 verbatim when cfg = PRIOR_R17_C.

    Args:
      master:  reference master AnnData.
      dataset: one of 'nadig25hepg2', 'nadig25jurkat', 'replogle22k562', 'replogle22rpe1'.
      seed:    which split-{seed} column in the AUCell pivot to use as train/test.
      cfg:     override HeadConfig; default HEAD_CONFIGS['perturbhd_hit'] = PRIOR_R17_C.

    Returns:
      DataFrame with columns ['pert', 'pheno', 'hit_score'] restricted to
      test + val perts.
    """
    if cfg is None:
        cfg = HEAD_CONFIGS['perturbhd_hit']

    # Load pheno pivot (targets).
    paper_key = DS_TO_PAPER_KEY[dataset]
    aucell_path = fb.MEAN_EFFECT_DIR / f'{paper_key}-h.all-all.pq'
    df = pd.read_parquet(aucell_path)
    df = df.drop_duplicates(subset=['pert', 'pheno'], keep=False)
    pivot = df.pivot(index='pert', columns='pheno', values='mean_diff')
    split_col = f'split-{seed}' if seed in [1, 2, 3] else 'split-1'
    condition_to_split = df.drop_duplicates(subset=['pert']).set_index('pert')[split_col]
    pivot['split'] = pd.Series(pivot.index).map(condition_to_split).to_numpy()

    # Build features (only rows in valid_perts are used).
    sd = build_shared_features(dataset)
    E_all = sd['E_all']
    all_perts_str = sd['all_perts']
    pert_to_idx = {p: i for i, p in enumerate(all_perts_str)}
    valid_perts = [p for p in pivot.index if p in pert_to_idx]
    idx = np.array([pert_to_idx[p] for p in valid_perts])
    E_valid = E_all[idx]
    pivot = pivot.loc[valid_perts]

    train_mask = (pivot['split'] == 'train').to_numpy()
    Y_all = pivot.drop(columns=['split']).to_numpy()
    pheno_cols = pivot.drop(columns=['split']).columns.tolist()
    split_labels = pivot['split'].to_numpy()

    train_row_idx = np.where(train_mask)[0]
    Y_train_valid = Y_all[train_mask]

    Y_pred = head_predict(E_valid, train_row_idx, Y_train_valid, cfg)

    pred_df = pd.DataFrame(index=valid_perts, columns=pheno_cols, data=Y_pred)
    pred_df = pred_df.reset_index().melt(
        id_vars='index', value_vars=pheno_cols,
        var_name='pheno', value_name='hit_score',
    )
    pred_df = pred_df.rename(columns={'index': 'pert'})
    condition_to_split_dict = dict(zip(valid_perts, split_labels))
    pred_df['split'] = pred_df['pert'].map(condition_to_split_dict)
    pred_df = pred_df[pred_df['split'].isin(['test', 'val'])]
    pred_df = pred_df.drop(columns=['split'])
    return pred_df


# ---------- Systema head ----------

def predict_systema(panel_master, seed: int,
                    cfg: Optional[HeadConfig] = None) -> np.ndarray:
    """Predict full log1p post-expression for the Systema benchmark.

    Reproduces systema_r1.res_wr20_wk70_wb10.predict verbatim when
    cfg = PRIOR_SYSTEMA.

    Args:
      panel_master: PanelMaster object with .all_perts / .train_perts /
                    .test_perts / .Y_train_post / .donor_id attributes.
      seed:         panel seed (used upstream for panel_master construction;
                    predict itself is deterministic).
      cfg:          override HeadConfig; default HEAD_CONFIGS['systema'] = PRIOR_SYSTEMA.

    Returns:
      Y_out: (n_test_perts, n_genes) predicted post-expression for panel_master.test_perts.
    """
    if cfg is None:
        cfg = HEAD_CONFIGS['systema']

    from .systema_adapter import build_systema_features

    all_perts = list(panel_master.all_perts)
    pert2idx = {p: i for i, p in enumerate(all_perts)}
    tr = np.array([pert2idx[p] for p in panel_master.train_perts])
    te = np.array([pert2idx[p] for p in panel_master.test_perts])

    donor_id = getattr(panel_master, 'donor_id', 'K562')

    E, offsets, stats = build_systema_features(all_perts, donor_id)
    Y_train_post = np.asarray(panel_master.Y_train_post, dtype=np.float32)

    # For head_predict: E is the 8604-D matrix, train_idx = tr, test_idx = te.
    # test_idx=te ensures Ridge/kNN/binary arms are computed only on test rows
    # so the z-blend matches the champion `res_wr20_wk70_wb10` semantics.
    Y_test = head_predict(E, tr, Y_train_post, cfg, test_idx=te)
    return Y_test
