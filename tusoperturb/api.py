"""TusoPerturb public API: 4 predict_* functions.

Each function wraps `head_predict` with the correct HeadConfig and IO shaping.
Signatures are unchanged from TusoPerturb v1, so an existing harness keeps
working; what changed is the feature space and the head table underneath (see
ARCHITECTURE.md).

Four of the five public keys resolve to the same `SHARED_HEAD`; only
`perturbhd_hit` uses a second config.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from .heads import HEAD_CONFIGS, HeadConfig
from .predictor import head_predict
from . import feature_builder as fb
from .feature_builder import build_features, build_shared_features, DS_TO_PAPER_KEY


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

    Args:
      master:  reference master AnnData (used for train/test split alignment
               via build_pred_adata_from_matrix; not touched by head_predict).
      dataset: one of 'nadig25hepg2', 'nadig25jurkat', 'replogle22k562', 'replogle22rpe1'.
      seed:    ignored by the regression path (the head is deterministic and
               does not depend on split-{seed}); kept in the signature for
               uniformity with the hit head.
      head:    which HeadConfig row to use. 'cellsim', 'scperturb' and
               'perturbhd_reg' are the same object (SHARED_HEAD).
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

    Args:
      master:  reference master AnnData.
      dataset: one of 'nadig25hepg2', 'nadig25jurkat', 'replogle22k562', 'replogle22rpe1'.
      seed:    which split-{seed} column in the AUCell pivot to use as train/test.
      cfg:     override HeadConfig; default HEAD_CONFIGS['perturbhd_hit'] = HIT_HEAD.

    Returns:
      DataFrame with columns ['pert', 'pheno', 'hit_score'] restricted to
      test + val perts.
    """
    if cfg is None:
        cfg = HEAD_CONFIGS['perturbhd_hit']

    # Load pheno pivot (targets).
    paper_key = DS_TO_PAPER_KEY[dataset]
    df = pd.read_parquet(fb.mean_effect_path(paper_key))
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

    Args:
      panel_master: PanelMaster object with .all_perts / .train_perts /
                    .test_perts / .Y_train_post / .donor_id attributes.
      seed:         panel seed (used upstream for panel_master construction;
                    predict itself is deterministic).
      cfg:          override HeadConfig; default HEAD_CONFIGS['systema'] = SHARED_HEAD.

    Returns:
      Y_test: (n_test_perts, n_genes) predicted post-expression for
      panel_master.test_perts.
    """
    if cfg is None:
        cfg = HEAD_CONFIGS['systema']

    all_perts = list(panel_master.all_perts)
    pert2idx = {p: i for i, p in enumerate(all_perts)}
    tr = np.array([pert2idx[p] for p in panel_master.train_perts])
    te = np.array([pert2idx[p] for p in panel_master.test_perts])

    donor_id = getattr(panel_master, 'donor_id', 'K562')

    E, _, _ = build_features(all_perts, donor_id)
    Y_train_post = np.asarray(panel_master.Y_train_post, dtype=np.float32)

    # test_idx=te restricts the ridge / kNN / binary arms to the test rows, so
    # the adaptive kNN bandwidth is the median over the test batch. This is the
    # batching the sealed Systema run used; the other four benchmarks predict
    # every row in one batch (test_idx=None).
    Y_test = head_predict(E, tr, Y_train_post, cfg, test_idx=te)
    return Y_test
