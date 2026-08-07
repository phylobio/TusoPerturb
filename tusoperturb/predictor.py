"""The predictor: one code path, one `HeadConfig` per head.

`head_predict` is the whole model. Seven steps:

  1. scale     RobustScaler(25, 75) fitted on the training rows only
  2. weight    the trailing `EMB_COLS` co-essentiality columns are multiplied
               by `cfg.emb_weight` after scaling
  3. shape     target -> raw, or residual after removing the per-gene mean of
               the training responses
  4. ridge     fixed-alpha Ridge, or RidgeCV over `cfg.ridge_alphas`
  5. kNN       cosine neighbours with supervised feature weighting,
               deterministic tie expansion and a per-row adaptive bandwidth
  6. binary    RidgeCV on the top-`top_p`% binarised target
  7. blend     fixed weights, optional per-arm z-scoring, then the target
               shaping is undone with the `shrink` amplitude factor

Nothing in this file is fitted on anything but `train_idx` rows.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler, normalize

from .heads import EMB_COLS, HeadConfig


# --------------------------------------------------------------------------
# feature path
# --------------------------------------------------------------------------
def _apply_scaler(X_train: np.ndarray, X_all: np.ndarray, cfg: HeadConfig
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the scaler on X_train, apply it to both X_train and X_all."""
    if cfg.scaler == 'robust_25_75':
        sc = RobustScaler(quantile_range=(25, 75)).fit(X_train)
        return (sc.transform(X_train).astype(np.float32),
                sc.transform(X_all).astype(np.float32))
    elif cfg.scaler == 'standard':
        sc = StandardScaler().fit(X_train)
        scale = np.where(sc.scale_ > 0, sc.scale_, 1.0)
        return (((X_train - sc.mean_) / scale).astype(np.float32),
                ((X_all - sc.mean_) / scale).astype(np.float32))
    elif cfg.scaler == 'none':
        return X_train.astype(np.float32), X_all.astype(np.float32)
    raise ValueError(f"Unknown scaler: {cfg.scaler}")


def _apply_emb_weight(X: np.ndarray, cfg: HeadConfig) -> np.ndarray:
    """Scale the trailing co-essentiality columns (post-scaling amplitude)."""
    w = float(cfg.emb_weight)
    if EMB_COLS <= 0 or w == 1.0:
        return X
    return np.concatenate([X[:, :-EMB_COLS], X[:, -EMB_COLS:] * np.float32(w)], axis=1)


def _zs(M: np.ndarray) -> np.ndarray:
    m = M.mean(axis=0, keepdims=True)
    s = M.std(axis=0, keepdims=True) + 1e-8
    return (M - m) / s


# --------------------------------------------------------------------------
# ridge arm
# --------------------------------------------------------------------------
def _fit_ridge(X_train: np.ndarray, Y_train: np.ndarray, cfg: HeadConfig):
    """Ridge (fixed alpha or RidgeCV) with optional Y standardization."""
    if cfg.y_standardize:
        sy = StandardScaler().fit(Y_train)
        Y_target = sy.transform(Y_train)
    else:
        sy = None
        Y_target = Y_train

    if cfg.ridge_estimator == 'ridge_fixed':
        r = Ridge(alpha=cfg.ridge_fixed_alpha, random_state=1, fit_intercept=True)
    elif cfg.ridge_estimator == 'ridgecv':
        r = RidgeCV(alphas=list(cfg.ridge_alphas))
    else:
        raise ValueError(f"Unknown ridge_estimator: {cfg.ridge_estimator}")

    r.fit(X_train, Y_target)
    return r, sy


def _predict_ridge(r, sy, X_all: np.ndarray) -> np.ndarray:
    Y_pred = r.predict(X_all)
    if sy is not None:
        Y_pred = sy.inverse_transform(Y_pred)
    return Y_pred


# --------------------------------------------------------------------------
# kNN arm
# --------------------------------------------------------------------------
def _feature_weights(X_train: np.ndarray, Y_train: np.ndarray,
                     cfg: HeadConfig) -> Optional[np.ndarray]:
    """Supervised column weights for the kNN metric, fitted on train targets only.

    Most of the 8700 columns carry no information about the response. Weighting
    each column by how strongly it covaries with the target block makes
    neighbour selection target-aware. The target block is first reduced to
    `knn_feat_rank` SVD components, so the correlation is one small matmul
    rather than a full (n_features x n_genes) product.
    """
    if cfg.knn_feat_weight == 'none':
        return None
    if cfg.knn_feat_weight != 'target_corr':
        raise ValueError(f"Unknown knn_feat_weight: {cfg.knn_feat_weight!r}")
    if min(Y_train.shape) < 2:
        return None
    k = int(min(cfg.knn_feat_rank, min(Y_train.shape) - 1))
    sv = TruncatedSVD(n_components=k, random_state=0).fit(Y_train)
    Yr = Y_train @ sv.components_.T
    Xc = X_train - X_train.mean(axis=0, keepdims=True)
    Yc = Yr - Yr.mean(axis=0, keepdims=True)
    xs = np.sqrt((Xc ** 2).sum(axis=0)) + 1e-12
    ys = np.sqrt((Yc ** 2).sum(axis=0)) + 1e-12
    C = (Xc.T @ Yc) / xs[:, None] / ys[None, :]
    wf = np.sqrt((C ** 2).sum(axis=1))
    wf = wf / (wf.mean() + 1e-12)
    return (wf ** cfg.knn_feat_pow).astype(np.float32)


def _knn_arm(X_train: np.ndarray, X_pred: np.ndarray, Y_train: np.ndarray,
             cfg: HeadConfig) -> np.ndarray:
    """Weighted-average kNN with deterministic ties and adaptive bandwidth."""
    wf = _feature_weights(X_train, Y_train, cfg)
    if wf is not None:
        X_train = X_train * wf[None, :]
        X_pred = X_pred * wf[None, :]

    if cfg.knn_metric == 'cosine':
        Xtr_n = normalize(X_train, axis=1)
        Xpr_n = normalize(X_pred, axis=1)
    else:
        Xtr_n, Xpr_n = X_train, X_pred

    n_train = len(X_train)
    K_eff = min(cfg.knn_K, n_train)
    # 'expand' needs headroom to see rows tied with the K-th neighbour.
    K_query = min(n_train, K_eff + (32 if cfg.knn_tie_break == 'expand' else 0))
    nn = NearestNeighbors(n_neighbors=K_query, metric=cfg.knn_metric).fit(Xtr_n)
    d, idx = nn.kneighbors(Xpr_n)

    d_k = d[:, K_eff - 1][:, None]
    if cfg.knn_tie_break == 'expand':
        # Deterministic neighbour set: keep every train row within the K-th
        # distance, with a float32 epsilon so ties that differ only by SIMD
        # reduction order are kept on every CPU.
        mask = d <= (d_k + 1e-6 * np.maximum(d_k, 1.0))
    elif cfg.knn_tie_break == 'none':
        mask = np.zeros_like(d, dtype=bool)
        mask[:, :K_eff] = True
    else:
        raise ValueError(f"Unknown knn_tie_break: {cfg.knn_tie_break!r}")

    if cfg.knn_adaptive_bw == 'none':
        w = np.exp(-cfg.knn_tau * d)
    elif cfg.knn_adaptive_bw == 'mean':
        # Measure each row's distances in units of its own mean neighbour
        # distance, then renormalise by the median across rows so that tau keeps
        # its global meaning and a typical row is left unchanged.
        cnt = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        scale = np.where(mask, d, 0.0).sum(axis=1, keepdims=True) / cnt
        scale = np.maximum(scale, 1e-8)
        scale = scale / max(float(np.median(scale)), 1e-8)
        w = np.exp(-cfg.knn_tau * (d / scale))
    else:
        raise ValueError(f"Unknown knn_adaptive_bw: {cfg.knn_adaptive_bw!r}")

    w = np.where(mask, w, 0.0)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    return np.einsum("nk,nkp->np", w, Y_train[idx])


# --------------------------------------------------------------------------
# binary arm
# --------------------------------------------------------------------------
def _fit_binary_ridge(X_train, X_pred, Y_train, cfg: HeadConfig) -> np.ndarray:
    """RidgeCV on the top-`top_p`%-binarized target. Signed or unsigned per cfg."""
    Y_bin = np.zeros_like(Y_train)
    p = cfg.top_p / 100.0
    for j in range(Y_train.shape[1]):
        col = Y_train[:, j]
        if cfg.signed_binary:
            thr = np.quantile(np.abs(col), 1 - p)
            Y_bin[:, j] = (np.abs(col) >= thr).astype(np.float32) * np.sign(col)
        else:
            thr = np.quantile(col, 1 - p)
            Y_bin[:, j] = (col >= thr).astype(np.float32)
    r = RidgeCV(alphas=list(cfg.bin_ridge_alphas)).fit(X_train, Y_bin)
    return r.predict(X_pred)


# --------------------------------------------------------------------------
# blend
# --------------------------------------------------------------------------
def _blend(Y_r, Y_k, Y_b, weights, z: bool) -> np.ndarray:
    wr, wk, wb = weights
    if wr > 0 and wk == 0 and wb == 0:
        return Y_r
    parts = []
    for w, Y in ((wr, Y_r), (wk, Y_k), (wb, Y_b)):
        if Y is not None:
            parts.append(w * (_zs(Y) if z else Y))
    return sum(parts)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def head_predict(E_full: np.ndarray,
                 train_idx: np.ndarray,
                 Y_train: np.ndarray,
                 cfg: HeadConfig,
                 test_idx: np.ndarray = None) -> np.ndarray:
    """Predict the response for every row of `E_full` (or just `test_idx`).

    Args:
      E_full:    (n_perts, 8700) shared feature matrix, from
                 `tusoperturb.feature_builder.build_features`.
      train_idx: indices of the training perturbations within `E_full`.
      Y_train:   (n_train, n_targets) training responses, in `train_idx` order.
      cfg:       `SHARED_HEAD` or `HIT_HEAD`.
      test_idx:  optional row indices to predict. If None, predict every row.

    Two output modes, both fitted on `train_idx` only:

    - `test_idx is None`: returns `(n_perts, n_targets)`; callers that only
      want the scored rows slice the result themselves.
    - `test_idx is not None`: returns `(len(test_idx), n_targets)`. Any pooling
      across rows -- per-arm z-scoring, and the adaptive kNN bandwidth's median
      normalisation -- is then computed over the predicted rows only. The
      Systema path uses this mode.
    """
    # 1-2. scale on train rows, then weight the co-essentiality columns.
    Xtr, Xal = _apply_scaler(E_full[train_idx], E_full, cfg)
    Xtr = _apply_emb_weight(Xtr, cfg)
    Xal = _apply_emb_weight(Xal, cfg)
    Xpred = Xal[test_idx] if test_idx is not None else Xal

    # 3. target shaping
    if cfg.target_shape == 'raw_delta':
        Y_target = Y_train.astype(np.float32)
        pert_mean = np.zeros(Y_train.shape[1], dtype=np.float32)
    elif cfg.target_shape == 'residual_pert_mean':
        pert_mean = Y_train.mean(axis=0).astype(np.float32)
        Y_target = (Y_train - pert_mean[None, :]).astype(np.float32)
    else:
        raise ValueError(f"Unknown target_shape: {cfg.target_shape}")

    # 4-6. arms (skip any arm with zero blend weight)
    wr, wk, wb = cfg.weights
    Y_r = _predict_ridge(*_fit_ridge(Xtr, Y_target, cfg), Xpred) if wr > 0 else None
    Y_k = _knn_arm(Xtr, Xpred, Y_target, cfg) if wk > 0 else None
    Y_b = _fit_binary_ridge(Xtr, Xpred, Y_target, cfg) if wb > 0 else None
    Y_pred = _blend(Y_r, Y_k, Y_b, (wr, wk, wb), cfg.y_z_score_arms)

    # 7. undo the target shaping, applying the amplitude factor
    lam = float(cfg.shrink)
    if cfg.target_shape == 'residual_pert_mean':
        if cfg.y_z_score_arms:
            # Per-arm z-scoring destroys the scale, so the prediction is put back
            # on the training residual scale per column. At correlation r the
            # MSE-optimal amplitude is r * std_tr, which is what `lam` sets.
            per_col_std_tr = (lam * Y_target.std(axis=0, keepdims=True)) + 1e-8
            per_col_std_pr = Y_pred.std(axis=0, keepdims=True) + 1e-8
            Y_pred = Y_pred * (per_col_std_tr / per_col_std_pr)
            per_col_mean_tr = Y_target.mean(axis=0, keepdims=True)
            Y_pred = Y_pred - Y_pred.mean(axis=0, keepdims=True) + per_col_mean_tr
        elif lam != 1.0:
            Y_pred = Y_pred * lam
        Y_out = Y_pred + pert_mean[None, :]
    else:
        Y_out = Y_pred if lam == 1.0 else Y_pred * lam
    return Y_out
