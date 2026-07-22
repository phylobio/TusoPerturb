"""Unified predictor -- one code path, per-head HeadConfig.

Reduces to plain Ridge(alpha, random_state=1) when weights = (1.0, 0.0, 0.0)
+ scaler='standard' + ridge_estimator='ridge_fixed' + y_standardize=False, so
`head_predict(cfg=PRIOR_A_v2_193)` is byte-identical to A_v2_193.

Falls through to the 3-arm z-blend when weights are mixed, matching R17_C
(hit head, raw target) and res_wr20_wk70_wb10 (systema head, residual target).
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler, normalize

from .heads import HeadConfig, FEATURE_BLOCKS


def _select_features(E: np.ndarray, cfg: HeadConfig) -> np.ndarray:
    """Slice/mask the shared 11680-D feature matrix per head config.

    NOTE: `systema_native_8604` is handled OUTSIDE this function --
    the Systema adapter builds its own 8604-D matrix from
    systema_features.py and passes it directly (E already the right shape).
    """
    if cfg.feature_subset == 'full_11680':
        return E
    elif cfg.feature_subset == 'no_genept_8608':
        return E[:, 3072:11680]
    elif cfg.feature_subset == 'no_genept_no_depmap_8603':
        # From shared 11680: drop genept [0:3072) AND depmap [11671:11676).
        # Result = 11680 - 3072 - 5 = 8603 columns.
        mask = np.ones(E.shape[1], dtype=bool)
        mask[0:3072] = False
        mask[11671:11676] = False
        return E[:, mask]
    elif cfg.feature_subset == 'systema_native_8604':
        # Systema adapter passes its own 8604-D matrix directly.
        return E
    else:
        raise ValueError(f"Unknown feature_subset: {cfg.feature_subset}")


def _apply_scaler(X_train: np.ndarray, X_all: np.ndarray, cfg: HeadConfig
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit scaler on X_train, apply to both X_train and X_all."""
    if cfg.scaler == 'robust_25_75':
        sc = RobustScaler(quantile_range=(25, 75)).fit(X_train)
        return sc.transform(X_train).astype(np.float32), sc.transform(X_all).astype(np.float32)
    elif cfg.scaler == 'standard':
        sc = StandardScaler()
        sc.fit(X_train)
        # Match A_v2_193 handling of zero-variance columns.
        scale = np.where(sc.scale_ > 0, sc.scale_, 1.0)
        Xtr = ((X_train - sc.mean_) / scale).astype(np.float32)
        Xal = ((X_all - sc.mean_) / scale).astype(np.float32)
        return Xtr, Xal
    elif cfg.scaler == 'none':
        return X_train.astype(np.float32), X_all.astype(np.float32)
    else:
        raise ValueError(f"Unknown scaler: {cfg.scaler}")


def _feature_subset_offsets(cfg: HeadConfig) -> Dict[str, Tuple[int, int]]:
    """Return the block offsets valid AFTER feature subsetting."""
    if cfg.feature_subset == 'full_11680':
        return FEATURE_BLOCKS.copy()
    elif cfg.feature_subset == 'no_genept_8608':
        return {name: (s - 3072, e - 3072)
                for name, (s, e) in FEATURE_BLOCKS.items() if name != 'genept'}
    elif cfg.feature_subset == 'no_genept_no_depmap_8603':
        names = ['reactome', 'go_bp', 'hallmark', 'progeny', 'collectri', 'string', 'baseline']
        out = {}; cursor = 0
        for n in names:
            s, e = FEATURE_BLOCKS[n]
            width = e - s
            out[n] = (cursor, cursor + width); cursor += width
        return out
    elif cfg.feature_subset == 'systema_native_8604':
        # Native systema layout, offsets provided externally if needed.
        return {}
    else:
        raise ValueError(f"Unknown feature_subset: {cfg.feature_subset}")


def _apply_block_weights(X: np.ndarray, cfg: HeadConfig,
                          offsets: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Multiply feature blocks by cfg.block_weights (post-scaling)."""
    if not cfg.block_weights:
        return X
    Xw = X.copy()
    for block_name, (start, end) in offsets.items():
        if block_name in cfg.block_weights:
            Xw[:, start:end] *= cfg.block_weights[block_name]
    return Xw


def _zs(M: np.ndarray) -> np.ndarray:
    m = M.mean(axis=0, keepdims=True)
    s = M.std(axis=0, keepdims=True) + 1e-8
    return (M - m) / s


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


def _predict_ridge(r, sy, X_all):
    Y_pred = r.predict(X_all)
    if sy is not None:
        Y_pred = sy.inverse_transform(Y_pred)
    return Y_pred


def _fit_knn_and_predict(X_train, X_all, Y_train, cfg: HeadConfig):
    """kNN cosine softmax weighted predict."""
    if cfg.knn_metric == 'cosine':
        Xtr_n = normalize(X_train, axis=1)
        Xal_n = normalize(X_all, axis=1)
    else:
        Xtr_n, Xal_n = X_train, X_all
    K_eff = min(cfg.knn_K, len(X_train))
    nn = NearestNeighbors(n_neighbors=K_eff, metric=cfg.knn_metric).fit(Xtr_n)
    d, idx = nn.kneighbors(Xal_n)
    w = np.exp(-cfg.knn_tau * d)
    w /= (w.sum(axis=1, keepdims=True) + 1e-12)
    return np.einsum("nk,nkp->np", w, Y_train[idx])


def _fit_binary_ridge(X_train, X_all, Y_train, cfg: HeadConfig):
    """Ridge on top-P%-binarized Y. Signed or unsigned per cfg."""
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
    return r.predict(X_all)


def head_predict(E_full: np.ndarray,
                 train_idx: np.ndarray,
                 Y_train: np.ndarray,
                 cfg: HeadConfig,
                 test_idx: np.ndarray = None) -> np.ndarray:
    """Predict Y per head config.

    Two output modes controlled by `test_idx`:

    - `test_idx is None` (default, regression heads): fits on `E[train_idx]`,
      predicts on ALL rows of E, returns `(n_all, n_targets)`. Blending
      z-scoring (if enabled) is computed across all rows. Callers that only
      want test rows slice `Y_out[test_idx]` themselves.

    - `test_idx is not None` (Systema head): fits on `E[train_idx]`, predicts
      only on `E[test_idx]`, returns `(n_test, n_targets)`. Blending
      z-scoring is computed across test rows only, matching the champion
      `res_wr20_wk70_wb10.predict` semantics.

    Args:
      E_full: (n_perts, D) feature matrix. D depends on cfg.feature_subset --
        for `systema_native_8604`, caller passes the 8604-D matrix directly;
        for `full_11680` variants, caller passes the shared 11680-D matrix.
      train_idx: indices of training perts within E_full.
      Y_train:   (n_train, n_targets) training targets.
                  For regression heads: delta from control.
                  For hit head:        AUCell phenotype scores.
                  For Systema:         raw log1p post-expression Y_train_post.
      cfg:       HeadConfig with all parameters.
      test_idx:  optional test-row indices. If None, predict on all rows.

    Returns:
      Y_out: (n_all, n_targets) or (n_test, n_targets) depending on test_idx.
             For target_shape='residual_pert_mean', pert_mean is added back
             after per-column rescaling.
    """
    # 1. Feature selection
    E = _select_features(E_full, cfg)
    # 2. Scaling (fit on train rows only)
    Xtr, Xal = _apply_scaler(E[train_idx], E, cfg)
    # 3. Block weights (post-scaling)
    offsets = _feature_subset_offsets(cfg)
    Xtr = _apply_block_weights(Xtr, cfg, offsets)
    Xal = _apply_block_weights(Xal, cfg, offsets)

    # 3b. If test_idx is provided, restrict prediction rows to just those.
    #     Ridge/kNN/binary arms all fit on Xtr only; only the "predict" side
    #     changes.
    if test_idx is not None:
        Xpred = Xal[test_idx]
    else:
        Xpred = Xal

    # 4. Target shaping
    if cfg.target_shape == 'raw_delta':
        Y_target = Y_train.astype(np.float32)
        pert_mean = np.zeros(Y_train.shape[1], dtype=np.float32)
    elif cfg.target_shape == 'residual_pert_mean':
        pert_mean = Y_train.mean(axis=0).astype(np.float32)
        Y_target = (Y_train - pert_mean[None, :]).astype(np.float32)
    else:
        raise ValueError(f"Unknown target_shape: {cfg.target_shape}")

    wr, wk, wb = cfg.weights

    # 5. Fit + predict per arm (skip disabled arms).
    Y_r = _predict_ridge(*_fit_ridge(Xtr, Y_target, cfg), Xpred) if wr > 0 else None
    Y_k = _fit_knn_and_predict(Xtr, Xpred, Y_target, cfg) if wk > 0 else None
    Y_b = _fit_binary_ridge(Xtr, Xpred, Y_target, cfg) if wb > 0 else None

    # 6. Blend
    if wr > 0 and wk == 0 and wb == 0:
        # Plain-Ridge fast path (A_v2_193-style: no z-scoring, no blend).
        Y_pred = Y_r
    else:
        parts = []
        if Y_r is not None:
            parts.append(wr * (_zs(Y_r) if cfg.y_z_score_arms else Y_r))
        if Y_k is not None:
            parts.append(wk * (_zs(Y_k) if cfg.y_z_score_arms else Y_k))
        if Y_b is not None:
            parts.append(wb * (_zs(Y_b) if cfg.y_z_score_arms else Y_b))
        Y_pred = sum(parts)

    # 7. Undo target shaping
    if cfg.target_shape == 'residual_pert_mean':
        if cfg.y_z_score_arms:
            # `_rescale_to_train`: match Y_pred's std/mean to Y_target's per column.
            per_col_std_tr = Y_target.std(axis=0, keepdims=True) + 1e-8
            per_col_std_pr = Y_pred.std(axis=0, keepdims=True) + 1e-8
            Y_pred = Y_pred * (per_col_std_tr / per_col_std_pr)
            per_col_mean_tr = Y_target.mean(axis=0, keepdims=True)
            Y_pred = Y_pred - Y_pred.mean(axis=0, keepdims=True) + per_col_mean_tr
        Y_out = Y_pred + pert_mean[None, :]
    else:
        Y_out = Y_pred

    return Y_out.astype(np.float32)
