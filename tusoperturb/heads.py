"""Per-benchmark head configuration for TusoPerturb.

Each of the 5 benchmark heads is one row in `HEAD_CONFIGS`. The configs are
frozen at the prior per-benchmark champions:

  cellsim / scperturb / perturbhd_reg -> PRIOR_A_v2_193
    (Ridge alpha=110 on StandardScaler + block_weights, 11680-D features)
  perturbhd_hit -> PRIOR_R17_C
    (3-arm blend WR=0.10/WK=0.65/WB=0.25, drop-GenePT 8608-D, RobustScaler,
     unsigned top-2% binary)
  systema -> PRIOR_SYSTEMA (res_wr20_wk70_wb10)
    (3-arm blend WR=0.20/WK=0.70/WB=0.10, drop-GenePT-and-DepMap 8599-D or the
     Systema-specific 8604-D block stack, RobustScaler, signed top-2% binary,
     residual-over-pert-mean target)

FEATURE_BLOCKS documents the 11680-D shared feature layout used by the
regression + hit heads. The Systema head uses a separate 8604-D block stack
built by `tusoperturb.systema_adapter` from the systema_features.py builder.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


# The 11680-D shared feature layout used by the CellSim / scPerturB /
# PerturbHD (both reg + hit) heads. Column offsets are (start, end).
FEATURE_BLOCKS: Dict[str, Tuple[int, int]] = {
    'genept':    (0, 3072),
    'reactome':  (3072, 4888),
    'go_bp':     (4888, 10294),
    'hallmark':  (10294, 10344),
    'progeny':   (10344, 10358),
    'collectri': (10358, 11543),
    'string':    (11543, 11671),
    'depmap':    (11671, 11676),
    'baseline':  (11676, 11680),
}
FEATURE_BLOCK_NAMES = list(FEATURE_BLOCKS.keys())


# Symbolic feature-subset labels for head configs.
FEATURE_SUBSETS = (
    'full_11680',                     # Regression heads (CellSim, scPerturB, PerturbHD-reg)
    'no_genept_8608',                 # PerturbHD-hit (drop GenePT)
    'no_genept_no_depmap_8603',       # Systema when built from the shared 11680-D stack
                                      # (drop GenePT and DepMap, no baseline)
    'systema_native_8604',            # Systema when built from systema_features.py
                                      # (reactome+go_bp+hallmark+progeny+collectri+string+depmap)
)


@dataclass
class HeadConfig:
    """Per-head parameters. Reduces to plain-Ridge if weights = (1.0, 0.0, 0.0)."""

    # Feature selection
    feature_subset: str = 'full_11680'
    scaler: str = 'robust_25_75'
    block_weights: Dict[str, float] = field(default_factory=dict)

    # Ridge continuous arm
    ridge_estimator: str = 'ridgecv'
    ridge_alphas: Tuple[float, ...] = (3.0, 10.0, 30.0, 100.0)
    ridge_fixed_alpha: float = 110.0
    y_standardize: bool = True

    # kNN arm
    knn_K: int = 13
    knn_tau: float = 11.0
    knn_metric: str = 'cosine'

    # Binary arm
    top_p: float = 2.0
    bin_ridge_alphas: Tuple[float, ...] = (20.0, 60.0, 180.0)
    signed_binary: bool = True

    # Blend
    weights: Tuple[float, float, float] = (0.20, 0.70, 0.10)  # (WR, WK, WB)
    y_z_score_arms: bool = True

    # Target shaping
    target_shape: str = 'residual_pert_mean'  # 'raw_delta' or 'residual_pert_mean'

    def __post_init__(self):
        wr, wk, wb = self.weights
        total = wr + wk + wb
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Blend weights must sum to 1.0, got {total}: ({wr},{wk},{wb})")


# ---- Prior per-benchmark champion configs (verbatim, frozen) ----

# A_v2_193_STR15_Dep100_a110 (regression heads):
#   StandardScaler + block weights {go_bp: 0.67, string: 1.5, depmap: 10.0}
#   Ridge(alpha=110, random_state=1) -- NOT RidgeCV
#   Trained on RAW Y_train (which is delta_mean from stage loader)
#   No blend: weights = (1.0, 0.0, 0.0)
PRIOR_A_v2_193 = HeadConfig(
    feature_subset='full_11680',
    scaler='standard',
    block_weights={'go_bp': 0.67, 'string': 1.5, 'depmap': 10.0},
    ridge_estimator='ridge_fixed',
    ridge_fixed_alpha=110.0,
    y_standardize=False,
    weights=(1.0, 0.0, 0.0),
    target_shape='raw_delta',
)

# R17_C_tau11_wr10_wk65_wb25 (PerturbHD-hit):
#   RobustScaler on X (drop GenePT: 8608-D)
#   3-arm blend WR=0.10 / WK=0.65 / WB=0.25 with per-arm z-scoring
#   K=13, TAU=11.0
#   Top-2%, unsigned binary
PRIOR_R17_C = HeadConfig(
    feature_subset='no_genept_8608',
    scaler='robust_25_75',
    block_weights={},
    ridge_estimator='ridgecv',
    ridge_alphas=(3.0, 10.0, 30.0, 100.0),
    y_standardize=True,
    knn_K=13,
    knn_tau=11.0,
    top_p=2.0,
    bin_ridge_alphas=(20.0, 60.0, 180.0),
    signed_binary=False,
    weights=(0.10, 0.65, 0.25),
    target_shape='raw_delta',
    y_z_score_arms=True,
)

# res_wr20_wk70_wb10 (Systema): built from systema_features.py 8604-D stack.
PRIOR_SYSTEMA = HeadConfig(
    feature_subset='systema_native_8604',
    scaler='robust_25_75',
    block_weights={},
    ridge_estimator='ridgecv',
    ridge_alphas=(3.0, 10.0, 30.0, 100.0),
    y_standardize=True,
    knn_K=13,
    knn_tau=11.0,
    top_p=2.0,
    bin_ridge_alphas=(20.0, 60.0, 180.0),
    signed_binary=True,
    weights=(0.20, 0.70, 0.10),
    target_shape='residual_pert_mean',
    y_z_score_arms=True,
)


# The 5-head unified config -- one row per benchmark.
HEAD_CONFIGS = {
    'cellsim':       PRIOR_A_v2_193,
    'scperturb':     PRIOR_A_v2_193,
    'perturbhd_reg': PRIOR_A_v2_193,
    'perturbhd_hit': PRIOR_R17_C,
    'systema':       PRIOR_SYSTEMA,
}
