"""Head configuration for TusoPerturb.

TusoPerturb v2 is a **two-head** method. Both heads read the same 8700-D
feature matrix; they differ only in head parameters:

  slot            head          target
  ---------------------------------------------------------------------
  cellsim         SHARED_HEAD   delta from control
  scperturb       SHARED_HEAD   delta from control
  perturbhd_reg   SHARED_HEAD   gene-level mean_diff
  systema         SHARED_HEAD   log1p post-expression
  perturbhd_hit   HIT_HEAD      AUCell phenotype scores

`FEATURE_BLOCKS` documents the shared 8700-D layout: the seven annotation
blocks (8604 columns, built by `tusoperturb.systema_adapter`) followed by the
two co-essentiality blocks (96 columns, built by
`tusoperturb.coessentiality`). Every head sees every column; there is no
feature subsetting.

The two heads are separate objects on purpose. `assert_two_head()` is the
acceptance test: the four shared slots must resolve to a field-identical
config, and the hit slot may differ only in head parameters -- never in the
feature space.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, Tuple


# Column bounds of the shared 8700-D feature space. (start, end), end-exclusive.
FEATURE_BLOCKS: Dict[str, Tuple[int, int]] = {
    'reactome':  (0, 1816),
    'go_bp':     (1816, 7222),
    'hallmark':  (7222, 7272),
    'progeny':   (7272, 7286),
    'collectri': (7286, 8471),
    'string':    (8471, 8599),
    'depmap':    (8599, 8604),
    'coess64':   (8604, 8668),
    'coess32':   (8668, 8700),
}
FEATURE_BLOCK_NAMES = list(FEATURE_BLOCKS.keys())

# Width of the annotation stack and of the trailing co-essentiality block.
ANNOT_WIDTH = 8604
EMB_COLS = 96
WIDTH = ANNOT_WIDTH + EMB_COLS  # 8700

SHARED_SLOTS = ('cellsim', 'scperturb', 'perturbhd_reg', 'systema')
HIT_SLOT = 'perturbhd_hit'


@dataclass(frozen=True)
class HeadConfig:
    """Every parameter of one head, as an explicit value.

    No parameter here is a selection rule: the ridge penalty of the shared head
    is written `alpha=10` rather than a grid, because that is the number that
    ran.
    """

    # Feature scaling. The co-essentiality columns are multiplied by
    # `emb_weight` AFTER scaling, so the weight is a pure amplitude knob and
    # cannot be absorbed by the (column-equivariant) scaler.
    scaler: str = 'robust_25_75'          # 'robust_25_75' | 'standard' | 'none'
    emb_weight: float = 1.15

    # Ridge (continuous) arm
    ridge_estimator: str = 'ridge_fixed'  # 'ridge_fixed' | 'ridgecv'
    ridge_fixed_alpha: float = 10.0
    ridge_alphas: Tuple[float, ...] = (3.0, 10.0, 30.0, 100.0)
    y_standardize: bool = True

    # kNN arm
    knn_K: int = 80
    knn_tau: float = 11.0
    knn_metric: str = 'cosine'
    knn_tie_break: str = 'expand'         # 'none' | 'expand'
    knn_adaptive_bw: str = 'mean'         # 'none' | 'mean'
    knn_feat_weight: str = 'target_corr'  # 'none' | 'target_corr'
    knn_feat_pow: float = 6.0
    knn_feat_rank: int = 512

    # Binary arm
    top_p: float = 2.0
    bin_ridge_alphas: Tuple[float, ...] = (20.0, 60.0, 180.0)
    signed_binary: bool = True

    # Blend
    weights: Tuple[float, float, float] = (0.10, 0.85, 0.05)  # (ridge, kNN, binary)
    y_z_score_arms: bool = False

    # Target shaping + amplitude
    target_shape: str = 'residual_pert_mean'  # 'raw_delta' | 'residual_pert_mean'
    shrink: float = 1.25

    def __post_init__(self):
        wr, wk, wb = self.weights
        total = wr + wk + wb
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Blend weights must sum to 1.0, got {total}: ({wr},{wk},{wb})")
        for name, allowed in (('scaler', ('robust_25_75', 'standard', 'none')),
                              ('ridge_estimator', ('ridge_fixed', 'ridgecv')),
                              ('knn_metric', ('cosine', 'euclidean')),
                              ('knn_tie_break', ('none', 'expand')),
                              ('knn_adaptive_bw', ('none', 'mean')),
                              ('knn_feat_weight', ('none', 'target_corr')),
                              ('target_shape', ('raw_delta', 'residual_pert_mean'))):
            v = getattr(self, name)
            if v not in allowed:
                raise ValueError(f"Unknown {name}: {v!r} (allowed: {allowed})")


# ---- The two shipped heads (verbatim, frozen) ----

# Shared head: cellsim, scperturb, perturbhd_reg, systema.
#   kNN-dominant blend on the 8700-D space, target-correlation feature
#   weighting to the 6th power over a rank-512 target basis, deterministic
#   tie expansion, per-row adaptive kernel bandwidth, residual-over-pert-mean
#   target with a 1.25 amplitude factor.
SHARED_HEAD = HeadConfig(
    scaler='robust_25_75',
    emb_weight=1.15,
    ridge_estimator='ridge_fixed',
    ridge_fixed_alpha=10.0,
    ridge_alphas=(3.0, 10.0, 30.0, 100.0),
    y_standardize=True,
    knn_K=80,
    knn_tau=11.0,
    knn_metric='cosine',
    knn_tie_break='expand',
    knn_adaptive_bw='mean',
    knn_feat_weight='target_corr',
    knn_feat_pow=6.0,
    knn_feat_rank=512,
    top_p=2.0,
    bin_ridge_alphas=(20.0, 60.0, 180.0),
    signed_binary=True,
    weights=(0.10, 0.85, 0.05),
    y_z_score_arms=False,
    target_shape='residual_pert_mean',
    shrink=1.25,
)

# Hit head: perturbhd_hit only.
#   Same feature matrix, unweighted kNN metric at K=13, RidgeCV, unsigned
#   top-2% binary arm, per-arm z-scoring, raw target, no shrink, and the
#   co-essentiality block at unit amplitude.
HIT_HEAD = HeadConfig(
    scaler='robust_25_75',
    emb_weight=1.0,
    ridge_estimator='ridgecv',
    ridge_fixed_alpha=110.0,       # inert under ridge_estimator='ridgecv'
    ridge_alphas=(3.0, 10.0, 30.0, 100.0),
    y_standardize=True,
    knn_K=13,
    knn_tau=11.0,
    knn_metric='cosine',
    knn_tie_break='none',
    knn_adaptive_bw='none',
    knn_feat_weight='none',
    knn_feat_pow=1.0,
    knn_feat_rank=128,
    top_p=2.0,
    bin_ridge_alphas=(20.0, 60.0, 180.0),
    signed_binary=False,
    weights=(0.10, 0.65, 0.25),
    y_z_score_arms=True,
    target_shape='raw_delta',
    shrink=1.0,
)


# The 5-slot head map -- one row per benchmark, two distinct configs.
HEAD_CONFIGS = {
    'cellsim':       SHARED_HEAD,
    'scperturb':     SHARED_HEAD,
    'perturbhd_reg': SHARED_HEAD,
    'systema':       SHARED_HEAD,
    'perturbhd_hit': HIT_HEAD,
}


def head_deviation() -> Dict[str, object]:
    """Fields where HIT_HEAD differs from SHARED_HEAD, as {field: hit_value}."""
    s, h = asdict(SHARED_HEAD), asdict(HIT_HEAD)
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in h.items() if s[k] != v}


def assert_two_head(configs: Dict[str, HeadConfig] = None) -> dict:
    """Acceptance test for the two-head architecture.

    Returns a dict of violations; empty means conforming. Checks that the four
    shared slots resolve to one field-identical config and that exactly two
    distinct configs are present.
    """
    cfgs = HEAD_CONFIGS if configs is None else configs
    out: dict = {}

    missing = sorted(set(SHARED_SLOTS + (HIT_SLOT,)) - set(cfgs))
    if missing:
        out['missing_slots'] = missing
        return out

    ref = asdict(cfgs[SHARED_SLOTS[0]])
    diffs: dict = {}
    for b in SHARED_SLOTS[1:]:
        for k, v in asdict(cfgs[b]).items():
            if ref[k] != v:
                diffs.setdefault(k, {})[SHARED_SLOTS[0]] = ref[k]
                diffs[k][b] = v
    if diffs:
        out['shared_head_not_identical'] = diffs

    if asdict(cfgs[HIT_SLOT]) == ref:
        out['hit_head_not_distinct'] = True

    return out
