"""TusoPerturb: unified perturbation-response predictor.

One feature space and two heads cover all five benchmarks. Every perturbation
is described by an 8700-D vector -- seven annotation blocks (Reactome, GO BP,
Hallmark, PROGENy, CollecTRI, STRING, DepMap essentiality; 8604 columns) plus
two truncated-SVD co-essentiality blocks built from the DepMap CRISPR
gene-effect matrix (96 columns) -- and every head runs the same three-arm
blend (ridge + weighted kNN + binary ridge) over it.

Four of the five public keys resolve to a single `SHARED_HEAD`; only
`perturbhd_hit` uses a second config, and it differs in head parameters only,
never in the feature space. `heads.assert_two_head()` is the executable form
of that claim.

Public API::

    from tusoperturb import (
        predict_regression,       # CellSim / scPerturB (Y_delta -> pred AnnData)
        predict_regression_gene,  # PerturbHD-reg (pert/gene/effect frame)
        predict_hit,              # PerturbHD-hit (AUCell phenotype scores)
        predict_systema,          # Systema (log1p post-expression)
        HEAD_CONFIGS,             # 5-key dict -> 2 distinct HeadConfig objects
    )

Head map::

    cellsim / scperturb / perturbhd_reg / systema -> SHARED_HEAD
    perturbhd_hit                                 -> HIT_HEAD
"""
from .heads import (
    HeadConfig,
    HEAD_CONFIGS,
    SHARED_HEAD,
    HIT_HEAD,
    FEATURE_BLOCKS,
    WIDTH,
    assert_two_head,
    head_deviation,
)
from .predictor import head_predict
from .feature_builder import build_features, build_shared_features
from .api import (
    predict_regression,
    predict_regression_gene,
    predict_hit,
    predict_systema,
)

__all__ = [
    'HeadConfig', 'HEAD_CONFIGS', 'SHARED_HEAD', 'HIT_HEAD',
    'FEATURE_BLOCKS', 'WIDTH', 'assert_two_head', 'head_deviation',
    'head_predict', 'build_features', 'build_shared_features',
    'predict_regression', 'predict_regression_gene',
    'predict_hit', 'predict_systema',
]
__version__ = "2.0.0"
