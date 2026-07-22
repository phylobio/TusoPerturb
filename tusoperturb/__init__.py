"""TusoPerturb: unified perturbation-response predictor.

Harmonizes two prior best-in-class codebases into one package:
  - `tusoskill/systema_r1` best on the Systema benchmark (Vinas et al. 2025).
  - `final_methodv3` best on CellSimBench, scPerturBench, and PerturbHD.

The unification is a single Python package with a single public API. Each of
the 5 benchmark heads runs the same feature builder + the same 3-arm blend
primitive, but each head selects the feature subset, scaler, ridge estimator,
target framing, and blend weights that were the prior per-benchmark champion.

Public API::

    from tusoperturb import (
        predict_regression,     # CellSim / scPerturB / PerturbHD-reg (Y_delta)
        predict_hit,            # PerturbHD-hit (AUCell phenotype scores)
        predict_systema,        # Systema (post-expression with residual framing)
        HEAD_CONFIGS,           # 5-entry dict of HeadConfig per benchmark
    )

The per-head champion configs are frozen in `HEAD_CONFIGS` and mirror the
prior per-benchmark sprints:

  cellsim / scperturb / perturbhd_reg -> PRIOR_A_v2_193
  perturbhd_hit                        -> PRIOR_R17_C
  systema                              -> PRIOR_SYSTEMA (res_wr20_wk70_wb10)
"""
from .heads import (
    HeadConfig,
    HEAD_CONFIGS,
    PRIOR_A_v2_193,
    PRIOR_R17_C,
    PRIOR_SYSTEMA,
    FEATURE_BLOCKS,
)
from .predictor import head_predict
from .api import (
    predict_regression,
    predict_hit,
    predict_systema,
)

__all__ = [
    'HeadConfig', 'HEAD_CONFIGS', 'PRIOR_A_v2_193', 'PRIOR_R17_C', 'PRIOR_SYSTEMA',
    'FEATURE_BLOCKS', 'head_predict',
    'predict_regression', 'predict_hit', 'predict_systema',
]
__version__ = "1.0.0"
