"""
Vendored dependencies used by TusoPerturb.

Files in this package are byte-identical copies of the corresponding modules
from the method development tree, with only the reference-data location logic
rewritten to prefer the vendored refs shipped inside
``TusoPerturb/data/embeddings/`` (see ``orth_features.REF_DIR`` /
``orth_features_v2.REF_DIR``).

Modules
-------
orth_features       (sprint 004) — STRING v12 / Reactome / GO BP loaders
orth_features_v2    (sprint 005) — adds Hallmark / CollecTRI / PROGENy on top

The public API used by the TusoPerturb pipeline is
``orth_features_v2.load_orth_features_v2``.
"""

from . import orth_features
from . import orth_features_v2

__all__ = ["orth_features", "orth_features_v2"]
