"""
Extended orthogonal feature loader for Sprint 005.

Adds support for new features on top of Sprint 004's set:
  - hallmark: MSigDB Hallmark 50 pathway multi-hot (50 d)
  - collectri: CollecTRI TF-target multi-hot per pert gene (1185 d)
  - progeny: PROGENy pathway response signatures (14 d)
  - reactome / go_bp / string: legacy from Sprint 004

Also supports a `pert_control_expr` interaction feature (built at candidate time
because it depends on dataset control expression, not a static ref file).

Usage:
    from orth_features_v2 import load_orth_features_v2
    feats, stats = load_orth_features_v2(all_perts, features=('hallmark','collectri','reactome'))
    # feats['hallmark'] -> (n_perts, 50)
    # feats['collectri'] -> (n_perts, 1185)
    # feats['progeny'] -> (n_perts, 14)
"""
import numpy as np
import json
import os
import sys
from pathlib import Path
from scipy import sparse

# Prefer the vendored ref/ shipped with TusoPerturb; fall back to the
# shared-workspace copy used during method development.
_HERE = Path(__file__).resolve().parent
_VENDORED_REF = _HERE.parent.parent / 'data' / 'embeddings' / 'ref'
_LEGACY_REF   = Path('/mnt/shared-workspace/biomni-method/perturb-2026/data/ref')
_ENV_REF      = os.environ.get('TUSOPERTURB_REF_DIR')

if _ENV_REF and Path(_ENV_REF).is_dir():
    REF_DIR = Path(_ENV_REF)
elif _VENDORED_REF.is_dir():
    REF_DIR = _VENDORED_REF
else:
    REF_DIR = _LEGACY_REF

# Load the v1 loader as the base. Prefer the sibling vendored copy so this
# module keeps working without the original sprint_004 tree on disk.
_LEGACY_V1 = '/mnt/shared-workspace/biomni-method/perturb-2026/experiments/sprints/sprint_004'
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from orth_features import load_orth_features as _load_orth_v1
except ImportError:
    if _LEGACY_V1 not in sys.path:
        sys.path.insert(0, _LEGACY_V1)
    from orth_features import load_orth_features as _load_orth_v1


def load_orth_features_v2(all_perts, features=('reactome',), ref_dir=REF_DIR):
    """
    Extended loader with new sprint_005 features.
    Falls through to v1 for legacy sources.
    """
    ref = Path(ref_dir)
    n = len(all_perts)
    out = {}
    stats = {}

    # Split into v1 features (delegate) and v2 features (implement here)
    v1_features = tuple(f for f in features if f in ('string', 'reactome', 'go_bp'))
    v2_features = tuple(f for f in features if f in ('hallmark', 'collectri', 'progeny'))

    if v1_features:
        v1_out, v1_stats = _load_orth_v1(all_perts, features=v1_features, ref_dir=ref)
        out.update(v1_out)
        stats.update(v1_stats)

    if 'hallmark' in v2_features:
        m = sparse.load_npz(ref / 'hallmark_multihot.npz')  # (n_all_genes, 50)
        with open(ref / 'hallmark_genes_list.json') as f:
            gene_list = json.load(f)
        gene_idx = {g: i for i, g in enumerate(gene_list)}
        A = np.zeros((n, m.shape[1]), dtype=np.float32)
        hit = 0
        for i, p in enumerate(all_perts):
            if p in gene_idx:
                row = m[gene_idx[p], :].toarray().flatten()
                if row.sum() > 0:
                    A[i] = row
                    hit += 1
        out['hallmark'] = A
        stats['hallmark'] = f"{hit}/{n} ({hit/n:.1%}) via MSigDB Hallmark 2020"

    if 'collectri' in v2_features:
        m = sparse.load_npz(ref / 'collectri_multihot.npz')  # (n_targets, 1185 TFs)
        with open(ref / 'collectri_targets.json') as f:
            target_list = json.load(f)
        target_idx = {g: i for i, g in enumerate(target_list)}
        A = np.zeros((n, m.shape[1]), dtype=np.float32)
        hit = 0
        for i, p in enumerate(all_perts):
            if p in target_idx:
                row = m[target_idx[p], :].toarray().flatten()
                if row.sum() > 0:
                    A[i] = row
                    hit += 1
        out['collectri'] = A
        stats['collectri'] = f"{hit}/{n} ({hit/n:.1%}) via CollecTRI TF regulon"

    if 'progeny' in v2_features:
        m = sparse.load_npz(ref / 'progeny.npz')  # (n_targets, 14 pathways) - signed
        with open(ref / 'progeny_targets.json') as f:
            target_list = json.load(f)
        target_idx = {g: i for i, g in enumerate(target_list)}
        A = np.zeros((n, m.shape[1]), dtype=np.float32)
        hit = 0
        for i, p in enumerate(all_perts):
            if p in target_idx:
                row = m[target_idx[p], :].toarray().flatten()
                if np.abs(row).sum() > 0:
                    A[i] = row
                    hit += 1
        out['progeny'] = A
        stats['progeny'] = f"{hit}/{n} ({hit/n:.1%}) via PROGENy 14-pathway signature"

    return out, stats


if __name__ == '__main__':
    ds = 'replogle22k562'
    with open(f'/mnt/results/biomni-method/perturb-2026/gpu_stage/{ds}/all_perts.json') as f:
        all_perts = json.load(f)
    feats, stats = load_orth_features_v2(all_perts, features=('reactome', 'hallmark', 'collectri', 'progeny'))
    print(f"{ds}: n_perts={len(all_perts)}")
    for name, matrix in feats.items():
        print(f"  {name}: shape={matrix.shape}, {stats[name]}")
