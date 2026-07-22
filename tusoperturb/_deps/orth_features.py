"""
Orthogonal feature loader for Sprint 004.

Loads STRING v12 node2vec, Reactome multi-hot, and GO BP multi-hot features,
indexed by gene symbol. Missing perts get zero rows (Ridge-neutral).

Usage:
    from orth_features import load_orth_features
    feats, stats = load_orth_features(all_perts, features=('string', 'reactome', 'go_bp'))
    # feats['string'] -> (n_perts, 128) float32
    # feats['reactome'] -> (n_perts, 1816) float32
    # feats['go_bp'] -> (n_perts, 5406) float32
"""
import numpy as np
import json
import os
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


def load_orth_features(all_perts, features=('string', 'reactome', 'go_bp'), ref_dir=REF_DIR):
    """
    Load orthogonal features for a list of pert gene symbols.
    
    Args:
        all_perts: list of pert gene symbols (order preserved)
        features: tuple of ('string', 'reactome', 'go_bp')
        ref_dir: path to data/ref/ directory
    
    Returns:
        feats: dict {feature_name: (n_perts, n_dims) float32 matrix}
        stats: dict {feature_name: coverage string}
    """
    ref = Path(ref_dir)
    n = len(all_perts)
    out = {}
    stats = {}
    
    # STRING v12 node2vec with legacy alias recovery
    if 'string' in features:
        emb = np.load(ref / 'string_v12_n2v_128.npy')
        with open(ref / 'string_v12_gene_names.json') as f:
            names = json.load(f)
        with open(ref / 'string_v12_pert_aliases.json') as f:
            aliases = json.load(f)
        n2idx = {g: i for i, g in enumerate(names)}
        
        A = np.zeros((n, emb.shape[1]), dtype=np.float32)
        hit = 0
        for i, p in enumerate(all_perts):
            if p in n2idx:
                A[i] = emb[n2idx[p]]
                hit += 1
            elif p in aliases and aliases[p] in n2idx:
                A[i] = emb[n2idx[aliases[p]]]
                hit += 1
        out['string'] = A
        stats['string'] = f"{hit}/{n} ({hit/n:.1%}) via STRING v12 + legacy aliases"
    
    # Reactome / GO BP share the same gene universe
    if 'reactome' in features or 'go_bp' in features:
        with open(ref / 'annot_gene_names.json') as f:
            annot_names = json.load(f)
        annot_idx = {g: i for i, g in enumerate(annot_names)}
    
    if 'reactome' in features:
        m = sparse.load_npz(ref / 'reactome_multihot.npz')
        A = np.zeros((n, m.shape[1]), dtype=np.float32)
        hit = 0
        for i, p in enumerate(all_perts):
            if p in annot_idx:
                row = m[annot_idx[p], :].toarray().flatten()
                if row.sum() > 0:
                    A[i] = row
                    hit += 1
        out['reactome'] = A
        stats['reactome'] = f"{hit}/{n} ({hit/n:.1%}) via Reactome 2022"
    
    if 'go_bp' in features:
        m = sparse.load_npz(ref / 'go_bp_multihot.npz')
        A = np.zeros((n, m.shape[1]), dtype=np.float32)
        hit = 0
        for i, p in enumerate(all_perts):
            if p in annot_idx:
                row = m[annot_idx[p], :].toarray().flatten()
                if row.sum() > 0:
                    A[i] = row
                    hit += 1
        out['go_bp'] = A
        stats['go_bp'] = f"{hit}/{n} ({hit/n:.1%}) via GO BP 2023"
    
    return out, stats


def concat_features(E_genept, feats, features=('string', 'reactome', 'go_bp')):
    """
    Concatenate GenePT (3072d) with selected orthogonal features.
    Returns concatenated matrix ordered: genept, string, reactome, go_bp.
    """
    parts = [E_genept]
    for f in features:
        if f in feats:
            parts.append(feats[f])
    return np.concatenate(parts, axis=1)


if __name__ == '__main__':
    import json as _j
    # Self-test: verify K562 coverage
    ds = 'replogle22k562'
    with open(f'/mnt/results/biomni-method/perturb-2026/gpu_stage/{ds}/all_perts.json') as f:
        all_perts = _j.load(f)
    feats, stats = load_orth_features(all_perts)
    print(f"{ds}: n_perts={len(all_perts)}")
    for name, matrix in feats.items():
        print(f"  {name}: shape={matrix.shape}, {stats[name]}")
