"""EMP probe: sample-level phylogenetic resolution on the Greengenes 97% tree.

Two per-sample statistics:

  1. abundance-weighted R_k = P[ Delta(T,S) = 0 ] over S ~ abundance-weighted
     k-taxa draws (the "bulk" resolution — dominated by the abundant, often
     phylogenetically diverse taxa).

  2. local confounding load  L = mean Delta(T, {t, n1, n2}) over each taxon t and
     its two nearest sample-neighbours (n1, n2) in tree distance.  This probes the
     *tight clades*, where the resolution limit actually lives.

Each is compared against a null model that preserves a sample's richness and
abundance vector but assigns the abundances to uniformly random taxa, so the
phylogenetic clustering is the only thing the null removes.

Usage:
    python emp_probe.py [--nsub 60] [--supports 60] [--max-per-env 0] [--seed 0]
"""
from __future__ import annotations

import argparse
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from bigres import BigTree, deficiency

RAW = "E:/RECOB_DATA/data/raw/emp"


def load_tree():
    bt = BigTree.from_newick(f"{RAW}/97_otus.tree")
    lab2node = {bt.lab[v]: v for v in range(bt.n) if bt.isleaf[v]}
    leaf_nodes = np.array([v for v in range(bt.n) if bt.isleaf[v]], dtype=np.int64)
    return bt, lab2node, leaf_nodes


def load_biom():
    f = h5py.File(f"{RAW}/emp_cr_gg_13_8.subset_2k.biom", "r")
    obs_ids = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["observation/ids"][:]]
    samp_ids = [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["sample/ids"][:]]
    data = f["sample/matrix/data"][:]
    idx = f["sample/matrix/indices"][:]
    ptr = f["sample/matrix/indptr"][:]
    M = sp.csr_matrix((data, idx, ptr), shape=(len(samp_ids), len(obs_ids)))
    f.close()
    return M, samp_ids, obs_ids


ENV_MAP = {
    "Animal distal gut": "gut", "Animal proximal gut": "gut",
    "Animal corpus": "gut", "Animal secretion": "gut",
    "Soil (non-saline)": "soil", "Sediment (non-saline)": "soil",
    "Water (saline)": "ocean", "Sediment (saline)": "ocean",
    "Surface (saline)": "ocean",
    "Plant surface": "plant", "Plant rhizosphere": "plant",
    "Plant corpus": "plant",
}


def tree_dist(bt, u, v):
    m = bt.lca(int(u), int(v))
    return int(bt.depth[u] + bt.depth[v] - 2 * bt.depth[m])


def abundance_Rk(bt, nodes, w, rng, k, M):
    """abundance-weighted R_k = P[Delta=0]."""
    r = len(nodes)
    if r < k:
        return np.nan
    p = w / w.sum()
    d = np.zeros(M, dtype=np.int64)
    for i in range(M):
        j = rng.choice(r, size=k, replace=False, p=p)
        d[i] = deficiency(bt, [int(nodes[x]) for x in j])
    return (d == 0).mean()


def confound_load(bt, nodes, rng, nsub, k=3):
    """mean Delta over nearest-neighbour triples (the tight-clade resolution)."""
    nodes = nodes.astype(np.int64)
    if len(nodes) < k:
        return np.nan
    if len(nodes) > nsub:
        nodes = rng.choice(nodes, size=nsub, replace=False)
    r = len(nodes)
    D = np.full((r, r), np.iinfo(np.int64).max, dtype=np.int64)
    for i in range(r):
        for j in range(i + 1, r):
            D[i, j] = D[j, i] = tree_dist(bt, nodes[i], nodes[j])
    ds = []
    for i in range(r):
        nn = np.argsort(D[i])[1:k]        # k-1 nearest neighbours
        tri = [int(nodes[i])] + [int(nodes[x]) for x in nn]
        ds.append(deficiency(bt, tri))
    ds = np.array(ds)
    return ds.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=60)
    ap.add_argument("--supports", type=int, default=60)
    ap.add_argument("--max-per-env", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    bt, lab2node, leaf_nodes = load_tree()
    M, samp_ids, obs_ids = load_biom()
    meta = pd.read_csv(f"{RAW}/emp_qiime_mapping_subset_2k.tsv", sep="\t",
                       dtype=str, comment=None, keep_default_na=False)
    obs_node = np.array([lab2node[o] for o in obs_ids], dtype=np.int64)
    meta_by = {s: i for i, s in enumerate(meta["#SampleID"])}
    sid_index = {s: i for i, s in enumerate(samp_ids)}
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for env in ["gut", "soil", "ocean", "plant"]:
        idxs = [sid_index[s] for s in samp_ids
                if ENV_MAP.get(meta.iloc[meta_by[s]]["empo_3"]) == env]
        if a.max_per_env:
            rng.shuffle(idxs)
            idxs = idxs[:a.max_per_env]
        for sidx in idxs:
            row = M.getrow(sidx).tocoo()
            nodes = obs_node[row.col]
            w = row.data.astype(np.float64)
            if len(nodes) < a.k:
                continue
            # real
            r_ab = abundance_Rk(bt, nodes, w, rng, a.k, a.supports)
            L_real = confound_load(bt, nodes, rng, a.nsub, a.k)
            # null (same richness + abundance vector, random taxa)
            rnodes = rng.choice(leaf_nodes, size=len(nodes), replace=False)
            r_ab_null = abundance_Rk(bt, rnodes, w, rng, a.k, a.supports)
            L_null = confound_load(bt, rnodes, rng, a.nsub, a.k)
            rows.append(dict(env=env, sample=samp_ids[sidx],
                             richness=len(nodes),
                             R_ab=r_ab, R_ab_null=r_ab_null,
                             L=L_real, L_null=L_null))
    D = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(f"EMP phylogenetic resolution probe  (k={a.k}, "
          f"nsub={a.nsub}, supports={a.supports})")
    print("=" * 78)
    g = D.groupby("env").agg(
        n=("sample", "size"),
        richness=("richness", "mean"),
        R_ab=("R_ab", "mean"),
        R_ab_null=("R_ab_null", "mean"),
        L_real=("L", "mean"),
        L_null=("L_null", "mean"),
    ).round(4)
    print(g.to_string())
    print("\nlocal confounding load L: real vs null (mean over samples)")
    D["L_gap"] = D.L - D.L_null
    print(D.groupby("env")["L_gap"].agg(
        mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)),
        n="size").round(5).to_string())
    print(f"\noverall: L_real={D.L.mean():.4f}  L_null={D.L_null.mean():.4f}  "
          f"gap={D.L_gap.mean():.4f}  (n={len(D)})")
    return D


if __name__ == "__main__":
    main()
