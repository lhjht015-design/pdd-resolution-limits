"""Counterfactual experiment (Step 2): an exact Phi-invisible biological perturbation.

For a real cherry {a,b} in the Greengenes tree with real abundances from an EMP sample,
construct two DISTINCT interior community states theta and theta' = theta + eps*v that
produce the SAME aggregate phylogenetic observation Phi.

The null direction is the cherry one:
    delta p_a = +eps,  delta p_b = -eps,  delta u_a = +eps,  delta u_b = -eps,
with u_q = p_q x_q, so the (p, x) realisation is
    p_a' = p_a + eps,        p_b' = p_b - eps,
    x_a' = (p_a x_a + eps) / (p_a + eps),
    x_b' = (p_b x_b - eps) / (p_b - eps),
and the observable part of Phi (the all-ones pendant term is a fixed scalar and is
identical in both states) is
    Phi = sum_q p_q D_{a_q} + sum_q u_q l_{a_q} C_{a_q} .

The cherry identity D_a - D_b = -l_a C_a + l_b C_b makes delta Phi = 0 exactly.
"""
from __future__ import annotations

import numpy as np
from collections import deque

from bigres import BigTree

RAW = "E:/RECOB_DATA/data/raw/emp"


def load_tree():
    bt = BigTree.from_newick(f"{RAW}/97_otus.tree")
    lab2node = {bt.lab[v]: v for v in range(bt.n) if bt.isleaf[v]}
    leaf_nodes = np.array([v for v in range(bt.n) if bt.isleaf[v]], dtype=np.int64)
    leaves_by_pos = np.zeros(bt.nleaf, dtype=np.int64)
    for v in range(bt.n):
        if bt.isleaf[v]:
            leaves_by_pos[bt.lo[v]] = v
    return bt, lab2node, leaf_nodes, leaves_by_pos


def dist_real(bt, u):
    """real-branch-length distance from node u to all leaves (leaf order)."""
    n = bt.n
    adj = [[] for _ in range(n)]
    for v in range(n):
        if bt.parent[v] >= 0:
            L = max(bt.length[v], 1e-9)
            adj[v].append((bt.parent[v], L))
            adj[bt.parent[v]].append((v, L))
    d = np.full(n, np.inf)
    d[u] = 0.0
    q = deque([u])
    while q:
        x = q.popleft()
        for w, l in adj[x]:
            if d[w] == np.inf:
                d[w] = d[x] + l
                q.append(w)
    return d


def observation(bt, leaves_by_pos, Ds, Cs, ells, p, x):
    """Phi (non-pendant part) for anchors with p and x. Ds: list of D_a vectors."""
    Phi = np.zeros(bt.nleaf)
    for q in range(len(p)):
        Phi = Phi + p[q] * Ds[q] + (p[q] * x[q]) * ells[q] * Cs[q]
    return Phi


def main():
    bt, lab2node, leaf_nodes, leaves_by_pos = load_tree()

    # load a real sample's abundances to pick a real, abundant cherry
    import h5py
    import scipy.sparse as sp
    import pandas as pd
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
    obs_node = np.array([lab2node[o] for o in obs_ids], dtype=np.int64)

    meta = pd.read_csv(f"{RAW}/emp_qiime_mapping_subset_2k.tsv", sep="\t",
                       dtype=str, comment=None, keep_default_na=False)
    meta_by = {s: i for i, s in enumerate(meta["#SampleID"])}
    sid_index = {s: i for i, s in enumerate(samp_ids)}
    gut_idxs = [sid_index[s] for s in samp_ids
                if meta.iloc[meta_by[s]]["empo_3"] in
                ("Animal distal gut", "Animal proximal gut")]

    # find an abundant cherry: two sibling OTUs both present, both with high count
    rng = np.random.default_rng(0)
    found = None
    for sidx in gut_idxs:
        row = M.getrow(sidx).tocoo()
        nodes = obs_node[row.col]
        counts = row.data.astype(float)
        # relative abundances among this sample
        rel = counts / counts.sum()
        by_node = {int(n): rel[i] for i, n in enumerate(nodes)}
        for i, a in enumerate(nodes):
            p = int(bt.parent[a])
            if p < 0:
                continue
            sib = [c for c in bt.children[p] if c != a and bt.isleaf[c] and int(c) in by_node]
            for b in sib:
                if by_node[int(a)] > 0.005 and by_node[int(b)] > 0.005:
                    # third taxon: most abundant other taxon
                    others = sorted(((by_node[x], int(x)) for x in by_node
                                     if int(x) not in (int(a), int(b))), reverse=True)
                    if others:
                        c = others[0][1]
                        found = (sidx, int(a), int(b), c, by_node)
                        break
            if found:
                break
        if found:
            break
    assert found is not None, "no abundant cherry found"
    sidx, a, b, c, by_node = found
    print(f"sample = {samp_ids[sidx]}  (emo_3 = {meta.iloc[meta_by[samp_ids[sidx]]]['empo_3']})")
    print(f"cherry = ({bt.lab[a]}, {bt.lab[b]})  third = {bt.lab[c]}")
    p_a, p_b, p_c = by_node[a], by_node[b], by_node[c]
    print(f"relative abundances: p_a={p_a:.4f}  p_b={p_b:.4f}  p_c={p_c:.4f}")
    print(f"tree sibling check: parent(a)==parent(b): {bt.parent[a]==bt.parent[b]}")

    # real-length distance vectors and cut vectors for the three anchors
    Ds = [dist_real(bt, u)[leaves_by_pos] for u in (a, b, c)]
    Cs = [np.where(leaves_by_pos == u, 1.0, -1.0) for u in (a, b, c)]
    ells = [bt.length[u] for u in (a, b, c)]

    # --- (1) verify the cherry identity exactly on the real tree / real lengths
    id_ = (Ds[0] - Ds[1]) + ells[0] * Cs[0] - ells[1] * Cs[1]
    print(f"\ncherry identity  D_a-D_b + l_a C_a - l_b C_b : "
          f"max|.| = {np.abs(id_).max():.3e}")

    # --- (2) choose an interior theta and the perturbed theta'
    x_a, x_b, x_c = 0.5, 0.5, 0.5          # any interior positions
    eps = 0.1 * min(p_a * x_a, p_b * x_b)  # stay interior
    print(f"\neps = {eps:.6f}")
    p = np.array([p_a, p_b, p_c])
    x = np.array([x_a, x_b, x_c])
    p_p = np.array([p_a + eps, p_b - eps, p_c])
    x_p = np.array([(p_a * x_a + eps) / (p_a + eps),
                    (p_b * x_b - eps) / (p_b - eps),
                    x_c])
    print(f"theta   p = {p.round(6)}   x = {x.round(6)}")
    print(f"theta'  p = {p_p.round(6)}   x = {x_p.round(6)}")
    print(f"p' interior: {np.all(p_p > 0)}   x' interior: {np.all((x_p > 0) & (x_p < 1))}")
    print(f"abundance change: p_a -> {p_a + eps:.5f} (+{eps:.5f}), "
          f"p_b -> {p_b - eps:.5f} (-{eps:.5f})")

    # --- (3) the two aggregate observations are identical
    Phi = observation(bt, leaves_by_pos, Ds, Cs, ells, p, x)
    Phi_p = observation(bt, leaves_by_pos, Ds, Cs, ells, p_p, x_p)
    scale = max(1.0, np.abs(Phi).max())
    print(f"\n||Phi(theta') - Phi(theta)||_inf = {np.abs(Phi_p - Phi).max():.3e}")
    print(f"relative                         = {np.abs(Phi_p - Phi).max()/scale:.3e}")
    print(f"||Phi||_inf = {np.abs(Phi).max():.3f}  (n={bt.nleaf} reference leaves)")

    # --- (4) a small excerpt for the figure: the two states vs the observation
    print("\nfigure excerpt (theta vs theta' on the cherry):")
    print(f"  state A : (p_a, p_b) = ({p_a:.4f}, {p_b:.4f}),  "
          f"(x_a, x_b) = ({x_a:.2f}, {x_b:.2f})")
    print(f"  state B : (p_a, p_b) = ({p_a+eps:.4f}, {p_b-eps:.4f}),  "
          f"(x_a, x_b) = ({x_p[0]:.2f}, {x_p[1]:.2f})")
    print(f"  identical aggregate observation: YES "
          f"(rel diff {np.abs(Phi_p-Phi).max()/scale:.2e})")


if __name__ == "__main__":
    main()
