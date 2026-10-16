"""Sibling co-occurrence census (Table S2, EMP prevalence rows).

Counts, across the EMP subset_2k samples, the sibling pairs of the Greengenes 97%
tree (two leaves sharing a parent) whose BOTH members co-occur in a sample. Reports
the total across all samples and the fraction of samples containing at least one.

Usage: python sibling_census.py
"""
from __future__ import annotations
import numpy as np
import h5py
import scipy.sparse as sp
from bigres import BigTree

RAW = "E:/RECOB_DATA/data/raw/emp"


def load_tree():
    bt = BigTree.from_newick(f"{RAW}/97_otus.tree")
    lab2node = {bt.lab[v]: v for v in range(bt.n) if bt.isleaf[v]}
    return bt, lab2node


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


def main():
    bt, lab2node = load_tree()
    leaves = [v for v in range(bt.n) if bt.isleaf[v]]

    # sibling pairs: leaf pairs sharing a parent; sibling_of[a] = b, a < b
    from collections import defaultdict
    children = defaultdict(list)
    for v in leaves:
        children[int(bt.parent[v])].append(v)
    sib_a, sib_b = [], []
    sibling_of = np.full(bt.n, -1, dtype=np.int64)
    for p, kids in children.items():
        leaf_kids = [k for k in kids if bt.isleaf[k]]
        if len(leaf_kids) == 2:
            a, b = sorted(leaf_kids)
            sib_a.append(a); sib_b.append(b)
            sibling_of[a] = b; sibling_of[b] = a
    sib_a = np.array(sib_a, dtype=np.int64)
    sib_b = np.array(sib_b, dtype=np.int64)
    print(f"[tree] {len(leaves)} leaves, {len(sib_a)} sibling pairs")

    M, samp_ids, obs_ids = load_biom()
    col2node = np.full(len(obs_ids), -1, dtype=np.int64)
    for j, oid in enumerate(obs_ids):
        col2node[j] = lab2node.get(oid, -1)

    total = 0
    samples_with = 0
    present = np.zeros(bt.n, dtype=bool)
    for i in range(M.shape[0]):
        present[:] = False
        row = M.getrow(i)
        for j in row.indices:
            v = col2node[j]
            if v >= 0:
                present[v] = True
        c = int(np.sum(present[sib_a] & present[sib_b]))
        total += c
        if c >= 1:
            samples_with += 1
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{M.shape[0]} samples, running total {total}", flush=True)

    n = M.shape[0]
    print(f"[census] co-occurring sibling pairs (across all samples): {total}")
    print(f"[census] samples with >=1 sibling pair: {samples_with}/{n} "
          f"= {100.0 * samples_with / n:.1f}%")


if __name__ == "__main__":
    main()
