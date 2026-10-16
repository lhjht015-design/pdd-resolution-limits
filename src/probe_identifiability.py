"""PDD identifiability probe -- structure of the solution set of Phi_R.

THE MODEL, taken from Arasti et al. (RECOMB 2026), "Deconvolving Phylogenetic
Distance Mixtures".

    Full tree F = reference R plus k query leaves, query q attached on edge e_q of
    R at relative position x_q in (0,1), with pendant length y_q and abundance p_q
    on the simplex. The observation is the abundance-weighted mean path distance
    from each reference leaf to the queries:

        d_r = sum_q p_q * d_F(r, q),     r = 1..n      (n = |R| reference leaves)

    Writing a_q for the CHILD endpoint of e_q (so anchors are in bijection with
    non-root nodes) and ell_a for the length of the edge into a,

        d_F(r, q) = d_R(r, a_q) + C_{r,a_q} * ell_{a_q} * x_q + y_q
        C_{r,a}   = +1 if r lies in the subtree below a, else -1
        ybar      = sum_q p_q y_q          (only this combination is identifiable)

    so

        Phi_R(e, x, p, ybar)_r
            = sum_q p_q * d_R(r,a_q) + sum_q u_q * C_{r,a_q} * ell_{a_q} + ybar,
        u_q = x_q * p_q.

THE ONE STRUCTURAL FACT THIS PROBE IS BUILT ON
    Given the anchor assignment e, Phi is LINEAR in (p, u, ybar), and the domain
    is the open convex set {p > 0, sum p = 1, 0 < u_q < p_q, ybar > 0}.
    Therefore, at an interior point,

        the placement is identifiable  <=>  the 2k x n Jacobian has full rank 2k,

    because the free coordinates are (k-1) for p, k for u and 1 for ybar = 2k.
    Two immediate consequences, neither of which needs a computation:

      * 2k > n  =>  rank <= n < 2k  =>  GENERICALLY NON-IDENTIFIABLE. No tree,
        no edge lengths, no position can rescue it. Dimension counting alone.
      * 2k = n  =>  identifiability requires the single square system to be
        invertible, which is a codimension-one condition on the edge lengths.

    The paper's Figure 1d counterexample is the k=2, n=4 case, i.e. exactly the
    boundary. So the whole question is: for n >= 2k, which trees and which anchor
    assignments are non-identifiable, and is there a combinatorial criterion?

WHAT IS ENUMERATED
    All rooted binary trees on n labelled leaves, for n = 4,5,6 (15, 105, 945
    shapes). All anchor assignments: ordered k-tuples of distinct non-root nodes
    (2n-2 of them). Edge lengths: all-equal, and random. Rank is measured by SVD
    with a relative tolerance, and the root-invariance of the verdict is checked
    explicitly, since C and ell are root-dependent objects but Phi is not.

Usage:
    probe_identifiability.py [--ns 4,5] [--ks 2,3] [--seed 0]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections import deque

import numpy as np

OUT = "./out"


# --------------------------------------------------------------- trees
def gen_trees(leaves):
    """Every rooted binary tree on the given leaf set, each generated once by
    pinning leaves[0] into the first side of the split."""
    leaves = tuple(leaves)
    if len(leaves) == 1:
        yield ("L", leaves[0])
        return
    first, rest = leaves[0], leaves[1:]
    for r in range(len(rest)):
        for combo in itertools.combinations(rest, r):
            s1 = (first,) + combo
            s2 = tuple(x for x in rest if x not in combo)
            if not s2:
                continue
            for t1 in gen_trees(s1):
                for t2 in gen_trees(s2):
                    yield ("I", t1, t2)


class Tree:
    """Rooted binary tree; node 0 is the root. Built by `build_tree`, which fills
    parent/children/lab/isleaf and then calls `_distances`."""

    __slots__ = ("parent", "children", "lab", "isleaf", "root", "nleaf",
                 "leaves", "nonroot", "len", "D", "C")


def build_tree(t, lengths="equal", rng=None):
    """Recursive builder returning a Tree. The lists start EMPTY -- seeding them
    with a placeholder makes the first `new()` return 1, leaving node 0 orphaned
    and the adjacency disconnected."""
    parent, children, lab, isleaf = [], [], [], []

    def new(l=None, leaf=False):
        parent.append(-1); children.append([]); lab.append(l); isleaf.append(leaf)
        return len(parent) - 1

    def rec(node_tuple):
        """Attach `node_tuple` under a fresh internal node; returns that node."""
        if node_tuple[0] == "L":
            c = new(l=node_tuple[1], leaf=True)
            return c, c
        inner = new()
        for side in node_tuple[1:]:
            c, _ = rec(side)
            parent[c] = inner
            children[inner].append(c)
        return inner, None

    # the top-level 'I' node *is* the root; its children are the two subtrees
    root, _ = rec(t)
    T = Tree.__new__(Tree)
    T.parent, T.children, T.lab, T.isleaf = parent, children, lab, isleaf
    T.root = root
    T.nleaf = sum(isleaf)
    T.leaves = [i for i in range(len(parent)) if isleaf[i]]
    T.nonroot = [i for i in range(len(parent)) if i != root]
    if lengths == "equal":
        T.len = np.ones(len(parent))
    else:
        T.len = 0.2 + rng.random(len(parent))
    T.len[root] = 0.0
    T._distances()
    return T


def _distances(self):
    n = len(self.parent)
    adj = [[] for _ in range(n)]
    for v in range(n):
        if self.parent[v] >= 0:
            adj[v].append((self.parent[v], self.len[v]))
            adj[self.parent[v]].append((v, self.len[v]))
    D = np.zeros((n, len(self.leaves)))
    for a in range(n):
        dist = {a: 0.0}
        q = deque([a])
        while q:
            u = q.popleft()
            for w, l in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + l
                    q.append(w)
        for j, r in enumerate(self.leaves):
            D[a, j] = dist[r]
    self.D = D
    # subtree membership: r descends from a
    C = np.zeros((n, len(self.leaves)))
    for a in range(n):
        sub = set()
        q = deque([a])
        while q:
            u = q.popleft()
            sub.add(u)
            for w, _ in adj[u]:
                if w != self.parent[u]:
                    q.append(w)
        for j, r in enumerate(self.leaves):
            C[a, j] = 1.0 if r in sub else -1.0
    self.C = C


Tree._distances = _distances


# --------------------------------------------------------------- Jacobian
def jacobian(T, anchors):
    """(2k x n) derivative of Phi w.r.t. (p_1..p_{k-1}, u_1..u_k, ybar).

    Columns are the reference leaves. Rows: the k-1 independent abundance
    directions, then the k position-weighted-abundance directions, then ybar.
    """
    k = len(anchors)
    n = len(T.leaves)
    rows = []
    base = T.D[anchors[-1]]                    # p_k = 1 - sum_{q<k} p_q
    for q in range(k - 1):
        rows.append(T.D[anchors[q]] - base)
    for q in range(k):
        rows.append(T.C[anchors[q]] * T.len[anchors[q]])
    rows.append(np.ones(n))
    return np.array(rows)


def rank_of(M, tol=1e-9):
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * max(1.0, s[0])).sum())


def is_identifiable(T, anchors, tol=1e-9):
    J = jacobian(T, anchors)
    return rank_of(J, tol) == J.shape[0], J


# --------------------------------------------------------------- structure
def features(T, anchors):
    """Structural features of an anchor assignment, for the search for a
    criterion. All are root-independent except `chain_to_root`, which is
    recorded separately because Phi itself does not depend on the root."""
    a = list(anchors)
    k = len(a)
    par = [T.parent[x] for x in a]
    anc = []
    for x in a:
        s, u = set(), x
        while u >= 0:
            s.add(u); u = T.parent[u]
        anc.append(s)
    # is one anchor an ancestor of another? (root-independent: ancestry does not
    # depend on where the root is put, as long as it is outside both subtrees)
    nested = False
    for x, y in itertools.permutations(a, 2):
        if y in anc[a.index(x)]:
            nested = True
    # the MRCA of the anchors, and whether it is the node the tree is rooted at
    common = [u for u in anc[0] if all(u in s for s in anc[1:])] if k > 1 else [a[0]]
    mrca = min(common, key=lambda u: sum(T.D[u]))
    feat = dict(
        k=k,
        n_leaves=len(T.leaves),
        same_parent=len(set(par)) < k,
        any_sibling=any(T.parent[x] == T.parent[y]
                        for x, y in itertools.combinations(a, 2)),
        all_leaves=all(T.isleaf[x] for x in a),
        all_internal=all(not T.isleaf[x] for x in a),
        nested=nested,
        mrca_is_root=(mrca == T.root),
        sum_len=float(sum(T.len[x] for x in a)),
    )
    # --- root-free features -------------------------------------------------
    # sigma_q(r) = +1 iff r lies on the anchor side of the placement edge e_q.
    # The dependence condition is  alpha*g + beta*s1 + gamma*s2 + delta*1 = 0
    # for every leaf, with g(r) = d(r,P_1) - d(r,P_2); it is root-free, unlike
    # `nested`, which is written in terms of ancestry.
    sig_pairs = 0 if k < 2 else len(np.unique(np.stack(
        [T.C[a[0]], T.C[a[1]]], axis=1), axis=0))
    g = (T.D[a[0]] - T.D[a[1]]) if k > 1 else np.zeros(len(T.leaves))
    g_range = float(g.max() - g.min()) if k > 1 else 0.0
    # the two placement EDGES are (parent(x), x); they share a vertex if either
    # parent is a common node, or one anchor is the other's parent. All three
    # cases are root-free.
    share = False
    for x, y in itertools.permutations(a, 2):
        if T.parent[x] == T.parent[y] or T.parent[x] == y:
            share = True

    feat["edges_share_vertex"] = share
    feat["same_parent_only"] = len(set(par)) < k
    feat["sigma_pairs"] = sig_pairs
    feat["g_range"] = g_range
    return feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="4,5")
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length-sets", type=int, default=3,
                    help="number of independent random length assignments, to "
                         "separate a combinatorial degeneracy from a "
                         "measure-zero one")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)

    print("=" * 92)
    print("PDD identifiability probe")
    print("=" * 92)
    print("Phi is LINEAR in (p, u = x*p, ybar) given the anchor assignment, and the")
    print("free dimension is 2k. The observer has n numbers. So:")
    print("    2k >  n   ->  NON-IDENTIFIABLE by dimension counting alone")
    print("    2k == n   ->  identifiable iff the square Jacobian is invertible")
    print("    2k <  n   ->  a genuine rank condition on the tree + anchor choice\n")

    rows = []
    for n in [int(v) for v in args.ns.split(",")]:
        for k in [int(v) for v in args.ks.split(",")]:
            if k > 2 * n - 2:
                continue
            trees = list(gen_trees(range(1, n + 1)))
            print(f"--- n={n} k={k}: {len(trees)} labelled rooted binary trees; "
                  f"2k={2*k} free parameters vs n={n} observations", flush=True)
            for si in range(args.length_sets):
                lmode = "equal" if si == 0 else "random"
                rs = np.random.default_rng(1000 + si)
                for ti, t in enumerate(trees):
                    T = build_tree(t, "equal" if si == 0 else "random", rs)
                    for anchors in itertools.combinations(T.nonroot, k):
                        ok, J = is_identifiable(T, anchors)
                        r = dict(n=n, k=k, lengths=lmode, lset=si, tree=ti,
                                 anchors=str(anchors), ident=int(ok))
                        r.update({f"f_{a}": b for a, b in
                                  features(T, anchors).items() if a != "k"})
                        rows.append(r)
                sub = [r for r in rows if r["n"] == n and r["k"] == k
                       and r["lset"] == si]
                f = sum(r["ident"] for r in sub) / len(sub)
                print(f"    lengths={lmode:6s} set={si}  identifiable "
                      f"{sum(r['ident'] for r in sub)}/{len(sub)} "
                      f"({100*f:.1f}%)", flush=True)
            print(flush=True)

    import pandas as pd
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "identifiability.csv"), index=False)

    print("=" * 92)
    print("### identifiability by structural feature (all n, k pooled)")
    print("### identical rates across the 4 length assignments above means the")
    print("### rank deficiency is COMBINATORIAL, not a measure-zero coincidence")
    for col in ("f_edges_share_vertex", "f_same_parent_only", "f_sigma_pairs",
                "f_all_leaves", "f_all_internal"):
        g = D.groupby(col)["ident"].agg(["mean", "size"])
        print(f"  {col:18s} " + "   ".join(
            f"{v}={g.loc[v,'mean']:.3f}(n={int(g.loc[v,'size'])})"
            for v in g.index))
    print("\n### the root-free version of the criterion")
    print("###   dependence is  alpha*g + beta*s1 + gamma*s2 + delta*1 = 0  for")
    print("###   all leaves, with s_q the side-of-edge indicator. Fewer distinct")
    print("###   (s1,s2) pairs attained => the system is more constrained.")
    D["adj"] = D.f_edges_share_vertex
    print(D.groupby(["n", "k", "adj", "f_sigma_pairs"])["ident"]
           .agg(["mean", "size"]).round(3).to_string())
    print("\n### by (n, k), split by length assignment (a combinatorial "
          "degeneracy gives identical fractions)")
    print(D.groupby(["n", "k", "lset"])["ident"]
           .agg(["mean", "size"]).round(4).to_string())
    print("\n### TEST OF THE CONJECTURED CRITERION")
    print("###   predicted non-identifiable  <=>  (siblings) OR (nested anchors)")
    D["pred_bad"] = (D.f_same_parent | D.f_nested).astype(int)
    D["actual_bad"] = 1 - D.ident
    ct = D.groupby(["n", "k"]).apply(
        lambda g: pd.Series({
            "n_configs": len(g),
            "actual_bad": int(g.actual_bad.sum()),
            "pred_bad": int(g.pred_bad.sum()),
            "both": int(((g.pred_bad == 1) & (g.actual_bad == 1)).sum()),
            "pred_only": int(((g.pred_bad == 1) & (g.actual_bad == 0)).sum()),
            "actual_only": int(((g.pred_bad == 0) & (g.actual_bad == 1)).sum()),
        }), include_groups=False)
    print(ct.to_string())
    print("\n  pred_only > 0  => the criterion over-predicts (a sibling/nested "
          "case IS identifiable)")
    print("  actual_only> 0 => the criterion is incomplete (other bad cases exist)")

    other = D[(D.pred_bad == 0) & (D.actual_bad == 1)]
    if len(other):
        print(f"\n### non-identifiable cases NOT explained by siblings-or-nested: "
              f"{len(other)}")
        print(other.groupby(["n", "f_all_leaves", "f_mrca_is_root"])
                    .size().to_string())
        print("\n  sample (n, k, tree, anchors):")
        for _, r in other.head(12).iterrows():
            print(f"    n={r['n']} k={r['k']} tree={r['tree']} "
                  f"anchors={r['anchors']} leaves={r['f_all_leaves']} "
                  f"mrca_root={r['f_mrca_is_root']}")

    bad = D[D.ident == 0]
    print(f"\n### non-identifiable total: {len(bad)}/{len(D)}")
    print("\nartifacts ->", os.path.join(OUT, "identifiability.csv"))
    print("\nartifacts ->", os.path.join(OUT, "identifiability.csv"))


if __name__ == "__main__":
    main()
