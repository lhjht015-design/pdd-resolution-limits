"""k = 2: is the identifiability criterion combinatorial, and what exactly is it?

THE CONJECTURE UNDER TEST
    For two placements P_1, P_2 the Jacobian's rows span {1, sigma_1, sigma_2, g}
    with g(r) = d(r,P_1) - d(r,P_2). Since two tree edges induce LAMINAR splits,
    at most three of the four (sigma_1, sigma_2) quadrants are non-empty, so
    span{1,sigma_1,sigma_2} is exactly the set of ALL functions of (sigma_1,sigma_2)
    on the attained pairs. Hence

        rank J < 4  <=>  g in span{1,sigma_1,sigma_2}  <=>  g is CONSTANT on every
                                                             sigma-class.

    And g depends on r only through pi_B(r), the projection of r onto the
    backbone path B = P_1..P_2:
        g(r) = 2*d(P_1, pi_B(r)) - L,    L = d(P_1,P_2),
    because d(pi,P_1) + d(pi,P_2) = L. So "g constant on a class" is the same as
    "the projection is constant on that class".

    Two consequences the probe should confirm exactly:
      A. edges sharing a vertex  =>  g is constant on all three classes  =>  never
         identifiable. (Three regions: subtree(a_1) projects to P_1, subtree(a_2)
         to P_2, everything else to the shared vertex v.)
      B. edges disjoint  =>  the two outer classes project to P_1 and P_2
         automatically; identifiability then hinges ONLY on whether the middle
         class has >= 2 distinct projections onto B.

WHAT IS MEASURED
    For every labelled tree and every pair of distinct edges, at midpoint
    positions: the rank of J, the sigma-classes, whether g is constant on each,
    whether the edges share a vertex, and the projection diversity of the middle
    class. Then the counts of agreement.

Usage:
    verify_k2.py [--max-n 7]
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from probe_identifiability import build_tree, gen_trees, rank_of

OUT = "./out"
TOL = 1e-9


def components_without(T, e1, e2):
    """Vertices grouped by connectivity in R - {e1, e2}; the two edges are
    given as (parent, child) pairs."""
    n = len(T.parent)
    cut = {frozenset((T.parent[e[1]], e[1])) for e in (e1, e2)}
    adj = [[] for _ in range(n)]
    for v in range(n):
        if T.parent[v] >= 0 and frozenset((T.parent[v], v)) not in cut:
            adj[v].append(T.parent[v]); adj[T.parent[v]].append(v)
    comp = [-1] * n
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack, comp[s] = [s], c
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if comp[w] < 0:
                    comp[w] = c; stack.append(w)
        c += 1
    return comp, c


def edge_params(T, anchors):
    """(s1, s2, g) for two anchors. g is taken to be the SAME vector that appears
    in the Jacobian, D[a_1] - D[a_2], not the placement-point version. The two
    differ by c_1*sigma_1 - c_2*sigma_2, which lies in the span, so membership in
    span{1,sigma_1,sigma_2} -- and constancy on each sigma-class -- is identical
    for both. Using the Jacobian's own row keeps the iff test exactly aligned."""
    s1, s2 = T.C[anchors[0]], T.C[anchors[1]]
    g = T.D[anchors[0]] - T.D[anchors[1]]
    return s1, s2, g


def sigma_classes(s1, s2):
    classes = {}
    for j in range(len(s1)):
        classes.setdefault((s1[j], s2[j]), []).append(j)
    return classes


def const_on_class(g, idx):
    return float(g[idx].max() - g[idx].min()) < 1e-9


def jacobian_k2(T, anchors):
    J = np.array([np.ones(len(T.leaves)),
                  T.C[anchors[0]] * T.len[anchors[0]],
                  T.C[anchors[1]] * T.len[anchors[1]],
                  T.D[anchors[0]] - T.D[anchors[1]]])
    return J


def node_path(T, a, b):
    """Nodes on the tree path from a to b, inclusive."""
    par = {a: None}
    stack = [a]
    while stack:
        u = stack.pop()
        if u == b:
            break
        for w in ([T.parent[u]] if T.parent[u] >= 0 else []) + T.children[u]:
            if w not in par:
                par[w] = u; stack.append(w)
    out, u = [], b
    while u is not None:
        out.append(u); u = par[u]
    return out


def projection_nodes(T, a1, a2):
    """For each leaf, the single node of the a1-a2 path its component attaches
    to. This is the projection pi_B in the combinatorial (node) form, and g_a is
    2*d(a1, pi) - L on it, so 'g_a constant on a set' <=> 'the set has one
    projection node'."""
    P = node_path(T, a1, a2)
    Pset = set(P)
    # distance from each node to the nearest path node, via multi-source BFS
    owner = {u: u for u in P}
    stack = list(P)
    while stack:
        u = stack.pop()
        for w in ([T.parent[u]] if T.parent[u] >= 0 else []) + T.children[u]:
            if w not in owner:
                owner[w] = owner[u]; stack.append(w)
    return np.array([owner[r] for r in T.leaves])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=7)
    ap.add_argument("--verbose", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 96)
    print("k = 2 identifiability -- testing  rank J < 4  <=>  g constant on every "
          "sigma-class")
    print("=" * 96)

    rows = []
    mismatch_iff = 0
    mismatch_proj = 0
    shown = 0
    shown2 = 0
    for n in range(4, args.max_n + 1):
        ntree = 0
        agg = dict(tot=0, ident=0, adj=0, adj_ident=0, disj=0, disj_ident=0,
                   disj_middlevar=0, disj_middlevar_ident=0)
        for ti, t in enumerate(gen_trees(range(1, n + 1))):
            T = build_tree(t, "random", np.random.default_rng(11 + ti))
            ntree += 1
            edges = [(T.parent[v], v) for v in T.nonroot]
            for (e1, e2) in itertools.combinations(edges, 2):
                anchors = (e1[1], e2[1])
                s1, s2, g = edge_params(T, anchors)
                J = jacobian_k2(T, anchors)
                rk = rank_of(J)
                ident = (rk == 4)
                classes = sigma_classes(s1, s2)
                g_const = all(const_on_class(g, idx) for idx in classes.values())
                proj = projection_nodes(T, anchors[0], anchors[1])
                proj_const = all(len(set(proj[idx].tolist())) == 1
                                 for idx in classes.values())
                if (rk < 4) != proj_const:
                    mismatch_proj += 1
                    if shown2 < args.verbose:
                        print(f"  PROJ MISMATCH n={n} tree={ti} anchors={anchors}"
                              f" rank={rk} proj_const={proj_const}")
                        shown2 += 1
                adjacent = (e1[0] == e2[0]) or (e1[0] == e2[1]) or (e1[1] == e2[0])
                # for disjoint edges only the "rest" sigma-class can vary; count
                # its distinct projection nodes
                rest_idx = [j for j in range(len(T.leaves))
                            if not (T.C[anchors[0]][j] > 0 or T.C[anchors[1]][j] > 0)]
                rest_proj = (len(set(proj[rest_idx].tolist()))
                             if rest_idx else 0)
                # the middle class = the leaves in neither outer component
                comp, nc = components_without(T, e1, e2)
                if adjacent:
                    midvar = None
                else:
                    # removing e_1 = (par_1, a_1) and e_2 = (par_2, a_2) leaves
                    # three components: the a_1 side, the a_2 side, and the middle
                    # (which contains par_1 and par_2). Only the middle can vary.
                    outer = {comp[e1[1]], comp[e2[1]]}
                    leaf_comp = {}
                    for j, r in enumerate(T.leaves):
                        leaf_comp.setdefault(comp[r], []).append(j)
                    mids = [v for k_, v in leaf_comp.items() if k_ not in outer]
                    midvar = bool(mids and max(
                        g[v].max() - g[v].min() for v in mids) > 1e-9)
                if (rk < 4) != g_const:
                    mismatch_iff += 1
                    if shown < args.verbose:
                        print(f"  MISMATCH n={n} tree={ti} anchors={anchors} "
                              f"rank={rk} g_const={g_const} adj={adjacent}")
                        shown += 1
                agg["tot"] += 1
                agg["ident"] += ident
                if adjacent:
                    agg["adj"] += 1; agg["adj_ident"] += ident
                else:
                    agg["disj"] += 1; agg["disj_ident"] += ident
                    agg["disj_middlevar"] += midvar
                    agg["disj_middlevar_ident"] += (midvar and ident)
                rows.append(dict(n=n, adj=adjacent, ident=int(ident),
                                 g_const=int(g_const), rk=rk,
                                 midvar=-1 if midvar is None else int(midvar),
                                 rest_proj=rest_proj))
        print(f"n={n}: {ntree} trees, {agg['tot']} edge pairs | "
              f"identifiable {agg['ident']}/{agg['tot']} "
              f"({100*agg['ident']/agg['tot']:.1f}%)")
        print(f"      adjacent {agg['adj']}: identifiable {agg['adj_ident']}")
        print(f"      disjoint {agg['disj']}: identifiable {agg['disj_ident']}"
              f" ({100*agg['disj_ident']/max(1,agg['disj']):.1f}%)")
        print(f"      of disjoint, middle-class projection VARIES in "
              f"{agg['disj_middlevar']}; of those, identifiable "
              f"{agg['disj_middlevar_ident']}")
        print(flush=True)

    print("=" * 96)
    print(f"### IFF TEST 1: (rank<4) vs (g constant on every sigma-class): "
          f"{mismatch_iff} disagreements")
    print(f"### IFF TEST 2: (rank<4) vs (every sigma-class has ONE projection "
          f"node): {mismatch_proj} disagreements")
    if mismatch_iff == 0 and mismatch_proj == 0:
        print("### PASS -- the criterion  rank J < 4  <=>  g const on sigma-classes")
        print("### holds exactly, and it is purely combinatorial (no edge lengths,")
        print("### no rooting: sigma-classes are level sets of edge-side indicators).")

    import pandas as pd
    D = pd.DataFrame(rows)
    print("\n### the structural form")
    print("###   PREDICTION: adjacent  =>  never identifiable")
    print("###               disjoint  =>  identifiable  <=>  middle projections "
          "take >= 2 values")
    a = D[D.adj]
    d = D[~D.adj]
    print(f"  adjacent: {a.ident.sum()}/{len(a)} identifiable")
    print(f"  disjoint: {d.ident.sum()}/{len(d)} identifiable")
    ct = pd.crosstab(d.midvar, d.ident)
    print("\n  disjoint, crosstab of (middle projection varies) x (identifiable):")
    print(ct.to_string())
    agree = int(((d.midvar == 1) == (d.ident == 1)).sum())
    print(f"  agreement: {agree}/{len(d)}")
    print("\n### random-placement ambiguity rate, P(non-identifiable | n)")
    print(D.groupby("n")["ident"].agg(["mean", "size"]).round(4).to_string())
    D.to_csv(os.path.join(OUT, "k2_theorem.csv"), index=False)
    print("\nartifacts ->", os.path.join(OUT, "k2_theorem.csv"))


if __name__ == "__main__":
    main()
