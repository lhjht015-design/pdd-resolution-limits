"""k = 2: the complete criterion, verified and extrapolated.

THE CRITERION, in its final form
    Two placements on edges e_1, e_2 of R, anchors a_q = child endpoint of e_q.
    For every leaf r let pi(r) be the node of the a_1-a_2 path its component
    attaches to, and sigma_q(r) = +1 iff r is in subtree(a_q). Then

        rank J < 4   <=>   every (sigma_1, sigma_2) class has a single pi.

    Equivalently, and this is the statement to prove:

        non-identifiable
          <=>  e_1 and e_2 SHARE A VERTEX
               OR  (they lie in different branches AND every leaf outside
                    subtree(a_1) U subtree(a_2) attaches to the a_1-a_2 path
                    at one and the same node).

    The case analysis behind it:
      * share a vertex        -> the three classes project to a_1, a_2 and the
                                 shared vertex. Never identifiable.
      * nested, no shared vertex -> subtree(a_1) \\ subtree(a_2) contains leaves
                                 attached to at least two path nodes, because the
                                 path a_1..a_2 has >= 3 nodes. Always identifiable.
      * neither               -> subtree(a_1) -> a_1 and subtree(a_2) -> a_2 are
                                 single; everything hinges on the rest class.

WHAT THIS SCRIPT DOES
    1. Full enumeration at n = 4,5,6: verifies the criterion above as a
       predictor of rank J < 4, and reports every disagreement.
    2. Random sampling at n = 7..40: the frequency of each structural case and
       the resulting P(non-identifiable), to see the asymptotic behaviour and
       check it against the naive O(1/n) adjacency estimate.

Usage:
    criterion_k2.py [--max-enum 6] [--samples 20000]
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from probe_identifiability import build_tree, gen_trees, rank_of
from verify_k2 import (jacobian_k2, node_path, projection_nodes, sigma_classes)

OUT = "./out"


# --------------------------------------------------------------- random trees
def random_tree(n, rng):
    """A random labelled binary tree, built by repeatedly joining two uniformly
    chosen subtrees. Cheaper than drawing from gen_trees, which enumerates."""
    pool = [("L", i) for i in range(1, n + 1)]
    while len(pool) > 1:
        i = rng.integers(len(pool)); a = pool.pop(i)
        j = rng.integers(len(pool)); b = pool.pop(j)
        pool.append(("I", a, b))
    return pool[0]


def classify(T, a1, a2):
    """(share_vertex, nested, rest_single) for two anchors."""
    def ancset(v):
        s, u = set(), v
        while u >= 0:
            s.add(u); u = T.parent[u]
        return s
    A1, A2 = ancset(a1), ancset(a2)
    share = (T.parent[a1] == T.parent[a2] or T.parent[a1] == a2
             or T.parent[a2] == a1)
    nested = (a1 in A2) or (a2 in A1)
    proj = projection_nodes(T, a1, a2)
    rest = [j for j in range(len(T.leaves))
            if not (T.C[a1][j] > 0 or T.C[a2][j] > 0)]
    rest_single = (len(set(proj[rest].tolist())) <= 1) if rest else True
    return share, nested, rest_single


def predict(share, nested, rest_single):
    """The criterion: non-identifiable?"""
    if share:
        return True
    if nested:
        return False
    return rest_single


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-enum", type=int, default=6)
    ap.add_argument("--samples", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 96)
    print("k = 2 -- the complete criterion")
    print("=" * 96)
    print("non-identifiable  <=>  edges SHARE A VERTEX, or (different branches AND")
    print("                       the rest class attaches to the a1-a2 path at "
          "one node)\n")

    # ------------------------------------------------------------ enumeration
    print("--- full enumeration")
    rows = []
    bad = 0
    shown = 0
    for n in range(4, args.max_enum + 1):
        for ti, t in enumerate(gen_trees(range(1, n + 1))):
            T = build_tree(t, "random", np.random.default_rng(11 + ti))
            for e1, e2 in itertools.combinations(
                    [(T.parent[v], v) for v in T.nonroot], 2):
                a1, a2 = e1[1], e2[1]
                rk = rank_of(jacobian_k2(T, (a1, a2)))
                actual = (rk < 4)
                sh, ne, rs = classify(T, a1, a2)
                pr = predict(sh, ne, rs)
                if pr != actual:
                    bad += 1
                    if shown < 8:
                        print(f"    MISMATCH n={n} tree={ti} a=({a1},{a2}) "
                              f"share={sh} nested={ne} rest_single={rs} "
                              f"rank={rk} predicted={pr}")
                        shown += 1
                rows.append(dict(n=n, share=sh, nested=ne, rest_single=rs,
                                 actual=int(actual), predicted=int(pr)))
        print(f"  n={n} done", flush=True)
    print(f"\n### criterion as a predictor: {bad} disagreements out of {len(rows)}")
    if bad == 0:
        print("### EXACT at n = 4..%d. The criterion is verified on every "
              "configuration." % args.max_enum)

    import pandas as pd
    D = pd.DataFrame(rows)
    print("\n### case frequencies and identifiability (enumeration)")
    print(D.groupby(["share", "nested", "rest_single"])["actual"]
           .agg(["mean", "size"]).round(4).to_string())

    # ------------------------------------------------------------- sampling
    print(f"\n--- random sampling, n = 7..40, {args.samples} draws each")
    rng = np.random.default_rng(2026)
    srows = []
    for n in [7, 8, 9, 10, 12, 15, 20, 30, 40]:
        for _ in range(args.samples):
            T = build_tree(random_tree(n, rng), "random", rng)
            edges = [(T.parent[v], v) for v in T.nonroot]
            i, j = rng.choice(len(edges), size=2, replace=False)
            e1, e2 = edges[i], edges[j]
            a1, a2 = e1[1], e2[1]
            sh, ne, rs = classify(T, a1, a2)
            rk = rank_of(jacobian_k2(T, (a1, a2)))
            srows.append(dict(n=n, share=sh, nested=ne, rest_single=rs,
                              actual=int(rk < 4), predicted=int(predict(sh, ne, rs))))
    S = pd.DataFrame(srows)
    bad_s = int((S.actual != S.predicted).sum())
    print(f"  criterion disagreements on samples: {bad_s}/{len(S)}")
    agg = S.groupby("n").agg(p_nonident=("actual", "mean"),
                             p_share=("share", "mean"),
                             p_nested=("nested", "mean")).round(4)
    agg["n_times_p"] = (agg.index * agg.p_nonident).round(3)
    print(agg.to_string())
    print("\n  n * P(non-identifiable) flattening means the rate is ~ c/n;")
    print("  the share-vertex case alone should contribute ~ (edges counted)")
    print("  For a binary tree: 2n-3 edges, sum over edges of (deg-1) = 2*(2n-3)-... ;")
    print("  the sibling pairs number n-1 and the parent-child pairs 2n-3, so")
    print("  P(share) ~ (3n)/C(2n-3,2) ~ 3n/(2n^2) = 3/(2n).")
    S.to_csv(os.path.join(OUT, "k2_sampling.csv"), index=False)
    print("\nartifacts ->", os.path.join(OUT, "k2_sampling.csv"))


if __name__ == "__main__":
    main()
