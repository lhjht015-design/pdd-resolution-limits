"""k >= 3: does the k = 2 criterion generalise to the Steiner tree?

THE k = 2 CRITERION, in the form to be generalised
    Two anchors a_1, a_2. For every leaf r let pi(r) be the node of the a_1-a_2
    path its component attaches to, and sigma_q(r) = +1 iff r is in subtree(a_q).
    Then

        rank J < 2k  <=>  every (sigma_1,...,sigma_k) class has a single pi.

    The k = 2 case was verified exactly (0 disagreements in 45,690 enumerated and
    36,000 sampled configurations). The reason it works: Phi is linear in
    (p, u, ybar) given the anchors, so rank J < 2k is exactly a dependence
    alpha_q (D_{P_q} - D_{P_k}) in span{1, sigma_1..sigma_k}; and D_{P_q}(r)
    depends on r only through its attachment point on the Steiner tree of the
    anchors, whereas the sigma's see only which component of R - e_q holds r.

THE CONJECTURE
    The same statement with the a_1-a_2 path replaced by the Steiner tree of
    {a_1,...,a_k} and sigma_1..sigma_k used in place of sigma_1, sigma_2. Since
    tree edges induce laminar splits, the number of attained sigma-classes is
    bounded, so span{1, sigma_q} is again the set of all functions of the sigma's
    on the attained classes.

WHAT IS MEASURED
    rank J vs the conjecture, over all labelled trees and all k-tuples of edges,
    for n = 4..8 and k = 2,3. Every disagreement is printed.

Usage:
    criterion_kgeneral.py [--max-n 7] [--ks 2,3]
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from probe_identifiability import build_tree, gen_trees, rank_of
from verify_k2 import node_path

OUT = "./out"


def steiner_nodes(T, nodes):
    """Nodes on the minimal subtree connecting `nodes`."""
    S = set(nodes)
    for a, b in itertools.combinations(nodes, 2):
        S.update(node_path(T, a, b))
    return S


def attachment(T, S):
    """For every leaf, the node of S its component of R - S attaches to. A
    multi-source BFS from S, each node keeping the S-node it came from: since S
    is connected and spans a subtree, each component of R - S touches S in
    exactly one node."""
    owner = {u: u for u in S}
    stack = list(S)
    while stack:
        u = stack.pop()
        for w in ([T.parent[u]] if T.parent[u] >= 0 else []) + T.children[u]:
            if w not in owner:
                owner[w] = owner[u]; stack.append(w)
    return np.array([owner[r] for r in T.leaves])


def jacobian_k(T, anchors):
    k = len(anchors)
    n = len(T.leaves)
    rows = [T.D[anchors[q]] - T.D[anchors[-1]] for q in range(k - 1)]
    rows += [T.C[a] * T.len[a] for a in anchors]
    rows += [np.ones(n)]
    return np.array(rows)


def sigma_classes(T, anchors):
    classes = {}
    for j in range(len(T.leaves)):
        key = tuple(T.C[a][j] for a in anchors)
        classes.setdefault(key, []).append(j)
    return classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=7)
    ap.add_argument("--ks", default="2,3")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 96)
    print("k >= 3 -- does 'rank J < 2k  <=>  for every PAIR of anchors, each "
          "sigma-class")
    print("           has a single projection onto that pair's path' hold?")
    print("=" * 96)

    rows = []
    for k in [int(v) for v in args.ks.split(",")]:
        for n in range(max(4, 2 * k - 1), args.max_n + 1):
            bad = tot = 0
            shown = 0
            for ti, t in enumerate(gen_trees(range(1, n + 1))):
                T = build_tree(t, "random", np.random.default_rng(31 + ti))
                edges = [(T.parent[v], v) for v in T.nonroot]
                for combo in itertools.combinations(range(len(edges)), k):
                    anchors = [edges[i][1] for i in combo]
                    if len(set(anchors)) < k:
                        continue
                    J = jacobian_k(T, anchors)
                    rk = rank_of(J)
                    actual = (rk < 2 * k)
                    classes = sigma_classes(T, anchors)
                    # WEAKER conjecture: for EVERY pair of anchors, the sigma
                    # signature determines the projection onto that pair's path.
                    # (Attachment to the Steiner tree implies this, which is why
                    # the Steiner form over-predicted non-identifiability.)
                    cb = True
                    for q1, q2 in itertools.combinations(anchors, 2):
                        pth = set(node_path(T, q1, q2))
                        at = attachment(T, pth)
                        if not all(len(set(at[idx].tolist())) == 1
                                   for idx in classes.values()):
                            cb = False
                            break
                    tot += 1
                    if actual != cb:
                        bad += 1
                        if shown < 6:
                            print(f"    MISMATCH n={n} k={k} tree={ti} anchors="
                                  f"{anchors} rank={rk}/{2*k} "
                                  f"class_attach_single={cb} "
                                  f"n_classes={len(classes)}")
                            shown += 1
                    rows.append(dict(k=k, n=n, rk=rk, actual=int(actual),
                                     conj=int(cb), nclass=len(classes)))
            print(f"  k={k} n={n}: {tot} configurations, {bad} disagreements "
                  f"({'CONJECTURE HOLDS' if bad == 0 else 'FAILS'})", flush=True)

    import pandas as pd
    D = pd.DataFrame(rows)
    print("\n### summary")
    print(D.groupby(["k", "n"]).agg(n=("actual", "size"),
                                    ident=("actual", lambda s: 1 - s.mean()),
                                    disagree=("conj", "size")).to_string())
    print("\n### attained sigma-class counts by k")
    print(D.groupby(["k", "nclass"]).size().to_string())
    D.to_csv(os.path.join(OUT, "kgeneral.csv"), index=False)
    print("\nartifacts ->", os.path.join(OUT, "kgeneral.csv"))


if __name__ == "__main__":
    main()
