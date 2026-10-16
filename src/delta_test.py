"""The alignment as a division-free Delta test, plus the per-component count identity.

TWO THINGS ARE CHECKED HERE.

(i) THE PER-COMPONENT DELETION COUNT. Remove the anchor edges A from T and let
    C_0, ..., C_c be the components. Every demand edge of D_I has both endpoints
    inside one sigma-class, hence inside one C_alpha, so on the augmented graph
    restricted to C_alpha a spanning tree needs

        (|C_alpha| - 1 - |E AND E(C_alpha)|) + |D_{I,alpha}|  =  |C_alpha| - 1,

    i.e.  |E AND E(C_alpha)| = |D_{I,alpha}|  for EVERY common basis E. Since the
    right side depends only on I, the number of deleted edges inside each component
    is FIXED for fixed I even though E itself varies. That is what removes the
    E-dependence of the permutation signs.

(ii) THE ALIGNMENT, DIVISION-FREE. Fix a reference common basis E_0 and test

        Delta(E, E_0) = det U_J(E)*det U_I(E_0) - det U_J(E_0)*det U_I(E) = 0

    for every other common basis E. All determinants are in {-1,+1}, so this is
    zero exactly when det U_J(E)/det U_I(E) equals det U_J(E_0)/det U_I(E_0);
    multiplying instead of dividing avoids any zero or sign implementation trap.

    U_I(E) = [ R_Ebar | Q_I ]  and  U_J(E) = [ R_Ebar | Q'_J ] with the SAME R
    (one reference tree), so det R and (-1)^chi(E) are common factors and cancel.

Usage:
    delta_test.py [--max-n 7] [--ks 2,3]
"""

from __future__ import annotations

import argparse
import itertools
import os
from collections import Counter

from probe_identifiability import build_tree, gen_trees
from pathsplit import build_P_M, build_B, sigma_classes
from sign_search import bareiss_det
from complement import incidence

OUT = "./out"


def components_without(N, kept_edges, removed_nodes=None):
    """Connected components of the graph on nodes 0..N-1 with `kept_edges`.
    Edges are (a, b) unordered."""
    par = list(range(N))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in kept_edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    return [find(i) for i in range(N)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=7)
    ap.add_argument("--ks", default="2,3")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 96)
    print("Delta(E,E0) = 0 test, and the per-component deletion-count identity")
    print("=" * 96)

    cnt = Counter()
    delta = Counter()
    maxabs = 0
    witness = None
    nrows = 0

    for k in [int(s) for s in args.ks.split(",")]:
        for n in range(max(4, k + 1), args.max_n + 1):
            for ti, t in enumerate(gen_trees(range(1, n + 1))):
                T = build_tree(t, "equal", None)
                N = len(T.parent)
                tn = list(T.nonroot)
                m = len(tn)
                pos = {v: j for j, v in enumerate(tn)}
                Te = [(T.parent[v], v) for v in tn]
                R = incidence(N, 0, Te)
                E = [(T.parent[v], v) for v in T.nonroot]
                for combo in itertools.combinations(range(len(E)), k):
                    anchors = [E[i][1] for i in combo]
                    if len(set(anchors)) < k:
                        continue
                    cls = sigma_classes(T, anchors)
                    if len(cls) != k + 1:
                        continue
                    B = build_B(T, anchors, cls)
                    r = k - 1
                    if B.shape[0] < r:
                        continue
                    P, M, edges, rowmeta = build_P_M(T, anchors, cls)
                    if P.shape[0] != B.shape[0]:
                        continue

                    # ---- components of T minus the anchor edges
                    anchor_e = set(anchors)
                    kept_all = [(T.parent[v], v) for v in tn if v not in anchor_e]
                    comp = components_without(N, kept_all)

                    Dp = [(T.leaves[rowmeta[i][2]], T.leaves[rowmeta[i][1]])
                          for i in range(P.shape[0])]
                    Dm = [(anchors[q], anchors[k - 1]) for q in range(r)]
                    QJ = incidence(N, 0, Dm)

                    for I in itertools.combinations(range(P.shape[0]), r):
                        DI = [Dp[i] for i in I]
                        QI = incidence(N, 0, DI)
                        # per-component demand EDGE counts for this I. Counting
                        # endpoints here instead of edges doubles every entry and
                        # makes the identity look false.
                        dem_of_comp = Counter(comp[e[0]] for e in DI)
                        shared = []
                        for Es in itertools.combinations(range(m), r):
                            cols = [tn[e] for e in Es]
                            dp = bareiss_det([[int(P[i][pos[c]]) for c in cols]
                                              for i in I])
                            if dp == 0:
                                continue
                            dm = bareiss_det([[int(M[pos[c]][q]) for q in range(r)]
                                              for c in cols])
                            if dm == 0:
                                continue
                            kept = [j for j in range(m) if j not in set(Es)]
                            UI = [[row[j] for j in kept] + [QI[i][c] for c in range(r)]
                                  for i, row in enumerate(R)]
                            UJ = [[row[j] for j in kept] + [QJ[i][c] for c in range(r)]
                                  for i, row in enumerate(R)]
                            dUI = bareiss_det(UI)
                            dUJ = bareiss_det(UJ)
                            if dUI == 0 or dUJ == 0:
                                continue
                            # ---- (i) per-component deletion count
                            del_of_comp = Counter(comp[tn[e]] for e in Es)
                            same = all(del_of_comp[c] == dem_of_comp[c]
                                       for c in dem_of_comp)
                            cnt["count identity holds" if same
                                else "COUNT IDENTITY FAILS"] += 1
                            if not same and witness is None:
                                witness = ("count", n, k, anchors, I, Es,
                                           dict(del_of_comp), dict(dem_of_comp))
                            shared.append((Es, dUI, dUJ))
                        if not shared:
                            continue
                        nrows += 1
                        Es0, u0I, u0J = shared[0]
                        loc = 0
                        for Es, dUI, dUJ in shared[1:]:
                            d = dUJ * u0I - u0J * dUI
                            loc = max(loc, abs(d))
                            if d != 0 and witness is None:
                                witness = ("delta", n, k, anchors, I, Es0, Es,
                                           u0I, u0J, dUI, dUJ)
                        maxabs = max(maxabs, loc)
                        delta["Delta == 0" if loc == 0 else "DELTA NONZERO"] += 1
                        delta[f"common_bases={len(shared)}"] += 0
            print(f"  k={k} n={n} done ({nrows} row sets)", flush=True)

    print("\n### (i) per-component deletion count:  |E AND E(C_a)| = |D_I AND E(C_a)|")
    for kk, v in cnt.items():
        print(f"    {kk:24s} {v}")
    if witness and witness[0] == "count":
        print("    witness:", witness)

    print("\n### (ii) Delta(E, E0) over common bases, E0 = the first one")
    for kk, v in delta.items():
        if kk.startswith("Delta") or kk.startswith("DELTA"):
            print(f"    {kk:24s} {v}")
    print(f"    row sets tested        : {nrows}")
    print(f"    max |Delta| observed   : {maxabs}")
    print("\n### distribution of the number of common bases per row set")
    import re
    print("    (not tabulated here; see the counter)")

    print("\n" + "=" * 96)
    ok1 = cnt.get("COUNT IDENTITY FAILS", 0) == 0
    ok2 = delta.get("DELTA NONZERO", 0) == 0
    print("COUNT IDENTITY:", "CONFIRMED" if ok1 else "REFUTED")
    print("ALIGNMENT (Delta == 0):", "CONFIRMED" if ok2 else "REFUTED")
    if ok1 and ok2:
        print("\nThe per-component deletion count is fixed by I alone, and the")
        print("alignment holds in division-free form on every common basis. The")
        print("next step is the component-wise elimination, not more enumeration.")


if __name__ == "__main__":
    main()
