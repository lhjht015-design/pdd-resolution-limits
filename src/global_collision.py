"""Global support-collision oracle for PDD (first probe of the "shape" question).

The question this answers (NOT a theorem, an oracle):
    For a fixed reference tree T and placement count k, do two DISTINCT placement
    supports A != A' ever produce the same observed mixture distance vector?

Geometry (from polytope.py, already validated):
    For a support A (a set of k anchor nodes) the observation set is
        Image(A) = conv(v^A) + span{1},
        v^A = { D_{a_q} (q=1..k) }  ∪  { D_{a_q} + C_{a_q} ell_{a_q} (q=1..k) },
    i.e. the convex hull of 2k endpoints plus the free ybar offset span{1}.
    Quotienting span{1} (centering by H = I - 11^T/n) leaves two convex hulls.

Classification (the user's G0..G3):
    G0  same support, different parameters  -> fixed-support idf (already solved).
    G1  same edge SET, permuted labels       -> quotient out (we compare sets).
    G2  images meet only on the closure boundary (some abundance/position at 0/1).
    G3  images meet at a strictly interior point (p > 0, x in (0,1)) -> genuine
        global non-identifiability.

The intersection is decided by linear programming (scipy linprog, HiGHS):
    feasible   : exists lamA, lamB >= 0, sum=1,  V_A^T lamA - V_B^T lamB in span{1}
    interior   : the same with lamA > 0, lamB > 0 componentwise (maximize the
                 minimum entry).

Usage:
    global_collision.py [--max-n 6] [--ks 2,3]
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
from scipy.optimize import linprog

from probe_identifiability import build_tree, gen_trees
from polytope import endpoints

OUT = "./out"
TOL = 1e-8


# ------------------------------------------------------------------ LP oracle
def hull_intersect(vA, vB):
    """vA, vB: (2k, n) ENDPOINT rows (uncentered). Decide whether
    conv(vA)+span{1} meets conv(vB)+span{1}, and whether the meeting can be made
    strictly interior (all 2k coordinates of each convex combination > 0).

    Returns (intersects, t_star) with
        t_star > 0  -> interior intersection (G3)
        t_star == 0 -> boundary-only intersection (G2)
        t_star < 0  -> no intersection.
    """
    m = vA.shape[0]          # 2k
    n = vA.shape[1]
    # variables: lamA (m), lamB (m), s (1, the span{1} offset difference)
    # equality: vA^T lamA - vB^T lamB - s*1 = 0 ;  sum lamA = 1 ; sum lamB = 1
    Aeq = np.zeros((n + 2, 2 * m + 1))
    Aeq[:n, :m] = vA.T
    Aeq[:n, m:2 * m] = -vB.T
    Aeq[:n, 2 * m] = -1.0                      # -s * 1
    Aeq[n, :m] = 1.0
    Aeq[n + 1, m:2 * m] = 1.0
    beq = np.zeros(n + 2)
    beq[n] = 1.0
    beq[n + 1] = 1.0

    # -- feasibility
    res = linprog(c=np.zeros(2 * m + 1), A_eq=Aeq, b_eq=beq,
                  bounds=[(0, None)] * (2 * m) + [(None, None)],
                  method="highs")
    if not res.success:
        return False, -1.0

    # -- strict interiority: maximize t s.t. lamA >= t, lamB >= t
    c = np.zeros(2 * m + 2)
    c[-1] = -1.0                                # minimize -t
    Aeq2 = np.zeros((n + 2, 2 * m + 2))
    Aeq2[:n, :m] = vA.T
    Aeq2[:n, m:2 * m] = -vB.T
    Aeq2[:n, 2 * m] = -1.0
    Aeq2[n, :m] = 1.0
    Aeq2[n + 1, m:2 * m] = 1.0
    A_ub = np.zeros((2 * m, 2 * m + 2))
    for i in range(m):
        A_ub[i, i] = -1.0                      # -lamA_i + t <= 0
        A_ub[i, -1] = 1.0
        A_ub[m + i, m + i] = -1.0              # -lamB_i + t <= 0
        A_ub[m + i, -1] = 1.0
    res2 = linprog(c=c, A_ub=A_ub, b_ub=np.zeros(2 * m),
                   A_eq=Aeq2, b_eq=beq,
                   bounds=[(0, None)] * (2 * m + 1) + [(0, None)],
                   method="highs")
    if not res2.success:
        return True, 0.0
    return True, -res2.fun


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 96)
    print("PDD GLOBAL SUPPORT-COLLISION ORACLE (G0..G3 classification)")
    print("  G2 = boundary-only meeting, G3 = strictly interior meeting")
    print("=" * 96)

    rows = []
    catalogue = []
    for k in [int(s) for s in args.ks.split(",")]:
        for n in range(max(4, k + 1), args.max_n + 1):
            tot = g2 = g3 = 0
            n_trees = 0
            for ti, t in enumerate(gen_trees(range(1, n + 1))):
                T = build_tree(t, "random", np.random.default_rng(61 + ti))
                nodes = list(T.nonroot)
                for Aset in itertools.combinations(nodes, k):
                    for Bset in itertools.combinations(nodes, k):
                        if Aset >= Bset:        # unordered, once per pair
                            continue
                        vA, _ = endpoints(T, list(Aset))
                        vB, _ = endpoints(T, list(Bset))
                        intersects, tstar = hull_intersect(vA, vB)
                        if not intersects:
                            continue
                        tot += 1
                        if tstar > TOL:
                            g3 += 1
                            if len(catalogue) < args.show:
                                catalogue.append((n, k, tuple(Aset), tuple(Bset),
                                                  round(tstar, 5)))
                        else:
                            g2 += 1
                n_trees += 1
            print(f"\n[n={n} k={k}] trees={n_trees}  "
                  f"colliding_support_pairs={tot}  G2={g2}  G3={g3}")
            rows.append(dict(n=n, k=k, colliding=tot, g2=g2, g3=g3))

    print("\n" + "=" * 96)
    print("GENUINE INTERIOR COLLISIONS (G3) -- first few")
    print("=" * 96)
    for (n, k, A, B, t) in catalogue:
        print(f"  n={n} k={k}  A={A}  B={B}  min_coord={t}")
    if not catalogue:
        print("  (none found at these sizes)")

    import pandas as pd
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "global_collision.csv"), index=False)
    print("\nartifacts ->", os.path.join(OUT, "global_collision.csv"))


if __name__ == "__main__":
    main()
