"""B = P L M^T exactly, and what that says about sign coherence.

THE DERIVATION (verified below, not assumed)
    Let edge e split the tree into two sides and write side(u,e) = +1 if node u
    lies on the child side of e (i.e. in subtree(child(e))), -1 otherwise. Then
    e separates r from a iff side(r,e)*side(a,e) = -1, so

        chi_{r,a}(e) := [e on the path from r to a] = (1 - side(r,e)*side(a,e))/2.

    B's entry is a four-term combination of such indicators,

        B[row,q] = [chi_{r,a_q} - chi_{r0,a_q}] - [chi_{r,a_k} - chi_{r0,a_k}],

    and substituting the formula above the `1`s cancel and the whole thing
    FACTORS:

        coeff of ell_e in B[row,q]
             = ( side(r,e) - side(r0,e) ) * ( side(a_k,e) - side(a_q,e) ) / 2.

    So with

        P[row, e] = ( side(r,e) - side(r0,e) ) / 2       in {-1,0,+1}
        M[e,   q] =   side(a_k,e) - side(a_q,e)          in {-2,0,+2}
        L         = diag(ell_e)

    we have B = P L M^T as an identity. P is +-1 exactly on edges lying on the
    leaf-pair path (r, r0) -- a LEAF-PAIR PATH INCIDENCE matrix. M is +-2
    exactly on edges lying on the anchor-pair path (a_k, a_q) -- a NODE-PAIR
    PATH INCIDENCE matrix. Both are purely combinatorial; only L carries the
    branch lengths.

WHY IT MATTERS
    Cauchy-Binet then gives, for an r x r minor,

        det(B_{I,J}) = sum_{|E|=r} det(P_{I,E}) * (prod_{e in E} ell_e)
                       * det(M_{E,J}),

    so the coefficient of each monomial in the ell's is a PRODUCT OF TWO
    DETERMINANTS OF PATH-INCIDENCE SUBMATRICES. If both determinants are always
    in {-1,0,+1} and all nonzero products have one sign, the minor is sign
    coherent -- which is exactly what the symbolic experiment observed and could
    not explain.

WHAT THIS SCRIPT DOES
    1. verifies B = P L M^T numerically on every configuration it visits;
    2. verifies the coefficient formula entrywise;
    3. checks total unimodularity of P and of M on all square submatrices;
    4. computes the Cauchy-Binet coefficients c_E for every maximal minor and
       reports whether they are sign-coherent, and whether |c_E| is always 1.

Usage:
    pathsplit.py [--max-n 7] [--ks 2,3]
"""

from __future__ import annotations

import argparse
import itertools
import os
from collections import Counter

import numpy as np

from probe_identifiability import build_tree, gen_trees, rank_of

OUT = "./out"


def side(T, u, v):
    """+1 if node u lies in subtree(v), else -1. v != root."""
    x = u
    while x >= 0:
        if x == v:
            return 1
        x = T.parent[x]
    return -1


def build_P_M(T, anchors, classes):
    """P (rows x edges) and M (edges x (k-1)) per the derivation."""
    edges = [v for v in T.nonroot]                 # edge into v
    k = len(anchors)
    rows, rowmeta = [], []
    for key, jj in sorted(classes.items()):
        if len(jj) <= 1:
            continue
        j0 = jj[0]
        for j in jj[1:]:
            r, r0 = T.leaves[j], T.leaves[j0]
            rows.append([(side(T, r, e) - side(T, r0, e)) / 2 for e in edges])
            rowmeta.append((key, j, j0))
    P = np.array(rows, dtype=float) if rows else np.zeros((0, len(edges)))
    # M' = M/2 is the +-1 node-pair path incidence; M carries a factor 2
    # because side() returns +-1 rather than an indicator.
    Mp = np.array([[(side(T, anchors[k - 1], e)
                     - side(T, anchors[q], e)) / 2
                    for q in range(k - 1)] for e in edges], dtype=float)
    return P, Mp, edges, rowmeta


def build_B(T, anchors, classes):
    """B exactly as defined in probe 3: within-class contrasts of the k-1
    difference vectors D_{a_q} - D_{a_k}."""
    k = len(anchors)
    Dm = np.array([T.D[anchors[q]] - T.D[anchors[k - 1]] for q in range(k - 1)])
    rows = []
    for key, jj in sorted(classes.items()):
        if len(jj) <= 1:
            continue
        j0 = jj[0]
        for j in jj[1:]:
            rows.append(Dm[:, j] - Dm[:, j0])
    return np.array(rows, dtype=float) if rows else np.zeros((0, k - 1))


def is_TU(Mx, tol=1e-9):
    """Total unimodularity check: every square submatrix has det in {-1,0,1}."""
    m, n = Mx.shape
    r = min(m, n)
    worst = 0.0
    for size in range(1, r + 1):
        for ri in itertools.combinations(range(m), size):
            for ci in itertools.combinations(range(n), size):
                d = np.linalg.det(Mx[np.ix_(ri, ci)])
                if abs(d) > tol and abs(abs(d) - 1) > 1e-7:
                    return False, float(d), (ri, ci)
                worst = max(worst, abs(d))
    return True, worst, None


def sigma_classes(T, anchors):
    c = {}
    for j in range(len(T.leaves)):
        c.setdefault(tuple(int(T.C[a][j]) for a in anchors), []).append(j)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--tu-n", type=int, default=6, help="max n for the TU scan")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 96)
    print("B = P L M^T -- verifying the factorisation, then its Cauchy-Binet "
          "consequences")
    print("=" * 96)

    rng = np.random.default_rng(9)
    worst_F = 0.0
    worst_coef = 0.0
    tu_bad = 0
    tu_tested = 0
    cb_signs = Counter()
    cb_mag = Counter()
    nconf = 0
    shown = 0

    for k in [int(s) for s in args.ks.split(",")]:
        for n in range(max(4, k + 1), args.max_n + 1):
            for ti, t in enumerate(gen_trees(range(1, n + 1))):
                T = build_tree(t, "random", np.random.default_rng(51 + ti))
                E = [(T.parent[v], v) for v in T.nonroot]
                for combo in itertools.combinations(range(len(E)), k):
                    anchors = [E[i][1] for i in combo]
                    if len(set(anchors)) < k:
                        continue
                    cls = sigma_classes(T, anchors)
                    if len(cls) != k + 1:
                        continue
                    B = build_B(T, anchors, cls)
                    if B.shape[0] < (k - 1):
                        continue
                    P, M, edges, _ = build_P_M(T, anchors, cls)
                    if P.shape[0] != B.shape[0]:
                        print("  ROW COUNT MISMATCH", n, k, anchors,
                              B.shape, P.shape)
                        continue
                    L = np.diag([T.len[e] for e in edges])
                    # M is stored as (edges x (k-1)), so B = P L M. (Writing it
                    # as P L M^T corresponds to storing M transposed.)
                    Bf = 2.0 * (P @ L @ M)
                    worst_F = max(worst_F, np.abs(Bf - B).max())

                    # entrywise coefficient check
                    for i in range(B.shape[0]):
                        for q in range(B.shape[1]):
                            c_actual = B[i, q]
                            c_pred = 2.0 * sum(T.len[e] * P[i, ei] * M[ei, q]
                                               for ei, e in enumerate(edges))
                            worst_coef = max(worst_coef, abs(c_actual - c_pred))

                    # TU scan (bounded, it is exponential in the size)
                    if n <= args.tu_n:
                        for name, Mx in (("P", P), ("M", M)):
                            if Mx.size == 0 or min(Mx.shape) == 0:
                                continue
                            ok, worst, wit = is_TU(Mx)
                            tu_tested += 1
                            if not ok:
                                tu_bad += 1
                                if shown < 5:
                                    print(f"  NOT TU: {name} n={n} k={k} "
                                          f"anchors={anchors} det={worst} at "
                                          f"{wit}")
                                    shown += 1

                    # Cauchy-Binet coefficients for every maximal minor
                    size = k - 1
                    if B.shape[0] >= size:
                        for sel in itertools.combinations(range(B.shape[0]), size):
                            rows_s = list(sel)
                            # the two determinant factors
                            dp, dm = {}, {}
                            for Eset in itertools.combinations(range(len(edges)),
                                                              size):
                                dp[Eset] = np.linalg.det(P[np.ix_(rows_s, Eset)])
                                dm[Eset] = np.linalg.det(M[np.ix_(Eset,
                                                                 range(size))])
                            terms = []
                            for Eset in dp:
                                c = dp[Eset] * dm[Eset]
                                if abs(c) > 1e-7:
                                    terms.append(c)
                            if not terms:
                                cb_signs["identically zero"] += 1
                                continue
                            signs = {1 if c > 0 else -1 for c in terms}
                            cb_signs["sign-coherent" if len(signs) == 1
                                     else "MIXED SIGN"] += 1
                            mags = {round(abs(c)) for c in terms}
                            vals = {round(c) for c in terms}
                            cb_mag[tuple(sorted(mags))] += 1
                            if len(mags) != 1:
                                cb_signs["|c_E| NOT constant"] += 1
                            elif len(vals) == 1:
                                cb_signs["ALL c_E EQUAL"] += 1
                            elif len(signs) == 1:
                                cb_signs["same sign, differing magnitude"] += 1
                    nconf += 1

    print("\n[FACTORISATION]")
    print(f"  configurations visited                       : {nconf}")
    print(f"  max |P L M^T - B|                            : {worst_F:.3e}")
    print(f"  max entrywise |coefficient formula - B|      : {worst_coef:.3e}")
    print("\n[TOTAL UNIMODULARITY]  (all square submatrices, n <= "
          f"{args.tu_n})")
    print(f"  matrices tested : {tu_tested}   failures : {tu_bad}")
    print("\n[CAUCHY-BINET]  coefficients c_E = det(P)*det(M) per minor")
    for k_, v in cb_signs.items():
        print(f"  {k_:18s} {v}")
    print("  |c_E| values seen:", dict(cb_mag))

    print("\n" + "=" * 96)
    ok = worst_F < 1e-9 and worst_coef < 1e-9
    print("FACTORISATION:", "EXACT" if ok else "FAILS")
    if cb_signs.get("MIXED SIGN", 0) == 0 and cb_signs.get("sign-coherent", 0) > 0:
        print("SIGN COHERENCE: every maximal minor has all its nonzero Cauchy-")
        print("Binet coefficients of one sign. This is the mechanism behind the")
        print("Class A / Class B split -- and it is a statement about two path-")
        print("incidence matrices, not about branch lengths.")


if __name__ == "__main__":
    main()
