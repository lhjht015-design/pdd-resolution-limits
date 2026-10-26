"""Cross-check of the necessity proof's algebraic reduction (Theorem 7).

The necessity proof (main text, Theorem 7) reduces an interior collision of the two
supports {a,s} and {b,s} to a relation among the tree's cut vectors:

    sum_{v != r0} c_v C_v = s * 1 ,   with
    c_v = ell_v * ( -alpha 1[a in subtree(v)] + beta 1[b in subtree(v)]
                    - (beta-alpha) 1[s in subtree(v)] )
          + u_a 1[v=a] - u_b 1[v=b] + w 1[v=s] ,

where alpha, beta in (0,1), u_a = alpha x_a ell_a, u_b = beta x_b ell_b, and
w = [(1-alpha) x_s - (1-beta) x_s'] ell_s.  By the cut-vector *dependence* lemma, this
relation holds iff the ancestor sums F(ell) = sum_{v ancestor of ell} c_v are constant
over leaves (the earlier "support = {u', v'}" reading followed from an incorrect lemma
and is not asserted here).

This script takes every interior collision found by the LP oracle of
global_collision.py, re-extracts the algebraic parameters (alpha, beta, u_a, u_b, w)
directly from the LP solution, and verifies to machine precision that

  1. the relation sum_v c_v C_v = s * 1 holds,
  2. the ancestor sums F(ell) = sum_{v ancestor of ell} c_v are constant over leaves,
  3. alpha != beta (the alpha = beta branch degenerates to a boundary contact).

Agreement on every collision cross-checks the algebraic reduction against the
independent geometric (LP) oracle, the two routes of the proof.

Usage:
    reduction_crosscheck.py [--ns 5,6,7] [--tol 1e-7] [--seed 0]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.optimize import linprog

from probe_identifiability import build_tree
from polytope import endpoints
from shared_edge_regression import unlabeled_shapes


def interior_lp(vA, vB, n):
    """Maximize the minimum convex coefficient of a common point of
    conv(vA) + span{1} and conv(vB) + span{1}; return the full LP result (None if
    infeasible).  Variables: lamA (m), lamB (m), s, t."""
    m = vA.shape[0]
    c = np.zeros(2 * m + 2)
    c[-1] = -1.0                                # minimize -t
    Aeq = np.zeros((n + 2, 2 * m + 2))
    Aeq[:n, :m] = vA.T
    Aeq[:n, m:2 * m] = -vB.T
    Aeq[:n, 2 * m] = -1.0                       # -s * 1
    Aeq[n, :m] = 1.0
    Aeq[n + 1, m:2 * m] = 1.0
    beq = np.zeros(n + 2)
    beq[n] = 1.0
    beq[n + 1] = 1.0
    Aub = np.zeros((2 * m, 2 * m + 2))
    for i in range(m):
        Aub[i, i] = -1.0                        # -lamA_i + t <= 0
        Aub[i, -1] = 1.0
        Aub[m + i, m + i] = -1.0                # -lamB_i + t <= 0
        Aub[m + i, -1] = 1.0
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(2 * m), A_eq=Aeq, b_eq=beq,
                  bounds=[(0, None)] * (2 * m + 1) + [(0, None)], method="highs")
    return res if res.success else None


def on_path(T, v, x):
    """1 if v is on the path root -> x, i.e. x lies in subtree(v)."""
    u = x
    while u >= 0:
        if u == v:
            return 1.0
        u = T.parent[u]
    return 0.0


def verify(T, a, b, s, tol=1e-7):
    """Run the LP and check the four assertions of the reduction.  Returns a dict of
    the four flags (all True on a genuine Theorem-7 collision)."""
    vA, _ = endpoints(T, [a, s])
    vB, _ = endpoints(T, [b, s])
    res = interior_lp(vA, vB, len(T.leaves))
    if res is None or res.x[-1] <= tol:
        return None                              # not an interior collision
    lam = res.x[0:4]
    mu = res.x[4:8]
    soffs = res.x[8]

    alpha = lam[0] + lam[2]
    beta = mu[0] + mu[2]
    u_a = lam[2] * T.len[a]
    u_b = mu[2] * T.len[b]
    w = (lam[3] - mu[3]) * T.len[s]

    # coefficients c_v for v != root
    c = {}
    for v in T.nonroot:
        cv = T.len[v] * (-alpha * on_path(T, v, a) + beta * on_path(T, v, b)
                         - (beta - alpha) * on_path(T, v, s))
        if v == a:
            cv += u_a
        if v == b:
            cv -= u_b
        if v == s:
            cv += w
        c[v] = cv

    # (1) relation holds: sum_v c_v C_v == s * 1
    lhs = np.zeros(len(T.leaves))
    for v, cv in c.items():
        lhs += cv * T.C[v]
    rel_ok = np.abs(lhs - soffs * np.ones(len(T.leaves))).max() < 1e-6

    # (2) ancestor sums F(ell) = sum_{v ancestor of ell} c_v are constant over leaves
    F = np.zeros(len(T.leaves))
    for li, ell in enumerate(T.leaves):
        for v, cv in c.items():
            if on_path(T, v, ell):
                F[li] += cv
    const_ok = F.max() - F.min() < 1e-6

    # (3) alpha != beta
    alpha_beta_ok = abs(alpha - beta) > 1e-6

    return dict(rel=rel_ok, const=const_ok, alpha_beta=alpha_beta_ok,
                alpha=alpha, beta=beta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5,6,7")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ncol = 0
    bad = {k: 0 for k in ("rel", "const", "alpha_beta")}
    for n in [int(v) for v in args.ns.split(",")]:
        for t in unlabeled_shapes(n):
            T = build_tree(t, "equal", rng)
            for a, b in itertools.combinations(T.nonroot, 2):
                for s in T.nonroot:
                    if s == a or s == b:
                        continue
                    r = verify(T, a, b, s, tol=args.tol)
                    if r is None:
                        continue
                    ncol += 1
                    for k in bad:
                        if not r[k]:
                            bad[k] += 1

    print(f"interior collisions checked: {ncol}")
    for k, v in bad.items():
        print(f"  {k}: {v} violations")
    print("REDUCTION CROSS-CHECK PASSED" if all(v == 0 for v in bad.values())
          else "REDUCTION CROSS-CHECK FAILED")


if __name__ == "__main__":
    main()
