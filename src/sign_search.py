"""ADVERSARIAL SEARCH for a counterexample to the joint sign coherence.

THE CONJECTURE UNDER ATTACK
    For every anchor configuration A, every maximal minor of B = 2 P L M', the
    Cauchy-Binet coefficients

        c_E = det( P[I, E] ) * det( M'[E, :] ),        |I| = |E| = k-1,

    are either all zero (the minor vanishes identically) or ALL OF ONE SIGN. If
    one were positive and another negative, the minor could vanish for some
    positive length vector and the rank would not be topology-determined.

    This is the whole remaining theorem. The searcher's job is to BREAK it, not
    to confirm it: it stops at the first pair (E, E') with c_E and c_E' of
    opposite sign and prints the minimal reproduction.

EXACT ARITHMETIC, DELIBERATELY
    P and M' have entries in {-1,0,+1} and are totally unimodular, so every
    determinant is in {-1,0,+1}. Computing them in floating point would round
    near-zero minors to 0 and hide exactly the kind of near-cancellation that
    would be a counterexample. So determinants are computed EXACTLY with the
    Bareiss fraction-free algorithm over Python ints, and the result is asserted
    to lie in {-1,0,1} -- if it does not, TU has failed and that is itself the
    headline.

SEARCH SHAPE
    small n: full enumeration of labelled rooted binary trees and of anchor
             k-subsets;
    large n: random trees and random anchor subsets, many draws, because the
             enumeration is exponential.
    In both regimes, all (k-1)-subsets I of B's rows and all (k-1)-subsets E of
    the tree's edges are exhausted -- that part is cheap and is where the
    counterexample would live.

Usage:
    sign_search.py [--enum-n 7] [--ks 2,3] [--rand-n 12] [--draws 4000]
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys

from probe_identifiability import build_tree, gen_trees
from pathsplit import build_P_M, build_B, sigma_classes

OUT = "./out"


def bareiss_det(a):
    """Exact integer determinant, fraction-free. Returns None if not square."""
    n = len(a)
    if n == 0:
        return 1
    if any(len(r) != n for r in a):
        return None
    M = [row[:] for row in a]
    sign, prev = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            piv = next((i for i in range(k + 1, n) if M[i][k] != 0), None)
            if piv is None:
                return 0
            M[k], M[piv] = M[piv], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
        for i in range(k + 1, n):
            M[i][k] = 0
    return sign * M[n - 1][n - 1]


def rand_tree(n, rng):
    pool = [("L", i) for i in range(1, n + 1)]
    while len(pool) > 1:
        a = pool.pop(rng.randrange(len(pool)))
        b = pool.pop(rng.randrange(len(pool)))
        pool.append(("I", a, b))
    return pool[0]


def attack(T, anchors, k, report):
    """Return a counterexample dict, or None."""
    cls = sigma_classes(T, anchors)
    if len(cls) != k + 1:
        return None
    B = build_B(T, anchors, cls)
    size = k - 1
    if B.shape[0] < size:
        return None
    P, M, edges, rowmeta = build_P_M(T, anchors, cls)
    if P.shape[0] != B.shape[0]:
        return None
    nedge = len(edges)

    # every maximal minor of B, and its Cauchy-Binet coefficients
    for I in itertools.combinations(range(B.shape[0]), size):
        terms = []
        for E in itertools.combinations(range(nedge), size):
            dp = bareiss_det([[int(P[i][e]) for e in E] for i in I])
            dm = bareiss_det([[int(M[e][q]) for q in range(size)] for e in E])
            if dp is None or dm is None:
                continue
            if abs(dp) > 1 or abs(dm) > 1:
                # TU would be violated; that is a headline in itself
                return dict(kind="TU_violation", dp=dp, dm=dm, I=I, E=E,
                            anchors=anchors, rowmeta=rowmeta, edges=edges)
            c = dp * dm
            if c != 0:
                terms.append((E, c, dp, dm))
        signs = {c for _, c, _, _ in terms}
        if len(signs) > 1:
            pos = next(t for t in terms if t[1] > 0)
            neg = next(t for t in terms if t[1] < 0)
            return dict(kind="MIXED_SIGN", I=I, pos=pos, neg=neg,
                        n_terms=len(terms), anchors=anchors,
                        rowmeta=rowmeta, edges=edges)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum-n", type=int, default=7)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--rand-n", type=int, default=12)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(args.seed)

    print("=" * 96)
    print("ADVERSARIAL SEARCH for a counterexample to joint sign coherence")
    print("=" * 96)
    print("target:  c_E = det(P[I,E]) * det(M'[E,:])  all of one sign, or all zero")
    print("stop at the first (E, E') with opposite signs\n")

    tested = 0
    minors = 0
    found = None

    # ---- full enumeration on small n
    for k in [int(s) for s in args.ks.split(",")]:
        for n in range(max(4, k + 1), args.enum_n + 1):
            for ti, t in enumerate(gen_trees(range(1, n + 1))):
                T = build_tree(t, "equal", None)
                E = [(T.parent[v], v) for v in T.nonroot]
                for combo in itertools.combinations(range(len(E)), k):
                    anchors = [E[i][1] for i in combo]
                    if len(set(anchors)) < k:
                        continue
                    tested += 1
                    r = attack(T, anchors, k, True)
                    if r:
                        found = (n, k, T, r)
                        break
                if found:
                    break
            if found:
                break
            print(f"  enum k={k} n={n} done ({tested} configs so far)",
                  flush=True)
        if found:
            break

    # ---- random search on larger n, larger k
    if not found:
        ks = sorted({int(s) for s in args.ks.split(",")} | {4, 5})
        for k in ks:
            for n in range(max(k + 2, args.enum_n + 1), args.rand_n + 1):
                edges_idx = None
                for _ in range(args.draws):
                    T = build_tree(rand_tree(n, rng), "equal", None)
                    E = [(T.parent[v], v) for v in T.nonroot]
                    if edges_idx is None:
                        edges_idx = list(range(len(E)))
                    combo = rng.sample(edges_idx, k)
                    anchors = [E[i][1] for i in combo]
                    if len(set(anchors)) < k:
                        continue
                    tested += 1
                    r = attack(T, anchors, k, False)
                    if r:
                        found = (n, k, T, r)
                        break
                if found:
                    break
                print(f"  rand k={k} n={n} done ({tested} configs so far)",
                      flush=True)
            if found:
                break

    print("\n" + "=" * 96)
    if not found:
        print(f"NO COUNTEREXAMPLE in {tested} configurations "
              f"(exhaustive to n={args.enum_n}, random to n={args.rand_n}, "
              f"k up to 5).")
        print("The conjecture survives an adversarial search. Next step is the")
        print("split-sign lemma, not more search.")
    else:
        n, k, T, r = found
        print(f"COUNTEREXAMPLE ({r['kind']}) at n={n} k={k}")
        print(f"  parent array : {T.parent}")
        print(f"  leaves       : {T.leaves}")
        print(f"  edge lengths : all 1")
        print(f"  anchors      : {r['anchors']}")
        print(f"  edges        : {r['edges']}")
        if r["kind"] == "MIXED_SIGN":
            print(f"  row subset I : {r['I']}")
            print(f"  # nonzero terms: {r['n_terms']}")
            print(f"  POSITIVE term  : E={r['pos'][0]} c={r['pos'][1]} "
                  f"det P={r['pos'][2]} det M'={r['pos'][3]}")
            print(f"  NEGATIVE term  : E={r['neg'][0]} c={r['neg'][1]} "
                  f"det P={r['neg'][2]} det M'={r['neg'][3]}")
            print("  -> the minor can vanish for a suitable positive length "
                  "vector, so the rank is NOT topology-determined here.")
        else:
            print(f"  det P = {r['dp']}, det M' = {r['dm']}  -- TU violated")


if __name__ == "__main__":
    main()
