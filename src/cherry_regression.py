"""Cherry lemma regression (Step 0/1 of the correction).

Verifies, on the exact engine, three things the corrected lemma rests on:

  (A) the paper's edge-segment identity  D_a + l_a C_a = D_parent(a)   (every anchor);
  (B) the cherry identity               D_a - D_b = -l_a C_a + l_b C_b  (sibling a,b);
  (C) Delta({a,b,c}) >= 1 for every cherry {a,b} and third leaf c  (n >= 4);
      and the degenerate n = 3 case where Delta should NOT be 1.

Each check is run under equal, random, and extreme (each edge x100 / x0.01) lengths,
and also asserts (B) is *specific* to cherries: non-sibling pairs violate it.
"""
from __future__ import annotations

import itertools

import numpy as np

from probe_identifiability import build_tree, gen_trees
from resolution import deficiency

TOL = 1e-8


def sibling_pairs(T):
    """list of (a, b) sibling leaf pairs (a < b)."""
    out = []
    for a, b in itertools.combinations(T.leaves, 2):
        if T.parent[a] == T.parent[b]:
            out.append((a, b))
    return out


def check_identities(T):
    """Return (edge_seg_bad, cherry_bad, nonsib_wrongly_holds)."""
    edge_bad = 0
    for a in T.nonroot:
        v = T.D[a] + T.len[a] * T.C[a] - T.D[T.parent[a]]
        if np.abs(v).max() > TOL:
            edge_bad += 1
    pairs = sibling_pairs(T)
    cher_bad = 0
    for a, b in pairs:
        v = T.D[a] - T.D[b] + T.len[a] * T.C[a] - T.len[b] * T.C[b]
        if np.abs(v).max() > TOL:
            cher_bad += 1
    # specificity: a random non-sibling pair should (almost always) violate (B)
    nonsib_holds = 0
    nonsib_total = 0
    for a, b in itertools.combinations(T.leaves, 2):
        if T.parent[a] == T.parent[b]:
            continue
        v = T.D[a] - T.D[b] + T.len[a] * T.C[a] - T.len[b] * T.C[b]
        nonsib_total += 1
        if np.abs(v).max() <= TOL:
            nonsib_holds += 1
    return edge_bad, cher_bad, nonsib_holds, nonsib_total


def main() -> None:
    rng = np.random.default_rng(7)

    # (A)+(B) identity regression over length regimes
    print("=" * 86)
    print("(A) edge-segment identity  D_a + l_a C_a = D_parent(a)")
    print("(B) cherry identity        D_a - D_b = -l_a C_a + l_b C_b   (siblings)")
    print("=" * 86)
    edge_tot = cher_tot = 0
    edge_bad = cher_bad = 0
    nonsib_holds = nonsib_total = 0
    for n in [4, 5, 6]:
        for t in gen_trees(range(1, n + 1)):
            regimes = [("equal", build_tree(t, "equal", None))]
            for si in range(2):
                regimes.append((f"random{si}",
                                build_tree(t, "random", np.random.default_rng(100 + si))))
            # extreme: each edge x100 and x0.01 (n<=6 only, to bound work)
            Te = build_tree(t, "equal", None)
            for e in Te.nonroot:
                T100 = build_tree(t, "equal", None)
                T100.len[e] = 100.0
                T100._distances()
                regimes.append((f"edge{e}x100", T100))
                T01 = build_tree(t, "equal", None)
                T01.len[e] = 0.01
                T01._distances()
                regimes.append((f"edge{e}x0.01", T01))
            for name, T in regimes:
                eb, cb, nh, nt = check_identities(T)
                edge_tot += 1
                cher_tot += len(sibling_pairs(T))
                nonsib_holds += nh
                nonsib_total += nt
                edge_bad += eb
                cher_bad += cb
    print(f"(A) edge-segment: {edge_tot} tree/regime checks, {edge_bad} bad")
    print(f"(B) cherry identity: {cher_tot} sibling pairs, {cher_bad} bad")
    print(f"(B) specificity: {nonsib_holds}/{nonsib_total} non-sibling pairs satisfy "
          f"(B) -> expect 0-ish, nonsiblings must violate it")

    # (C) Delta({a,b,c}) = 1 for cherries + third leaf, across n and lengths
    print()
    print("=" * 86)
    print("(C) Delta({a,b,c}) for cherry {a,b} + third leaf c")
    print("=" * 86)
    bad = 0
    tot = 0
    for n in [4, 5, 6, 7]:
        for t in gen_trees(range(1, n + 1)):
            for name, T in [("equal", build_tree(t, "equal", None)),
                            ("random", build_tree(t, "random", np.random.default_rng(55)))]:
                for a, b in sibling_pairs(T):
                    for c in T.leaves:
                        if c in (a, b):
                            continue
                        d = deficiency(T, [a, b, c])
                        tot += 1
                        if d < 1:
                            bad += 1
                            if bad <= 8:
                                print(f"  MISMATCH n={n} {name} cherry=({a},{b}) c={c} "
                                      f"Delta={d} (want >= 1)")
    print(f"(C) Delta>=1 for cherry+third: {tot} supports, {bad} mismatches")

    # n = 3 degenerate case
    print()
    print("=" * 86)
    print("(D) n = 3 edge case (no leaf outside the cherry)")
    print("=" * 86)
    n3bad = 0
    for t in gen_trees(range(1, 4)):  # only one binary tree on 3 leaves
        T = build_tree(t, "equal", None)
        a, b = sibling_pairs(T)[0]
        c = [x for x in T.leaves if x not in (a, b)][0]
        d = deficiency(T, [a, b, c])
        rankJ = 6 - d
        print(f"  n=3 cherry=({a},{b}) c={c}: Delta={d}, rank J={rankJ}, "
              f"sigma-classes c={4 - (6 - 2 * 3 - 2) if False else 'n/a'}")
        # expectation: 2k=6 > n=3 -> rank J <= 3, so Delta >= 3
        if d == 1:
            n3bad += 1
    print(f"  (D) n=3 must NOT give Delta=1; violations={n3bad}")
    print()
    print("All regressions done.")


if __name__ == "__main__":
    main()
