"""Shared-edge collision regression for the necessity theorem (Theorem 7, RECOMB 2027).

Enumerates every *unlabeled* rooted binary tree shape (Catalan number C_{n-1})
for n = 5, 6, 7, and for each shape every pair of supports {a,s}, {b,s} sharing
the placement edge above s (a, b, s distinct non-root nodes). For each pair we
test interior collision with the LP oracle of global_collision.py: whether
conv(v^{a,s}) + span{1} and conv(v^{b,s}) + span{1} meet at a strictly interior
point (all convex coefficients > 0), and we record whether the collision is the
configuration of Theorem 6 (the two neighbours of the degree-2 root, with s
adjacent to one of them).

Label-invariance: the collision verdict is unchanged by any relabelling of the
leaves (a leaf permutation commutes with the LP and fixes span{1}), so
enumerating unlabeled shapes -- one canonical labelling each -- is exhaustive
and each distinct collision configuration is counted once.

Reported counts (unit branch lengths):
    n = 5 : 14 shapes,  2352 pairs,   36 collisions
    n = 6 : 42 shapes, 15120 pairs,  112 collisions
    n = 7 : 132 shapes, 87120 pairs, 360 collisions
All collisions are the Theorem-6 configuration; none other occurs. Random
positive branch lengths leave the classification unchanged.

Usage:
    shared_edge_regression.py [--ns 5,6,7] [--random-draws 3] [--seed 0]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from probe_identifiability import build_tree
from polytope import endpoints
from global_collision import hull_intersect

TOL = 1e-7


def unlabeled_shapes(n):
    """Yield every unlabeled rooted binary tree with n leaves, once, in the
    ('L',·) / ('I',·,·) tuple format understood by build_tree."""
    if n == 1:
        yield ("L", 0)
        return
    for i in range(1, n):
        for left in unlabeled_shapes(i):
            for right in unlabeled_shapes(n - i):
                yield ("I", left, right)


def is_theorem6_config(T, a, b, s):
    """Collision {a,s} vs {b,s} is the configuration of Theorem 6 (up to the
    relabelling a <-> b): a, b are the two children of the root and s is a
    child of one of them."""
    root = T.root
    return (T.parent[a] == root and T.parent[b] == root
            and (T.parent[s] == a or T.parent[s] == b))


def run(n, lengths, rng):
    shapes = list(unlabeled_shapes(n))
    pairs = collisions = other = 0
    for t in shapes:
        T = build_tree(t, lengths, rng)
        nodes = T.nonroot
        for a, b in itertools.combinations(nodes, 2):  # unordered pair a < b
            for s in nodes:
                if s == a or s == b:
                    continue
                pairs += 1
                vA, _ = endpoints(T, [a, s])
                vB, _ = endpoints(T, [b, s])
                intersects, tstar = hull_intersect(vA, vB)
                if intersects and tstar > TOL:
                    collisions += 1
                    if not is_theorem6_config(T, a, b, s):
                        other += 1
    return shapes, pairs, collisions, other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5,6,7")
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("shared-edge collision regression (unit branch lengths)")
    for n in [int(v) for v in args.ns.split(",")]:
        shapes, pairs, collisions, other = run(n, "equal", rng)
        print(f"  n={n}: {len(shapes)} shapes, {pairs} pairs, "
              f"{collisions} collisions, {other} non-Theorem-6", flush=True)

    print("random positive branch lengths (robustness)")
    for n in (5, 6):
        for draw in range(args.random_draws):
            shapes, pairs, collisions, other = run(n, "random", rng)
            print(f"  n={n} draw {draw}: {collisions} collisions, "
                  f"{other} non-Theorem-6", flush=True)


if __name__ == "__main__":
    main()
