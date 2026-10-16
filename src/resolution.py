"""Phylogenetic resolution engine: the biological-analysis object on top of the PDD
identifiability theory.

For a reference tree T and a support S of k anchors (the placement edges, i.e. the taxa
of a community), the fixed-support identifiability rank is

    R(T,S) = rank J = c + rank B      (c = #sigma-classes, B = within-class contrast)

and the *identifiability deficiency* / *unresolved degrees of freedom* is

    Delta(T,S) = 2k - R(T,S) .

Delta = 0  <=> the k-component mixture is locally identifiable on T;
Delta > 0  <=> at least Delta parameter directions are invisible to the measurement.

The computation avoids materialising the full 2k x n Jacobian: c is the number of level
sets of the sigma-signature, and rank B is the rank of the (k-1) within-class contrast
vectors, both computable from the k distance vectors to the anchors.

This is the engine for the biological question: what can a reference phylogeny resolve
in a mixed community?
"""
from __future__ import annotations

import numpy as np

from probe_identifiability import Tree


def sigma_classes(T, anchors):
    """Map (sigma-signature tuple) -> list of leaf indices. c = len of this map."""
    classes = {}
    for j in range(len(T.leaves)):
        key = tuple(T.C[a][j] for a in anchors)
        classes.setdefault(key, []).append(j)
    return classes


def rank_B(T, anchors):
    """Rank of the within-class contrast matrix B (rows = within-class contrasts of the
    k-1 difference vectors D_{a_q} - D_{a_k})."""
    k = len(anchors)
    n = len(T.leaves)
    if k <= 1:
        return 0
    # the k-1 difference vectors (as n-columns)
    Dm = np.array([T.D[anchors[q]] - T.D[anchors[-1]] for q in range(k - 1)])  # (k-1, n)
    classes = sigma_classes(T, anchors)
    rows = []
    for key, jj in sorted(classes.items()):
        blk = Dm[:, jj]                      # (k-1, m) block for this class
        if blk.shape[1] > 1:
            rows.append(blk[:, 1:] - blk[:, 0:1])   # within-class contrasts
    if not rows:
        return 0
    B = np.hstack(rows)                      # (k-1, n-c)
    return int(np.linalg.matrix_rank(B, tol=1e-9))


def rank_J(T, anchors):
    """rank J = c + rank B."""
    c = len(sigma_classes(T, anchors))
    return c + rank_B(T, anchors)


def deficiency(T, anchors):
    """Delta = 2k - rank J = unresolved degrees of freedom."""
    k = len(anchors)
    return 2 * k - rank_J(T, anchors)


def unresolved_directions(T, anchors):
    """Null-space of the Jacobian J (the invisible parameter directions), for small n.
    Returns an orthonormal basis (as columns), or empty if fully identifiable."""
    from probe_identifiability import jacobian
    J = jacobian(T, list(anchors))
    # nullspace via SVD of J^T (n x 2k) -> left null of J is null of J^T
    u, s, vh = np.linalg.svd(J, full_matrices=True)
    # null space of J (right null of J^T?): J is 2k x n, null of J is in R^n
    # rank = number of nonzero singular values; nullspace = vh[rank:] (rows)
    rank = int((s > 1e-9 * max(1.0, s[0])).sum())
    return vh[rank:].T if rank < vh.shape[0] else np.zeros((J.shape[1], 0))


if __name__ == "__main__":
    # self-test: Delta >= 0, and topology-only (equal vs random lengths).
    from probe_identifiability import build_tree, gen_trees
    import itertools
    bad = 0
    tot = 0
    for ti, t in enumerate(gen_trees(range(1, 6))):
        Te = build_tree(t, "equal", None)
        Tr = build_tree(t, "random", np.random.default_rng(42 + ti))
        nodes = list(Te.nonroot)
        for k in [2, 3]:
            for anchors in itertools.combinations(nodes, k):
                d_e = deficiency(Te, list(anchors))
                d_r = deficiency(Tr, list(anchors))
                tot += 1
                if d_e != d_r or d_e < 0:
                    bad += 1
                    if bad <= 5:
                        print(f"MISMATCH n=5 k={k} anchors={anchors} equal={d_e} random={d_r}")
    print(f"self-test: {tot} configs, {bad} mismatches (Delta >= 0 and length-invariant)")
