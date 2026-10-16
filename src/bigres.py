"""Big-tree resolution engine.

For a reference tree T (the Greengenes 97% tree, ~99k leaves) and a support S of k
anchor taxa, compute the identifiability deficiency

    Delta(T,S) = 2k - rank J = (k-1) - rank B        (for distinct LEAF anchors)

without materialising any n x n or 2k x n matrix. Two facts make this O(|minimal
subtree|) per support instead of O(n):

  1. For leaf anchors the sigma-class count is c = k+1, so Delta = k-1 - rank B.
  2. The difference vector D[a_q] - D[a_k] is CONSTANT on every component of
     (tree minus the minimal subtree of the anchors): on each off-path subtree it
     equals dist(a_q,v) - dist(a_k,v) at the attachment node v, and on the region
     above the lca it equals depth[a_q] - depth[a_k].  Hence rank B is the rank of a
     (k-1) x R weighted-centred matrix with R ~ O(anchor path length), not n.

Delta is length-invariant (proved for the PDD Jacobian), so unit edge lengths are
used; real lengths are kept only for the length-invariance spot check.
"""
from __future__ import annotations

import numpy as np

from newick import parse_newick


class BigTree:
    __slots__ = ("parent", "children", "isleaf", "lab", "length", "root", "n",
                 "nleaf", "lo", "hi", "depth", "up", "LOG")

    def __init__(self, parent, children, isleaf, lab, length, root):
        self.parent = list(parent)
        self.children = [list(c) for c in children]
        self.isleaf = list(isleaf)
        self.lab = list(lab)
        self.length = list(length)
        self.root = root
        self.n = len(parent)
        self._index_leaves()
        self._depths()
        self._prep_lca()

    # ---- leaf ranges (pre-order DFS; subtree(u) = leaves lo[u]:hi[u]) ----
    def _index_leaves(self):
        n = self.n
        lo = np.zeros(n, dtype=np.int64)
        hi = np.zeros(n, dtype=np.int64)
        cnt = 0
        stack = [(self.root, False)]
        while stack:
            u, exiting = stack.pop()
            if exiting:
                hi[u] = cnt
            else:
                lo[u] = cnt
                if self.isleaf[u]:
                    hi[u] = cnt + 1
                    cnt += 1
                else:
                    stack.append((u, True))
                    for c in reversed(self.children[u]):
                        stack.append((c, False))
        self.nleaf = cnt
        self.lo = lo
        self.hi = hi

    def _depths(self):
        n = self.n
        depth = np.zeros(n, dtype=np.int64)
        order = [self.root]
        for u in order:
            for c in self.children[u]:
                depth[c] = depth[u] + 1
                order.append(c)
        self.depth = depth

    def _prep_lca(self):
        n = self.n
        maxd = int(self.depth.max())
        self.LOG = max(2, int(np.ceil(np.log2(maxd + 1))) + 1)
        up = np.full((self.LOG, n), -1, dtype=np.int64)
        up[0] = np.array(self.parent, dtype=np.int64)
        up[0][self.root] = self.root
        for k in range(1, self.LOG):
            up[k] = up[k - 1][up[k - 1]]
        self.up = up

    def lca(self, u, v):
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = int(self.depth[u] - self.depth[v])
        k = 0
        while diff:
            if diff & 1:
                u = int(self.up[k][u])
            diff >>= 1
            k += 1
        if u == v:
            return u
        for k in range(self.LOG - 1, -1, -1):
            if self.up[k][u] != self.up[k][v]:
                u = int(self.up[k][u])
                v = int(self.up[k][v])
        return int(self.parent[u])

    def lca_all(self, anchors):
        m = anchors[0]
        for a in anchors[1:]:
            m = self.lca(m, a)
        return m

    def is_ancestor(self, v, a):
        return self.lo[v] <= self.lo[a] < self.hi[v]

    @classmethod
    def from_newick(cls, path):
        parent, children, lab, length, isleaf, root = parse_newick(open(path).read())
        return cls(parent, children, isleaf, lab, length, root)


def _dist_to_anchor(bt, aq, v):
    """Unit-length distance from anchor leaf aq to node v on the minimal subtree."""
    w = bt.lca(aq, v)
    return int(bt.depth[aq] + bt.depth[v] - 2 * bt.depth[w])


def rank_B_regions(bt, anchors):
    """rank of the within-class contrast matrix, computed over the compressed
    regions of (tree minus minimal-subtree-of-anchors)."""
    k = len(anchors)
    if k <= 1:
        return 0
    anchors = [int(a) for a in anchors]
    m = bt.lca_all(anchors)
    # minimal-subtree node set (deduplicated)
    mst = {m}
    for aq in anchors:
        v = aq
        while v != m:
            mst.add(v)
            v = int(bt.parent[v])
    # collect regions: (size, value-vector of the k-1 difference coordinates)
    sizes = []
    vals = []  # each a list of k-1 floats
    # above-lca region
    sizeA = bt.nleaf - (bt.hi[m] - bt.lo[m])
    if sizeA > 0:
        sizes.append(sizeA)
        vals.append([int(bt.depth[aq] - bt.depth[anchors[-1]]) for aq in anchors[:-1]])
    # off-path subtrees of every minimal-subtree internal node
    for v in mst:
        if bt.isleaf[v]:
            continue
        for c in bt.children[v]:
            if c in mst:
                continue
            sz = bt.hi[c] - bt.lo[c]
            dk = _dist_to_anchor(bt, anchors[-1], v)
            sizes.append(sz)
            vals.append([_dist_to_anchor(bt, aq, v) - dk for aq in anchors[:-1]])
    if not sizes:
        return 0
    sizes = np.array(sizes, dtype=np.float64)
    M = np.array(vals, dtype=np.float64).T            # (k-1, R)
    tot = sizes.sum()
    mean = (M @ sizes) / tot                          # weighted mean per row
    C = M - mean[:, None]
    W = C * np.sqrt(sizes)[None, :]
    s = np.linalg.svd(W, compute_uv=False)
    tol = 1e-9 * (s[0] if s.size else 1.0)
    return int((s > max(tol, 1e-12)).sum())


def deficiency(bt, anchors):
    k = len(anchors)
    return k - 1 - rank_B_regions(bt, anchors)


def deficiency3(bt, anchors):
    """Fast k=3 leaf-anchor deficiency via arm structure (no per-region LCA).

    For three leaf anchors the closest pair (x,y) and the outlier z determine the
    minimal subtree: dist(x,v) is depth[x]-depth[v] when v is an ancestor of x,
    else depth[x]+depth[v]-2*depth[lca], where lca is m1=lca(x,y) if v lies on the
    other arm of the closest pair and m=lca(x,y,z) if v lies on the z-arm.  Only 3
    LCA calls are needed in total.  Verified 0/383,838 mismatches vs the exact
    resolution.deficiency (including n=3).
    """
    a, b, c = [int(x) for x in anchors]
    lab = bt.lca(a, b); lac = bt.lca(a, c); lbc = bt.lca(b, c)
    d_ab = int(bt.depth[lab]); d_ac = int(bt.depth[lac]); d_bc = int(bt.depth[lbc])
    if d_ab >= d_ac and d_ab >= d_bc:
        m1, m = lab, (lac if d_ac <= d_bc else lbc); x, y, z = a, b, c
    elif d_ac >= d_ab and d_ac >= d_bc:
        m1, m = lac, (lab if d_ab <= d_bc else lbc); x, y, z = a, c, b
    else:
        m1, m = lbc, (lab if d_ab <= d_ac else lac); x, y, z = b, c, a
    dx = int(bt.depth[x]); dy = int(bt.depth[y]); dz = int(bt.depth[z])
    dm = int(bt.depth[m]); dm1 = int(bt.depth[m1])
    lox, loy, loz = int(bt.lo[x]), int(bt.lo[y]), int(bt.lo[z])
    hix, hiy, hiz = int(bt.hi[x]), int(bt.hi[y]), int(bt.hi[z])

    sizes = []
    vals = []
    sizeA = bt.nleaf - (bt.hi[m] - bt.lo[m])
    if sizeA > 0:
        sizes.append(sizeA)
        vals.append([dx - dz, dy - dz])
    mst = {m}
    for aq in (x, y, z):
        v = aq
        while v != m:
            mst.add(v)
            v = int(bt.parent[v])
    for v in mst:
        if bt.isleaf[v]:
            continue
        dv = int(bt.depth[v]); lo = int(bt.lo[v]); hi = int(bt.hi[v])
        ancx = lo <= lox < hi
        ancy = lo <= loy < hi
        ancz = lo <= loz < hi
        # distances from v to the three anchors
        dxx = dx - dv if ancx else (dx + dv - 2 * dm1 if ancy else dx + dv - 2 * dm)
        dyy = dy - dv if ancy else (dy + dv - 2 * dm1 if ancx else dy + dv - 2 * dm)
        dzz = dz - dv if ancz else dz + dv - 2 * dm
        for ch in bt.children[v]:
            if ch in mst:
                continue
            sizes.append(bt.hi[ch] - bt.lo[ch])
            vals.append([dxx - dzz, dyy - dzz])
    if not sizes:
        c = 3 if bt.nleaf <= 3 else 4
        return 6 - c  # rank B = 0
    S = np.array(sizes, dtype=np.float64)
    M = np.array(vals, dtype=np.float64).T
    mean = (M @ S) / S.sum()
    C = M - mean[:, None]
    W = C * np.sqrt(S)[None, :]
    s = np.linalg.svd(W, compute_uv=False)
    r = int((s > 1e-9 * max(1.0, s[0])).sum())
    c = 3 if bt.nleaf <= 3 else 4
    return 6 - c - r


# ---------------------------------------------------------------- small-tree
# verification: rebuild a small rooted binary tree as a BigTree and compare with
# resolution.deficiency on the same anchors.
def bigtree_from_small(T):
    parent = list(T.parent)
    children = [list(c) for c in T.children]
    isleaf = list(T.isleaf)
    lab = list(T.lab)
    length = [0.0] * len(parent)
    return BigTree(parent, children, isleaf, lab, length, T.root)


if __name__ == "__main__":
    import itertools
    from probe_identifiability import build_tree, gen_trees
    from resolution import deficiency as deficiency_small

    bad = 0
    tot = 0
    for n in range(4, 8):
        for t in gen_trees(range(1, n + 1)):
            T = build_tree(t, "equal", None)
            bt = bigtree_from_small(T)
            for anchors in itertools.combinations(T.leaves, 3):
                d1 = deficiency_small(T, list(anchors))
                d2 = deficiency(bt, list(anchors))
                tot += 1
                if d1 != d2:
                    bad += 1
                    if bad <= 5:
                        print(f"MISMATCH n={n} anchors={anchors} "
                              f"resolution={d1} bigres={d2}")
    print(f"verification: {tot} 3-leaf supports, {bad} mismatches")
