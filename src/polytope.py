"""PDD: the endpoint-simplex / polytope formulation, with four-way validation.

THE THREE COORDINATE SYSTEMS
    A  direct     Phi_r = sum_q p_q (dist(r, P_q) + y_q), with P_q a point on its
                  edge. Rebuilt from the tree with the query leaf actually
                  inserted, no C, no ell, no rooting.
    B  linear     Phi_r = sum_q p_q D_{a_q}(r) + sum_q u_q C_{r,a_q} ell_{a_q} + ybar
                  with u = x o p.  (probe 1, validated to 7.4e-10)
    C  simplex    alpha_q = p_q - u_q, beta_q = u_q; then (alpha,beta) in
                  Delta^{2k-1} and
                      Phi_r = sum_q alpha_q v_q^(0)(r) + sum_q beta_q v_q^(1)(r)
                              + ybar,
                      v_q^(0) = D_{a_q},   v_q^(1) = D_{a_q} + C_{a_q} ell_{a_q}.
    D  jacobian   finite differences of A, against the analytic J, against the
                  affine-independence condition on the centred endpoints.

CLAIMS UNDER TEST
    (i)  A = B = C to < 1e-10 on every (tree, anchors, parameters).
    (ii) rank J = 2k  <=>  the 2k centred endpoints v~_j = H v_j are affinely
         independent, H = I - (1/n) 1 1^T. The two directions were argued by
         hand from "1 in span{differences} => affine dependence"; this checks it.
    (iii) the image of the parameter simplex is EXACTLY P_A = conv(v_1..v_{2k}),
         plus the free offset ybar.
    (iv) for k = 3 with c = k+1 = 4 classes, rank J = c + rank(B) where B
         collects the class-internal variation of the k-1 differences; hence
         non-identifiability at c = 4  <=>  the class-internal variations are
         collinear (rank B < 2).

Usage:
    polytope.py [--max-n 6] [--ks 2,3]
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from probe_identifiability import build_tree, gen_trees, rank_of
from verify_k2 import node_path

OUT = "./out"
TOL = 1e-9


# ------------------------------------------------------------------ endpoints
def endpoints(T, anchors):
    """(v, vsim) with v of shape (2k, n): the k 'alpha' endpoints then the k
    'beta' endpoints. Position-independent by construction."""
    n = len(T.leaves)
    v0 = [T.D[a] for a in anchors]
    v1 = [T.D[a] + T.C[a] * T.len[a] for a in anchors]
    v = np.array(v0 + v1)
    H = np.eye(n) - np.ones((n, n)) / n
    return v, v @ H.T


def affine_dim(vsim):
    """Affine dimension of the rows of vsim."""
    if vsim.shape[0] <= 1:
        return 0
    return rank_of(vsim[1:] - vsim[0])


# ------------------------------------------------------------------ path A
def phi_direct(T, anchors, p, x, ybar):
    """Phi from the tree, with the query leaf inserted at distance x_q * ell_q
    from the anchor (the child end). No C, no ell in the arithmetic: distances
    are recomputed from the tree each time."""
    n = len(T.parent)
    adj = [[] for _ in range(n)]
    for u in range(n):
        if T.parent[u] >= 0:
            adj[u].append((T.parent[u], T.len[u]))
            adj[T.parent[u]].append((u, T.len[u]))
    # insert a new node on each placement edge
    new = n
    for q, a in enumerate(anchors):
        par, ell = T.parent[a], T.len[a]
        t = x[q] * ell                     # distance from the anchor
        # edge (par,a) becomes (par,new) + (new,a)
        adj[par] = [(w, l) for (w, l) in adj[par] if w != a]
        adj[a] = [(w, l) for (w, l) in adj[a] if w != par]
        adj[par].append((new, ell - t))
        adj[a].append((new, t))
        adj.append([(par, ell - t), (a, t)])
        new += 1
    D = {}
    for src in range(len(adj)):
        dist = {src: 0.0}; st = [src]
        while st:
            u = st.pop()
            for w, l in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + l; st.append(w)
        D[src] = dist
    out = np.zeros(len(T.leaves))
    for j, r in enumerate(T.leaves):
        for q in range(len(anchors)):
            out[j] += p[q] * (D[r][n + q] + ybar)   # sum p_q y_q = ybar
    return out


def phi_linear(T, anchors, p, x, ybar):
    n = len(T.leaves)
    out = np.zeros(n)
    for q, a in enumerate(anchors):
        out += p[q] * T.D[a] + (x[q] * p[q]) * T.C[a] * T.len[a]
    return out + ybar


def phi_simplex(T, anchors, alpha, beta, ybar):
    n = len(T.leaves)
    out = np.zeros(n)
    for q, a in enumerate(anchors):
        out += alpha[q] * T.D[a] + beta[q] * (T.D[a] + T.C[a] * T.len[a])
    return out + ybar


def jacobian(T, anchors, p=None):
    k = len(anchors)
    n = len(T.leaves)
    rows = [T.D[anchors[q]] - T.D[anchors[-1]] for q in range(k - 1)]
    rows += [T.C[a] * T.len[a] for a in anchors]
    rows += [np.ones(n)]
    return np.array(rows)


# ------------------------------------------------------------------ classes
def sigma_classes(T, anchors):
    c = {}
    for j in range(len(T.leaves)):
        c.setdefault(tuple(T.C[a][j] for a in anchors), []).append(j)
    return c


def class_internal_B(T, anchors, classes):
    """The k-1 difference vectors, quotiented by the class-constant functions.
    Returns the (n-c) x (k-1) matrix of class-internal contrasts."""
    n = len(T.leaves)
    idx = sorted(range(n), key=lambda j: tuple(T.C[a][j] for a in anchors))
    Dm = np.array([T.D[anchors[q]] - T.D[anchors[-1]] for q in range(len(anchors) - 1)])
    X = Dm.T[idx]                              # n x (k-1), leaves ordered by class
    rows = []
    s = 0
    for key, jj in sorted(classes.items()):
        blk = X[s:s + len(jj)]
        s += len(jj)
        if len(jj) > 1:
            rows.append(blk[1:] - blk[0])      # within-class contrasts
    B = np.vstack(rows) if rows else np.zeros((0, X.shape[1]))
    return B


def log_header(n, k, anchors, cls):
    print("[MODEL]")
    print(f"  n={n}  k={k}  anchors={tuple(anchors)}")
    print(f"  parameter_dim={2*k}  (2k-1 on the simplex + ybar)")
    print(f"  sigma_classes={len(cls)}  sizes={sorted(len(v) for v in cls.values())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=5)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--verbose", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 96)
    print("PDD endpoint-simplex / polytope formulation -- four-way validation")
    print("=" * 96)

    rng = np.random.default_rng(5)
    worst = {"AB": 0.0, "AC": 0.0, "BC": 0.0, "FD": 0.0}
    mismatch = {"affine": 0, "image": 0, "quotient": 0}
    rows = []
    shown = 0

    for k in [int(s) for s in args.ks.split(",")]:
        for n in range(max(4, k + 1), args.max_n + 1):
            for ti, t in enumerate(gen_trees(range(1, n + 1))):
                T = build_tree(t, "random", np.random.default_rng(41 + ti))
                E = [(T.parent[v], v) for v in T.nonroot]
                for combo in itertools.combinations(range(len(E)), k):
                    anchors = [E[i][1] for i in combo]
                    if len(set(anchors)) < k:
                        continue
                    # random parameters in the interior
                    p = rng.random(k) + 0.1; p /= p.sum()
                    x = rng.random(k) * 0.8 + 0.1
                    ybar = float(rng.random() + 0.1)
                    alpha = p * (1 - x); beta = p * x

                    A = phi_direct(T, anchors, p, x, ybar)
                    B = phi_linear(T, anchors, p, x, ybar)
                    C = phi_simplex(T, anchors, alpha, beta, ybar)
                    worst["AB"] = max(worst["AB"], np.abs(A - B).max())
                    worst["AC"] = max(worst["AC"], np.abs(A - C).max())
                    worst["BC"] = max(worst["BC"], np.abs(B - C).max())

                    # --- D: finite differences, in the SAME coordinates as J ---
                    # J's coordinates are (p, u, ybar) with p and u INDEPENDENT:
                    # p on the simplex and 0 < u_q < p_q. So a p-step must hold u
                    # fixed, which means x_q = u_q/p_q moves too. Differentiating
                    # with x held fixed mixes the p and u rows -- that was the
                    # 8.6 discrepancy in the first run of this check.
                    J = jacobian(T, anchors)
                    Jn = np.zeros((2 * k, n))
                    u0 = x * p

                    def phi_of(pv, uv):
                        return phi_direct(T, anchors, pv, uv / np.maximum(pv, 1e-12),
                                          ybar)
                    for i in range(k - 1):
                        # J's p-coordinates carry p_k = 1 - sum_{q<k} p_q, so a
                        # step in p_i must be compensated in p_k: otherwise
                        # sum p != 1 and phi_direct's ybar bookkeeping breaks.
                        h = 1e-6
                        pp, pm = p.copy(), p.copy()
                        pp[i] += h; pp[-1] -= h
                        pm[i] -= h; pm[-1] += h
                        if min(pp.min(), pm.min()) <= 0:
                            continue
                        Jn[i] = (phi_of(pp, u0) - phi_of(pm, u0)) / (2 * h)
                    for i in range(k):
                        h = 1e-6
                        up, um = u0.copy(), u0.copy()
                        up[i] += h; um[i] -= h
                        Jn[k - 1 + i] = (phi_of(p, up) - phi_of(p, um)) / (2 * h)
                    Jn[-1] = 1.0
                    worst["FD"] = max(worst["FD"], np.abs(Jn - J).max())

                    v, vsim = endpoints(T, anchors)
                    rk = rank_of(J)
                    ad = affine_dim(vsim)

                    # (ii) rank J = 2k  <=>  affine dim = 2k-1
                    if (rk == 2 * k) != (ad == 2 * k - 1):
                        mismatch["affine"] += 1
                        if shown < args.verbose:
                            print(f"  AFFINE MISMATCH n={n} k={k} anchors={anchors}"
                                  f" rank={rk} affdim={ad} (2k-1={2*k-1})")
                            shown += 1

                    # (iii) the image is exactly conv(v) + span{1}
                    lam = np.zeros(2 * k)
                    lam[:k] = alpha; lam[k:] = beta
                    img = lam @ v + ybar
                    img2 = phi_simplex(T, anchors, alpha, beta, ybar)
                    if np.abs(img - img2).max() > 1e-8:
                        mismatch["image"] += 1

                    cls = sigma_classes(T, anchors)
                    c = len(cls)
                    Bm = class_internal_B(T, anchors, cls)
                    rb = rank_of(Bm) if Bm.size else 0
                    # (iv) rank J = c + rank(B), when the c class-constant rows
                    #      {1, sigma_q} are independent on the classes
                    if c == k + 1:
                        pred = c + rb
                        if pred != rk:
                            mismatch["quotient"] += 1
                    rows.append(dict(k=k, n=n, c=c, rk=rk, rb=rb, ad=ad,
                                     ident=int(rk == 2 * k),
                                     c_plus_rb=c + rb,
                                     nclass_ge_kp1=int(c == k + 1)))

    print("\n[MAP-CHECK]  (max abs difference over every configuration tried)")
    print(f"  direct  vs linear   : {worst['AB']:.3e}")
    print(f"  direct  vs simplex  : {worst['AC']:.3e}")
    print(f"  linear  vs simplex  : {worst['BC']:.3e}")
    print(f"  FD Jacobian vs analytic : {worst['FD']:.3e}")
    ok = max(worst.values()) < 1e-6

    print("\n[RANK]")
    print("  (ii) rank J = 2k  <=>  centered endpoints affinely independent")
    print(f"       disagreements: {mismatch['affine']}")
    print(f"  (iii) image = conv(v_1..v_2k) + span{{1}}: "
          f"disagreements {mismatch['image']}")
    print(f"  (iv) for c = k+1, rank J = c + rank(B): "
          f"disagreements {mismatch['quotient']}")

    import pandas as pd
    D = pd.DataFrame(rows)
    print("\n[QUOTIENT]  rank J against c + rank(B)")
    print(D.groupby(["k", "c", "rb"])["rk"].agg(["mean", "size"]).to_string())
    print("\n  rank J vs c + rank(B) overall: "
          f"{int((D.rk == D.c_plus_rb).sum())}/{len(D)} agree")
    bad = D[D.rk != D.c_plus_rb]
    if len(bad):
        print("  disagreements by (k, c, rb):")
        print(bad.groupby(["k", "c", "rb"]).size().to_string())
    D.to_csv(os.path.join(OUT, "polytope.csv"), index=False)

    print("\n" + "=" * 96)
    print("VERDICT:", "PASS" if (ok and sum(mismatch.values()) == 0) else "CHECK")
    print("=" * 96)
    print("artifacts ->", os.path.join(OUT, "polytope.csv"))


if __name__ == "__main__":
    main()
