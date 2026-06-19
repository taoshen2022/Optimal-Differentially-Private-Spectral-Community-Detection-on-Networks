"""Core utilities for real-data NetPTR network experiments.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd

from typing import Optional, Dict

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

def _upper_edges_from_csr(A):
    """Return upper-triangular edges (i<j) as (ei, ej) from symmetric CSR adjacency."""
    A = A.tocsr()
    r, c = A.nonzero()
    m = r < c
    ei = r[m].astype(np.int64)
    ej = c[m].astype(np.int64)
    return ei, ej

def _tri_prefix(n: int) -> np.ndarray:
    c = np.arange(n - 1, 0, -1, dtype=np.int64)
    prefix = np.zeros(n, dtype=np.int64)
    prefix[1:] = np.cumsum(c, dtype=np.int64)
    return prefix

def _tri_index_to_ij(k: np.ndarray, n: int, prefix: np.ndarray):
    k = np.asarray(k, dtype=np.int64)
    i = np.searchsorted(prefix, k, side="right") - 1
    off = k - prefix[i]
    j = i + 1 + off
    return i.astype(np.int64), j.astype(np.int64)

def _build_sparse_A_from_edges(n: int, ei: np.ndarray, ej: np.ndarray):
    m = int(ei.size)
    if m == 0:
        return coo_matrix((n, n), dtype=float).tocsr()
    data = np.ones(2 * m, dtype=float)
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    A = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=float).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    if A.nnz:
        A.data[:] = 1.0
    return A

def _edgeflip_upper_edges_rr(
    n: int,
    ei: np.ndarray,
    ej: np.ndarray,
    eps: float,
    rng: np.random.Generator,
    codes_orig_sorted: np.ndarray,
    prefix: np.ndarray,
    max_edges_upper_priv: int = 30_000_000,
    oversample: float = 1.5,
    max_rounds: int = 200,
):
    """
    Randomized response edge flip on ALL pairs:
      keep existing edge w.p. 1-q, drop w.p. q
      add non-edge w.p. q
    We allow duplicates among added non-edges (they get collapsed by sum_duplicates).
    """
    eps = float(eps)
    q = 1.0 / (np.exp(eps) + 1.0)

    ei = np.asarray(ei, dtype=np.int64)
    ej = np.asarray(ej, dtype=np.int64)
    m = int(ei.size)

    # remove original edges
    keep = rng.random(m) >= q
    ei_keep = ei[keep]
    ej_keep = ej[keep]

    # how many non-edges flip to edges
    N_total = n * (n - 1) // 2
    N_non = N_total - m
    add_m = int(rng.binomial(N_non, q))

    # membership test for "is this an original edge?"
    def _is_in_orig(codes: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(codes_orig_sorted, codes)
        ok = (idx < codes_orig_sorted.size)
        out = np.zeros(codes.size, dtype=bool)
        out[ok] = (codes_orig_sorted[idx[ok]] == codes[ok])
        return out

    # sample added non-edges via rejection against orig edges
    need = int(add_m)
    ei_add_chunks = []
    ej_add_chunks = []
    rounds = 0

    while need > 0 and rounds < max_rounds:
        rounds += 1
        draw = int(np.ceil(need * oversample)) + 1000
        k = rng.integers(0, N_total, size=draw, dtype=np.int64)
        ci, cj = _tri_index_to_ij(k, n, prefix)
        codes = ci * np.int64(n) + cj

        mask = ~_is_in_orig(codes)
        ci = ci[mask]
        cj = cj[mask]
        if ci.size == 0:
            continue

        take = min(need, int(ci.size))
        ei_add_chunks.append(ci[:take])
        ej_add_chunks.append(cj[:take])
        need -= take

    if need > 0:
        raise RuntimeError(f"EdgeFlip rejection sampler failed: still need {need} non-edges (try higher max_rounds).")

    if add_m > 0:
        ei_add = np.concatenate(ei_add_chunks).astype(np.int64, copy=False)
        ej_add = np.concatenate(ej_add_chunks).astype(np.int64, copy=False)
        ei_priv = np.concatenate([ei_keep, ei_add])
        ej_priv = np.concatenate([ej_keep, ej_add])
    else:
        ei_priv, ej_priv = ei_keep, ej_keep

    return ei_priv, ej_priv, float(q), int(add_m), int(ei_keep.size)

# -----------------------------
# stable sigmoid + clustering
# -----------------------------
def _stable_sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)

def _cluster_from_embedding(Xi: np.ndarray, K: int, nstart: int = 25, norm_tol: float = 1e-12, seed=None) -> np.ndarray:
    Xi = np.asarray(Xi, dtype=float)
    rn = np.sqrt((Xi * Xi).sum(axis=1))
    rn = np.maximum(rn, norm_tol)
    Xn = Xi / rn[:, None]
    km = KMeans(n_clusters=int(K), n_init=int(nstart), random_state=None if seed is None else int(seed))
    return km.fit_predict(Xn)

def clustering_error(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    labels_true = np.asarray(labels_true, dtype=int)
    labels_pred = np.asarray(labels_pred, dtype=int)
    if labels_true.shape != labels_pred.shape:
        raise ValueError("labels_true and labels_pred must have the same shape.")
    n = labels_true.size
    if n == 0:
        return 0.0

    _, true_inv = np.unique(labels_true, return_inverse=True)
    _, pred_inv = np.unique(labels_pred, return_inverse=True)
    k_true = int(true_inv.max()) + 1
    k_pred = int(pred_inv.max()) + 1

    counts = np.zeros((k_true, k_pred), dtype=int)
    np.add.at(counts, (true_inv, pred_inv), 1)

    maxc = counts.max() if counts.size else 0
    cost = maxc - counts
    r, c = linear_sum_assignment(cost)
    matched = int(counts[r, c].sum())
    return 1.0 - matched / n


# -----------------------------
# load BlogCatalog graph
# -----------------------------
def _read_int_list_from_fileobj(f) -> np.ndarray:
    vals = []
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "," in s:
            tok = s.split(",", 1)[0].strip()
        else:
            tok = s.split()[0].strip()
        try:
            vals.append(int(tok))
        except ValueError:
            continue
    return np.asarray(vals, dtype=np.int64)

def _iter_edges_fileobj(f):
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
        else:
            parts = s.split()
        if len(parts) < 2:
            continue
        try:
            u = int(parts[0]); v = int(parts[1])
        except ValueError:
            continue
        yield u, v

def load_blogcatalog_graph(base_dir=None, zip_path=None):
    """
    Returns:
      A (csr), id_map (orig_id -> 0..n-1)
    """
    if (base_dir is None) == (zip_path is None):
        raise ValueError("Provide exactly one of base_dir or zip_path.")

    if zip_path is not None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            def find_member(basename: str):
                b = basename.lower()
                cands = [n for n in zf.namelist() if n.lower().endswith("/" + b) or n.lower() == b]
                if not cands:
                    return None
                cands.sort(key=lambda s: (s.count("/"), len(s)))
                return cands[0]

            m_nodes = find_member("nodes.csv")
            m_edges = find_member("edges.csv")
            if m_nodes is None or m_edges is None:
                raise ValueError("zip must contain nodes.csv and edges.csv (possibly inside a subfolder).")

            with io.TextIOWrapper(zf.open(m_nodes, "r"), encoding="utf-8", errors="ignore") as f:
                node_ids = _read_int_list_from_fileobj(f)

            id_map = {int(x): i for i, x in enumerate(node_ids.tolist())}
            n = len(id_map)
            if n == 0:
                raise ValueError("nodes.csv appears empty/unreadable.")

            src, dst = [], []
            with io.TextIOWrapper(zf.open(m_edges, "r"), encoding="utf-8", errors="ignore") as f:
                for u0, v0 in _iter_edges_fileobj(f):
                    if u0 == v0:
                        continue
                    u = id_map.get(u0, None)
                    v = id_map.get(v0, None)
                    if u is None or v is None:
                        continue
                    src.append(u); dst.append(v)
    else:
        nodes_path = os.path.join(base_dir, "nodes.csv")
        edges_path = os.path.join(base_dir, "edges.csv")
        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            raise ValueError("base_dir must contain nodes.csv and edges.csv")

        with open(nodes_path, "rt", encoding="utf-8", errors="ignore") as f:
            node_ids = _read_int_list_from_fileobj(f)

        id_map = {int(x): i for i, x in enumerate(node_ids.tolist())}
        n = len(id_map)
        if n == 0:
            raise ValueError("nodes.csv appears empty/unreadable.")

        src, dst = [], []
        with open(edges_path, "rt", encoding="utf-8", errors="ignore") as f:
            for u0, v0 in _iter_edges_fileobj(f):
                if u0 == v0:
                    continue
                u = id_map.get(u0, None)
                v = id_map.get(v0, None)
                if u is None or v is None:
                    continue
                src.append(u); dst.append(v)

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    data = np.ones(src.size, dtype=np.float64)
    A = coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    A.sum_duplicates()
    if A.nnz:
        A.data[:] = 1.0
    return A.tocsr(), id_map


def dmax_from_csr(A) -> int:
    """For 0/1 CSR adjacency, max degree = max nnz per row."""
    # degrees = number of nonzeros in each row
    deg = np.diff(A.indptr).astype(np.int64)
    return int(deg.max(initial=0))


# -----------------------------
# eigs -> Xi_hat and lambdaK/lambdaKp1
# -----------------------------
def compute_Xi_and_lambdas_from_A(A, K: int, eig_sort: str = "abs", tol: float = 1e-3, maxiter: int = 200000):
    n = A.shape[0]
    k = int(min(K + 1, n - 2))
    if k < K + 1:
        raise ValueError(f"Need K+1 eigenpairs but n too small: n={n}, K={K}")

    try:
        vals, vecs = eigsh(A, k=k, which="LM", tol=tol, maxiter=maxiter)
    except ArpackNoConvergence as e:
        vals = e.eigenvalues
        vecs = e.eigenvectors
        if vals is None or vecs is None or vals.size < K + 1:
            raise RuntimeError(f"ARPACK did not converge enough eigenpairs: got {0 if vals is None else vals.size}")

    vals = np.asarray(vals, dtype=float)
    vecs = np.asarray(vecs, dtype=float)

    if eig_sort == "abs":
        key = np.abs(vals)
        ord_idx = np.argsort(key)[::-1]
        key_sorted = key[ord_idx]
    elif eig_sort == "raw_desc":
        ord_idx = np.argsort(vals)[::-1]
        key_sorted = vals[ord_idx]
    else:
        raise ValueError("eig_sort must be 'abs' or 'raw_desc'")

    Xi_hat = vecs[:, ord_idx[:K]].copy()
    lamK = float(key_sorted[K - 1])
    lamKp1 = float(key_sorted[K])
    return Xi_hat, lamK, lamKp1


# -----------------------------
# theta0_tilde from dmax(A)
# -----------------------------
def theta0_tilde_from_dmaxA(
    dmaxA: float,
    n: int,
    eps1: float,
    rng: Optional[np.random.Generator] = None,
    floor: float = 1e-12,
) -> Dict[str, float]:
    """
    thetatilde0(A) = sqrt( dmax(A)/n + Lap(1)/(eps1*n) ), clipped at 0 inside sqrt.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if eps1 is None or eps1 <= 0:
        raise ValueError("eps1 must be > 0.")
    if rng is None:
        rng = np.random.default_rng()

    lap = float(rng.laplace(loc=0.0, scale=1.0))
    inside = (float(dmaxA) / float(n)) + (lap / (float(eps1) * float(n)))
    inside_clipped = max(inside, 0.0)

    theta_tilde = float(np.sqrt(inside_clipped))
    theta_tilde = max(theta_tilde, float(floor))

    return {
        "theta0_tilde": theta_tilde,
        "lap": lap,
        "inside": float(inside),
        "inside_clipped": float(inside_clipped),
        "A_inf": float(dmaxA),
        "eps1": float(eps1),
    }


# -----------------------------
# UPDATED gamma_E + alpha (matches your revised code)
# -----------------------------
def _norm_2_infty(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    rn = np.sqrt((X * X).sum(axis=1))
    return float(np.max(rn)) if rn.size else 0.0

def compute_gamma_E(
    Xi_hat: np.ndarray,
    dmaxA: float,
    lambdaK: float,
    lambdaKp1: float,
    a0: float,
    A0: float,
    theta0: float = 1.0,
) -> dict:
    r"""
    gamma_E(A) = min{
      ( (1+a0)*theta0^2*n - ||A||_inf )/sqrt(2),
      ( lambdaK(A) - a0*theta0^2*n - 3*sqrt(2) )/sqrt(2),
      ( a0*theta0^2*n - lambda_{K+1}(A) )/sqrt(2),
      ( A0/sqrt(n) - ||Xi_A||_{2,inf} ) / U0
    }_+

    with U0:
      2*sqrt(2)*(1+a0)*A0/(a0^2*theta0^2*n*sqrt(n))
      + A0/(a0*theta0^2*n*sqrt(n))
      + sqrt(2)*A0^2/(a0*theta0^2*n^2)
      + 2*sqrt(2)*A0/(a0^2*theta0^2*n^2*sqrt(n))
      + 8*A0^3/(a0^2*theta0^2*n^3*sqrt(n)).
    """
    Xi_hat = np.asarray(Xi_hat, dtype=float)
    n = int(Xi_hat.shape[0])
    if n <= 0:
        raise ValueError("Xi_hat has zero rows.")
    if a0 <= 0 or A0 <= 0:
        raise ValueError("Require a0 > 0 and A0 > 0.")
    if theta0 <= 0:
        raise ValueError("Require theta0 > 0.")
    if dmaxA < 0:
        raise ValueError("Require dmaxA >= 0.")

    sq2 = np.sqrt(2.0)
    sqn = np.sqrt(float(n))
    th2 = float(theta0) ** 2

    xi_2inf = _norm_2_infty(Xi_hat)

    U0 = (
        (4.0 * sq2 + 1) * A0 / (a0 * th2 * n * sqn) #(2.0 * sq2 * (1.0 + a0) * A0) / (a0**2 * th2 * n * sqn)
        + (A0) / (a0 * th2 * n * sqn)
        + (sq2 * A0**2) / (a0 * th2 * n**2)
        + (5.0 * sq2 * A0) / (a0**2 * th2**2 * n**2 * sqn)
        + (50.0 * A0**3) / (a0**2 * th2**2 * n**3 * sqn)
    )

    t_deg = (((1.0 + a0) * th2 * n) - float(dmaxA)) / sq2
    t1 = (float(lambdaK) - a0 * th2 * n - 3.0 * sq2) / sq2
    t2 = (0.8 * a0 * th2 * n - float(lambdaKp1)) / sq2
    t3 = (A0 / sqn - xi_2inf) / float(U0)
    
    gamma_E = max(min(t_deg, t1, t2, t3), 0.0)

    # "good set" diagnostics (same structure as your runner)
    cond_deg = ((1.0 + a0) * th2 * n >= float(dmaxA))
    cond_lamK = (float(lambdaK) >= a0 * th2 * n + 3.0 * sq2)
    cond_lamKp1 = (float(lambdaKp1) <= a0 * th2 * n)
    cond_xi = (xi_2inf <= A0 / sqn)
    good_set = bool(cond_deg and cond_lamK and cond_lamKp1 and cond_xi)

    return {
        "n": n,
        "theta0": float(theta0),
        "dmaxA": float(dmaxA),
        "xi_2inf": float(xi_2inf),
        "U0": float(U0),
        "t_deg": float(t_deg),
        "t1": float(t1),
        "t2": float(t2),
        "t3": float(t3),
        "gamma_E": float(gamma_E),
        "good_set": bool(good_set),
        "cond_deg": bool(cond_deg),
        "cond_lamK": bool(cond_lamK),
        "cond_lamKp1": bool(cond_lamKp1),
        "cond_xi": bool(cond_xi),
    }

def alpha_netptr(n: int, K: int, a0: float, A0: float) -> float:
    n = int(n); K = int(K)
    sqn = np.sqrt(float(n))
    return (2.0 * np.sqrt(2.0 * K) * A0) / (a0 * n * sqn) + (16.0 * np.sqrt(float(K))) / (a0 ** 2 * n ** 2)


# -----------------------------
# ONE-SHOT tester: one network, one eps, one rep; scan a0 (UPDATED)
# -----------------------------

def _induced_subgraph(A, nodes: np.ndarray):
    """Return the induced subgraph A[nodes, nodes] as symmetric 0/1 CSR."""
    nodes = np.asarray(nodes, dtype=np.int64)
    Asub = A[nodes][:, nodes].tocsr()
    Asub.setdiag(0)
    Asub.eliminate_zeros()
    Asub.sum_duplicates()
    if Asub.nnz:
        Asub.data[:] = 1.0
    return Asub

def _sample_subgraph_nodes(n: int, subgraph_n: int, rng: np.random.Generator):
    if subgraph_n <= 0 or subgraph_n > n:
        raise ValueError(f"subgraph_n must be in [1, n], got {subgraph_n} with n={n}")
    return np.sort(rng.choice(n, size=subgraph_n, replace=False))

def _compute_netptr_labels_on_A(
    A_sub,
    K: int,
    eps: float,
    delta: float,
    a0: float,
    A0: float,
    A0_mode: str,
    A0_factor: float,
    theta0_input: float,
    theta0_mode: str,
    eps1: Optional[float],
    theta0_floor: float,
    eig_sort: str,
    eig_tol: float,
    eig_maxiter: int,
    kmeans_nstart: int,
    norm_tol: float,
    seed_base_local: int,
    rep: int,
    eps_tag: int,
    eps1_tag: int,
    verbose: bool = False,
):
    """
    Run non-private SC on A_sub to get baseline labels_sc,
    then run NetPTR on the SAME subgraph and return ARI against labels_sc.
    """
    n = int(A_sub.shape[0])
    dmaxA = float(dmax_from_csr(A_sub))

    Xi_hat, lamK, lamKp1 = compute_Xi_and_lambdas_from_A(
        A_sub, K=K, eig_sort=eig_sort, tol=float(eig_tol), maxiter=int(eig_maxiter)
    )
    Xi_hat = np.asarray(Xi_hat, dtype=float)

    if A0_mode == "fixed":
        A0_use = float(A0)
    elif A0_mode == "from_2toinfty":
        xi_2inf = _norm_2_infty(Xi_hat)
        A0_use = (1.0 + float(A0_factor)) * np.sqrt(float(n)) * float(xi_2inf)
    else:
        raise ValueError("A0_mode must be 'fixed' or 'from_2toinfty'.")

    # shared randomness within this (rep, subgraph, eps), so different eps1 compare fairly
    rng_shared = np.random.default_rng(seed_base_local + 100000 * rep + 1000 * eps_tag)
    u_accept = float(rng_shared.random())
    X_rand = rng_shared.random(size=(n, K))
    seed_km = int(rng_shared.integers(1, 1_000_000))

    # keep zeta fixed across eps1 for fairness
    zeta = np.random.default_rng(seed_base_local + 7777 + rep).standard_normal(size=(n, K))

    labels_sc = _cluster_from_embedding(
        Xi_hat, K, nstart=kmeans_nstart, norm_tol=norm_tol, seed=seed_km
    )

    # theta0 choice
    theta0_used = float(theta0_input)
    theta0_tilde = np.nan
    lap_theta0 = np.nan
    eps1_used = np.nan

    if theta0_mode != "fixed":
        if eps1 is None or float(eps1) <= 0:
            raise ValueError("eps1 must be > 0 when theta0_mode != 'fixed'.")
        eps1_used = float(eps1)
        rng_th = np.random.default_rng(seed_base_local + 500000 * rep + 1000 * eps_tag + 17 * eps1_tag)
        th_info = theta0_tilde_from_dmaxA(
            dmaxA=dmaxA,
            n=n,
            eps1=float(eps1_used),
            rng=rng_th,
            floor=float(theta0_floor),
        )
        theta0_tilde = float(th_info["theta0_tilde"])
        lap_theta0 = float(th_info["lap"])

        if theta0_mode == "use_tilde":
            theta0_used = theta0_tilde
        elif theta0_mode == "floor_by_tilde":
            theta0_used = max(float(theta0_input), theta0_tilde)
        else:
            raise ValueError("theta0_mode must be 'fixed', 'use_tilde', or 'floor_by_tilde'.")

    g = compute_gamma_E(
        Xi_hat=Xi_hat,
        dmaxA=dmaxA,
        lambdaK=float(lamK),
        lambdaKp1=float(lamKp1),
        a0=float(a0),
        A0=float(A0_use),
        theta0=float(theta0_used),
    )
    gamma_E = float(g["gamma_E"])

    M = 1.0 + (2.0 / float(eps)) * np.log(1.0 / float(delta))
    betaA = gamma_E
    if betaA > 2.0 * M:
        pA = 1.0
    else:
        x = 0.5 * float(eps) * (betaA - M)
        pA = float(_stable_sigmoid(x))

    accept = (u_accept < pA)
    alpha = float(alpha_netptr(n=n, K=K, a0=float(a0), A0=float(A0_use)))
    noise_scale = (alpha / float(eps)) * np.sqrt(2.0 * np.log(1.25 / float(delta)))

    Xi_tilde = (Xi_hat + noise_scale * zeta) if accept else X_rand
    labels_gap = _cluster_from_embedding(
        Xi_tilde, K, nstart=kmeans_nstart, norm_tol=norm_tol, seed=seed_km
    )
    ari_gap = float(adjusted_rand_score(labels_sc, labels_gap))

    if verbose:
        print(
            f"    [NetPTR eps={eps:.3g}, eps1={eps1}] "
            f"ARI={ari_gap:.4f}, accept={accept}, pA={pA:.4g}, gamma_E={gamma_E:.4g}"
        )

    return {
        "labels_sc": labels_sc,
        "ARI_NetPTR_vs_SC": ari_gap,
        "accept": bool(accept),
        "pA": float(pA),
        "gamma_E": float(gamma_E),
        "noise_scale": float(noise_scale),
        "theta0_used": float(theta0_used),
        "theta0_tilde": float(theta0_tilde) if np.isfinite(theta0_tilde) else np.nan,
        "lap_theta0": float(lap_theta0) if np.isfinite(lap_theta0) else np.nan,
        "eps1_used": float(eps1_used) if np.isfinite(eps1_used) else np.nan,
        "A0_use": float(A0_use),
        "dmaxA": float(dmaxA),
        "nnz": int(A_sub.nnz),
        "good_set": bool(g["good_set"]),
        "t_deg": float(g["t_deg"]),
        "t1": float(g["t1"]),
        "t2": float(g["t2"]),
        "t3": float(g["t3"]),
    }

def edgeflip_specclu_from_A(
    A,
    K: int,
    eps: float,
    seed_flip: int,
    seed_km: int,
    nstart: int = 25,
    norm_tol: float = 1e-12,
    max_edges_upper_priv: int = 1e10,
):
    A = A.tocsr()
    n = int(A.shape[0])

    ei, ej = _upper_edges_from_csr(A)
    m = int(ei.size)

    # precompute codes for orig edges (sorted int64)
    codes_orig = (ei * np.int64(n) + ej).astype(np.int64)
    codes_orig_sorted = np.sort(codes_orig)
    prefix = _tri_prefix(n)

    rng = np.random.default_rng(seed_flip)

    ei_priv, ej_priv, q, add_m, kept_m = _edgeflip_upper_edges_rr(
        n=n, ei=ei, ej=ej, eps=float(eps), rng=rng,
        codes_orig_sorted=codes_orig_sorted, prefix=prefix,
        max_edges_upper_priv=max_edges_upper_priv,
    )
   
    A_priv = _build_sparse_A_from_edges(n, ei_priv, ej_priv)

    vals, vecs = eigsh(A_priv, k=K + 1, which="LM")
    order = np.argsort(np.abs(vals))[::-1]
    vecs = vecs[:, order]

    Xi = vecs[:, :K]
    labels = _cluster_from_embedding(Xi, K, nstart=nstart, norm_tol=norm_tol, seed=seed_km)

    return {
        "labels": labels,
        "q_flip": float(q),
        "m_upper": m,
        "kept_upper": kept_m,
        "add_upper": add_m,
        "m_upper_priv": int(ei_priv.size),
    }




# Backward-compatible aliases for old notebooks/scripts.
compute_gamma_E_updated = compute_gamma_E
alpha_improved = alpha_netptr
_compute_gaptr_labels_on_A = _compute_netptr_labels_on_A
