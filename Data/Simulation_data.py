from __future__ import annotations

import os
import json
import numpy as np
import math, struct
from typing import Optional, Tuple

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def _save_theta_z_csv(
    path: str,
    theta: np.ndarray,
    z0: np.ndarray,
    node_base: int = 0,
    z_base: int = 1,
    theta_decimals: int = 8,
) -> None:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    z0 = np.asarray(z0, dtype=int).reshape(-1)
    if theta.shape[0] != z0.shape[0]:
        raise ValueError("theta and z0 must have the same length.")

    n = theta.shape[0]
    nodes = (np.arange(n) + node_base).astype(np.int64)
    z_out = (z0 + z_base).astype(np.int64)
    th = np.round(theta, decimals=theta_decimals)

    out = np.column_stack([nodes, z_out, th])
    fmt = ["%d", "%d", f"%.{theta_decimals}f"]
    np.savetxt(path, out, delimiter=",", header="node,z,theta", comments="", fmt=fmt)


def _save_matrix_csv(
    path: str,
    M: np.ndarray,
    decimals: int = 8,
    row_base: int = 1,
    col_base: int = 1,
) -> None:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square 2D array.")
    K = M.shape[0]
    Mr = np.round(M, decimals=decimals)

    col_names = [f"k{j+col_base}" for j in range(K)]
    header = ",".join(["row"] + col_names)

    rows = (np.arange(K) + row_base).astype(int).reshape(-1, 1)
    out = np.concatenate([rows, Mr], axis=1)

    fmt = ["%d"] + [f"%.{decimals}f"] * K
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt=fmt)


def _save_omega_components(
    rep_dir: str,
    theta: np.ndarray,
    z0: np.ndarray,
    P0: np.ndarray,
    P_eff: np.ndarray,
    scale_s: float,
    node_base: int = 0,
    z_base: int = 1,
) -> None:
    # 1) CSVs (easy to inspect / read in R)
    _save_theta_z_csv(
        os.path.join(rep_dir, "omega_theta_z.csv"),
        theta=theta, z0=z0,
        node_base=node_base, z_base=z_base,
        theta_decimals=8,
    )
    _save_matrix_csv(os.path.join(rep_dir, "P0.csv"), P0, decimals=8, row_base=1, col_base=1)
    _save_matrix_csv(os.path.join(rep_dir, "P_eff.csv"), P_eff, decimals=8, row_base=1, col_base=1)

    # 2) NPZ (compact + lossless-ish; float32 is usually enough)
    np.savez_compressed(
        os.path.join(rep_dir, "omega_components.npz"),
        theta=theta.astype(np.float32),
        z0=z0.astype(np.int32),
        P0=P0.astype(np.float32),
        P_eff=P_eff.astype(np.float32),
        scale_s=np.array([scale_s], dtype=np.float32),
        node_base=np.array([node_base], dtype=np.int32),
        z_base=np.array([z_base], dtype=np.int32),
    )


def omega_top_eigs_lowrank(
    theta: np.ndarray,
    z0: np.ndarray,
    P_eff: np.ndarray,
    Keig: int,
    eig_order: str = "abs"
) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    z0 = np.asarray(z0, dtype=int)
    P_eff = np.asarray(P_eff, dtype=float)

    n = theta.shape[0]
    K = P_eff.shape[0]
    if P_eff.shape != (K, K):
        raise ValueError("P_eff must be K x K.")
    if not (1 <= Keig <= K):
        raise ValueError("Require 1 <= Keig <= K.")
    if z0.min() < 0 or z0.max() >= K:
        raise ValueError("z0 must be in {0,...,K-1}.")

    d = np.bincount(z0, weights=theta**2, minlength=K).astype(float)
    if np.any(d <= 0):
        raise ValueError("Each community must have at least one node with theta>0.")

    D_sqrt = np.sqrt(d)
    D_inv_sqrt = 1.0 / D_sqrt

    B = (D_sqrt[:, None] * P_eff) * D_sqrt[None, :]
    lam, V = np.linalg.eigh(B)  # ascending

    if eig_order == "abs":
        order = np.argsort(np.abs(lam))[::-1]
    elif eig_order == "value":
        order = np.argsort(lam)[::-1]
    else:
        raise ValueError("eig_order must be 'abs' or 'value'.")

    lam = lam[order]
    V = V[:, order]

    lam_top = lam[:Keig]
    V_top = V[:, :Keig]

    DV = (D_inv_sqrt[:, None] * V_top)   # K x Keig
    Xi_top = theta[:, None] * DV[z0, :]  # n x Keig
    return lam_top.astype(float), Xi_top.astype(float)


def _make_balanced_sizes(n: int, K: int) -> np.ndarray:
    base = n // K
    sizes = np.full(K, base, dtype=int)
    rem = n - sizes.sum()
    if rem > 0:
        sizes[:rem] += 1
    return sizes

def _sample_sizes_unbalanced(n: int, K: int, dirichlet_alpha: float, rng: np.random.Generator) -> np.ndarray:
    alpha_vec = np.full(K, dirichlet_alpha, dtype=float)
    g = rng.gamma(shape=alpha_vec, scale=1.0)
    probs = g / g.sum()
    sizes = np.floor(probs * n).astype(int)
    rem = n - sizes.sum()
    if rem > 0:
        frac = probs - sizes / n
        idx_add = np.argsort(-frac)[:rem]
        sizes[idx_add] += 1
    return sizes

def _build_P0(K: int, p_in: float, p_out: float, P_noise: float, rng: np.random.Generator) -> np.ndarray:
    P0 = np.full((K, K), p_out, dtype=float)
    np.fill_diagonal(P0, p_in)
    if P_noise > 0:
        noise = rng.uniform(-P_noise, P_noise, size=(K, K))
        noise = (noise + noise.T) / 2.0
        np.fill_diagonal(noise, 0.0)
        P0 = P0 + noise
    return np.clip(P0, 0.0, 1.0)

def _generate_theta(
    n: int,
    theta_dist: str,
    theta_min: float,
    theta_max: float,
    theta_beta: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    if theta_dist == "uniform":
        theta = rng.uniform(theta_min, theta_max, size=n)

    elif theta_dist == "uniform_mixture_low":
        # Main group: 60% of nodes have theta in [theta_min, theta_max].
        # Low-degree group: 40% of nodes have theta in
        # [0.1 * theta_min, 0.1 * theta_max].
        theta = rng.uniform(theta_min, theta_max, size=n)
        low_mask = rng.random(n) < 0.4
        theta[low_mask] = rng.uniform(
            0.1 * theta_min,
            0.1 * theta_max,
            size=int(low_mask.sum()),
        )

    elif theta_dist == "beta":
        a, b = theta_beta
        u = rng.beta(a, b, size=n)
        theta = theta_min + (theta_max - theta_min) * u

    else:
        raise ValueError(
            "theta_dist must be 'uniform', 'uniform_mixture_low', or 'beta'"
        )

    return theta.astype(float)


def _scale_P_to_keep_probs_le1(theta: np.ndarray, z0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, float]:
    K = P0.shape[0]
    max_theta_by_comm = np.zeros(K, dtype=float)
    for k in range(K):
        mask = (z0 == k)
        if not np.any(mask):
            raise ValueError("Empty community encountered.")
        max_theta_by_comm[k] = float(theta[mask].max())

    M = (max_theta_by_comm[:, None] * max_theta_by_comm[None, :]) * P0
    np.fill_diagonal(M, 0.0)
    max_prob = float(M.max(initial=0.0))
    s = 1.0 if max_prob <= 1.0 else (1.0 / max_prob)
    return (s * P0).astype(float), float(s)


def _sample_edges_and_write_csv(
    path_csv: str,
    n: int,
    theta: np.ndarray,
    z0: np.ndarray,
    P_eff: np.ndarray,
    rng: np.random.Generator,
    block: int = 512,
    index_base: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    z0 = np.asarray(z0, dtype=int)

    ei_chunks = []
    ej_chunks = []

    with open(path_csv, "w") as f:
        f.write("i,j\n")

        for i0 in range(0, n, block):
            i1 = min(n, i0 + block)
            I = np.arange(i0, i1)
            tI = theta[I]
            zI = z0[I]

            for j0 in range(i0, n, block):
                j1 = min(n, j0 + block)
                J = np.arange(j0, j1)
                tJ = theta[J]
                zJ = z0[J]

                probs = (tI[:, None] * tJ[None, :]) * P_eff[zI[:, None], zJ[None, :]]
                probs = np.clip(probs, 0.0, 1.0)

                U = rng.random(probs.shape)

                if i0 == j0:
                    mask = np.triu(np.ones_like(probs, dtype=bool), k=1)
                    hit = (U < probs) & mask
                else:
                    hit = (U < probs)

                if not np.any(hit):
                    continue

                ii_loc, jj_loc = np.nonzero(hit)
                ei = (I[ii_loc]).astype(np.int64)
                ej = (J[jj_loc]).astype(np.int64)

                # write (with optional 1-based etc.)
                out = np.column_stack([ei + index_base, ej + index_base])
                np.savetxt(f, out, fmt="%d", delimiter=",")

                ei_chunks.append(ei)
                ej_chunks.append(ej)

    if ei_chunks:
        ei_all = np.concatenate(ei_chunks)
        ej_all = np.concatenate(ej_chunks)
    else:
        ei_all = np.empty(0, dtype=np.int64)
        ej_all = np.empty(0, dtype=np.int64)

    return ei_all, ej_all

def _save_A_upper_triangle_bitpack(
    path: str,
    n: int,
    ei: np.ndarray,
    ej: np.ndarray,
    index_base: int = 0,   # stored in header only (for reference)
    bitorder: str = "little",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ei = np.asarray(ei, dtype=np.int64)
    ej = np.asarray(ej, dtype=np.int64)

    # --- detect whether edges are 0-based or 1-based ---
    if ei.size:
        mn = int(min(ei.min(), ej.min()))
        mx = int(max(ei.max(), ej.max()))

        if mn >= 0 and mx < n:
            input_base = 0
        elif mn >= 1 and mx <= n:
            input_base = 1
            ei = ei - 1
            ej = ej - 1
        else:
            raise ValueError(
                f"Edge indices out of range. Detected min={mn}, max={mx}, n={n}. "
                f"Expected 0-based in [0,{n-1}] or 1-based in [1,{n}]."
            )

    # ensure i<j (swap if needed)
    swap = ei > ej
    if np.any(swap):
        tmp = ei[swap].copy()
        ei[swap] = ej[swap]
        ej[swap] = tmp

    # keep only strict upper triangle
    mask = ei < ej
    ei = ei[mask]
    ej = ej[mask]

    # sort by (i, j)
    if ei.size:
        order = np.lexsort((ej, ei))
        ei = ei[order]
        ej = ej[order]

    bitflag = 1 if bitorder == "little" else 0

    with open(path, "wb") as f:
        # header
        f.write(b"ABIT")
        f.write(struct.pack("<I", int(n)))
        f.write(struct.pack("<B", int(index_base)))  # just metadata for you
        f.write(struct.pack("<B", int(bitflag)))

        if ei.size:
            rows = np.arange(n - 1, dtype=np.int64)
            start = np.searchsorted(ei, rows, side="left")
            end   = np.searchsorted(ei, rows, side="right")
        else:
            start = end = None

        for i in range(n - 1):
            m = n - i - 1
            nbytes = (m + 7) // 8
            buf = np.zeros(nbytes, dtype=np.uint8)

            if ei.size:
                s = start[i]
                t = end[i]
                if t > s:
                    js = ej[s:t]
                    pos = js - (i + 1)      # 0..m-1
                    byte_idx = pos >> 3
                    bit = pos & 7

                    if bitorder == "little":
                        maskbits = (1 << bit).astype(np.uint8)
                    else:
                        maskbits = (1 << (7 - bit)).astype(np.uint8)

                    np.bitwise_or.at(buf, byte_idx, maskbits)

            f.write(buf.tobytes())

def _build_sparse_A_from_edges(n: int, ei: np.ndarray, ej: np.ndarray) -> coo_matrix:
    m = ei.size
    if m == 0:
        return coo_matrix((n, n), dtype=float)

    data = np.ones(2 * m, dtype=float)
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    A = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=float).tocsr()
    return A


def _top_abs_eigs_sparse_A(A_csr, K: int) -> Tuple[np.ndarray, np.ndarray]:
    n = A_csr.shape[0]
    if K + 1 >= n:
        raise ValueError(f"Need K+1 < n for eigsh; got K={K}, n={n}.")
    vals, vecs = eigsh(A_csr, k=K + 1, which="LM")
    order = np.argsort(np.abs(vals))[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return vals.astype(float), vecs[:, :K].astype(float)

def _save_eigvals_with_abs_csv(path: str, vals: np.ndarray, decimals: int = 6) -> None:
    vals = np.asarray(vals, dtype=float).reshape(-1)
    vals = np.round(vals, decimals=decimals)
    absvals = np.round(np.abs(vals), decimals=decimals)

    out = np.column_stack([vals, absvals])
    fmt = [f"%.{decimals}f", f"%.{decimals}f"]
    np.savetxt(path, out, delimiter=",", header="eigval,abs_eigval", comments="", fmt=fmt)

def _save_eigvecs_csv(
    path: str,
    vecs: np.ndarray,
    node_base: int = 0,
    vec_decimals: int = 6,
    z: Optional[np.ndarray] = None,
    z_base: int = 1,
) -> None:
    vecs = np.asarray(vecs, dtype=float)
    n, K = vecs.shape
    nodes = (np.arange(n) + node_base).astype(np.int64)

    vecs = np.round(vecs, decimals=vec_decimals)
    vec_fmt = f"%.{vec_decimals}f"

    if z is None:
        out = np.column_stack([nodes, vecs])
        header = "node," + ",".join([f"vec{k+1}" for k in range(K)])
        fmt = ["%d"] + [vec_fmt] * K
    else:
        z = np.asarray(z, dtype=int).reshape(-1)
        if z.shape[0] != n:
            raise ValueError("z must have length n.")
        z_out = (z + z_base).astype(np.int64) 
        out = np.column_stack([nodes, z_out, vecs])
        header = "node,z," + ",".join([f"vec{k+1}" for k in range(K)])
        fmt = ["%d", "%d"] + [vec_fmt] * K

    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt=fmt)

def save_network(
    out_dir: str,
    n_rep: int,
    n: int = 1000,
    K: int = 3,
    balanced: bool = True,
    p_in: float = 0.15,
    p_out: float = 0.03,
    P_noise: float = 0.0,
    theta_dist: str = "uniform",
    theta_min: float = 0.7,
    theta_max: float = 1.3,
    theta_beta: Tuple[float, float] = (2.0, 5.0),
    min_comm_prop: float = 0.05,
    dirichlet_alpha: float = 1.0,
    max_tries: int = 100,
    eig_order_omega: str = "abs",
    block: int = 512,
    index_base_edges: int = 0,
    base_seed: Optional[int] = 0,
) -> None:

    os.makedirs(out_dir, exist_ok=True)
    width = max(3, len(str(n_rep - 1))) if n_rep > 1 else 3

    if not (n > K and K >= 2):
        raise ValueError("Require n > K and K >= 2.")
    if eig_order_omega not in {"abs", "value"}:
        raise ValueError("eig_order_omega must be 'abs' or 'value'.")
    if not (0 < min_comm_prop <= 1 / K):
        raise ValueError("Require 0 < min_comm_prop <= 1/K.")

    min_size = int(np.ceil(min_comm_prop * n))

    for rep in range(n_rep):
        rep_dir = os.path.join(out_dir, f"rep{rep:0{width}d}")
        os.makedirs(rep_dir, exist_ok=True)

        seed = None if base_seed is None else int(base_seed + rep)
        rng = np.random.default_rng(seed)

        # ---- 1) sample community sizes + labels ----
        attempt = 1
        while True:
            sizes = _make_balanced_sizes(n, K) if balanced else _sample_sizes_unbalanced(n, K, dirichlet_alpha, rng)
            if sizes.min() >= min_size:
                break
            attempt += 1
            if attempt > max_tries:
                raise RuntimeError(
                    f"Failed to sample community sizes with min >= {min_size} after {max_tries} tries."
                )

        z0 = np.repeat(np.arange(K), sizes)
        rng.shuffle(z0)

        # ---- 2) P and theta ----
        P0 = _build_P0(K, p_in, p_out, P_noise, rng)
        theta = _generate_theta(n, theta_dist, theta_min, theta_max, theta_beta, rng)
        P_eff, scale_s = _scale_P_to_keep_probs_le1(theta, z0, P0)

        _save_omega_components(
            rep_dir=rep_dir,
            theta=theta,
            z0=z0,
            P0=P0,
            P_eff=P_eff,
            scale_s=scale_s,
            node_base=index_base_edges,
            z_base=1,
        )

        # ---- 3) sample A edges ----
        edges_csv = os.path.join(rep_dir, "A_edges.csv")
        ei, ej = _sample_edges_and_write_csv(
            path_csv=edges_csv,
            n=n,
            theta=theta,
            z0=z0,
            P_eff=P_eff,
            rng=rng,
            block=block,
            index_base=index_base_edges,
        )
        bit_path = os.path.join(rep_dir, "A_upper_bitpack.bin")
        _save_A_upper_triangle_bitpack(
            path=bit_path,
            n=n,
            ei=ei,
            ej=ej,
            index_base=index_base_edges,
            bitorder="little",
        )
        try:
            os.remove(edges_csv)
        except Exception:
            pass

        A_csr = _build_sparse_A_from_edges(n, ei, ej)

        deg = np.diff(A_csr.indptr)          # degree for each node (since A is 0/1 CSR)
        d_max = int(deg.max(initial=0))      # safe if n=0 edge-case; here n>0 anyway


        # ---- 4) A eigs ----
        A_vals, A_vecs = _top_abs_eigs_sparse_A(A_csr, K=K)
        _save_eigvals_with_abs_csv(os.path.join(rep_dir, "A_eigvals_topKp1.csv"), A_vals, decimals=4)
        _save_eigvecs_csv(
            os.path.join(rep_dir, "A_eigvecs_topK.csv"),
            A_vecs,
            node_base=index_base_edges,
            vec_decimals=6,
            z=z0,
            z_base=1
        )

        # ---- 5) Omega eigs ----
        Om_vals, Om_vecs = omega_top_eigs_lowrank(theta, z0, P_eff, Keig=K, eig_order=eig_order_omega)
        _save_eigvals_with_abs_csv(os.path.join(rep_dir, "Omega_eigvals_topK.csv"), Om_vals, decimals=4)
        _save_eigvecs_csv(
            os.path.join(rep_dir, "Omega_eigvecs_topK.csv"),
            Om_vecs,
            node_base=index_base_edges,
            vec_decimals=6,
            z=z0,
            z_base=1
        )

        # ---- 6) meta ----
        meta = {
            "rep": int(rep),
            "seed": seed,
            "n": int(n),
            "K": int(K),
            "balanced": bool(balanced),
            "sizes": sizes.tolist(),
            "attempts": int(attempt),
            "p_in": float(p_in),
            "p_out": float(p_out),
            "P_noise": float(P_noise),
            "theta_dist": theta_dist,
            "theta_min": float(theta_min),
            "theta_max": float(theta_max),
            "theta_beta": [float(theta_beta[0]), float(theta_beta[1])],
            "min_comm_prop": float(min_comm_prop),
            "dirichlet_alpha": float(dirichlet_alpha),
            "scale_s": float(scale_s),
            "eig_order_omega": eig_order_omega,
            "block": int(block),
            "index_base_edges": int(index_base_edges),
            "num_edges_upper": int(ei.size),
            "d_max": int(d_max),   # ---- NEW ----
            "omega_saved_files": [
                "omega_theta_z.csv",
                "P0.csv",
                "P_eff.csv",
                "omega_components.npz",
            ],
        }
        with open(os.path.join(rep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


##example
save_network(
    out_dir="./theta14",
    n_rep = 10,
    n=20000,
    K=2,
    balanced=True,
    #min_comm_prop = 0.4,
    p_in=0.4,
    p_out=0.1,
    theta_dist = "uniform",
    theta_min = 0.6,
    theta_max = 0.9,
    base_seed=123,
    index_base_edges=1,
)
