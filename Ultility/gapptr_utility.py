"""
GapPTR simulation utilities.

This module contains reusable helper functions for Experiments 1--3.  It is
intended to be placed under ``Utility/`` and imported by experiment-specific
scripts, rather than executed directly.

The functions in this file assume that each simulated network replication is
stored in a folder such as ``rep000`` or ``rep_000`` and contains the spectral
and population files produced by the data-generation utility, for example

    A_eigvecs_topK.csv
    A_eigvals_topKp1.csv
    meta.json                         # preferred source for d_max(A)
    A_upper_bitpack.bin               # fallback source for d_max(A)
    omega_components.npz              # preferred source for oracle theta0

Main public functions
---------------------
``run_improved_gapptr_from_saved``
    Run the improved GapPTR mechanism on saved simulation replications and
    return a pandas DataFrame of diagnostics and clustering errors.

``compute_gamma_E``
    Compute the empirical stability margin used by the precheck step.

``theta0_oracle_from_omega``
    Compute the oracle degree-scale parameter from saved population quantities.

``theta0_tilde_from_dmaxA``
    Compute the noisy DP-style degree-scale estimate from d_max(A).
"""

from __future__ import annotations

import json
import os
import re
import struct
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# -----------------------------------------------------------------------------
# Basic clustering helpers
# -----------------------------------------------------------------------------


def _stable_sigmoid(x: float) -> float:
    """Evaluate the logistic sigmoid without numerical overflow."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def _cluster_from_embedding(
    Xi: np.ndarray,
    K: int,
    nstart: int = 25,
    norm_tol: float = 1e-12,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Row-normalize a spectral embedding and apply k-means.

    Parameters
    ----------
    Xi:
        ``n x K`` embedding matrix.
    K:
        Number of clusters.
    nstart:
        Number of k-means initializations.
    norm_tol:
        Minimum row norm used to avoid division by zero.
    seed:
        Random seed for k-means.
    """
    Xi = np.asarray(Xi, dtype=float)
    row_norm = np.sqrt((Xi * Xi).sum(axis=1))
    row_norm = np.maximum(row_norm, norm_tol)
    X_normalized = Xi / row_norm[:, None]

    km = KMeans(n_clusters=K, n_init=nstart, random_state=seed)
    return km.fit_predict(X_normalized)


def clustering_error(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """
    Compute clustering error after optimal label permutation.

    The returned value is

        1 - max_permutation accuracy.

    This is appropriate for comparing community labels because labels are only
    identifiable up to permutation.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if labels_true.shape != labels_pred.shape:
        raise ValueError("labels_true and labels_pred must have the same shape.")

    n = labels_true.size
    if n == 0:
        return 0.0

    true_vals, true_inv = np.unique(labels_true, return_inverse=True)
    pred_vals, pred_inv = np.unique(labels_pred, return_inverse=True)
    k_true, k_pred = true_vals.size, pred_vals.size

    counts = np.zeros((k_true, k_pred), dtype=int)
    np.add.at(counts, (true_inv, pred_inv), 1)

    try:
        from scipy.optimize import linear_sum_assignment

        max_count = counts.max() if counts.size else 0
        cost = max_count - counts
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = int(counts[row_ind, col_ind].sum())
    except Exception:
        # Fallback used only if scipy.optimize is unavailable.
        matched = 0
        used_rows, used_cols = set(), set()
        pairs = [(counts[i, j], i, j) for i in range(k_true) for j in range(k_pred)]
        pairs.sort(reverse=True)
        for count_ij, i, j in pairs:
            if i in used_rows or j in used_cols:
                continue
            used_rows.add(i)
            used_cols.add(j)
            matched += int(count_ij)

    return 1.0 - matched / n


# -----------------------------------------------------------------------------
# Loading saved spectral information
# -----------------------------------------------------------------------------


def load_Xi_hat_from_csv(path: str, K: int) -> np.ndarray:
    """Load the top-``K`` sample eigenvectors from ``A_eigvecs_topK.csv``."""
    dat = np.genfromtxt(path, delimiter=",", names=True)
    nodes = dat["node"].astype(int)
    order = np.argsort(nodes)
    Xi_hat = np.column_stack([dat[f"vec{i + 1}"] for i in range(K)])[order, :]
    return Xi_hat.astype(float)


def load_eigs_topKp1_from_csv(path: str, K: int, sort_by: str = "abs") -> dict:
    """
    Load the top eigenvalues and return lambda_K and lambda_{K+1}.

    Parameters
    ----------
    path:
        Path to ``A_eigvals_topKp1.csv``.
    K:
        Number of target communities.
    sort_by:
        ``"abs"`` sorts by decreasing absolute value; ``"raw_desc"`` sorts by
        decreasing signed eigenvalue.
    """
    dat = np.genfromtxt(path, delimiter=",", names=True)
    if dat.dtype.names is None or "eigval" not in dat.dtype.names:
        raise ValueError(f"Expected a named-column CSV with an 'eigval' column: {path}")

    eigval = np.asarray(dat["eigval"], dtype=float)
    abs_eig = np.asarray(dat["abs_eigval"], dtype=float) if "abs_eigval" in dat.dtype.names else np.abs(eigval)

    if sort_by not in {"abs", "raw_desc"}:
        raise ValueError("sort_by must be 'abs' or 'raw_desc'.")

    if sort_by == "abs":
        order = np.argsort(abs_eig)[::-1]
        key_sorted = abs_eig[order]
        eig_sorted = eigval[order]
    else:
        order = np.argsort(eigval)[::-1]
        key_sorted = eigval[order]
        eig_sorted = eigval[order]

    if key_sorted.size < K + 1:
        raise ValueError(f"Need at least K+1 eigenvalues; got {key_sorted.size} in {path}")

    return {
        "lambdaK": float(key_sorted[K - 1]),
        "lambdaKp1": float(key_sorted[K]),
        "eig_sorted": eig_sorted[: K + 1].copy(),
        "key_sorted": key_sorted[: K + 1].copy(),
    }


def try_load_labels_true(rep_dir: str) -> Optional[np.ndarray]:
    """
    Try to load true community labels from ``A_eigvecs_topK.csv``.

    The data-generation utility stores the true label column ``z`` together with
    the eigenvectors.  If that column is not available, this function returns
    ``None`` instead of failing.
    """
    path = os.path.join(rep_dir, "A_eigvecs_topK.csv")
    if not os.path.exists(path):
        return None

    dat = np.genfromtxt(path, delimiter=",", names=True)
    if isinstance(dat, np.ndarray) and dat.dtype.names is not None and "z" in dat.dtype.names:
        z = dat["z"].astype(int)
        if "node" in dat.dtype.names:
            nodes = dat["node"].astype(int)
            z = z[np.argsort(nodes)]
        return z

    raw = np.genfromtxt(path, delimiter=",")
    return np.asarray(raw).reshape(-1).astype(int)


# -----------------------------------------------------------------------------
# Loading or computing d_max(A) = ||A||_infty for 0/1 adjacency matrices
# -----------------------------------------------------------------------------


def _try_read_max_degree_from_report_json(path: str) -> Optional[int]:
    """Read max degree from a report-style JSON file, if available."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        key_paths = [
            ("basic_stats", "max_degree"),
            ("basic_stats", "maxDegree"),
            ("max_degree",),
            ("maxDegree",),
        ]
        for key_path in key_paths:
            cur = obj
            ok = True
            for key in key_path:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    ok = False
                    break
            if ok:
                return int(cur)
    except Exception:
        return None
    return None


def _try_read_dmax_from_degrees_csv(path: str) -> Optional[int]:
    """Read max degree from a degree CSV, if available."""
    try:
        deg = np.genfromtxt(path, delimiter=",", names=True)
        if isinstance(deg, np.ndarray) and deg.dtype.names is not None:
            if "degree" in deg.dtype.names:
                values = np.asarray(deg["degree"], dtype=float)
            else:
                first_col = list(deg.dtype.names)[0]
                values = np.asarray(deg[first_col], dtype=float)
        else:
            values = np.asarray(deg, dtype=float).reshape(-1)

        if values.size == 0:
            return None
        return int(np.nanmax(values))
    except Exception:
        return None


def _detect_edge_index_base(edge_pairs: list[tuple[int, int]], n: int) -> int:
    """
    Detect whether an edge list is 0-based or 1-based.

    The detection uses all observed endpoints.  If the file is ambiguous, the
    function defaults to 0-based indexing, which is safer for files generated by
    Python utilities.  Use ``index_base_edges=1`` in the generator when R-style
    1-based edge labels are needed.
    """
    if not edge_pairs:
        return 0

    endpoints = np.asarray(edge_pairs, dtype=np.int64).reshape(-1)
    min_id = int(endpoints.min())
    max_id = int(endpoints.max())

    if min_id >= 0 and max_id < n:
        return 0
    if min_id >= 1 and max_id <= n:
        return 1

    raise ValueError(
        f"Edge indices are out of range: min={min_id}, max={max_id}, n={n}. "
        "Expected 0-based labels in [0, n-1] or 1-based labels in [1, n]."
    )


def _dmax_from_edges_csv_stream(path: str, n: int) -> int:
    """Compute max degree by streaming through an edge-list CSV file."""
    edge_pairs: list[tuple[int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        has_header = any(ch.isalpha() for ch in first_line)

        if first_line and not has_header:
            parts = first_line.split(",") if "," in first_line else first_line.split()
            if len(parts) >= 2:
                edge_pairs.append((int(parts[0]), int(parts[1])))

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) < 2:
                continue
            edge_pairs.append((int(parts[0]), int(parts[1])))

    if not edge_pairs:
        return 0

    index_base = _detect_edge_index_base(edge_pairs, n=n)
    degree = np.zeros(n, dtype=np.int64)

    for u, v in edge_pairs:
        if index_base == 1:
            u -= 1
            v -= 1
        if u == v:
            continue
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge out of range in {path}: ({u}, {v}) with n={n}")
        degree[u] += 1
        degree[v] += 1

    return int(degree.max(initial=0))


def _dmax_from_abit_bitpack(path: str) -> int:
    """
    Compute max degree from the ``ABIT`` upper-triangle bit-packed adjacency.

    This is slower than reading ``d_max`` from ``meta.json`` but avoids storing a
    large dense or sparse adjacency matrix.
    """
    bits_little = [[] for _ in range(256)]
    bits_big = [[] for _ in range(256)]
    for byte_value in range(256):
        for bit in range(8):
            if (byte_value >> bit) & 1:
                bits_little[byte_value].append(bit)
            if (byte_value >> (7 - bit)) & 1:
                bits_big[byte_value].append(bit)

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ABIT":
            raise ValueError(f"Not an ABIT file: {path}")

        n = struct.unpack("<I", f.read(4))[0]
        _index_base = struct.unpack("<B", f.read(1))[0]
        bitflag = struct.unpack("<B", f.read(1))[0]
        table = bits_little if bitflag == 1 else bits_big

        degree = np.zeros(n, dtype=np.int32)

        for i in range(n - 1):
            row_length = n - i - 1
            nbytes = (row_length + 7) // 8
            buffer = f.read(nbytes)
            if len(buffer) != nbytes:
                raise ValueError(f"Unexpected EOF in {path} at row i={i}")

            for byte_idx, byte_value in enumerate(buffer):
                if byte_value == 0:
                    continue
                for bit in table[byte_value]:
                    pos = 8 * byte_idx + bit
                    if pos >= row_length:
                        continue
                    j = i + 1 + pos
                    degree[i] += 1
                    degree[j] += 1

    return int(degree.max(initial=0))


def _try_read_dmax_from_meta_json(path: str) -> Optional[int]:
    """Read ``d_max`` from a generator metadata JSON file, if available."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for key in ["d_max", "dmaxA", "dmax", "max_degree", "maxDegree"]:
            if isinstance(obj, dict) and key in obj and obj[key] is not None:
                return int(obj[key])
    except Exception:
        return None
    return None


def load_dmaxA(rep_dir: str, n: int) -> int:
    """
    Load or compute ``d_max(A)`` for one saved replication.

    The search order is deliberately arranged from fastest to slowest:

    1. ``meta.json`` or ``A_meta.json``;
    2. ``report.json`` or ``A_report.json``;
    3. degree CSV files;
    4. edge-list CSV files;
    5. the compact ``ABIT`` bit-packed adjacency file.
    """
    for filename in ["meta.json", "A_meta.json"]:
        path = os.path.join(rep_dir, filename)
        if os.path.exists(path):
            value = _try_read_dmax_from_meta_json(path)
            if value is not None:
                return int(value)

    for filename in ["report.json", "A_report.json"]:
        path = os.path.join(rep_dir, filename)
        if os.path.exists(path):
            value = _try_read_max_degree_from_report_json(path)
            if value is not None:
                return int(value)

    for filename in ["degrees.csv", "A_degrees.csv", "deg.csv"]:
        path = os.path.join(rep_dir, filename)
        if os.path.exists(path):
            value = _try_read_dmax_from_degrees_csv(path)
            if value is not None:
                return int(value)

    for filename in ["A_edges.csv", "edges.csv"]:
        path = os.path.join(rep_dir, filename)
        if os.path.exists(path):
            return int(_dmax_from_edges_csv_stream(path, n=n))

    for filename in ["A_upper_bitpack.bin", "A_upper_bitpack_little.bin"]:
        path = os.path.join(rep_dir, filename)
        if os.path.exists(path):
            return int(_dmax_from_abit_bitpack(path))

    raise FileNotFoundError(
        f"Cannot obtain d_max(A) in {rep_dir}. Provide meta.json with d_max, "
        "or degrees.csv, or A_edges.csv, or A_upper_bitpack.bin."
    )


# -----------------------------------------------------------------------------
# Oracle and private estimates of the degree-scale parameter theta0
# -----------------------------------------------------------------------------


def _load_omega_components(rep_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``theta``, labels ``z0``, and the effective block matrix ``P_eff``."""
    npz_path = os.path.join(rep_dir, "omega_components.npz")
    if os.path.exists(npz_path):
        obj = np.load(npz_path)
        theta = np.asarray(obj["theta"], dtype=float).reshape(-1)
        z0 = np.asarray(obj["z0"], dtype=int).reshape(-1)
        P_eff = np.asarray(obj["P_eff"], dtype=float)
        return theta, z0, P_eff

    theta_path = os.path.join(rep_dir, "omega_theta_z.csv")
    peff_path = os.path.join(rep_dir, "P_eff.csv")
    if not (os.path.exists(theta_path) and os.path.exists(peff_path)):
        raise FileNotFoundError("Need omega_components.npz or both omega_theta_z.csv and P_eff.csv.")

    dat = np.genfromtxt(theta_path, delimiter=",", names=True)
    theta = np.asarray(dat["theta"], dtype=float).reshape(-1)
    z = np.asarray(dat["z"], dtype=int).reshape(-1)
    z0 = z - 1 if z.min() == 1 else z

    mat = np.genfromtxt(peff_path, delimiter=",", names=True)
    colnames = [name for name in mat.dtype.names if name != "row"]
    P_eff = np.column_stack([mat[name] for name in colnames]).astype(float)

    return theta, z0.astype(int), P_eff


def theta0_oracle_from_omega(rep_dir: str, n: int, K: int, floor: float = 1e-12) -> dict:
    r"""
    Compute the oracle degree-scale parameter from saved population quantities.

    For a binary adjacency matrix generated from ``Omega``, this uses

    .. math::

        \theta_{0,\mathrm{oracle}} = \sqrt{\|\Omega\|_\infty / n},

    where ``||Omega||_infty`` is the maximum expected row sum.
    """
    theta, z0, P_eff = _load_omega_components(rep_dir)

    if theta.shape[0] != n or z0.shape[0] != n:
        raise ValueError(f"Omega component length mismatch: got {theta.shape[0]} versus n={n}.")
    if P_eff.shape != (K, K):
        raise ValueError(f"P_eff shape mismatch: got {P_eff.shape}, expected {(K, K)}.")
    if z0.min() < 0 or z0.max() >= K:
        raise ValueError("z0 must be encoded as {0, ..., K-1}.")

    # S_l = sum_{j:z_j=l} theta_j.
    S = np.bincount(z0, weights=theta, minlength=K).astype(float)
    c = P_eff @ S

    # Exclude the diagonal self-loop contribution.
    diagP = np.diag(P_eff)
    expected_degree = theta * c[z0] - (theta * theta) * diagP[z0]
    omega_inf = float(np.max(expected_degree, initial=0.0))

    theta0_oracle = float(np.sqrt(max(omega_inf / float(n), 0.0)))
    theta0_oracle = max(theta0_oracle, float(floor))

    return {
        "theta0_oracle": theta0_oracle,
        "omega_inf": omega_inf,
    }


def theta0_tilde_from_dmaxA(
    dmaxA: float,
    n: int,
    eps1: float,
    rng: Optional[np.random.Generator] = None,
    floor: float = 1e-12,
) -> dict:
    r"""
    Compute the noisy DP-style estimate of ``theta0``.

    The estimate is

    .. math::

        \widetilde\theta_0(A)
        = \sqrt{\|A\|_\infty/n + \mathrm{Lap}(1)/(\varepsilon_1 n)}.

    For a 0/1 adjacency matrix, ``||A||_infty`` is the maximum degree
    ``dmaxA``.  The expression inside the square root is clipped at zero for
    numerical stability.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if eps1 is None or eps1 <= 0:
        raise ValueError("eps1 must be provided and positive.")
    if rng is None:
        rng = np.random.default_rng()

    lap = float(rng.laplace(loc=0.0, scale=1.0))
    inside = (float(dmaxA) / float(n)) + (lap / (float(eps1) * float(n)))
    inside_clipped = max(inside, 0.0)

    theta0_tilde = float(np.sqrt(inside_clipped))
    theta0_tilde = max(theta0_tilde, float(floor))

    return {
        "theta0_tilde": theta0_tilde,
        "lap": lap,
        "inside": float(inside),
        "inside_clipped": float(inside_clipped),
        "A_inf": float(dmaxA),
        "eps1": float(eps1),
    }


# -----------------------------------------------------------------------------
# Stability margin and improved GapPTR mechanism
# -----------------------------------------------------------------------------


def _norm_2_infty(X: np.ndarray) -> float:
    """Return the maximum row Euclidean norm of a matrix."""
    row_norm = np.sqrt((X * X).sum(axis=1))
    return float(np.max(row_norm)) if row_norm.size else 0.0


def compute_gamma_E(
    Xi_hat: np.ndarray,
    dmaxA: float,
    lambdaK: float,
    lambdaKp1: float,
    a0: float,
    A0: float,
    theta0: float = 1.0,
    verbose: bool = False,
) -> dict:
    r"""
    Compute the empirical stability margin ``gamma_E(A)``.

    The margin is the positive part of the minimum of four good-set diagnostics:
    a degree condition, a lower eigengap condition, an upper noise-eigenvalue
    condition, and a row-wise eigenvector condition.  The returned dictionary
    also records the four raw terms and the corresponding Boolean checks.
    """
    Xi_hat = np.asarray(Xi_hat, dtype=float)
    n = int(Xi_hat.shape[0])

    if n <= 0:
        raise ValueError("Xi_hat must have at least one row.")
    if a0 <= 0:
        raise ValueError("a0 must be positive.")
    if A0 <= 0:
        raise ValueError("A0 must be positive.")
    if theta0 <= 0:
        raise ValueError("theta0 must be positive.")
    if dmaxA < 0:
        raise ValueError("dmaxA must be nonnegative.")

    sqrt2 = np.sqrt(2.0)
    sqrtn = np.sqrt(float(n))
    theta0_sq = float(theta0) ** 2
    xi_2inf = _norm_2_infty(Xi_hat)

    # Lipschitz-type denominator for the row-wise eigenvector condition.
    # Keep this expression synchronized with the theory used in the manuscript.
    U0 = (
        (4.0 * sqrt2 * A0) / (a0 * theta0_sq * n * sqrtn)
        + A0 / (a0 * theta0_sq * n * sqrtn)
        + (sqrt2 * A0**2) / (a0 * theta0_sq * n**2)
        + (5.0 * sqrt2 * A0) / (a0**2 * theta0_sq**2 * n**2 * sqrtn)
        + (50.0 * A0**3) / (a0**2 * theta0_sq**2 * n**3 * sqrtn)
    )

    t_deg = (((1.0 + a0) * theta0_sq * n) - float(dmaxA)) / sqrt2
    t1 = (float(lambdaK) - a0 * theta0_sq * n - 3.0 * sqrt2) / sqrt2
    t2 = (0.8 * a0 * theta0_sq * n - float(lambdaKp1)) / sqrt2
    t3 = (A0 / sqrtn - xi_2inf) / float(U0)

    gamma_E = max(min(t_deg, t1, t2, t3), 0.0)

    cond_deg = (1.0 + a0) * theta0_sq * n >= float(dmaxA)
    cond_lamK = float(lambdaK) >= a0 * theta0_sq * n + 3.0 * sqrt2
    cond_lamKp1 = float(lambdaKp1) <= a0 * theta0_sq * n
    cond_xi = xi_2inf <= A0 / sqrtn
    in_good_set = bool(cond_deg and cond_lamK and cond_lamKp1 and cond_xi)

    if verbose:
        print(
            f"t_deg={t_deg:.6g}, t1={t1:.6g}, t2={t2:.6g}, t3={t3:.6g}, "
            f"gamma_E={gamma_E:.6g}, good_set={in_good_set}"
        )

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
        "good_set": in_good_set,
        "cond_deg": bool(cond_deg),
        "cond_lamK": bool(cond_lamK),
        "cond_lamKp1": bool(cond_lamKp1),
        "cond_xi": bool(cond_xi),
    }


def GapPrecheckPTR_improved_from_embedding(
    Xi_hat: np.ndarray,
    dmaxA: float,
    lambdaK: float,
    lambdaKp1: float,
    K: int,
    eps_list: Iterable[float],
    delta: float,
    a0: float,
    A0: float,
    theta0: float = 1.0,
    nstart: int = 25,
    norm_tol: float = 1e-12,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Run the improved GapPTR release step from a saved embedding.

    The same Gaussian perturbation direction is reused across all epsilon values
    in ``eps_list`` for a fixed replication call.  This makes curves across
    privacy budgets less noisy while preserving independent randomness across
    replications through the supplied seed.
    """
    Xi_hat = np.asarray(Xi_hat, dtype=float)
    n = Xi_hat.shape[0]

    if delta <= 0 or delta >= 1:
        raise ValueError("delta must lie in (0, 1).")
    if K <= 0:
        raise ValueError("K must be positive.")
    if Xi_hat.shape[1] != K:
        raise ValueError(f"Xi_hat must have shape (n, K); got {Xi_hat.shape} with K={K}.")
    if a0 <= 0 or A0 <= 0:
        raise ValueError("a0 and A0 must be positive.")
    if theta0 <= 0:
        raise ValueError("theta0 must be positive.")

    if np.isscalar(eps_list):
        eps_arr = np.array([float(eps_list)], dtype=float)
    else:
        eps_arr = np.asarray(list(eps_list), dtype=float).reshape(-1)

    rng = np.random.default_rng(seed)
    zeta = rng.standard_normal(size=(n, K))

    gamma_info = compute_gamma_E(
        Xi_hat=Xi_hat,
        dmaxA=float(dmaxA),
        lambdaK=float(lambdaK),
        lambdaKp1=float(lambdaKp1),
        a0=float(a0),
        A0=float(A0),
        theta0=float(theta0),
        verbose=False,
    )
    gamma_E = float(gamma_info["gamma_E"])

    sqrtn = np.sqrt(float(n))
    alpha = (
        (2.0 * np.sqrt(2.0 * K) * A0) / (a0 * n * sqrtn)
        + (16.0 * np.sqrt(float(K))) / (a0**2 * n**2)
    )

    results: list[dict] = []
    for eps in eps_arr:
        eps = float(eps)
        if eps <= 0:
            raise ValueError("All epsilon values must be positive.")

        M = 1.0 + (2.0 / eps) * np.log(1.0 / delta)
        betaA = gamma_E
        pA = 1.0 if betaA > 2.0 * M else float(_stable_sigmoid(0.5 * eps * (betaA - M)))

        noise_scale = (alpha / eps) * np.sqrt(2.0 * np.log(1.25 / delta))

        if rng.random() < pA:
            Xi_tilde = Xi_hat + noise_scale * zeta
        else:
            Xi_tilde = rng.random(size=(n, K))

        labels_DP = _cluster_from_embedding(
            Xi_tilde,
            K,
            nstart=nstart,
            norm_tol=norm_tol,
            seed=seed,
        )

        results.append(
            {
                "epsilon": eps,
                "labels_DP": labels_DP,
                "lambdaK": float(lambdaK),
                "lambdaKp1": float(lambdaKp1),
                "a0": float(a0),
                "A0": float(A0),
                "theta0": float(theta0),
                "dmaxA": float(dmaxA),
                "gamma_E": float(gamma_E),
                "M": float(M),
                "betaA": float(betaA),
                "pA": float(pA),
                "alpha": float(alpha),
                "noise_scale": float(noise_scale),
                "xi_2inf": float(gamma_info["xi_2inf"]),
                "U0": float(gamma_info["U0"]),
                "t_deg": float(gamma_info["t_deg"]),
                "t1": float(gamma_info["t1"]),
                "t2": float(gamma_info["t2"]),
                "t3": float(gamma_info["t3"]),
                "delta": float(delta),
                "good_set": bool(gamma_info["good_set"]),
                "cond_deg": bool(gamma_info["cond_deg"]),
                "cond_lamK": bool(gamma_info["cond_lamK"]),
                "cond_lamKp1": bool(gamma_info["cond_lamKp1"]),
                "cond_xi": bool(gamma_info["cond_xi"]),
            }
        )

    return results


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------


def _discover_rep_ids(sim_dir: str) -> list[int]:
    """Discover replication IDs from folders named ``rep000`` or ``rep_000``."""
    rep_ids: list[int] = []
    for name in sorted(os.listdir(sim_dir)):
        match = re.match(r"rep[_]?(\d+)$", name)
        if match:
            rep_ids.append(int(match.group(1)))
    return rep_ids


def delta_default(eps: float, n: int) -> float:
    """Default delta choice used when no explicit delta is supplied."""
    del eps
    return 1.0 / n


def run_improved_gapptr_from_saved(
    sim_dir: str,
    rep_ids: Optional[Sequence[int]] = None,
    K: int = 2,
    eps_list: Iterable[float] = (0.5,),
    nstart: int = 25,
    base_seed: int = 5000,
    verbose: bool = True,
    a0: float = 1e-3,
    A0: float = 2.0,
    theta0: float = 1.0,
    theta0_mode: str = "fixed",
    eps1_mode: str = "fixed",
    eps1: Optional[float] = None,
    theta0_floor: float = 1e-12,
    A0_mode: str = "fixed",
    A0_factor: float = 1.05,
    eig_sort: str = "abs",
    delta: Optional[float] = None,
    delta_fn=delta_default,
) -> pd.DataFrame:
    r"""
    Run improved GapPTR on saved simulation replications.

    Parameters
    ----------
    sim_dir:
        Directory containing replication folders such as ``rep000``.
    rep_ids:
        Replication IDs to use.  If ``None``, IDs are discovered automatically.
    K:
        Number of communities.
    eps_list:
        Privacy-budget values used by the release step.
    a0, A0:
        Good-set constants appearing in the stability margin.
    theta0:
        Fixed input value used when ``theta0_mode='fixed'``.
    theta0_mode:
        One of ``'fixed'``, ``'oracle'``, ``'use_tilde'``, or
        ``'floor_by_tilde'``.

        * ``'fixed'`` uses the supplied ``theta0``.
        * ``'oracle'`` uses ``sqrt(||Omega||_infty / n)`` from saved population
          quantities.
        * ``'use_tilde'`` uses the noisy estimate based on ``d_max(A)``.
        * ``'floor_by_tilde'`` uses ``max(theta0, theta0_tilde)``.

    eps1_mode:
        Relevant only for tilde-based modes.  Use ``'fixed'`` to use the same
        ``eps1`` for all release budgets, or ``'same_as_eps'`` to set
        ``eps1=eps`` separately for each release budget.
    A0_mode:
        ``'fixed'`` uses the supplied ``A0``.  ``'from_2toinfty'`` sets ``A0``
        from the observed ``2-to-infinity`` norm of the embedding.

    Returns
    -------
    pandas.DataFrame
        One row for each ``(replication, epsilon)`` pair.  The table includes
        clustering error, release probability, noise scale, stability margin,
        theta0 diagnostics, and good-set diagnostics.
    """
    if rep_ids is None:
        rep_ids = _discover_rep_ids(sim_dir)

    if theta0_mode not in {"fixed", "use_tilde", "floor_by_tilde", "oracle"}:
        raise ValueError("theta0_mode must be 'fixed', 'use_tilde', 'floor_by_tilde', or 'oracle'.")

    uses_tilde = theta0_mode in {"use_tilde", "floor_by_tilde"}
    if uses_tilde:
        if eps1_mode not in {"fixed", "same_as_eps"}:
            raise ValueError("eps1_mode must be 'fixed' or 'same_as_eps'.")
        if eps1_mode == "fixed" and (eps1 is None or float(eps1) <= 0):
            raise ValueError("eps1 must be positive when eps1_mode='fixed' and theta0_mode uses tilde.")

    eps_values = [float(x) for x in (np.atleast_1d(eps_list))]
    records: list[dict] = []

    for rid in rep_ids:
        rep_dir_a = os.path.join(sim_dir, f"rep{rid:03d}")
        rep_dir_b = os.path.join(sim_dir, f"rep_{rid:03d}")
        rep_dir = rep_dir_a if os.path.isdir(rep_dir_a) else rep_dir_b
        if not os.path.isdir(rep_dir):
            raise FileNotFoundError(f"Cannot find replication directory for rid={rid}.")

        Xi_hat = load_Xi_hat_from_csv(os.path.join(rep_dir, "A_eigvecs_topK.csv"), K=K)
        n = int(Xi_hat.shape[0])

        eig_info = load_eigs_topKp1_from_csv(
            os.path.join(rep_dir, "A_eigvals_topKp1.csv"),
            K=K,
            sort_by=eig_sort,
        )
        lambdaK = float(eig_info["lambdaK"])
        lambdaKp1 = float(eig_info["lambdaKp1"])
        dmaxA = float(load_dmaxA(rep_dir, n=n))

        oracle_info = None
        if theta0_mode == "oracle":
            oracle_info = theta0_oracle_from_omega(rep_dir, n=n, K=K, floor=theta0_floor)

        if A0_mode == "fixed":
            A0_use = float(A0)
        elif A0_mode == "from_2toinfty":
            xi_2inf = _norm_2_infty(Xi_hat)
            A0_use = (1.0 + float(A0_factor)) * np.sqrt(float(n)) * float(xi_2inf)
        else:
            raise ValueError("A0_mode must be 'fixed' or 'from_2toinfty'.")

        z_true = try_load_labels_true(rep_dir)
        if z_true is not None:
            z_true = np.asarray(z_true, dtype=int).reshape(-1)
            if z_true.min() == 1 and z_true.max() == K:
                z_true = z_true - 1

        base_rng = np.random.default_rng(base_seed + rid)

        fixed_tilde_info = None
        if uses_tilde and eps1_mode == "fixed":
            rng_theta = np.random.default_rng(base_seed + 123456 + rid)
            fixed_tilde_info = theta0_tilde_from_dmaxA(
                dmaxA=dmaxA,
                n=n,
                eps1=float(eps1),
                rng=rng_theta,
                floor=theta0_floor,
            )

        for j, eps in enumerate(eps_values):
            delta_use = float(delta) if delta is not None else float(delta_fn(eps, n))
            seed_run = int(base_rng.integers(1, 1_000_000))

            eps1_used = np.nan
            theta0_tilde = np.nan
            lap_theta0 = np.nan
            theta0_oracle = np.nan
            omega_inf = np.nan

            theta0_input = float(theta0)
            theta0_used = theta0_input

            if theta0_mode == "oracle":
                theta0_oracle = float(oracle_info["theta0_oracle"])
                omega_inf = float(oracle_info["omega_inf"])
                theta0_used = theta0_oracle

            elif uses_tilde:
                if eps1_mode == "same_as_eps":
                    eps1_used = float(eps)
                    rng_theta = np.random.default_rng(base_seed + 654321 + rid * 1000 + j)
                    tilde_info = theta0_tilde_from_dmaxA(
                        dmaxA=dmaxA,
                        n=n,
                        eps1=eps1_used,
                        rng=rng_theta,
                        floor=theta0_floor,
                    )
                else:
                    eps1_used = float(eps1)
                    tilde_info = fixed_tilde_info

                theta0_tilde = float(tilde_info["theta0_tilde"])
                lap_theta0 = float(tilde_info["lap"])

                if theta0_mode == "use_tilde":
                    theta0_used = theta0_tilde
                else:
                    theta0_used = max(theta0_input, theta0_tilde)

            mechanism_output = GapPrecheckPTR_improved_from_embedding(
                Xi_hat=Xi_hat,
                dmaxA=dmaxA,
                lambdaK=lambdaK,
                lambdaKp1=lambdaKp1,
                K=K,
                eps_list=[eps],
                delta=delta_use,
                a0=float(a0),
                A0=float(A0_use),
                theta0=float(theta0_used),
                nstart=nstart,
                seed=seed_run,
            )[0]

            clustering_err = np.nan
            if z_true is not None:
                clustering_err = float(clustering_error(z_true, mechanism_output["labels_DP"]))

            records.append(
                {
                    "rep": int(rid),
                    "n": int(n),
                    "K": int(K),
                    "eps": float(eps),
                    "delta": float(delta_use),
                    "eig_sort": str(eig_sort),
                    "lambdaK": float(lambdaK),
                    "lambdaKp1": float(lambdaKp1),
                    "dmaxA": float(dmaxA),
                    "theta0_mode": str(theta0_mode),
                    "eps1_mode": str(eps1_mode),
                    "eps1_used": float(eps1_used) if np.isfinite(eps1_used) else np.nan,
                    "theta0_input": float(theta0_input),
                    "theta0_tilde": float(theta0_tilde) if np.isfinite(theta0_tilde) else np.nan,
                    "lap_theta0": float(lap_theta0) if np.isfinite(lap_theta0) else np.nan,
                    "theta0_oracle": float(theta0_oracle) if np.isfinite(theta0_oracle) else np.nan,
                    "omega_inf": float(omega_inf) if np.isfinite(omega_inf) else np.nan,
                    "theta0_used": float(theta0_used),
                    "a0": float(a0),
                    "A0": float(A0_use),
                    "gamma_E": float(mechanism_output["gamma_E"]),
                    "t_deg": float(mechanism_output["t_deg"]),
                    "t1": float(mechanism_output["t1"]),
                    "t2": float(mechanism_output["t2"]),
                    "t3": float(mechanism_output["t3"]),
                    "U0": float(mechanism_output["U0"]),
                    "pA": float(mechanism_output["pA"]),
                    "M": float(mechanism_output["M"]),
                    "alpha": float(mechanism_output["alpha"]),
                    "noise_scale": float(mechanism_output["noise_scale"]),
                    "xi_2inf": float(mechanism_output["xi_2inf"]),
                    "clustering_error_vs_truth": float(clustering_err) if np.isfinite(clustering_err) else np.nan,
                    "good_set": bool(mechanism_output["good_set"]),
                    "cond_deg": bool(mechanism_output["cond_deg"]),
                    "cond_lamK": bool(mechanism_output["cond_lamK"]),
                    "cond_lamKp1": bool(mechanism_output["cond_lamKp1"]),
                    "cond_xi": bool(mechanism_output["cond_xi"]),
                }
            )

            if verbose:
                good_status = "GOOD" if mechanism_output["good_set"] else "BAD"
                if theta0_mode == "oracle":
                    theta_msg = f"theta0={theta0_used:.4g} (oracle, omega_inf={omega_inf:.4g})"
                elif uses_tilde:
                    theta_msg = f"theta0={theta0_used:.4g} (tilde={theta0_tilde:.4g}, eps1={eps1_used:.4g})"
                else:
                    theta_msg = f"theta0={theta0_used:.4g} (fixed)"

                err_msg = f"{clustering_err:.4g}" if np.isfinite(clustering_err) else "NA"
                print(
                    f"rep {rid:03d}, eps={eps:.4g}: "
                    f"gamma_E={mechanism_output['gamma_E']:.3g}, "
                    f"pA={mechanism_output['pA']:.3f}, "
                    f"alpha={mechanism_output['alpha']:.3g}, "
                    f"err={err_msg}, {theta_msg}, good_set={good_status}"
                )

        if verbose:
            print()

    return pd.DataFrame(records)


__all__ = [
    "GapPrecheckPTR_improved_from_embedding",
    "clustering_error",
    "compute_gamma_E",
    "delta_default",
    "load_Xi_hat_from_csv",
    "load_dmaxA",
    "load_eigs_topKp1_from_csv",
    "run_improved_gapptr_from_saved",
    "theta0_oracle_from_omega",
    "theta0_tilde_from_dmaxA",
    "try_load_labels_true",
]
