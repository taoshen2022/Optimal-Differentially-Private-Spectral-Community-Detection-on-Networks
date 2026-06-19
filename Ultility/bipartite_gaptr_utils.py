"""Utility functions for bipartite GapPTR simulations.

This module is shared by the bipartite simulation scripts.  It does not read
pre-generated networks from disk.  Instead, each replication generates a fresh
bipartite DCSBM network, runs the non-private spectral clustering benchmark,
and then runs the bipartite GapPTR mechanism under the requested privacy split.

Typical project layout
----------------------

    ProjectRoot/
      Utility/
        bipartite_gaptr_utils.py
      BipartiteExperiment/
        run_bipartite_vary_eps.py
        run_bipartite_vary_m.py

The experiment scripts should import this file as

    from Utility.bipartite_gaptr_utils import ...

The main user-facing functions are

    run_one_replication(...)
    summarize_records(...)
    plot_summary(...)

Notation
--------
The generated matrix B is n x m.  Clustering is evaluated on the left-side
nodes.  The population matrix follows a bipartite degree-corrected block model,

    Omega_ij = theta_i * P_B[z_i, w_j] * phi_j,

where z_i is the left-side community label and w_j is the right-side community
label.  Conditional on Omega, entries of B are sampled independently as
Bernoulli(Omega_ij).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans


# -----------------------------------------------------------------------------
# Scenario definitions
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for one bipartite DCSBM scenario.

    Parameters
    ----------
    name:
        Scenario name used in output tables.
    K, J:
        Number of left- and right-side communities.
    pb:
        K x J baseline connectivity matrix.
    theta_mode, phi_mode:
        Degree-parameter distributions.  Use "uniform" for a single uniform
        distribution and "mixture" to introduce a lower-degree subpopulation.
    theta_low, theta_high, phi_low, phi_high:
        Baseline uniform ranges.
    mix_prob:
        Probability of drawing from the baseline range in mixture mode.
        The remaining probability is assigned to the low-degree component.
    mix_low_scale:
        Multiplier defining the low-degree range in mixture mode.
        For example, if theta is uniform on [0.6, 1.0] and mix_low_scale=0.25,
        then the low-degree component is uniform on [0.15, 0.25].
    """

    name: str = "scenario1"
    K: int = 2
    J: int = 2
    pb: tuple[tuple[float, ...], ...] = ((0.35, 0.08), (0.08, 0.35))
    theta_mode: str = "uniform"
    phi_mode: str = "uniform"
    theta_low: float = 0.7
    theta_high: float = 1.0
    phi_low: float = 0.7
    phi_high: float = 1.0
    mix_prob: float = 0.6
    mix_low_scale: float = 0.2


SCENARIOS: dict[str, ScenarioConfig] = {
    "scenario1": ScenarioConfig(
        name="scenario1",
        pb=((0.7, 0.1), (0.1, 0.7)),
        theta_mode="uniform",
        phi_mode="uniform",
        theta_low=0.7,
        theta_high=1.0,
        phi_low=0.7,
        phi_high=1.0,
    ),
    "scenario2": ScenarioConfig(
        name="scenario2",
        pb=((0.50, 0.15), (0.15, 0.50)),
        theta_mode="mixture",
        phi_mode="mixture",
        theta_low=0.6,
        theta_high=1.0,
        phi_low=0.6,
        phi_high=1.0,
        mix_prob=0.65,
        mix_low_scale=0.25,
    ),
}


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def build_rng(seed: Optional[int]) -> np.random.Generator:
    """Construct a NumPy random generator."""

    return np.random.default_rng(seed)


def balanced_labels(n: int, k: int) -> np.ndarray:
    """Return nearly balanced labels in {0, ..., k-1}."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if k <= 0:
        raise ValueError("k must be positive.")
    base = np.repeat(np.arange(k), n // k)
    rem = n - len(base)
    if rem > 0:
        base = np.concatenate([base, np.arange(rem)])
    return base.astype(int)


def row_degree_max(B: np.ndarray) -> float:
    """Return ||B||_infty, the maximum left-side row degree."""

    return float(np.asarray(B.sum(axis=1)).reshape(-1).max())


def sample_degree_parameters(
    size: int,
    mode: str,
    low: float,
    high: float,
    rng: np.random.Generator,
    mix_prob: float = 0.6,
    mix_low_scale: float = 0.2,
) -> np.ndarray:
    """Sample degree parameters for one side of the bipartite network.

    In mixture mode, a node is drawn from the baseline range with probability
    ``mix_prob`` and from a lower-degree range with probability ``1-mix_prob``.
    """

    if size <= 0:
        raise ValueError("size must be positive.")
    if not (0.0 < low <= high):
        raise ValueError("Require 0 < low <= high.")
    if not (0.0 <= mix_prob <= 1.0):
        raise ValueError("mix_prob must lie in [0, 1].")
    if mix_low_scale <= 0:
        raise ValueError("mix_low_scale must be positive.")

    if mode == "uniform":
        return rng.uniform(low, high, size=size)

    if mode == "mixture":
        baseline = rng.uniform(low, high, size=size)
        low_component = rng.uniform(mix_low_scale * low, mix_low_scale * high, size=size)
        use_baseline = rng.random(size=size) <= mix_prob
        return np.where(use_baseline, baseline, low_component)

    raise ValueError(f"Unknown degree mode: {mode}. Use 'uniform' or 'mixture'.")


# -----------------------------------------------------------------------------
# Bipartite DCSBM generation
# -----------------------------------------------------------------------------


def simulate_bi_dcsbm(
    n: int,
    m: int,
    scenario: str,
    seed: Optional[int] = None,
) -> dict[str, np.ndarray | float | str | int]:
    """Generate one bipartite DCSBM network.

    Parameters
    ----------
    n, m:
        Number of left- and right-side nodes.
    scenario:
        Key in ``SCENARIOS``.
    seed:
        Random seed for this replication.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario={scenario!r}. Available: {sorted(SCENARIOS)}")
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive.")

    cfg = SCENARIOS[scenario]
    rng = build_rng(seed)

    left_labels = balanced_labels(n, cfg.K)
    right_labels = balanced_labels(m, cfg.J)
    rng.shuffle(left_labels)
    rng.shuffle(right_labels)

    theta = sample_degree_parameters(
        size=n,
        mode=cfg.theta_mode,
        low=cfg.theta_low,
        high=cfg.theta_high,
        mix_prob=cfg.mix_prob,
        mix_low_scale=cfg.mix_low_scale,
        rng=rng,
    )
    phi = sample_degree_parameters(
        size=m,
        mode=cfg.phi_mode,
        low=cfg.phi_low,
        high=cfg.phi_high,
        mix_prob=cfg.mix_prob,
        mix_low_scale=cfg.mix_low_scale,
        rng=rng,
    )

    pb = np.asarray(cfg.pb, dtype=float)
    if pb.shape != (cfg.K, cfg.J):
        raise ValueError("pb has incompatible shape.")

    omega = theta[:, None] * pb[left_labels][:, right_labels] * phi[None, :]
    if np.any(omega > 1.0):
        max_prob = float(omega.max())
        raise ValueError(
            f"Some Bernoulli probabilities exceed 1; max probability is {max_prob:.4g}. "
            "Reduce pb or degree scales."
        )

    B = rng.binomial(1, omega).astype(float)

    # For the bipartite scaling used in the PTR test, theta0 is tied to the
    # maximum expected left degree divided by m.
    theta0_oracle = math.sqrt(max(float(omega.sum(axis=1).max()) / max(m, 1), 1e-12))

    return {
        "B": B,
        "omega": omega,
        "left_labels": left_labels,
        "right_labels": right_labels,
        "theta": theta,
        "phi": phi,
        "theta0_oracle": theta0_oracle,
        "scenario": scenario,
        "n": n,
        "m": m,
    }


# -----------------------------------------------------------------------------
# Spectral clustering and error metric
# -----------------------------------------------------------------------------


def _top_left_singular_vectors_dense(B: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    return U[:, :k], s[:k]


def top_left_singular_vectors(B: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the top left singular vectors and singular values of B."""

    n, m = B.shape
    max_rank = min(n, m)
    k_eff = min(k, max_rank)
    if k_eff < 1:
        raise ValueError("k must be at least 1.")

    # scipy.sparse.linalg.svds requires k < min(B.shape), so use dense SVD when
    # the requested rank is close to the matrix rank.
    if max_rank <= 50 or k_eff >= max_rank:
        return _top_left_singular_vectors_dense(B, k_eff)

    try:
        U, s, _ = svds(sparse.csr_matrix(B), k=k_eff)
        order = np.argsort(s)[::-1]
        return U[:, order], s[order]
    except Exception:
        return _top_left_singular_vectors_dense(B, k_eff)


def row_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows of a matrix, leaving near-zero rows unchanged."""

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def kmeans_labels(X: np.ndarray, K: int, random_state: int = 0) -> np.ndarray:
    """Run K-means and return labels."""

    km = KMeans(n_clusters=K, n_init=20, random_state=random_state)
    return km.fit_predict(X)


def bipartite_spectral_clustering(
    B: np.ndarray,
    K: int = 2,
    random_state: int = 0,
) -> dict[str, np.ndarray | float]:
    """Spectral clustering of left-side nodes in a bipartite matrix."""

    U, s = top_left_singular_vectors(B, k=K + 1)
    Uk = U[:, :K]
    lambdas = (s**2) / B.shape[1]
    gap = float(lambdas[K - 1] - lambdas[K]) if len(lambdas) > K else float(lambdas[K - 1])
    labels = kmeans_labels(row_normalize(Uk), K=K, random_state=random_state)
    return {
        "embedding": Uk,
        "labels": labels,
        "singular_values": s,
        "eigenvalues": lambdas,
        "gap": gap,
    }


def clustering_error(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Misclustering error after optimal label permutation."""

    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)
    if true_labels.shape[0] != pred_labels.shape[0]:
        raise ValueError("true_labels and pred_labels must have the same length.")

    k_true = int(true_labels.max()) + 1
    k_pred = int(pred_labels.max()) + 1
    K = max(k_true, k_pred)

    contingency = np.zeros((K, K), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        contingency[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    return 1.0 - matched / len(true_labels)


# -----------------------------------------------------------------------------
# Bipartite GapPTR mechanism
# -----------------------------------------------------------------------------


def private_theta0_estimate(B: np.ndarray, eps1: float, rng: np.random.Generator) -> float:
    """Private estimate of theta0 based on the maximum left-side degree.

    The estimator is

        sqrt( ||B||_infty / m + Lap(1)/(eps1*m) ).
    """

    m = B.shape[1]
    if eps1 <= 0:
        raise ValueError("eps1 must be positive for private theta0 estimation.")
    inside = row_degree_max(B) / m + rng.laplace(0.0, 1.0 / (eps1 * m))
    return math.sqrt(max(inside, 1e-12))


def fptr_acceptance_probability(gamma: float, eps: float, delta: float) -> tuple[float, float]:
    """fPTR release probability used by the proposal-test-release step."""

    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")

    M = 1.0 + 2.0 / eps * math.log(1.0 / delta)
    if gamma > 2.0 * M:
        return 1.0, M

    z = 0.5 * eps * (gamma - M)
    if z >= 0:
        p = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        p = ez / (1.0 + ez)
    return p, M


def bipartite_gapptr(
    B: np.ndarray,
    K: int,
    eps: float,
    delta: float,
    a0: float,
    theta0: float,
    rng: np.random.Generator,
    random_state: int,
    noise_mult: float = 1.0,
) -> dict[str, np.ndarray | float | int]:
    """Run the bipartite GapPTR mechanism for one matrix B."""

    if eps <= 0:
        raise ValueError("eps must be positive.")
    if a0 <= 0:
        raise ValueError("a0 must be positive.")
    if theta0 <= 0:
        raise ValueError("theta0 must be positive.")

    n, m = B.shape
    spec = bipartite_spectral_clustering(B, K=K, random_state=random_state)
    Uhat = np.asarray(spec["embedding"], dtype=float)
    gap = float(spec["gap"])

    gamma = (m / (2.0 * n)) * max(gap - a0 * theta0 * theta0 * n, 0.0)
    p_release, M = fptr_acceptance_probability(gamma, eps=eps, delta=delta)

    alpha = 4.0 * math.sqrt(2.0) / (a0 * theta0 * theta0 * m)
    sigma = noise_mult * alpha / eps * math.sqrt(2.0 * math.log(1.25 / delta))

    released = rng.uniform() <= p_release
    if released:
        Upriv = Uhat + sigma * rng.normal(size=Uhat.shape)
        labels = kmeans_labels(row_normalize(Upriv), K=K, random_state=random_state)
    else:
        Upriv = np.full_like(Uhat, np.nan)
        labels = rng.integers(low=0, high=K, size=n)

    return {
        "labels": labels,
        "released": int(released),
        "p_release": float(p_release),
        "M": float(M),
        "gamma": float(gamma),
        "gap": float(gap),
        "alpha": float(alpha),
        "sigma": float(sigma),
        "theta0_used": float(theta0),
        "embedding": Upriv,
    }


# -----------------------------------------------------------------------------
# Replication runner and summaries
# -----------------------------------------------------------------------------


def run_one_replication(
    *,
    n: int,
    m: int,
    scenario: str,
    eps_total: float,
    eps1_values: Iterable[float],
    delta: float,
    a0: float,
    rep: int,
    base_seed: int,
    K: int = 2,
    noise_mult: float = 1.0,
) -> list[dict[str, float | int | str]]:
    """Generate one network and evaluate NonPrivate plus GapPTR variants."""

    if eps_total <= 0:
        raise ValueError("eps_total must be positive.")

    sim_seed = int(base_seed + 100_000 * rep + 97 * m + 11 * n)
    sim = simulate_bi_dcsbm(n=n, m=m, scenario=scenario, seed=sim_seed)
    B = np.asarray(sim["B"], dtype=float)
    left_labels = np.asarray(sim["left_labels"], dtype=int)
    theta0_oracle = float(sim["theta0_oracle"])

    nonprivate = bipartite_spectral_clustering(B, K=K, random_state=sim_seed)
    nonprivate_error = clustering_error(left_labels, np.asarray(nonprivate["labels"], dtype=int))

    records: list[dict[str, float | int | str]] = [
        {
            "scenario": scenario,
            "rep": rep,
            "n": n,
            "m": m,
            "eps_total": eps_total,
            "eps1": np.nan,
            "eps_main": np.nan,
            "method": "NonPrivate",
            "theta_source": "none",
            "error": nonprivate_error,
            "released": 1,
            "p_release": 1.0,
            "gamma": np.nan,
            "gap": float(nonprivate["gap"]),
            "theta0_oracle": theta0_oracle,
            "theta0_used": theta0_oracle,
            "sigma": 0.0,
        }
    ]

    for idx, eps1_raw in enumerate(eps1_values):
        eps1 = float(eps1_raw)
        if eps1 < 0:
            raise ValueError("eps1 values must be nonnegative.")
        if eps1 >= eps_total:
            raise ValueError(f"eps1={eps1} must be smaller than eps_total={eps_total}.")

        eps_main = eps_total - eps1
        rng = build_rng(sim_seed + 10_000 + idx)

        if eps1 == 0:
            theta0_used = theta0_oracle
            theta_source = "oracle"
        else:
            theta0_used = private_theta0_estimate(B, eps1=eps1, rng=rng)
            theta_source = "private"

        priv = bipartite_gapptr(
            B=B,
            K=K,
            eps=eps_main,
            delta=delta,
            a0=a0,
            theta0=theta0_used,
            rng=rng,
            random_state=sim_seed + 200 + idx,
            noise_mult=noise_mult,
        )
        err = clustering_error(left_labels, np.asarray(priv["labels"], dtype=int))
        records.append(
            {
                "scenario": scenario,
                "rep": rep,
                "n": n,
                "m": m,
                "eps_total": eps_total,
                "eps1": eps1,
                "eps_main": eps_main,
                "method": "GapPTR",
                "theta_source": theta_source,
                "error": err,
                "released": int(priv["released"]),
                "p_release": float(priv["p_release"]),
                "gamma": float(priv["gamma"]),
                "gap": float(priv["gap"]),
                "theta0_oracle": theta0_oracle,
                "theta0_used": float(theta0_used),
                "sigma": float(priv["sigma"]),
            }
        )

    return records


def summarize_records(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Aggregate raw replication records by scenario, x variable, method, and eps1."""

    required = {"scenario", x_col, "method", "error"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cannot summarize records; missing columns: {sorted(missing)}")

    group_cols = ["scenario", x_col, "method"]
    if "eps1" in df.columns:
        group_cols.append("eps1")

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_error=("error", "mean"),
            sd_error=("error", "std"),
            n_rep=("error", "size"),
            mean_release=("released", "mean"),
            mean_p_release=("p_release", "mean"),
            mean_gamma=("gamma", "mean"),
            mean_gap=("gap", "mean"),
            mean_theta0_used=("theta0_used", "mean"),
            mean_sigma=("sigma", "mean"),
        )
        .reset_index()
    )
    summary["sd_error"] = summary["sd_error"].fillna(0.0)
    summary["se_error"] = summary["sd_error"] / np.sqrt(summary["n_rep"].clip(lower=1))
    return summary


def method_label(method: str, eps1: float | None) -> str:
    """Human-readable label for plotting."""

    if method == "NonPrivate":
        return "NonPrivate"
    if eps1 is None or pd.isna(eps1):
        return method
    if float(eps1) == 0.0:
        return r"GapPTR ($\epsilon_1=0$, oracle $\theta_0$)"
    return rf"GapPTR ($\epsilon_1={float(eps1):g}$)"


def plot_summary(
    summary: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    outpath: Path,
    log_x: bool = False,
    show_ci: bool = True,
) -> None:
    """Plot mean clustering error with optional 95% SE bands."""

    ensure_dir(outpath.parent)
    plt.figure(figsize=(8, 5))

    plot_df = summary.copy().sort_values(["method", "eps1", x_col], na_position="first")
    for (method, eps1), sub in plot_df.groupby(["method", "eps1"], dropna=False):
        label = method_label(str(method), None if pd.isna(eps1) else float(eps1))
        x = sub[x_col].to_numpy(dtype=float)
        y = sub["mean_error"].to_numpy(dtype=float)
        se = sub["se_error"].to_numpy(dtype=float)

        linestyle = "--" if str(method) == "NonPrivate" else "-"
        marker = "s" if str(method) == "NonPrivate" else "o"
        plt.plot(x, y, marker=marker, linestyle=linestyle, label=label)
        if show_ci:
            plt.fill_between(
                x,
                np.maximum(0.0, y - 1.96 * se),
                np.minimum(1.0, y + 1.96 * se),
                alpha=0.18,
            )

    if log_x:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Mean clustering error")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Command-line helpers shared by scripts
# -----------------------------------------------------------------------------


def save_csv(path: Path, df: pd.DataFrame) -> None:
    """Save a dataframe and create the parent folder if necessary."""

    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def default_m_values() -> list[int]:
    """Default right-side sizes for the varying-m experiment."""

    return [400, 800, 1200, 1600, 2400, 3200]


def default_eps_values() -> list[float]:
    """Default total privacy budgets for the varying-epsilon experiment."""

    return [0.35, 0.5, 0.8, 1.2, 1.6, 2.0]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments common to the two bipartite experiment scripts."""

    parser.add_argument("--scenario", type=str, default="scenario1", choices=sorted(SCENARIOS))
    parser.add_argument("--n", type=int, default=400, help="Number of left-side nodes.")
    parser.add_argument("--K", type=int, default=2, help="Number of left-side communities.")
    parser.add_argument("--delta", type=float, default=1e-3, help="PTR/Gaussian-mechanism delta.")
    parser.add_argument("--a0", type=float, default=0.5, help="Gap threshold constant.")
    parser.add_argument("--reps", type=int, default=30, help="Number of Monte Carlo replications.")
    parser.add_argument("--seed_base", type=int, default=20260418, help="Base random seed.")
    parser.add_argument(
        "--eps1_values",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2],
        help="Privacy budgets used for theta0 estimation. eps1=0 uses oracle theta0.",
    )
    parser.add_argument(
        "--noise_mult",
        type=float,
        default=1.0,
        help="Multiplier for the Gaussian noise scale. Use 2.0 for a more conservative scale.",
    )
    parser.add_argument("--outdir", type=str, default="./outputs", help="Output directory.")
    return parser
