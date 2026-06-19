#!/usr/bin/env python3
"""
Experiment 3: varying the degree scale theta0 under two DCSBM settings.

This script should be placed under the Experiment3/ folder and should call
shared routines from Utility/gapptr_utility.py and Utility/Edge_flip.py.

Required project layout
-----------------------
ProjectRoot/
  Utility/
    gapptr_utility.py
    Edge_flip.py
  Experiment3/
    run_exp3_vary_theta0_all_settings.py
    E3S1theta1/rep000/ ...
    E3S1theta2/rep000/ ...
    ...
    E3S1theta7/rep000/ ...
    E3S2theta1/rep000/ ...
    E3S2theta2/rep000/ ...
    ...
    E3S2theta7/rep000/ ...

Setting 1 corresponds to the regular heterogeneous-degree design:
  (p_in, p_out) = (0.4, 0.1), theta_i ~ Unif(theta_min, theta_max).

Setting 2 corresponds to the low-degree mixture design:
  (p_in, p_out) = (0.9, 0.3), theta_i ~ Unif(theta_min, theta_max) first,
  then 40% of theta_i values are multiplied by 0.1.

Mapping used by this runner:
  E3S1theta1--E3S1theta7 -> 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40
  E3S2theta1--E3S2theta7 -> 0.18, 0.24, 0.30, 0.36, 0.42, 0.50, 0.55

For both settings, the default privacy split is
  oracle GapPTR: eps = eps_all, eps1 = 0,
  tilde GapPTR:  eps = eps_all - eps1_tilde, eps1 = eps1_tilde,
with eps_all = 0.8 and eps1_tilde = 0.2 by default.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------
# Import project utilities
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = THIS_FILE.parent
PROJECT_ROOT = EXPERIMENT_DIR.parent

# Allow running from Experiment3/ while importing Utility/* from project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Utility.gapptr_utility import run_improved_gapptr_from_saved
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import run_improved_gapptr_from_saved from Utility/gapptr_utility.py. "
        "Please check that this script is under Experiment3/ and that Utility/ exists "
        "at the project root."
    ) from exc

try:
    from Utility.Edge_flip import run_edgeflip_from_saved
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import run_edgeflip_from_saved from Utility/Edge_flip.py. "
        "Please put Edge_flip.py under Utility/."
    ) from exc


# ---------------------------------------------------------------------
# Basic clustering helpers for the non-private benchmark
# ---------------------------------------------------------------------
def clustering_error(z_true: np.ndarray, z_pred: np.ndarray) -> float:
    """Return label-switching-invariant clustering error."""
    z_true = np.asarray(z_true).astype(int).reshape(-1)
    z_pred = np.asarray(z_pred).astype(int).reshape(-1)
    if z_true.shape[0] != z_pred.shape[0]:
        raise ValueError("z_true and z_pred must have the same length.")

    # Convert possible 1-based labels to 0-based labels.
    if z_true.min() == 1:
        z_true = z_true - 1
    if z_pred.min() == 1:
        z_pred = z_pred - 1

    labels_true = np.unique(z_true)
    labels_pred = np.unique(z_pred)
    K = max(len(labels_true), len(labels_pred))

    # Relabel to compact integer labels.
    true_map = {lab: i for i, lab in enumerate(labels_true)}
    pred_map = {lab: i for i, lab in enumerate(labels_pred)}
    zt = np.array([true_map[x] for x in z_true], dtype=int)
    zp = np.array([pred_map[x] for x in z_pred], dtype=int)

    contingency = np.zeros((K, K), dtype=int)
    for a, b in zip(zt, zp):
        contingency[a, b] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(1.0 - matched / len(z_true))


def _cluster_from_embedding(
    Xi: np.ndarray,
    K: int,
    nstart: int = 25,
    norm_tol: float = 1e-12,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Row-normalize a spectral embedding and run k-means."""
    Xi = np.asarray(Xi, dtype=float)
    rn = np.sqrt((Xi * Xi).sum(axis=1))
    rn = np.maximum(rn, norm_tol)
    Xn = Xi / rn[:, None]
    km = KMeans(n_clusters=K, n_init=nstart, random_state=seed)
    return km.fit_predict(Xn)


def _find_rep_dir(sim_dir: Path, rep: int) -> Path:
    """Locate a replication folder with common naming conventions."""
    candidates = [
        sim_dir / f"rep{rep:03d}",
        sim_dir / f"rep_{rep:03d}",
        sim_dir / f"rep{rep:04d}",
        sim_dir / f"rep_{rep:04d}",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Cannot locate rep directory for rep={rep} under {sim_dir}")


def _find_joint_eigvec_csv(rep_dir: Path) -> Path:
    """Find the CSV containing columns node, z, vec1, ..., vecK."""
    candidates = sorted(glob.glob(str(rep_dir / "*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {rep_dir}")

    for path in candidates:
        try:
            header = pd.read_csv(path, nrows=5)
            cols = {str(c).strip().lower() for c in header.columns}
            if ("node" in cols) and ("z" in cols) and any(c.startswith("vec") for c in cols):
                return Path(path)
        except Exception:
            continue

    raise FileNotFoundError(f"Found CSVs in {rep_dir}, but none has columns node,z,vec*.")


def _load_truth_and_embedding(csv_path: Path, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Load true labels and the top-K eigenvectors from a joint CSV file."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    cols_lower = {c.lower(): c for c in df.columns}

    if "z" not in cols_lower:
        raise ValueError(f"{csv_path} is missing column 'z'.")
    z_col = cols_lower["z"]

    vec_cols = []
    for k in range(1, K + 1):
        key = f"vec{k}"
        if key not in cols_lower:
            raise ValueError(f"{csv_path} is missing column '{key}' for K={K}.")
        vec_cols.append(cols_lower[key])

    z_true = df[z_col].to_numpy(dtype=int)
    Xi = df[vec_cols].to_numpy(dtype=float)

    if "node" in cols_lower:
        node_col = cols_lower["node"]
        order = np.argsort(df[node_col].to_numpy())
        z_true = z_true[order]
        Xi = Xi[order, :]

    return z_true, Xi


def run_nonprivate_from_saved(
    sim_dir: Path,
    rep_ids: Iterable[int],
    K: int,
    theta0_value: float,
    nstart: int = 25,
    base_seed: int = 5000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the non-private spectral clustering benchmark from saved eigenvectors."""
    rows = []
    for rep in rep_ids:
        rep_dir = _find_rep_dir(sim_dir, rep)
        csv_path = _find_joint_eigvec_csv(rep_dir)
        z_true, Xi = _load_truth_and_embedding(csv_path, K=K)

        labels = _cluster_from_embedding(Xi, K=K, nstart=nstart, seed=base_seed + int(rep))
        err = clustering_error(z_true, labels)

        rows.append({
            "rep": int(rep),
            "theta0": float(theta0_value),
            "method": "NonPrivate",
            "clustering_error_vs_truth": float(err),
            "csv_path": csv_path.name,
        })

        if verbose:
            print(f"[NonPrivate] {sim_dir.name} rep={rep:03d} theta0={theta0_value:.4g} err={err:.4f}")

    return pd.DataFrame(rows)


def summarize_rep_df(df: pd.DataFrame, by=("eps",)) -> pd.DataFrame:
    """Summarize replicate-level clustering errors."""
    return (
        df.groupby(list(by), as_index=False)
        .agg(
            mean_err=("clustering_error_vs_truth", "mean"),
            sd_err=("clustering_error_vs_truth", "std"),
            n_rep=("rep", "nunique"),
        )
    )


# ---------------------------------------------------------------------
# Experiment 3 configurations
# ---------------------------------------------------------------------
def default_setting_configs() -> dict[str, dict]:
    """Default folder/theta0/a0 configurations for the two Experiment 3 settings."""
    return {
        "setting1": {
            "label": "Setting 1: regular heterogeneous degrees",
            "folders": [f"E3S1theta{i}" for i in range(1, 8)],
            "theta0_map": {
                "E3S1theta1": 0.10,
                "E3S1theta2": 0.15,
                "E3S1theta3": 0.20,
                "E3S1theta4": 0.25,
                "E3S1theta5": 0.30,
                "E3S1theta6": 0.35,
                "E3S1theta7": 0.40,
            },
            "a0_default": 0.40,
        },
        "setting2": {
            "label": "Setting 2: low-degree mixture",
            "folders": [f"E3S2theta{i}" for i in range(1, 8)],
            "theta0_map": {
                "E3S2theta1": 0.18,
                "E3S2theta2": 0.24,
                "E3S2theta3": 0.30,
                "E3S2theta4": 0.36,
                "E3S2theta5": 0.42,
                "E3S2theta6": 0.50,
                "E3S2theta7": 0.55,
            },
            "a0_default": 0.25,
        },
    }


def resolve_setting_folders(exp_dir: Path, cfg: dict) -> tuple[Path, list[str], dict[str, float]]:
    """Resolve the current flat data folders for one Experiment 3 setting."""
    folders = list(cfg["folders"])
    theta0_map = dict(cfg["theta0_map"])
    return exp_dir, folders, theta0_map


# ---------------------------------------------------------------------
# Running methods
# ---------------------------------------------------------------------
def run_gap_oracle(
    sim_dir: Path,
    rep_ids: list[int],
    K: int,
    eps: float,
    delta: float,
    a0: float,
    A0: float,
    A0_mode: str,
    A0_factor: float,
    nstart: int,
    base_seed: int,
    verbose: bool,
) -> pd.DataFrame:
    """GapPTR with oracle theta0 and eps1=0."""
    return run_improved_gapptr_from_saved(
        sim_dir=str(sim_dir),
        rep_ids=rep_ids,
        K=K,
        eps_list=[float(eps)],
        nstart=nstart,
        base_seed=base_seed,
        verbose=verbose,
        a0=float(a0),
        A0=float(A0),
        A0_mode=A0_mode,
        A0_factor=float(A0_factor),
        eig_sort="abs",
        delta=float(delta),
        theta0_mode="oracle",
        eps1_mode="same_as_eps",  # ignored for oracle mode
        theta0_floor=1e-12,
    )


def run_gap_tilde(
    sim_dir: Path,
    rep_ids: list[int],
    K: int,
    eps: float,
    eps1: float,
    delta: float,
    a0: float,
    A0: float,
    A0_mode: str,
    A0_factor: float,
    nstart: int,
    base_seed: int,
    verbose: bool,
) -> pd.DataFrame:
    """GapPTR with noisy/estimated theta0 using a fixed eps1 budget."""
    return run_improved_gapptr_from_saved(
        sim_dir=str(sim_dir),
        rep_ids=rep_ids,
        K=K,
        eps_list=[float(eps)],
        nstart=nstart,
        base_seed=base_seed,
        verbose=verbose,
        a0=float(a0),
        A0=float(A0),
        A0_mode=A0_mode,
        A0_factor=float(A0_factor),
        eig_sort="abs",
        delta=float(delta),
        theta0_mode="use_tilde",
        eps1_mode="fixed",
        eps1=float(eps1),
        theta0_floor=1e-12,
    )


def run_edgeflip_safe(
    sim_dir: Path,
    rep_ids: list[int],
    K: int,
    eps: float,
    delta: Optional[float],
    nstart: int,
    base_seed: int,
    index_base_edges: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Run EdgeFlip from saved networks.

    The current Utility/Edge_flip.py is expected to accept delta. For compatibility
    with older versions, this wrapper falls back to calling without delta.
    """
    base_kwargs = dict(
        sim_dir=str(sim_dir),
        rep_ids=rep_ids,
        K=K,
        eps_list=[float(eps)],
        nstart=nstart,
        base_seed=base_seed,
        verbose=verbose,
        index_base_edges=index_base_edges,
    )

    if delta is not None:
        try:
            return run_edgeflip_from_saved(**base_kwargs, delta=float(delta))
        except TypeError:
            print("[warn] Utility.Edge_flip.run_edgeflip_from_saved does not accept delta; calling without delta.")

    return run_edgeflip_from_saved(**base_kwargs)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_setting(df: pd.DataFrame, setting: str, out_path: Path, shade: str = "none") -> None:
    """Plot mean clustering error versus theta0 for one setting."""
    sub = df[df["setting"] == setting].copy()
    if sub.empty:
        print(f"[plot] no rows for {setting}; skip {out_path}")
        return

    label_map = {
        "GapPTR_oracle": r"GapPTR oracle $(\epsilon_1=0)$",
        "GapPTR_tilde": r"GapPTR tilde $(\epsilon_1>0)$",
        "EdgeFlip": "EdgeFlip",
        "NonPrivate": "NonPrivate",
    }
    color_map = {
        "GapPTR_oracle": "#D55E00",
        "GapPTR_tilde": "#E69F00",
        "EdgeFlip": "#0072B2",
        "NonPrivate": "#9467bd",
    }
    linestyle_map = {
        "GapPTR_oracle": "-",
        "GapPTR_tilde": "-.",
        "EdgeFlip": "-",
        "NonPrivate": "--",
    }
    marker_map = {
        "GapPTR_oracle": "o",
        "GapPTR_tilde": "^",
        "EdgeFlip": "o",
        "NonPrivate": "s",
    }
    plot_order = ["GapPTR_oracle", "GapPTR_tilde", "EdgeFlip", "NonPrivate"]

    plt.figure(figsize=(6.2, 4.4))
    for method in plot_order:
        ss = sub[sub["method"] == method].sort_values("theta0")
        if ss.empty:
            continue

        x = ss["theta0"].to_numpy(dtype=float)
        y = ss["mean_err"].to_numpy(dtype=float)
        plt.plot(
            x,
            y,
            marker=marker_map[method],
            linestyle=linestyle_map[method],
            color=color_map[method],
            linewidth=2,
            markersize=6,
            label=label_map[method],
        )

        if shade in {"sd", "se"} and "sd_err" in ss.columns:
            sd = ss["sd_err"].to_numpy(dtype=float)
            n_rep = ss["n_rep"].to_numpy(dtype=float)
            band = sd if shade == "sd" else sd / np.sqrt(np.maximum(n_rep, 1.0))
            if np.any(np.isfinite(band)):
                lo = np.maximum(0.0, y - band)
                hi = np.minimum(1.0, y + band)
                plt.fill_between(x, lo, hi, color=color_map[method], alpha=0.15, linewidth=0)

    eps_all_vals = sorted(v for v in sub["eps_all"].dropna().unique())
    eps_title = f"$\\epsilon_{{all}}={eps_all_vals[0]:.3g}$" if len(eps_all_vals) == 1 else ""
    plt.xlabel(r"$\theta_0$")
    plt.ylabel("Mean clustering error")
    if eps_title:
        plt.title(eps_title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_combined(df: pd.DataFrame, out_path: Path, shade: str = "none") -> None:
    """Save a compact combined figure with one panel per setting."""
    # To obey simple plotting requirements in earlier scripts, this function makes
    # a separate image but uses matplotlib subplots only if explicitly called by this script.
    # The main setting plots are still saved separately.
    settings = [s for s in ["setting1", "setting2"] if s in set(df["setting"])]
    if not settings:
        return

    # Create one simple combined plot by offsetting labels; users can ignore it if not needed.
    plt.figure(figsize=(6.8, 4.8))
    method_order = ["GapPTR_oracle", "GapPTR_tilde", "EdgeFlip", "NonPrivate"]
    markers = {"setting1": "o", "setting2": "^"}
    labels = {"setting1": "Setting 1", "setting2": "Setting 2"}
    colors = {
        "GapPTR_oracle": "#D55E00",
        "GapPTR_tilde": "#E69F00",
        "EdgeFlip": "#0072B2",
        "NonPrivate": "#9467bd",
    }
    for method in method_order:
        for setting in settings:
            ss = df[(df["setting"] == setting) & (df["method"] == method)].sort_values("theta0")
            if ss.empty:
                continue
            plt.plot(
                ss["theta0"],
                ss["mean_err"],
                marker=markers.get(setting, "o"),
                linestyle="-" if setting == "setting1" else "--",
                color=colors[method],
                label=f"{method}, {labels.get(setting, setting)}",
            )
    plt.xlabel(r"$\theta_0$")
    plt.ylabel("Mean clustering error")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved combined plot: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 3 for both theta0 settings.")
    parser.add_argument("--settings", nargs="+", default=["setting1", "setting2"], choices=["setting1", "setting2"],
                        help="Which Experiment 3 settings to run.")
    parser.add_argument("--n-rep", type=int, default=50, help="Number of replications per theta0 folder.")
    parser.add_argument("--rep-start", type=int, default=0, help="First replication index.")
    parser.add_argument("--K", type=int, default=2, help="Number of communities.")
    parser.add_argument("--eps-all", type=float, default=0.8, help="Total privacy budget eps_all = eps + eps1.")
    parser.add_argument("--eps1-tilde", type=float, default=0.2, help="Budget used to estimate theta0 for GapPTR_tilde.")
    parser.add_argument("--delta-gap", type=float, default=0.01, help="Delta for GapPTR.")
    parser.add_argument("--delta-edge", type=float, default=0.01, help="Delta for EdgeFlip if supported by the utility.")
    parser.add_argument("--a0-setting1", type=float, default=None, help="Override a0 for setting1.")
    parser.add_argument("--a0-setting2", type=float, default=None, help="Override a0 for setting2.")
    parser.add_argument("--A0", type=float, default=50.0, help="A0 parameter for GapPTR.")
    parser.add_argument("--A0-mode", type=str, default="from_2toinfty", choices=["fixed", "from_2toinfty"],
                        help="How Utility/gapptr_utility should choose A0.")
    parser.add_argument("--A0-factor", type=float, default=1.05, help="Multiplicative factor when A0-mode is from_2toinfty.")
    parser.add_argument("--nstart", type=int, default=25, help="K-means n_init.")
    parser.add_argument("--base-seed", type=int, default=5000, help="Base random seed.")
    parser.add_argument("--index-base-edges", type=int, default=1, help="Index base used by saved edge files.")
    parser.add_argument("--skip-edgeflip", action="store_true", help="Skip EdgeFlip.")
    parser.add_argument("--skip-tilde", action="store_true", help="Skip GapPTR_tilde.")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip GapPTR_oracle.")
    parser.add_argument("--skip-nonprivate", action="store_true", help="Skip NonPrivate.")
    parser.add_argument("--shade", choices=["none", "sd", "se"], default="none", help="Plot uncertainty band.")
    parser.add_argument("--outdir", type=str, default="results_exp3", help="Output directory.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-replication printing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eps1_tilde >= args.eps_all:
        raise ValueError("Require eps1_tilde < eps_all so that eps_gap_tilde = eps_all - eps1_tilde > 0.")

    rep_ids = list(range(args.rep_start, args.rep_start + args.n_rep))
    eps_gap_oracle = float(args.eps_all)
    eps1_gap_oracle = 0.0
    eps_gap_tilde = float(args.eps_all - args.eps1_tilde)
    eps1_gap_tilde = float(args.eps1_tilde)
    eps_edge = float(args.eps_all)

    outdir = EXPERIMENT_DIR / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    configs = default_setting_configs()
    verbose = not args.quiet
    all_rows = []

    for setting in args.settings:
        cfg = configs[setting]
        base_dir, folders, theta0_map = resolve_setting_folders(EXPERIMENT_DIR, cfg)
        a0 = cfg["a0_default"]
        if setting == "setting1" and args.a0_setting1 is not None:
            a0 = float(args.a0_setting1)
        if setting == "setting2" and args.a0_setting2 is not None:
            a0 = float(args.a0_setting2)

        print("\n" + "=" * 72)
        print(f"{setting}: {cfg['label']}")
        print(f"base_dir={base_dir}")
        print(f"eps_all={args.eps_all}, eps_gap_oracle={eps_gap_oracle}, eps_gap_tilde={eps_gap_tilde}, eps1_tilde={eps1_gap_tilde}")
        print(f"a0={a0}, A0={args.A0}, A0_mode={args.A0_mode}")
        print("=" * 72)

        for folder in folders:
            sim_dir = base_dir / folder
            if not sim_dir.is_dir():
                print(f"[SKIP] missing folder: {sim_dir}")
                continue

            theta0_value = float(theta0_map[folder])
            print(f"\n----- {setting} / {folder}: theta0={theta0_value:.4g} -----")

            if not args.skip_nonprivate:
                np_df = run_nonprivate_from_saved(
                    sim_dir=sim_dir,
                    rep_ids=rep_ids,
                    K=args.K,
                    theta0_value=theta0_value,
                    nstart=args.nstart,
                    base_seed=args.base_seed,
                    verbose=verbose,
                )
                np_avg = summarize_rep_df(np_df, by=("theta0",))
                for _, row in np_avg.iterrows():
                    all_rows.append({
                        "setting": setting,
                        "setting_label": cfg["label"],
                        "folder": folder,
                        "theta0": float(row["theta0"]),
                        "method": "NonPrivate",
                        "eps_all": np.nan,
                        "eps": np.nan,
                        "eps1": np.nan,
                        "delta": np.nan,
                        "mean_err": float(row["mean_err"]),
                        "sd_err": float(row["sd_err"]) if pd.notna(row["sd_err"]) else np.nan,
                        "n_rep": int(row["n_rep"]),
                        "a0": float(a0),
                        "source": "fresh",
                    })

            if not args.skip_edgeflip:
                ef_df = run_edgeflip_safe(
                    sim_dir=sim_dir,
                    rep_ids=rep_ids,
                    K=args.K,
                    eps=eps_edge,
                    delta=args.delta_edge,
                    nstart=args.nstart,
                    base_seed=args.base_seed,
                    index_base_edges=args.index_base_edges,
                    verbose=verbose,
                )
                ef_avg = summarize_rep_df(ef_df, by=("eps",))
                for _, row in ef_avg.iterrows():
                    all_rows.append({
                        "setting": setting,
                        "setting_label": cfg["label"],
                        "folder": folder,
                        "theta0": theta0_value,
                        "method": "EdgeFlip",
                        "eps_all": float(args.eps_all),
                        "eps": float(row["eps"]),
                        "eps1": np.nan,
                        "delta": float(args.delta_edge),
                        "mean_err": float(row["mean_err"]),
                        "sd_err": float(row["sd_err"]) if pd.notna(row["sd_err"]) else np.nan,
                        "n_rep": int(row["n_rep"]),
                        "a0": float(a0),
                        "source": "fresh",
                    })

            if not args.skip_oracle:
                gap_oracle_df = run_gap_oracle(
                    sim_dir=sim_dir,
                    rep_ids=rep_ids,
                    K=args.K,
                    eps=eps_gap_oracle,
                    delta=args.delta_gap,
                    a0=a0,
                    A0=args.A0,
                    A0_mode=args.A0_mode,
                    A0_factor=args.A0_factor,
                    nstart=args.nstart,
                    base_seed=args.base_seed,
                    verbose=verbose,
                )
                gap_oracle_avg = summarize_rep_df(gap_oracle_df, by=("eps",))
                for _, row in gap_oracle_avg.iterrows():
                    all_rows.append({
                        "setting": setting,
                        "setting_label": cfg["label"],
                        "folder": folder,
                        "theta0": theta0_value,
                        "method": "GapPTR_oracle",
                        "eps_all": float(args.eps_all),
                        "eps": float(row["eps"]),
                        "eps1": eps1_gap_oracle,
                        "delta": float(args.delta_gap),
                        "mean_err": float(row["mean_err"]),
                        "sd_err": float(row["sd_err"]) if pd.notna(row["sd_err"]) else np.nan,
                        "n_rep": int(row["n_rep"]),
                        "a0": float(a0),
                        "theta0_mode": "oracle",
                        "eps1_mode": "fixed_zero",
                        "source": "fresh",
                    })

            if not args.skip_tilde:
                gap_tilde_df = run_gap_tilde(
                    sim_dir=sim_dir,
                    rep_ids=rep_ids,
                    K=args.K,
                    eps=eps_gap_tilde,
                    eps1=eps1_gap_tilde,
                    delta=args.delta_gap,
                    a0=a0,
                    A0=args.A0,
                    A0_mode=args.A0_mode,
                    A0_factor=args.A0_factor,
                    nstart=args.nstart,
                    base_seed=args.base_seed,
                    verbose=verbose,
                )
                gap_tilde_avg = summarize_rep_df(gap_tilde_df, by=("eps",))
                for _, row in gap_tilde_avg.iterrows():
                    all_rows.append({
                        "setting": setting,
                        "setting_label": cfg["label"],
                        "folder": folder,
                        "theta0": theta0_value,
                        "method": "GapPTR_tilde",
                        "eps_all": float(args.eps_all),
                        "eps": float(row["eps"]),
                        "eps1": eps1_gap_tilde,
                        "delta": float(args.delta_gap),
                        "mean_err": float(row["mean_err"]),
                        "sd_err": float(row["sd_err"]) if pd.notna(row["sd_err"]) else np.nan,
                        "n_rep": int(row["n_rep"]),
                        "a0": float(a0),
                        "theta0_mode": "use_tilde",
                        "eps1_mode": "fixed",
                        "source": "fresh",
                    })

    if not all_rows:
        raise RuntimeError("No results were produced. Please check the data folders.")

    res = pd.DataFrame(all_rows)
    res = res.sort_values(["setting", "theta0", "method"]).reset_index(drop=True)

    out_csv = outdir / "summary_exp3_all_settings_theta0.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nSaved table: {out_csv}")
    print(res.to_string(index=False))

    for setting in args.settings:
        out_png = outdir / f"plot_exp3_{setting}_theta0.png"
        plot_setting(res, setting=setting, out_path=out_png, shade=args.shade)

    # Optional combined comparison plot.
    out_combined = outdir / "plot_exp3_combined_settings_theta0.png"
    plot_combined(res, out_path=out_combined, shade=args.shade)


if __name__ == "__main__":
    main()
