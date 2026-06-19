"""
Experiment 2 runner: varying the total privacy budget.

This script compares GapPTR with EdgeFlip and the non-private spectral
clustering benchmark on the two Experiment 2 simulation folders

    E2S1/    # Scenario 1
    E2S2/    # Scenario 2

It is intended to be placed under the Experiment2/ folder, with the shared
GapPTR utility stored under ../Utility/gapptr_utility.py.

Default folder layout
---------------------
ProjectRoot/
  Utility/
    gapptr_utility.py
    Edge_flip.py              # optional, if EdgeFlip is stored as a utility
  Experiment2/
    run_exp2_vary_eps_all_scenarios.py
    E2S1/
      rep000/
      rep001/
      ...
    E2S2/
      rep000/
      rep001/
      ...

Main privacy split
------------------
For each total privacy budget eps_all:

* GapPTR_oracle: eps = eps_all, eps1 = 0.
* GapPTR_tilde:  eps = eps_all - 0.2, eps1 = 0.2.
* EdgeFlip:      eps = eps_all.
* NonPrivate:    no privacy noise; repeated across the eps_all grid for plotting.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
DEFAULT_PROJECT_ROOT = THIS_DIR.parent


for p in [DEFAULT_PROJECT_ROOT, DEFAULT_PROJECT_ROOT / "Utility", THIS_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from Utility.gapptr_utility import (  # noqa: E402
    clustering_error,
    load_Xi_hat_from_csv,
    run_improved_gapptr_from_saved,
    try_load_labels_true,
)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def discover_rep_ids(sim_dir: str | Path) -> list[int]:
    """Discover replication IDs from folders named rep000 or rep_000."""
    sim_dir = Path(sim_dir)
    rep_ids: list[int] = []
    for item in sorted(sim_dir.iterdir()):
        if not item.is_dir():
            continue
        match = re.match(r"rep[_]?(\d+)$", item.name)
        if match:
            rep_ids.append(int(match.group(1)))
    return rep_ids


def make_rep_ids(sim_dir: str | Path, n_rep: Optional[int]) -> list[int]:
    """
    Return the replication IDs to use.

    If n_rep is supplied, use range(n_rep). Otherwise discover all available
    replication folders under sim_dir.
    """
    if n_rep is not None:
        return list(range(int(n_rep)))
    rep_ids = discover_rep_ids(sim_dir)
    if not rep_ids:
        raise FileNotFoundError(f"No rep folders found under {sim_dir}.")
    return rep_ids


def rep_dir_from_id(sim_dir: str | Path, rep: int) -> Path:
    """Find one replication folder."""
    sim_dir = Path(sim_dir)
    candidates = [sim_dir / f"rep{rep:03d}", sim_dir / f"rep_{rep:03d}", sim_dir / f"rep{rep:04d}"]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Cannot locate replication {rep} under {sim_dir}.")


def cluster_from_embedding(
    Xi: np.ndarray,
    K: int,
    nstart: int = 25,
    norm_tol: float = 1e-12,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Row-normalize a spectral embedding and run k-means."""
    Xi = np.asarray(Xi, dtype=float)
    row_norm = np.sqrt((Xi * Xi).sum(axis=1))
    row_norm = np.maximum(row_norm, norm_tol)
    X_normalized = Xi / row_norm[:, None]
    km = KMeans(n_clusters=K, n_init=nstart, random_state=seed)
    return km.fit_predict(X_normalized)


def run_nonprivate_from_saved(
    sim_dir: str | Path,
    rep_ids: Sequence[int],
    K: int,
    nstart: int = 25,
    base_seed: int = 5000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run non-private spectral clustering from saved eigenvectors."""
    rows: list[dict] = []
    for rep in rep_ids:
        rep_dir = rep_dir_from_id(sim_dir, rep)
        Xi_hat = load_Xi_hat_from_csv(str(rep_dir / "A_eigvecs_topK.csv"), K=K)
        z_true = try_load_labels_true(str(rep_dir))
        if z_true is None:
            raise FileNotFoundError(f"Cannot load true labels from {rep_dir}.")
        z_true = np.asarray(z_true, dtype=int).reshape(-1)

        labels = cluster_from_embedding(
            Xi_hat,
            K=K,
            nstart=nstart,
            seed=base_seed + int(rep),
        )
        err = clustering_error(z_true, labels)
        rows.append({"rep": int(rep), "clustering_error_vs_truth": float(err)})

        if verbose:
            print(f"[NonPrivate] rep={rep:03d}, err={err:.6f}")

    return pd.DataFrame(rows)


def summarize_rep_df(df: pd.DataFrame, by: Sequence[str] = ("eps",)) -> pd.DataFrame:
    """Summarize replication-level clustering errors."""
    out = (
        df.groupby(list(by), as_index=False)
        .agg(
            mean_err=("clustering_error_vs_truth", "mean"),
            sd_err=("clustering_error_vs_truth", "std"),
            n_rep=("rep", "nunique"),
        )
        .sort_values(list(by))
    )
    out["se_err"] = out["sd_err"] / np.sqrt(out["n_rep"].clip(lower=1))
    return out


# -----------------------------------------------------------------------------
# EdgeFlip import
# -----------------------------------------------------------------------------


def import_edgeflip_runner() -> Callable:
    
    candidate_modules = [
        "Utility.Edge_flip",
        "Utility.edge_flip",
        "Edge_flip",
        "edge_flip",
    ]

    errors: list[str] = []
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        if hasattr(module, "run_edgeflip_from_saved"):
            return getattr(module, "run_edgeflip_from_saved")
        errors.append(f"{module_name}: no run_edgeflip_from_saved")

    msg = "\n".join(errors)
    raise ImportError(
        "Cannot import run_edgeflip_from_saved. Put Edge_flip.py under Utility/ "
        "or next to this script, or run with --skip-edgeflip.\n"
        f"Import attempts:\n{msg}"
    )


def run_edgeflip_with_fallback(
    edgeflip_runner: Callable,
    sim_dir: str | Path,
    rep_ids: Sequence[int],
    K: int,
    eps_list: Iterable[float],
    delta: float,
    nstart: int,
    base_seed: int,
    verbose: bool,
    index_base_edges: int,
) -> pd.DataFrame:
    """
    Call the project EdgeFlip function while allowing minor signature changes.
    """
    kwargs = dict(
        sim_dir=str(sim_dir),
        rep_ids=list(rep_ids),
        K=K,
        eps_list=list(eps_list),
        delta=delta,
        nstart=nstart,
        base_seed=base_seed,
        verbose=verbose,
        index_base_edges=index_base_edges,
    )

    try:
        return edgeflip_runner(**kwargs)
    except TypeError:
        # Some older versions may not include index_base_edges.
        kwargs2 = dict(kwargs)
        kwargs2.pop("index_base_edges", None)
        return edgeflip_runner(**kwargs2)


# -----------------------------------------------------------------------------
# Main Experiment 2 computation
# -----------------------------------------------------------------------------


def run_one_scenario(
    scenario: int,
    sim_dir: str | Path,
    rep_ids: Sequence[int],
    eps_all_list: np.ndarray,
    eps1_tilde: float,
    K: int,
    delta: float,
    a0: float,
    A0: float,
    A0_mode: str,
    A0_factor: float,
    nstart: int,
    base_seed: int,
    verbose: bool,
    index_base_edges: int,
    skip_edgeflip: bool,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run all methods for one scenario folder."""
    sim_dir = Path(sim_dir)
    if not sim_dir.is_dir():
        raise FileNotFoundError(f"Scenario folder does not exist: {sim_dir}")

    eps_all_list = np.asarray(eps_all_list, dtype=float)
    eps_gap_tilde_list = eps_all_list - float(eps1_tilde)
    if np.any(eps_gap_tilde_list <= 0):
        raise ValueError(
            "Some eps_all values are too small for the tilde split. "
            f"Need eps_all > eps1_tilde={eps1_tilde}."
        )

    all_rows: list[dict] = []
    raw: dict[str, pd.DataFrame] = {}

    print(f"\n{'=' * 80}")
    print(f"Experiment 2, Scenario {scenario}: {sim_dir}")
    print(f"Using rep_ids={list(rep_ids)}")
    print(f"{'=' * 80}")

    # ------------------------------------------------------------------
    # GapPTR oracle: eps = eps_all, eps1 = 0
    # ------------------------------------------------------------------
    print("\n=== GapPTR oracle: eps = eps_all, eps1 = 0 ===")
    gap_oracle_df = run_improved_gapptr_from_saved(
        sim_dir=str(sim_dir),
        rep_ids=list(rep_ids),
        K=K,
        eps_list=list(eps_all_list),
        nstart=nstart,
        base_seed=base_seed,
        verbose=verbose,
        theta0_mode="oracle",
        eps1_mode="same_as_eps",  # ignored under oracle mode
        theta0_floor=1e-12,
        a0=a0,
        A0=A0,
        A0_mode=A0_mode,
        A0_factor=A0_factor,
        eig_sort="abs",
        delta=delta,
    )
    raw["GapPTR_oracle"] = gap_oracle_df
    gap_oracle_avg = summarize_rep_df(gap_oracle_df, by=("eps",))

    for _, r in gap_oracle_avg.iterrows():
        eps = float(r["eps"])
        all_rows.append(
            dict(
                scenario=int(scenario),
                scenario_dir=str(sim_dir),
                log_eps_all=float(np.log(eps)),
                eps_all=float(eps),
                eps=float(eps),
                eps1=0.0,
                method="GapPTR_oracle",
                mean_err=float(r["mean_err"]),
                sd_err=float(r["sd_err"]) if pd.notna(r["sd_err"]) else np.nan,
                se_err=float(r["se_err"]) if pd.notna(r["se_err"]) else np.nan,
                n_rep=int(r["n_rep"]),
                a0=float(a0),
                delta=float(delta),
            )
        )

    # ------------------------------------------------------------------
    # GapPTR tilde: eps = eps_all - eps1_tilde, eps1 = eps1_tilde
    # ------------------------------------------------------------------
    print(f"\n=== GapPTR tilde: eps1 = {eps1_tilde}, eps = eps_all - {eps1_tilde} ===")
    gap_tilde_df = run_improved_gapptr_from_saved(
        sim_dir=str(sim_dir),
        rep_ids=list(rep_ids),
        K=K,
        eps_list=list(eps_gap_tilde_list),
        nstart=nstart,
        base_seed=base_seed,
        verbose=verbose,
        theta0_mode="use_tilde",
        eps1_mode="fixed",
        eps1=float(eps1_tilde),
        theta0_floor=1e-12,
        a0=a0,
        A0=A0,
        A0_mode=A0_mode,
        A0_factor=A0_factor,
        eig_sort="abs",
        delta=delta,
    )
    raw["GapPTR_tilde"] = gap_tilde_df
    gap_tilde_avg = summarize_rep_df(gap_tilde_df, by=("eps",))

    for eps_all, eps_run in zip(eps_all_list, eps_gap_tilde_list):
        matched = gap_tilde_avg[np.isclose(gap_tilde_avg["eps"].astype(float), eps_run)]
        if matched.empty:
            raise RuntimeError(f"Cannot match GapPTR_tilde summary for eps={eps_run}.")
        r = matched.iloc[0]
        all_rows.append(
            dict(
                scenario=int(scenario),
                scenario_dir=str(sim_dir),
                log_eps_all=float(np.log(eps_all)),
                eps_all=float(eps_all),
                eps=float(eps_run),
                eps1=float(eps1_tilde),
                method="GapPTR_tilde",
                mean_err=float(r["mean_err"]),
                sd_err=float(r["sd_err"]) if pd.notna(r["sd_err"]) else np.nan,
                se_err=float(r["se_err"]) if pd.notna(r["se_err"]) else np.nan,
                n_rep=int(r["n_rep"]),
                a0=float(a0),
                delta=float(delta),
            )
        )

    # ------------------------------------------------------------------
    # EdgeFlip: eps = eps_all
    # ------------------------------------------------------------------
    if not skip_edgeflip:
        print("\n=== EdgeFlip: eps = eps_all ===")
        edgeflip_runner = import_edgeflip_runner()
        ef_df = run_edgeflip_with_fallback(
            edgeflip_runner=edgeflip_runner,
            sim_dir=sim_dir,
            rep_ids=rep_ids,
            K=K,
            eps_list=eps_all_list,
            delta=delta,
            nstart=nstart,
            base_seed=base_seed,
            verbose=verbose,
            index_base_edges=index_base_edges,
        )
        raw["EdgeFlip_eps_delta"] = ef_df
        ef_avg = summarize_rep_df(ef_df, by=("eps",))

        for _, r in ef_avg.iterrows():
            eps = float(r["eps"])
            all_rows.append(
                dict(
                    scenario=int(scenario),
                    scenario_dir=str(sim_dir),
                    log_eps_all=float(np.log(eps)),
                    eps_all=float(eps),
                    eps=float(eps),
                    eps1=np.nan,
                    method="EdgeFlip_eps_delta",
                    mean_err=float(r["mean_err"]),
                    sd_err=float(r["sd_err"]) if pd.notna(r["sd_err"]) else np.nan,
                    se_err=float(r["se_err"]) if pd.notna(r["se_err"]) else np.nan,
                    n_rep=int(r["n_rep"]),
                    a0=np.nan,
                    delta=float(delta),
                )
            )

    # ------------------------------------------------------------------
    # Non-private: compute once, repeat across eps_all grid for plotting
    # ------------------------------------------------------------------
    print("\n=== NonPrivate spectral clustering ===")
    np_rep = run_nonprivate_from_saved(
        sim_dir=sim_dir,
        rep_ids=rep_ids,
        K=K,
        nstart=nstart,
        base_seed=base_seed,
        verbose=verbose,
    )
    raw["NonPrivate"] = np_rep
    np_mean = float(np_rep["clustering_error_vs_truth"].mean())
    np_sd = float(np_rep["clustering_error_vs_truth"].std())
    np_se = float(np_sd / np.sqrt(max(len(rep_ids), 1)))

    for eps_all in eps_all_list:
        all_rows.append(
            dict(
                scenario=int(scenario),
                scenario_dir=str(sim_dir),
                log_eps_all=float(np.log(eps_all)),
                eps_all=float(eps_all),
                eps=np.nan,
                eps1=np.nan,
                method="NonPrivate",
                mean_err=np_mean,
                sd_err=np_sd,
                se_err=np_se,
                n_rep=len(rep_ids),
                a0=np.nan,
                delta=np.nan,
            )
        )

    summary = pd.DataFrame(all_rows).sort_values(["scenario", "log_eps_all", "method"])
    return summary, raw


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_one_scenario(
    summary: pd.DataFrame,
    out_png: str | Path,
    title: Optional[str] = None,
    shade: str = "none",
) -> None:
    """Plot mean clustering error against log(eps_all)."""
    out_png = Path(out_png)

    color_map = {
        "GapPTR_oracle": "#D55E00",      # vermillion
        "GapPTR_tilde": "#E69F00",       # orange
        "EdgeFlip_eps_delta": "#0072B2", # blue
        "NonPrivate": "#9467bd",         # purple
    }
    label_map = {
        "EdgeFlip_eps_delta": r"EdgeFlip $(\epsilon,\delta)$",
        "GapPTR_oracle": r"GapPTR $(\theta_0\ \mathrm{oracle},\ \epsilon_1=0)$",
        "GapPTR_tilde": r"GapPTR $(\theta_0=\widetilde{\theta}_0,\ \epsilon_1=0.2)$",
        "NonPrivate": "NonPrivate",
    }
    marker_map = {
        "GapPTR_oracle": "o",
        "GapPTR_tilde": "^",
        "EdgeFlip_eps_delta": "o",
        "NonPrivate": "s",
    }
    linestyle_map = {
        "GapPTR_oracle": "-",
        "GapPTR_tilde": "-",
        "EdgeFlip_eps_delta": "-",
        "NonPrivate": "--",
    }
    plot_order = ["EdgeFlip_eps_delta", "GapPTR_oracle", "GapPTR_tilde", "NonPrivate"]

    plt.figure(figsize=(7.2, 4.8))
    for method in plot_order:
        sub = summary[summary["method"] == method].copy().sort_values("log_eps_all")
        if sub.empty:
            continue

        x = sub["log_eps_all"].to_numpy(dtype=float)
        y = sub["mean_err"].to_numpy(dtype=float)
        plt.plot(
            x,
            y,
            marker=marker_map[method],
            linestyle=linestyle_map[method],
            linewidth=2,
            color=color_map[method],
            label=label_map[method],
        )

        if shade in {"se", "sd"} and method != "NonPrivate":
            err_col = "se_err" if shade == "se" else "sd_err"
            err = sub[err_col].to_numpy(dtype=float)
            ok = np.isfinite(err)
            if ok.any():
                plt.fill_between(x[ok], y[ok] - err[ok], y[ok] + err[ok], alpha=0.15, color=color_map[method])

    plt.xlabel(r"$\log(\epsilon_{\mathrm{all}})$")
    plt.ylabel("Mean clustering error")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved plot: {out_png}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 2 for E2S1 and E2S2.")

    parser.add_argument("--data-root", type=str, default=".", help="Folder containing E2S1 and E2S2.")
    parser.add_argument("--outdir", type=str, default="results_exp2", help="Output folder for CSVs and plots.")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    parser.add_argument("--scenario-dirs", type=str, nargs="+", default=None,
                        help="Optional explicit scenario directories. Use two values for scenarios 1 and 2.")
    parser.add_argument("--n-rep", type=int, default=None,
                        help="Number of replications to use. If omitted, discover all rep folders.")

    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--a0", type=float, default=0.3)
    parser.add_argument("--A0", type=float, default=10.0)
    parser.add_argument("--A0-mode", type=str, default="from_2toinfty", choices=["fixed", "from_2toinfty"])
    parser.add_argument("--A0-factor", type=float, default=1.05)
    parser.add_argument("--eps1-tilde", type=float, default=0.2,
                        help="Fixed eps1 for the private theta0 estimate. Paper default is 0.2.")
    parser.add_argument("--log-eps-min", type=float, default=-1.0)
    parser.add_argument("--log-eps-max", type=float, default=1.0)
    parser.add_argument("--log-eps-step", type=float, default=0.25)

    parser.add_argument("--nstart", type=int, default=25)
    parser.add_argument("--base-seed", type=int, default=5000)
    parser.add_argument("--index-base-edges", type=int, default=1)
    parser.add_argument("--skip-edgeflip", action="store_true",
                        help="Run only GapPTR and NonPrivate if EdgeFlip code is unavailable.")
    parser.add_argument("--shade", type=str, default="none", choices=["none", "se", "sd"],
                        help="Optional uncertainty band for private methods.")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.scenario_dirs is None:
        scenario_dir_map = {1: data_root / "E2S1", 2: data_root / "E2S2"}
    else:
        if len(args.scenario_dirs) != len(args.scenarios):
            raise ValueError("--scenario-dirs must have the same length as --scenarios.")
        scenario_dir_map = {
            int(s): Path(d).expanduser().resolve()
            for s, d in zip(args.scenarios, args.scenario_dirs)
        }

    log_eps_all_list = np.arange(
        args.log_eps_min,
        args.log_eps_max + 0.5 * args.log_eps_step,
        args.log_eps_step,
    )
    eps_all_list = np.exp(log_eps_all_list)

    print("\nExperiment 2 privacy grid")
    print(pd.DataFrame({"log_eps_all": log_eps_all_list, "eps_all": eps_all_list}).to_string(index=False))
    print(f"\nGapPTR tilde split: eps1={args.eps1_tilde}, eps=eps_all-{args.eps1_tilde}")

    all_summaries: list[pd.DataFrame] = []

    for scenario in args.scenarios:
        sim_dir = scenario_dir_map[int(scenario)]
        rep_ids = make_rep_ids(sim_dir, args.n_rep)

        summary, raw = run_one_scenario(
            scenario=int(scenario),
            sim_dir=sim_dir,
            rep_ids=rep_ids,
            eps_all_list=eps_all_list,
            eps1_tilde=args.eps1_tilde,
            K=args.K,
            delta=args.delta,
            a0=args.a0,
            A0=args.A0,
            A0_mode=args.A0_mode,
            A0_factor=args.A0_factor,
            nstart=args.nstart,
            base_seed=args.base_seed,
            verbose=not args.quiet,
            index_base_edges=args.index_base_edges,
            skip_edgeflip=args.skip_edgeflip,
        )

        scen_tag = f"s{scenario}"
        summary_csv = outdir / f"summary_exp2_{scen_tag}.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"Saved table: {summary_csv}")

        # Save replication-level diagnostics where available.
        for method, df in raw.items():
            raw_csv = outdir / f"raw_exp2_{scen_tag}_{method}.csv"
            df.to_csv(raw_csv, index=False)
            print(f"Saved raw diagnostics: {raw_csv}")

        plot_one_scenario(
            summary=summary,
            out_png=outdir / f"plot_exp2_{scen_tag}.png",
            title=f"Experiment 2, Scenario {scenario}",
            shade=args.shade,
        )

        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True).sort_values(["scenario", "log_eps_all", "method"])
    combined_csv = outdir / "summary_exp2_all_scenarios.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"\nSaved combined table: {combined_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
