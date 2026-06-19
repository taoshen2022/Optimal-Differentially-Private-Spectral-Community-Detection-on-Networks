#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1 runner: GapPTR on saved DCSBM simulation folders.

This script is intended to be placed under the Experiment 1 folder, while the
shared GapPTR functions are placed under

    ProjectRoot/Utility/gapptr_utility.py

Recommended project layout:

    ProjectRoot/
      Utility/
        gapptr_utility.py
      Experiment1/
        run_exp1_all_scenarios.py
        E1S15000/
        E1S110000/
        ...
        E1S25000/
        E1S210000/
        ...

Folder naming convention
------------------------
For Scenario 1, the script expects folders of the form

    E1S1{n}

for example ``E1S15000`` and ``E1S110000``.

For Scenario 2, the script expects folders of the form

    E1S2{n}

for example ``E1S25000`` and ``E1S210000``.

Each folder should contain replication folders such as ``rep000``, ``rep001``,
..., with the files produced by the simulation data generator.

What this script runs
---------------------
For each scenario and each network size, the script runs:

1. NonPrivate spectral clustering, using the saved sample eigenvectors.
2. GapPTR Case A: oracle theta0, no eps1 budget.
3. GapPTR Case B: noisy/estimated theta0, with fixed eps1=0.2.

The output includes long-format CSV files and plots for each scenario/case, plus
one combined CSV over all scenarios and cases.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------
# Import the shared Utility module
# ---------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utility.gapptr_utility import (  # noqa: E402
    clustering_error,
    load_Xi_hat_from_csv,
    run_improved_gapptr_from_saved,
    try_load_labels_true,
)


# ---------------------------------------------------------------------
# Default experiment configuration
# ---------------------------------------------------------------------

DEFAULT_N_LIST = [
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000,
]

DEFAULT_EPS_LIST = [0.5, 0.8, 1.0, 2.0]

SCENARIOS = {
    1: {
        "name": "Scenario 1",
        "folder_prefix": "E1S1",
        "description": "relatively sparse baseline with regular heterogeneous degrees",
    },
    2: {
        "name": "Scenario 2",
        "folder_prefix": "E1S2",
        "description": "denser baseline with a low-degree node mixture",
    },
}


@dataclass(frozen=True)
class GapPTRCase:
    """Configuration for one GapPTR variant."""

    case_id: str
    case_name: str
    theta0_mode: str
    eps1_mode: str = "fixed"
    eps1: float | None = None
    theta0: float = 0.9
    plot_label: str = ""


DEFAULT_CASES = {
    "oracle": GapPTRCase(
        case_id="oracle",
        case_name="Case A: oracle theta0",
        theta0_mode="oracle",
        eps1_mode="fixed",
        eps1=None,
        theta0=0.9,
        plot_label=r"GapPTR oracle $\theta_0$",
    ),
    "tilde": GapPTRCase(
        case_id="tilde",
        case_name=r"Case B: noisy $\widetilde{\theta}_0$",
        theta0_mode="use_tilde",
        eps1_mode="fixed",
        eps1=0.2,
        theta0=0.9,
        plot_label=r"GapPTR $\widetilde{\theta}_0$, $\epsilon_1=0.2$",
    ),
}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def scenario_dir(data_root: Path, scenario_id: int, n: int) -> Path:
    """Return the expected folder for a scenario and network size."""
    prefix = SCENARIOS[scenario_id]["folder_prefix"]
    return data_root / f"{prefix}{n}"


def find_rep_dir(sim_dir: Path, rep: int) -> Path:
    """Find a replication directory under ``sim_dir``."""
    candidates = [
        sim_dir / f"rep{rep:03d}",
        sim_dir / f"rep_{rep:03d}",
        sim_dir / f"rep{rep:04d}",
        sim_dir / f"rep_{rep:04d}",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Cannot locate rep={rep} under {sim_dir}")


def cluster_from_embedding(
    Xi: np.ndarray,
    K: int,
    nstart: int = 25,
    norm_tol: float = 1e-12,
    seed: int | None = None,
) -> np.ndarray:
    """Run row-normalized K-means on the spectral embedding."""
    Xi = np.asarray(Xi, dtype=float)
    row_norm = np.sqrt((Xi * Xi).sum(axis=1))
    row_norm = np.maximum(row_norm, norm_tol)
    Xn = Xi / row_norm[:, None]

    km = KMeans(n_clusters=K, n_init=nstart, random_state=seed)
    return km.fit_predict(Xn)


def run_nonprivate_from_saved(
    sim_dir: Path,
    rep_ids: Sequence[int],
    K: int,
    nstart: int = 25,
    base_seed: int = 5000,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run the non-private spectral clustering baseline from saved eigenvectors.

    This function reads ``A_eigvecs_topK.csv`` from each replication folder and
    compares the resulting K-means labels against the saved ground-truth labels.
    """
    records = []

    for rep in rep_ids:
        rep_dir = find_rep_dir(sim_dir, rep)
        eigvec_path = rep_dir / "A_eigvecs_topK.csv"

        if not eigvec_path.exists():
            raise FileNotFoundError(f"Missing {eigvec_path}")

        Xi = load_Xi_hat_from_csv(str(eigvec_path), K=K)
        z_true = try_load_labels_true(str(rep_dir))

        if z_true is None:
            raise FileNotFoundError(
                f"Cannot find ground-truth labels in {rep_dir}. "
                "Expected labels in A_eigvecs_topK.csv or omega_theta_z.csv."
            )

        z_true = np.asarray(z_true, dtype=int).reshape(-1)
        if z_true.min() == 1 and z_true.max() == K:
            z_true = z_true - 1

        labels = cluster_from_embedding(
            Xi=Xi,
            K=K,
            nstart=nstart,
            seed=base_seed + int(rep),
        )

        err = clustering_error(z_true, labels)

        records.append(
            {
                "rep": int(rep),
                "n": int(Xi.shape[0]),
                "clustering_error_vs_truth": float(err),
            }
        )

        if verbose:
            print(f"[NonPrivate] {sim_dir.name}, rep={rep:03d}, err={err:.4g}")

    return pd.DataFrame(records)


def summarize_gap_df(
    gap_df: pd.DataFrame,
    scenario_id: int,
    case: GapPTRCase,
) -> pd.DataFrame:
    """Summarize GapPTR output by epsilon."""
    out = (
        gap_df.groupby(["n", "eps"], as_index=False)
        .agg(
            mean_err=("clustering_error_vs_truth", "mean"),
            sd_err=("clustering_error_vs_truth", "std"),
            n_rep=("rep", "nunique"),
            mean_pA=("pA", "mean"),
            mean_noise=("noise_scale", "mean"),
            mean_gammaE=("gamma_E", "mean"),
            mean_theta0_used=("theta0_used", "mean"),
            mean_dmaxA=("dmaxA", "mean"),
        )
        .sort_values(["n", "eps"])
    )

    out["se_err"] = out["sd_err"] / np.sqrt(out["n_rep"])
    out["scenario"] = int(scenario_id)
    out["scenario_name"] = SCENARIOS[scenario_id]["name"]
    out["case"] = case.case_id
    out["case_name"] = case.case_name
    out["method"] = "GapPTR"

    return out


def summarize_nonprivate_df(np_df: pd.DataFrame, scenario_id: int) -> pd.DataFrame:
    """Summarize the non-private baseline by network size."""
    out = (
        np_df.groupby("n", as_index=False)
        .agg(
            mean_err=("clustering_error_vs_truth", "mean"),
            sd_err=("clustering_error_vs_truth", "std"),
            n_rep=("rep", "nunique"),
        )
        .sort_values("n")
    )
    out["se_err"] = out["sd_err"] / np.sqrt(out["n_rep"])
    out["scenario"] = int(scenario_id)
    out["scenario_name"] = SCENARIOS[scenario_id]["name"]
    out["method"] = "NonPrivate"
    return out


def plot_one_case(
    summary: pd.DataFrame,
    nonprivate_summary: pd.DataFrame,
    scenario_id: int,
    case: GapPTRCase,
    out_path: Path,
) -> None:
    """Plot mean clustering error versus n for one scenario and one case."""
    plt.figure(figsize=(7.0, 4.8))

    for eps in sorted(summary["eps"].unique()):
        tmp = summary.loc[summary["eps"] == eps].sort_values("n")
        eps1_text = ""
        if case.eps1 is not None:
            eps1_text = rf", $\epsilon_1={case.eps1:g}$"
        plt.plot(
            tmp["n"].values,
            tmp["mean_err"].values,
            marker="o",
            label=rf"$\epsilon={eps:g}$" + eps1_text,
        )

    np_tmp = nonprivate_summary.sort_values("n")
    plt.plot(
        np_tmp["n"].values,
        np_tmp["mean_err"].values,
        marker="s",
        linestyle="--",
        label="NonPrivate",
    )

    title = f"Experiment 1, {SCENARIOS[scenario_id]['name']}, {case.case_name}"
    plt.xlabel("n")
    plt.ylabel("Mean clustering error")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_all_cases_for_scenario(
    scenario_summary: pd.DataFrame,
    nonprivate_summary: pd.DataFrame,
    scenario_id: int,
    out_path: Path,
) -> None:
    """
    Plot all GapPTR cases for one scenario.

    This plot can be crowded because it overlays both cases and all epsilon
    values. The per-case plots are usually cleaner for the paper.
    """
    plt.figure(figsize=(8.2, 5.4))

    for (case_id, eps), tmp in scenario_summary.groupby(["case", "eps"]):
        tmp = tmp.sort_values("n")
        case_name = str(tmp["case_name"].iloc[0])
        plt.plot(
            tmp["n"].values,
            tmp["mean_err"].values,
            marker="o",
            label=f"{case_id}, eps={eps:g}",
        )

    np_tmp = nonprivate_summary.sort_values("n")
    plt.plot(
        np_tmp["n"].values,
        np_tmp["mean_err"].values,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="NonPrivate",
    )

    plt.xlabel("n")
    plt.ylabel("Mean clustering error")
    plt.title(f"Experiment 1, {SCENARIOS[scenario_id]['name']}: all cases")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


# ---------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------

def run_exp1(
    data_root: Path,
    out_dir: Path,
    scenarios: Sequence[int],
    cases: Sequence[GapPTRCase],
    n_list: Sequence[int],
    rep_ids: Sequence[int],
    eps_list: Sequence[float],
    K: int = 2,
    delta: float = 0.01,
    a0: float = 0.30,
    A0: float = 50.0,
    nstart: int = 25,
    base_seed: int = 5000,
    theta0_floor: float = 1e-12,
    A0_mode: str = "fixed",
    A0_factor: float = 1.05,
    eig_sort: str = "abs",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all requested Experiment 1 scenarios and cases."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gap_summaries = []
    all_gap_long = []
    all_nonprivate_summaries = []

    for scenario_id in scenarios:
        scenario_name = SCENARIOS[scenario_id]["name"]
        print(f"\n================ {scenario_name} ================")

        # NonPrivate is shared across all GapPTR cases.
        np_by_n = []

        for n in n_list:
            sim_dir = scenario_dir(data_root, scenario_id, n)

            if not sim_dir.is_dir():
                print(f"[WARN] Missing folder: {sim_dir}. Skipping n={n}.")
                continue

            print(f"\n--- NonPrivate | {scenario_name}, n={n}, dir={sim_dir.name} ---")
            np_df = run_nonprivate_from_saved(
                sim_dir=sim_dir,
                rep_ids=rep_ids,
                K=K,
                nstart=nstart,
                base_seed=base_seed,
                verbose=False,
            )
            np_df["scenario"] = scenario_id
            np_by_n.append(np_df)

        if not np_by_n:
            print(f"[WARN] No folders found for {scenario_name}.")
            continue

        np_all = pd.concat(np_by_n, ignore_index=True)
        np_summary = summarize_nonprivate_df(np_all, scenario_id=scenario_id)
        np_summary.to_csv(out_dir / f"summary_exp1_s{scenario_id}_nonprivate.csv", index=False)
        all_nonprivate_summaries.append(np_summary)

        # Run each GapPTR case.
        scenario_case_summaries = []

        for case in cases:
            print(f"\n============ {scenario_name} | {case.case_name} ============")
            case_long_parts = []
            case_summary_parts = []

            for n in n_list:
                sim_dir = scenario_dir(data_root, scenario_id, n)

                if not sim_dir.is_dir():
                    print(f"[WARN] Missing folder: {sim_dir}. Skipping n={n}.")
                    continue

                print(f"\n--- GapPTR | {scenario_name}, {case.case_name}, n={n}, dir={sim_dir.name} ---")

                gap_df = run_improved_gapptr_from_saved(
                    sim_dir=str(sim_dir),
                    rep_ids=rep_ids,
                    K=K,
                    eps_list=eps_list,
                    nstart=nstart,
                    base_seed=base_seed,
                    verbose=verbose,
                    theta0_mode=case.theta0_mode,
                    eps1_mode=case.eps1_mode,
                    eps1=case.eps1,
                    theta0_floor=theta0_floor,
                    a0=a0,
                    A0=A0,
                    theta0=case.theta0,
                    A0_mode=A0_mode,
                    A0_factor=A0_factor,
                    eig_sort=eig_sort,
                    delta=delta,
                )

                gap_df["scenario"] = scenario_id
                gap_df["scenario_name"] = scenario_name
                gap_df["case"] = case.case_id
                gap_df["case_name"] = case.case_name

                case_long_parts.append(gap_df)
                case_summary_parts.append(
                    summarize_gap_df(gap_df, scenario_id=scenario_id, case=case)
                )

            if not case_summary_parts:
                print(f"[WARN] No results collected for {scenario_name}, {case.case_name}.")
                continue

            case_long = pd.concat(case_long_parts, ignore_index=True)
            case_summary = pd.concat(case_summary_parts, ignore_index=True)

            case_long_path = out_dir / f"long_exp1_s{scenario_id}_{case.case_id}.csv"
            case_summary_path = out_dir / f"summary_exp1_s{scenario_id}_{case.case_id}.csv"
            case_plot_path = out_dir / f"plot_exp1_s{scenario_id}_{case.case_id}.png"

            case_long.to_csv(case_long_path, index=False)
            case_summary.to_csv(case_summary_path, index=False)

            plot_one_case(
                summary=case_summary,
                nonprivate_summary=np_summary,
                scenario_id=scenario_id,
                case=case,
                out_path=case_plot_path,
            )

            print(f"[SAVE] {case_summary_path}")
            print(f"[SAVE] {case_plot_path}")

            all_gap_long.append(case_long)
            all_gap_summaries.append(case_summary)
            scenario_case_summaries.append(case_summary)

        # Combined plot for this scenario.
        if scenario_case_summaries:
            scenario_summary = pd.concat(scenario_case_summaries, ignore_index=True)
            plot_all_cases_for_scenario(
                scenario_summary=scenario_summary,
                nonprivate_summary=np_summary,
                scenario_id=scenario_id,
                out_path=out_dir / f"plot_exp1_s{scenario_id}_all_cases.png",
            )

    if not all_gap_summaries:
        raise RuntimeError("No GapPTR results were collected. Check folder names and paths.")

    gap_summary_all = pd.concat(all_gap_summaries, ignore_index=True)
    gap_long_all = pd.concat(all_gap_long, ignore_index=True)

    gap_summary_all.to_csv(out_dir / "summary_exp1_all_scenarios_cases.csv", index=False)
    gap_long_all.to_csv(out_dir / "long_exp1_all_scenarios_cases.csv", index=False)

    if all_nonprivate_summaries:
        nonprivate_all = pd.concat(all_nonprivate_summaries, ignore_index=True)
        nonprivate_all.to_csv(out_dir / "summary_exp1_all_nonprivate.csv", index=False)
    else:
        nonprivate_all = pd.DataFrame()

    print(f"\n[SAVE] {out_dir / 'summary_exp1_all_scenarios_cases.csv'}")
    print(f"[SAVE] {out_dir / 'long_exp1_all_scenarios_cases.csv'}")

    return gap_summary_all, nonprivate_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1 for all saved DCSBM scenarios and GapPTR cases."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Folder containing E1S1{n} and E1S2{n} simulation folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results_exp1",
        help="Output folder for CSV summaries and plots.",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=[1, 2],
        choices=[1, 2],
        help="Scenario IDs to run.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=["oracle", "tilde"],
        choices=list(DEFAULT_CASES.keys()),
        help="GapPTR cases to run.",
    )
    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=DEFAULT_N_LIST,
        help="Network sizes to search for.",
    )
    parser.add_argument(
        "--n-rep",
        type=int,
        default=5,
        help="Number of replications per folder. Uses rep IDs 0, ..., n_rep-1.",
    )
    parser.add_argument(
        "--eps-list",
        type=float,
        nargs="+",
        default=DEFAULT_EPS_LIST,
        help="Release epsilon values for GapPTR.",
    )

    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--a0", type=float, default=0.30)
    parser.add_argument("--A0", type=float, default=50.0)
    parser.add_argument("--base-seed", type=int, default=5000)
    parser.add_argument("--nstart", type=int, default=25)
    parser.add_argument("--theta0-floor", type=float, default=1e-12)
    parser.add_argument("--A0-mode", type=str, default="fixed", choices=["fixed", "from_2toinfty"])
    parser.add_argument("--A0-factor", type=float, default=1.05)
    parser.add_argument("--eig-sort", type=str, default="abs", choices=["abs", "value"])
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-replication GapPTR printouts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = [DEFAULT_CASES[name] for name in args.cases]
    rep_ids = list(range(args.n_rep))

    run_exp1(
        data_root=Path(args.data_root).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        scenarios=args.scenarios,
        cases=cases,
        n_list=args.n_list,
        rep_ids=rep_ids,
        eps_list=args.eps_list,
        K=args.K,
        delta=args.delta,
        a0=args.a0,
        A0=args.A0,
        nstart=args.nstart,
        base_seed=args.base_seed,
        theta0_floor=args.theta0_floor,
        A0_mode=args.A0_mode,
        A0_factor=args.A0_factor,
        eig_sort=args.eig_sort,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
