"""Run the bipartite GapPTR simulation while varying total epsilon.

This script does not require pre-generated networks.  Each Monte Carlo
replication generates a fresh bipartite DCSBM network internally through
``Utility.bipartite_gaptr_utils``.

Recommended layout
------------------

    ProjectRoot/
      Utility/
        bipartite_gaptr_utils.py
      BipartiteExperiment/
        run_bipartite_vary_eps.py

Run from ``ProjectRoot/BipartiteExperiment`` or from ``ProjectRoot``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if (PROJECT_ROOT / "Utility").is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Utility.bipartite_gaptr_utils import (
        add_common_args,
        default_eps_values,
        plot_summary,
        run_one_replication,
        save_csv,
        summarize_records,
    )
except ModuleNotFoundError:
    from bipartite_gaptr_utils import (  # type: ignore
        add_common_args,
        default_eps_values,
        plot_summary,
        run_one_replication,
        save_csv,
        summarize_records,
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build command-line parser for the varying-epsilon experiment."""

    parser = argparse.ArgumentParser(
        description=(
            "Bi-DCSBM simulation for bipartite GapPTR: vary total epsilon and "
            "compare NonPrivate against GapPTR under several eps1 splits."
        )
    )
    add_common_args(parser)
    parser.add_argument("--m", type=int, default=1600, help="Number of right-side nodes.")
    parser.add_argument(
        "--eps_total_values",
        type=float,
        nargs="+",
        default=default_eps_values(),
        help="Grid of total privacy budgets eps_total = eps_main + eps1.",
    )
    parser.add_argument(
        "--log_x",
        action="store_true",
        help="Use a logarithmic x-axis for epsilon in the output figure.",
    )
    parser.add_argument(
        "--no_ci",
        action="store_true",
        help="Do not draw 95% standard-error bands in the output figure.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments before launching a possibly long simulation."""

    if args.n <= 0 or args.m <= 0:
        raise ValueError("n and m must be positive.")
    if args.reps <= 0:
        raise ValueError("reps must be positive.")
    if any(eps <= 0 for eps in args.eps_total_values):
        raise ValueError("All eps_total_values must be positive.")
    if any(eps1 < 0 for eps1 in args.eps1_values):
        raise ValueError("All eps1_values must be nonnegative.")
    min_eps_total = min(args.eps_total_values)
    max_eps1 = max(args.eps1_values)
    if max_eps1 >= min_eps_total:
        raise ValueError(
            "The largest eps1 value must be smaller than every eps_total value. "
            f"Got max eps1={max_eps1} and min eps_total={min_eps_total}."
        )


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    total_jobs = len(args.eps_total_values) * args.reps
    job_id = 0

    for eps_total in args.eps_total_values:
        print(f"\n=== eps_total={eps_total:g} ===")
        for rep in range(args.reps):
            job_id += 1
            print(f"  replication {rep + 1}/{args.reps}  ({job_id}/{total_jobs})")
            all_records.extend(
                run_one_replication(
                    n=args.n,
                    m=args.m,
                    scenario=args.scenario,
                    eps_total=float(eps_total),
                    eps1_values=args.eps1_values,
                    delta=args.delta,
                    a0=args.a0,
                    rep=rep,
                    base_seed=args.seed_base,
                    K=args.K,
                    noise_mult=args.noise_mult,
                )
            )

    raw_df = pd.DataFrame(all_records)
    summary_df = summarize_records(raw_df, x_col="eps_total")

    raw_path = outdir / "vary_eps_raw.csv"
    summary_path = outdir / "vary_eps_summary.csv"
    figure_path = outdir / "vary_eps_clustering_error.png"

    save_csv(raw_path, raw_df)
    save_csv(summary_path, summary_df)

    plot_summary(
        summary=summary_df,
        x_col="eps_total",
        title=(
            f"Bi-DCSBM GapPTR: varying total epsilon "
            f"(scenario={args.scenario}, n={args.n}, m={args.m})"
        ),
        xlabel=r"Total privacy budget $\epsilon_{\mathrm{total}}$",
        outpath=figure_path,
        log_x=args.log_x,
        show_ci=not args.no_ci,
    )

    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
