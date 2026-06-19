"""Repeated-subgraph real-data experiment for NetPTR vs EdgeFlip.

This is the main real-data script corresponding to the monolithic version.
"""

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from netptr_realdata_utils import (
    _cluster_from_embedding,
    _compute_netptr_labels_on_A,
    _induced_subgraph,
    _sample_subgraph_nodes,
    compute_Xi_and_lambdas_from_A,
    edgeflip_specclu_from_A,
    load_blogcatalog_graph,
)

def sweep_eps_subgraphs_NetPTR_vs_edgeflip_multi_eps1(
    base_dir=None,
    zip_path=None,
    K: int = 2,
    a0: float = 0.11,
    eps_list=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    eps1_list=(0.1, 0.2, 0.5),

    # subgraph repetition design
    n_reps: int = 20,
    subgraph_n: Optional[int] = None,
    subgraph_frac: Optional[float] = 0.5,

    delta: Optional[float] = 0.01,

    # NetPTR params
    A0: float = 50.0,
    A0_mode: str = "from_2toinfty",
    A0_factor: float = 1.05,
    eig_sort: str = "abs",
    eig_tol: float = 1e-3,
    eig_maxiter: int = 200000,

    theta0_input: float = 0.3,
    theta0_mode: str = "use_tilde",    # fixed | use_tilde | floor_by_tilde
    theta0_floor: float = 1e-12,

    # kmeans / randomness
    seed_base: int = 24680,
    kmeans_nstart: int = 25,
    norm_tol: float = 1e-12,

    # edgeflip guard
    edgeflip_max_edges_upper_priv: int = 30_000_000,
    edgeflip_eps_offset: float = 0.95,

    # output
    out_prefix: str = "subgraph_eps_scan",
    verbose: bool = True,
):
    """
    Real-data analysis by repeated induced subgraphs.

    For each repetition:
      1) sample an induced subgraph
      2) use SC on that subgraph as non-private baseline
      3) compare EdgeFlip and NetPTR against that baseline

    Outputs:
      raw_df      : long-format results per rep / eps / method / eps1
      summary_df  : aggregated mean/sd/se over reps
    """
    if (base_dir is None) == (zip_path is None):
        raise ValueError("Provide exactly one of base_dir or zip_path.")

    A_full, _ = load_blogcatalog_graph(base_dir=base_dir, zip_path=zip_path)
    n_full = int(A_full.shape[0])

    if subgraph_n is None:
        if subgraph_frac is None:
            raise ValueError("Provide either subgraph_n or subgraph_frac.")
        subgraph_n = int(round(float(subgraph_frac) * n_full))

    subgraph_n = max(subgraph_n, K + 2)
    if subgraph_n > n_full:
        raise ValueError(f"subgraph_n={subgraph_n} exceeds full graph size {n_full}.")

    if delta is None:
        delta_global = np.nan
    else:
        delta_global = float(delta)

    rows = []
    skipped = 0

    for rep in range(int(n_reps)):
        rng_sub = np.random.default_rng(seed_base + 10000 * rep + 17)
        nodes = _sample_subgraph_nodes(n_full, subgraph_n, rng_sub)
        A_sub = _induced_subgraph(A_full, nodes)

        n_sub = int(A_sub.shape[0])
        nnz_sub = int(A_sub.nnz)

        if verbose:
            print(f"\n[rep {rep:02d}] subgraph_n={n_sub}, nnz={nnz_sub}")

        # skip pathological tiny/empty subgraphs
        if n_sub <= K + 1 or nnz_sub == 0:
            skipped += 1
            if verbose:
                print(f"  skipped rep {rep}: subgraph too small / empty")
            continue

        delta_use = (1.0 / float(n_sub)) if delta is None else float(delta_global)

        # -------------------------
        # EdgeFlip: one line per eps
        # -------------------------
        try:
            Xi_hat_sub, _, _ = compute_Xi_and_lambdas_from_A(
                A_sub, K=K, eig_sort=eig_sort, tol=float(eig_tol), maxiter=int(eig_maxiter)
            )
            seed_km_base = int(np.random.default_rng(seed_base + 200000 * rep).integers(1, 1_000_000))
            labels_sc_sub = _cluster_from_embedding(
                Xi_hat_sub, K, nstart=kmeans_nstart, norm_tol=norm_tol, seed=seed_km_base
            )
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"  skipped rep {rep}: SC baseline failed with error: {e}")
            continue


        # -------------------------
        # NetPTR: multiple eps1 lines
        # -------------------------
        for ie, eps in enumerate(eps_list):
            eps = float(eps)
            for j, eps1 in enumerate(eps1_list):
                try:
                    res_gap = _compute_netptr_labels_on_A(
                        A_sub=A_sub,
                        K=K,
                        eps=float(eps),
                        delta=float(delta_use),
                        a0=float(a0),
                        A0=float(A0),
                        A0_mode=A0_mode,
                        A0_factor=float(A0_factor),
                        theta0_input=float(theta0_input),
                        theta0_mode=theta0_mode,
                        eps1=float(eps1),
                        theta0_floor=float(theta0_floor),
                        eig_sort=eig_sort,
                        eig_tol=float(eig_tol),
                        eig_maxiter=int(eig_maxiter),
                        kmeans_nstart=int(kmeans_nstart),
                        norm_tol=float(norm_tol),
                        seed_base_local=int(seed_base + 333333),
                        rep=int(rep),
                        eps_tag=int(round(1000 * eps)),
                        eps1_tag=int(round(1000 * float(eps1))),
                        verbose=verbose,
                    )

                    rows.append({
                        "rep": int(rep),
                        "subgraph_n": int(n_sub),
                        "eps": float(eps),
                        "eps1": float(eps1),
                        "method": "NetPTR",
                        "ARI_vs_SC": float(res_gap["ARI_NetPTR_vs_SC"]),
                        "delta": float(delta_use),
                        "nnz": int(res_gap["nnz"]),
                        "q_flip": np.nan,
                        "m_upper_priv": np.nan,
                        "add_upper": np.nan,
                        "accept": bool(res_gap["accept"]),
                        "pA": float(res_gap["pA"]),
                        "gamma_E": float(res_gap["gamma_E"]),
                        "noise_scale": float(res_gap["noise_scale"]),
                        "theta0_used": float(res_gap["theta0_used"]),
                        "theta0_tilde": float(res_gap["theta0_tilde"]) if np.isfinite(res_gap["theta0_tilde"]) else np.nan,
                        "lap_theta0": float(res_gap["lap_theta0"]) if np.isfinite(res_gap["lap_theta0"]) else np.nan,
                        "A0_use": float(res_gap["A0_use"]),
                        "good_set": bool(res_gap["good_set"]),
                    })
                except Exception as e:
                    if verbose:
                        print(f"  [NetPTR eps={eps:.3g}, eps1={eps1}] failed: {e}")

        for ie, eps in enumerate(eps_list):
            eps = float(eps)
            try:
                seed_flip = int(seed_base + 900000 + 1000 * rep + int(round(1000 * eps)))
                eps_edgeflip = eps - float(edgeflip_eps_offset)
                if eps_edgeflip <= 0:
                    raise ValueError(f"EdgeFlip effective eps must be positive; got eps - edgeflip_eps_offset = {eps_edgeflip:g}")

                ef = edgeflip_specclu_from_A(
                    A=A_sub,
                    K=K,
                    eps=eps_edgeflip,
                    seed_flip=seed_flip,
                    seed_km=seed_km_base,
                    nstart=kmeans_nstart,
                    norm_tol=norm_tol,
                    max_edges_upper_priv=int(edgeflip_max_edges_upper_priv),
                )
                ari_edge = float(adjusted_rand_score(labels_sc_sub, ef["labels"]))

                if verbose:
                    print(
                        f"  [EdgeFlip eps={eps:.3g}] ARI={ari_edge:.4f}, "
                        f"q={ef['q_flip']:.4g}, m_priv={ef['m_upper_priv']}"
                    )

                rows.append({
                    "rep": int(rep),
                    "subgraph_n": int(n_sub),
                    "eps": float(eps),
                    "eps1": np.nan,
                    "method": "EdgeFlip",
                    "ARI_vs_SC": float(ari_edge),
                    "delta": float(delta_use),
                    "nnz": int(nnz_sub),
                    "q_flip": float(ef["q_flip"]),
                    "m_upper_priv": int(ef["m_upper_priv"]),
                    "add_upper": int(ef["add_upper"]),
                    "accept": np.nan,
                    "pA": np.nan,
                    "gamma_E": np.nan,
                    "noise_scale": np.nan,
                    "theta0_used": np.nan,
                    "theta0_tilde": np.nan,
                    "lap_theta0": np.nan,
                    "A0_use": np.nan,
                    "good_set": np.nan,
                })
            except Exception as e:
                if verbose:
                    print(f"  [EdgeFlip eps={eps:.3g}] failed: {e}")

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise RuntimeError("No successful runs. Try larger subgraph_n or fewer K/easier eps settings.")

    summary_df = (
        raw_df.groupby(["method", "eps", "eps1"], dropna=False)
        .agg(
            mean_ARI=("ARI_vs_SC", "mean"),
            sd_ARI=("ARI_vs_SC", "std"),
            n_runs=("ARI_vs_SC", "size"),
            mean_subgraph_n=("subgraph_n", "mean"),
            mean_nnz=("nnz", "mean"),
        )
        .reset_index()
        .sort_values(["method", "eps1", "eps"])
        .reset_index(drop=True)
    )
    summary_df["sd_ARI"] = summary_df["sd_ARI"].fillna(0.0)
    summary_df["se_ARI"] = summary_df["sd_ARI"] / np.sqrt(summary_df["n_runs"])

    raw_csv = f"{out_prefix}_raw.csv"
    summary_csv = f"{out_prefix}_summary.csv"
    fig_path = f"{out_prefix}_meanARI.png"

    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    # plot mean ARI curves; put NetPTR first in the legend
    plt.figure()

    # NetPTR, one line per eps1
    sub_gap_all = summary_df[summary_df["method"] == "NetPTR"].copy()
    eps1_vals = sorted([x for x in sub_gap_all["eps1"].dropna().unique()])

    for eps1 in eps1_vals:
        sub_gap = sub_gap_all[sub_gap_all["eps1"] == eps1].sort_values("eps")
        plt.plot(
            sub_gap["eps"].values,
            sub_gap["mean_ARI"].values,
            marker="o",
            label=fr"NetPTR ($\varepsilon_1={eps1:g}$)",
        )

    # EdgeFlip
    sub_edge = summary_df[summary_df["method"] == "EdgeFlip"].sort_values("eps")
    if len(sub_edge) > 0:
        plt.plot(
            sub_edge["eps"].values,
            sub_edge["mean_ARI"].values,
            marker="s",
            linestyle="--",
            label="EdgeFlip",
        )

    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("Mean ARI vs non-private SC")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    if verbose:
        print(f"\nSaved raw results   : {raw_csv}")
        print(f"Saved summary       : {summary_csv}")
        print(f"Saved figure        : {fig_path}")
        print(f"Skipped repetitions : {skipped}")

    return raw_df, summary_df



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run repeated-subgraph NetPTR vs EdgeFlip real-data experiment.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base_dir", type=str, help="Folder containing nodes.csv and edges.csv.")
    src.add_argument("--zip_path", type=str, help="Zip containing nodes.csv and edges.csv.")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--a0", type=float, default=0.12)
    p.add_argument("--eps_list", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--eps1_list", type=float, nargs="+", default=[0.2, 0.5])
    p.add_argument("--n_reps", type=int, default=10)
    p.add_argument("--subgraph_n", type=int, default=None)
    p.add_argument("--subgraph_frac", type=float, default=0.9)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--A0", type=float, default=50.0)
    p.add_argument("--A0_mode", type=str, default="from_2toinfty", choices=["fixed", "from_2toinfty"])
    p.add_argument("--A0_factor", type=float, default=1.05)
    p.add_argument("--eig_sort", type=str, default="abs", choices=["abs", "raw_desc"])
    p.add_argument("--eig_tol", type=float, default=1e-3)
    p.add_argument("--eig_maxiter", type=int, default=200000)
    p.add_argument("--theta0_input", type=float, default=0.3)
    p.add_argument("--theta0_mode", type=str, default="use_tilde", choices=["fixed", "use_tilde", "floor_by_tilde"])
    p.add_argument("--theta0_floor", type=float, default=1e-12)
    p.add_argument("--seed_base", type=int, default=24680)
    p.add_argument("--kmeans_nstart", type=int, default=25)
    p.add_argument("--norm_tol", type=float, default=1e-12)
    p.add_argument("--edgeflip_max_edges_upper_priv", type=int, default=30_000_000)
    p.add_argument("--edgeflip_eps_offset", type=float, default=0.95,
                   help="Run EdgeFlip with eps - offset. Default 0.95 reproduces the uploaded script; use 0 for the same epsilon.")
    p.add_argument("--out_prefix", type=str, default="flickr_subgraph_netptr_edgeflip")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    raw_df, summary_df = sweep_eps_subgraphs_netptr_vs_edgeflip_multi_eps1(
        base_dir=args.base_dir,
        zip_path=args.zip_path,
        K=args.K,
        a0=args.a0,
        eps_list=tuple(args.eps_list),
        eps1_list=tuple(args.eps1_list),
        n_reps=args.n_reps,
        subgraph_n=args.subgraph_n,
        subgraph_frac=args.subgraph_frac,
        delta=args.delta,
        A0=args.A0,
        A0_mode=args.A0_mode,
        A0_factor=args.A0_factor,
        eig_sort=args.eig_sort,
        eig_tol=args.eig_tol,
        eig_maxiter=args.eig_maxiter,
        theta0_input=args.theta0_input,
        theta0_mode=args.theta0_mode,
        theta0_floor=args.theta0_floor,
        seed_base=args.seed_base,
        kmeans_nstart=args.kmeans_nstart,
        norm_tol=args.norm_tol,
        edgeflip_max_edges_upper_priv=args.edgeflip_max_edges_upper_priv,
        edgeflip_eps_offset=args.edgeflip_eps_offset,
        out_prefix=args.out_prefix,
        verbose=not args.quiet,
    )

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
