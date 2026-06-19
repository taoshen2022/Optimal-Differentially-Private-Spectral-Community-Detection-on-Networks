"""Scan NetPTR stability over a grid of a0 values on one real network.

Example:
    python exp_a0_scan_netptr.py --base_dir ./flickr-dataset/pruned_mindeg100_single \
        --eps 3.0 --a0_grid 0.08 0.10 0.12
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from netptr_realdata_utils import (
    _cluster_from_embedding,
    _norm_2_infty,
    _stable_sigmoid,
    alpha_netptr,
    clustering_error,
    compute_gamma_E,
    compute_Xi_and_lambdas_from_A,
    dmax_from_csr,
    load_blogcatalog_graph,
    theta0_tilde_from_dmaxA,
)

def test_a0_one_round_NetPTR_blogcatalog_updated(
    base_dir=None,
    zip_path=None,
    K: int = 2,
    eps: float = 1.0,
    rep: int = 0,

    # A0 handling (like your runner)
    A0: float = 50.0,
    A0_mode: str = "fixed",            # "fixed" | "from_2toinfty"
    A0_factor: float = 1.05,           # used if A0_mode="from_2toinfty"

    # scan grid
    a0_grid=(0.01, 0.02, 0.03),

    # eig options
    eig_sort: str = "abs",
    eig_tol: float = 1e-3,
    eig_maxiter: int = 200000,

    # kmeans options
    kmeans_nstart: int = 25,
    norm_tol: float = 1e-12,

    # DP params
    delta: Optional[float] = None,     # if None: default 1/n (or set explicitly e.g. 0.01)
    cache_npz: str = "cache_realdata_A_eigs.npz",

    # theta0 options (like your runner)
    theta0_input: float = 1.0,
    theta0_mode: str = "fixed",        # "fixed" | "use_tilde" | "floor_by_tilde"
    eps1_mode: str = "fixed",          # "fixed" | "same_as_eps"
    eps1: Optional[float] = None,      # required if eps1_mode="fixed" and theta0_mode!="fixed"
    theta0_floor: float = 1e-12,

    # randomness control
    seed_base: int = 24680,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Outputs:
      DataFrame with columns including gamma_E, pA, accept, noise_scale, mismatch(SC, NetPTR),
      plus diagnostics t_deg, t1, t2, t3, good_set flags, and theta0_tilde details (if enabled).
    """
    # --- checks ---
    if (base_dir is None) == (zip_path is None):
        raise ValueError("Provide exactly one of base_dir or zip_path.")
    A0_mode = str(A0_mode)
    if A0_mode not in ("fixed", "from_2toinfty"):
        raise ValueError("A0_mode must be 'fixed' or 'from_2toinfty'.")
    theta0_mode = str(theta0_mode)
    if theta0_mode not in ("fixed", "use_tilde", "floor_by_tilde"):
        raise ValueError("theta0_mode must be 'fixed', 'use_tilde', or 'floor_by_tilde'.")
    eps1_mode = str(eps1_mode)
    if eps1_mode not in ("fixed", "same_as_eps"):
        raise ValueError("eps1_mode must be 'fixed' or 'same_as_eps'.")

    # --- load graph ---
    A, _ = load_blogcatalog_graph(base_dir=base_dir, zip_path=zip_path)
    n = int(A.shape[0])
    K = int(K)
    if not (1 <= K <= n - 2):
        raise ValueError(f"Require 1 <= K <= n-2, got K={K}, n={n}")

    dmaxA = float(dmax_from_csr(A))

    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be > 0")

    if delta is None:
        delta_use = 1.0 / float(n)
    else:
        delta_use = float(delta)
    if not (0.0 < delta_use < 1.0):
        raise ValueError("delta must be in (0,1)")

    if theta0_mode != "fixed" and eps1_mode == "fixed":
        if eps1 is None or float(eps1) <= 0:
            raise ValueError("eps1 must be provided and > 0 when eps1_mode='fixed' and theta0_mode!='fixed'.")

    if verbose:
        print(f"[data] n={n}, nnz={A.nnz}, dmaxA={dmaxA:g}, K={K}, eps={eps:g}, delta={delta_use:g}, eig_sort={eig_sort}")

    # --- compute/load eigen-embedding ONCE ---
    recompute = True
    Xi_hat = lamK = lamKp1 = None

    if os.path.exists(cache_npz):
        try:
            obj = np.load(cache_npz, allow_pickle=False)
            Xi_hat_c = obj["Xi_hat"]
            lamK_c = float(obj["lamK"])
            lamKp1_c = float(obj["lamKp1"])
            cached_K = int(obj["K"])
            cached_sort = str(obj["eig_sort"])
            cached_n = int(obj["n"]) if "n" in obj.files else Xi_hat_c.shape[0]
            if cached_K == K and cached_sort == eig_sort and cached_n == n:
                Xi_hat = Xi_hat_c
                lamK = lamK_c
                lamKp1 = lamKp1_c
                recompute = False
                if verbose:
                    print(f"[cache] loaded Xi_hat/lamK/lamKp1 from {cache_npz}")
        except Exception:
            recompute = True

    if recompute:
        if verbose:
            print("[eigs] computing Xi_hat + lambdaK/lambdaKp1 from A (once)...")
        Xi_hat, lamK, lamKp1 = compute_Xi_and_lambdas_from_A(
            A, K=K, eig_sort=eig_sort, tol=float(eig_tol), maxiter=int(eig_maxiter)
        )
        np.savez(cache_npz, Xi_hat=Xi_hat, lamK=lamK, lamKp1=lamKp1, K=K, eig_sort=eig_sort, n=n, dmaxA=dmaxA)
        if verbose:
            print(f"[cache] saved to {cache_npz}")

    Xi_hat = np.asarray(Xi_hat, dtype=float)

    # --- choose A0_use ---
    if A0_mode == "fixed":
        A0_use = float(A0)
    else:
        xi_2inf = _norm_2_infty(Xi_hat)
        A0_use = (1.0 + float(A0_factor)) * np.sqrt(float(n)) * float(xi_2inf)

    # --- theta0 choice (tilde) ---
    eps1_used = np.nan
    theta0_tilde = np.nan
    lap_theta0 = np.nan
    theta0_used = float(theta0_input)

    if theta0_mode != "fixed":
        if eps1_mode == "same_as_eps":
            eps1_used = float(eps)
            seed_th = int(seed_base + 654321 + 1000 * int(rep) + int(round(eps * 1000)))
        else:
            eps1_used = float(eps1)
            seed_th = int(seed_base + 123456 + int(rep))

        rng_th = np.random.default_rng(seed_th)
        th_info = theta0_tilde_from_dmaxA(
            dmaxA=dmaxA, n=n, eps1=float(eps1_used), rng=rng_th, floor=float(theta0_floor)
        )
        theta0_tilde = float(th_info["theta0_tilde"])
        lap_theta0 = float(th_info["lap"])

        if theta0_mode == "use_tilde":
            theta0_used = theta0_tilde
        elif theta0_mode == "floor_by_tilde":
            theta0_used = max(float(theta0_input), theta0_tilde)

    if verbose:
        if theta0_mode == "fixed":
            print(f"[theta0] mode=fixed, theta0_used={theta0_used:g}")
        else:
            print(
                f"[theta0] mode={theta0_mode}, theta0_input={float(theta0_input):g}, "
                f"theta0_tilde={theta0_tilde:g}, theta0_used={theta0_used:g}, eps1_used={eps1_used:g}, lap={lap_theta0:g}"
            )
        print(f"[A0] mode={A0_mode}, A0_use={A0_use:g}")

    # --- fixed randomness for this (rep, eps): fair comparison across a0 ---
    seed = int(seed_base + 100000 * int(rep) + int(round(eps * 1000)))
    rng = np.random.default_rng(seed)
    u_accept = float(rng.random())
    X_rand = rng.random(size=(n, K))
    seed_km = int(rng.integers(1, 1_000_000))
    zeta = np.random.default_rng(7777 + int(rep)).standard_normal(size=(n, K))  # fixed per rep

    # baseline SC labels on Xi_hat (NON-private)
    labels_sc = _cluster_from_embedding(Xi_hat, K, nstart=kmeans_nstart, norm_tol=norm_tol, seed=seed_km)

    # constants for this eps
    M = 1.0 + (2.0 / eps) * np.log(1.0 / delta_use)

    if verbose:
        print(f"\n[fixed for scan] u_accept={u_accept:.6f}, M={M:.6g}, seed_km={seed_km}\n")

    rows = []
    for a0 in a0_grid:
        a0 = float(a0)
        if a0 <= 0:
            raise ValueError("a0_grid must contain only positive values.")

        g = compute_gamma_E(
            Xi_hat=Xi_hat,
            dmaxA=dmaxA,
            lambdaK=float(lamK),
            lambdaKp1=float(lamKp1),
            a0=a0,
            A0=float(A0_use),
            theta0=float(theta0_used),
        )
        gamma_E = float(g["gamma_E"])
        alpha = float(alpha_netptr(n=n, K=K, a0=a0, A0=float(A0_use)))

        betaA = gamma_E
        if betaA > 2.0 * M:
            pA = 1.0
        else:
            x = 0.5 * eps * (betaA - M)
            pA = float(_stable_sigmoid(x))

        accept = (u_accept < pA)

        noise_scale = (alpha / eps) * np.sqrt(2.0 * np.log(1.25 / delta_use))
        Xi_tilde = (Xi_hat + noise_scale * zeta) if accept else X_rand
        labels_gap = _cluster_from_embedding(Xi_tilde, K, nstart=kmeans_nstart, norm_tol=norm_tol, seed=seed_km)

        mismatch = float(clustering_error(labels_sc, labels_gap))

        if verbose:
            good_str = "GOOD" if g["good_set"] else "BAD"
            print(
                f"[a0={a0:g}] gamma_E={gamma_E:.6g}, pA={pA:.6g}, accept={accept}, "
                f"mismatch(SC,NetPTR)={mismatch:.4f}, A_good={good_str}, "
                f"t_deg={g['t_deg']:.3g}, t1={g['t1']:.3g}, t2={g['t2']:.3g}, t3={g['t3']:.3g}"
            )

        rows.append({
            "rep": int(rep),
            "n": int(n),
            "K": int(K),
            "eps": float(eps),
            "delta": float(delta_use),
            "eig_sort": str(eig_sort),

            # network diagnostics
            "nnz": int(A.nnz),
            "dmaxA": float(dmaxA),

            # theta0 diagnostics
            "theta0_mode": str(theta0_mode),
            "eps1_mode": str(eps1_mode),
            "eps1_used": float(eps1_used) if np.isfinite(eps1_used) else np.nan,
            "theta0_input": float(theta0_input),
            "theta0_tilde": float(theta0_tilde) if np.isfinite(theta0_tilde) else np.nan,
            "lap_theta0": float(lap_theta0) if np.isfinite(lap_theta0) else np.nan,
            "theta0_used": float(theta0_used),

            # A0 diagnostics
            "A0_mode": str(A0_mode),
            "A0_factor": float(A0_factor),
            "A0_use": float(A0_use),

            # scan parameter
            "a0": float(a0),

            # acceptance/noise
            "M": float(M),
            "gamma_E": float(gamma_E),
            "pA": float(pA),
            "accept": bool(accept),
            "alpha": float(alpha),
            "noise_scale": float(noise_scale),

            # mismatch vs baseline spectral clustering
            "mismatch": float(mismatch),

            # gamma_E internals
            "xi_2inf": float(g["xi_2inf"]),
            "U0": float(g["U0"]),
            "t_deg": float(g["t_deg"]),
            "t1": float(g["t1"]),
            "t2": float(g["t2"]),
            "t3": float(g["t3"]),

            # good-set flags
            "good_set": bool(g["good_set"]),
            "cond_deg": bool(g["cond_deg"]),
            "cond_lamK": bool(g["cond_lamK"]),
            "cond_lamKp1": bool(g["cond_lamKp1"]),
            "cond_xi": bool(g["cond_xi"]),
        })

    out = pd.DataFrame(rows).sort_values(["pA", "a0"], ascending=[False, True]).reset_index(drop=True)
    return out



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan NetPTR a0 values on a real network.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base_dir", type=str, help="Folder containing nodes.csv and edges.csv.")
    src.add_argument("--zip_path", type=str, help="Zip containing nodes.csv and edges.csv.")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--eps", type=float, default=3.0)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--a0_grid", type=float, nargs="+", default=[0.08, 0.10, 0.12])
    p.add_argument("--A0", type=float, default=50.0)
    p.add_argument("--A0_mode", type=str, default="from_2toinfty", choices=["fixed", "from_2toinfty"])
    p.add_argument("--A0_factor", type=float, default=1.05)
    p.add_argument("--eig_sort", type=str, default="abs", choices=["abs", "raw_desc"])
    p.add_argument("--eig_tol", type=float, default=1e-3)
    p.add_argument("--eig_maxiter", type=int, default=200000)
    p.add_argument("--theta0_input", type=float, default=0.3)
    p.add_argument("--theta0_mode", type=str, default="use_tilde", choices=["fixed", "use_tilde", "floor_by_tilde"])
    p.add_argument("--eps1_mode", type=str, default="same_as_eps", choices=["fixed", "same_as_eps"])
    p.add_argument("--eps1", type=float, default=None)
    p.add_argument("--theta0_floor", type=float, default=1e-12)
    p.add_argument("--seed_base", type=int, default=24680)
    p.add_argument("--kmeans_nstart", type=int, default=25)
    p.add_argument("--norm_tol", type=float, default=1e-12)
    p.add_argument("--cache_npz", type=str, default="cache_realdata_A_eigs.npz")
    p.add_argument("--out_csv", type=str, default="netptr_a0_scan.csv")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    df = test_a0_one_round_netptr_blogcatalog(
        base_dir=args.base_dir,
        zip_path=args.zip_path,
        K=args.K,
        eps=args.eps,
        rep=args.rep,
        A0=args.A0,
        A0_mode=args.A0_mode,
        A0_factor=args.A0_factor,
        a0_grid=tuple(args.a0_grid),
        eig_sort=args.eig_sort,
        eig_tol=args.eig_tol,
        eig_maxiter=args.eig_maxiter,
        delta=args.delta,
        cache_npz=args.cache_npz,
        theta0_input=args.theta0_input,
        theta0_mode=args.theta0_mode,
        eps1_mode=args.eps1_mode,
        eps1=args.eps1,
        theta0_floor=args.theta0_floor,
        seed_base=args.seed_base,
        kmeans_nstart=args.kmeans_nstart,
        norm_tol=args.norm_tol,
        verbose=not args.quiet,
    )
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
