"""Full-graph epsilon sweep comparing NetPTR and EdgeFlip.

This version uses the full graph rather than repeated induced subgraphs.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from netptr_realdata_utils import (
    _cluster_from_embedding,
    _norm_2_infty,
    _stable_sigmoid,
    alpha_netptr,
    compute_gamma_E,
    compute_Xi_and_lambdas_from_A,
    dmax_from_csr,
    edgeflip_specclu_from_A,
    load_blogcatalog_graph,
    theta0_tilde_from_dmaxA,
)

def sweep_eps_ari_NetPTR_vs_edgeflip(
    base_dir=None,
    zip_path=None,
    K: int = 2,
    a0: float = 0.01,
    eps_list=(1.0, 1.5, 2.0, 2.5, 3.0),
    rep: int = 0,
    delta: float = 0.01,

    # NetPTR params
    A0: float = 50.0,
    A0_mode: str = "from_2toinfty",
    A0_factor: float = 1.05,
    eig_sort: str = "abs",
    eig_tol: float = 1e-3,
    eig_maxiter: int = 200000,
    cache_npz: str = "cache_flickr_A_eigs_K2_abs.npz",

    theta0_input: float = 0.3,
    theta0_mode: str = "use_tilde",      # fixed | use_tilde | floor_by_tilde
    eps1_mode: str = "same_as_eps",      # fixed | same_as_eps
    eps1: Optional[float] = None,
    theta0_floor: float = 1e-12,

    # randomness / kmeans
    seed_base: int = 24680,
    kmeans_nstart: int = 25,
    norm_tol: float = 1e-12,

    # edgeflip guard
    edgeflip_max_edges_upper_priv: int = 30_000_000,

    verbose: bool = True,
):
    # --- load graph ---
    A, _ = load_blogcatalog_graph(base_dir=base_dir, zip_path=zip_path)
    n = int(A.shape[0])
    dmaxA = float(dmax_from_csr(A))

    # --- compute/load Xi_hat ---
    Xi_hat = lamK = lamKp1 = None

    if verbose:
        print("[eigs] computing Xi_hat + lambdaK/lambdaKp1 from A (no cache)...")

    Xi_hat, lamK, lamKp1 = compute_Xi_and_lambdas_from_A(
        A, K=K, eig_sort=eig_sort, tol=float(eig_tol), maxiter=int(eig_maxiter)
    )
    Xi_hat = np.asarray(Xi_hat, dtype=float)

    # --- choose A0_use (once) ---
    if A0_mode == "fixed":
        A0_use = float(A0)
    else:
        xi_2inf = _norm_2_infty(Xi_hat)
        A0_use = (1.0 + float(A0_factor)) * np.sqrt(float(n)) * float(xi_2inf)

    # --- FIX randomness across eps (so eps effect is clearer) ---
    rng_shared = np.random.default_rng(int(seed_base + 100000 * int(rep)))
    u_accept = float(rng_shared.random())
    X_rand = rng_shared.random(size=(n, K))
    seed_km = int(rng_shared.integers(1, 1_000_000))
    zeta = np.random.default_rng(7777 + int(rep)).standard_normal(size=(n, K))

    # baseline non-private SC labels (fixed)
    labels_sc = _cluster_from_embedding(Xi_hat, K, nstart=kmeans_nstart, norm_tol=norm_tol, seed=seed_km)

    rows = []
    for eps in eps_list:
        eps = float(eps)

        # --- theta0 (possibly depends on eps) ---
        theta0_used = float(theta0_input)
        eps1_used = np.nan
        theta0_tilde = np.nan
        lap_theta0 = np.nan

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

        # --- NetPTR for this eps (single a0) ---
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

        M = 1.0 + (2.0 / eps) * np.log(1.0 / float(delta))
        betaA = gamma_E

        if betaA > 2.0 * M:
            pA = 1.0
        else:
            x = 0.5 * eps * (betaA - M)
            pA = float(_stable_sigmoid(x))

        accept = (u_accept < pA)
        alpha = float(alpha_netptr(n=n, K=K, a0=float(a0), A0=float(A0_use)))
        noise_scale = (alpha / eps) * np.sqrt(2.0 * np.log(1.25 / float(delta)))

        Xi_tilde = (Xi_hat + noise_scale * zeta) if accept else X_rand
        labels_gap = _cluster_from_embedding(Xi_tilde, K, nstart=kmeans_nstart, norm_tol=norm_tol, seed=seed_km)

        ari_gap = float(adjusted_rand_score(labels_sc, labels_gap))

        # --- edge-flip spectral clustering for this eps ---
        seed_flip = int(seed_base + 900000 + 1000 * int(rep) + int(round(eps * 1000)))
        ef = edgeflip_specclu_from_A(
            A=A, K=K, eps=eps,
            seed_flip=seed_flip,
            seed_km=seed_km,
            nstart=kmeans_nstart,
            norm_tol=norm_tol,
            max_edges_upper_priv=int(edgeflip_max_edges_upper_priv),
        )
        if ef["labels"] is None:
            ari_edge = np.nan
        else:
            ari_edge = float(adjusted_rand_score(labels_sc, ef["labels"]))

        if verbose:
            print(
                f"[eps={eps:.2f}] NetPTR: accept={accept}, pA={pA:.4g}, ARI={ari_gap:.4f} | "
                f"edgeflip: ARI={ari_edge if np.isfinite(ari_edge) else 'NA'} "
                f"(q={ef['q_flip']:.4g}, m_priv={ef['m_upper_priv']})"
            )

        rows.append({
            "rep": int(rep),
            "n": int(n),
            "K": int(K),
            "a0": float(a0),
            "eps": float(eps),
            "delta": float(delta),

            "ARI_NetPTR_vs_SC": float(ari_gap),
            "ARI_edgeflip_vs_SC": float(ari_edge) if np.isfinite(ari_edge) else np.nan,

            # optional diagnostics
            "accept": bool(accept),
            "pA": float(pA),
            "gamma_E": float(gamma_E),
            "noise_scale": float(noise_scale),
            "theta0_used": float(theta0_used),
            "theta0_tilde": float(theta0_tilde) if np.isfinite(theta0_tilde) else np.nan,

            "q_flip": float(ef["q_flip"]),
            "m_upper_priv": ef["m_upper_priv"],
            "add_upper": int(ef["add_upper"]),
        })

    out = pd.DataFrame(rows).sort_values("eps").reset_index(drop=True)

    # --- plot eps vs ARI ---
    plt.figure()
    plt.plot(out["eps"], out["ARI_NetPTR_vs_SC"], marker="o", label="NetPTR")
    plt.plot(out["eps"], out["ARI_edgeflip_vs_SC"], marker="o", label="EdgeFlip")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("Adjusted Rand Index (ARI)")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eps_vs_ari.png", dpi=200)
    plt.close()

    out.to_csv("eps_vs_ari.csv", index=False)
    if verbose:
        print("\nSaved: eps_vs_ari.csv and eps_vs_ari.png")

    return out



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a full-graph NetPTR vs EdgeFlip epsilon sweep.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base_dir", type=str)
    src.add_argument("--zip_path", type=str)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--a0", type=float, default=0.12)
    p.add_argument("--eps_list", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.01)
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
    p.add_argument("--edgeflip_max_edges_upper_priv", type=int, default=30_000_000)
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out = sweep_eps_ari_netptr_vs_edgeflip(
        base_dir=args.base_dir,
        zip_path=args.zip_path,
        K=args.K,
        a0=args.a0,
        eps_list=tuple(args.eps_list),
        rep=args.rep,
        delta=args.delta,
        A0=args.A0,
        A0_mode=args.A0_mode,
        A0_factor=args.A0_factor,
        eig_sort=args.eig_sort,
        eig_tol=args.eig_tol,
        eig_maxiter=args.eig_maxiter,
        theta0_input=args.theta0_input,
        theta0_mode=args.theta0_mode,
        eps1_mode=args.eps1_mode,
        eps1=args.eps1,
        theta0_floor=args.theta0_floor,
        seed_base=args.seed_base,
        kmeans_nstart=args.kmeans_nstart,
        norm_tol=args.norm_tol,
        edgeflip_max_edges_upper_priv=args.edgeflip_max_edges_upper_priv,
        verbose=not args.quiet,
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
