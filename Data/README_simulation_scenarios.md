# README: Generating DCSBM Simulation Scenarios

This README explains how to generate the synthetic network scenarios in the simulation experiments using "Simulation_data.py". The data-generating model is a degree-corrected stochastic block model (DCSBM). The same generator can be reused across Experiment 1, Experiment 2, and Experiment 3 by changing only the connectivity parameters, degree-parameter distribution, network size, number of replications, and output folder.

---

## 1. Dependencies

The generator requires:

```bash
pip install numpy scipy
```

The script uses:

```python
import os
import json
import numpy as np
import struct
from typing import Optional, Tuple
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
```

---

## 2. Main generator

The main function is:

```python
save_network(
    out_dir: str,
    n_rep: int,
    n: int = 1000,
    K: int = 2,
    balanced: bool = True,
    p_in: float = 0.4,
    p_out: float = 0.1,
    P_noise: float = 0.0,
    theta_dist: str = "uniform",
    theta_min: float = 0.1,
    theta_max: float = 0.5,
    theta_beta: Tuple[float, float] = (2.0, 5.0),
    min_comm_prop: float = 0.05,
    dirichlet_alpha: float = 1.0,
    max_tries: int = 100,
    eig_order_omega: str = "abs",
    block: int = 512,
    index_base_edges: int = 1,
    base_seed: Optional[int] = 123,
)
```

Important arguments:

| Argument | Meaning |
|---|---|
| `out_dir` | Output folder for this scenario |
| `n_rep` | Number of independent network replications |
| `n` | Number of nodes |
| `K` | Number of communities; use `K=2` in the experiments below |
| `balanced` | If `True`, assigns approximately equal community sizes; for even `n` and `K=2`, this gives exactly `n/2` per community |
| `p_in` | Baseline within-community connectivity |
| `p_out` | Baseline between-community connectivity |
| `theta_dist` | Degree-parameter distribution |
| `theta_min`, `theta_max` | Range of the main degree parameters |
| `theta_beta` | Shape parameters for the optional beta degree distribution |
| `block` | Block size used when sampling edges; useful for memory control |
| `index_base_edges` | Use `1` if downstream R code expects 1-based node labels |
| `base_seed` | Replication `r` uses seed `base_seed + r` |

---
## 3. Output files

For each replication, the generator creates a folder such as:

```text
out_dir/
  rep000/
  rep001/
  rep002/
  ...
```

Each replication folder contains:

| File | Description |
|---|---|
| `A_upper_bitpack.bin` | Compact bit-packed upper triangle of the sampled adjacency matrix |
| `omega_theta_z.csv` | Node ID, true community label, and degree parameter \(\theta_i\) |
| `P0.csv` | Original baseline block matrix before any probability rescaling |
| `P_eff.csv` | Effective block matrix after possible rescaling to keep probabilities valid |
| `omega_components.npz` | Compact NumPy archive containing `theta`, `z0`, `P0`, `P_eff`, and scaling metadata |
| `A_eigvals_topKp1.csv` | Top \(K+1\) eigenvalues of the sampled adjacency matrix, sorted by absolute value |
| `A_eigvecs_topK.csv` | Top \(K\) eigenvectors of the sampled adjacency matrix, with node labels and true labels |
| `Omega_eigvals_topK.csv` | Top \(K\) eigenvalues of the population matrix \(\Omega\) |
| `Omega_eigvecs_topK.csv` | Top \(K\) eigenvectors of \(\Omega\), with node labels and true labels |
| `meta.json` | Metadata for the replication, including seed, network size, number of edges, maximum degree, and scenario parameters |

The temporary edge list `A_edges.csv` is written during generation but removed after `A_upper_bitpack.bin` is created.

---

## 4. Experiment 1

Networks are generated under two scenarios.

### Scenario 1: relatively sparse baseline with regular heterogeneous degrees

Use:

\[
(p_{\mathrm{in}},p_{\mathrm{out}})=(0.4,0.1),
\qquad
\theta_i\sim \mathrm{Unif}(0.1,0.5).
\]

Example generation:

```python
from generate_dcsbm_network import save_network

save_network(
    out_dir="./outputs/exp1_scenario1",
    n_rep=50,
    n=20000,
    K=2,
    balanced=True,
    p_in=0.4,
    p_out=0.1,
    theta_dist="uniform",
    theta_min=0.1,
    theta_max=0.5,
    base_seed=123,
    index_base_edges=1,
)
```

### Scenario 2: denser baseline with low-degree nodes

Use:

\[
(p_{\mathrm{in}},p_{\mathrm{out}})=(0.9,0.3).
\]

The degree parameters follow the mixture distribution:

\[
\theta_i\sim \mathrm{Unif}(0.1,0.5)
\quad \text{with probability }0.6,
\]

and

\[
\theta_i\sim \mathrm{Unif}(0.01,0.05)
\quad \text{with probability }0.4.
\]

Example generation:

```python
from generate_dcsbm_network import save_network

save_network(
    out_dir="./outputs/exp1_scenario2",
    n_rep=50,
    n=20000,
    K=2,
    balanced=True,
    p_in=0.9,
    p_out=0.3,
    theta_dist="uniform_mixture_low",
    theta_min=0.1,
    theta_max=0.5,
    base_seed=223,
    index_base_edges=1,
)
```

---

## 5. Experiment 2


The network-generation settings are the same as in Experiment 1, but the network size is fixed at
\(
n=20000.
\)
The overall privacy budget is
\(
\varepsilon_{\mathrm{all}}=\varepsilon+\varepsilon_1.
\)

The budget grid is:

\[
\log(\varepsilon_{\mathrm{all}})
\in
\{-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1\}.
\]

In Python:

```python
import numpy as np

eps_all_grid = np.exp(np.arange(-1, 1.0001, 0.25))
```

For each value of \(\varepsilon_{\mathrm{all}}\), run:

| Method | Privacy-budget setting |
|---|---|
| `GapPTR_oracle` | Known \(\theta_0\): \(\varepsilon=\varepsilon_{\mathrm{all}}\), \(\varepsilon_1=0\) |
| `GapPTR_tilde` | Estimated \(\theta_0\): \(\varepsilon=\varepsilon_{\mathrm{all}}-0.2\), \(\varepsilon_1=0.2\) |
| `EdgeFlip` | \(\varepsilon=\varepsilon_{\mathrm{all}}\) |
| `NonPrivate` | Non-private spectral clustering benchmark |

Generate the networks once and reuse the same replications across all privacy budgets and all methods.

### Experiment 2, Scenario 1

```python
from generate_dcsbm_network import save_network

save_network(
    out_dir="./outputs/exp2_scenario1_n20000",
    n_rep=50,
    n=20000,
    K=2,
    balanced=True,
    p_in=0.4,
    p_out=0.1,
    theta_dist="uniform",
    theta_min=0.1,
    theta_max=0.5,
    base_seed=123,
    index_base_edges=1,
)
```

### Experiment 2, Scenario 2

```python
from generate_dcsbm_network import save_network

save_network(
    out_dir="./outputs/exp2_scenario2_n20000",
    n_rep=50,
    n=20000,
    K=2,
    balanced=True,
    p_in=0.9,
    p_out=0.3,
    theta_dist="uniform_mixture_low",
    theta_min=0.1,
    theta_max=0.5,
    base_seed=223,
    index_base_edges=1,
)
```


---

## 6. Experiment 3

The network size and privacy budget are fixed as
\(
n=20000,
\varepsilon_{\mathrm{all}}=0.8.
\)
The degree scale is varied by changing \((\theta_{\min},\theta_{\max})\).

A suggested grid is:

```python
theta_grid = [
    (0.1, 0.5),
    (0.2, 0.6),
    (0.3, 0.7),
    (0.4, 0.8),
    (0.6, 0.9),
]
```

### Setting 1: regular heterogeneous degrees

Use:

\[
(p_{\mathrm{in}},p_{\mathrm{out}})=(0.4,0.1),
\qquad
\theta_i\sim \mathrm{Unif}(\theta_{\min},\theta_{\max}).
\]

Example generation loop:

```python
from generate_dcsbm_network import save_network

n_rep = 50
n = 20000

theta_grid = [
    (0.1, 0.5),
    (0.2, 0.6),
    (0.3, 0.7),
    (0.4, 0.8),
    (0.6, 0.9),
]

for g, (theta_min, theta_max) in enumerate(theta_grid):
    save_network(
        out_dir=f"./outputs/exp3_setting1_theta{g}",
        n_rep=n_rep,
        n=n,
        K=2,
        balanced=True,
        p_in=0.4,
        p_out=0.1,
        theta_dist="uniform",
        theta_min=theta_min,
        theta_max=theta_max,
        base_seed=1000 + 100 * g,
        index_base_edges=1,
    )
```

### Setting 2: heterogeneous degrees with low-degree nodes

Use:

\[
(p_{\mathrm{in}},p_{\mathrm{out}})=(0.9,0.3),
\]

with

\[
\theta_i\sim \mathrm{Unif}(\theta_{\min},\theta_{\max})
\quad \text{with probability }0.6,
\]

and

\[
\theta_i\sim \mathrm{Unif}(0.1\theta_{\min},0.1\theta_{\max})
\quad \text{with probability }0.4.
\]

Example generation loop:

```python
from generate_dcsbm_network import save_network

n_rep = 50
n = 20000

theta_grid = [
    (0.1, 0.5),
    (0.2, 0.6),
    (0.3, 0.7),
    (0.4, 0.8),
    (0.6, 0.9),
]

for g, (theta_min, theta_max) in enumerate(theta_grid):
    save_network(
        out_dir=f"./outputs/exp3_setting2_theta{g}",
        n_rep=n_rep,
        n=n,
        K=2,
        balanced=True,
        p_in=0.9,
        p_out=0.3,
        theta_dist="uniform_mixture_low",
        theta_min=theta_min,
        theta_max=theta_max,
        base_seed=2000 + 100 * g,
        index_base_edges=1,
    )
```

---
