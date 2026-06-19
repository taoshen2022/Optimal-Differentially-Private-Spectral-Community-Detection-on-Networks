# Real-data NetPTR experiment：Guide for Analysis on Senate Data
This folder repackages the uploaded Senate roll-call real-data scripts into one utility file and three runnable experiment/figure scripts.

## Files

- `binetptr_senate_utils.R`  
  Shared utilities: data loading, bipartite spectral clustering, Bi-NetPTR, EdgeFlip, party-composition helpers, and test-statistic diagnostics.

- `exp_senate_mean_ari.R`  
  Runs repetitions over an epsilon grid and produces the mean ARI figure. The figure uses blue and green curves for Bi-NetPTR, red squares for EdgeFlip, and a black dashed NonPrivate reference line.

- `fig_senate_party_composition_eps8.R`  
  Produces the eps_all = 8 party-composition figure. By default, the Bi-NetPTR cluster labels 1 and 2 are swapped for display, matching the requested relabeling.

- `fig_senate_test_statistics_histogram.R`  
  Produces the t1/t2 diagnostic histogram for Party-flip R subjects.

## Requirements

```r
install.packages(c("pscl", "ggplot2"))
```

## Run from this folder

```bash
Rscript exp_senate_mean_ari.R
Rscript fig_senate_party_composition_eps8.R
Rscript fig_senate_test_statistics_histogram.R
```

Outputs are saved by default to `outputs_senate_binetptr/`.

## Useful options

Arguments use the form `--name=value`.

```bash
Rscript exp_senate_mean_ari.R \
  --eps_grid=1,2,4,8,12 \
  --reps=300 \
  --show_error=none \
  --x_transform=log2
```

```bash
Rscript fig_senate_party_composition_eps8.R \
  --eps_show=8 \
  --eps1_private=0.5 \
  --swap_private_clusters=TRUE
```

```bash
Rscript fig_senate_test_statistics_histogram.R \
  --eps_diag=8 \
  --eps1_private=0.5 \
  --bins=30 \
  --facet=TRUE
```
