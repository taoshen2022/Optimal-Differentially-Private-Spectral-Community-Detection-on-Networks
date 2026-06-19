# Real-data NetPTR experiment：Guide for Analysis on Flickr Data

## Files

- `netptr_realdata_utils.py`: reusable functions only: graph loading, sparse edge utilities, spectral clustering, NetPTR release, and EdgeFlip.
- `exp_flickr_subgraph_netptr_edgeflip.py`: main repeated-induced-subgraph real-data experiment. 
- `exp_fullgraph_eps_netptr_edgeflip.py`: optional full-graph epsilon sweep.
- `exp_a0_scan_netptr.py`: optional one-round `a0` scan.

## Main command matching the uploaded settings

```bash
python exp_flickr_subgraph_netptr_edgeflip.py \
  --base_dir ./flickr-dataset/pruned_mindeg100_single \
  --K 2 \
  --a0 0.12 \
  --eps_list 2.0 2.5 3.0 3.5 4.0 \
  --eps1_list 0.2 0.5 \
  --n_reps 10 \
  --subgraph_frac 0.9 \
  --delta 0.01 \
  --A0 50.0 \
  --A0_mode from_2toinfty \
  --A0_factor 1.05 \
  --eig_sort abs \
  --eig_tol 1e-3 \
  --eig_maxiter 200000 \
  --theta0_input 0.3 \
  --theta0_mode use_tilde \
  --theta0_floor 1e-12 \
  --edgeflip_max_edges_upper_priv 30000000 \
  --edgeflip_eps_offset 0.0 \
  --out_prefix flickr_subgraph_netptr_edgeflip
```

## Outputs

The main script writes:

- `<out_prefix>_raw.csv`
- `<out_prefix>_summary.csv`
- `<out_prefix>_meanARI.png`

