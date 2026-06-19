############################################################
# Experiment: mean ARI over repetitions for Senate roll calls
# Methods: Bi-NetPTR oracle theta0, Bi-NetPTR private theta0, EdgeFlip
############################################################

source("binetptr_senate_utils.R")

outdir <- get_arg("outdir", "outputs_senate_binetptr")
eps_grid <- get_num_vec_arg("eps_grid", c(1, 2, 4, 8, 12))
reps <- get_int_arg("reps", 300)
delta <- get_num_arg("delta", 0.1)
K <- get_int_arg("K", 2)

max_missing <- get_num_arg("max_missing", 0.05)
min_yes <- get_num_arg("min_yes", 0.10)
max_yes <- get_num_arg("max_yes", 0.90)

# Bi-NetPTR tuning, following the uploaded real-data script.
a0_frac <- get_num_arg("a0_frac", 0.50)
a0_floor <- get_num_arg("a0_floor", 0.14)
eps1_private <- get_num_arg("eps1_private", 0.5)
nstart_base <- get_int_arg("nstart_base", 100)
nstart_private <- get_int_arg("nstart_private", 50)

seed_base <- get_int_arg("seed_base", 123)
seed_oracle <- get_int_arg("seed_oracle", 100)
seed_private <- get_int_arg("seed_private", 200)
seed_edgeflip <- get_int_arg("seed_edgeflip", 300)

x_transform <- get_arg("x_transform", "log2")      # log2 or identity
show_error <- get_arg("show_error", "none")        # none, se, or sd
error_multiplier <- get_num_arg("error_multiplier", 1)

ensure_dir(outdir)

cat("Loading Senate roll-call data...\n")
dat <- load_senate_bipartite_data(max_missing = max_missing, min_yes = min_yes, max_yes = max_yes)
B <- dat$B
legis_info <- dat$legis_info
party <- get_party_vector(legis_info)

cat("Senate bipartite matrix:", nrow(B), "legislators x", ncol(B), "votes\n")
cat("Total Yea edges:", sum(B), "\n\n")

cat("Fitting non-private spectral clustering baseline...\n")
base_fit <- nonprivate_bipartite_sc(B, K = K, nstart = nstart_base, seed = seed_base)
ref_labels <- base_fit$labels
theta0_np <- estimate_theta0_bipartite(B, eps1 = NULL)

cat("Non-private gap(A):", round(base_fit$gapA, 6), "\n")
cat("Non-private theta0 proxy:", round(theta0_np, 6), "\n\n")
cat("Party x non-private cluster table:\n")
print(table(standardize_party_two(party), ref_labels))
cat("\n")

cat("Running Bi-NetPTR with oracle theta0...\n")
res_oracle <- run_binitptr_grid(
  B = B,
  ref_labels = ref_labels,
  K = K,
  eps_total_grid = eps_grid,
  eps1 = 0,
  theta0_oracle = theta0_np,
  delta = delta,
  reps = reps,
  a0_frac = a0_frac,
  a0_floor = a0_floor,
  nstart = nstart_private,
  seed = seed_oracle,
  method = "Bi-NetPTR_oracle"
)

cat("Running Bi-NetPTR with private theta0...\n")
res_private <- run_binitptr_grid(
  B = B,
  ref_labels = ref_labels,
  K = K,
  eps_total_grid = eps_grid,
  eps1 = eps1_private,
  theta0_oracle = NULL,
  delta = delta,
  reps = reps,
  a0_frac = a0_frac,
  a0_floor = a0_floor,
  nstart = nstart_private,
  seed = seed_private,
  method = "Bi-NetPTR_private"
)

cat("Running EdgeFlip...\n")
res_edgeflip <- run_edgeflip_grid(
  B = B,
  ref_labels = ref_labels,
  K = K,
  eps_total_grid = eps_grid,
  privacy_model = "node_column",
  debias = FALSE,
  reps = reps,
  nstart = nstart_private,
  seed = seed_edgeflip,
  method = "EdgeFlip"
)

write.csv(res_oracle$raw, file.path(outdir, "mean_ari_raw_binetptr_oracle.csv"), row.names = FALSE)
write.csv(res_private$raw, file.path(outdir, "mean_ari_raw_binetptr_private.csv"), row.names = FALSE)
write.csv(res_edgeflip$raw, file.path(outdir, "mean_ari_raw_edgeflip.csv"), row.names = FALSE)

align_summary_columns <- function(df, cols) {
  missing_cols <- setdiff(cols, names(df))
  for (nm in missing_cols) df[[nm]] <- NA
  df[, cols, drop = FALSE]
}

summary_cols <- Reduce(
  union,
  list(names(res_oracle$summary), names(res_private$summary), names(res_edgeflip$summary))
)

summary_all <- do.call(
  rbind,
  list(
    align_summary_columns(res_oracle$summary, summary_cols),
    align_summary_columns(res_private$summary, summary_cols),
    align_summary_columns(res_edgeflip$summary, summary_cols)
  )
)
write.csv(summary_all, file.path(outdir, "mean_ari_summary_all.csv"), row.names = FALSE)

cat("Summary:\n")
print(summary_all)

p <- plot_mean_ari_compare(
  sum_oracle = res_oracle$summary,
  sum_private = res_private$summary,
  sum_edgeflip = res_edgeflip$summary,
  outfile_pdf = file.path(outdir, "fig_mean_ari_binetptr_edgeflip.pdf"),
  outfile_png = file.path(outdir, "fig_mean_ari_binetptr_edgeflip.png"),
  title = get_arg("title", "Senate roll-call data"),
  show_error = show_error,
  x_transform = "identity",
  error_multiplier = error_multiplier
)

print(p)
cat("Saved outputs in:", outdir, "\n")
