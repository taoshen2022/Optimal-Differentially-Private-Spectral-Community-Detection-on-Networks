############################################################
# Figure: party composition at eps_all = 8
# The Bi-NetPTR cluster labels 1 and 2 are swapped for display by default.
############################################################

source("binetptr_senate_utils.R")

outdir <- get_arg("outdir", "outputs_senate_binetptr")
eps_show <- get_num_arg("eps_show", 8)
eps1_private <- get_num_arg("eps1_private", 0.5)
delta <- get_num_arg("delta", 0.1)
K <- get_int_arg("K", 2)

max_missing <- get_num_arg("max_missing", 0.05)
min_yes <- get_num_arg("min_yes", 0.10)
max_yes <- get_num_arg("max_yes", 0.90)

# Defaults follow the uploaded party-composition script.
a0_frac <- get_num_arg("a0_frac", 0.80)
a0_floor <- get_num_arg("a0_floor", 0.12)
nstart_base <- get_int_arg("nstart_base", 100)
nstart_private <- get_int_arg("nstart_private", 50)
seed_base <- get_int_arg("seed_base", 123)
seed_private_show <- get_int_arg("seed_private_show", 1235)
swap_private_clusters <- get_bool_arg("swap_private_clusters", FALSE)

ensure_dir(outdir)

dat <- load_senate_bipartite_data(max_missing = max_missing, min_yes = min_yes, max_yes = max_yes)
B <- dat$B
party <- get_party_vector(dat$legis_info)

base_fit <- nonprivate_bipartite_sc(B, K = K, nstart = nstart_base, seed = seed_base)
ref_labels <- base_fit$labels

fit_private_show <- binitptr_bipartite_once(
  B = B,
  K = K,
  eps = eps_show - eps1_private,
  delta = delta,
  eps1 = eps1_private,
  theta0_oracle = NULL,
  a0_frac = a0_frac,
  a0_floor = a0_floor,
  nstart = nstart_private,
  seed = seed_private_show
)

private_labels_for_plot <- fit_private_show$labels
if (swap_private_clusters) {
  private_labels_for_plot <- swap_two_cluster_labels(private_labels_for_plot)
}

plot_df <- rbind(
  make_cluster_party_plot_df(ref_labels, party, "NonPrivate"),
  make_cluster_party_plot_df(private_labels_for_plot, party, sprintf("Bi-NetPTR (εall = %g, ε1 = %g)", eps_show, eps1_private))
)
write.csv(plot_df, file.path(outdir, sprintf("party_composition_eps%s_plot_data.csv", eps_show)), row.names = FALSE)

p <- plot_party_composition(
  np_labels = ref_labels,
  private_labels = private_labels_for_plot,
  party = party,
  outfile_pdf = file.path(outdir, sprintf("fig_party_composition_eps%s_binetptr.pdf", eps_show)),
  outfile_png = file.path(outdir, sprintf("fig_party_composition_eps%s_binetptr.png", eps_show)),
  eps_all = eps_show,
  eps1 = eps1_private,
  swap_nonprivate_bar_position = TRUE
)

print(p)
cat("Bi-NetPTR released:", fit_private_show$released, "\n")
cat("Bi-NetPTR p_release:", round(fit_private_show$p_release, 6), "\n")
cat("Saved outputs in:", outdir, "\n")
