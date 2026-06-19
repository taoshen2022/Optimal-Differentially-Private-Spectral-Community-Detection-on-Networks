############################################################
# Figure: t1/t2 diagnostic histogram for party-flip R subjects
# t1: Party R yes-rate vs Party D yes-rate
# t2: Party-flip R yes-rate vs Party D yes-rate
############################################################

source("binetptr_senate_utils.R")

outdir <- get_arg("outdir", "outputs_senate_binetptr")
eps_diag <- get_num_arg("eps_diag", 8)
eps1_private <- get_num_arg("eps1_private", 0.5)
delta <- get_num_arg("delta", 0.1)
K <- get_int_arg("K", 2)
rep_diag <- get_int_arg("rep_diag", 1)
eps_grid <- get_num_vec_arg("eps_grid", c(1, 2, 4, 8, 12))

max_missing <- get_num_arg("max_missing", 0.05)
min_yes <- get_num_arg("min_yes", 0.10)
max_yes <- get_num_arg("max_yes", 0.90)

# Defaults follow the uploaded diagnostic script.
a0_frac <- get_num_arg("a0_frac", 0.80)
a0_floor <- get_num_arg("a0_floor", 0.15)
nstart_base <- get_int_arg("nstart_base", 100)
nstart_private <- get_int_arg("nstart_private", 50)
seed_base <- get_int_arg("seed_base", 123)
seed_private_grid <- get_int_arg("seed_private_grid", 200)
bins <- get_int_arg("bins", 30)
facet <- get_bool_arg("facet", TRUE)
min_n_flagged_R <- get_int_arg("min_n_flagged_R", 1)

ensure_dir(outdir)

dat <- load_senate_bipartite_data(max_missing = max_missing, min_yes = min_yes, max_yes = max_yes)
B <- dat$B
party <- get_party_vector(dat$legis_info)

base_fit <- nonprivate_bipartite_sc(B, K = K, nstart = nstart_base, seed = seed_base)
ref_labels <- base_fit$labels

j_diag <- match(eps_diag, eps_grid)
if (is.na(j_diag)) {
  warning("eps_diag is not in eps_grid; using j_diag = 1 for the diagnostic seed.")
  j_diag <- 1L
}
seed_diag <- seed_private_grid + 10000L * j_diag + rep_diag

fit_diag <- binitptr_bipartite_once(
  B = B,
  K = K,
  eps = eps_diag - eps1_private,
  delta = delta,
  eps1 = eps1_private,
  theta0_oracle = NULL,
  a0_frac = a0_frac,
  a0_floor = a0_floor,
  nstart = nstart_private,
  seed = seed_diag
)

subj_obj <- build_flagged_subject_table(
  np_labels = ref_labels,
  private_labels = fit_diag$labels,
  B = B,
  party = party,
  legis_info = dat$legis_info
)

flagged_subjects <- subj_obj$subject_table[subj_obj$subject_table$flagged, , drop = FALSE]
write.csv(flagged_subjects, file.path(outdir, sprintf("diag_eps%s_party_flip_subjects.csv", eps_diag)), row.names = FALSE)

cat("Number of party-flip subjects:", nrow(flagged_subjects), "\n")
print(table(flagged_subjects$party, useNA = "ifany"))

# The histogram focuses on Party-flip R subjects, matching the paper diagnostic.
test_df <- build_t1_t2_vote_tests(
  B = B,
  party = party,
  flagged = subj_obj$subject_table$flagged,
  vote_data = dat$vote_data,
  pooled = TRUE,
  min_n_flagged_R = min_n_flagged_R
)

if (nrow(test_df) == 0) {
  stop("No roll calls available after requiring at least one Party-flip R subject. Try another diagnostic seed or lower min_n_flagged_R.")
}

write.csv(test_df, file.path(outdir, sprintf("diag_eps%s_t1_t2_tests.csv", eps_diag)), row.names = FALSE)

p <- plot_t1_t2_histogram(
  test_df = test_df,
  outfile_pdf = file.path(outdir, sprintf("fig_t1_t2_hist_eps%s_binetptr.pdf", eps_diag)),
  outfile_png = file.path(outdir, sprintf("fig_t1_t2_hist_eps%s_binetptr.png", eps_diag)),
  bins = bins,
  facet = facet
)

print(p)
cat("Bi-NetPTR released:", fit_diag$released, "\n")
cat("Bi-NetPTR p_release:", round(fit_diag$p_release, 6), "\n")
cat("Saved outputs in:", outdir, "\n")
