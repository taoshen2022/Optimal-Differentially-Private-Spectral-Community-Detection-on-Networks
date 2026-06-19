############################################################
# Utility functions for Senate bipartite network privacy
#
# User-facing method name in all figures: Bi-NetPTR
# Data source: pscl::s109, 109th U.S. Senate roll-call voting
############################################################

## ---------------- package helpers ----------------

require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' is required. Please install it first.", pkg), call. = FALSE)
  }
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

## ---------------- command-line helpers ----------------

get_arg <- function(name, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  key <- paste0("--", name, "=")
  hit <- grep(paste0("^", key), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(key, "", hit[length(hit)], fixed = TRUE)
}

get_num_arg <- function(name, default) {
  val <- get_arg(name, NULL)
  if (is.null(val) || !nzchar(val)) return(default)
  as.numeric(val)
}

get_int_arg <- function(name, default) {
  val <- get_arg(name, NULL)
  if (is.null(val) || !nzchar(val)) return(default)
  as.integer(val)
}

get_bool_arg <- function(name, default = FALSE) {
  val <- get_arg(name, NULL)
  if (is.null(val) || !nzchar(val)) return(default)
  tolower(val) %in% c("1", "true", "t", "yes", "y")
}

get_num_vec_arg <- function(name, default) {
  val <- get_arg(name, NULL)
  if (is.null(val) || !nzchar(val)) return(default)
  as.numeric(strsplit(val, ",")[[1]])
}

## ---------------- basic utilities ----------------

rLaplace <- function(n, scale = 1) {
  u <- runif(n, min = -0.5, max = 0.5)
  -scale * sign(u) * log(1 - 2 * abs(u))
}

row_normalize <- function(X) {
  nr <- sqrt(rowSums(X^2))
  nr[nr == 0] <- 1
  X / nr
}

choose2_safe <- function(x) x * (x - 1) / 2

adjusted_rand_index <- function(x, y) {
  x <- as.integer(factor(x))
  y <- as.integer(factor(y))
  tab <- table(x, y)
  n <- sum(tab)
  if (n < 2) return(NA_real_)

  sum_ij <- sum(choose2_safe(tab))
  a_i <- rowSums(tab)
  b_j <- colSums(tab)
  sum_a <- sum(choose2_safe(a_i))
  sum_b <- sum(choose2_safe(b_j))
  expected <- sum_a * sum_b / choose2_safe(n)
  max_index <- 0.5 * (sum_a + sum_b)
  denom <- max_index - expected
  if (abs(denom) < .Machine$double.eps) return(0)

  (sum_ij - expected) / denom
}

eig_gap <- function(vals, K) {
  if (length(vals) < K + 1) stop("Need at least K+1 eigenvalues.")
  vals[K] - vals[K + 1]
}

## ---------------- data loading ----------------

load_senate_bipartite_data <- function(max_missing = 0.05,
                                       min_yes = 0.10,
                                       max_yes = 0.90,
                                       drop_zero_rows = TRUE) {
  require_pkg("pscl")
  data("s109", package = "pscl")
  rc <- pscl::s109

  X <- rc$votes
  yea_codes <- rc$codes$yea
  nay_codes <- rc$codes$nay

  B <- matrix(NA_real_, nrow = nrow(X), ncol = ncol(X), dimnames = dimnames(X))
  B[X %in% yea_codes] <- 1
  B[X %in% nay_codes] <- 0

  miss_frac <- colMeans(is.na(B))
  keep1 <- miss_frac <= max_missing
  B <- B[, keep1, drop = FALSE]

  yes_rate <- colMeans(B, na.rm = TRUE)
  keep2 <- !is.na(yes_rate) & yes_rate >= min_yes & yes_rate <= max_yes
  B <- B[, keep2, drop = FALSE]

  vote_keep <- which(keep1)[keep2]
  vote_data <- NULL
  if (!is.null(rc$vote.data)) {
    vote_data <- as.data.frame(rc$vote.data)
    if (nrow(vote_data) >= max(vote_keep)) {
      vote_data <- vote_data[vote_keep, , drop = FALSE]
    }
  }

  B[is.na(B)] <- 0

  keep_rows <- rep(TRUE, nrow(B))
  if (drop_zero_rows) {
    keep_rows <- rowSums(B) > 0
    B <- B[keep_rows, , drop = FALSE]
  }

  legis_info <- rc$legis.data
  if (!is.null(legis_info)) {
    legis_info <- as.data.frame(legis_info)
    legis_info <- legis_info[keep_rows, , drop = FALSE]
  }

  list(B = B, legis_info = legis_info, vote_data = vote_data, rc = rc)
}

pick_first_col <- function(df, patterns) {
  if (is.null(df)) return(NULL)
  nm <- names(df)
  idx <- unique(unlist(lapply(patterns, function(p) grep(p, nm, ignore.case = TRUE))))
  if (length(idx) == 0) return(NULL)
  nm[idx[1]]
}

standardize_party_two <- function(party) {
  z <- trimws(as.character(party))
  out <- rep(NA_character_, length(z))
  out[grepl("^d$|dem", z, ignore.case = TRUE)] <- "D"
  out[grepl("^r$|rep", z, ignore.case = TRUE)] <- "R"

  if (all(is.na(out))) {
    lev <- sort(unique(z[!is.na(z)]))
    if (length(lev) >= 2) {
      out[z == lev[1]] <- lev[1]
      out[z == lev[2]] <- lev[2]
    }
  }
  out
}

get_party_vector <- function(legis_info) {
  if (is.null(legis_info)) stop("legis_info is NULL; cannot extract party.")
  party_col <- grep("party", names(legis_info), ignore.case = TRUE, value = TRUE)
  if (length(party_col) == 0) stop("No party column found in legis_info.")
  legis_info[[party_col[1]]]
}

get_subject_names <- function(legis_info) {
  if (is.null(legis_info)) return(NULL)
  ncol_name <- pick_first_col(legis_info, c("^name$", "last", "legis", "member", "senator"))
  if (is.null(ncol_name)) return(rownames(legis_info))
  as.character(legis_info[[ncol_name]])
}

get_subject_state <- function(legis_info) {
  if (is.null(legis_info)) return(NULL)
  scol <- pick_first_col(legis_info, c("state", "st$", "stateabb"))
  if (is.null(scol)) return(NULL)
  as.character(legis_info[[scol]])
}

align_vote_data <- function(B, vote_data) {
  if (is.null(vote_data)) return(NULL)
  vd <- as.data.frame(vote_data)
  if (nrow(vd) == ncol(B)) return(vd)

  b_ids <- colnames(B)
  v_ids <- rownames(vd)
  if (!is.null(b_ids) && !is.null(v_ids)) {
    idx <- match(b_ids, v_ids)
    if (sum(!is.na(idx)) > 0) {
      vd2 <- vd[idx, , drop = FALSE]
      rownames(vd2) <- b_ids
      return(vd2)
    }
  }

  n_use <- min(nrow(vd), ncol(B))
  vd2 <- vd[seq_len(n_use), , drop = FALSE]
  warning(sprintf("vote_data could not be cleanly aligned to B by names; using first %d rows only.", n_use))
  vd2
}

## ---------------- spectral clustering and privacy mechanisms ----------------

estimate_theta0_bipartite <- function(B, eps1 = NULL) {
  m <- ncol(B)
  max_deg <- max(rowSums(B))
  val <- max_deg / m
  if (!is.null(eps1) && eps1 > 0) {
    val <- val + rLaplace(1, scale = 1 / (eps1 * m))
  }
  sqrt(max(val, 1e-10))
}

nonprivate_bipartite_sc <- function(B, K = 2, nstart = 50, seed = 1) {
  set.seed(seed)
  A <- tcrossprod(B) / ncol(B)
  ee <- eigen(A, symmetric = TRUE)

  Xi_hat <- ee$vectors[, seq_len(K), drop = FALSE]
  R_hat <- row_normalize(Xi_hat)
  km <- kmeans(R_hat, centers = K, nstart = nstart)

  list(
    labels = km$cluster,
    Xi_hat = Xi_hat,
    eigvals = ee$values,
    gapA = eig_gap(ee$values, K),
    A = A
  )
}

binitptr_bipartite_once <- function(B,
                                    K = 2,
                                    eps = 1,
                                    delta = 1e-6,
                                    eps1 = 0,
                                    theta0_oracle = NULL,
                                    a0_frac = 0.80,
                                    a0_floor = 0.05,
                                    nstart = 50,
                                    seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (eps <= 0) stop("eps must be positive.")
  if (delta <= 0 || delta >= 1) stop("delta must lie in (0,1).")

  n <- nrow(B)
  m <- ncol(B)
  A <- tcrossprod(B) / m
  ee <- eigen(A, symmetric = TRUE)
  Xi_hat <- ee$vectors[, seq_len(K), drop = FALSE]
  gapA <- eig_gap(ee$values, K)

  theta0_hat <- if (!is.null(theta0_oracle)) {
    theta0_oracle
  } else {
    estimate_theta0_bipartite(B, eps1 = eps1)
  }

  a0_hat <- a0_frac * gapA / max(n * theta0_hat^2, 1e-10)
  a0_hat <- max(a0_hat, a0_floor)

  gammaI <- (m / (2 * n)) * max(gapA - a0_hat * theta0_hat^4 * n, 0)

  Mthr <- 1 + (2 / eps) * log(1 / delta)
  p_release <- if (gammaI > 2 * Mthr) {
    1
  } else {
    plogis(0.5 * eps * (gammaI - Mthr))
  }

  alpha <- 4 * sqrt(2) / (a0_hat * theta0_hat^4 * m)
  noise_sd <- (alpha / eps) * sqrt(2 * log(1.25 / delta))

  released <- as.logical(rbinom(1, size = 1, prob = p_release))

  Xi_priv <- if (released) {
    Xi_hat + matrix(rnorm(n * K, sd = noise_sd), nrow = n, ncol = K)
  } else {
    matrix(rnorm(n * K), nrow = n, ncol = K)
  }

  R_priv <- row_normalize(Xi_priv)
  km_priv <- kmeans(R_priv, centers = K, nstart = nstart)

  list(
    labels = km_priv$cluster,
    gammaI = gammaI,
    p_release = p_release,
    released = released,
    theta0_hat = theta0_hat,
    a0_hat = a0_hat,
    gapA = gapA,
    noise_sd = noise_sd
  )
}

# Backward-compatible alias for old scripts; figures still use Bi-NetPTR labels.
gapptr_bipartite_once <- binitptr_bipartite_once

run_binitptr_grid <- function(B,
                              ref_labels,
                              K = 2,
                              eps_total_grid = c(1, 2, 4, 8),
                              eps1 = 0,
                              theta0_oracle = NULL,
                              delta = 1e-6,
                              reps = 300,
                              a0_frac = 0.80,
                              a0_floor = 0.05,
                              nstart = 50,
                              seed = 1,
                              method = "Bi-NetPTR") {
  out <- vector("list", length(eps_total_grid) * reps)
  idx <- 1L

  for (j in seq_along(eps_total_grid)) {
    eps_total <- eps_total_grid[j]
    eps_main <- eps_total - eps1
    if (eps_main <= 0) stop("eps_total must exceed eps1.")

    for (r in seq_len(reps)) {
      fit <- binitptr_bipartite_once(
        B = B,
        K = K,
        eps = eps_main,
        delta = delta,
        eps1 = if (is.null(theta0_oracle)) eps1 else 0,
        theta0_oracle = theta0_oracle,
        a0_frac = a0_frac,
        a0_floor = a0_floor,
        nstart = nstart,
        seed = seed + 10000L * j + r
      )

      out[[idx]] <- data.frame(
        method = method,
        eps_total = eps_total,
        eps_main = eps_main,
        eps1 = if (is.null(theta0_oracle)) eps1 else 0,
        rep = r,
        ARI_vs_nonprivate = adjusted_rand_index(ref_labels, fit$labels),
        p_release = fit$p_release,
        released = as.numeric(fit$released),
        gammaI = fit$gammaI,
        gapA = fit$gapA,
        theta0_hat = fit$theta0_hat,
        a0_hat = fit$a0_hat,
        noise_sd = fit$noise_sd
      )
      idx <- idx + 1L
    }
  }

  raw <- do.call(rbind, out)
  summary <- do.call(rbind, lapply(split(raw, raw$eps_total), function(d) {
    data.frame(
      method = unique(d$method),
      eps_total = unique(d$eps_total),
      eps1 = unique(d$eps1),
      ARI_mean = mean(d$ARI_vs_nonprivate, na.rm = TRUE),
      ARI_sd = sd(d$ARI_vs_nonprivate, na.rm = TRUE),
      ARI_se = sd(d$ARI_vs_nonprivate, na.rm = TRUE) / sqrt(nrow(d)),
      p_release_mean = mean(d$p_release, na.rm = TRUE),
      released_mean = mean(d$released, na.rm = TRUE),
      gammaI_mean = mean(d$gammaI, na.rm = TRUE),
      theta0_mean = mean(d$theta0_hat, na.rm = TRUE),
      noise_sd_mean = mean(d$noise_sd, na.rm = TRUE)
    )
  }))
  rownames(summary) <- NULL

  list(raw = raw, summary = summary)
}

edgeflip_bipartite_once <- function(B,
                                    K = 2,
                                    eps_total = 1,
                                    privacy_model = c("node_column", "edge_entry"),
                                    debias = FALSE,
                                    nstart = 50,
                                    seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  privacy_model <- match.arg(privacy_model)

  n <- nrow(B)

  # Node-DP on bipartite data: one neighboring change = one whole vote column changes.
  # Independent randomized response per entry uses eps_entry = eps_total / n.
  eps_entry <- if (privacy_model == "node_column") eps_total / n else eps_total

  keep_prob <- exp(eps_entry) / (1 + exp(eps_entry))
  flip_prob <- 1 - keep_prob

  flips <- matrix(rbinom(length(B), size = 1, prob = flip_prob), nrow = nrow(B), ncol = ncol(B))
  Y <- (B + flips) %% 2

  B_used <- if (debias) {
    q <- flip_prob
    if (abs(1 - 2 * q) < 1e-8) stop("Debiasing unstable because flip_prob is too close to 0.5.")
    (Y - q) / (1 - 2 * q)
  } else {
    Y
  }

  fit <- nonprivate_bipartite_sc(B = B_used, K = K, nstart = nstart, seed = if (is.null(seed)) 1 else seed + 1)

  list(labels = fit$labels, eps_entry = eps_entry, flip_prob = flip_prob, B_priv = Y)
}

run_edgeflip_grid <- function(B,
                              ref_labels,
                              K = 2,
                              eps_total_grid = c(1, 2, 4, 8),
                              privacy_model = c("node_column", "edge_entry"),
                              debias = FALSE,
                              reps = 300,
                              nstart = 50,
                              seed = 1,
                              method = "EdgeFlip") {
  privacy_model <- match.arg(privacy_model)
  out <- vector("list", length(eps_total_grid) * reps)
  idx <- 1L

  for (j in seq_along(eps_total_grid)) {
    eps_total <- eps_total_grid[j]
    for (r in seq_len(reps)) {
      fit <- edgeflip_bipartite_once(
        B = B,
        K = K,
        eps_total = eps_total,
        privacy_model = privacy_model,
        debias = debias,
        nstart = nstart,
        seed = seed + 10000L * j + r
      )

      out[[idx]] <- data.frame(
        method = method,
        privacy_model = privacy_model,
        eps_total = eps_total,
        rep = r,
        ARI_vs_nonprivate = adjusted_rand_index(ref_labels, fit$labels),
        eps_entry = fit$eps_entry,
        flip_prob = fit$flip_prob
      )
      idx <- idx + 1L
    }
  }

  raw <- do.call(rbind, out)
  summary <- do.call(rbind, lapply(split(raw, raw$eps_total), function(d) {
    data.frame(
      method = unique(d$method),
      privacy_model = unique(d$privacy_model),
      eps_total = unique(d$eps_total),
      ARI_mean = mean(d$ARI_vs_nonprivate, na.rm = TRUE),
      ARI_sd = sd(d$ARI_vs_nonprivate, na.rm = TRUE),
      ARI_se = sd(d$ARI_vs_nonprivate, na.rm = TRUE) / sqrt(nrow(d)),
      eps_entry_mean = mean(d$eps_entry, na.rm = TRUE),
      flip_prob_mean = mean(d$flip_prob, na.rm = TRUE)
    )
  }))
  rownames(summary) <- NULL

  list(raw = raw, summary = summary)
}

## ---------------- mean ARI plot ----------------

plot_mean_ari_compare <- function(sum_oracle,
                                  sum_private,
                                  sum_edgeflip,
                                  outfile_pdf = NULL,
                                  outfile_png = NULL,
                                  title = "Senate roll-call data",
                                  x_transform = c("log2", "identity"),
                                  show_error = c("none", "se", "sd"),
                                  error_multiplier = 1) {
  require_pkg("ggplot2")
  x_transform <- match.arg(x_transform)
  show_error <- match.arg(show_error)
  label_oracle <- "Bi-NetPTR_eps1_0"
  label_private <- "Bi-NetPTR_eps1_05"
  label_edge <- "EdgeFlip"
  
  method_levels <- c(label_oracle, label_private, label_edge)
  
  legend_labels <- c(
    expression(paste("Bi-NetPTR (", epsilon[1], " = 0)")),
    expression(paste("Bi-NetPTR (", epsilon[1], " = 0.5)")),
    "EdgeFlip"
  )
  
  method_levels <- c(label_oracle, label_private, label_edge)
  
  make_df <- function(s, lab) {
    err <- switch(show_error,
                  none = rep(0, nrow(s)),
                  se = s$ARI_se,
                  sd = s$ARI_sd)
    
    data.frame(
      eps_total = s$eps_total,
      x = if (x_transform == "log2") log2(s$eps_total) else s$eps_total,
      mean = s$ARI_mean,
      ymin = pmax(0, s$ARI_mean - error_multiplier * err),
      ymax = pmin(1, s$ARI_mean + error_multiplier * err),
      method = lab,
      stringsAsFactors = FALSE
    )
  }
  
  df <- rbind(
    make_df(sum_oracle, label_oracle),
    make_df(sum_private, label_private),
    make_df(sum_edgeflip, label_edge)
  )
  
  df <- df[is.finite(df$x) & is.finite(df$mean), , drop = FALSE]
  df$method <- factor(df$method, levels = method_levels)
  
  color_vals <- setNames(
    c("blue", "darkgreen", "red"),
    method_levels
  )
  
  linetype_vals <- setNames(
    c("solid", "solid", "solid"),
    method_levels
  )
  
  shape_vals <- setNames(
    c(16, 16, 15),
    method_levels
  )
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = mean, group = method)) +
    ggplot2::geom_line(
      ggplot2::aes(color = method, linetype = method),
      linewidth = 1.15
    ) +
    ggplot2::geom_point(
      ggplot2::aes(color = method, shape = method),
      size = 3.0
    ) +
    ggplot2::scale_color_manual(
      values = color_vals,
      breaks = method_levels,
      labels = legend_labels,
      drop = FALSE
    ) +
    ggplot2::scale_linetype_manual(
      values = linetype_vals,
      breaks = method_levels,
      labels = legend_labels,
      drop = FALSE
    ) +
    ggplot2::scale_shape_manual(
      values = shape_vals,
      breaks = method_levels,
      labels = legend_labels,
      drop = FALSE
    ) +
    ggplot2::coord_cartesian(ylim = c(0, 1.02)) +
    ggplot2::labs(
      x = if (x_transform == "log2") expression(log[2](epsilon[all])) else expression(epsilon[all]),
      y = "Mean ARI",
      title = title,
      color = NULL,
      linetype = NULL,
      shape = NULL
    ) +
    ggplot2::theme_bw(base_size = 14) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, size = 17),
      
      # legend outside, right-top
      legend.position = "right",
      legend.box.just = "top",
      legend.justification = "top",
      
      legend.background = ggplot2::element_rect(fill = "white", color = "grey80"),
      legend.key = ggplot2::element_blank(),
      
      panel.grid.major = ggplot2::element_line(linetype = "dotted", color = "grey75"),
      panel.grid.minor = ggplot2::element_line(linetype = "dotted", color = "grey90")
    )
  
  if (show_error != "none") {
    p <- p + ggplot2::geom_errorbar(
      ggplot2::aes(x = x, ymin = ymin, ymax = ymax, color = method),
      width = 0.035 * diff(range(df$x, na.rm = TRUE)),
      linewidth = 0.55,
      alpha = 0.75
    )
  }
  
  if (!is.null(outfile_pdf)) {
    ggplot2::ggsave(outfile_pdf, p, width = 10.5, height = 5.8)
  }
  if (!is.null(outfile_png)) {
    ggplot2::ggsave(outfile_png, p, width = 10.5, height = 5.8, dpi = 300)
  }
  
  p
}
## ---------------- party-composition helpers ----------------

cluster_party_metrics <- function(labels, party) {
  ok <- !(is.na(labels) | is.na(party))
  labels <- labels[ok]
  party <- party[ok]

  labels <- factor(labels)
  party <- factor(standardize_party_two(party))

  tab <- table(cluster = labels, party = party)
  prop <- prop.table(tab, margin = 1)

  purity_each <- apply(prop, 1, max)
  cluster_sizes <- rowSums(tab)
  weighted_purity <- sum((cluster_sizes / sum(cluster_sizes)) * purity_each)
  min_purity <- min(purity_each)

  list(tab = tab, prop = prop, purity_each = purity_each, weighted_purity = weighted_purity, min_purity = min_purity)
}

swap_two_cluster_labels <- function(labels) {
  out <- labels
  out[labels == 1] <- -1
  out[labels == 2] <- 1
  out[labels == -1] <- 2
  out
}

make_cluster_party_plot_df <- function(labels, party, method_label) {
  party_std <- standardize_party_two(party)
  ok <- !(is.na(labels) | is.na(party_std))
  tab <- as.data.frame(table(cluster = labels[ok], party = party_std[ok]), stringsAsFactors = FALSE)
  names(tab) <- c("cluster", "party", "count")
  tab$count <- as.numeric(tab$count)
  totals <- aggregate(count ~ cluster, data = tab, sum)
  names(totals)[2] <- "total"
  tab <- merge(tab, totals, by = "cluster", all.x = TRUE)
  tab$prop <- ifelse(tab$total > 0, tab$count / tab$total, NA_real_)
  tab$method <- method_label
  tab$cluster_label <- paste0("Cluster ", tab$cluster)
  tab
}

plot_party_composition <- function(np_labels,
                                   private_labels,
                                   party,
                                   outfile_pdf = NULL,
                                   outfile_png = NULL,
                                   title = NULL,
                                   eps_all = 8,
                                   eps1 = 0.5,
                                   swap_nonprivate_bar_position = TRUE) {
  require_pkg("ggplot2")
  
  label_np <- "NonPrivate"
  label_private <- "BiNetPTR"
  
  df_np <- make_cluster_party_plot_df(np_labels, party, label_np)
  df_private <- make_cluster_party_plot_df(private_labels, party, label_private)
  
  df <- rbind(df_np, df_private)
  
  # Only keep Cluster 1 and Cluster 2
  df$cluster <- as.character(df$cluster)
  df <- df[df$cluster %in% c("1", "2"), , drop = FALSE]
  
  # Default x-position is the original cluster label
  df$x_cluster <- paste0("Cluster ", df$cluster)
  
  # For the NonPrivate panel only, swap the displayed positions
  # so original Cluster 1 is drawn at the Cluster 2 position,
  # and original Cluster 2 is drawn at the Cluster 1 position.
  if (swap_nonprivate_bar_position) {
    idx_np <- df$method == label_np
    df$x_cluster[idx_np & df$cluster == "1"] <- "Cluster 2"
    df$x_cluster[idx_np & df$cluster == "2"] <- "Cluster 1"
  }
  
  df$x_cluster <- factor(df$x_cluster, levels = c("Cluster 1", "Cluster 2"))
  df$party <- as.character(df$party)
  df$party[df$party == "D"] <- "Democratic"
  df$party[df$party == "R"] <- "Republican"
  df$party <- factor(df$party, levels = c("Democratic", "Republican"))
  
  # Internal facet labels. We use label_parsed below to display eps_all and eps_1 properly.
  df$method_facet <- ifelse(
    df$method == label_np,
    "'NonPrivate'",
    paste0(
      "'Bi-NetPTR'~(",
      "epsilon[all]==", eps_all,
      "*','~epsilon[1]==", eps1,
      ")"
    )
  )
  
  df$method_facet <- factor(
    df$method_facet,
    levels = c(
      "'NonPrivate'",
      paste0(
        "'Bi-NetPTR'~(",
        "epsilon[all]==", eps_all,
        "*','~epsilon[1]==", eps1,
        ")"
      )
    )
  )
  
  if (is.null(title)) {
    title <- bquote("Party composition")
  }
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = x_cluster, y = prop, fill = party)) +
    ggplot2::geom_col(width = 0.68, color = "white", linewidth = 0.45) +
    ggplot2::geom_text(
      ggplot2::aes(label = ifelse(prop >= 0.06, sprintf("%.0f%%", 100 * prop), "")),
      position = ggplot2::position_stack(vjust = 0.5),
      color = "white",
      size = 4.0,
      fontface = "bold"
    ) +
    ggplot2::facet_wrap(
      ~ method_facet,
      nrow = 1,
      labeller = ggplot2::label_parsed
    ) +
    ggplot2::scale_y_continuous(
      labels = function(x) paste0(round(100 * x), "%"),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    ggplot2::scale_fill_manual(
      values = c(
        "Democratic" = "#4C78A8",
        "Republican" = "#E45756"
      ),
      drop = FALSE
    ) +
    ggplot2::labs(
      x = NULL,
      y = "Party proportion",
      fill = "Party",
      title = title
    ) +
    ggplot2::theme_bw(base_size = 14) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, size = 17),
      strip.background = ggplot2::element_rect(fill = "grey96", color = "grey80"),
      strip.text = ggplot2::element_text(face = "plain", size = 13),
      legend.position = "bottom",
      legend.title = ggplot2::element_text(face = "bold"),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank()
    )
  
  if (!is.null(outfile_pdf)) {
    ggplot2::ggsave(outfile_pdf, p, width = 8.8, height = 5.0)
  }
  if (!is.null(outfile_png)) {
    ggplot2::ggsave(outfile_png, p, width = 8.8, height = 5.0, dpi = 300)
  }
  
  p
}

## ---------------- diagnostic helpers for party-flip R subjects ----------------

majority_party_map <- function(labels, party_std) {
  ok <- !(is.na(labels) | is.na(party_std))
  if (!any(ok)) {
    return(list(tab = matrix(0, nrow = 0, ncol = 0), majority = setNames(character(0), character(0))))
  }
  tab <- table(cluster = labels[ok], party = party_std[ok])
  maj <- apply(tab, 1, function(x) colnames(tab)[which.max(x)])
  list(tab = tab, majority = maj)
}

build_flagged_subject_table <- function(np_labels, private_labels, B, party, legis_info = NULL) {
  stopifnot(length(np_labels) == nrow(B), length(private_labels) == nrow(B), length(party) == nrow(B))

  party_std <- standardize_party_two(party)
  subj_names <- get_subject_names(legis_info)
  subj_state <- get_subject_state(legis_info)

  np_maj_obj <- majority_party_map(np_labels, party_std)
  private_maj_obj <- majority_party_map(private_labels, party_std)

  np_cluster_majority_party <- unname(np_maj_obj$majority[as.character(np_labels)])
  private_cluster_majority_party <- unname(private_maj_obj$majority[as.character(private_labels)])

  np_party_match <- !is.na(party_std) & !is.na(np_cluster_majority_party) & (party_std == np_cluster_majority_party)
  private_party_mismatch <- !is.na(party_std) & !is.na(private_cluster_majority_party) & (party_std != private_cluster_majority_party)
  flagged <- np_party_match & private_party_mismatch

  out <- data.frame(
    subject_id = seq_len(nrow(B)),
    name = if (is.null(subj_names)) paste0("subject_", seq_len(nrow(B))) else subj_names,
    state = if (is.null(subj_state)) NA_character_ else subj_state,
    party = party_std,
    np_label = np_labels,
    private_label = private_labels,
    np_cluster_majority_party = np_cluster_majority_party,
    private_cluster_majority_party = private_cluster_majority_party,
    np_party_match = np_party_match,
    private_party_mismatch = private_party_mismatch,
    flagged = flagged,
    yes_rate = rowMeans(B, na.rm = TRUE),
    stringsAsFactors = FALSE
  )

  out$direction <- ifelse(out$flagged, paste0(out$party, "_to_", out$private_cluster_majority_party), NA_character_)
  out$group_party <- ifelse(out$flagged, paste0("Party-flip ", out$party), NA_character_)
  out <- out[order(-out$flagged, out$party, out$name), ]

  list(
    subject_table = out,
    np_cluster_party_table = np_maj_obj$tab,
    private_cluster_party_table = private_maj_obj$tab
  )
}

two_prop_z_stat <- function(k1, n1, k2, n2, pooled = TRUE) {
  if (n1 <= 0 || n2 <= 0) return(NA_real_)

  p1 <- k1 / n1
  p2 <- k2 / n2

  if (pooled) {
    p_pool <- (k1 + k2) / (n1 + n2)
    se <- sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
  } else {
    se <- sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
  }

  if (is.na(se) || se <= 0) return(NA_real_)
  (p1 - p2) / se
}

build_t1_t2_vote_tests <- function(B,
                                   party,
                                   flagged,
                                   vote_data = NULL,
                                   pooled = TRUE,
                                   min_n_flagged_R = 1) {
  party_std <- standardize_party_two(party)
  D_idx <- which(party_std == "D")
  R_idx <- which(party_std == "R")
  flip_R_idx <- which(flagged & party_std == "R")
  vd <- align_vote_data(B, vote_data)

  rows <- vector("list", ncol(B))
  for (j in seq_len(ncol(B))) {
    k_D <- sum(B[D_idx, j], na.rm = TRUE)
    k_R <- sum(B[R_idx, j], na.rm = TRUE)
    k_F <- sum(B[flip_R_idx, j], na.rm = TRUE)

    n_D <- length(D_idx)
    n_R <- length(R_idx)
    n_F <- length(flip_R_idx)

    p_D <- k_D / n_D
    p_R <- k_R / n_R
    p_F <- ifelse(n_F > 0, k_F / n_F, NA_real_)

    t1 <- two_prop_z_stat(k_R, n_R, k_D, n_D, pooled = pooled)
    t2 <- two_prop_z_stat(k_F, n_F, k_D, n_D, pooled = pooled)

    rows[[j]] <- data.frame(
      vote_id = j,
      D_yes_rate = p_D,
      R_yes_rate = p_R,
      flipped_R_yes_rate = p_F,
      n_D = n_D,
      n_R = n_R,
      n_flipped_R = n_F,
      t1_R_vs_D = t1,
      t2_flippedR_vs_D = t2,
      abs_t1 = abs(t1),
      abs_t2 = abs(t2),
      gap_party = abs(p_R - p_D),
      flip_D_distance = abs(p_F - p_D),
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, rows)
  if (!is.null(vd)) {
    qcol <- pick_first_col(vd, c("question", "description", "desc", "bill", "issue", "title"))
    dcol <- pick_first_col(vd, c("date", "time"))
    sicol <- pick_first_col(vd, c("subject", "topic", "policy", "category"))
    if (!is.null(qcol)) out$question <- as.character(vd[[qcol]])
    if (!is.null(dcol)) out$date <- as.character(vd[[dcol]])
    if (!is.null(sicol)) out$subject <- as.character(vd[[sicol]])
  }

  out <- out[out$n_flipped_R >= min_n_flagged_R, , drop = FALSE]
  out
}

plot_t1_t2_histogram <- function(test_df,
                                 outfile_pdf = NULL,
                                 outfile_png = NULL,
                                 width = 8.6,
                                 height = 5.2,
                                 bins = 30,
                                 facet = TRUE) {
  require_pkg("ggplot2")
  if (nrow(test_df) == 0) stop("test_df is empty; no test statistics to plot.")
  
  # Internal labels for parsed facet labels
  stat_t1 <- "paste(t[1], ': Republican vs Democratic')"
  stat_t2 <- "paste(t[2], ': Flipped subjects vs Democratic')"
  
  df_long <- rbind(
    data.frame(
      statistic = stat_t1,
      value = test_df$t1_R_vs_D,
      stringsAsFactors = FALSE
    ),
    data.frame(
      statistic = stat_t2,
      value = test_df$t2_flippedR_vs_D,
      stringsAsFactors = FALSE
    )
  )
  
  df_long <- df_long[is.finite(df_long$value), , drop = FALSE]
  df_long$statistic <- factor(df_long$statistic, levels = c(stat_t1, stat_t2))
  
  fill_vals <- setNames(
    c("#4C78A8", "#59A14F"),
    c(stat_t1, stat_t2)
  )
  
  fill_labs <- as.expression(list(
    bquote(t[1] * ": Republican vs Democratic"),
    bquote(t[2] * ": Flipped subjects vs Democratic")
  ))
  
  p <- ggplot2::ggplot(df_long, ggplot2::aes(x = value, fill = statistic)) +
    ggplot2::geom_histogram(
      bins = bins,
      alpha = 0.75,
      position = "identity",
      color = "white",
      linewidth = 0.25
    ) +
    # Keep only the zero reference line; remove +/- 1.96 lines.
    ggplot2::geom_vline(
      xintercept = 0,
      linetype = "solid",
      linewidth = 0.35,
      color = "grey25"
    ) +
    ggplot2::scale_fill_manual(
      values = fill_vals,
      breaks = c(stat_t1, stat_t2),
      labels = fill_labs,
      drop = FALSE
    ) +
    ggplot2::labs(
      x = "Test statistic",
      y = "Number of roll calls",
      fill = NULL
    ) +
    ggplot2::theme_bw(base_size = 14) +
    ggplot2::theme(
      legend.position = if (facet) "none" else "bottom",
      panel.grid.major = ggplot2::element_line(linetype = "dotted", color = "grey82"),
      panel.grid.minor = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "grey95", color = "grey80"),
      strip.text = ggplot2::element_text(face = "plain", size = 12.5)
    )
  
  if (facet) {
    p <- p +
      ggplot2::facet_wrap(
        ~ statistic,
        ncol = 1,
        scales = "free_y",
        labeller = ggplot2::label_parsed
      )
  }
  
  if (!is.null(outfile_pdf)) {
    ggplot2::ggsave(outfile_pdf, p, width = width, height = height)
  }
  if (!is.null(outfile_png)) {
    ggplot2::ggsave(outfile_png, p, width = width, height = height, dpi = 300)
  }
  
  p
}
