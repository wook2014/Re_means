#' @useDynLib Remeans, .registration = TRUE
#' @importFrom stats predict fitted
#' @keywords internal
"_PACKAGE"

# Check and coerce to double
as_data_matrix <- function(x, what = "x") {
  if (inherits(x, "tbl_df") || is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x) || !is.numeric(x)) {
    stop("`", what, "` must be a numeric matrix, data frame or tibble", call. = FALSE) # nolint: line_length_linter.
  }
  if (anyNA(x) || any(!is.finite(x))) stop("`", what, "` contains NA or non-finite values", call. = FALSE) # nolint: line_length_linter.
  storage.mode(x) <- "double"
  x
}

# Threads: argument, option, cores
default_threads <- function(n_threads) {
  if (!is.null(n_threads)) return(max(1L, as.integer(n_threads)))
  opt <- getOption("Remeans.threads")
  if (!is.null(opt)) return(max(1L, as.integer(opt)))
  cores <- max(1L, parallel::detectCores(logical = FALSE) - 1L, na.rm = TRUE)
  # R CMD check --as-cran sets this; CRAN allows at most two cores.
  limit <- tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_", ""))
  if (nzchar(limit) && limit != "false") cores <- min(cores, 2L)
  cores
}

# Warn on non-unit rows
check_unit_rows <- function(x) {
  rows <- if (nrow(x) > 1000) x[seq(1, nrow(x), length.out = 1000), , drop = FALSE] else x # nolint: line_length_linter.
  if (any(abs(rowSums(rows^2) - 1) > 1e-3)) {
    warning("spherical = TRUE expects unit-norm rows; normalize x first", call. = FALSE) # nolint: line_length_linter.
  }
}

#' e-means clustering
#'
#' Mini-batch k-means with exponential-moving-average centroid updates
#' (Bessa et al., 2026, arXiv:2611.09501). Batch size and step size come from
#' fitted scaling laws. The epoch budget defaults to five passes.
#'
#' Each epoch visits the rows in a fresh random order in batches of
#' `batch_size`. Every batch is assigned to the nearest centroids, and each
#' touched centroid moves towards its batch mean with weight
#' `alpha * n_j / N_j`, where the mass `N_j` is itself an exponential moving
#' average and `alpha` follows a cosine schedule from `alpha0` to
#' `alpha0 / 10`. Every five updates, clusters with mass below
#' `unused_threshold * mean(mass)` are revived by splitting large clusters
#' (picked with probability proportional to mass). Each nonempty Euclidean
#' center is the mean of its assigned rows from the last training pass.
#'
#' @param x Numeric matrix or data frame, one row per point.
#' @param centers Number of clusters, or a matrix with one row per initial
#'   center and the same columns as `x`.
#' @param nepochs Epoch budget; fractions allowed (e.g. `1.5`).
#' @param batch_law Batch size rule. `"auto"` uses `"quality"` at 16
#'   dimensions or fewer and `"throughput"` otherwise. Use `"quality"`
#'   for batches roughly one eighth as large.
#' @param batch_size,alpha0 Optional overrides for batch size and initial EMA
#'   step. A custom batch size still determines the default step size.
#' @param unused_threshold Revive clusters below this fraction of the mean
#'   running mass. Defaults to 0 at two dimensions or fewer and 0.1
#'   otherwise. Set to 0 to disable revival.
#' @param spherical Cluster unit vectors by cosine similarity. Rows of `x`
#'   must already be unit-norm; centers are returned as unit vectors.
#' @param max_points_per_centroid If set, cap the training sample at this
#'   many rows per requested cluster, drawn at random.
#' @param precision `"double"`, or `"single"` to store the training data and
#'   compute distances in float32. Batch
#'   statistics, centroids and masses always accumulate in double precision,
#'   and `withinss` is computed from the original data. As in the faiss FP32
#'   path, single precision assigns poorly when the data sit far from the
#'   origin (the error grows with the squared norm of the rows); centre the
#'   data first, e.g. with `scale(x, scale = FALSE)`. Double precision hits
#'   the same limit at offsets around 1e8.
#' @param final_assign If `TRUE` (default), assign every row of `x` to the
#'   final centers, so `cluster`, `withinss` and `size` describe the returned
#'   centers (as [stats::kmeans()]). If `FALSE`, `cluster` holds labels from
#'   the last training pass. Rows omitted by subsampling or by an epoch
#'   budget below one have `NA` labels.
#' @param seed Integer seed for initialization, shuffling and revival.
#'   Does not touch R's random number generator.
#' @param n_threads Threads for assignment. Defaults to
#'   `getOption("Remeans.threads")`, else physical cores minus one (at most
#'   two when the `_R_CHECK_LIMIT_CORES_` environment variable is set).
#' @param verbose Print the recipe and per-epoch objective.
#' @return An object inheriting from `remeans`, `emeans` and `kmeans`, with
#'   components
#'   `cluster`, `centers`, `totss`, `withinss`, `tot.withinss`, `betweenss`,
#'   `size`, `iter` (epochs), `ifault` (0), plus `obj` (moving-pass objective
#'   per epoch), `iteration_stats`, `recipe`, `n_splits`, `spherical` and
#'   `precision`.
#' @examples
#' set.seed(1)
#' x <- rbind(matrix(rnorm(2000, 0), ncol = 2), matrix(rnorm(2000, 5), ncol = 2))
#' fit <- e_means(x, centers = 2, nepochs = 3, n_threads = 1)
#' fit$centers
#' table(fit$cluster)
#' @export
e_means <- function(x, centers, nepochs = 5, final_assign = TRUE,
                    batch_law = c("auto", "throughput", "quality"),
                    batch_size = NULL, alpha0 = NULL, spherical = FALSE,
                    unused_threshold = NULL, max_points_per_centroid = NULL,
                    precision = c("double", "single"), seed = 1234L,
                    n_threads = NULL, verbose = FALSE) {
  # Validate inputs
  if (!is.numeric(nepochs) || length(nepochs) != 1L ||
      !is.finite(nepochs) || nepochs <= 0) {
    stop("`nepochs` must be a positive finite number.", call. = FALSE)
  }
  if (!is.logical(final_assign) || length(final_assign) != 1L || is.na(final_assign)) {
    stop("`final_assign` must be TRUE or FALSE.", call. = FALSE)
  }
  batch_law <- match.arg(batch_law)
  precision <- match.arg(precision)
  single <- precision == "single"
  x <- as_data_matrix(x)
  n <- nrow(x)
  d <- ncol(x)
  init <- NULL
  if (is.matrix(centers)) {
    init <- as_data_matrix(centers, "centers")
    if (ncol(init) != d) stop("initial centers must have ", d, " columns", call. = FALSE)
    k <- nrow(init)
  } else {
    k <- centers
  }
  if (!is.numeric(k) || length(k) != 1L || !is.finite(k) || k < 1 || k != floor(k)) {
    stop("`centers` must be a positive integer or a matrix of initial centers.",
         call. = FALSE)
  }
  if (k > n) stop("more clusters than rows", call. = FALSE)
  k <- as.integer(k)
  if (spherical) check_unit_rows(x)
  threads <- default_threads(n_threads)
  seed <- as.numeric(seed)

  # Optional training subset
  keep <- NULL
  if (!is.null(max_points_per_centroid)) {
    if (max_points_per_centroid < 1) stop("max_points_per_centroid must be >= 1", call. = FALSE) # nolint: line_length_linter.
    limit <- floor(max_points_per_centroid) * k  # whole points, as faiss
    if (n > limit) {
      keep <- sort(em_sample(n, limit, seed + 1))
      if (verbose) message(sprintf("Sampling %d of %d rows (max_points_per_centroid = %d)", # nolint: line_length_linter.
                                   length(keep), n, as.integer(max_points_per_centroid))) # nolint: line_length_linter.
    }
  }
  xt <- if (is.null(keep)) x else x[keep, , drop = FALSE]
  nt <- nrow(xt)
  if (k > nt) stop("more clusters than training rows", call. = FALSE)

  # Recipe and initial centers
  recipe <- emeans_recipe(nt, k, d, nepochs, batch_law, batch_size, alpha0, unused_threshold) # nolint: line_length_linter.
  if (is.null(init)) {
    init <- if (nt == k) xt else xt[em_sample(nt, k, seed), , drop = FALSE]
  }
  if (verbose) {
    message(sprintf("Clustering %d points in %dD to %d clusters, %g epochs (batch %d, alpha0 %.3g, %s law%s)", # nolint: line_length_linter.
                    nt, d, k, sum(recipe$epoch_rows) / nt, as.integer(recipe$batch_size), recipe$alpha0, # nolint: line_length_linter.
                    if (is.na(recipe$batch_law)) "given" else recipe$batch_law,
                    if (spherical) ", spherical" else ""))
  }

  # Rust data and state
  data <- em_data(xt, single)
  st <- em_new(init, nt, spherical, single, recipe$batch_size, recipe$alpha0,
               recipe$epoch_rows, recipe$unused_threshold, seed, threads)
  # Train in chunks; allow interrupts
  chunk_steps <- max(1, floor(2e9 / (recipe$batch_size * k * d)))
  last_epoch <- 0
  start <- proc.time()[["elapsed"]]
  repeat {
    done <- em_train_steps(st, data, chunk_steps, !final_assign)
    if (verbose) {
      # Report finished epochs
      info <- em_info(st)
      while (last_epoch < length(info$obj)) {
        last_epoch <- last_epoch + 1
        message(sprintf("  Epoch %d / %d (%.2f s): objective=%g, splits=%d",
                        last_epoch, info$n_epochs, proc.time()[["elapsed"]] - start, # nolint: line_length_linter.
                        info$obj[last_epoch], as.integer(info$nsplit[last_epoch]))) # nolint: line_length_linter.
      }
    }
    if (done) break
  }
  rm(data)
  # Exact closing centers
  centers <- em_finalize(st)
  info <- em_info(st)
  colnames(centers) <- colnames(x)

  # Labels: reassign or last pass
  dist2 <- NULL
  if (final_assign) {
    a <- em_assign(centers, x, spherical, single, threads)
    cluster <- a$cluster
    # Spherical gives similarities
    if (!spherical) dist2 <- a$distance
  } else {
    labels <- em_labels(st)
    cluster <- rep(NA_integer_, n)
    cluster[if (is.null(keep)) seq_len(n) else keep] <- labels
  }
  new_emeans(x, centers, cluster, k, recipe, info, spherical, precision, dist2)
}

# Build kmeans-compatible result
new_emeans <- function(
  x, centers, cluster, k,
  recipe, info, spherical, precision, dist2 = NULL
) {

  ok <- !is.na(cluster)
  all_rows <- all(ok)
  xs <- if (all_rows) x else x[ok, , drop = FALSE]
  cl <- cluster[ok]
  if (is.null(dist2)) {
    dist2 <- if (length(cl)) rowSums((xs - centers[cl, , drop = FALSE])^2) else numeric() # nolint: line_length_linter.
  }
  # Withinss from squared distances
  withinss <- numeric(k)
  if (length(cl)) {
    s <- rowsum(dist2, cl, reorder = TRUE)
    withinss[as.integer(rownames(s))] <- s[, 1]
  }
  totss <- em_totss(xs)
  names(cluster) <- rownames(x)
  structure(
    list(
      cluster = cluster,
      centers = centers,
      totss = totss,
      withinss = withinss,
      tot.withinss = sum(withinss),
      betweenss = totss - sum(withinss),
      size = tabulate(cl, k),
      iter = length(info$obj),
      ifault = 0L,
      obj = info$obj,
      iteration_stats = data.frame(epoch = seq_along(info$obj), obj = info$obj,
                                   imbalance = info$imbalance, nsplit = info$nsplit), # nolint: line_length_linter.
      recipe = recipe,
      n_splits = info$n_splits,
      spherical = spherical,
      precision = precision
    ),
    class = c("remeans", "emeans", "kmeans")
  )
}

# Print fit summary
#' @export
print.remeans <- function(x, ...) {
  cat(sprintf("e-means clustering with %d clusters of sizes %s\n", nrow(x$centers), # nolint: line_length_linter.
              paste(utils::head(x$size, 20), collapse = ", ")),
      if (length(x$size) > 20) "...\n" else "", sep = "")
  cat(sprintf("Epochs: %g (batch %d, alpha0 %.3g), splits: %d%s\n",
              x$recipe$nepochs,
              as.integer(x$recipe$batch_size), x$recipe$alpha0, as.integer(x$n_splits), # nolint: line_length_linter.
              if (x$spherical) ", spherical" else ""))
  cat(sprintf("Within cluster sum of squares: %g (between_SS / total_SS = %5.1f %%)\n", # nolint: line_length_linter.
              x$tot.withinss, 100 * x$betweenss / x$totss))
  cat("Available components:\n")
  print(names(x))
  invisible(x)
}

#' Assign new data to e-means centers
#'
#' @param object An `e_means()` fit.
#' @param newdata Numeric matrix or data frame with the same columns.
#' @param distance If `TRUE`, return a data frame with `cluster` and
#'   `distance`: the squared Euclidean distance, or, when spherical, the
#'   inner product with the unit centers (the cosine similarity if `newdata`
#'   rows are unit-norm).
#' @param n_threads Threads for assignment. Uses the same default as `e_means()`.
#' @param ... Unused.
#' @return Integer cluster labels, or a data frame when `distance = TRUE`.
#' @export
predict.remeans <- function(
  object, newdata,
  distance = FALSE, n_threads = NULL, ...
) {

  # Nearest center per row
  newdata <- as_data_matrix(newdata, "newdata")
  res <- em_assign(object$centers, newdata, object$spherical,
                   object$precision == "single", default_threads(n_threads))
  if (distance) data.frame(cluster = res$cluster, distance = res$distance) else res$cluster # nolint: line_length_linter.
}

# Centers or labels per row
#' @export
fitted.remeans <- function(object, method = c("centers", "classes"), ...) {
  method <- match.arg(method)
  if (is.null(object$cluster)) stop("no cluster labels; use predict()", call. = FALSE) # nolint: line_length_linter.
  if (method == "centers") object$centers[object$cluster, , drop = FALSE] else object$cluster # nolint: line_length_linter.
}
