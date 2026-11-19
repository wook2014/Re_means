# Scaling laws, paper Eqs 6-7

# faiss batch law: C, exponents
.batch_laws <- list(
  throughput = c(0.035134, 0.134749, 0.876473, 0.813344),
  quality = c(0.00439175, 0.134749, 0.876473, 0.813344)
)
# faiss step law: C, exponents
.alpha_law <- c(0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
# Revival and auto-law cutoffs
.unused_threshold_default <- 0.1
.no_revival_max_d <- 2
.auto_quality_max_d <- 16

# Nearest power of two
round_pow2 <- function(value) {
  if (value <= 1) return(1)
  low <- 2^floor(log2(value))
  high <- 2 * low
  if (value / low < high / value) low else high
}

# Initial step, clamped
step_law <- function(n, k, nepochs, batch_size) {
  a <- .alpha_law
  alpha0 <- a[1] * k^a[2] * batch_size^a[3] * n^a[4] * nepochs^a[5]
  min(max(alpha0, 1e-4), 1)
}

# Rows per epoch
epoch_rows <- function(n, nepochs) {
  whole <- floor(nepochs)
  remainder <- round((nepochs - whole) * n)  # ties to even
  if (whole == 0) return(remainder)
  c(rep(n, whole - 1), if (remainder > 0) remainder, n)
}

# Resolve the batch size, step size and pass lengths for one training run.
emeans_recipe <- function(n, k, d, nepochs = 5,
                          batch_law = c("auto", "throughput", "quality"),
                          batch_size = NULL, alpha0 = NULL,
                          unused_threshold = NULL) {
  batch_law <- match.arg(batch_law)
  n <- as.numeric(n)
  k <- as.numeric(k)
  nepochs <- as.numeric(nepochs)
  stopifnot(n >= 1, k >= 1, d >= 1, nepochs > 0)
  # Resolve auto batch law
  law <- if (batch_law == "auto") {
    if (d <= .auto_quality_max_d) "quality" else "throughput"
  } else {
    batch_law
  }

  if (is.null(batch_size)) {
    # Batch size from law
    b <- .batch_laws[[law]]
    raw <- b[1] * k^b[2] * n^b[3] * nepochs^b[4]
    batch_size <- round_pow2(raw)
    batch_size <- if (law == "throughput") {
      min(batch_size, round_pow2(n / 2))
    } else {
      min(batch_size, n %/% 2)
    }
    batch_size <- max(1, min(batch_size, n))
  } else {
    # Manual batch size
    law <- NA_character_
    batch_size <- min(as.integer(batch_size), n)
    stopifnot(batch_size >= 1)
  }
  # Step size from law
  if (is.null(alpha0)) {
    alpha0 <- step_law(n, k, nepochs, batch_size)
  } else {
    stopifnot(alpha0 > 0, alpha0 <= 1)
  }
  # Default revival threshold
  if (is.null(unused_threshold)) {
    unused_threshold <- if (d <= .no_revival_max_d) 0 else .unused_threshold_default # nolint: line_length_linter.
  } else {
    stopifnot(unused_threshold >= 0)
  }
  # Rows and steps per epoch
  rows <- epoch_rows(n, nepochs)
  steps <- ceiling(rows / batch_size)
  if (sum(steps) == 0) {
    stop("nepochs too small: no rows to train on", call. = FALSE)
  }
  list(
    batch_size = batch_size,
    alpha0 = as.numeric(alpha0),
    epoch_rows = rows,
    epoch_steps = steps,
    total_steps = sum(steps),
    batch_law = law,
    unused_threshold = unused_threshold,
    nepochs = nepochs
  )
}
