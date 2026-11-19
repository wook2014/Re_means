# Run from the repository root with Rscript tools/benchmark.R.
library(Remeans)

set.seed(2026)
n <- 50000L
d <- 32L
k <- 128L
population <- matrix(rnorm(k * d, sd = 1.6), k, d)
simulate_rows <- function(n) {
  population[sample.int(k, n, replace = TRUE), , drop = FALSE] +
    matrix(rnorm(n * d), n, d)
}
x <- simulate_rows(n)
test <- simulate_rows(2000L)

test_mse <- function(centers) {
  nearest <- rep(Inf, nrow(test))
  for (i in seq_len(nrow(centers))) {
    distance <- rowSums(sweep(test, 2L, centers[i, ], "-")^2)
    nearest <- pmin(nearest, distance)
  }
  mean(nearest)
}

run_benchmark <- function(seed) {
  set.seed(seed)
  initial <- x[sample.int(n, k), , drop = FALSE]
  methods <- list(
    "e-means (3 epochs, 1 thread)" = function() {
      e_means(x, initial, nepochs = 3, seed = seed, n_threads = 1L)
    },
    "e-means (5 epochs, 1 thread)" = function() {
      e_means(x, initial, seed = seed, n_threads = 1L)
    },
    "Lloyd (up to 25 iterations)" = function() {
      stats::kmeans(x, initial, iter.max = 25, algorithm = "Lloyd")
    },
    "Hartigan-Wong (up to 25 iterations)" = function() {
      stats::kmeans(x, initial, iter.max = 25, algorithm = "Hartigan-Wong")
    }
  )

  do.call(rbind, lapply(names(methods), function(method) {
    seconds <- system.time(fit <- methods[[method]]())[["elapsed"]]
    ifault <- if (is.null(fit$ifault)) NA_integer_ else fit$ifault
    data.frame(method, seed, seconds, test_mse = test_mse(fit$centers),
               iter = fit$iter, ifault)
  }))
}

# Load the backend before timing the fits.
invisible(e_means(x[1:100, ], 2L, nepochs = 1, n_threads = 1L))
runs <- do.call(rbind, lapply(1:3, run_benchmark))
print(runs, row.names = FALSE)
cat("\nMedian across seeds\n")
print(aggregate(cbind(seconds, test_mse) ~ method, runs, median), row.names = FALSE)
