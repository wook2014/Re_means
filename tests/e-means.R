library(Remeans)

x <- as.matrix(datasets::iris[, 1:4])
initial <- x[c(1, 51, 101), , drop = FALSE]
set.seed(17)
rng_before <- .Random.seed
fit <- e_means(x, initial, nepochs = 2.5, n_threads = 1)
last_pass <- e_means(x, initial, nepochs = 2.5, final_assign = FALSE,
                     n_threads = 2)

# Closing means come from last-pass labels. The default labels use those
# centers for a fresh assignment.
closing <- rowsum(x, last_pass$cluster) / as.numeric(table(last_pass$cluster))
distance <- vapply(seq_len(nrow(fit$centers)), function(j) {
  rowSums(sweep(x, 2L, fit$centers[j, ], "-")^2)
}, numeric(nrow(x)))
nearest <- max.col(-distance, ties.method = "first")

stopifnot(
  identical(.Random.seed, rng_before),
  inherits(fit, "kmeans"),
  identical(colnames(fit$centers), colnames(x)),
  identical(fit$recipe$epoch_rows, c(150, 75, 150)),
  identical(fit$centers, last_pass$centers),
  isTRUE(all.equal(unname(last_pass$centers), unname(closing))),
  identical(unname(fit$cluster), nearest),
  identical(predict(fit, x, n_threads = 1), nearest),
  identical(fitted(fit, method = "classes"), fit$cluster),
  isTRUE(all.equal(fit$tot.withinss, sum(apply(distance, 1L, min))))
)

partial <- e_means(x, 3, nepochs = 0.5, final_assign = FALSE, n_threads = 1)
stopifnot(sum(is.na(partial$cluster)) == 75L, sum(partial$size) == 75L)
