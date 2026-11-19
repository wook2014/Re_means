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

# Known top-1 results through the same Faiss call used by training.
query <- rbind(c(0, 0), c(3, 4), c(-2, 1))
targets <- rbind(c(0, 1), c(4, 4), c(-3, 0))
exact <- Remeans:::em_assign(targets, query, FALSE, 1L)
stopifnot(identical(exact$cluster, 1:3), identical(exact$distance, c(1, 1, 2)))

# More than 20 queries exercises the Faiss 1.11 CPU BLAS path.
set.seed(4321)
query <- matrix(rnorm(128L * 8L), ncol = 8L)
targets <- matrix(rnorm(17L * 8L), ncol = 8L)
reference <- vapply(seq_len(nrow(targets)), function(j) {
  rowSums(sweep(query, 2L, targets[j, ], "-")^2)
}, numeric(nrow(query)))
assigned <- Remeans:::em_assign(targets, query, FALSE, 2L)
stopifnot(
  identical(assigned$cluster, max.col(-reference, ties.method = "first")),
  isTRUE(all.equal(assigned$distance, apply(reference, 1L, min)))
)

sampled <- e_means(x, initial, max_points_per_centroid = 20, n_threads = 1)
sampled_last <- e_means(x, initial, max_points_per_centroid = 20,
                        final_assign = FALSE, n_threads = 1)
stopifnot(all(!is.na(sampled$cluster)), sum(!is.na(sampled_last$cluster)) == 60L,
          identical(sampled$centers, sampled_last$centers))

unit <- query / sqrt(rowSums(query^2))
sphere <- e_means(unit, 4, nepochs = 1.5, spherical = TRUE, precision = "single",
                  n_threads = 1)
similarity <- unit %*% t(sphere$centers)
spherical_labels <- predict(sphere, unit, distance = TRUE, n_threads = 1)
stopifnot(
  isTRUE(all.equal(rowSums(sphere$centers^2), rep(1, 4L))),
  identical(spherical_labels$cluster, max.col(similarity, ties.method = "first")),
  isTRUE(all.equal(spherical_labels$distance, apply(similarity, 1L, max)))
)
