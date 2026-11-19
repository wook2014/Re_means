// E-means training rules adapted from the Faiss reference and emeansR.
// Copyright notices and source references are in inst/COPYRIGHTS.
#include <Rcpp.h>
#include <faiss/utils/distances.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

using std::size_t;
constexpr size_t search_chunk = 8192;

// The one-time input conversion and the reusable batch buffer are row-major.
std::vector<float> pack_rows(const Rcpp::NumericMatrix& x) {
    const size_t n = x.nrow(), d = x.ncol();
    std::vector<float> result(n * d);
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < d; ++col) {
            const float value = static_cast<float>(x[row + col * n]);
            if (!std::isfinite(value)) {
                Rcpp::stop("Data must be finite in Faiss float32 precision.");
            }
            result[row * d + col] = value;
        }
    }
    return result;
}

class ThreadScope {
    const int previous = omp_get_max_threads();
public:
    explicit ThreadScope(int count) { omp_set_num_threads(count); }
    ~ThreadScope() { omp_set_num_threads(previous); }
};

// Preserve the deterministic seed streams used by the previous implementation.
class Random {
    uint64_t state[4];

    uint64_t next() {
        const uint64_t value = state[1] * 5;
        const uint64_t result = ((value << 7) | (value >> 57)) * 9;
        const uint64_t shifted = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= shifted;
        state[3] = (state[3] << 45) | (state[3] >> 19);
        return result;
    }

    uint32_t below(uint32_t bound) {
        const uint64_t threshold = (uint64_t{0} - bound) % bound;
        for (;;) {
            const uint64_t value = next();
            const uint64_t low_half = (value & 0xffffffffULL) * bound;
            const uint64_t high_half = (value >> 32) * bound;
            const uint64_t low = low_half + (high_half << 32);
            if (low >= threshold) {
                return static_cast<uint32_t>((high_half >> 32) + (low < low_half));
            }
        }
    }

public:
    explicit Random(uint64_t seed) {
        for (auto& word : state) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t value = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
            word = value ^ (value >> 31);
        }
    }

    double uniform() { return static_cast<double>(next() >> 11) * 0x1p-53; }

    void permute(std::vector<int>& order, size_t take) {
        std::iota(order.begin(), order.end(), 0);
        for (size_t i = 0; i < take; ++i) {
            const size_t j = i + below(static_cast<uint32_t>(order.size() - i));
            std::swap(order[i], order[j]);
        }
    }
};

double norm(const double* row, size_t d) {
    double largest = 0;
    for (size_t j = 0; j < d; ++j) largest = std::max(largest, std::abs(row[j]));
    if (largest == 0) return 1;
    double squares = 0;
    for (size_t j = 0; j < d; ++j) {
        const double scaled = row[j] / largest;
        squares += scaled * scaled;
    }
    return largest * std::sqrt(squares);
}

void normalize(double* row, size_t d) {
    const double scale = norm(row, d);
    for (size_t j = 0; j < d; ++j) row[j] /= scale;
}

void search_top1(const float* x, const float* centers, size_t d, size_t n,
                 size_t k, bool spherical, float* scores, int64_t* labels) {
    if (spherical) {
        faiss::knn_inner_product(x, centers, d, n, k, 1, scores, labels);
    } else {
        faiss::knn_L2sqr(x, centers, d, n, k, 1, scores, labels);
    }
    for (size_t i = 0; i < n; ++i) {
        if (labels[i] < 0 || static_cast<size_t>(labels[i]) >= k || !std::isfinite(scores[i])) {
            Rcpp::stop("Faiss could not assign a row. Check the scale of the data and centers.");
        }
    }
}

Rcpp::List assign_rows(const Rcpp::NumericMatrix& x, const Rcpp::NumericMatrix& centers,
                       const std::vector<float>& packed, bool spherical) {
    const size_t n = x.nrow(), d = x.ncol(), k = centers.nrow();
    Rcpp::IntegerVector cluster(n);
    Rcpp::NumericVector distance(n);
    const auto packed_centers = pack_rows(centers);
    std::vector<float> scores(std::min(n, search_chunk));
    std::vector<int64_t> labels(scores.size());
    for (size_t base = 0; base < n; base += search_chunk) {
        Rcpp::checkUserInterrupt();
        const size_t count = std::min(search_chunk, n - base);
        search_top1(packed.data() + base * d, packed_centers.data(), d, count, k,
                    spherical, scores.data(), labels.data());
        for (size_t i = 0; i < count; ++i) {
            const size_t row = base + i, center = labels[i];
            cluster[row] = static_cast<int>(center + 1);
            double score = 0;
            for (size_t col = 0; col < d; ++col) {
                const double value = x[row + col * n], target = centers[center + col * k];
                const double delta = value - target;
                score += spherical ? value * target : delta * delta;
            }
            distance[row] = score;
        }
    }
    return Rcpp::List::create(Rcpp::Named("cluster") = cluster,
                              Rcpp::Named("distance") = distance);
}

size_t revive(std::vector<double>& centers, std::vector<double>& usage,
              size_t d, double coefficient, Random& random) {
    const double threshold = coefficient *
        (std::accumulate(usage.begin(), usage.end(), 0.0) / usage.size());
    std::vector<size_t> targets, donors;
    for (size_t j = 0; j < usage.size(); ++j) {
        if (usage[j] < threshold) targets.push_back(j);
        else if (usage[j] > 0) donors.push_back(j);
    }
    const size_t splits = std::min(targets.size(), donors.size());
    if (splits == 0) return 0;

    // Gumbel top-k samples donors without replacement, weighted by EMA mass.
    std::vector<double> keys(usage.size());
    for (size_t j = 0; j < usage.size(); ++j) {
        keys[j] = std::max(random.uniform(), std::numeric_limits<double>::min());
    }
    for (const auto donor : donors) {
        keys[donor] = std::log(usage[donor]) - std::log(-std::log(keys[donor]));
    }
    std::partial_sort(donors.begin(), donors.begin() + splits, donors.end(),
        [&](size_t a, size_t b) { return keys[a] == keys[b] ? a < b : keys[a] > keys[b]; });

    for (size_t split = 0; split < splits; ++split) {
        const size_t src = donors[split], dst = targets[split];
        auto* donor = centers.data() + src * d;
        auto* child = centers.data() + dst * d;
        for (size_t col = 0; col < d; ++col) {
            const double offset = (col % 2 == 0) ? 1e-4 : -1e-4;
            child[col] = donor[col] * (1 + offset);
            donor[col] *= 1 - offset;
        }
        if (std::equal(donor, donor + d, child)) {
            const float value = static_cast<float>(child[0]);
            const float up = std::nextafter(value, std::numeric_limits<float>::infinity());
            child[0] = std::isfinite(up) ? up : std::nextafter(value, -std::numeric_limits<float>::infinity());
        }
        usage[src] /= 2;
        usage[dst] = usage[src];
    }
    return splits;
}

}  // namespace

// [[Rcpp::export(rng = false)]]
Rcpp::IntegerVector em_sample(int n, int take, double seed) {
    std::vector<int> order(n);
    Random random(static_cast<uint64_t>(static_cast<int64_t>(seed)));
    random.permute(order, take);
    Rcpp::IntegerVector result(take);
    for (int i = 0; i < take; ++i) result[i] = order[i] + 1;
    return result;
}

// [[Rcpp::export(rng = false)]]
double em_totss(const Rcpp::NumericMatrix& x) {
    const size_t n = x.nrow();
    if (n == 0) return 0;
    double result = 0;
    for (int j = 0; j < x.ncol(); ++j) {
        const auto* col = x.begin() + j * n;
        const double mean = std::accumulate(col, col + n, 0.0) / n;
        for (size_t i = 0; i < n; ++i) {
            const double delta = col[i] - mean;
            result += delta * delta;
        }
    }
    return result;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List em_assign(const Rcpp::NumericMatrix& centers, const Rcpp::NumericMatrix& x,
                     bool spherical, int n_threads) {
    if (x.ncol() != centers.ncol() || centers.ncol() == 0 || centers.nrow() == 0) {
        Rcpp::stop("newdata and centers must have the same positive number of columns and at least one center.");
    }
    const ThreadScope threads(n_threads);
    return assign_rows(x, centers, pack_rows(x), spherical);
}

// [[Rcpp::export(rng = false)]]
Rcpp::List em_train(const Rcpp::NumericMatrix& x, const Rcpp::NumericMatrix& initial,
                    const Rcpp::List& recipe, bool spherical, bool single, double seed_value,
                    int n_threads, bool capture_last, bool assign_final, bool verbose) {
    const size_t n = x.nrow(), d = x.ncol(), k = initial.nrow();
    const size_t batch_size = Rcpp::as<size_t>(recipe["batch_size"]);
    const double alpha0 = recipe["alpha0"], total_steps = recipe["total_steps"];
    const double unused_threshold = recipe["unused_threshold"];
    const Rcpp::NumericVector epoch_rows = recipe["epoch_rows"];
    const auto seed = static_cast<uint64_t>(static_cast<int64_t>(seed_value));
    const ThreadScope threads(n_threads);
    const auto packed = pack_rows(x);

    std::vector<double> centers(k * d), usage(k, 0x1p-16);
    for (size_t j = 0; j < k; ++j) {
        for (size_t col = 0; col < d; ++col) centers[j * d + col] = initial[j + col * k];
        if (spherical) normalize(centers.data() + j * d, d);
    }
    for (double value : centers) {
        if (!std::isfinite(static_cast<float>(value))) {
            Rcpp::stop("Initial centers must be finite in Faiss float32 precision.");
        }
    }
    const size_t work_rows = std::min(batch_size, search_chunk);
    std::vector<float> batch(work_rows * d), snapshot(k * d), scores(work_rows);
    std::vector<int64_t> labels(work_rows);
    std::vector<int> order(n);
    std::vector<double> batch_sums(k * d), closing_sums(k * d), counts(k), pass_counts(k);
    Rcpp::IntegerVector cluster(capture_last ? n : 0, NA_INTEGER);
    Rcpp::NumericVector objective(epoch_rows.size()), imbalance(epoch_rows.size()), splits(epoch_rows.size());
    Random split_random(seed);
    size_t step = 0, n_splits = 0;

    for (R_xlen_t epoch = 0; epoch < epoch_rows.size(); ++epoch) {
        const size_t rows = static_cast<size_t>(epoch_rows[epoch]);
        const bool closing = epoch + 1 == epoch_rows.size();
        const size_t splits_before = n_splits;
        Random shuffle((seed * 1000003ULL + static_cast<uint64_t>(epoch)) & 0xffffffffULL);
        shuffle.permute(order, rows);
        std::fill(pass_counts.begin(), pass_counts.end(), 0);
        double loss = 0;

        for (size_t start = 0; start < rows; start += batch_size, ++step) {
            const size_t end = std::min(start + batch_size, rows);
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(batch_sums.begin(), batch_sums.end(), 0);
            for (size_t j = 0; j < k; ++j) {
                const auto* center = centers.data() + j * d;
                const double scale = spherical ? norm(center, d) : 1;
                for (size_t col = 0; col < d; ++col) {
                    snapshot[j * d + col] = static_cast<float>(center[col] / scale);
                }
            }
            // A chunk bounds scratch size. Every chunk of this batch uses the
            // same snapshot and contributes to one EMA update.
            for (size_t base = start; base < end; base += work_rows) {
                Rcpp::checkUserInterrupt();
                const size_t count = std::min(work_rows, end - base);
                for (size_t i = 0; i < count; ++i) {
                    const auto* row = packed.data() + static_cast<size_t>(order[base + i]) * d;
                    std::copy_n(row, d, batch.data() + i * d);
                }
                search_top1(batch.data(), snapshot.data(), d, count, k, spherical,
                            scores.data(), labels.data());
                for (size_t i = 0; i < count; ++i) {
                    const size_t row = order[base + i], j = labels[i];
                    ++counts[j];
                    for (size_t col = 0; col < d; ++col) {
                        const double value = single ? batch[i * d + col] : x[row + col * n];
                        batch_sums[j * d + col] += value;
                    }
                    loss += spherical ? 2 - 2.0 * scores[i] : scores[i];
                    if (closing && capture_last) cluster[row] = static_cast<int>(j + 1);
                }
            }
            const double cosine = 0.5 * (1 + std::cos(std::acos(-1.0) * step / total_steps));
            const double alpha = alpha0 * (0.1 + 0.9 * cosine);
            for (size_t j = 0; j < k; ++j) {
                const double count = counts[j];
                pass_counts[j] += count;
                if (alpha != 1) usage[j] *= 1 - alpha;
                if (count == 0) continue;
                usage[j] = (alpha == 1 ? 0 : usage[j]) + alpha * count;
                const double weight = alpha * count / usage[j];
                for (size_t col = 0; col < d; ++col) {
                    const size_t index = j * d + col;
                    const double sum = batch_sums[index], mean = sum / count;
                    if (closing) closing_sums[index] += sum;
                    centers[index] = weight == 1 ? mean : centers[index] + weight * (mean - centers[index]);
                }
            }
            if (unused_threshold > 0 && (step + 1) % 5 == 0) {
                n_splits += revive(centers, usage, d, unused_threshold, split_random);
            }
        }
        objective[epoch] = loss;
        imbalance[epoch] = k * std::inner_product(pass_counts.begin(), pass_counts.end(),
            pass_counts.begin(), 0.0) / (static_cast<double>(rows) * rows);
        splits[epoch] = n_splits - splits_before;
        if (verbose) {
            Rcpp::Rcout << "  Epoch " << (epoch + 1) << " / " << epoch_rows.size()
                        << ": objective=" << loss << ", splits=" << splits[epoch] << '\n';
        }
    }

    Rcpp::NumericMatrix final_centers(k, d);
    for (size_t j = 0; j < k; ++j) {
        auto* center = centers.data() + j * d;
        if (pass_counts[j] > 0) {
            for (size_t col = 0; col < d; ++col) center[col] = closing_sums[j * d + col] / pass_counts[j];
        }
        if (spherical) normalize(center, d);
        for (size_t col = 0; col < d; ++col) final_centers(j, col) = center[col];
    }
    Rcpp::List assignment;
    if (assign_final) assignment = assign_rows(x, final_centers, packed, spherical);
    return Rcpp::List::create(Rcpp::Named("centers") = final_centers,
                              Rcpp::Named("cluster") = cluster,
                              Rcpp::Named("assignment") = assignment,
                              Rcpp::Named("obj") = objective,
                              Rcpp::Named("imbalance") = imbalance,
                              Rcpp::Named("nsplit") = splits,
                              Rcpp::Named("n_splits") = static_cast<double>(n_splits));
}
