/// @file rolling_corr_matrix.cpp
/// Rolling correlation matrix across the columns of a matrix.
///
/// The p(p-1)/2 upper-triangle pairs are computed independently, each by the
/// pairwise rolling_corr kernel writing its own contiguous output column —
/// no shared state, no locks.  OpenMP parallelises over pairs with
/// schedule(dynamic, 4).  The flat pair index is decoded via a precomputed
/// lookup table (p ≤ 50 → at most 1225 entries).
#include "fastwindow.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace fastwindow {

static int resolve_threads(int n_threads) {
#ifdef _OPENMP
    return (n_threads > 0) ? n_threads : omp_get_max_threads();
#else
    (void)n_threads;
    return 1;
#endif
}

void rolling_corr_matrix(
        const double* X, double* dst,
        size_t n, int p, size_t window, int min_periods, int n_threads) {
    if (n == 0 || window == 0 || p < 2 || p > FW_MAX_CORR_COLS) return;

    const int n_pairs = p * (p - 1) / 2;

    //Lookup table: flat pair index → (i, j), i < j
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int i = 0, k = 0; i < p - 1; i++)
        for (int j = i + 1; j < p; j++, k++) {
            pi[k] = i;
            pj[k] = j;
        }

    const int nt = resolve_threads(n_threads);
    (void)nt;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4) num_threads(nt)
#endif
    for (int pair = 0; pair < n_pairs; pair++) {
        rolling_corr(X + static_cast<size_t>(pi[pair]) * n,
                     X + static_cast<size_t>(pj[pair]) * n,
                     dst + static_cast<size_t>(pair) * n,
                     nullptr,
                     n, window, min_periods, /*skip_nan=*/false);
    }
}

void rolling_corr_matrix_full(
        const double* X, double* out, size_t n, int p, size_t window,
        int min_periods, bool r_layout, int n_threads) {
    if (n == 0 || window == 0 || p < 2 || p > FW_MAX_CORR_COLS) return;

    const int n_pairs = p * (p - 1) / 2;
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int i = 0, k = 0; i < p - 1; i++)
        for (int j = i + 1; j < p; j++, k++) {
            pi[k] = i;
            pj[k] = j;
        }

    const int nt = resolve_threads(n_threads);
    (void)nt;
    const size_t pp = static_cast<size_t>(p) * p;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int d = 0; d < p; d++) {
        if (r_layout) {
            double* diag = out + static_cast<size_t>(d) * n * p
                               + static_cast<size_t>(d) * n;
            for (size_t t = 0; t < n; t++) diag[t] = 1.0;
        } else {
            for (size_t t = 0; t < n; t++) out[t * pp + d * p + d] = 1.0;
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4) num_threads(nt)
#endif
    for (int pair = 0; pair < n_pairs; pair++) {
        const int i = pi[pair], j = pj[pair];
        if (r_layout) {
            double* upper = out + static_cast<size_t>(j) * n * p
                                + static_cast<size_t>(i) * n;
            double* lower = out + static_cast<size_t>(i) * n * p
                                + static_cast<size_t>(j) * n;
            rolling_corr(X + static_cast<size_t>(i) * n,
                         X + static_cast<size_t>(j) * n, upper, nullptr,
                         n, window, min_periods, false);
            std::copy(upper, upper + n, lower);
        } else {
            std::vector<double> values(n);
            rolling_corr(X + static_cast<size_t>(i) * n,
                         X + static_cast<size_t>(j) * n, values.data(), nullptr,
                         n, window, min_periods, false);
            for (size_t t = 0; t < n; t++) {
                out[t * pp + i * p + j] = values[t];
                out[t * pp + j * p + i] = values[t];
            }
        }
    }
}

void corr_matrix_expand(
        const double* tri, double* out,
        size_t n, int p, bool r_layout, int n_threads) {
    const int n_pairs = p * (p - 1) / 2;

    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int i = 0, k = 0; i < p - 1; i++)
        for (int j = i + 1; j < p; j++, k++) {
            pi[k] = i;
            pj[k] = j;
        }

    const int nt = resolve_threads(n_threads);
    (void)nt;

    const size_t pp = static_cast<size_t>(p) * p;
    if (!r_layout) {
        //C-order (n, p, p): out[t·p² + i·p + j].  Processing t sequentially
        //keeps the n_pairs strided tri streams cache-line-resident.
        //(Non-temporal stores were tried here and measured slower: the
        //800-byte per-slice runs thrash the write-combining buffers.)
        //signed loop index: MSVC only implements OpenMP 2.0, which rejects
        //unsigned induction variables (C3016)
        const long long ns = static_cast<long long>(n);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(nt)
#endif
        for (long long ts = 0; ts < ns; ts++) {
            const size_t t = static_cast<size_t>(ts);
            double* FW_RESTRICT o = out + t * pp;
            for (int d = 0; d < p; d++) o[d * p + d] = 1.0;
            for (int k = 0; k < n_pairs; k++) {
                double v = tri[static_cast<size_t>(k) * n + t];
                o[pi[k] * p + pj[k]] = v;
                o[pj[k] * p + pi[k]] = v;
            }
        }
    } else {
        //R column-major dim c(n, p, p): out[t + i·n + j·n·p].
        //The (i,j) planes are n-contiguous, so copy per pair, not per t.
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(nt)
#endif
        for (int k = -p; k < n_pairs; k++) {
            if (k < 0) {           //negative indices fill the unit diagonal
                int d = k + p;
                double* FW_RESTRICT o = out +
                    static_cast<size_t>(d) * n * p + static_cast<size_t>(d) * n;
                for (size_t t = 0; t < n; t++) o[t] = 1.0;
            } else {
                const double* FW_RESTRICT src = tri + static_cast<size_t>(k) * n;
                double* FW_RESTRICT up = out +
                    static_cast<size_t>(pj[k]) * n * p +
                    static_cast<size_t>(pi[k]) * n;
                double* FW_RESTRICT lo = out +
                    static_cast<size_t>(pi[k]) * n * p +
                    static_cast<size_t>(pj[k]) * n;
                for (size_t t = 0; t < n; t++) {
                    up[t] = src[t];
                    lo[t] = src[t];
                }
            }
        }
    }
}

} //namespace fastwindow
