/// @file rolling_matrix.cpp
/// Multi-column dispatch: column-wise rolling statistics over a matrix.
/// Each column is fully independent, so OpenMP parallelises over columns
/// with per-thread stack state — no sharing, no locks.
#include "fastwindow.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace fastwindow {

static int mat_threads(int n_threads) {
#ifdef _OPENMP
    return (n_threads > 0) ? n_threads : omp_get_max_threads();
#else
    (void)n_threads;
    return 1;
#endif
}

void rolling_mean_matrix(const double* X, double* dst, size_t n, int p,
                         size_t window, int min_periods, int n_threads) {
    const int nt = mat_threads(n_threads);
    (void)nt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int c = 0; c < p; c++)
        rolling_mean(X + (size_t)c * n, dst + (size_t)c * n,
                     n, window, min_periods, false);
}

void rolling_std_matrix(const double* X, double* dst, size_t n, int p,
                        size_t window, int min_periods, bool ddof1,
                        int n_threads) {
    const int nt = mat_threads(n_threads);
    (void)nt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int c = 0; c < p; c++)
        rolling_std(X + (size_t)c * n, dst + (size_t)c * n,
                    n, window, min_periods, ddof1, false);
}

void rolling_sum_matrix(const double* X, double* dst, size_t n, int p,
                        size_t window, int min_periods, int n_threads) {
    const int nt = mat_threads(n_threads);
    (void)nt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int c = 0; c < p; c++)
        rolling_sum(X + (size_t)c * n, dst + (size_t)c * n,
                    n, window, min_periods, false);
}

void rolling_min_matrix(const double* X, double* dst, size_t n, int p,
                        size_t window, int min_periods, bool skip_nan,
                        int n_threads) {
    const int nt = mat_threads(n_threads);
    (void)nt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int c = 0; c < p; c++)
        rolling_min(X + (size_t)c * n, dst + (size_t)c * n, n, window,
                    min_periods, skip_nan);
}

void rolling_max_matrix(const double* X, double* dst, size_t n, int p,
                        size_t window, int min_periods, bool skip_nan,
                        int n_threads) {
    const int nt = mat_threads(n_threads);
    (void)nt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int c = 0; c < p; c++)
        rolling_max(X + (size_t)c * n, dst + (size_t)c * n, n, window,
                    min_periods, skip_nan);
}

} //namespace fastwindow
