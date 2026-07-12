/// @file fastwindow.h
/// Master header for the fastwindow C++17 core.
/// All kernel function declarations live here; implementations in rolling_basic.cpp.
#pragma once

#include <vector>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>

//MSVC spells the non-standard restrict qualifier without trailing underscores.
#if defined(_MSC_VER)
  #define FW_RESTRICT __restrict
#else
  #define FW_RESTRICT __restrict__
#endif

//AVX2 kernels are compiled on any x86-64 target and selected at RUNTIME via
//cpu_has_avx2(): when the baseline ISA lacks AVX2 (portable wheels, CRAN
//builds) the intrinsics are enabled per-function through FW_TARGET_AVX2
//target attributes (MSVC compiles intrinsics without any flag, so the
//attribute is empty there).  -DFASTWINDOW_NO_AVX2 forces the scalar
//fallback at compile time (CI flag).
#if !defined(FASTWINDOW_NO_AVX2) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__AVX2__))
  #include <immintrin.h>
  #define FW_SIMD 1
  #if defined(__AVX2__) || defined(_MSC_VER)
    #define FW_TARGET_AVX2
  #else
    #define FW_TARGET_AVX2 __attribute__((target("avx2,fma")))
  #endif
#else
  #define FW_SIMD 0
#endif

namespace fastwindow {

/// True iff the CPU and OS support the AVX2 kernel paths (AVX2 + FMA
/// instruction sets and OS-saved YMM state).  Result is cached; always
/// false in FASTWINDOW_NO_AVX2 / non-x86-64 builds.
bool cpu_has_avx2() noexcept;

//NaN-safe finite check
//std::isfinite / std::isnan are optimised away by -ffast-math / -ffinite-math-only.
//Use a bit-cast approach instead; std::memcpy is a pure memory op, unaffected.

inline bool fw_isfinite(double x) noexcept {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    //Exponent bits all-1 → Inf or NaN
    return (u & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

inline bool fw_isfinite(float x) noexcept {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return (u & 0x7F800000U) != 0x7F800000U;
}

//Data structures

/// Cache-efficient circular buffer for fixed-capacity rolling windows.
template <typename T>
struct RollingBuffer {
    std::vector<T> data;
    size_t head     = 0;   ///< index of oldest element
    size_t count    = 0;   ///< number of elements currently stored
    size_t capacity;       ///< maximum capacity (= window size)

    explicit RollingBuffer(size_t cap)
        : data(cap, T(0)), capacity(cap) {}

    /// Push a new value; overwrites oldest element when full.
    void push(T val);
    /// Return the oldest element (about to be evicted on next push).
    T oldest() const { return data[head]; }
    /// Return the newest element.
    T newest() const { return data[(head + count - 1) % capacity]; }
    bool full()  const { return count == capacity; }
};

/// Monotonic deque for O(1) amortised rolling min/max.
/// Stores indices into the source array, not values.
struct MonotonicDeque {
    std::deque<size_t> idx;

    /// Maintain ascending order (for rolling_min).
    void push_min(size_t i, const double* src, size_t window);
    /// Maintain descending order (for rolling_max).
    void push_max(size_t i, const double* src, size_t window);
    size_t front() const { return idx.front(); }
    bool   empty() const { return idx.empty(); }
};

/// Welford online algorithm adapted for sliding fixed-size windows.
/// Tracks mean and M2 (sum of squared deviations) over valid (finite) values.
template <typename T>
struct RunningStats {
    T      mean   = T(0);
    T      M2     = T(0);
    size_t n      = 0;       ///< count of finite values currently included
    size_t window;           ///< fixed window capacity

    explicit RunningStats(size_t w) : window(w) {}

    /// Standard (growing) Welford add — use during warmup.
    void grow(T x_new);
    /// Sliding Welford — use once window is full; both x_new and x_old are finite.
    void slide(T x_new, T x_old);
    /// Reverse Welford — remove one finite element.
    void remove(T x_old);
    /// Reset to empty state.
    void reset() { mean = T(0); M2 = T(0); n = 0; }

    T variance(bool ddof1) const;
    T stddev(bool ddof1)   const;
};

/// Running OLS sums for simple linear regression (k=1) over a sliding window.
/// Maintains Σx, Σy, Σx², Σxy, Σyy; all updates O(1).
struct RunningRegression {
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    double sum_yy = 0;   ///< needed for R²
    size_t n = 0;        ///< number of points currently included

    /// Add one point during warmup (window not yet full).
    void grow(double x, double y);
    /// Sliding update: add (x,y), evict (x_old,y_old).  Window stays size n.
    void add(double x, double y, double x_old, double y_old);
    /// Reset to the empty state.
    void reset();
    /// OLS coefficients; writes NaN if the design is degenerate (denom ≈ 0).
    void coeffs(double& b0, double& b1) const;
    /// Coefficient of determination; NaN when y is constant or n < 2.
    double r_squared() const;
};

/// Rolling OLS with k regressors plus intercept.  Maintains (X'X)⁻¹
/// incrementally via Sherman-Morrison rank-1 updates; X'X itself is kept
/// exactly (cheap rank-1 sums) so the inverse can always be rebuilt by
/// Cholesky when an update is rejected as near-singular.
struct RunningMultipleRegression {
    int k;                         ///< number of regressors INCLUDING intercept
    std::vector<double> XtX;       ///< k×k, row-major, maintained exactly
    std::vector<double> XtX_inv;   ///< k×k, row-major, Sherman-Morrison updated
    std::vector<double> Xty;       ///< k×1
    double yty   = 0;
    double sum_y = 0;
    size_t n     = 0;              ///< rows currently in the window
    bool inv_valid = false;        ///< false ⇒ XtX_inv is stale, refresh needed

    explicit RunningMultipleRegression(int k_incl_intercept);

    /// Reset all sums to the empty state.
    void reset();
    /// Rank-1 add of row u (length k, u[0]=1 for the intercept) with target y.
    void add_row(const double* u, double y);
    /// Rank-1 removal of a row previously added.
    void remove_row(const double* u, double y);
    /// Combined sliding update per spec: add (x_new,y_new), evict (x_old,y_old).
    void add(const double* x_new, double y_new,
             const double* x_old, double y_old);
    /// Rebuild XtX_inv from XtX via Cholesky; returns false if singular.
    bool refresh_inverse();
    /// β = (X'X)⁻¹ X'y; writes k doubles.  Requires inv_valid.
    void coeffs(double* beta_out) const;
    /// Coefficient of determination for the current window.
    double r_squared() const;
    /// Residual standard error, sqrt(SSR / (n - k)).
    double residual_std() const;
};

//Basic statistics kernels

/// Rolling mean over a sliding window.
/// @param src         Input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid observations required to emit a value;
///                    positions with fewer get NaN
/// @param skip_nan    If true, non-finite values are excluded from the
///                    window count; if false they propagate as NaN output
/// @param n_threads   OpenMP threads over blocks in AVX2 builds; 0 (the
///                    default) means single-threaded
/// @note Running sums are recomputed every 4096 steps to suppress
///       floating-point drift (AVX2 builds use exact block-local sums).
void rolling_mean(const double* src, double* dst, size_t n,
                  size_t window, int min_periods, bool skip_nan,
                  int n_threads = 0);

/// Rolling variance over a sliding window (sliding Welford).
/// @param src         Input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid observations required to emit a value
/// @param ddof1       true → sample estimator (n-1), false → population (n)
/// @param skip_nan    If true, non-finite values are excluded; else NaN out
void rolling_var(const double* src, double* dst, size_t n,
                 size_t window, int min_periods, bool ddof1, bool skip_nan,
                 int n_threads = 0);

/// Rolling standard deviation; sqrt of rolling_var with the same semantics.
/// @copydetails rolling_var
void rolling_std(const double* src, double* dst, size_t n,
                 size_t window, int min_periods, bool ddof1, bool skip_nan,
                 int n_threads = 0);

/// Rolling sum over a sliding window (Kahan-compensated recomputation).
/// @param src         Input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid observations required to emit a value
/// @param skip_nan    If true, non-finite values are excluded; else NaN out
void rolling_sum(const double* src, double* dst, size_t n,
                 size_t window, int min_periods, bool skip_nan,
                 int n_threads = 0);

/// Rolling minimum.
/// @param src         Input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid observations required to emit a value
/// @param skip_nan    If true, non-finite values are excluded from the
///                    window (pandas semantics); if false any non-finite
///                    value inside the window yields NaN for that position
/// @note AVX2 builds use the van Herk blocked algorithm for window ≥ 16
///       (skip_nan=false); otherwise a monotonic deque.
void rolling_min(const double* src, double* dst, size_t n, size_t window,
                 int min_periods, bool skip_nan, int n_threads = 0);

/// Rolling maximum; mirror of rolling_min.
/// @copydetails rolling_min
void rolling_max(const double* src, double* dst, size_t n, size_t window,
                 int min_periods, bool skip_nan, int n_threads = 0);

//Simple linear regression kernels

/// Rolling simple OLS regression of y on time (x = 0,1,…,window-1 within
/// each window); Σx and Σx² are closed-form constants once the window fills.
/// @param y           Dependent variable, length n
/// @param b0          Output intercepts, length n
/// @param b1          Output slopes, length n
/// @param r2          Optional output R², length n; nullptr to skip
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum observations required to emit a value
/// @param time_as_x   Must be true; for an explicit regressor use the
///                    _xy variant
void rolling_simple_regression(
    const double* y, double* b0, double* b1, double* r2,
    size_t n, size_t window, int min_periods, bool time_as_x);

/// Rolling simple OLS regression of y on an explicit regressor series x.
/// @param x           Regressor series, length n
/// @param y           Dependent variable, length n
/// @param b0          Output intercepts, length n
/// @param b1          Output slopes, length n
/// @param r2          Optional output R², length n; nullptr to skip
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid pairs required to emit a value
void rolling_simple_regression_xy(
    const double* x, const double* y,
    double* b0, double* b1, double* r2,
    size_t n, size_t window, int min_periods);

/// Pearson correlation / covariance between two series over a sliding
/// window.  Maintains the five running sums Σx, Σy, Σx², Σy², Σxy over the
/// valid (pairwise-finite) observations; all updates O(1).
struct RunningCorrelation {
    double sum_x = 0, sum_y = 0;
    double sum_xx = 0, sum_yy = 0, sum_xy = 0;
    size_t n = 0;   ///< count of valid pairs currently included

    /// Add one valid pair.
    void grow(double x, double y);
    /// Remove one previously added pair.
    void remove(double x, double y);
    /// Combined sliding update per spec: add (x_new,y_new), evict (x_old,y_old).
    void add(double x_new, double y_new, double x_old, double y_old);
    /// Reset to the empty state.
    void reset();
    /// Pearson correlation; NaN when n < 2 or either series is constant.
    double corr() const;
    /// Covariance (ddof1=true → sample, n-1 denominator).
    double cov(bool ddof1) const;
};

//Multiple regression kernel

/// Maximum number of user-supplied regressors (excluding the intercept).
constexpr int FW_MAX_REGRESSORS = 16;

/// Rolling OLS with k regressors; the intercept is added internally
/// (the user passes k regressor columns, receives k+1 coefficients).
/// @param Y           Dependent variable, length n
/// @param X           Regressors, n×k column-major, NO intercept column
/// @param beta        Output coefficients, n×(k+1) column-major,
///                    intercept first
/// @param r2          Optional output R², length n; nullptr to skip
/// @param res_std     Optional output residual standard error, length n;
///                    nullptr to skip
/// @param n           Number of observations
/// @param k           Number of regressors, in [1, FW_MAX_REGRESSORS]
/// @param window      Window length (> 0)
/// @param min_periods Minimum observations required to emit a value
/// @note (X'X)⁻¹ is maintained by Sherman-Morrison rank-1 updates; a
///       Woodbury denominator below 1e-12 triggers a Cholesky rebuild for
///       that step only.  Full recompute every 4096 steps.
void rolling_multiple_regression(
    const double* Y, const double* X,
    double* beta, double* r2, double* res_std,
    size_t n, int k, size_t window, int min_periods);

//Correlation and covariance kernels

/// Rolling Pearson correlation between two series.
/// @param x           First input array of length n
/// @param y           Second input array of length n
/// @param dst_corr    Output correlations, length n
/// @param dst_cov     Optional output sample covariances (n-1 denominator),
///                    length n; pass nullptr to skip
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid pairs required to emit a value
/// @param skip_nan    If true, pairs with a non-finite member are excluded;
///                    if false they propagate as NaN output
/// @note Zero variance in either series within a window yields NaN, never ±Inf.
void rolling_corr(
    const double* x, const double* y,
    double* dst_corr, double* dst_cov,
    size_t n, size_t window, int min_periods, bool skip_nan);

/// Rolling covariance between two series.
/// @param x           First input array of length n
/// @param y           Second input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid pairs required to emit a value
/// @param ddof1       true → sample estimator (n-1), false → population (n)
/// @param skip_nan    If true, invalid pairs are excluded; else NaN output
void rolling_cov(
    const double* x, const double* y, double* dst,
    size_t n, size_t window, int min_periods, bool ddof1, bool skip_nan);

//Correlation matrix kernels (OpenMP over column pairs)

/// Practical upper limit on the number of columns for the corr matrix.
constexpr int FW_MAX_CORR_COLS = 50;

/// Rolling Pearson correlation matrix across the columns of X.
/// @param X           Input matrix, n×p column-major (2 ≤ p ≤ FW_MAX_CORR_COLS)
/// @param dst         Output: the p(p-1)/2 upper-triangle pair series,
///                    column-major dst[pair·n + t], pairs ordered
///                    (0,1),(0,2),…,(1,2),…
/// @param n           Number of rows
/// @param p           Number of columns
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid pairs required to emit a value
/// @param n_threads   OpenMP threads over pairs; 0 = library default
void rolling_corr_matrix(
    const double* X, double* dst,
    size_t n, int p, size_t window, int min_periods, int n_threads);

/// Expand the pair-major triangle into full symmetric matrices with unit
/// diagonal.  r_layout=false → C-order (n,p,p): out[t·p² + i·p + j];
/// r_layout=true → R column-major dim c(n,p,p): out[t + i·n + j·n·p].
void corr_matrix_expand(
    const double* tri, double* out,
    size_t n, int p, bool r_layout, int n_threads);

//Higher moments and z-score

/// Rolling skewness, bias-corrected (matches pandas .rolling().skew() and
/// scipy.stats.skew(bias=False)).  Requires ≥ 3 valid observations;
/// zero-variance windows give NaN.
/// @param src         Input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid observations required to emit a value
/// @param skip_nan    If true, non-finite values are excluded; else NaN out
void rolling_skew(const double* src, double* dst, size_t n, size_t window,
                  int min_periods, bool skip_nan);

/// Rolling EXCESS kurtosis, bias-corrected (matches pandas
/// .rolling().kurt() and scipy.stats.kurtosis(bias=False)).  Requires ≥ 4
/// valid observations; zero-variance windows give NaN.
/// @copydetails rolling_skew
void rolling_kurt(const double* src, double* dst, size_t n, size_t window,
                  int min_periods, bool skip_nan);

/// Rolling z-score: (x[i] − window mean) / window stddev, NaN when the
/// input is non-finite, the window doesn't emit, or the stddev is 0.
/// @param src         Input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid observations required to emit a value
/// @param ddof1       true → sample stddev (n-1), false → population
/// @param skip_nan    If true, non-finite values are excluded; else NaN out
/// @param n_threads   Forwarded to the mean/std kernels (AVX2 builds)
void rolling_zscore(const double* src, double* dst, size_t n, size_t window,
                    int min_periods, bool ddof1, bool skip_nan,
                    int n_threads = 0);

//Rolling quantile

/// Rolling quantile.
/// @param src         Input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param q           Quantile, strictly in (0, 1)
/// @param min_periods Minimum valid observations required to emit a value
/// @param exact       false → P² streaming approximation (Jain & Chlamtac,
///                    Comm. ACM 28(10), 1985; O(1)/step); true → exact
///                    two-heap order statistics (O(log window) amortised
///                    per step), interpolating like numpy.percentile
/// @note Any non-finite value inside the window yields NaN for that position.
void rolling_quantile(
    const double* src, double* dst,
    size_t n, size_t window, double q, int min_periods, bool exact);

//Expanding windows

/// Expanding mean over valid (finite) values.
/// @param src         Input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param min_periods Minimum valid observations required to emit a value
/// @note Non-finite values are skipped (not counted), like pandas .expanding().
void expanding_mean(const double* src, double* dst, size_t n, int min_periods);
/// Expanding variance via Welford's algorithm (ddof1=true → sample).
void expanding_var(const double* src, double* dst, size_t n,
                   int min_periods, bool ddof1);
/// Expanding standard deviation.
void expanding_std(const double* src, double* dst, size_t n,
                   int min_periods, bool ddof1);
/// Expanding sum (Kahan-compensated).
void expanding_sum(const double* src, double* dst, size_t n, int min_periods);
/// Expanding OLS of y on the global time index 0..i.
/// @param y           Dependent variable, length n
/// @param b0          Output intercepts, length n
/// @param b1          Output slopes, length n
/// @param r2          Optional output R², length n; nullptr to skip
/// @param n           Array length
/// @param min_periods Minimum observations required (effective minimum 2)
void expanding_regression(
    const double* y, double* b0, double* b1, double* r2,
    size_t n, int min_periods);

//Multi-column dispatch (OpenMP over columns)

/// Column-wise rolling mean over an n×p column-major matrix.
/// @param X           Input matrix, n×p column-major
/// @param dst         Output matrix, n×p column-major
/// @param n           Number of rows
/// @param p           Number of columns
/// @param window      Window length (> 0)
/// @param min_periods Minimum observations required to emit a value
/// @param n_threads   OpenMP threads over columns; 0 = library default
void rolling_mean_matrix(const double* X, double* dst, size_t n, int p,
                         size_t window, int min_periods, int n_threads);
void rolling_std_matrix(const double* X, double* dst, size_t n, int p,
                        size_t window, int min_periods, bool ddof1,
                        int n_threads);
void rolling_sum_matrix(const double* X, double* dst, size_t n, int p,
                        size_t window, int min_periods, int n_threads);
void rolling_min_matrix(const double* X, double* dst, size_t n, int p,
                        size_t window, int min_periods, bool skip_nan,
                        int n_threads);
void rolling_max_matrix(const double* X, double* dst, size_t n, int p,
                        size_t window, int min_periods, bool skip_nan,
                        int n_threads);

//Spearman rank correlation

/// Rolling Spearman rank correlation (average ranks for ties, matching
/// scipy.stats.spearmanr and R's cor(method = "spearman")).
/// @param x           First input array of length n
/// @param y           Second input array of length n
/// @param dst         Output array of length n
/// @param n           Array length
/// @param window      Window length (> 0)
/// @param min_periods Minimum valid pairs required to emit a value
/// @note O(window · log window) per step — intended for small/moderate
///       windows (up to a few hundred elements).
void rolling_spearman(
    const double* x, const double* y, double* dst,
    size_t n, size_t window, int min_periods);

} //namespace fastwindow
