/// @file bindings_r.cpp
/// Rcpp bindings for the rolling statistics kernels.  Thin .Call wrappers;
/// all argument validation happens in the R layer (R/fastroll.R).
#include <Rcpp.h>
#include "fastwindow.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace Rcpp;

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_mean(NumericVector x, int window,
                               int min_periods, bool skip_nan,
                               int n_threads) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_mean(REAL(x), REAL(out), n,
                             (size_t)window, min_periods, skip_nan,
                             n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_var(NumericVector x, int window,
                              int min_periods, int ddof, bool skip_nan,
                              int n_threads) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_var(REAL(x), REAL(out), n,
                            (size_t)window, min_periods, ddof == 1, skip_nan,
                            n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_std(NumericVector x, int window,
                              int min_periods, int ddof, bool skip_nan,
                              int n_threads) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_std(REAL(x), REAL(out), n,
                            (size_t)window, min_periods, ddof == 1, skip_nan,
                            n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_sum(NumericVector x, int window,
                              int min_periods, bool skip_nan,
                              int n_threads) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_sum(REAL(x), REAL(out), n,
                            (size_t)window, min_periods, skip_nan,
                            n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_min(NumericVector x, int window, int n_threads) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_min(REAL(x), REAL(out), n, (size_t)window, n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_max(NumericVector x, int window, int n_threads) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_max(REAL(x), REAL(out), n, (size_t)window, n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
List cpp_rolling_regression(NumericVector y, int window, int min_periods) {
    size_t n = y.size();
    NumericVector b0(n), b1(n), r2(n);
    fastwindow::rolling_simple_regression(
        REAL(y), REAL(b0), REAL(b1), REAL(r2),
        n, (size_t)window, min_periods, true);
    return List::create(_["intercept"] = b0,
                        _["slope"]     = b1,
                        _["r2"]        = r2);
}

// [[Rcpp::export(rng = false)]]
List cpp_rolling_regression_xy(NumericVector y, NumericVector x,
                               int window, int min_periods) {
    size_t n = y.size();
    NumericVector b0(n), b1(n), r2(n);
    fastwindow::rolling_simple_regression_xy(
        REAL(x), REAL(y), REAL(b0), REAL(b1), REAL(r2),
        n, (size_t)window, min_periods);
    return List::create(_["intercept"] = b0,
                        _["slope"]     = b1,
                        _["r2"]        = r2);
}

// [[Rcpp::export(rng = false)]]
List cpp_rolling_multiple_regression(NumericVector y, NumericMatrix X,
                                     int window, int min_periods) {
    size_t n = y.size();
    int k = X.ncol();
    NumericMatrix beta(n, k + 1);   //R matrices are column-major natively
    NumericVector r2(n), res_std(n);
    fastwindow::rolling_multiple_regression(
        REAL(y), REAL(X), REAL(beta), REAL(r2), REAL(res_std),
        n, k, (size_t)window, min_periods);
    return List::create(_["coef"]         = beta,
                        _["r2"]           = r2,
                        _["residual_std"] = res_std);
}

// [[Rcpp::export(rng = false)]]
List cpp_rolling_corr(NumericVector x, NumericVector y, int window,
                      int min_periods, bool return_cov, bool skip_nan) {
    size_t n = x.size();
    NumericVector corr(n);
    if (return_cov) {
        NumericVector cov(n);
        fastwindow::rolling_corr(REAL(x), REAL(y), REAL(corr), REAL(cov),
                                 n, (size_t)window, min_periods, skip_nan);
        return List::create(_["corr"] = corr, _["cov"] = cov);
    }
    fastwindow::rolling_corr(REAL(x), REAL(y), REAL(corr), nullptr,
                             n, (size_t)window, min_periods, skip_nan);
    return List::create(_["corr"] = corr);
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_cov(NumericVector x, NumericVector y, int window,
                              int min_periods, int ddof, bool skip_nan) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_cov(REAL(x), REAL(y), REAL(out),
                            n, (size_t)window, min_periods,
                            ddof == 1, skip_nan);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_corr_matrix(NumericMatrix X, int window,
                                      int min_periods, int n_threads) {
    size_t n = X.nrow();
    int p    = X.ncol();
    NumericVector out(n * static_cast<size_t>(p) * p);
    size_t n_pairs = static_cast<size_t>(p) * (p - 1) / 2;
    std::vector<double> tri(n_pairs * n);
    fastwindow::rolling_corr_matrix(REAL(X), tri.data(),
                                    n, p, (size_t)window, min_periods,
                                    n_threads);
    fastwindow::corr_matrix_expand(tri.data(), REAL(out), n, p,
                                   /*r_layout=*/true, n_threads);
    out.attr("dim") = IntegerVector::create((int)n, p, p);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_quantile(NumericVector x, int window, double q,
                                   int min_periods, bool exact) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_quantile(REAL(x), REAL(out), n, (size_t)window, q,
                                 min_periods, exact);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_expanding_mean(NumericVector x, int min_periods) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::expanding_mean(REAL(x), REAL(out), n, min_periods);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_expanding_var(NumericVector x, int min_periods, int ddof) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::expanding_var(REAL(x), REAL(out), n, min_periods, ddof == 1);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_expanding_std(NumericVector x, int min_periods, int ddof) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::expanding_std(REAL(x), REAL(out), n, min_periods, ddof == 1);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_expanding_sum(NumericVector x, int min_periods) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::expanding_sum(REAL(x), REAL(out), n, min_periods);
    return out;
}

// [[Rcpp::export(rng = false)]]
List cpp_expanding_regression(NumericVector y, int min_periods) {
    size_t n = y.size();
    NumericVector b0(n), b1(n), r2(n);
    fastwindow::expanding_regression(REAL(y), REAL(b0), REAL(b1), REAL(r2),
                                     n, min_periods);
    return List::create(_["intercept"] = b0,
                        _["slope"]     = b1,
                        _["r2"]        = r2);
}

// [[Rcpp::export(rng = false)]]
NumericMatrix cpp_rolling_mean_matrix(NumericMatrix X, int window,
                                      int min_periods, int n_threads) {
    size_t n = X.nrow();
    int p = X.ncol();
    NumericMatrix out(n, p);
    fastwindow::rolling_mean_matrix(REAL(X), REAL(out), n, p,
                                    (size_t)window, min_periods, n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericMatrix cpp_rolling_std_matrix(NumericMatrix X, int window,
                                     int min_periods, int ddof,
                                     int n_threads) {
    size_t n = X.nrow();
    int p = X.ncol();
    NumericMatrix out(n, p);
    fastwindow::rolling_std_matrix(REAL(X), REAL(out), n, p,
                                   (size_t)window, min_periods,
                                   ddof == 1, n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericMatrix cpp_rolling_sum_matrix(NumericMatrix X, int window,
                                     int min_periods, int n_threads) {
    size_t n = X.nrow();
    int p = X.ncol();
    NumericMatrix out(n, p);
    fastwindow::rolling_sum_matrix(REAL(X), REAL(out), n, p,
                                   (size_t)window, min_periods, n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericMatrix cpp_rolling_min_matrix(NumericMatrix X, int window,
                                     int n_threads) {
    size_t n = X.nrow();
    int p = X.ncol();
    NumericMatrix out(n, p);
    fastwindow::rolling_min_matrix(REAL(X), REAL(out), n, p,
                                   (size_t)window, n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericMatrix cpp_rolling_max_matrix(NumericMatrix X, int window,
                                     int n_threads) {
    size_t n = X.nrow();
    int p = X.ncol();
    NumericMatrix out(n, p);
    fastwindow::rolling_max_matrix(REAL(X), REAL(out), n, p,
                                   (size_t)window, n_threads);
    return out;
}

// [[Rcpp::export(rng = false)]]
NumericVector cpp_rolling_spearman(NumericVector x, NumericVector y,
                                   int window, int min_periods) {
    size_t n = x.size();
    NumericVector out(n);
    fastwindow::rolling_spearman(REAL(x), REAL(y), REAL(out),
                                 n, (size_t)window, min_periods);
    return out;
}

// [[Rcpp::export(rng = false)]]
void cpp_set_num_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

// [[Rcpp::export(rng = false)]]
int cpp_get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// [[Rcpp::export(rng = false)]]
bool cpp_has_avx2() {
    return FW_SIMD && fastwindow::cpu_has_avx2();
}
