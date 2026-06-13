/// @file expanding.cpp
/// Expanding-window statistics (window grows from 1 to n).
/// Pure accumulation — no eviction, so Welford / Kahan running
/// state is numerically stable without periodic reinitialisation.
/// Non-finite values are skipped (not counted), matching pandas .expanding().
#include "fastwindow.h"

namespace fastwindow {

static constexpr double DNAN = std::numeric_limits<double>::quiet_NaN();

void expanding_mean(const double* src, double* dst, size_t n,
                    int min_periods) {
    double sum = 0.0, comp = 0.0;   //Kahan-compensated
    size_t nv = 0;
    for (size_t i = 0; i < n; i++) {
        double v = src[i];
        if (fw_isfinite(v)) {
            double yk = v - comp;
            double t  = sum + yk;
            comp = (t - sum) - yk;
            sum = t;
            nv++;
        }
        dst[i] = ((int)nv >= min_periods && nv > 0)
            ? sum / (double)nv : DNAN;
    }
}

void expanding_var(const double* src, double* dst, size_t n,
                   int min_periods, bool ddof1) {
    RunningStats<double> rs(0);   //window field unused for pure growth
    for (size_t i = 0; i < n; i++) {
        double v = src[i];
        if (fw_isfinite(v)) rs.grow(v);
        double out = DNAN;
        if ((int)rs.n >= min_periods) {
            double var = rs.variance(ddof1);
            if (fw_isfinite(var)) out = var;
        }
        dst[i] = out;
    }
}

void expanding_std(const double* src, double* dst, size_t n,
                   int min_periods, bool ddof1) {
    expanding_var(src, dst, n, min_periods, ddof1);
    for (size_t i = 0; i < n; i++)
        if (fw_isfinite(dst[i])) dst[i] = std::sqrt(dst[i]);
}

void expanding_sum(const double* src, double* dst, size_t n,
                   int min_periods) {
    double sum = 0.0, comp = 0.0;   //Kahan-compensated
    size_t nv = 0;
    for (size_t i = 0; i < n; i++) {
        double v = src[i];
        if (fw_isfinite(v)) {
            double yk = v - comp;
            double t  = sum + yk;
            comp = (t - sum) - yk;
            sum = t;
            nv++;
        }
        dst[i] = ((int)nv >= min_periods && nv > 0) ? sum : DNAN;
    }
}

void expanding_regression(
        const double* y, double* b0, double* b1, double* r2,
        size_t n, int min_periods) {
    RunningRegression rr;
    for (size_t i = 0; i < n; i++) {
        if (fw_isfinite(y[i]))
            rr.grow((double)i, y[i]);   //x = global time index, gaps allowed
        b0[i] = DNAN;
        b1[i] = DNAN;
        if (r2) r2[i] = DNAN;
        if ((int)rr.n >= min_periods && rr.n >= 2) {
            double icpt, slope;
            rr.coeffs(icpt, slope);
            b0[i] = icpt;
            b1[i] = slope;
            if (r2) r2[i] = rr.r_squared();
        }
    }
}

} //namespace fastwindow
