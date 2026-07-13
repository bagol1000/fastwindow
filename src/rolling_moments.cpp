/// @file rolling_moments.cpp
/// Rolling skewness, excess kurtosis and z-score.
///
/// skew/kurt maintain sliding raw power sums Σd, Σd², Σd³, Σd⁴ of SHIFTED
/// values d = x − K.  Central moments of degree 3–4 computed from raw sums
/// of x itself lose all precision once |mean| >> stddev (catastrophic
/// cancellation ~ (mean/std)^4); anchoring K to the data keeps the sums
/// conditioned.  K is re-anchored to the current window mean at every
/// 4096-step exact rescan, which also flushes sliding-update drift.
///
/// Output conventions match pandas .rolling().skew()/.kurt() (scipy
/// bias=False): skew needs ≥ 3 observations, kurt (excess) needs ≥ 4;
/// zero-variance windows give NaN.
#include "fastwindow.h"

#include <algorithm>

namespace fastwindow {

static constexpr double DNAN = std::numeric_limits<double>::quiet_NaN();
static constexpr size_t REINIT_INTERVAL = 4096;

namespace {

template <bool SKIP, bool KURT>
static void run_moments_kernel(const double* FW_RESTRICT x,
                               double* FW_RESTRICT dst,
                               size_t n, size_t window, int min_periods) {
    double K = 0.0;               //current shift (anchor)
    bool   K_set = false;
    double s1 = 0, s2 = 0, s3 = 0, s4 = 0;   //Σd, Σd², Σd³, Σd⁴
    size_t nv = 0;                //finite values currently in the window
    size_t nan_count = 0;
    size_t reinit_ctr = 0;

    auto rescan = [&](size_t lo, size_t hi) {
        double sum = 0.0;
        size_t cnt = 0;
        for (size_t j = lo; j <= hi; j++)
            if (fw_isfinite(x[j])) { sum += x[j]; cnt++; }
        K = cnt ? sum / (double)cnt : 0.0;
        K_set = cnt > 0;
        s1 = s2 = s3 = s4 = 0.0;
        nv = cnt;
        for (size_t j = lo; j <= hi; j++) {
            if (!fw_isfinite(x[j])) continue;
            double d = x[j] - K, d2 = d * d;
            s1 += d; s2 += d2; s3 += d2 * d; s4 += d2 * d2;
        }
    };

    for (size_t i = 0; i < n; i++) {
        if (i >= window) {
            double xo = x[i - window];
            if (fw_isfinite(xo)) {
                double d = xo - K, d2 = d * d;
                s1 -= d; s2 -= d2; s3 -= d2 * d; s4 -= d2 * d2;
                nv--;
            } else {
                nan_count--;
            }
        }
        double xn = x[i];
        if (fw_isfinite(xn)) {
            if (!K_set) { K = xn; K_set = true; }
            double d = xn - K, d2 = d * d;
            s1 += d; s2 += d2; s3 += d2 * d; s4 += d2 * d2;
            nv++;
        } else {
            nan_count++;
        }

        if (++reinit_ctr >= REINIT_INTERVAL) {
            reinit_ctr = 0;
            rescan(i + 1 >= window ? i + 1 - window : 0, i);
        }

        size_t m = std::min(i + 1, window);
        bool emit = SKIP ? ((int)nv >= min_periods)
                         : (nan_count == 0 && (int)m >= min_periods);
        const size_t need = KURT ? 4 : 3;
        if (!emit || nv < need) continue;

        double dm   = (double)nv;
        double mean = s1 / dm;
        double M2 = s2 - dm * mean * mean;
        if (!(M2 > 0.0)) continue;   //constant window → NaN stays
        if (KURT) {
            double M4 = s4 - 4.0 * mean * s3 + 6.0 * mean * mean * s2
                        - 3.0 * dm * mean * mean * mean * mean;
            double g2 = dm * M4 / (M2 * M2) - 3.0;   //biased excess kurtosis
            dst[i] = (dm - 1.0) / ((dm - 2.0) * (dm - 3.0))
                     * ((dm + 1.0) * g2 + 6.0);
        } else {
            double g1 = (s3 - 3.0 * mean * s2
                         + 2.0 * dm * mean * mean * mean) / dm
                        / std::pow(M2 / dm, 1.5);    //biased skewness
            dst[i] = std::sqrt(dm * (dm - 1.0)) / (dm - 2.0) * g1;
        }
    }
}

} //anonymous namespace

void rolling_skew(const double* src, double* dst, size_t n, size_t window,
                  int min_periods, bool skip_nan) {
    for (size_t i = 0; i < n; i++) dst[i] = DNAN;
    if (n == 0 || window == 0) return;
    if (skip_nan)
        run_moments_kernel<true,  false>(src, dst, n, window, min_periods);
    else
        run_moments_kernel<false, false>(src, dst, n, window, min_periods);
}

void rolling_kurt(const double* src, double* dst, size_t n, size_t window,
                  int min_periods, bool skip_nan) {
    for (size_t i = 0; i < n; i++) dst[i] = DNAN;
    if (n == 0 || window == 0) return;
    if (skip_nan)
        run_moments_kernel<true,  true>(src, dst, n, window, min_periods);
    else
        run_moments_kernel<false, true>(src, dst, n, window, min_periods);
}

void rolling_zscore(const double* src, double* dst, size_t n, size_t window,
                    int min_periods, bool ddof1, bool skip_nan,
                    int n_threads) {
    if (n == 0) return;
    if (window == 0) {
        for (size_t i = 0; i < n; i++) dst[i] = DNAN;
        return;
    }

    if (n_threads > 1) {
        // The parallel blocked kernels still save one full temporary array:
        // std is written to dst and replaced in-place during combination.
        std::vector<double> mu(n);
        rolling_mean(src, mu.data(), n, window, min_periods, skip_nan, n_threads);
        rolling_std(src, dst, n, window, min_periods, ddof1, skip_nan, n_threads);
        for (size_t i = 0; i < n; i++) {
            double xv = src[i], m = mu[i], s = dst[i];
            dst[i] = (fw_isfinite(xv) && fw_isfinite(m) && fw_isfinite(s) &&
                      s > 0.0) ? (xv - m) / s : DNAN;
        }
        return;
    }

    // Fused scalar path: one rolling state, no n-sized temporaries.
    RunningStats<double> rs(window);
    size_t invalid = 0, reinit_ctr = 0;
    for (size_t i = 0; i < n; i++) {
        if (i >= window) {
            double old = src[i - window];
            if (fw_isfinite(old)) rs.remove(old);
            else                  invalid--;
        }
        double current = src[i];
        if (fw_isfinite(current)) rs.grow(current);
        else                      invalid++;

        if (++reinit_ctr >= 4096 && i + 1 >= window) {
            reinit_ctr = 0;
            rs.reset();
            invalid = 0;
            const size_t start = i + 1 - window;
            for (size_t j = start; j <= i; j++) {
                if (fw_isfinite(src[j])) rs.grow(src[j]);
                else                     invalid++;
            }
        }

        const size_t total = std::min(i + 1, window);
        const bool emit = skip_nan
            ? (static_cast<int>(rs.n) >= min_periods)
            : (invalid == 0 && static_cast<int>(total) >= min_periods);
        double value = DNAN;
        if (emit && fw_isfinite(current)) {
            const double var = rs.variance(ddof1);
            if (fw_isfinite(var) && var > 0.0)
                value = (current - rs.mean) / std::sqrt(var);
        }
        dst[i] = value;
    }
}

} //namespace fastwindow
