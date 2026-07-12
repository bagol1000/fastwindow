/// @file rolling_corr.cpp
/// Rolling Pearson correlation and covariance.
///
/// A pair (x_i, y_i) is "valid" iff both values are finite.  Only valid
/// pairs enter the five running sums, so no window buffer is needed:
/// evicted values are re-read straight from the source arrays.  Output is
/// gated on valid count vs window count (skip_nan=false requires all pairs
/// valid).  All sums are recomputed from the source window every 4096 steps.
#include "fastwindow.h"
#include "simd_scan.h"

#include <algorithm>

namespace fastwindow {

static constexpr size_t REINIT_INTERVAL = 4096;
static constexpr double DNAN = std::numeric_limits<double>::quiet_NaN();

/// Finite check reading the bits straight from memory: compiles to a single
/// integer load, avoiding the FP→GPR domain crossing that fw_isfinite incurs
/// on a value already in an xmm register.
static inline bool finite_mem(const double* p) {
    uint64_t u;
    std::memcpy(&u, p, sizeof(u));
    return (u & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

//RunningCorrelation

void RunningCorrelation::grow(double x, double y) {
    sum_x  += x;
    sum_y  += y;
    sum_xx += x * x;
    sum_yy += y * y;
    sum_xy += x * y;
    n++;
}

void RunningCorrelation::remove(double x, double y) {
    sum_x  -= x;
    sum_y  -= y;
    sum_xx -= x * x;
    sum_yy -= y * y;
    sum_xy -= x * y;
    n--;
}

void RunningCorrelation::add(double x_new, double y_new,
                             double x_old, double y_old) {
    sum_x  += x_new - x_old;
    sum_y  += y_new - y_old;
    sum_xx += x_new * x_new - x_old * x_old;
    sum_yy += y_new * y_new - y_old * y_old;
    sum_xy += x_new * y_new - x_old * y_old;
}

void RunningCorrelation::reset() {
    sum_x = sum_y = sum_xx = sum_yy = sum_xy = 0.0;
    n = 0;
}

double RunningCorrelation::corr() const {
    if (n < 2) return DNAN;
    double dn  = static_cast<double>(n);
    double vx  = dn * sum_xx - sum_x * sum_x;
    double vy  = dn * sum_yy - sum_y * sum_y;
    double cxy = dn * sum_xy - sum_x * sum_y;
    //Relative zero-variance guard: a constant series cancels to rounding
    //noise of order eps·n·Σx², which must map to NaN, never ±Inf.
    if (!(vx > 1e-14 * dn * std::abs(sum_xx)) ||
        !(vy > 1e-14 * dn * std::abs(sum_yy)))
        return DNAN;
    double r = cxy / std::sqrt(vx * vy);
    if (r >  1.0) r =  1.0;
    if (r < -1.0) r = -1.0;
    return r;
}

double RunningCorrelation::cov(bool ddof1) const {
    size_t min_n = ddof1 ? 2 : 1;
    if (n < min_n) return DNAN;
    double dn = static_cast<double>(n);
    //Single-division form of (Σxy − ΣxΣy/n) / (n − ddof)
    double ddof = ddof1 ? 1.0 : 0.0;
    return (dn * sum_xy - sum_x * sum_y) / (dn * (dn - ddof));
}

//Shared kernel loop.  Emit decides what gets written for each position.
//SKIP is a compile-time flag so the steady-state loop carries no per-step
//mode branch; warmup (growing window) and steady state are separate loops.

template <bool SKIP, typename Emit>
static void run_corr_kernel(
        const double* FW_RESTRICT x, const double* FW_RESTRICT y,
        size_t n, size_t window, int min_periods, Emit&& emit) {
    //Local scalars (not a struct passed by reference) so the sums live in
    //registers across iterations instead of bouncing through the stack.
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    size_t nv = 0;            //count of valid pairs in the window
    size_t reinit_ctr = 0;
    const size_t warm = std::min(n, window);

    auto rescan = [&](size_t lo, size_t hi) {   //exact recompute [lo, hi]
        sx = sy = sxx = syy = sxy = 0.0;
        nv = 0;
        for (size_t j = lo; j <= hi; j++) {
            double xj = x[j], yj = y[j];
            if (fw_isfinite(xj) && fw_isfinite(yj)) {
                sx += xj;  sy += yj;
                sxx += xj * xj;  syy += yj * yj;  sxy += xj * yj;
                nv++;
            }
        }
    };

    //warmup: window still growing, no evictions
    for (size_t i = 0; i < warm; i++) {
        double xn = x[i], yn = y[i];
        if (fw_isfinite(xn) && fw_isfinite(yn)) {
            sx += xn;  sy += yn;
            sxx += xn * xn;  syy += yn * yn;  sxy += xn * yn;
            nv++;
        }
        ++reinit_ctr;
        //Every position is written exactly once: gating positions pass
        //nv_eff = 0, which the finalisers map to NaN (no prefill pass).
        size_t nv_eff;
        if (SKIP) {
            nv_eff = ((int)nv >= min_periods) ? nv : 0;
        } else {
            size_t m = i + 1;   //nv < m ⇒ invalid pair present
            nv_eff = (nv == m && (int)m >= min_periods) ? nv : 0;
        }
        emit(i, nv_eff, sx, sy, sxx, syy, sxy);
    }

    //steady state: fused evict+add, one update per sum per step
    const bool full_meets_mp = (int)window >= min_periods;
    for (size_t i = warm; i < n; i++) {
        //Branchless: invalid pairs are blended to (0,0) — an exact no-op
        //for every sum — and the valid count adjusted with integer math.
        double xo = x[i - window], yo = y[i - window];
        double xn = x[i],          yn = y[i];
        bool old_ok = fw_isfinite(xo) && fw_isfinite(yo);
        bool new_ok = fw_isfinite(xn) && fw_isfinite(yn);
        xo = old_ok ? xo : 0.0;
        yo = old_ok ? yo : 0.0;
        xn = new_ok ? xn : 0.0;
        yn = new_ok ? yn : 0.0;
        sx  += xn - xo;
        sy  += yn - yo;
        sxx += xn * xn - xo * xo;
        syy += yn * yn - yo * yo;
        sxy += xn * yn - xo * yo;
        nv  += static_cast<size_t>(new_ok) - static_cast<size_t>(old_ok);

        //Periodic exact recompute from the source window
        if (++reinit_ctr >= REINIT_INTERVAL) {
            reinit_ctr = 0;
            rescan(i + 1 - window, i);
        }

        size_t nv_eff;
        if (SKIP) {
            nv_eff = ((int)nv >= min_periods) ? nv : 0;
        } else {
            nv_eff = (nv == window && full_meets_mp) ? nv : 0;
        }
        emit(i, nv_eff, sx, sy, sxx, syy, sxy);
    }
}

//Scalar-argument forms of the statistics (shared by the emit lambdas)

static inline double corr_from_sums(size_t n, double sx, double sy,
                                    double sxx, double syy, double sxy) {
    if (n < 2) return DNAN;
    double dn  = static_cast<double>(n);
    double vx  = dn * sxx - sx * sx;
    double vy  = dn * syy - sy * sy;
    double cxy = dn * sxy - sx * sy;
    if (!(vx > 1e-14 * dn * std::abs(sxx)) ||
        !(vy > 1e-14 * dn * std::abs(syy)))
        return DNAN;
    double r = cxy / std::sqrt(vx * vy);
    if (r >  1.0) r =  1.0;
    if (r < -1.0) r = -1.0;
    return r;
}

static inline double cov_from_sums(size_t n, double sx, double sy,
                                   double sxy, bool ddof1) {
    if (n < (ddof1 ? 2u : 1u)) return DNAN;
    double dn = static_cast<double>(n);
    double ddof = ddof1 ? 1.0 : 0.0;
    return (dn * sxy - sx * sy) / (dn * (dn - ddof));
}

template <typename Emit>
static void run_corr_dispatch(
        const double* x, const double* y,
        size_t n, size_t window, int min_periods, bool skip_nan,
        Emit&& emit) {
    if (skip_nan)
        run_corr_kernel<true>(x, y, n, window, min_periods,
                              std::forward<Emit>(emit));
    else
        run_corr_kernel<false>(x, y, n, window, min_periods,
                               std::forward<Emit>(emit));
}


//Blocked (van Herk) corr/cov kernel — AVX2, skip_nan = false only.
//
//All five sums are additive, so the same block-local prefix/suffix scan
//used for mean/std applies with five streams (x, y, x², y², xy).  This
//removes the loop-carried dependency of the sliding kernel AND lets the
//finalisation (sqrt + divide) run four-wide inside the emit pass — bolted
//onto the serial kernel that finalisation was measured slower, here it is
//free.  Window sums are block-local (≤ 2w elements), so accuracy is at
//machine precision and no periodic reinitialisation is needed.

#if FW_SIMD

using simd::scan_fwd_add;
using simd::scan_bwd_add;
using simd::bad_lanes;

FW_TARGET_AVX2
static void run_corr_blocked(
        const double* FW_RESTRICT x, const double* FW_RESTRICT y,
        double* FW_RESTRICT dst_corr, double* FW_RESTRICT dst_cov,
        size_t n, size_t w, int min_periods, bool cov_ddof1) {
    const double QNAN = std::numeric_limits<double>::quiet_NaN();

    {   //scalar warmup: growing window, positions 0 .. w-2
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        size_t nv = 0;
        size_t lim = std::min(n, w - 1);
        for (size_t i = 0; i < lim; i++) {
            double xv = x[i], yv = y[i];
            if (fw_isfinite(xv) && fw_isfinite(yv)) {
                sx += xv;  sy += yv;
                sxx += xv * xv;  syy += yv * yv;  sxy += xv * yv;
                nv++;
            }
            size_t m = i + 1;
            bool emit = (nv == m && (int)m >= min_periods);
            if (dst_corr)
                dst_corr[i] = emit ? corr_from_sums(nv, sx, sy, sxx, syy, sxy)
                                   : QNAN;
            if (dst_cov)
                dst_cov[i] = emit ? cov_from_sums(nv, sx, sy, sxy, cov_ddof1)
                                  : QNAN;
        }
        if (n < w) return;
    }

    const double dw = (double)w;
    const __m256d vw     = _mm256_set1_pd(dw);
    const __m256d veps   = _mm256_set1_pd(1e-14 * dw);
    const __m256d vone   = _mm256_set1_pd(1.0);
    const __m256d vmone  = _mm256_set1_pd(-1.0);
    const __m256d vnan   = _mm256_set1_pd(QNAN);
    const __m256d vzero  = _mm256_setzero_pd();
    const __m256d absmsk = _mm256_castsi256_pd(
        _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    const double cov_den = dw * (dw - (cov_ddof1 ? 1.0 : 0.0));
    const __m256d vcovden = _mm256_set1_pd(1.0 / cov_den);

    //suffix arrays for the five streams, previous and current block
    constexpr int NS = 5;
    std::vector<double> Sp((w + 1) * NS, 0.0), Sc((w + 1) * NS, 0.0);
    auto SP = [&](int s) { return Sp.data() + (size_t)s * (w + 1); };
    auto SC = [&](int s) { return Sc.data() + (size_t)s * (w + 1); };
    std::vector<uint8_t> nanSuf(w + 1, 0);
    bool bad_prev = false;

    const size_t nblocks = (n + w - 1) / w;
    for (size_t k = 0; k < nblocks; k++) {
        const size_t s = k * w;
        const size_t L = std::min(w, n - s);

        bool bad_cur = false;
        {   //backward suffix scans (invalid pairs blended to (0,0))
            size_t j = L;
            double c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;
            while (j > (L & ~size_t(3))) {
                j--;
                double xv = x[s + j], yv = y[s + j];
                bool bad = !(fw_isfinite(xv) && fw_isfinite(yv));
                bad_cur |= bad;
                if (bad) { xv = 0.0; yv = 0.0; }
                c1 += xv;  c2 += yv;
                c3 += xv * xv;  c4 += yv * yv;  c5 += xv * yv;
                SC(0)[j] = c1;  SC(1)[j] = c2;
                SC(2)[j] = c3;  SC(3)[j] = c4;  SC(4)[j] = c5;
            }
            __m256d v1 = _mm256_set1_pd(c1), v2 = _mm256_set1_pd(c2);
            __m256d v3 = _mm256_set1_pd(c3), v4 = _mm256_set1_pd(c4);
            __m256d v5 = _mm256_set1_pd(c5);
            int badacc = 0;
            while (j >= 4) {
                j -= 4;
                __m256d vx = _mm256_loadu_pd(x + s + j);
                __m256d vy = _mm256_loadu_pd(y + s + j);
                __m256d bad = _mm256_or_pd(bad_lanes(vx), bad_lanes(vy));
                badacc |= _mm256_movemask_pd(bad);
                vx = _mm256_blendv_pd(vx, vzero, bad);
                vy = _mm256_blendv_pd(vy, vzero, bad);
                _mm256_storeu_pd(SC(0) + j, scan_bwd_add(vx, v1));
                _mm256_storeu_pd(SC(1) + j, scan_bwd_add(vy, v2));
                _mm256_storeu_pd(SC(2) + j,
                    scan_bwd_add(_mm256_mul_pd(vx, vx), v3));
                _mm256_storeu_pd(SC(3) + j,
                    scan_bwd_add(_mm256_mul_pd(vy, vy), v4));
                _mm256_storeu_pd(SC(4) + j,
                    scan_bwd_add(_mm256_mul_pd(vx, vy), v5));
            }
            bad_cur |= (badacc != 0);
        }
        for (int st = 0; st < NS; st++)
            for (size_t j = L; j <= w; j++) SC(st)[j] = 0.0;

        const size_t r_lo = std::max(s, w - 1);
        if (r_lo < s + L) {
            if (!bad_prev && !bad_cur) {
                if (k == 0) {
                    if (dst_corr)
                        dst_corr[w - 1] = corr_from_sums(
                            w, SC(0)[0], SC(1)[0], SC(2)[0], SC(3)[0],
                            SC(4)[0]);
                    if (dst_cov)
                        dst_cov[w - 1] = cov_from_sums(
                            w, SC(0)[0], SC(1)[0], SC(4)[0], cov_ddof1);
                } else {
                    size_t j = 0, r = s;
                    __m256d c1 = vzero, c2 = vzero, c3 = vzero,
                            c4 = vzero, c5 = vzero;
                    for (; j + 4 <= L; j += 4, r += 4) {
                        __m256d vx = _mm256_loadu_pd(x + r);
                        __m256d vy = _mm256_loadu_pd(y + r);
                        __m256d sx = _mm256_add_pd(scan_fwd_add(vx, c1),
                            _mm256_loadu_pd(SP(0) + j + 1));
                        __m256d sy = _mm256_add_pd(scan_fwd_add(vy, c2),
                            _mm256_loadu_pd(SP(1) + j + 1));
                        __m256d sxx = _mm256_add_pd(
                            scan_fwd_add(_mm256_mul_pd(vx, vx), c3),
                            _mm256_loadu_pd(SP(2) + j + 1));
                        __m256d syy = _mm256_add_pd(
                            scan_fwd_add(_mm256_mul_pd(vy, vy), c4),
                            _mm256_loadu_pd(SP(3) + j + 1));
                        __m256d sxy = _mm256_add_pd(
                            scan_fwd_add(_mm256_mul_pd(vx, vy), c5),
                            _mm256_loadu_pd(SP(4) + j + 1));

                        __m256d cxy = _mm256_sub_pd(_mm256_mul_pd(vw, sxy),
                                                    _mm256_mul_pd(sx, sy));
                        if (dst_cov)
                            _mm256_storeu_pd(dst_cov + r,
                                _mm256_mul_pd(cxy, vcovden));
                        if (dst_corr) {
                            __m256d vxv = _mm256_sub_pd(
                                _mm256_mul_pd(vw, sxx), _mm256_mul_pd(sx, sx));
                            __m256d vyv = _mm256_sub_pd(
                                _mm256_mul_pd(vw, syy), _mm256_mul_pd(sy, sy));
                            //relative zero-variance guard (constant series)
                            __m256d ok = _mm256_and_pd(
                                _mm256_cmp_pd(vxv, _mm256_mul_pd(veps,
                                    _mm256_and_pd(sxx, absmsk)), _CMP_GT_OQ),
                                _mm256_cmp_pd(vyv, _mm256_mul_pd(veps,
                                    _mm256_and_pd(syy, absmsk)), _CMP_GT_OQ));
                            __m256d rr = _mm256_div_pd(cxy,
                                _mm256_sqrt_pd(_mm256_mul_pd(vxv, vyv)));
                            rr = _mm256_min_pd(rr, vone);
                            rr = _mm256_max_pd(rr, vmone);
                            rr = _mm256_blendv_pd(vnan, rr, ok);
                            _mm256_storeu_pd(dst_corr + r, rr);
                        }
                    }
                    //scalar tail (carries = lane 0 of each broadcast)
                    double t1 = _mm_cvtsd_f64(_mm256_castpd256_pd128(c1));
                    double t2 = _mm_cvtsd_f64(_mm256_castpd256_pd128(c2));
                    double t3 = _mm_cvtsd_f64(_mm256_castpd256_pd128(c3));
                    double t4 = _mm_cvtsd_f64(_mm256_castpd256_pd128(c4));
                    double t5 = _mm_cvtsd_f64(_mm256_castpd256_pd128(c5));
                    for (; j < L; j++, r++) {
                        double xv = x[r], yv = y[r];
                        t1 += xv;  t2 += yv;
                        t3 += xv * xv;  t4 += yv * yv;  t5 += xv * yv;
                        double sx = t1 + SP(0)[j + 1], sy = t2 + SP(1)[j + 1];
                        double sxx = t3 + SP(2)[j + 1];
                        double syy = t4 + SP(3)[j + 1];
                        double sxy = t5 + SP(4)[j + 1];
                        if (dst_corr)
                            dst_corr[r] = corr_from_sums(w, sx, sy, sxx, syy,
                                                         sxy);
                        if (dst_cov)
                            dst_cov[r] = cov_from_sums(w, sx, sy, sxy,
                                                       cov_ddof1);
                    }
                }
            } else {
                //dirty path: scalar with explicit NaN-presence tracking
                if (bad_prev) {
                    bool acc = false;
                    for (size_t j = w; j-- > 0;) {
                        acc |= !(fw_isfinite(x[s - w + j]) &&
                                 fw_isfinite(y[s - w + j]));
                        nanSuf[j] = acc ? 1 : 0;
                    }
                    nanSuf[w] = 0;
                } else {
                    std::fill(nanSuf.begin(), nanSuf.end(), 0);
                }
                double p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0;
                bool nanP = false;
                for (size_t j = 0; j < L; j++) {
                    size_t r = s + j;
                    double xv = x[r], yv = y[r];
                    bool bad = !(fw_isfinite(xv) && fw_isfinite(yv));
                    nanP |= bad;
                    if (bad) { xv = 0.0; yv = 0.0; }
                    p1 += xv;  p2 += yv;
                    p3 += xv * xv;  p4 += yv * yv;  p5 += xv * yv;
                    if (r < w - 1) continue;
                    if (nanP || nanSuf[j + 1]) {
                        if (dst_corr) dst_corr[r] = QNAN;
                        if (dst_cov)  dst_cov[r]  = QNAN;
                    } else {
                        double sx = p1 + SP(0)[j + 1], sy = p2 + SP(1)[j + 1];
                        double sxx = p3 + SP(2)[j + 1];
                        double syy = p4 + SP(3)[j + 1];
                        double sxy = p5 + SP(4)[j + 1];
                        if (dst_corr)
                            dst_corr[r] = corr_from_sums(w, sx, sy, sxx, syy,
                                                         sxy);
                        if (dst_cov)
                            dst_cov[r] = cov_from_sums(w, sx, sy, sxy,
                                                       cov_ddof1);
                    }
                }
            }
        }

        Sp.swap(Sc);
        bad_prev = bad_cur;
    }
}

#endif //FW_SIMD

//Public kernels

void rolling_corr(
        const double* x, const double* y,
        double* dst_corr, double* dst_cov,
        size_t n, size_t window, int min_periods, bool skip_nan) {
    if (n == 0 || window == 0) {
        //Degenerate parameters: NaN-fill and return (the kernel otherwise
        //writes every position itself — no prefill pass needed).
        for (size_t i = 0; i < n; i++) {
            dst_corr[i] = DNAN;
            if (dst_cov) dst_cov[i] = DNAN;
        }
        return;
    }

#if FW_SIMD
    if (cpu_has_avx2() && !skip_nan && window >= 16 &&
        min_periods <= (int)window) {
        run_corr_blocked(x, y, dst_corr, dst_cov, n, window, min_periods,
                         /*cov_ddof1=*/true);
        return;
    }
#endif

    if (dst_cov) {
        double* FW_RESTRICT dc = dst_corr;
        double* FW_RESTRICT dv = dst_cov;
        run_corr_dispatch(x, y, n, window, min_periods, skip_nan,
            [=](size_t i, size_t nv, double sx, double sy,
                double sxx, double syy, double sxy) {
                dc[i] = corr_from_sums(nv, sx, sy, sxx, syy, sxy);
                dv[i] = cov_from_sums(nv, sx, sy, sxy, true);  //pandas-compatible
            });
    } else {
        //NOTE: a two-pass vectorised finalisation (sqrt+div in a separate
        //loop) was implemented and measured SLOWER here — out-of-order
        //execution already hides the per-step sqrt+div behind the serial
        //running-sum chain, so the extra pass only added memory traffic.
        double* FW_RESTRICT dc = dst_corr;
        run_corr_dispatch(x, y, n, window, min_periods, skip_nan,
            [=](size_t i, size_t nv, double sx, double sy,
                double sxx, double syy, double sxy) {
                dc[i] = corr_from_sums(nv, sx, sy, sxx, syy, sxy);
            });
    }
}

void rolling_cov(
        const double* x, const double* y, double* dst,
        size_t n, size_t window, int min_periods, bool ddof1, bool skip_nan) {
    if (n == 0 || window == 0) {
        for (size_t i = 0; i < n; i++) dst[i] = DNAN;
        return;
    }

#if FW_SIMD
    if (cpu_has_avx2() && !skip_nan && window >= 16 &&
        min_periods <= (int)window) {
        run_corr_blocked(x, y, nullptr, dst, n, window, min_periods, ddof1);
        return;
    }
#endif

    double* FW_RESTRICT d = dst;
    run_corr_dispatch(x, y, n, window, min_periods, skip_nan,
        [=](size_t i, size_t nv, double sx, double sy,
            double /*sxx*/, double /*syy*/, double sxy) {
            d[i] = cov_from_sums(nv, sx, sy, sxy, ddof1);
        });
}

} //namespace fastwindow
