/// @file rolling_basic.cpp
/// Rolling mean, var, std, sum, min, max.
/// Algorithms: sliding Welford for mean/variance, monotonic deque for min/max.
/// Running sums are reinitialised from scratch every 4096 steps for numerical stability.
#include "fastwindow.h"
#include "simd_scan.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <algorithm>
#include <cassert>

namespace fastwindow {

//RollingBuffer

template <typename T>
void RollingBuffer<T>::push(T val) {
    if (count < capacity) {
        data[(head + count) % capacity] = val;
        count++;
    } else {
        data[head] = val;
        head = (head + 1) % capacity;
    }
}

template struct RollingBuffer<double>;
template struct RollingBuffer<float>;

//MonotonicDeque

void MonotonicDeque::push_min(size_t i, const double* src, size_t window) {
    while (!idx.empty() && idx.front() + window <= i)
        idx.pop_front();
    while (!idx.empty() && src[idx.back()] >= src[i])
        idx.pop_back();
    idx.push_back(i);
}

void MonotonicDeque::push_max(size_t i, const double* src, size_t window) {
    while (!idx.empty() && idx.front() + window <= i)
        idx.pop_front();
    while (!idx.empty() && src[idx.back()] <= src[i])
        idx.pop_back();
    idx.push_back(i);
}

//RunningStats

template <typename T>
void RunningStats<T>::grow(T x_new) {
    n++;
    T delta = x_new - mean;
    mean += delta / static_cast<T>(n);
    T delta2 = x_new - mean;
    M2 += delta * delta2;
}

template <typename T>
void RunningStats<T>::slide(T x_new, T x_old) {
    //Sliding Welford: n stays constant (= window capacity for finite values)
    T delta1 = x_new - mean;
    mean += (x_new - x_old) / static_cast<T>(n);
    T delta2 = x_old - mean;  //after mean update
    M2 += (x_new - x_old) * (delta1 + delta2);
    if (M2 < T(0)) M2 = T(0);  //guard against floating-point underflow
}

template <typename T>
void RunningStats<T>::remove(T x_old) {
    if (n == 0) return;
    if (n == 1) { reset(); return; }
    T old_mean = mean;
    n--;
    mean = (mean * static_cast<T>(n + 1) - x_old) / static_cast<T>(n);
    M2  -= (x_old - old_mean) * (x_old - mean);
    if (M2 < T(0)) M2 = T(0);
}

template <typename T>
T RunningStats<T>::variance(bool ddof1) const {
    const T nan = std::numeric_limits<T>::quiet_NaN();
    if (n == 0) return nan;
    if (ddof1 && n < 2) return nan;
    size_t denom = n - (ddof1 ? 1 : 0);
    return M2 / static_cast<T>(denom);
}

template <typename T>
T RunningStats<T>::stddev(bool ddof1) const {
    T v = variance(ddof1);
    return fw_isfinite(v) ? std::sqrt(v) : v;
}

template struct RunningStats<double>;
template struct RunningStats<float>;

//Internal helpers

static constexpr size_t REINIT_INTERVAL = 4096;
static constexpr double DNAN = std::numeric_limits<double>::quiet_NaN();

#if FW_SIMD
//Blocked AVX2 kernels (van Herk structure); defined further below.
namespace {
static void run_sum_blocked(const double* FW_RESTRICT x,
                            double* FW_RESTRICT dst,
                            size_t n, size_t w, int min_periods,
                            bool mean_div, int n_threads);
static void run_var_blocked(const double* FW_RESTRICT x,
                            double* FW_RESTRICT dst,
                            size_t n, size_t w, int min_periods,
                            bool ddof1, bool do_sqrt, int n_threads);
}
#endif

/// Reinit RunningStats from the circular buffer using standard Welford.
static void reinit_welford(const RollingBuffer<double>& buf,
                           RunningStats<double>& rs, size_t& nan_count) {
    rs.reset(); nan_count = 0;
    for (size_t j = 0; j < buf.count; j++) {
        double v = buf.data[(buf.head + j) % buf.capacity];
        if (fw_isfinite(v)) rs.grow(v);
        else nan_count++;
    }
}

//rolling_mean / rolling_sum
//
//Buffer-free design: evicted values are re-read from the source array, the
//fused evict+add keeps one update per step on the running sum (kept in a
//register-resident local), invalid values are blended to 0.0 branchlessly,
//and every output position is written exactly once (NaN via blend).
//Kahan-compensated rescan of the source window every 4096 steps.

template <bool SKIP, bool MEAN>
static void run_sum_kernel(const double* FW_RESTRICT x,
                           double* FW_RESTRICT dst,
                           size_t n, size_t window, int min_periods) {
    double sum = 0.0;
    size_t nv = 0;             //valid (finite) values in the window
    size_t reinit_ctr = 0;
    const size_t warm = std::min(n, window);

    auto rescan = [&](size_t lo, size_t hi) {   //Kahan rescan of [lo, hi]
        double ksum = 0.0, comp = 0.0;
        nv = 0;
        for (size_t j = lo; j <= hi; j++) {
            double v = x[j];
            if (fw_isfinite(v)) {
                double yk = v - comp;
                double t  = ksum + yk;
                comp = (t - ksum) - yk;
                ksum = t;
                nv++;
            }
        }
        sum = ksum;
    };

    for (size_t i = 0; i < warm; i++) {
        double v = x[i];
        bool ok = fw_isfinite(v);
        sum += ok ? v : 0.0;
        nv  += static_cast<size_t>(ok);
        ++reinit_ctr;
        size_t m = i + 1;
        bool emit = SKIP ? ((int)nv >= min_periods && nv > 0)
                         : (nv == m && (int)m >= min_periods);
        size_t den = SKIP ? nv : m;
        double val = MEAN ? sum / (double)(den ? den : 1) : sum;
        dst[i] = emit ? val : std::numeric_limits<double>::quiet_NaN();
    }

    const bool full_ok = (int)window >= min_periods;
    for (size_t i = warm; i < n; i++) {
        double xo = x[i - window], xn = x[i];
        bool ok_o = fw_isfinite(xo);
        bool ok_n = fw_isfinite(xn);
        sum += (ok_n ? xn : 0.0) - (ok_o ? xo : 0.0);
        nv  += static_cast<size_t>(ok_n) - static_cast<size_t>(ok_o);

        if (++reinit_ctr >= REINIT_INTERVAL) {
            reinit_ctr = 0;
            rescan(i + 1 - window, i);
        }

        bool emit = SKIP ? ((int)nv >= min_periods && nv > 0)
                         : (nv == window && full_ok);
        size_t den = SKIP ? nv : window;
        double val = MEAN ? sum / (double)(den ? den : 1) : sum;
        dst[i] = emit ? val : std::numeric_limits<double>::quiet_NaN();
    }
}

void rolling_mean(const double* src, double* dst, size_t n,
                  size_t window, int min_periods, bool skip_nan,
                  int n_threads) {
    (void)n_threads;
    if (n == 0 || window == 0) {
        for (size_t i = 0; i < n; i++) dst[i] = DNAN;
        return;
    }
#if FW_SIMD
    if (!skip_nan && window >= 16 && min_periods <= (int)window) {
        run_sum_blocked(src, dst, n, window, min_periods,
                        /*mean_div=*/true, n_threads);
        return;
    }
#endif
    if (skip_nan) run_sum_kernel<true,  true>(src, dst, n, window, min_periods);
    else          run_sum_kernel<false, true>(src, dst, n, window, min_periods);
}

//rolling_var

void rolling_var(const double* src, double* dst, size_t n,
                 size_t window, int min_periods, bool ddof1, bool skip_nan,
                 int n_threads) {
    (void)n_threads;
#if FW_SIMD
    if (n > 0 && !skip_nan && window >= 16 && min_periods <= (int)window) {
        run_var_blocked(src, dst, n, window, min_periods, ddof1,
                        /*do_sqrt=*/false, n_threads);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) dst[i] = DNAN;
    if (n == 0 || window == 0) return;

    RollingBuffer<double> buf(window);
    RunningStats<double>  rs(window);
    size_t nan_count = 0;
    size_t reinit_ctr = 0;

    for (size_t i = 0; i < n; i++) {
        double x_new    = src[i];
        bool   x_new_ok = fw_isfinite(x_new);
        bool   was_full = buf.full();

        double x_old    = was_full ? buf.oldest() : 0.0;
        bool   x_old_ok = was_full && fw_isfinite(x_old);

        buf.push(x_new_ok ? x_new : DNAN);

        if (was_full && !x_old_ok) nan_count--;
        if (!x_new_ok)              nan_count++;

        if (!was_full) {
            if (x_new_ok) rs.grow(x_new);
        } else {
            if (x_new_ok && x_old_ok) {
                rs.slide(x_new, x_old);            //both finite: optimal path
            } else if (x_new_ok && !x_old_ok) {
                rs.grow(x_new);                    //evict NaN, add finite
            } else if (!x_new_ok && x_old_ok) {
                rs.remove(x_old);                  //evict finite, add NaN
            }
            //else both NaN: rs unchanged
        }

        //4096-step exact recompute suppresses sliding-Welford drift
        if (++reinit_ctr >= REINIT_INTERVAL) {
            reinit_ctr = 0;
            reinit_welford(buf, rs, nan_count);
        }

        if (skip_nan) {
            if ((int)rs.n >= min_periods) {
                double v = rs.variance(ddof1);
                if (fw_isfinite(v)) dst[i] = v;
            }
        } else {
            size_t total = buf.count;
            if (nan_count == 0 && (int)total >= min_periods) {
                double v = rs.variance(ddof1);
                if (fw_isfinite(v)) dst[i] = v;
            }
        }
    }
}

//rolling_std

void rolling_std(const double* src, double* dst, size_t n,
                 size_t window, int min_periods, bool ddof1, bool skip_nan,
                 int n_threads) {
    (void)n_threads;
#if FW_SIMD
    if (n > 0 && !skip_nan && window >= 16 && min_periods <= (int)window) {
        run_var_blocked(src, dst, n, window, min_periods, ddof1,
                        /*do_sqrt=*/true, n_threads);
        return;
    }
#endif
    rolling_var(src, dst, n, window, min_periods, ddof1, skip_nan);
    for (size_t i = 0; i < n; i++) {
        if (fw_isfinite(dst[i])) dst[i] = std::sqrt(dst[i]);
    }
}

//rolling_sum

void rolling_sum(const double* src, double* dst, size_t n,
                 size_t window, int min_periods, bool skip_nan,
                 int n_threads) {
    (void)n_threads;
    if (n == 0 || window == 0) {
        for (size_t i = 0; i < n; i++) dst[i] = DNAN;
        return;
    }
#if FW_SIMD
    if (!skip_nan && window >= 16 && min_periods <= (int)window) {
        run_sum_blocked(src, dst, n, window, min_periods,
                        /*mean_div=*/false, n_threads);
        return;
    }
#endif
    if (skip_nan) run_sum_kernel<true,  false>(src, dst, n, window, min_periods);
    else          run_sum_kernel<false, false>(src, dst, n, window, min_periods);
}

//rolling_min / rolling_max
//
//AVX2 builds use the van Herk–Gil-Werman blocked algorithm for window ≥ 16:
//the input is split into blocks of exactly `window` elements; within each
//block a suffix scan S and a forward prefix scan P are computed with SIMD
//in-register scans, and the window result is op(S[l], P[r]) — O(1) per
//element, O(window) extra memory, no branches in the steady state.
//Non-finite handling: blocks containing any non-finite value take a scalar
//"dirty" path with explicit NaN-presence tracking; clean blocks (the common
//case) skip that logic entirely.  Scalar builds and window < 16 use the
//monotonic deque.

#if FW_SIMD
namespace {

template <bool MIN>
static inline __m256d vop(__m256d a, __m256d b) {
    return MIN ? _mm256_min_pd(a, b) : _mm256_max_pd(a, b);
}

template <bool MIN>
static inline double sop(double a, double b) {
    return MIN ? (a < b ? a : b) : (a > b ? a : b);
}

using simd::bad_lanes;

//Processes blocks [k_lo, k_hi); when k_lo > 0 the loop starts one block
//early with emit disabled, so the seed Sprev is computed by the exact
//same code path — results are bitwise identical for any partition.
template <bool MIN>
static void minmax_blocked_range(const double* FW_RESTRICT x,
                                 double* FW_RESTRICT dst,
                                 size_t n, size_t w,
                                 size_t k_lo, size_t k_hi) {
    const double SENT = MIN ?  std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity();
    const double QNAN = std::numeric_limits<double>::quiet_NaN();
    const __m256d vsent = _mm256_set1_pd(SENT);

    //Sprev[w] = SENT handles the "window exactly equals current block" case
    std::vector<double>  Sprev(w + 1, SENT), Scur(w + 1, SENT);
    std::vector<uint8_t> nanSuf(w + 1, 0);   //built lazily for dirty blocks
    bool bad_prev = false;

    for (size_t k = (k_lo ? k_lo - 1 : 0); k < k_hi; k++) {
        const size_t s = k * w;
        const size_t L = std::min(w, n - s);

        //backward suffix scan of block k into Scur
        bool bad_cur = false;
        {
            size_t j = L;
            double carry = SENT;
            //scalar tail at the high end seeds the vector carry
            while (j > (L & ~size_t(3))) {
                j--;
                double xv = x[s + j];
                bool bad = !fw_isfinite(xv);
                bad_cur |= bad;
                carry = sop<MIN>(carry, bad ? SENT : xv);
                Scur[j] = carry;
            }
            __m256d vcarry = _mm256_set1_pd(carry);
            int badacc = 0;
            while (j >= 4) {
                j -= 4;
                __m256d v   = _mm256_loadu_pd(x + s + j);
                __m256d bad = bad_lanes(v);
                badacc |= _mm256_movemask_pd(bad);
                v = _mm256_blendv_pd(v, vsent, bad);
                //in-register suffix scan: t = op(v, [v1,v2,v3,SENT]) etc.
                __m256d t = vop<MIN>(v, _mm256_blend_pd(
                    _mm256_permute4x64_pd(v, _MM_SHUFFLE(3, 3, 2, 1)),
                    vsent, 0x8));
                t = vop<MIN>(t, _mm256_blend_pd(
                    _mm256_permute4x64_pd(t, _MM_SHUFFLE(3, 3, 3, 2)),
                    vsent, 0xC));
                t = vop<MIN>(t, vcarry);
                _mm256_storeu_pd(Scur.data() + j, t);
                vcarry = _mm256_permute4x64_pd(t, 0x00);   //broadcast lane 0
            }
            bad_cur |= (badacc != 0);
        }
        for (size_t j = L; j <= w; j++) Scur[j] = SENT;   //identity padding

        //emit pass over block k
        const size_t r_lo = std::max(s, w - 1);
        if (k >= k_lo && r_lo < s + L) {
            if (!bad_prev && !bad_cur) {
                if (k == 0) {
                    //only r = w-1 emits in block 0; its window IS the block
                    dst[w - 1] = Scur[0];
                } else {
                    //fast path: SIMD prefix scan fused with the combine.
                    //dst is written as one long sequential stream, so use
                    //non-temporal stores when aligned (skips the RFO).
                    size_t j = 0, r = s;
                    __m256d vcarry = vsent;
                    const bool nt =
                        (reinterpret_cast<uintptr_t>(dst + s) & 31) == 0;
                    for (; j + 4 <= L; j += 4, r += 4) {
                        __m256d v = _mm256_loadu_pd(x + r);
                        __m256d t = vop<MIN>(v, _mm256_blend_pd(
                            _mm256_permute4x64_pd(v, _MM_SHUFFLE(2, 1, 0, 0)),
                            vsent, 0x1));
                        t = vop<MIN>(t, _mm256_blend_pd(
                            _mm256_permute4x64_pd(t, _MM_SHUFFLE(1, 0, 0, 0)),
                            vsent, 0x3));
                        t = vop<MIN>(t, vcarry);
                        vcarry = _mm256_permute4x64_pd(t, 0xFF);
                        __m256d sv = _mm256_loadu_pd(Sprev.data() + j + 1);
                        __m256d o  = vop<MIN>(t, sv);
                        if (nt) _mm256_stream_pd(dst + r, o);
                        else    _mm256_storeu_pd(dst + r, o);
                    }
                    //scalar tail (carry = lane 0 of vcarry broadcast)
                    double pc = _mm_cvtsd_f64(
                        _mm256_castpd256_pd128(vcarry));
                    for (; j < L; j++, r++) {
                        pc = sop<MIN>(pc, x[r]);
                        dst[r] = sop<MIN>(pc, Sprev[j + 1]);
                    }
                }
            } else {
                //dirty path: scalar with explicit NaN-presence tracking
                if (bad_prev) {
                    bool acc = false;
                    for (size_t j = w; j-- > 0;) {
                        acc |= !fw_isfinite(x[s - w + j]);
                        nanSuf[j] = acc ? 1 : 0;
                    }
                    nanSuf[w] = 0;
                } else {
                    std::fill(nanSuf.begin(), nanSuf.end(), 0);
                }
                double P = SENT;
                bool nanP = false;
                for (size_t j = 0; j < L; j++) {
                    size_t r = s + j;
                    double xv = x[r];
                    bool bad = !fw_isfinite(xv);
                    nanP |= bad;
                    P = sop<MIN>(P, bad ? SENT : xv);
                    if (r < w - 1) continue;
                    dst[r] = (nanP || nanSuf[j + 1])
                        ? QNAN : sop<MIN>(P, Sprev[j + 1]);
                }
            }
        }

        Sprev.swap(Scur);
        bad_prev = bad_cur;
    }
    _mm_sfence();   //flush pending non-temporal stores
}

static int blk_threads(int n_threads) {
#ifdef _OPENMP
    return (n_threads <= 1) ? 1 : n_threads;   //0 = single-threaded
#else
    (void)n_threads;
    return 1;
#endif
}

template <bool MIN>
static void run_minmax_blocked(const double* FW_RESTRICT x,
                               double* FW_RESTRICT dst,
                               size_t n, size_t w, int n_threads) {
    const double QNAN = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i + 1 < w && i < n; i++) dst[i] = QNAN;
    const size_t nblocks = (n + w - 1) / w;
    const int nt = blk_threads(n_threads);
    if (nt > 1 && nblocks >= (size_t)(2 * nt)) {
#ifdef _OPENMP
        #pragma omp parallel num_threads(nt)
        {
            const size_t T   = (size_t)omp_get_num_threads();
            const size_t tid = (size_t)omp_get_thread_num();
            size_t lo = nblocks * tid / T, hi = nblocks * (tid + 1) / T;
            if (lo < hi) minmax_blocked_range<MIN>(x, dst, n, w, lo, hi);
        }
#endif
    } else {
        minmax_blocked_range<MIN>(x, dst, n, w, 0, nblocks);
    }
}

//Blocked additive scans — same van Herk structure with op = add.
//Window sums come from two block-local partial sums (each over ≤ w
//elements), so accuracy is better than the sliding update and no periodic
//reinitialisation is needed.  Used by mean/sum (one stream) and var/std
//(two streams: x and x²) when skip_nan=false and window ≥ 16.

using simd::scan_fwd_add;
using simd::scan_bwd_add;

/// Blocked rolling sum/mean (skip_nan = false).  mean_div selects division
/// by the window length; the warmup region is handled scalar so that
/// min_periods < window still emits partial results.
static void sum_blocked_range(const double* FW_RESTRICT x,
                              double* FW_RESTRICT dst,
                              size_t n, size_t w, bool mean_div,
                              size_t k_lo, size_t k_hi) {
    const double QNAN = std::numeric_limits<double>::quiet_NaN();
    const double scale = mean_div ? 1.0 / (double)w : 1.0;
    const __m256d vscale = _mm256_set1_pd(scale);

    std::vector<double>  Sprev(w + 1, 0.0), Scur(w + 1, 0.0);
    std::vector<uint8_t> nanSuf(w + 1, 0);
    bool bad_prev = false;

    for (size_t k = (k_lo ? k_lo - 1 : 0); k < k_hi; k++) {
        const size_t s = k * w;
        const size_t L = std::min(w, n - s);

        bool bad_cur = false;
        {   //backward suffix-sum scan (non-finite blended to 0, flagged)
            size_t j = L;
            double carry = 0.0;
            while (j > (L & ~size_t(3))) {
                j--;
                double xv = x[s + j];
                bool bad = !fw_isfinite(xv);
                bad_cur |= bad;
                carry += bad ? 0.0 : xv;
                Scur[j] = carry;
            }
            __m256d vcarry = _mm256_set1_pd(carry);
            int badacc = 0;
            while (j >= 4) {
                j -= 4;
                __m256d v   = _mm256_loadu_pd(x + s + j);
                __m256d bad = bad_lanes(v);
                badacc |= _mm256_movemask_pd(bad);
                v = _mm256_blendv_pd(v, _mm256_setzero_pd(), bad);
                _mm256_storeu_pd(Scur.data() + j, scan_bwd_add(v, vcarry));
            }
            bad_cur |= (badacc != 0);
        }
        for (size_t j = L; j <= w; j++) Scur[j] = 0.0;

        const size_t r_lo = std::max(s, w - 1);
        if (k >= k_lo && r_lo < s + L) {
            if (!bad_prev && !bad_cur) {
                if (k == 0) {
                    dst[w - 1] = Scur[0] * scale;
                } else {
                    size_t j = 0, r = s;
                    __m256d vcarry = _mm256_setzero_pd();
                    const bool nt =
                        (reinterpret_cast<uintptr_t>(dst + s) & 31) == 0;
                    for (; j + 4 <= L; j += 4, r += 4) {
                        __m256d v = _mm256_loadu_pd(x + r);
                        __m256d p = scan_fwd_add(v, vcarry);
                        __m256d o = _mm256_mul_pd(
                            _mm256_add_pd(p,
                                _mm256_loadu_pd(Sprev.data() + j + 1)),
                            vscale);
                        if (nt) _mm256_stream_pd(dst + r, o);
                        else    _mm256_storeu_pd(dst + r, o);
                    }
                    double pc = _mm_cvtsd_f64(
                        _mm256_castpd256_pd128(vcarry));
                    for (; j < L; j++, r++) {
                        pc += x[r];
                        dst[r] = (pc + Sprev[j + 1]) * scale;
                    }
                }
            } else {
                if (bad_prev) {
                    bool acc = false;
                    for (size_t j = w; j-- > 0;) {
                        acc |= !fw_isfinite(x[s - w + j]);
                        nanSuf[j] = acc ? 1 : 0;
                    }
                    nanSuf[w] = 0;
                } else {
                    std::fill(nanSuf.begin(), nanSuf.end(), 0);
                }
                double P = 0.0;
                bool nanP = false;
                for (size_t j = 0; j < L; j++) {
                    size_t r = s + j;
                    double xv = x[r];
                    bool bad = !fw_isfinite(xv);
                    nanP |= bad;
                    P += bad ? 0.0 : xv;
                    if (r < w - 1) continue;
                    dst[r] = (nanP || nanSuf[j + 1])
                        ? QNAN : (P + Sprev[j + 1]) * scale;
                }
            }
        }

        Sprev.swap(Scur);
        bad_prev = bad_cur;
    }
    _mm_sfence();
}

static void run_sum_blocked(const double* FW_RESTRICT x,
                            double* FW_RESTRICT dst,
                            size_t n, size_t w, int min_periods,
                            bool mean_div, int n_threads) {
    const double QNAN = std::numeric_limits<double>::quiet_NaN();
    {   //scalar warmup: positions 0 .. w-2 (growing window)
        double sum = 0.0;
        size_t nv = 0;
        size_t lim = std::min(n, w - 1);
        for (size_t i = 0; i < lim; i++) {
            double v = x[i];
            bool ok = fw_isfinite(v);
            sum += ok ? v : 0.0;
            nv  += static_cast<size_t>(ok);
            size_t m = i + 1;
            bool emit = (nv == m && (int)m >= min_periods);
            dst[i] = emit ? (mean_div ? sum / (double)m : sum) : QNAN;
        }
        if (n < w) return;
    }
    const size_t nblocks = (n + w - 1) / w;
    const int nt = blk_threads(n_threads);
    if (nt > 1 && nblocks >= (size_t)(2 * nt)) {
#ifdef _OPENMP
        #pragma omp parallel num_threads(nt)
        {
            const size_t T   = (size_t)omp_get_num_threads();
            const size_t tid = (size_t)omp_get_thread_num();
            size_t lo = nblocks * tid / T, hi = nblocks * (tid + 1) / T;
            if (lo < hi) sum_blocked_range(x, dst, n, w, mean_div, lo, hi);
        }
#endif
    } else {
        sum_blocked_range(x, dst, n, w, mean_div, 0, nblocks);
    }
}

/// Blocked rolling variance/std (skip_nan = false): two additive streams
/// (x and x²); var = (Σx² − (Σx)²/w) / (w − ddof), clamped at 0; std takes
/// _mm256_sqrt_pd in the same write pass.
static void var_blocked_range(const double* FW_RESTRICT x,
                              double* FW_RESTRICT dst,
                              size_t n, size_t w,
                              bool ddof1, bool do_sqrt,
                              size_t k_lo, size_t k_hi) {
    const double QNAN = std::numeric_limits<double>::quiet_NaN();
    const double dw    = (double)w;
    const double inv_w = 1.0 / dw;
    const double denom = dw - (ddof1 ? 1.0 : 0.0);
    const double inv_d = 1.0 / denom;
    const __m256d vinvw = _mm256_set1_pd(inv_w);
    const __m256d vinvd = _mm256_set1_pd(inv_d);
    const __m256d vzero = _mm256_setzero_pd();

    std::vector<double>  S1p(w + 1, 0.0), S1c(w + 1, 0.0);   //Σx suffix
    std::vector<double>  S2p(w + 1, 0.0), S2c(w + 1, 0.0);   //Σx² suffix
    std::vector<uint8_t> nanSuf(w + 1, 0);
    bool bad_prev = false;

    for (size_t k = (k_lo ? k_lo - 1 : 0); k < k_hi; k++) {
        const size_t s = k * w;
        const size_t L = std::min(w, n - s);

        bool bad_cur = false;
        {
            size_t j = L;
            double c1 = 0.0, c2 = 0.0;
            while (j > (L & ~size_t(3))) {
                j--;
                double xv = x[s + j];
                bool bad = !fw_isfinite(xv);
                bad_cur |= bad;
                double cv = bad ? 0.0 : xv;
                c1 += cv;
                c2 += cv * cv;
                S1c[j] = c1;
                S2c[j] = c2;
            }
            __m256d vc1 = _mm256_set1_pd(c1), vc2 = _mm256_set1_pd(c2);
            int badacc = 0;
            while (j >= 4) {
                j -= 4;
                __m256d v   = _mm256_loadu_pd(x + s + j);
                __m256d bad = bad_lanes(v);
                badacc |= _mm256_movemask_pd(bad);
                v = _mm256_blendv_pd(v, vzero, bad);
                _mm256_storeu_pd(S1c.data() + j, scan_bwd_add(v, vc1));
                _mm256_storeu_pd(S2c.data() + j,
                                 scan_bwd_add(_mm256_mul_pd(v, v), vc2));
            }
            bad_cur |= (badacc != 0);
        }
        for (size_t j = L; j <= w; j++) { S1c[j] = 0.0; S2c[j] = 0.0; }

        const size_t r_lo = std::max(s, w - 1);
        if (k >= k_lo && r_lo < s + L) {
            if (!bad_prev && !bad_cur) {
                if (k == 0) {
                    double sx = S1c[0], sxx = S2c[0];
                    double var = (sxx - sx * sx * inv_w) * inv_d;
                    if (var < 0.0) var = 0.0;
                    dst[w - 1] = do_sqrt ? std::sqrt(var) : var;
                } else {
                    size_t j = 0, r = s;
                    __m256d vc1 = vzero, vc2 = vzero;
                    const bool nt =
                        (reinterpret_cast<uintptr_t>(dst + s) & 31) == 0;
                    for (; j + 4 <= L; j += 4, r += 4) {
                        __m256d v  = _mm256_loadu_pd(x + r);
                        __m256d p1 = scan_fwd_add(v, vc1);
                        __m256d p2 = scan_fwd_add(_mm256_mul_pd(v, v), vc2);
                        __m256d sx = _mm256_add_pd(p1,
                            _mm256_loadu_pd(S1p.data() + j + 1));
                        __m256d sxx = _mm256_add_pd(p2,
                            _mm256_loadu_pd(S2p.data() + j + 1));
                        __m256d var = _mm256_mul_pd(
                            _mm256_sub_pd(sxx, _mm256_mul_pd(
                                _mm256_mul_pd(sx, sx), vinvw)),
                            vinvd);
                        var = _mm256_max_pd(var, vzero);
                        if (do_sqrt) var = _mm256_sqrt_pd(var);
                        if (nt) _mm256_stream_pd(dst + r, var);
                        else    _mm256_storeu_pd(dst + r, var);
                    }
                    double c1 = _mm_cvtsd_f64(_mm256_castpd256_pd128(vc1));
                    double c2 = _mm_cvtsd_f64(_mm256_castpd256_pd128(vc2));
                    for (; j < L; j++, r++) {
                        c1 += x[r];
                        c2 += x[r] * x[r];
                        double sx  = c1 + S1p[j + 1];
                        double sxx = c2 + S2p[j + 1];
                        double var = (sxx - sx * sx * inv_w) * inv_d;
                        if (var < 0.0) var = 0.0;
                        dst[r] = do_sqrt ? std::sqrt(var) : var;
                    }
                }
            } else {
                if (bad_prev) {
                    bool acc = false;
                    for (size_t j = w; j-- > 0;) {
                        acc |= !fw_isfinite(x[s - w + j]);
                        nanSuf[j] = acc ? 1 : 0;
                    }
                    nanSuf[w] = 0;
                } else {
                    std::fill(nanSuf.begin(), nanSuf.end(), 0);
                }
                double P1 = 0.0, P2 = 0.0;
                bool nanP = false;
                for (size_t j = 0; j < L; j++) {
                    size_t r = s + j;
                    double xv = x[r];
                    bool bad = !fw_isfinite(xv);
                    nanP |= bad;
                    double cv = bad ? 0.0 : xv;
                    P1 += cv;
                    P2 += cv * cv;
                    if (r < w - 1) continue;
                    if (nanP || nanSuf[j + 1]) {
                        dst[r] = QNAN;
                    } else {
                        double sx  = P1 + S1p[j + 1];
                        double sxx = P2 + S2p[j + 1];
                        double var = (sxx - sx * sx * inv_w) * inv_d;
                        if (var < 0.0) var = 0.0;
                        dst[r] = do_sqrt ? std::sqrt(var) : var;
                    }
                }
            }
        }

        S1p.swap(S1c);
        S2p.swap(S2c);
        bad_prev = bad_cur;
    }
    _mm_sfence();
}

static void run_var_blocked(const double* FW_RESTRICT x,
                            double* FW_RESTRICT dst,
                            size_t n, size_t w, int min_periods,
                            bool ddof1, bool do_sqrt, int n_threads) {
    const double QNAN = std::numeric_limits<double>::quiet_NaN();
    {   //scalar warmup via Welford (matches the scalar kernel's gating)
        RunningStats<double> rs(w);
        size_t nan_count = 0;
        size_t lim = std::min(n, w - 1);
        for (size_t i = 0; i < lim; i++) {
            double v = x[i];
            if (fw_isfinite(v)) rs.grow(v);
            else                nan_count++;
            double out = QNAN;
            if (nan_count == 0 && (int)(i + 1) >= min_periods) {
                double var = rs.variance(ddof1);
                if (fw_isfinite(var)) out = do_sqrt ? std::sqrt(var) : var;
            }
            dst[i] = out;
        }
        if (n < w) return;
    }
    const size_t nblocks = (n + w - 1) / w;
    const int nt = blk_threads(n_threads);
    if (nt > 1 && nblocks >= (size_t)(2 * nt)) {
#ifdef _OPENMP
        #pragma omp parallel num_threads(nt)
        {
            const size_t T   = (size_t)omp_get_num_threads();
            const size_t tid = (size_t)omp_get_thread_num();
            size_t lo = nblocks * tid / T, hi = nblocks * (tid + 1) / T;
            if (lo < hi)
                var_blocked_range(x, dst, n, w, ddof1, do_sqrt, lo, hi);
        }
#endif
    } else {
        var_blocked_range(x, dst, n, w, ddof1, do_sqrt, 0, nblocks);
    }
}

} //anonymous namespace
#endif //FW_SIMD

void rolling_min(const double* src, double* dst, size_t n, size_t window,
                 int n_threads) {
    (void)n_threads;
#if FW_SIMD
    if (window >= 16 && n >= window) {
        run_minmax_blocked<true>(src, dst, n, window, n_threads);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) dst[i] = DNAN;
    if (n == 0 || window == 0) return;

    MonotonicDeque dq;
    size_t nan_count = 0;

    for (size_t i = 0; i < n; i++) {
        //NaN bookkeeping for the departing element
        if (i >= window) {
            if (!fw_isfinite(src[i - window])) nan_count--;
        }

        if (!fw_isfinite(src[i])) {
            //NaN / Inf: don't add to deque, bump nan counter
            nan_count++;
        } else {
            //Maintain ascending deque (front = current min)
            while (!dq.empty() && src[dq.idx.back()] >= src[i])
                dq.idx.pop_back();
            dq.idx.push_back(i);
        }

        while (!dq.empty() && dq.front() + window <= i)
            dq.idx.pop_front();

        if (i + 1 >= window) {
            if (nan_count == 0 && !dq.empty())
                dst[i] = src[dq.front()];
            //else: NaN in window → dst[i] stays NaN
        }
    }
}

void rolling_max(const double* src, double* dst, size_t n, size_t window,
                 int n_threads) {
    (void)n_threads;
#if FW_SIMD
    if (window >= 16 && n >= window) {
        run_minmax_blocked<false>(src, dst, n, window, n_threads);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) dst[i] = DNAN;
    if (n == 0 || window == 0) return;

    MonotonicDeque dq;
    size_t nan_count = 0;

    for (size_t i = 0; i < n; i++) {
        if (i >= window) {
            if (!fw_isfinite(src[i - window])) nan_count--;
        }

        if (!fw_isfinite(src[i])) {
            nan_count++;
        } else {
            //Maintain descending deque (front = current max)
            while (!dq.empty() && src[dq.idx.back()] <= src[i])
                dq.idx.pop_back();
            dq.idx.push_back(i);
        }

        while (!dq.empty() && dq.front() + window <= i)
            dq.idx.pop_front();

        if (i + 1 >= window) {
            if (nan_count == 0 && !dq.empty())
                dst[i] = src[dq.front()];
        }
    }
}

} //namespace fastwindow
