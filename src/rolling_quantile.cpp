/// @file rolling_quantile.cpp
/// Rolling quantile.
///
/// exact=false: P² streaming approximation (Jain & Chlamtac, 1985) — five
/// markers tracking {min, q/2, q, (1+q)/2, max}, O(1) per step, no window
/// buffer.  Note that P² estimates the quantile of the whole stream seen so
/// far; for stationary series this tracks the window quantile well, which is
/// the documented approximation contract.
///
/// exact=true: two max/min heaps with lazy deletion straddling the target
/// order statistic, O(log window) amortised per step; linear interpolation
/// identical to numpy.percentile.  Any window size.
///
/// NaN policy (both modes): non-finite values are never fed to the
/// estimator, and any non-finite value inside the current window forces a
/// NaN output until it slides out.
#include "fastwindow.h"

#include <algorithm>
#include <queue>
#include <unordered_map>

namespace fastwindow {

static constexpr double DNAN = std::numeric_limits<double>::quiet_NaN();

//P² marker state

namespace {

struct P2State {
    double q;
    double h[5];      //marker heights
    double pos[5];    //actual marker positions (1-based counts)
    double dpos[5];   //desired positions
    double incr[5];   //desired-position increments per observation
    int    count = 0; //observations absorbed (5 needed to initialise)
    double init_buf[5];

    explicit P2State(double q_) : q(q_) {
        incr[0] = 0.0;
        incr[1] = q / 2.0;
        incr[2] = q;
        incr[3] = (1.0 + q) / 2.0;
        incr[4] = 1.0;
    }

    double parabolic(int i, double s) const {
        return h[i] + s / (pos[i + 1] - pos[i - 1]) *
            ((pos[i] - pos[i - 1] + s) * (h[i + 1] - h[i]) /
                 (pos[i + 1] - pos[i]) +
             (pos[i + 1] - pos[i] - s) * (h[i] - h[i - 1]) /
                 (pos[i] - pos[i - 1]));
    }

    double linear(int i, int s) const {
        return h[i] + s * (h[i + s] - h[i]) / (pos[i + s] - pos[i]);
    }

    void add(double x) {
        if (count < 5) {
            init_buf[count++] = x;
            if (count == 5) {
                std::sort(init_buf, init_buf + 5);
                for (int i = 0; i < 5; i++) {
                    h[i]   = init_buf[i];
                    pos[i] = i + 1;
                }
                dpos[0] = 1.0;
                dpos[1] = 1.0 + 2.0 * q;
                dpos[2] = 1.0 + 4.0 * q;
                dpos[3] = 3.0 + 2.0 * q;
                dpos[4] = 5.0;
            }
            return;
        }
        count++;

        int k;
        if (x < h[0])       { h[0] = x; k = 0; }
        else if (x >= h[4]) { h[4] = x; k = 3; }
        else { k = 0; while (k < 3 && x >= h[k + 1]) k++; }

        for (int i = k + 1; i < 5; i++) pos[i] += 1.0;
        for (int i = 0; i < 5; i++)     dpos[i] += incr[i];

        for (int i = 1; i <= 3; i++) {
            double d = dpos[i] - pos[i];
            if ((d >= 1.0 && pos[i + 1] - pos[i] > 1.0) ||
                (d <= -1.0 && pos[i - 1] - pos[i] < -1.0)) {
                double s = (d >= 0.0) ? 1.0 : -1.0;
                double hp = parabolic(i, s);
                if (h[i - 1] < hp && hp < h[i + 1]) h[i] = hp;
                else h[i] = linear(i, (int)s);
                pos[i] += s;
            }
        }
    }

    /// Current estimate.  For fewer than 5 observations: exact quantile of
    /// the values seen so far (sorted copy, linear interpolation).
    double value() const {
        if (count == 0) return DNAN;
        if (count < 5) {
            double tmp[5];
            std::copy(init_buf, init_buf + count, tmp);
            std::sort(tmp, tmp + count);
            double rank = q * (count - 1);
            int lo = (int)rank;
            double w = rank - lo;
            return (lo + 1 < count) ? tmp[lo] + w * (tmp[lo + 1] - tmp[lo])
                                    : tmp[lo];
        }
        return h[2];
    }
};

} //anonymous namespace

//Kernel

void rolling_quantile(
        const double* src, double* dst,
        size_t n, size_t window, double q, int min_periods, bool exact) {
    for (size_t i = 0; i < n; i++) dst[i] = DNAN;
    if (n == 0 || window == 0 || !(q > 0.0 && q < 1.0)) return;

    size_t nan_count = 0;

    if (!exact) {
        P2State p2(q);
        for (size_t i = 0; i < n; i++) {
            if (i >= window && !fw_isfinite(src[i - window])) nan_count--;
            if (fw_isfinite(src[i])) p2.add(src[i]);
            else                     nan_count++;

            size_t m = std::min(i + 1, window);
            if (nan_count == 0 && (int)m >= min_periods)
                dst[i] = p2.value();
        }
    } else {
        //Two heaps with lazy deletion: lo (max-heap) holds the k+1
        //smallest valid values, hi (min-heap) the rest, so the k-th and
        //(k+1)-th order statistics are the two tops — O(log w) amortised
        //per step.  Deletions are deferred via a counter map and resolved
        //when a dead value surfaces at a top.  Monotone inputs can strand
        //dead entries deep in a heap, so the structure is rebuilt from the
        //source window whenever its physical size exceeds 3× the logical
        //one — amortised O(log w) overall.
        std::priority_queue<double> lo;
        std::priority_queue<double, std::vector<double>,
                            std::greater<double>> hi;
        std::unordered_map<uint64_t, int> dead;
        size_t lo_n = 0, hi_n = 0;

        auto key = [](double v) {
            uint64_t u;
            __builtin_memcpy(&u, &v, sizeof(u));
            if (u == 0x8000000000000000ULL) u = 0;   //-0.0 and +0.0 are equal
            return u;
        };
        auto prune_lo = [&] {
            while (!lo.empty()) {
                auto it = dead.find(key(lo.top()));
                if (it == dead.end()) break;
                if (--it->second == 0) dead.erase(it);
                lo.pop();
            }
        };
        auto prune_hi = [&] {
            while (!hi.empty()) {
                auto it = dead.find(key(hi.top()));
                if (it == dead.end()) break;
                if (--it->second == 0) dead.erase(it);
                hi.pop();
            }
        };
        auto insert = [&](double v) {
            //membership by boundary: when lo is logically empty the
            //comparison must fall back to hi's minimum, or the heap
            //ordering invariant (max lo <= min hi) breaks
            prune_lo();
            bool to_lo;
            if (lo_n > 0) {
                to_lo = v <= lo.top();
            } else {
                prune_hi();
                to_lo = (hi_n == 0) || v <= hi.top();
            }
            if (to_lo) { lo.push(v); lo_n++; }
            else       { hi.push(v); hi_n++; }
        };
        auto erase = [&](double v) {
            prune_lo();
            dead[key(v)]++;
            //equal values may straddle the boundary; either side works
            //numerically because the straddling values are identical
            if (lo_n > 0 && v <= lo.top()) lo_n--;
            else                           hi_n--;
        };
        auto rebalance = [&](size_t target_lo) {
            while (lo_n > target_lo) {
                prune_lo();
                double t = lo.top(); lo.pop(); hi.push(t);
                lo_n--; hi_n++;
            }
            while (lo_n < target_lo) {
                prune_hi();
                double t = hi.top(); hi.pop(); lo.push(t);
                hi_n--; lo_n++;
            }
        };

        for (size_t i = 0; i < n; i++) {
            if (i >= window) {
                double xo = src[i - window];
                if (fw_isfinite(xo)) erase(xo);
                else                 nan_count--;
            }
            double xn = src[i];
            if (fw_isfinite(xn)) insert(xn);
            else                 nan_count++;

            if (lo.size() + hi.size() > 3 * (lo_n + hi_n) + 16) {
                //rebuild from the source window to evict stranded entries
                lo = {}; hi = {}; dead.clear();
                lo_n = hi_n = 0;
                size_t start = (i + 1 >= window) ? i + 1 - window : 0;
                for (size_t j = start; j <= i; j++)
                    if (fw_isfinite(src[j])) { lo.push(src[j]); lo_n++; }
            }

            size_t m = std::min(i + 1, window);
            if (nan_count == 0 && (int)m >= min_periods && lo_n + hi_n > 0) {
                size_t cnt = lo_n + hi_n;   //== m when nan_count == 0
                double rank = q * (double)(cnt - 1);
                size_t k = (size_t)rank;
                double frac = rank - (double)k;
                rebalance(k + 1);
                prune_lo();
                double vk = lo.top();
                if (frac > 0.0 && k + 1 < cnt) {
                    prune_hi();
                    dst[i] = vk + frac * (hi.top() - vk);
                } else {
                    dst[i] = vk;
                }
            }
        }
    }
}

} //namespace fastwindow
