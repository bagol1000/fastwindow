/// @file rolling_spearman.cpp
/// Rolling Spearman rank correlation (average ranks for ties, matching
/// scipy.stats.spearmanr and R's cor(method = "spearman")).
///
/// Two sorted arrays of (value, time-slot) entries are maintained by
/// binary-search insertion/removal, O(window) per step.  At each output
/// position the rank statistics are produced by three LINEAR passes:
/// one walk over sorted-y assigning average ranks per time slot, one walk
/// over sorted-x accumulating Σrx² and Σrx·ry through the slot table, and
/// the Pearson finalisation.  With average ranks, Σrx = Σry = m(m+1)/2
/// exactly, so those never need computing.  No per-element binary searches
/// — this replaced an O(w log w) variant and is ~20× faster in practice.
///
/// A pair is valid iff both values are finite; any invalid pair inside the
/// window forces NaN output until it slides out (the walks are skipped for
/// such positions, so NaN stretches cost O(1) per step).
#include "fastwindow.h"

#include <algorithm>
#include <utility>

namespace fastwindow {

static constexpr double DNAN = std::numeric_limits<double>::quiet_NaN();

namespace {

using Entry = std::pair<double, int>;   //(value, time slot)

struct ByValue {
    bool operator()(const Entry& a, double v) const { return a.first < v; }
    bool operator()(double v, const Entry& a) const { return v < a.first; }
};

void sorted_insert(std::vector<Entry>& v, double val, int slot) {
    auto it = std::lower_bound(v.begin(), v.end(), val, ByValue{});
    v.insert(it, {val, slot});
}

void sorted_erase(std::vector<Entry>& v, double val, int slot) {
    auto it = std::lower_bound(v.begin(), v.end(), val, ByValue{});
    while (it->second != slot) ++it;    //scan the tie run for our slot
    v.erase(it);
}

} //anonymous namespace

void rolling_spearman(
        const double* x, const double* y, double* dst,
        size_t n, size_t window, int min_periods) {
    for (size_t i = 0; i < n; i++) dst[i] = DNAN;
    if (n == 0 || window == 0) return;

    std::vector<Entry> sx, sy;
    sx.reserve(window);
    sy.reserve(window);
    std::vector<double> ry_of_slot(window);
    size_t nan_count = 0;

    auto pair_valid = [&](size_t i) {
        return fw_isfinite(x[i]) && fw_isfinite(y[i]);
    };

    for (size_t i = 0; i < n; i++) {
        int slot = (int)(i % window);
        if (i >= window) {
            size_t o = i - window;
            if (pair_valid(o)) {
                sorted_erase(sx, x[o], slot);
                sorted_erase(sy, y[o], slot);
            } else {
                nan_count--;
            }
        }
        if (pair_valid(i)) {
            sorted_insert(sx, x[i], slot);
            sorted_insert(sy, y[i], slot);
        } else {
            nan_count++;
        }

        size_t m = std::min(i + 1, window);
        if (nan_count > 0 || (int)m < min_periods || m < 2)
            continue;

        //pass 1: average y-ranks per time slot + Σry²  (tie runs share
        //the mean of the 1-based positions they occupy)
        double sryy = 0.0;
        for (size_t a = 0; a < m; ) {
            size_t b = a + 1;
            while (b < m && sy[b].first == sy[a].first) b++;
            double rank = 0.5 * (double)(a + b + 1);   //mean of a+1 .. b
            for (size_t t = a; t < b; t++) ry_of_slot[sy[t].second] = rank;
            sryy += (double)(b - a) * rank * rank;
            a = b;
        }

        //pass 2: Σrx² and Σrx·ry through the slot table
        double srxx = 0.0, srxy = 0.0;
        for (size_t a = 0; a < m; ) {
            size_t b = a + 1;
            while (b < m && sx[b].first == sx[a].first) b++;
            double rank = 0.5 * (double)(a + b + 1);
            double ry_sum = 0.0;
            for (size_t t = a; t < b; t++) ry_sum += ry_of_slot[sx[t].second];
            srxx += (double)(b - a) * rank * rank;
            srxy += rank * ry_sum;
            a = b;
        }

        //Pearson on ranks; Σrx = Σry = m(m+1)/2 holds exactly under
        //average ranking, ties included
        double dm = (double)m;
        double sr = dm * (dm + 1.0) * 0.5;
        double vx  = dm * srxx - sr * sr;
        double vy  = dm * sryy - sr * sr;
        double cxy = dm * srxy - sr * sr;
        if (!(vx > 1e-12) || !(vy > 1e-12))   //all-tied ranks → undefined
            continue;
        double r = cxy / std::sqrt(vx * vy);
        if (r >  1.0) r =  1.0;
        if (r < -1.0) r = -1.0;
        dst[i] = r;
    }
}

} //namespace fastwindow
