/// @file rolling_regression.cpp
/// Rolling simple linear regression (k=1).
/// time_as_x variant: x = 0,1,...,window-1 within each window, so Σx and Σx²
/// are constants once the window is full and the cross sum Σ(j·y_j) shifts in
/// O(1) per step.  Explicit-x variant uses the standard five running sums.
/// All sums are recomputed from the window buffer every 4096 steps.
#include "fastwindow.h"

namespace fastwindow {

static constexpr size_t REINIT_INTERVAL = 4096;
static constexpr double DNAN  = std::numeric_limits<double>::quiet_NaN();
static constexpr double DENOM_EPS = 1e-12;

//RunningRegression

void RunningRegression::grow(double x, double y) {
    n++;
    sum_x  += x;
    sum_y  += y;
    sum_xx += x * x;
    sum_xy += x * y;
    sum_yy += y * y;
}

void RunningRegression::add(double x, double y, double x_old, double y_old) {
    sum_x  += x - x_old;
    sum_y  += y - y_old;
    sum_xx += x * x - x_old * x_old;
    sum_xy += x * y - x_old * y_old;
    sum_yy += y * y - y_old * y_old;
}

void RunningRegression::reset() {
    sum_x = sum_y = sum_xx = sum_xy = sum_yy = 0.0;
    n = 0;
}

void RunningRegression::coeffs(double& b0, double& b1) const {
    double dn    = static_cast<double>(n);
    double denom = dn * sum_xx - sum_x * sum_x;
    if (n < 2 || !(denom > DENOM_EPS)) {  //degenerate or near-singular design
        b0 = DNAN; b1 = DNAN;
        return;
    }
    b1 = (dn * sum_xy - sum_x * sum_y) / denom;
    b0 = (sum_y - b1 * sum_x) / dn;
}

double RunningRegression::r_squared() const {
    double dn  = static_cast<double>(n);
    double sxx = dn * sum_xx - sum_x * sum_x;
    double syy = dn * sum_yy - sum_y * sum_y;
    double sxy = dn * sum_xy - sum_x * sum_y;
    double den = sxx * syy;
    if (n < 2 || !(den > DENOM_EPS)) return DNAN;  //constant x or constant y
    double r2 = (sxy * sxy) / den;
    if (r2 > 1.0) r2 = 1.0;  //clamp floating-point overshoot
    if (r2 < 0.0) r2 = 0.0;
    return r2;
}

//rolling_simple_regression (x = window position 0..w-1)
//
//Sliding update of Sjy = Σ j·y_j (j = 0-based position within window):
//when the window shifts, every surviving point's position decrements, so
//Sjy_new = Sjy_old - (Sy_old - y_evicted) + (w-1)·y_new
//NaN values enter the buffer and sums as 0.0 with nan_count tracking; the
//substitution keeps the shift identity exact, and outputs are NaN while
//nan_count > 0, so the zeros are never observable.

void rolling_simple_regression(
        const double* y, double* b0, double* b1, double* r2,
        size_t n, size_t window, int min_periods, bool time_as_x) {
    for (size_t i = 0; i < n; i++) {
        b0[i] = DNAN;
        b1[i] = DNAN;
        if (r2) r2[i] = DNAN;
    }
    if (n == 0 || window == 0 || !time_as_x) return;

    RollingBuffer<double> buf(window);
    double Sy = 0.0, Syy = 0.0, Sjy = 0.0;
    size_t nan_count = 0;
    size_t reinit_ctr = 0;

    for (size_t i = 0; i < n; i++) {
        bool   y_ok  = fw_isfinite(y[i]);
        double y_new = y_ok ? y[i] : 0.0;   //NaN substituted as 0, tracked below

        if (buf.full()) {
            double y_old    = buf.oldest();
            bool   y_old_nan = !fw_isfinite(y[i - window]);
            //Position shift: subtract the post-eviction window sum (Sy - y_old)
            Sjy -= (Sy - y_old);
            Sy  -= y_old;
            Syy -= y_old * y_old;
            Sjy += static_cast<double>(window - 1) * y_new;
            if (y_old_nan) nan_count--;
        } else {
            //Growing phase: new point lands at position count (0-based)
            Sjy += static_cast<double>(buf.count) * y_new;
        }
        Sy  += y_new;
        Syy += y_new * y_new;
        buf.push(y_new);
        if (!y_ok) nan_count++;

        //Periodic exact recompute to suppress floating-point drift
        if (++reinit_ctr >= REINIT_INTERVAL) {
            reinit_ctr = 0;
            Sy = Syy = Sjy = 0.0;
            for (size_t j = 0; j < buf.count; j++) {
                double v = buf.data[(buf.head + j) % buf.capacity];
                Sy  += v;
                Syy += v * v;
                Sjy += static_cast<double>(j) * v;
            }
        }

        size_t m = buf.count;
        if (nan_count > 0 || (int)m < min_periods || m < 2) continue;

        //Closed-form Σx, Σx² for x = 0..m-1
        double dm = static_cast<double>(m);
        double Sx  = dm * (dm - 1.0) / 2.0;
        double Sxx = (dm - 1.0) * dm * (2.0 * dm - 1.0) / 6.0;

        double denom = dm * Sxx - Sx * Sx;
        if (!(denom > DENOM_EPS)) continue;
        double slope     = (dm * Sjy - Sx * Sy) / denom;
        double intercept = (Sy - slope * Sx) / dm;
        b1[i] = slope;
        b0[i] = intercept;

        if (r2) {
            double syy = dm * Syy - Sy * Sy;
            double sxy = dm * Sjy - Sx * Sy;
            double den = denom * syy;
            if (den > DENOM_EPS) {
                double v = (sxy * sxy) / den;
                if (v > 1.0) v = 1.0;
                if (v < 0.0) v = 0.0;
                r2[i] = v;
            }
        }
    }
}

//rolling_simple_regression_xy (explicit regressor series)
//
//A pair is "valid" iff both x and y are finite; invalid pairs enter the
//buffers and sums as (0,0) with nan_count tracking, and outputs stay NaN
//while any invalid pair is inside the window.

void rolling_simple_regression_xy(
        const double* x, const double* y,
        double* b0, double* b1, double* r2,
        size_t n, size_t window, int min_periods) {
    for (size_t i = 0; i < n; i++) {
        b0[i] = DNAN;
        b1[i] = DNAN;
        if (r2) r2[i] = DNAN;
    }
    if (n == 0 || window == 0) return;

    RollingBuffer<double> bufx(window), bufy(window);
    RunningRegression rr;
    size_t nan_count = 0;
    size_t reinit_ctr = 0;

    for (size_t i = 0; i < n; i++) {
        bool   ok    = fw_isfinite(x[i]) && fw_isfinite(y[i]);
        double x_new = ok ? x[i] : 0.0;
        double y_new = ok ? y[i] : 0.0;

        if (bufx.full()) {
            double x_old = bufx.oldest();
            double y_old = bufy.oldest();
            bool old_invalid =
                !(fw_isfinite(x[i - window]) && fw_isfinite(y[i - window]));
            rr.add(x_new, y_new, x_old, y_old);
            if (old_invalid) nan_count--;
        } else {
            rr.grow(x_new, y_new);
        }
        bufx.push(x_new);
        bufy.push(y_new);
        if (!ok) nan_count++;

        if (++reinit_ctr >= REINIT_INTERVAL) {
            reinit_ctr = 0;
            size_t m = bufx.count;
            rr.reset();
            for (size_t j = 0; j < m; j++) {
                double vx = bufx.data[(bufx.head + j) % bufx.capacity];
                double vy = bufy.data[(bufy.head + j) % bufy.capacity];
                rr.grow(vx, vy);
            }
        }

        size_t m = bufx.count;
        if (nan_count > 0 || (int)m < min_periods || m < 2) continue;

        double intercept, slope;
        rr.coeffs(intercept, slope);
        b0[i] = intercept;
        b1[i] = slope;
        if (r2) r2[i] = rr.r_squared();
    }
}

} //namespace fastwindow
