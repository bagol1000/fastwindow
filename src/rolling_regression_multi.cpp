/// @file rolling_regression_multi.cpp
/// Rolling multiple OLS regression.
///
/// (X'X)⁻¹ is maintained incrementally with Sherman-Morrison rank-1 updates,
/// O(k²) per step.  X'X itself is also maintained (exact rank-1 sums), so
/// whenever an update denominator falls below 1e-12 the inverse is rebuilt
/// from X'X by Cholesky for that step only.  Everything is recomputed from
/// the window buffers every 4096 steps.
///
/// The kernel is a single template dispatched on the design dimension
/// p = k+1: for k ≤ 4 it is instantiated with a compile-time P, so every
/// matrix loop has constexpr bounds and is fully unrolled by the compiler
/// (the spec's "explicit unrolled ops" requirement); k > 4 uses the P=0
/// instantiation with runtime-dimension nested loops.
#include "fastwindow.h"

#include <algorithm>

namespace fastwindow {

static constexpr size_t REINIT_INTERVAL = 4096;
static constexpr double DNAN      = std::numeric_limits<double>::quiet_NaN();
static constexpr double SM_EPS    = 1e-12;   //Woodbury denominator guard
static constexpr int    FW_MAX_P  = FW_MAX_REGRESSORS + 1;  //17

//Cholesky inversion of a p×p SPD matrix (row-major).  Returns false when
//the matrix is not positive definite (singular / degenerate window).
//Cold path: used for initialisation, the per-step Sherman-Morrison
//fallback, and the 4096-step reinit.

static bool cholesky_invert(const double* A, double* Ainv, int p) {
    double L[FW_MAX_P * FW_MAX_P];

    double diag_scale = 1.0;
    for (int i = 0; i < p; i++)
        diag_scale = std::max(diag_scale, std::abs(A[i * p + i]));
    const double piv_eps = diag_scale * 1e-13;

    for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[i * p + j];
            for (int m = 0; m < j; m++) s -= L[i * p + m] * L[j * p + m];
            if (i == j) {
                if (!(s > piv_eps)) return false;
                L[i * p + i] = std::sqrt(s);
            } else {
                L[i * p + j] = s / L[j * p + j];
            }
        }
    }

    //Solve A z = e_col for each unit vector via forward + back substitution.
    for (int col = 0; col < p; col++) {
        double z[FW_MAX_P];
        for (int i = 0; i < p; i++) {                 //L w = e_col
            double s = (i == col) ? 1.0 : 0.0;
            for (int m = 0; m < i; m++) s -= L[i * p + m] * z[m];
            z[i] = s / L[i * p + i];
        }
        for (int i = p - 1; i >= 0; i--) {            //Lᵀ z = w
            double s = z[i];
            for (int m = i + 1; m < p; m++) s -= L[m * p + i] * z[m];
            z[i] = s / L[i * p + i];
        }
        for (int i = 0; i < p; i++) Ainv[i * p + col] = z[i];
    }
    return true;
}

//RunningMultipleRegression  (spec 5.6 struct; the hot kernel below uses an
//equivalent inlined state so the small-p instantiations stay in registers)

RunningMultipleRegression::RunningMultipleRegression(int k_incl_intercept)
    : k(k_incl_intercept),
      XtX(static_cast<size_t>(k) * k, 0.0),
      XtX_inv(static_cast<size_t>(k) * k, 0.0),
      Xty(static_cast<size_t>(k), 0.0) {}

void RunningMultipleRegression::reset() {
    std::fill(XtX.begin(), XtX.end(), 0.0);
    std::fill(Xty.begin(), Xty.end(), 0.0);
    yty = 0.0;
    sum_y = 0.0;
    n = 0;
    inv_valid = false;
}

/// Sherman-Morrison rank-1 update of A⁻¹ (runtime dimension):
///   (A + s·uuᵀ)⁻¹ = A⁻¹ − s·(A⁻¹u)(A⁻¹u)ᵀ / (1 + s·uᵀA⁻¹u)
static bool sm_update(double* Ainv, const double* u, double s, int p) {
    double v[FW_MAX_P];
    for (int i = 0; i < p; i++) {
        double acc = 0.0;
        for (int j = 0; j < p; j++) acc += Ainv[i * p + j] * u[j];
        v[i] = acc;
    }
    double utv = 0.0;
    for (int i = 0; i < p; i++) utv += u[i] * v[i];
    double denom = 1.0 + s * utv;
    if (!(denom > SM_EPS) && !(denom < -SM_EPS)) return false;
    double f = s / denom;
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++)
            Ainv[i * p + j] -= f * v[i] * v[j];
    return true;
}

void RunningMultipleRegression::add_row(const double* u, double y) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) XtX[i * k + j] += u[i] * u[j];
        Xty[i] += u[i] * y;
    }
    yty   += y * y;
    sum_y += y;
    n++;
    if (inv_valid && !sm_update(XtX_inv.data(), u, +1.0, k))
        inv_valid = false;
}

void RunningMultipleRegression::remove_row(const double* u, double y) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) XtX[i * k + j] -= u[i] * u[j];
        Xty[i] -= u[i] * y;
    }
    yty   -= y * y;
    sum_y -= y;
    n--;
    if (inv_valid && !sm_update(XtX_inv.data(), u, -1.0, k))
        inv_valid = false;
}

void RunningMultipleRegression::add(const double* x_new, double y_new,
                                    const double* x_old, double y_old) {
    remove_row(x_old, y_old);
    add_row(x_new, y_new);
}

bool RunningMultipleRegression::refresh_inverse() {
    inv_valid = (n >= static_cast<size_t>(k)) &&
                cholesky_invert(XtX.data(), XtX_inv.data(), k);
    return inv_valid;
}

void RunningMultipleRegression::coeffs(double* beta_out) const {
    for (int i = 0; i < k; i++) {
        double acc = 0.0;
        for (int j = 0; j < k; j++) acc += XtX_inv[i * k + j] * Xty[j];
        beta_out[i] = acc;
    }
}

double RunningMultipleRegression::r_squared() const {
    if (!inv_valid || n < 2) return DNAN;
    double beta[FW_MAX_P];
    coeffs(beta);
    double bXty = 0.0;
    for (int i = 0; i < k; i++) bXty += beta[i] * Xty[i];
    double sst = yty - sum_y * sum_y / static_cast<double>(n);
    if (!(sst > SM_EPS)) return DNAN;       //constant y
    double ssr = yty - bXty;
    double r2 = 1.0 - ssr / sst;
    if (r2 > 1.0) r2 = 1.0;
    if (r2 < 0.0) r2 = 0.0;
    return r2;
}

double RunningMultipleRegression::residual_std() const {
    if (!inv_valid || n <= static_cast<size_t>(k)) return DNAN;
    double beta[FW_MAX_P];
    coeffs(beta);
    double bXty = 0.0;
    for (int i = 0; i < k; i++) bXty += beta[i] * Xty[i];
    double ssr = yty - bXty;
    if (ssr < 0.0) ssr = 0.0;
    return std::sqrt(ssr / static_cast<double>(n - k));
}

//Hot kernel.  P_CT > 0 → compile-time dimension (loops fully unrolled);
//P_CT == 0 → runtime dimension p_rt (generic nested loops, k > 4).
//
//Rows where y or any regressor is non-finite enter the sums as all-zero
//rows (a zero row is an exact no-op for every rank-1 update), with a NaN
//counter gating output — same strategy as the simple-regression kernel.

template <int P_CT>
static void run_multi_kernel(
        const double* Y, const double* X,
        double* beta, double* r2, double* res_std,
        size_t n, int k, size_t window, int min_periods, int p_rt) {
    constexpr int PMAX = (P_CT > 0) ? P_CT : FW_MAX_P;
    const int p = (P_CT > 0) ? P_CT : p_rt;

    //X'X is NOT maintained per step: it is only needed for the (rare)
    //Cholesky refresh, so it is rebuilt from the window buffer on demand.
    double Ainv[PMAX * PMAX];
    double Xty[PMAX];
    for (int i = 0; i < p; i++) Xty[i] = 0.0;
    double yty = 0.0, sum_y = 0.0;
    bool inv_valid = false;

    std::vector<double>        rowbuf(window * static_cast<size_t>(p), 0.0);
    std::vector<double>        ybuf(window, 0.0);
    std::vector<unsigned char> validbuf(window, 1);
    size_t nan_count = 0;
    size_t reinit_ctr = 0;

    double u[PMAX], v[PMAX], beta_tmp[PMAX];

    //Rebuild X'X from the buffered rows and re-invert by Cholesky.
    auto refresh_inverse = [&](size_t m) -> bool {
        double XtX[PMAX * PMAX];
        for (int a = 0; a < p * p; a++) XtX[a] = 0.0;
        for (size_t s = 0; s < m; s++) {
            const double* ur = &rowbuf[s * p];
            for (int a = 0; a < p; a++)
                for (int b = a; b < p; b++)
                    XtX[a * p + b] += ur[a] * ur[b];
        }
        for (int a = 1; a < p; a++)              //mirror upper → lower
            for (int b = 0; b < a; b++)
                XtX[a * p + b] = XtX[b * p + a];
        return cholesky_invert(XtX, Ainv, p);
    };

    for (size_t i = 0; i < n; i++) {
        size_t slot = i % window;

        //evict the row falling out of the window
        if (i >= window) {
            const double* uo = &rowbuf[slot * p];
            double yo = ybuf[slot];
            for (int a = 0; a < p; a++) Xty[a] -= uo[a] * yo;
            yty   -= yo * yo;
            sum_y -= yo;
            if (!validbuf[slot]) nan_count--;

            if (inv_valid) {
                for (int a = 0; a < p; a++) {
                    double acc = 0.0;
                    for (int b = 0; b < p; b++) acc += Ainv[a * p + b] * uo[b];
                    v[a] = acc;
                }
                double utv = 0.0;
                for (int a = 0; a < p; a++) utv += uo[a] * v[a];
                double denom = 1.0 - utv;
                if (!(denom > SM_EPS) && !(denom < -SM_EPS)) {
                    inv_valid = false;
                } else {
                    double f = -1.0 / denom;
                    for (int a = 0; a < p; a++)
                        for (int b = 0; b < p; b++)
                            Ainv[a * p + b] -= f * v[a] * v[b];
                }
            }
        }

        //build and add the new design row u = [1, x_1..x_k]
        bool ok = fw_isfinite(Y[i]);
        for (int j = 0; ok && j < k; j++)
            ok = fw_isfinite(X[static_cast<size_t>(j) * n + i]);
        double y_new;
        if (ok) {
            u[0] = 1.0;
            for (int j = 0; j < k; j++)
                u[j + 1] = X[static_cast<size_t>(j) * n + i];
            y_new = Y[i];
        } else {
            for (int j = 0; j < p; j++) u[j] = 0.0;
            y_new = 0.0;
            nan_count++;
        }
        validbuf[slot] = ok ? 1 : 0;
        for (int j = 0; j < p; j++) rowbuf[slot * p + j] = u[j];
        ybuf[slot] = y_new;

        for (int a = 0; a < p; a++) Xty[a] += u[a] * y_new;
        yty   += y_new * y_new;
        sum_y += y_new;

        if (inv_valid) {
            for (int a = 0; a < p; a++) {
                double acc = 0.0;
                for (int b = 0; b < p; b++) acc += Ainv[a * p + b] * u[b];
                v[a] = acc;
            }
            double utv = 0.0;
            for (int a = 0; a < p; a++) utv += u[a] * v[a];
            double denom = 1.0 + utv;
            if (!(denom > SM_EPS)) {
                inv_valid = false;
            } else {
                double f = 1.0 / denom;
                for (int a = 0; a < p; a++)
                    for (int b = 0; b < p; b++)
                        Ainv[a * p + b] -= f * v[a] * v[b];
            }
        }

        //periodic exact recompute from the window buffers
        if (++reinit_ctr >= REINIT_INTERVAL) {
            reinit_ctr = 0;
            size_t m = std::min(i + 1, window);
            for (int a = 0; a < p; a++) Xty[a] = 0.0;
            yty = 0.0; sum_y = 0.0;
            for (size_t s = 0; s < m; s++) {
                const double* ur = &rowbuf[s * p];
                double yr = ybuf[s];
                for (int a = 0; a < p; a++) Xty[a] += ur[a] * yr;
                yty   += yr * yr;
                sum_y += yr;
            }
            inv_valid = false;   //force a Cholesky refresh below
        }

        //output
        size_t m = std::min(i + 1, window);
        if (nan_count > 0 || (int)m < min_periods || m < static_cast<size_t>(p))
            continue;

        //Cholesky fallback whenever the Sherman-Morrison chain was broken
        if (!inv_valid) {
            inv_valid = refresh_inverse(m);
            if (!inv_valid) continue;   //genuinely singular window
        }

        for (int a = 0; a < p; a++) {
            double acc = 0.0;
            for (int b = 0; b < p; b++) acc += Ainv[a * p + b] * Xty[b];
            beta_tmp[a] = acc;
        }
        for (int a = 0; a < p; a++)
            beta[static_cast<size_t>(a) * n + i] = beta_tmp[a];

        if (r2 || res_std) {
            double bXty = 0.0;
            for (int a = 0; a < p; a++) bXty += beta_tmp[a] * Xty[a];
            double ssr = yty - bXty;
            if (ssr < 0.0) ssr = 0.0;
            if (r2) {
                double sst = yty - sum_y * sum_y / static_cast<double>(m);
                if (sst > SM_EPS) {
                    double val = 1.0 - ssr / sst;
                    if (val > 1.0) val = 1.0;
                    if (val < 0.0) val = 0.0;
                    r2[i] = val;
                }
            }
            if (res_std && m > static_cast<size_t>(p))
                res_std[i] = std::sqrt(ssr / static_cast<double>(m - p));
        }
    }
}

//Public kernel: validates, NaN-fills, dispatches on p = k+1

void rolling_multiple_regression(
        const double* Y, const double* X,
        double* beta, double* r2, double* res_std,
        size_t n, int k, size_t window, int min_periods) {
    const int p = k + 1;   //intercept added internally
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) beta[static_cast<size_t>(j) * n + i] = DNAN;
        if (r2)      r2[i]      = DNAN;
        if (res_std) res_std[i] = DNAN;
    }
    if (n == 0 || window == 0 || k < 1 || k > FW_MAX_REGRESSORS) return;

    switch (p) {
        case 2:  run_multi_kernel<2>(Y, X, beta, r2, res_std, n, k, window, min_periods, p); break;
        case 3:  run_multi_kernel<3>(Y, X, beta, r2, res_std, n, k, window, min_periods, p); break;
        case 4:  run_multi_kernel<4>(Y, X, beta, r2, res_std, n, k, window, min_periods, p); break;
        case 5:  run_multi_kernel<5>(Y, X, beta, r2, res_std, n, k, window, min_periods, p); break;
        default: run_multi_kernel<0>(Y, X, beta, r2, res_std, n, k, window, min_periods, p); break;
    }
}

} //namespace fastwindow
