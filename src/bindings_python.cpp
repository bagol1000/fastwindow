/// @file bindings_python.cpp
/// pybind11 bindings for the rolling statistics kernels.
/// Guarded by FASTWINDOW_PYTHON (defined in setup.py) so that R CMD INSTALL,
/// which compiles every .cpp under src/, skips this translation unit.
#ifdef FASTWINDOW_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "fastwindow.h"

namespace py = pybind11;

//Helper: guarantee a 1-D C-contiguous double array; copy if needed.
//array_t<double, c_style|forcecast>::ensure() handles dtype casting,
//non-contiguous strides, and Fortran-order arrays transparently.
using ContiguousDouble =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

static ContiguousDouble ensure_contiguous(const py::array& arr) {
    if (arr.ndim() != 1)
        throw std::invalid_argument("Input array must be 1-D");
    auto result = ContiguousDouble::ensure(arr);
    if (!result)
        throw std::invalid_argument("Could not convert input to a 1-D float64 array");
    return result;
}


/// Resolve the optional out= argument: validate and reuse the caller's
/// buffer, or allocate a fresh array.  Reuse avoids the page-fault cost of
/// a new 8n-byte allocation per call in tight loops.  The kernels re-read
/// evicted source values, so out must not alias an input buffer.
static py::array_t<double> resolve_out(const py::object& out, py::ssize_t n,
                                       const void* src1,
                                       const void* src2 = nullptr) {
    if (out.is_none())
        return py::array_t<double>(n);
    auto arr = py::cast<py::array>(out);
    bool ok = arr.ndim() == 1 && arr.shape(0) == n &&
              arr.dtype().num() == py::dtype::of<double>().num() &&
              (arr.flags() & py::array::c_style) && arr.writeable();
    if (!ok)
        throw std::invalid_argument(
            "out must be a writable C-contiguous float64 array with the "
            "same length as the input");
    if (arr.data() == src1 || (src2 && arr.data() == src2))
        throw std::invalid_argument(
            "out must not be the same array as the input");
    return arr.cast<py::array_t<double>>();
}

static py::array_t<double> make_output(py::ssize_t n) {
    return py::array_t<double>(n);
}

//Validation helpers
static void check_window(size_t window) {
    if (window == 0)
        throw std::invalid_argument("window must be > 0");
}

static int resolve_min_periods(int min_periods, size_t window) {
    if (min_periods == -1)
        return static_cast<int>(window);
    if (min_periods < 0)
        throw std::invalid_argument("min_periods must be >= 0");
    return min_periods;
}

static int resolve_min_periods_default(int min_periods, int default_value) {
    if (min_periods == -1)
        return default_value;
    if (min_periods < 0)
        throw std::invalid_argument("min_periods must be >= 0");
    return min_periods;
}

static void check_n_threads(int n_threads) {
    if (n_threads < 0)
        throw std::invalid_argument("n_threads must be >= 0");
}

//Wrappers

static py::array_t<double> py_rolling_mean(
        const py::array& x, size_t window,
        int min_periods, bool skip_nan, int n_threads,
        const py::object& out) {
    check_window(window);
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    check_n_threads(n_threads);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_mean(
            src.data(), dst.mutable_data(), n, window, mp, skip_nan,
            n_threads);
    }
    return dst;
}

static py::array_t<double> py_rolling_var(
        const py::array& x, size_t window,
        int min_periods, int ddof, bool skip_nan, int n_threads,
        const py::object& out) {
    check_window(window);
    if (ddof < 0 || ddof > 1)
        throw std::invalid_argument("ddof must be 0 or 1");
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    check_n_threads(n_threads);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_var(
            src.data(), dst.mutable_data(), n,
            window, mp, ddof == 1, skip_nan, n_threads);
    }
    return dst;
}

static py::array_t<double> py_rolling_std(
        const py::array& x, size_t window,
        int min_periods, int ddof, bool skip_nan, int n_threads,
        const py::object& out) {
    check_window(window);
    if (ddof < 0 || ddof > 1)
        throw std::invalid_argument("ddof must be 0 or 1");
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    check_n_threads(n_threads);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_std(
            src.data(), dst.mutable_data(), n,
            window, mp, ddof == 1, skip_nan, n_threads);
    }
    return dst;
}

static py::array_t<double> py_rolling_sum(
        const py::array& x, size_t window,
        int min_periods, bool skip_nan, int n_threads,
        const py::object& out) {
    check_window(window);
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    check_n_threads(n_threads);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_sum(
            src.data(), dst.mutable_data(), n, window, mp, skip_nan,
            n_threads);
    }
    return dst;
}

static py::array_t<double> py_rolling_min(
        const py::array& x, size_t window, int min_periods, bool skip_nan,
        int n_threads, const py::object& out) {
    check_window(window);
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    check_n_threads(n_threads);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_min(src.data(), dst.mutable_data(), n, window,
                                mp, skip_nan, n_threads);
    }
    return dst;
}

static py::array_t<double> py_rolling_max(
        const py::array& x, size_t window, int min_periods, bool skip_nan,
        int n_threads, const py::object& out) {
    check_window(window);
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    check_n_threads(n_threads);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_max(src.data(), dst.mutable_data(), n, window,
                                mp, skip_nan, n_threads);
    }
    return dst;
}

//Regression wrappers — return dicts with intercept / slope / r2

static py::dict py_rolling_regression(
        const py::array& y, size_t window,
        int min_periods, bool time_as_x) {
    check_window(window);
    if (!time_as_x)
        throw std::invalid_argument(
            "time_as_x=False requires an explicit x series; "
            "use rolling_regression_xy(y, x, window) instead");
    auto ysrc = ensure_contiguous(y);
    size_t n  = static_cast<size_t>(ysrc.shape(0));
    auto b0 = make_output(static_cast<py::ssize_t>(n));
    auto b1 = make_output(static_cast<py::ssize_t>(n));
    auto r2 = make_output(static_cast<py::ssize_t>(n));
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_simple_regression(
            ysrc.data(), b0.mutable_data(), b1.mutable_data(),
            r2.mutable_data(), n, window, mp, true);
    }
    py::dict out;
    out["intercept"] = b0;
    out["slope"]     = b1;
    out["r2"]        = r2;
    return out;
}

static py::dict py_rolling_regression_xy(
        const py::array& y, const py::array& x,
        size_t window, int min_periods) {
    check_window(window);
    auto ysrc = ensure_contiguous(y);
    auto xsrc = ensure_contiguous(x);
    if (ysrc.shape(0) != xsrc.shape(0))
        throw std::invalid_argument("x and y must have the same length");
    size_t n = static_cast<size_t>(ysrc.shape(0));
    auto b0 = make_output(static_cast<py::ssize_t>(n));
    auto b1 = make_output(static_cast<py::ssize_t>(n));
    auto r2 = make_output(static_cast<py::ssize_t>(n));
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_simple_regression_xy(
            xsrc.data(), ysrc.data(),
            b0.mutable_data(), b1.mutable_data(), r2.mutable_data(),
            n, window, mp);
    }
    py::dict out;
    out["intercept"] = b0;
    out["slope"]     = b1;
    out["r2"]        = r2;
    return out;
}

//Multiple regression — X is converted to Fortran (column-major) order to
//match the kernel's layout; beta is returned as an F-ordered (n, k+1) array.
using FortranDouble =
    py::array_t<double, py::array::f_style | py::array::forcecast>;

static py::dict py_rolling_multiple_regression(
        const py::array& y, const py::array& X,
        size_t window, int min_periods) {
    check_window(window);
    if (X.ndim() != 2)
        throw std::invalid_argument("X must be a 2-D array of shape (n, k)");
    auto ysrc = ensure_contiguous(y);
    auto Xsrc = FortranDouble::ensure(X);
    if (!Xsrc)
        throw std::invalid_argument("Could not convert X to a float64 array");
    size_t n = static_cast<size_t>(ysrc.shape(0));
    if (static_cast<size_t>(Xsrc.shape(0)) != n)
        throw std::invalid_argument("y and X must have the same number of rows");
    int k = static_cast<int>(Xsrc.shape(1));
    if (k < 1 || k > fastwindow::FW_MAX_REGRESSORS)
        throw std::invalid_argument(
            "k must be between 1 and 16 regressors (X has shape (n, k))");

    auto beta    = FortranDouble({static_cast<py::ssize_t>(n),
                                  static_cast<py::ssize_t>(k + 1)});
    auto r2      = make_output(static_cast<py::ssize_t>(n));
    auto res_std = make_output(static_cast<py::ssize_t>(n));
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_multiple_regression(
            ysrc.data(), Xsrc.data(),
            beta.mutable_data(), r2.mutable_data(), res_std.mutable_data(),
            n, k, window, mp);
    }
    py::dict out;
    out["coef"]         = beta;
    out["r2"]           = r2;
    out["residual_std"] = res_std;
    return out;
}

//Correlation / covariance wrappers

static py::object py_rolling_corr(
        const py::array& x, const py::array& y, size_t window,
        int min_periods, bool return_cov, bool skip_nan,
        const py::object& out) {
    check_window(window);
    if (return_cov && !out.is_none())
        throw std::invalid_argument(
            "out= is not supported together with return_cov=True");
    auto xsrc = ensure_contiguous(x);
    auto ysrc = ensure_contiguous(y);
    if (xsrc.shape(0) != ysrc.shape(0))
        throw std::invalid_argument("x and y must have the same length");
    size_t n = static_cast<size_t>(xsrc.shape(0));
    auto corr = resolve_out(out, static_cast<py::ssize_t>(n), xsrc.data(), ysrc.data());
    int mp = resolve_min_periods(min_periods, window);
    if (return_cov) {
        auto cov = make_output(static_cast<py::ssize_t>(n));
        {
            py::gil_scoped_release release;
            fastwindow::rolling_corr(
                xsrc.data(), ysrc.data(),
                corr.mutable_data(), cov.mutable_data(),
                n, window, mp, skip_nan);
        }
        return py::make_tuple(corr, cov);
    }
    {
        py::gil_scoped_release release;
        fastwindow::rolling_corr(
            xsrc.data(), ysrc.data(), corr.mutable_data(), nullptr,
            n, window, mp, skip_nan);
    }
    return corr;
}

static py::array_t<double> py_rolling_cov(
        const py::array& x, const py::array& y, size_t window,
        int min_periods, int ddof, bool skip_nan,
        const py::object& out) {
    check_window(window);
    if (ddof < 0 || ddof > 1)
        throw std::invalid_argument("ddof must be 0 or 1");
    auto xsrc = ensure_contiguous(x);
    auto ysrc = ensure_contiguous(y);
    if (xsrc.shape(0) != ysrc.shape(0))
        throw std::invalid_argument("x and y must have the same length");
    size_t n = static_cast<size_t>(xsrc.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), xsrc.data(), ysrc.data());
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_cov(
            xsrc.data(), ysrc.data(), dst.mutable_data(),
            n, window, mp, ddof == 1, skip_nan);
    }
    return dst;
}

//Correlation matrix — returns (n, p, p) C-ordered symmetric array

static py::array_t<double> py_rolling_corr_matrix(
        const py::array& X, size_t window, int min_periods, int n_threads) {
    check_window(window);
    check_n_threads(n_threads);
    if (X.ndim() != 2)
        throw std::invalid_argument("X must be a 2-D array of shape (n, p)");
    auto Xsrc = FortranDouble::ensure(X);
    if (!Xsrc)
        throw std::invalid_argument("Could not convert X to a float64 array");
    size_t n = static_cast<size_t>(Xsrc.shape(0));
    int p    = static_cast<int>(Xsrc.shape(1));
    if (p < 2 || p > fastwindow::FW_MAX_CORR_COLS)
        throw std::invalid_argument("X must have between 2 and 50 columns");

    auto out = py::array_t<double>({static_cast<py::ssize_t>(n),
                                    static_cast<py::ssize_t>(p),
                                    static_cast<py::ssize_t>(p)});
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;   //long-running, OpenMP inside
        size_t n_pairs = static_cast<size_t>(p) * (p - 1) / 2;
        std::vector<double> tri(n_pairs * n);
        fastwindow::rolling_corr_matrix(
            Xsrc.data(), tri.data(), n, p, window, mp, n_threads);
        fastwindow::corr_matrix_expand(
            tri.data(), out.mutable_data(), n, p,
            /*r_layout=*/false, n_threads);
    }
    return out;
}

/// Internal: pair-major triangle output (spec 6.8 kernel layout) without the
/// (n, p, p) expansion.  Column k holds corr(X[:,i], X[:,j]) for the k-th
/// upper-triangle pair (0,1),(0,2),…  Used for benchmarking and power users.
static py::array_t<double> py_corr_matrix_pairs(
        const py::array& X, size_t window, int min_periods, int n_threads) {
    check_window(window);
    check_n_threads(n_threads);
    if (X.ndim() != 2)
        throw std::invalid_argument("X must be a 2-D array of shape (n, p)");
    auto Xsrc = FortranDouble::ensure(X);
    if (!Xsrc)
        throw std::invalid_argument("Could not convert X to a float64 array");
    size_t n = static_cast<size_t>(Xsrc.shape(0));
    int p    = static_cast<int>(Xsrc.shape(1));
    if (p < 2 || p > fastwindow::FW_MAX_CORR_COLS)
        throw std::invalid_argument("X must have between 2 and 50 columns");
    py::ssize_t n_pairs = static_cast<py::ssize_t>(p) * (p - 1) / 2;
    auto out = FortranDouble({static_cast<py::ssize_t>(n), n_pairs});
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_corr_matrix(
            Xsrc.data(), out.mutable_data(), n, p, window, mp, n_threads);
    }
    return out;
}

//Higher moments and z-score

static py::array_t<double> py_rolling_skew(
        const py::array& x, size_t window,
        int min_periods, bool skip_nan, const py::object& out) {
    check_window(window);
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_skew(src.data(), dst.mutable_data(), n, window,
                                 mp, skip_nan);
    }
    return dst;
}

static py::array_t<double> py_rolling_kurt(
        const py::array& x, size_t window,
        int min_periods, bool skip_nan, const py::object& out) {
    check_window(window);
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_kurt(src.data(), dst.mutable_data(), n, window,
                                 mp, skip_nan);
    }
    return dst;
}

static py::array_t<double> py_rolling_zscore(
        const py::array& x, size_t window,
        int min_periods, int ddof, bool skip_nan, int n_threads,
        const py::object& out) {
    check_window(window);
    if (ddof < 0 || ddof > 1)
        throw std::invalid_argument("ddof must be 0 or 1");
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    check_n_threads(n_threads);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_zscore(src.data(), dst.mutable_data(), n, window,
                                   mp, ddof == 1, skip_nan, n_threads);
    }
    return dst;
}

//Quantile, expanding windows, 2-D dispatch, Spearman

static py::array_t<double> py_rolling_quantile(
        const py::array& x, size_t window, double q,
        int min_periods, bool exact, const py::object& out) {
    check_window(window);
    if (!(q > 0.0 && q < 1.0))
        throw std::invalid_argument("q must be strictly between 0 and 1");
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), src.data());
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_quantile(
            src.data(), dst.mutable_data(), n, window, q, mp, exact);
    }
    return dst;
}

static py::array_t<double> py_expanding_mean(const py::array& x,
                                             int min_periods) {
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = make_output(static_cast<py::ssize_t>(n));
    int mp = resolve_min_periods_default(min_periods, 1);
    {
        py::gil_scoped_release release;
        fastwindow::expanding_mean(src.data(), dst.mutable_data(), n, mp);
    }
    return dst;
}

static py::array_t<double> py_expanding_var(const py::array& x,
                                            int min_periods, int ddof) {
    if (ddof < 0 || ddof > 1)
        throw std::invalid_argument("ddof must be 0 or 1");
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = make_output(static_cast<py::ssize_t>(n));
    int mp = resolve_min_periods_default(min_periods, 2);
    {
        py::gil_scoped_release release;
        fastwindow::expanding_var(src.data(), dst.mutable_data(), n,
                                  mp, ddof == 1);
    }
    return dst;
}

static py::array_t<double> py_expanding_std(const py::array& x,
                                            int min_periods, int ddof) {
    if (ddof < 0 || ddof > 1)
        throw std::invalid_argument("ddof must be 0 or 1");
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = make_output(static_cast<py::ssize_t>(n));
    int mp = resolve_min_periods_default(min_periods, 2);
    {
        py::gil_scoped_release release;
        fastwindow::expanding_std(src.data(), dst.mutable_data(), n,
                                  mp, ddof == 1);
    }
    return dst;
}

static py::array_t<double> py_expanding_sum(const py::array& x,
                                            int min_periods) {
    auto src = ensure_contiguous(x);
    size_t n = static_cast<size_t>(src.shape(0));
    auto dst = make_output(static_cast<py::ssize_t>(n));
    int mp = resolve_min_periods_default(min_periods, 1);
    {
        py::gil_scoped_release release;
        fastwindow::expanding_sum(src.data(), dst.mutable_data(), n, mp);
    }
    return dst;
}

static py::dict py_expanding_regression(const py::array& y,
                                        int min_periods, bool time_as_x) {
    if (!time_as_x)
        throw std::invalid_argument(
            "expanding_regression supports time_as_x=True only");
    auto ysrc = ensure_contiguous(y);
    size_t n  = static_cast<size_t>(ysrc.shape(0));
    auto b0 = make_output(static_cast<py::ssize_t>(n));
    auto b1 = make_output(static_cast<py::ssize_t>(n));
    auto r2 = make_output(static_cast<py::ssize_t>(n));
    int mp = resolve_min_periods_default(min_periods, 2);
    {
        py::gil_scoped_release release;
        fastwindow::expanding_regression(
            ysrc.data(), b0.mutable_data(), b1.mutable_data(),
            r2.mutable_data(), n, mp);
    }
    py::dict out;
    out["intercept"] = b0;
    out["slope"]     = b1;
    out["r2"]        = r2;
    return out;
}

//2-D dispatch — shared scaffolding: ensure Fortran X, allocate F-ordered out
template <typename Kernel>
static py::array_t<double> run_2d(const py::array& X, Kernel&& kernel) {
    if (X.ndim() != 2)
        throw std::invalid_argument("X must be a 2-D array of shape (n, p)");
    auto Xsrc = FortranDouble::ensure(X);
    if (!Xsrc)
        throw std::invalid_argument("Could not convert X to a float64 array");
    size_t n = static_cast<size_t>(Xsrc.shape(0));
    int p    = static_cast<int>(Xsrc.shape(1));
    auto out = FortranDouble({static_cast<py::ssize_t>(n),
                              static_cast<py::ssize_t>(p)});
    {
        py::gil_scoped_release release;
        kernel(Xsrc.data(), out.mutable_data(), n, p);
    }
    return out;
}

static py::array_t<double> py_rolling_mean_2d(
        const py::array& X, size_t window, int min_periods, int n_threads) {
    check_window(window);
    check_n_threads(n_threads);
    int mp = resolve_min_periods(min_periods, window);
    return run_2d(X, [=](const double* src, double* dst, size_t n, int p) {
        fastwindow::rolling_mean_matrix(src, dst, n, p, window, mp, n_threads);
    });
}

static py::array_t<double> py_rolling_std_2d(
        const py::array& X, size_t window, int min_periods,
        int ddof, int n_threads) {
    check_window(window);
    check_n_threads(n_threads);
    if (ddof < 0 || ddof > 1)
        throw std::invalid_argument("ddof must be 0 or 1");
    int mp = resolve_min_periods(min_periods, window);
    return run_2d(X, [=](const double* src, double* dst, size_t n, int p) {
        fastwindow::rolling_std_matrix(src, dst, n, p, window, mp,
                                       ddof == 1, n_threads);
    });
}

static py::array_t<double> py_rolling_sum_2d(
        const py::array& X, size_t window, int min_periods, int n_threads) {
    check_window(window);
    check_n_threads(n_threads);
    int mp = resolve_min_periods(min_periods, window);
    return run_2d(X, [=](const double* src, double* dst, size_t n, int p) {
        fastwindow::rolling_sum_matrix(src, dst, n, p, window, mp, n_threads);
    });
}

static py::array_t<double> py_rolling_min_2d(
        const py::array& X, size_t window, int min_periods, bool skip_nan,
        int n_threads) {
    check_window(window);
    check_n_threads(n_threads);
    int mp = resolve_min_periods(min_periods, window);
    return run_2d(X, [=](const double* src, double* dst, size_t n, int p) {
        fastwindow::rolling_min_matrix(src, dst, n, p, window, mp, skip_nan,
                                       n_threads);
    });
}

static py::array_t<double> py_rolling_max_2d(
        const py::array& X, size_t window, int min_periods, bool skip_nan,
        int n_threads) {
    check_window(window);
    check_n_threads(n_threads);
    int mp = resolve_min_periods(min_periods, window);
    return run_2d(X, [=](const double* src, double* dst, size_t n, int p) {
        fastwindow::rolling_max_matrix(src, dst, n, p, window, mp, skip_nan,
                                       n_threads);
    });
}

static py::array_t<double> py_rolling_spearman(
        const py::array& x, const py::array& y, size_t window,
        int min_periods, const py::object& out) {
    check_window(window);
    auto xsrc = ensure_contiguous(x);
    auto ysrc = ensure_contiguous(y);
    if (xsrc.shape(0) != ysrc.shape(0))
        throw std::invalid_argument("x and y must have the same length");
    size_t n = static_cast<size_t>(xsrc.shape(0));
    auto dst = resolve_out(out, static_cast<py::ssize_t>(n), xsrc.data(), ysrc.data());
    int mp = resolve_min_periods(min_periods, window);
    {
        py::gil_scoped_release release;
        fastwindow::rolling_spearman(
            xsrc.data(), ysrc.data(), dst.mutable_data(), n, window, mp);
    }
    return dst;
}

//Module info helpers

static bool py_has_avx2() {
    return FW_SIMD && fastwindow::cpu_has_avx2();
}

static void py_set_num_threads(int n) {
    if (n < 1)
        throw std::invalid_argument("n must be a positive integer");
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

static int py_get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

//Module definition

PYBIND11_MODULE(_core, m) {
    m.doc() = "fastwindow C++ core — rolling window statistics";

    m.def("rolling_mean", &py_rolling_mean,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("skip_nan")    = false,
          py::arg("n_threads")   = 0,
          py::arg("out")         = py::none(),
          "Rolling mean over a 1-D double array.");

    m.def("rolling_var", &py_rolling_var,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("ddof")        = 1,
          py::arg("skip_nan")    = false,
          py::arg("n_threads")   = 0,
          py::arg("out")         = py::none(),
          "Rolling variance (ddof=1 for sample, 0 for population).");

    m.def("rolling_std", &py_rolling_std,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("ddof")        = 1,
          py::arg("skip_nan")    = false,
          py::arg("n_threads")   = 0,
          py::arg("out")         = py::none(),
          "Rolling standard deviation.");

    m.def("rolling_sum", &py_rolling_sum,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("skip_nan")    = false,
          py::arg("n_threads")   = 0,
          py::arg("out")         = py::none(),
          "Rolling sum.");

    m.def("rolling_min", &py_rolling_min,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("skip_nan")    = false,
          py::arg("n_threads")   = 0,
          py::arg("out")         = py::none(),
          "Rolling minimum.  skip_nan=True ignores NaN values "
          "(pandas semantics); default propagates NaN to the output.");

    m.def("rolling_max", &py_rolling_max,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("skip_nan")    = false,
          py::arg("n_threads")   = 0,
          py::arg("out")         = py::none(),
          "Rolling maximum.  skip_nan=True ignores NaN values "
          "(pandas semantics); default propagates NaN to the output.");

    m.def("rolling_regression", &py_rolling_regression,
          py::arg("y"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("time_as_x")   = true,
          "Rolling OLS of y on time (x = 0..window-1). "
          "Returns dict with 'intercept', 'slope', 'r2'.");

    m.def("rolling_regression_xy", &py_rolling_regression_xy,
          py::arg("y"), py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          "Rolling OLS of y on an explicit regressor x. "
          "Returns dict with 'intercept', 'slope', 'r2'.");

    m.def("rolling_multiple_regression", &py_rolling_multiple_regression,
          py::arg("y"), py::arg("X"), py::arg("window"),
          py::arg("min_periods") = -1,
          "Rolling OLS of y on k regressors (intercept added internally). "
          "Returns dict with 'coef' (n×(k+1), intercept first), 'r2', "
          "'residual_std'.");

    m.def("rolling_corr", &py_rolling_corr,
          py::arg("x"), py::arg("y"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("return_cov")  = false,
          py::arg("skip_nan")    = false,
          py::arg("out")         = py::none(),
          "Rolling Pearson correlation between x and y. "
          "return_cov=True returns a (corr, cov) tuple.");

    m.def("rolling_cov", &py_rolling_cov,
          py::arg("x"), py::arg("y"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("ddof")        = 1,
          py::arg("skip_nan")    = false,
          py::arg("out")         = py::none(),
          "Rolling covariance between x and y (ddof=1 sample, 0 population).");

    m.def("rolling_corr_matrix", &py_rolling_corr_matrix,
          py::arg("X"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("n_threads")   = 0,
          "Rolling correlation matrix across the columns of X. "
          "Returns an (n, p, p) symmetric array with unit diagonal.");

    m.def("_corr_matrix_pairs", &py_corr_matrix_pairs,
          py::arg("X"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("n_threads")   = 0,
          "Internal: upper-triangle pair series without matrix expansion.");

    m.def("rolling_skew", &py_rolling_skew,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("skip_nan")    = false,
          py::arg("out")         = py::none(),
          "Rolling skewness, bias-corrected (matches pandas "
          ".rolling().skew()).  Requires >= 3 valid observations.");

    m.def("rolling_kurt", &py_rolling_kurt,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("skip_nan")    = false,
          py::arg("out")         = py::none(),
          "Rolling excess kurtosis, bias-corrected (matches pandas "
          ".rolling().kurt()).  Requires >= 4 valid observations.");

    m.def("rolling_zscore", &py_rolling_zscore,
          py::arg("x"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("ddof")        = 1,
          py::arg("skip_nan")    = false,
          py::arg("n_threads")   = 0,
          py::arg("out")         = py::none(),
          "Rolling z-score: (x - rolling mean) / rolling std.  NaN where "
          "the input is NaN or the window stddev is zero.");

    m.def("rolling_quantile", &py_rolling_quantile,
          py::arg("x"), py::arg("window"),
          py::arg("q")           = 0.5,
          py::arg("min_periods") = -1,
          py::arg("exact")       = true,
          py::arg("out")         = py::none(),
          "Rolling quantile. exact=True (default) maintains an exact "
          "rolling window with lazy-deletion heaps; exact=False uses the "
          "faster P2 streaming approximation, which estimates the quantile "
          "of all observations seen so far, not of the window.");

    m.def("expanding_mean", &py_expanding_mean,
          py::arg("x"), py::arg("min_periods") = 1,
          "Expanding (growing-window) mean.");
    m.def("expanding_var", &py_expanding_var,
          py::arg("x"), py::arg("min_periods") = 2, py::arg("ddof") = 1,
          "Expanding variance.");
    m.def("expanding_std", &py_expanding_std,
          py::arg("x"), py::arg("min_periods") = 2, py::arg("ddof") = 1,
          "Expanding standard deviation.");
    m.def("expanding_sum", &py_expanding_sum,
          py::arg("x"), py::arg("min_periods") = 1,
          "Expanding sum (Kahan-compensated).");
    m.def("expanding_regression", &py_expanding_regression,
          py::arg("y"), py::arg("min_periods") = -1,
          py::arg("time_as_x") = true,
          "Expanding OLS of y on the time index. Returns dict with "
          "'intercept', 'slope', 'r2'.");

    m.def("rolling_mean_2d", &py_rolling_mean_2d,
          py::arg("X"), py::arg("window"),
          py::arg("min_periods") = -1, py::arg("n_threads") = 0,
          "Column-wise rolling mean over an (n, p) array (OpenMP).");
    m.def("rolling_std_2d", &py_rolling_std_2d,
          py::arg("X"), py::arg("window"),
          py::arg("min_periods") = -1, py::arg("ddof") = 1,
          py::arg("n_threads") = 0,
          "Column-wise rolling standard deviation (OpenMP).");
    m.def("rolling_sum_2d", &py_rolling_sum_2d,
          py::arg("X"), py::arg("window"),
          py::arg("min_periods") = -1, py::arg("n_threads") = 0,
          "Column-wise rolling sum (OpenMP).");
    m.def("rolling_min_2d", &py_rolling_min_2d,
          py::arg("X"), py::arg("window"),
          py::arg("min_periods") = -1, py::arg("skip_nan") = false,
          py::arg("n_threads") = 0,
          "Column-wise rolling minimum (OpenMP).");
    m.def("rolling_max_2d", &py_rolling_max_2d,
          py::arg("X"), py::arg("window"),
          py::arg("min_periods") = -1, py::arg("skip_nan") = false,
          py::arg("n_threads") = 0,
          "Column-wise rolling maximum (OpenMP).");

    m.def("rolling_spearman", &py_rolling_spearman,
          py::arg("x"), py::arg("y"), py::arg("window"),
          py::arg("min_periods") = -1,
          py::arg("out")         = py::none(),
          "Rolling Spearman rank correlation (average ranks for ties). "
          "O(window log window) per step — use small/moderate windows.");

    m.def("set_num_threads", &py_set_num_threads, py::arg("n"),
          "Set the default OpenMP thread count used when n_threads=0.");
    m.def("get_num_threads", &py_get_num_threads,
          "Return the current default OpenMP thread count.");

    m.def("has_avx2", &py_has_avx2,
          "Return True if the AVX2 kernel paths are active on this CPU "
          "(selected at runtime; False on non-x86-64 builds or when the "
          "CPU/OS lacks AVX2+FMA support).");
}
#endif  //FASTWINDOW_PYTHON
