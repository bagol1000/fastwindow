"""
Rolling window statistics backed by a C++17 core (AVX2 + OpenMP).

Inputs are converted to contiguous float64 on entry.  NaN values inside a
window propagate to the output by default; functions with a ``skip_nan``
parameter can exclude them instead.  See ``help()`` on each function or the
type stubs in ``__init__.pyi`` for details.
"""

__version__ = "0.1.0"

from fastwindow._core import (
    rolling_mean,
    rolling_var,
    rolling_std,
    rolling_sum,
    rolling_min,
    rolling_max,
    rolling_regression,
    rolling_regression_xy,
    rolling_multiple_regression,
    rolling_corr,
    rolling_cov,
    rolling_corr_matrix,
    rolling_quantile,
    rolling_spearman,
    expanding_mean,
    expanding_var,
    expanding_std,
    expanding_sum,
    expanding_regression,
    rolling_mean_2d,
    rolling_std_2d,
    rolling_sum_2d,
    rolling_min_2d,
    rolling_max_2d,
    set_num_threads,
    get_num_threads,
    has_avx2,
)

__all__ = [
    "rolling_mean",
    "rolling_var",
    "rolling_std",
    "rolling_sum",
    "rolling_min",
    "rolling_max",
    "rolling_regression",
    "rolling_regression_xy",
    "rolling_multiple_regression",
    "rolling_corr",
    "rolling_cov",
    "rolling_corr_matrix",
    "rolling_quantile",
    "rolling_spearman",
    "expanding_mean",
    "expanding_var",
    "expanding_std",
    "expanding_sum",
    "expanding_regression",
    "rolling_mean_2d",
    "rolling_std_2d",
    "rolling_sum_2d",
    "rolling_min_2d",
    "rolling_max_2d",
    "set_num_threads",
    "get_num_threads",
    "has_avx2",
]
