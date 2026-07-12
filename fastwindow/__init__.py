"""
Rolling window statistics backed by a C++17 core (AVX2 + OpenMP).

Inputs are converted to contiguous float64 on entry.  NaN values inside a
window propagate to the output by default; functions with a ``skip_nan``
parameter can exclude them instead.  See ``help()`` on each function or the
type stubs in ``__init__.pyi`` for details.

pandas objects are accepted everywhere (pandas itself is NOT required):
Series in → Series out with the index preserved, DataFrame in →
DataFrame out for the 2-D functions.  Plain arrays stay plain arrays.
"""

__version__ = "0.2.0"

import sys as _sys

import numpy as _np

from fastwindow import _core
from fastwindow._core import (
    set_num_threads,
    get_num_threads,
    has_avx2,
)


def _unwrap(obj):
    """Split a possibly-pandas input into (float64 values, container)."""
    pd = _sys.modules.get("pandas")
    if pd is not None and isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_numpy(dtype="float64", na_value=_np.nan), obj
    if not isinstance(obj, _np.ndarray):
        return _np.asarray(obj, dtype="float64"), None
    return obj, None


def _box_series(container, values):
    import pandas as pd
    name = container.name if isinstance(container, pd.Series) else None
    #copy=False: keep out= buffer reuse zero-copy under pandas CoW
    return pd.Series(values, index=container.index, name=name, copy=False)


def _copy_meta(fn, core_fn):
    fn.__name__ = core_fn.__name__
    fn.__qualname__ = core_fn.__name__
    fn.__doc__ = core_fn.__doc__
    return fn


def _wrap_1d(core_fn):
    """1-D array in → same-shape array out; Series in → Series out."""
    def fn(x, *args, **kwargs):
        arr, box = _unwrap(x)
        res = core_fn(arr, *args, **kwargs)
        return _box_series(box, res) if box is not None else res
    return _copy_meta(fn, core_fn)


def _wrap_xy(core_fn):
    """Two 1-D inputs; the box follows the first pandas argument.
    Handles tuple returns (rolling_corr with return_cov=True)."""
    def fn(x, y, *args, **kwargs):
        xarr, xbox = _unwrap(x)
        yarr, ybox = _unwrap(y)
        box = xbox if xbox is not None else ybox
        res = core_fn(xarr, yarr, *args, **kwargs)
        if box is None:
            return res
        if isinstance(res, tuple):
            return tuple(_box_series(box, r) for r in res)
        if isinstance(res, dict):
            return {k: _box_series(box, v) for k, v in res.items()}
        return _box_series(box, res)
    return _copy_meta(fn, core_fn)


def _wrap_dict(core_fn):
    """1-D input, dict-of-arrays output (regression results)."""
    def fn(y, *args, **kwargs):
        arr, box = _unwrap(y)
        res = core_fn(arr, *args, **kwargs)
        if box is not None:
            res = {k: _box_series(box, v) for k, v in res.items()}
        return res
    return _copy_meta(fn, core_fn)


def _wrap_2d(core_fn):
    """2-D input; DataFrame in → DataFrame out (index and columns kept)."""
    def fn(X, *args, **kwargs):
        arr, box = _unwrap(X)
        res = core_fn(arr, *args, **kwargs)
        if box is not None:
            import pandas as pd
            return pd.DataFrame(res, index=box.index, columns=box.columns,
                                copy=False)
        return res
    return _copy_meta(fn, core_fn)


def _wrap_2d_plain(core_fn):
    """2-D input accepted as DataFrame, output stays an ndarray
    (rolling_corr_matrix returns an (n, p, p) cube)."""
    def fn(X, *args, **kwargs):
        arr, _ = _unwrap(X)
        return core_fn(arr, *args, **kwargs)
    return _copy_meta(fn, core_fn)


def _wrap_multiple_regression(core_fn):
    """y (Series ok) + X (DataFrame ok) → dict; 'coef' becomes a
    DataFrame with columns ['intercept', *X.columns] when X is one."""
    def fn(y, X, *args, **kwargs):
        yarr, ybox = _unwrap(y)
        Xarr, Xbox = _unwrap(X)
        res = core_fn(yarr, Xarr, *args, **kwargs)
        box = ybox if ybox is not None else Xbox
        if box is None:
            return res
        import pandas as pd
        index = box.index
        coef_cols = (["intercept"] + [str(c) for c in Xbox.columns]
                     if Xbox is not None
                     else ["intercept"] + [f"x{i}" for i
                                           in range(res["coef"].shape[1] - 1)])
        return {
            "coef": pd.DataFrame(res["coef"], index=index,
                                 columns=coef_cols),
            "r2": pd.Series(res["r2"], index=index),
            "residual_std": pd.Series(res["residual_std"], index=index),
        }
    return _copy_meta(fn, core_fn)


rolling_mean     = _wrap_1d(_core.rolling_mean)
rolling_var      = _wrap_1d(_core.rolling_var)
rolling_std      = _wrap_1d(_core.rolling_std)
rolling_sum      = _wrap_1d(_core.rolling_sum)
rolling_min      = _wrap_1d(_core.rolling_min)
rolling_max      = _wrap_1d(_core.rolling_max)
rolling_quantile = _wrap_1d(_core.rolling_quantile)
rolling_skew     = _wrap_1d(_core.rolling_skew)
rolling_kurt     = _wrap_1d(_core.rolling_kurt)
rolling_zscore   = _wrap_1d(_core.rolling_zscore)
expanding_mean   = _wrap_1d(_core.expanding_mean)
expanding_var    = _wrap_1d(_core.expanding_var)
expanding_std    = _wrap_1d(_core.expanding_std)
expanding_sum    = _wrap_1d(_core.expanding_sum)

rolling_corr     = _wrap_xy(_core.rolling_corr)
rolling_cov      = _wrap_xy(_core.rolling_cov)
rolling_spearman = _wrap_xy(_core.rolling_spearman)

rolling_regression    = _wrap_dict(_core.rolling_regression)
expanding_regression  = _wrap_dict(_core.expanding_regression)
rolling_regression_xy = _wrap_xy(_core.rolling_regression_xy)

rolling_multiple_regression = \
    _wrap_multiple_regression(_core.rolling_multiple_regression)

rolling_mean_2d = _wrap_2d(_core.rolling_mean_2d)
rolling_std_2d  = _wrap_2d(_core.rolling_std_2d)
rolling_sum_2d  = _wrap_2d(_core.rolling_sum_2d)
rolling_min_2d  = _wrap_2d(_core.rolling_min_2d)
rolling_max_2d  = _wrap_2d(_core.rolling_max_2d)

rolling_corr_matrix = _wrap_2d_plain(_core.rolling_corr_matrix)

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
    "rolling_skew",
    "rolling_kurt",
    "rolling_zscore",
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
