"""Type stubs and reference documentation for the fastwindow public API."""

from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str

def rolling_mean(
    x: ArrayLike, window: int, min_periods: int = ..., skip_nan: bool = ...,
    n_threads: int = ..., out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling mean over a sliding window.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Input series (float64; other dtypes are converted).
    window : int
        Window length, > 0.
    min_periods : int, optional
        Minimum number of valid observations required to emit a value.
        Defaults to ``window``.
    skip_nan : bool, default False
        If True, NaN values are excluded from the window instead of
        propagating to the output.

    Returns
    -------
    np.ndarray, shape (n,)
        The first ``window - 1`` entries are NaN unless
        ``min_periods < window``.

    Examples
    --------
    >>> fw.rolling_mean(np.array([1., 2., 3., 4., 5.]), window=3)
    array([nan, nan,  2.,  3.,  4.])
    """
    ...

def rolling_var(
    x: ArrayLike, window: int, min_periods: int = ..., ddof: int = ...,
    skip_nan: bool = ..., n_threads: int = ...,
    out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling variance (sliding Welford algorithm).

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
        Defaults to ``window``.
    ddof : int, default 1
        1 for the sample estimator (n-1 denominator), 0 for population.
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def rolling_std(
    x: ArrayLike, window: int, min_periods: int = ..., ddof: int = ...,
    skip_nan: bool = ..., n_threads: int = ...,
    out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling standard deviation; square root of :func:`rolling_var`.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
    ddof : int, default 1
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)

    Examples
    --------
    >>> fw.rolling_std(np.array([1., 3., 5., 7.]), window=2)
    array([nan, 1.41421356, 1.41421356, 1.41421356])
    """
    ...

def rolling_sum(
    x: ArrayLike, window: int, min_periods: int = ..., skip_nan: bool = ...,
    n_threads: int = ..., out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling sum with compensated (Kahan) recomputation for stability.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def rolling_min(
    x: ArrayLike, window: int, min_periods: int = ..., skip_nan: bool = ...,
    n_threads: int = ..., out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling minimum.

    By default any NaN inside the window yields NaN for that position;
    ``skip_nan=True`` ignores NaN values instead (pandas semantics) and
    ``min_periods`` counts only the valid observations.  AVX2 builds use
    a blocked van Herk algorithm; otherwise a monotonic deque (O(n)
    total either way).

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
        Defaults to ``window``.
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def rolling_max(
    x: ArrayLike, window: int, min_periods: int = ..., skip_nan: bool = ...,
    n_threads: int = ..., out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling maximum; mirror of :func:`rolling_min`.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def rolling_regression(
    y: ArrayLike, window: int, min_periods: int = ..., time_as_x: bool = ...
) -> dict[str, NDArray[np.float64]]:
    """
    Rolling OLS of y on the within-window time index 0..window-1.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Dependent variable.
    window : int
    min_periods : int, optional
    time_as_x : bool, default True
        Must be True; pass an explicit regressor via
        :func:`rolling_regression_xy` instead.

    Returns
    -------
    dict
        Keys ``'intercept'``, ``'slope'``, ``'r2'``; each an array of
        shape (n,).
    """
    ...

def rolling_regression_xy(
    y: ArrayLike, x: ArrayLike, window: int, min_periods: int = ...
) -> dict[str, NDArray[np.float64]]:
    """
    Rolling OLS of y on an explicit regressor series x.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional

    Returns
    -------
    dict
        Keys ``'intercept'``, ``'slope'``, ``'r2'``.
    """
    ...

def rolling_multiple_regression(
    y: ArrayLike, X: ArrayLike, window: int, min_periods: int = ...
) -> dict[str, NDArray[np.float64]]:
    """
    Rolling OLS with k regressors (intercept added internally).

    Maintains (X'X)^-1 via Sherman-Morrison rank-1 updates with a Cholesky
    rebuild on near-singular steps.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    X : np.ndarray, shape (n, k)
        1 <= k <= 16 regressors; do not include an intercept column.
    window : int
    min_periods : int, optional

    Returns
    -------
    dict
        ``'coef'`` of shape (n, k+1) with the intercept first,
        ``'r2'`` and ``'residual_std'`` of shape (n,).

    Raises
    ------
    ValueError
        If k > 16.
    """
    ...

def rolling_corr(
    x: ArrayLike, y: ArrayLike, window: int, min_periods: int = ...,
    return_cov: bool = ..., skip_nan: bool = ...,
    out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Rolling Pearson correlation between two series.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
    return_cov : bool, default False
        If True, return a ``(corr, cov)`` tuple (sample covariance).
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        Windows where either series is constant yield NaN, never inf.
    """
    ...

def rolling_cov(
    x: ArrayLike, y: ArrayLike, window: int, min_periods: int = ...,
    ddof: int = ..., skip_nan: bool = ..., out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling covariance between two series.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
    ddof : int, default 1
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def rolling_corr_matrix(
    X: ArrayLike, window: int, min_periods: int = ..., n_threads: int = ...
) -> NDArray[np.float64]:
    """
    Rolling correlation matrix across the columns of X (OpenMP over pairs).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        2 <= p <= 50 columns.
    window : int
    min_periods : int, optional
    n_threads : int, default 0
        0 uses the library default (see :func:`set_num_threads`).

    Returns
    -------
    np.ndarray, shape (n, p, p)
        Symmetric slices with unit diagonal.
    """
    ...

def rolling_skew(
    x: ArrayLike, window: int, min_periods: int = ..., skip_nan: bool = ...,
    out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling skewness, bias-corrected.

    Matches ``pandas.Series.rolling(window).skew()`` and
    ``scipy.stats.skew(bias=False)``.  Requires at least 3 valid
    observations; zero-variance windows give NaN.  Internally uses
    shifted power sums re-anchored to the window mean every 4096 steps,
    so large mean offsets do not lose precision.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
        Defaults to ``window``.
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def rolling_kurt(
    x: ArrayLike, window: int, min_periods: int = ..., skip_nan: bool = ...,
    out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling excess kurtosis, bias-corrected.

    Matches ``pandas.Series.rolling(window).kurt()`` and
    ``scipy.stats.kurtosis(bias=False)``.  Requires at least 4 valid
    observations; zero-variance windows give NaN.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def rolling_zscore(
    x: ArrayLike, window: int, min_periods: int = ..., ddof: int = ...,
    skip_nan: bool = ..., n_threads: int = ...,
    out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling z-score: ``(x - rolling mean) / rolling std``.

    NaN where the input is NaN, the window does not emit (see
    ``min_periods``), or the window standard deviation is zero.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional
    ddof : int, default 1
    skip_nan : bool, default False

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def rolling_quantile(
    x: ArrayLike, window: int, q: float = ..., min_periods: int = ...,
    exact: bool = ..., out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling quantile.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    window : int
    q : float, default 0.5
        Strictly between 0 and 1.
    min_periods : int, optional
    exact : bool, default True
        True uses an exact two-heap order-statistic structure (O(log window)
        amortised per step; matches ``numpy.percentile``); any window size.
        False uses the P-squared streaming approximation (Jain & Chlamtac,
        1985) over observations seen so far, so it is O(1) per step but is
        not an exact rolling-window quantile on drifting distributions.

    Returns
    -------
    np.ndarray, shape (n,)

    Raises
    ------
    ValueError
        If q is not in (0, 1).
    """
    ...

def rolling_spearman(
    x: ArrayLike, y: ArrayLike, window: int, min_periods: int = ...,
    out: NDArray[np.float64] | None = ...
) -> NDArray[np.float64]:
    """
    Rolling Spearman rank correlation (average ranks for ties).

    O(window) per step via three linear passes over sorted window copies;
    practical for windows up to a few thousand elements.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
    window : int
    min_periods : int, optional

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def expanding_mean(x: ArrayLike, min_periods: int = ...) -> NDArray[np.float64]:
    """
    Expanding (growing-window) mean; NaN values are skipped.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    min_periods : int, default 1

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def expanding_var(
    x: ArrayLike, min_periods: int = ..., ddof: int = ...
) -> NDArray[np.float64]:
    """
    Expanding variance (Welford).

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    min_periods : int, default 2
    ddof : int, default 1

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def expanding_std(
    x: ArrayLike, min_periods: int = ..., ddof: int = ...
) -> NDArray[np.float64]:
    """
    Expanding standard deviation.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    min_periods : int, default 2
    ddof : int, default 1

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def expanding_sum(x: ArrayLike, min_periods: int = ...) -> NDArray[np.float64]:
    """
    Expanding sum (Kahan-compensated).

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    min_periods : int, default 1

    Returns
    -------
    np.ndarray, shape (n,)
    """
    ...

def expanding_regression(
    y: ArrayLike, min_periods: int = ..., time_as_x: bool = ...
) -> dict[str, NDArray[np.float64]]:
    """
    Expanding OLS of y on the time index.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    min_periods : int, optional
        Effective minimum is 2.
    time_as_x : bool, default True
        Only True is supported.

    Returns
    -------
    dict
        Keys ``'intercept'``, ``'slope'``, ``'r2'``.
    """
    ...

def rolling_mean_2d(
    X: ArrayLike, window: int, min_periods: int = ..., n_threads: int = ...
) -> NDArray[np.float64]:
    """
    Column-wise rolling mean over an (n, p) array (OpenMP over columns).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    window : int
    min_periods : int, optional
    n_threads : int, default 0

    Returns
    -------
    np.ndarray, shape (n, p)
    """
    ...

def rolling_std_2d(
    X: ArrayLike, window: int, min_periods: int = ..., ddof: int = ...,
    n_threads: int = ...
) -> NDArray[np.float64]:
    """
    Column-wise rolling standard deviation (OpenMP over columns).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    window : int
    min_periods : int, optional
    ddof : int, default 1
    n_threads : int, default 0

    Returns
    -------
    np.ndarray, shape (n, p)
    """
    ...

def rolling_sum_2d(
    X: ArrayLike, window: int, min_periods: int = ..., n_threads: int = ...
) -> NDArray[np.float64]:
    """
    Column-wise rolling sum (OpenMP over columns).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    window : int
    min_periods : int, optional
    n_threads : int, default 0

    Returns
    -------
    np.ndarray, shape (n, p)
    """
    ...

def rolling_min_2d(
    X: ArrayLike, window: int, min_periods: int = ..., skip_nan: bool = ...,
    n_threads: int = ...
) -> NDArray[np.float64]:
    """
    Column-wise rolling minimum (OpenMP over columns).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    window : int
    min_periods : int, optional
    skip_nan : bool, default False
    n_threads : int, default 0

    Returns
    -------
    np.ndarray, shape (n, p)
    """
    ...

def rolling_max_2d(
    X: ArrayLike, window: int, min_periods: int = ..., skip_nan: bool = ...,
    n_threads: int = ...
) -> NDArray[np.float64]:
    """
    Column-wise rolling maximum (OpenMP over columns).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    window : int
    min_periods : int, optional
    skip_nan : bool, default False
    n_threads : int, default 0

    Returns
    -------
    np.ndarray, shape (n, p)
    """
    ...

def set_num_threads(n: int) -> None:
    """
    Set the default OpenMP thread count used when ``n_threads=0``.

    Parameters
    ----------
    n : int
        A positive integer.
    """
    ...

def get_num_threads() -> int:
    """Return the current default OpenMP thread count."""
    ...

def has_avx2() -> bool:
    """Return True if the extension was compiled with AVX2 support."""
    ...
