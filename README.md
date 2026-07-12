# fastwindow / fastroll

[![PyPI version](https://img.shields.io/pypi/v/fastwindow.svg)](https://pypi.org/project/fastwindow/)
[![CI](https://github.com/bagol1000/fastwindow/actions/workflows/ci.yml/badge.svg)](https://github.com/bagol1000/fastwindow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Rolling-window statistics that **nothing else computes fast** — rolling
OLS regression, correlation matrices, Spearman rank correlation, exact
quantiles, skewness/kurtosis — plus the basics (mean/std/min/max) at
bottleneck-or-better speed.  One C++17 core (runtime-dispatched AVX2 +
OpenMP), two thin bindings: **Python** (`fastwindow`, pybind11) and
**R** (`fastroll`, Rcpp), with identical semantics.

```python
import numpy as np, fastwindow as fw

y = np.cumsum(np.random.randn(1_000_000))
fit = fw.rolling_regression(y, window=252)     # 13 ms — pandas .apply: ~28 s
fit["slope"], fit["intercept"], fit["r2"]
```

## Why this exists

`pandas.rolling().apply(fn)` runs a Python function once per window.
For the statistics that have no built-in fast path — regression,
correlation matrices, rank correlation — that means seconds to minutes
on series where an O(1)-per-step streaming kernel needs milliseconds.
bottleneck and polars solve this only for the simple moments.

1M doubles, window 252, single thread
([full results & methodology](docs/benchmarks.md)):

| Operation | fastwindow | pandas | polars |
|---|---|---|---|
| pairwise correlation | **3.9 ms** | 88 ms | 86 ms |
| simple regression (slope+intercept+R²) | **13.6 ms** | ~28 s (`.apply`) | no equivalent |
| multiple regression k=3 | **55.6 ms** | ~8 s (`.apply`) | no equivalent |
| correlation matrix p=10 (4 threads) | **464 ms** | ~3.9 s | no equivalent |
| Spearman rank correlation (100k) | **148 ms** | no equivalent | no equivalent |

And the basics hold their own (10M doubles, window 252):

| Operation | fastwindow | pandas | bottleneck | polars |
|---|---|---|---|---|
| mean | **45 ms** | 225 ms | 47 ms | 96 ms |
| std  | **39 ms** | 317 ms | 126 ms | 122 ms |
| min  | **35 ms** | 316 ms | 105 ms | 144 ms |
| skew | **91 ms** | 278 ms | n/a | 256 ms |
| kurt | **68 ms** | 293 ms | n/a | n/a |

When *not* to use fastwindow: if you only need rolling mean/sum,
bottleneck is just as fast; and polars beats us on exact rolling
*median* of huge series.  Everything else here is either faster or
doesn't exist elsewhere.

## Install

```bash
pip install fastwindow                # Python
R CMD INSTALL .                        # R, from this source tree
```

Wheels select the AVX2 kernels **at runtime** (CPUID), so a plain
`pip install` gets full speed on any x86-64 machine made this decade;
older CPUs fall back to scalar code automatically.  `fw.has_avx2()`
tells you which path is active.

## Usage

All examples assume `import numpy as np, fastwindow as fw`.  Every
function returns a NaN-padded array of the same length as the input
(the first `window - 1` positions are NaN unless you lower
`min_periods`).

**pandas objects work everywhere** (pandas is optional, not required):

```python
import pandas as pd
s = pd.Series(np.random.randn(1000),
              index=pd.date_range("2020-01-01", periods=1000))
fw.rolling_mean(s, window=50)          # -> Series, index preserved
```

### Basic rolling statistics

```python
x = np.array([1., 2., 3., 4., 5., 6.])

fw.rolling_mean(x, window=3)              # [nan, nan, 2., 3., 4., 5.]
fw.rolling_sum(x, window=3)
fw.rolling_min(x, window=3)               # min/max take min_periods/skip_nan too
fw.rolling_max(x, window=3, min_periods=1)

# variance / std take a ddof (1 = sample, 0 = population)
fw.rolling_std(x, window=3, ddof=1)
fw.rolling_var(x, window=3, ddof=0)

# emit values earlier with min_periods
fw.rolling_mean(x, window=3, min_periods=1)   # no leading NaNs
```

### Higher moments and z-score

```python
x = np.random.randn(10_000)

fw.rolling_skew(x, window=100)     # bias-corrected, matches pandas .skew()
fw.rolling_kurt(x, window=100)     # excess kurtosis, matches pandas .kurt()
fw.rolling_zscore(x, window=100)   # (x - rolling mean) / rolling std
```

### NaN handling

```python
x = np.array([1., np.nan, 3., 4., 5.])

# default: any NaN in the window propagates to the output
fw.rolling_mean(x, window=3)

# skip_nan=True ignores NaNs; min_periods controls the minimum valid count
fw.rolling_mean(x, window=3, skip_nan=True, min_periods=2)
fw.rolling_min(x, window=3, skip_nan=True, min_periods=1)   # pandas semantics
```

### Quantiles

```python
x = np.random.randn(1000)

fw.rolling_quantile(x, window=50, q=0.5)               # exact rolling median
fw.rolling_quantile(x, window=50, q=0.9, exact=False)  # fast P² streaming approximation
```

With `exact=True` (the default), the result is the exact quantile of the
current rolling window and matches NumPy/R type-7 interpolation.  With
`exact=False`, P² is an O(1) streaming estimator over observations seen so
far; it is useful for stationary series but is not an exact rolling-window
quantile on drifting distributions.

### Correlation and covariance

```python
x = np.random.randn(1000)
y = 0.5 * x + np.random.randn(1000)

fw.rolling_corr(x, y, window=60)                 # Pearson correlation
fw.rolling_cov(x, y, window=60)                  # covariance
fw.rolling_spearman(x, y, window=60)             # rank correlation

corr, cov = fw.rolling_corr(x, y, window=60, return_cov=True)

# pairwise correlation matrix across columns -> shape (n, p, p)
X = np.random.randn(1000, 4)
cubes = fw.rolling_corr_matrix(X, window=60)
cubes[-1]                                        # 4x4 matrix at the last step
```

### Rolling regression

```python
y = np.cumsum(np.random.randn(500))

# OLS of y on the within-window time index -> dict of arrays
res = fw.rolling_regression(y, window=60)
res["slope"], res["intercept"], res["r2"]

# OLS against an explicit regressor x
x = np.random.randn(500)
fw.rolling_regression_xy(y, x, window=60)

# multiple regression (k <= 16 regressors, intercept added automatically)
X = np.random.randn(500, 3)
out = fw.rolling_multiple_regression(y, X, window=60)
out["coef"]          # shape (n, k+1), intercept first
out["r2"], out["residual_std"]
```

With a DataFrame `X`, `out["coef"]` comes back as a DataFrame with
columns `["intercept", *X.columns]`.

### Expanding windows

```python
x = np.random.randn(1000)
fw.expanding_mean(x)
fw.expanding_std(x, min_periods=2)
fw.expanding_regression(y)
```

### Column-wise on 2-D arrays

Apply a rolling op independently to every column (OpenMP-parallelised):

```python
X = np.random.randn(10_000, 8)
fw.rolling_mean_2d(X, window=100)     # also _std / _sum / _min / _max
```

### Performance knobs

```python
fw.rolling_mean(x, window=252, n_threads=4)   # OpenMP over blocks, 1-D too
buf = np.empty_like(x)
fw.rolling_mean(x, window=252, out=buf)       # reuse output buffer

fw.set_num_threads(8)     # OpenMP default used when n_threads=0
fw.has_avx2()             # True if the AVX2 kernels are active on this CPU
```

`n_threads=` on the 1-D kernels splits the series into blocks with
bitwise-identical results for any thread count; combined with `out=`,
`rolling_mean` on 10M doubles drops to ~6 ms
([details](docs/benchmarks.md)).

### R

The R API mirrors the Python one (note `rolling_*_matrix` for the 2-D
variants):

```r
library(fastroll)
x <- rnorm(1000); y <- 0.5 * x + rnorm(1000)

rolling_mean(x, window = 50, min_periods = 1)
rolling_skew(x, window = 50)
rolling_zscore(x, window = 50)
rolling_quantile(x, window = 50, q = 0.9)

rolling_corr(x, y, window = 60)
rolling_spearman(x, y, window = 60)

fit <- rolling_regression(y, window = 60)   # list: intercept, slope, r2
fit$slope

X <- matrix(rnorm(10000 * 8), ncol = 8)
rolling_mean_matrix(X, window = 100)
```

## Documentation

- Python: numpy-style docstrings (`help(fw.rolling_mean)`) and type
  stubs in `fastwindow/__init__.pyi`
- R: `?rolling_mean` etc. (roxygen2-generated man pages)
- [Benchmark comparison page](docs/benchmarks.md)
- [Changelog](CHANGELOG.md)

## License

Released under the MIT License. See [LICENSE.md](LICENSE.md) for details.
