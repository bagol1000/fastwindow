# fastwindow / fastroll

[![PyPI version](https://img.shields.io/pypi/v/fastwindow.svg)](https://pypi.org/project/fastwindow/)
[![CI](https://github.com/bagol1000/fastwindow/actions/workflows/ci.yml/badge.svg)](https://github.com/bagol1000/fastwindow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance rolling window statistics with a shared C++17 core
(AVX2 + OpenMP), exposed to **Python** (pybind11) as `fastwindow` and to **R** (Rcpp) as `fastroll`.

Rolling mean / std / var / sum / min / max, simple and multiple OLS
regression, Pearson correlation and covariance, correlation matrices,
Spearman rank correlation, quantiles (exact and P²-approximate),
expanding windows, and OpenMP multi-column dispatch — all with
consistent NaN semantics and `min_periods` support.

## Quick start

**Python**

```python
import numpy as np, fastwindow as fw
fw.rolling_mean(np.random.randn(1_000_000), window=252)
```

**R**

```r
library(fastroll)
rolling_mean(rnorm(1e6), window = 252)
```

## Usage

All examples assume `import numpy as np, fastwindow as fw`. Every function
returns a NaN-padded array of the same length as the input (the first
`window - 1` positions are NaN unless you lower `min_periods`).

### Basic rolling statistics

```python
x = np.array([1., 2., 3., 4., 5., 6.])

fw.rolling_mean(x, window=3)              # [nan, nan, 2., 3., 4., 5.]
fw.rolling_sum(x, window=3)
fw.rolling_min(x, window=3)
fw.rolling_max(x, window=3)

# variance / std take a ddof (1 = sample, 0 = population)
fw.rolling_std(x, window=3, ddof=1)
fw.rolling_var(x, window=3, ddof=0)

# emit values earlier with min_periods
fw.rolling_mean(x, window=3, min_periods=1)   # no leading NaNs
```

### NaN handling

```python
x = np.array([1., np.nan, 3., 4., 5.])

# default: any NaN in the window propagates to the output
fw.rolling_mean(x, window=3)

# skip_nan=True ignores NaNs; min_periods controls the minimum valid count
fw.rolling_mean(x, window=3, skip_nan=True, min_periods=2)
```

### Quantiles

```python
x = np.random.randn(1000)

fw.rolling_quantile(x, window=50, q=0.5)               # exact rolling median
fw.rolling_quantile(x, window=50, q=0.9, exact=False)  # fast P²-approximate
```

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

### Threads and CPU features

```python
fw.set_num_threads(8)     # 0 = library default
fw.get_num_threads()
fw.has_avx2()             # True if the AVX2 kernels are active
```

### R

The R API mirrors the Python one (note `rolling_*_matrix` for the 2-D variants):

```r
library(fastroll)
x <- rnorm(1000); y <- 0.5 * x + rnorm(1000)

rolling_mean(x, window = 50, min_periods = 1)
rolling_std(x, window = 50, ddof = 1)
rolling_quantile(x, window = 50, q = 0.9, exact = FALSE)

rolling_corr(x, y, window = 60)
rolling_spearman(x, y, window = 60)

fit <- rolling_regression(y, window = 60)   # list: intercept, slope, r2
fit$slope

X <- matrix(rnorm(10000 * 8), ncol = 8)
rolling_mean_matrix(X, window = 100)
```

## Install

```bash
pip install fastwindow            # Python (not yet)
R CMD INSTALL fastroll_0.1.0.tar.gz   # R (not yet)
```

For maximum performance build from source on the target machine; the
distributed wheels use portable ISA with scalar fallbacks, while a
source build adds `-march=native` AVX2 kernels automatically.

## Benchmarks

10M doubles, window 252, single thread, median of 5 runs
([full results & methodology](docs/benchmarks.md)):

| Operation | fastwindow | pandas | bottleneck | speedup vs pandas |
|---|---|---|---|---|
| mean | **31 ms** | 193 ms | 30 ms | 6.2× |
| std  | **38 ms** | 278 ms | 90 ms | 7.4× |
| min  | **33 ms** | 287 ms | 103 ms | 8.7× |
| max  | **32 ms** | 290 ms | 102 ms | 9.1× |
| simple regression (1M) | **7.7 ms** | ~28 s (`.apply`) | N/A | ~3,700× |

## Documentation

- Python: numpy-style docstrings (`help(fw.rolling_mean)`) and type
  stubs in `fastwindow/__init__.pyi`
- R: `?rolling_mean` etc. (roxygen2-generated man pages)
- [Benchmark comparison page](docs/benchmarks.md)

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
