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

## Install

```bash
pip install fastwindow            # Python
R CMD INSTALL fastroll_0.1.0.tar.gz   # R (CRAN submission pending)
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
