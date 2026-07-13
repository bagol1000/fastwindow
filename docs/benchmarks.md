# fastwindow benchmarks

Current measurements for fastwindow 0.3.0 against pandas, Bottleneck and
Polars on identical deterministic inputs.

**Environment:** Intel Core Ultra 9 185H, Linux 6.17, Python 3.12,
runtime AVX2 dispatch active. Library versions: fastwindow 0.3.0,
NumPy 2.5.1, pandas 3.0.3, Bottleneck 1.6.0 and Polars 1.42.1.

**Method:** random N(0,1) doubles generated with seed 20260713, window 252,
one warm-up followed by five measured runs (three for the largest specialized
operations). Tables report the median wall-clock time and include output
allocation. Competitors and fastwindow 1-D calls use one thread; explicitly
multithreaded rows are labelled. Laptop frequency scaling means absolute
results can vary, so version, hardware and methodology are part of every
comparison.

## Reproduce

Run with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `POLARS_MAX_THREADS=1`:

```python
import gc, platform, statistics, time
import bottleneck as bn
import fastwindow as fw
import numpy as np
import pandas as pd
import polars as pl

rng = np.random.default_rng(20260713)
w = 252

def timed(fn, reps=5):
    fn()
    samples = []
    for _ in range(reps):
        gc.collect()
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000)
        del result
    return statistics.median(samples)

n = 10_000_000
x = rng.standard_normal(n)
s, ps = pd.Series(x), pl.Series(x)
cases = [
    ("mean", lambda: fw.rolling_mean(x, w), lambda: s.rolling(w).mean(),
     lambda: bn.move_mean(x, w), lambda: ps.rolling_mean(w)),
    ("std", lambda: fw.rolling_std(x, w), lambda: s.rolling(w).std(),
     lambda: bn.move_std(x, w, ddof=1), lambda: ps.rolling_std(w, ddof=1)),
    ("min", lambda: fw.rolling_min(x, w), lambda: s.rolling(w).min(),
     lambda: bn.move_min(x, w), lambda: ps.rolling_min(w)),
    ("max", lambda: fw.rolling_max(x, w), lambda: s.rolling(w).max(),
     lambda: bn.move_max(x, w), lambda: ps.rolling_max(w)),
    ("skew", lambda: fw.rolling_skew(x, w), lambda: s.rolling(w).skew(),
     None, lambda: ps.rolling_skew(w)),
    ("kurt", lambda: fw.rolling_kurt(x, w), lambda: s.rolling(w).kurt(),
     None, lambda: ps.rolling_kurtosis(w)),
    ("zscore", lambda: fw.rolling_zscore(x, w),
     lambda: (s - s.rolling(w).mean()) / s.rolling(w).std(), None,
     lambda: (ps - ps.rolling_mean(w)) / ps.rolling_std(w, ddof=1)),
    ("median", lambda: fw.rolling_quantile(x, w, q=0.5),
     lambda: s.rolling(w).median(), lambda: bn.move_median(x, w),
     lambda: ps.rolling_median(w)),
]
print(platform.platform(), fw.__version__, np.__version__, pd.__version__,
      bn.__version__, pl.__version__)
for name, *functions in cases:
    print(name, *[timed(fn, 3 if name == "median" else 5)
                   if fn else None for fn in functions])

buf = np.empty_like(x)
print("mean paths",
      timed(lambda: fw.rolling_mean(x, w, n_threads=1), 7),
      timed(lambda: fw.rolling_mean(x, w, n_threads=4), 7),
      timed(lambda: fw.rolling_mean(x, w, n_threads=4, out=buf), 7))

n2 = 1_000_000
y, x2 = rng.standard_normal(n2), rng.standard_normal(n2)
X3 = np.asfortranarray(rng.standard_normal((n2, 3)))
print("corr", timed(lambda: fw.rolling_corr(x2, y, w)),
      timed(lambda: pd.Series(x2).rolling(w).corr(pd.Series(y)), 3))
print("simple regression", timed(lambda: fw.rolling_regression(y, w)))
print("multiple regression", timed(
    lambda: fw.rolling_multiple_regression(y, X3, w), 3))
print("spearman", timed(
    lambda: fw.rolling_spearman(x2[:100_000], y[:100_000], w), 3))

M = np.asfortranarray(rng.standard_normal((200_000, 10)))
print("corr cube", timed(
    lambda: fw.rolling_corr_matrix(M, w, n_threads=4), 3))
print("corr pairs", timed(
    lambda: fw.rolling_corr_pairs(M, w, n_threads=4), 3))
print("output MiB", M.shape[0] * 10 * 10 * 8 / 1024**2,
      M.shape[0] * 45 * 8 / 1024**2)
```

## Basic statistics (10M elements, window 252)

Measured on 2026-07-13 with fastwindow 0.3.0:

| Operation | fastwindow | pandas | Bottleneck | Polars | speedup vs pandas |
|---|---:|---:|---:|---:|---:|
| mean | **24.30 ms** | 123.39 ms | **24.02 ms** | 57.10 ms | 5.08x |
| std | **26.11 ms** | 190.52 ms | 50.21 ms | 123.85 ms | 7.30x |
| min | **16.62 ms** | 217.27 ms | 86.62 ms | 144.24 ms | 13.07x |
| max | **16.28 ms** | 216.84 ms | 87.01 ms | 147.63 ms | 13.32x |
| skew | **85.32 ms** | 216.64 ms | n/a | 242.12 ms | 2.54x |
| kurt | **46.07 ms** | 197.99 ms | n/a | 201.80 ms | 4.30x |
| z-score | **186.63 ms** | 345.73 ms | n/a | 208.06 ms | 1.85x |
| exact median | 1303.44 ms | 3168.08 ms | **435.84 ms** | 654.74 ms | 2.43x |

The mean is effectively tied with Bottleneck. fastwindow is 1.92x faster
than Bottleneck for standard deviation and about 5.3x faster for min/max.
The fused z-score prioritizes low peak memory: it has no n-sized internal
temporary on the single-threaded path, while pandas and Polars evaluate two
rolling statistics before combining them. Exact median is the measured
counterexample: Bottleneck is 2.99x and Polars is 1.99x faster than fastwindow for that operation on this setup.

## Specialized operations

All values below are direct measurements; no extrapolated pandas timings are
used.

| Operation | Dataset | fastwindow | Comparison |
|---|---|---:|---:|
| pairwise correlation | n=1M | **6.26 ms** | pandas 85.04 ms (13.6x) |
| simple regression | n=1M | **12.61 ms** | no direct equivalent |
| multiple regression, k=3 | n=1M | **53.80 ms** | no direct equivalent |
| Spearman correlation | n=100k | **148.79 ms** | no direct rolling equivalent |
| full correlation cube, p=10, 4 threads | n=200k | **80.69 ms** | 152.6 MiB output |
| packed correlation pairs, p=10, 4 threads | n=200k | **18.91 ms** | 68.7 MiB output |

For p=10, `rolling_corr_pairs` is 4.27x faster in this measurement and its
output is 2.22x smaller because it stores only the 45 unique off-diagonal
pairs instead of 100 matrix cells per timestamp.

## Tuned-path timings

`rolling_mean` on 10M elements, median of seven runs:

| Path | Time |
|---|---:|
| one thread, fresh output | 16.47 ms |
| four threads, fresh output | **8.60 ms** |
| four threads, reused `out=` | 8.78 ms |

Four threads provide a 1.92x speedup here. Reusing `out=` removes allocation
ownership and is useful for controlling memory in loops, but it did not
improve the median time of a single call on this allocator and machine. This
is intentionally reported instead of claiming an allocation speedup that was
not reproduced.

## Memory behavior

`rolling_zscore` uses only its output array on the default fused path and one
additional n-element mean array on the explicit multithreaded path.
`rolling_corr_matrix` writes directly to the final cube with at most one
n-element pair buffer per worker. For n=200k and p=10, the measured output
sizes are 152.6 MiB for the full cube and 68.7 MiB for packed pairs.

## Interpretation limits

These are wall-clock measurements from one laptop, not universal constants.
CPU frequency, thermal state, allocator, compiler and dependency versions can
change absolute times. Relative claims above refer only to the stated setup.
The reproduction code deliberately includes allocation and uses medians after
a warm-up; it does not report extrapolated timings.
