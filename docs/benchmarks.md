# fastwindow benchmarks

Comparison against pandas, bottleneck and polars on identical inputs.

**Setup:** n = 10,000,000 random N(0,1) doubles, window = 252.
Median of 5 runs, including output-array allocation.
Recorded with fastwindow 0.2.0, pandas 3.0.3, bottleneck 1.6.0,
polars 1.41.2, numpy 1.26.4.

> Benchmarks run on an Intel Core Ultra 9 185H (laptop), single thread
> unless noted.  Laptop frequency scaling makes absolute numbers vary
> between runs by up to 2×; the relative ordering is stable.

## Reproduce

```python
import fastwindow as fw, pandas as pd, bottleneck as bn, polars as pl
import numpy as np, platform, statistics, time

n, w = 10_000_000, 252
rng = np.random.default_rng(20260713)
x = rng.standard_normal(n)
s = pd.Series(x)
ps = pl.Series(x)

def t(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

print("platform", platform.platform(), platform.processor())
print("versions", fw.__version__, np.__version__, pd.__version__, bn.__version__, pl.__version__)
for name, fast, pandas_fn, bottleneck_fn, polars_fn in [
    ("mean", lambda: fw.rolling_mean(x, w),
     lambda: s.rolling(w).mean(), lambda: bn.move_mean(x, w),
     lambda: ps.rolling_mean(w)),
    ("std", lambda: fw.rolling_std(x, w),
     lambda: s.rolling(w).std(), lambda: bn.move_std(x, w, ddof=1),
     lambda: ps.rolling_std(w, ddof=1)),
    ("min", lambda: fw.rolling_min(x, w),
     lambda: s.rolling(w).min(), lambda: bn.move_min(x, w),
     lambda: ps.rolling_min(w)),
    ("max", lambda: fw.rolling_max(x, w),
     lambda: s.rolling(w).max(), lambda: bn.move_max(x, w),
     lambda: ps.rolling_max(w)),
]:
    print(name, t(fast), t(pandas_fn), t(bottleneck_fn), t(polars_fn))
```

## Basic statistics (10M elements, window 252)

Historical 0.2.0 run (runtime AVX2 dispatch active; rerun the script for current 0.3.0 results):

| Operation | fastwindow | pandas | bottleneck | polars | speedup vs pandas |
|---|---|---|---|---|---|
| mean | **45 ms** | 225 ms | 47 ms | 96 ms | 5.0× |
| std  | **39 ms** | 317 ms | 126 ms | 122 ms | 8.1× |
| min  | **35 ms** | 316 ms | 105 ms | 144 ms | 9.0× |
| max  | **38 ms** | 311 ms | 103 ms | 139 ms | 8.3× |
| skew | **91 ms** | 278 ms | n/a | 256 ms | 3.1× |
| kurt | **68 ms** | 293 ms | n/a | n/a | 4.3× |

fastwindow matches bottleneck on `mean` and is 2.7–4.1× faster on
`std`/`min`/`max` (blocked van Herk AVX2 kernels vs bottleneck's scalar
loops), and 2.4–4.1× faster than polars across the board.

**Honest counterpoint:** polars wins on exact rolling *median*
(~635 ms vs ~1,506 ms for `rolling_quantile(q=0.5, exact=True)` on the
same input) — if the rolling median of a huge series is your bottleneck,
use polars for that call.

## Regression, correlation, rank statistics (1M elements, window 252)

The operations nothing else provides fast — bottleneck has none of
these; polars has only the pairwise correlation:

| Operation | fastwindow | pandas | polars | speedup |
|---|---|---|---|---|
| pairwise correlation | **3.9 ms** | 88 ms | 86 ms | 22× |
| simple regression (slope+intercept+R²) | **13.6 ms** | ~28,300 ms (`.apply` + polyfit, extrapolated from 20k) | no equivalent | ~2,000× |
| multiple regression k=3 (coef+R²+σ) | **55.6 ms** | ~8 s (`.apply`, extrapolated) | no equivalent | ~150× |
| correlation matrix p=10 (4 threads) | **464 ms** | ~3,860 ms (extrapolated from 100k) | no equivalent | ~8× |
| Spearman rank correlation (100k) | **148 ms** | no equivalent | no equivalent | — |
| z-score (10M) | **115 ms** | two rolling calls | two rolling calls | — |

## Tuned-path timings (optimization pass)

The 1-D kernels accept ``n_threads=`` (OpenMP over van Herk blocks,
bitwise-identical results for any thread count) and ``out=`` (buffer
reuse, skipping the fresh-allocation page faults).  10M doubles,
window 252, best of 7:

| Operation | default | n_threads=4 | n_threads=4 + out= |
|---|---|---|---|
| mean | 29.2 ms | 14.8 ms | **5.9 ms** |
| std  | 31 ms | 18.4 ms | ~8 ms |
| min  | 33 ms | 16.3 ms | ~8 ms |
| max  | 32 ms | 15.5 ms | ~8 ms |

In a tight loop of 100 calls, ``out=`` alone is 3.5× (4.01 s → 1.16 s
for ``rolling_mean`` on 10M).

Other algorithm upgrades in the same pass:

- **Spearman**: per-step binary searches replaced by three linear walks
  over sorted window copies — 1M elements at window 252 went from
  ~18.2 s to **1.4 s** (≈13×), and large windows are now practical.
- **Exact quantile**: sorted-buffer O(window) replaced by a two-heap
  lazy-deletion order statistic, O(log window) amortised — runtime is
  now window-independent (1M elements: ~134 ms at window 252 *or*
  window 2000) and the former window ≤ 500 limit is gone.
- **corr/cov**: rewritten on the blocked-scan structure.  Same speed as
  the previous serial kernel (the five independent sum chains already
  pipelined well — measured, not assumed), but accuracy improved to
  machine precision and the periodic recompute is no longer needed.

## Memory behavior in 0.3.0

`rolling_zscore` uses no n-sized temporary on its default fused path and one
on the explicit multithreaded path. `rolling_corr_matrix` writes directly to
the final cube with at most one n-element pair buffer per worker. Use
`rolling_corr_pairs` to avoid allocating the full `(n, p, p)` output.

## Kernel-only timings

The Python numbers above include allocating the output array on every
call.  Timed at the C++ kernel level (buffers reused, same machine at a
fixed ~2 GHz clock), with the scalar fallback (`-DFASTWINDOW_NO_AVX2`)
as reference:

| Kernel | scalar | AVX2 | speedup |
|---|---|---|---|
| min / max | 179 ms | 18.7 ms | 9.6× |
| mean | 24.7 ms | 17.8 ms | 1.4× |
| std | 101.5 ms | 23.7 ms | 4.3× |
| sum | 22.5 ms | 18.0 ms | 1.2× |
| corr | 57.9 ms | 58.3 ms | — (serial-chain bound) |
