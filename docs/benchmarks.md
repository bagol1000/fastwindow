# fastwindow benchmarks

Comparison against pandas, bottleneck and polars on identical inputs.

**Setup:** n = 10,000,000 random N(0,1) doubles, window = 252.
Median of 5 runs, including output-array allocation.
Versions: pandas 3.0.3, bottleneck 1.6.0, polars 1.41.2, numpy 1.26.4.

> Benchmarks run on an Intel Core Ultra 9 185H (laptop), single thread
> unless noted.  Laptop frequency scaling makes absolute numbers vary
> between runs by up to 2×; the relative ordering is stable.

## Reproduce

```python
import fastwindow as fw, pandas as pd, bottleneck as bn
import numpy as np, time, statistics

n, w = 10_000_000, 252
x = np.random.randn(n)
s = pd.Series(x)

def t(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

print("mean", t(lambda: fw.rolling_mean(x, w)),
              t(lambda: s.rolling(w).mean()),
              t(lambda: bn.move_mean(x, w)))
# ... same pattern for std / min / max
```

## Basic statistics (10M elements, window 252)

| Operation | n | window | fastwindow | pandas | bottleneck | polars | speedup vs pandas |
|---|---|---|---|---|---|---|---|
| mean | 10M | 252 | **31.0 ms** | 192.7 ms | 30.0 ms | 53.2 ms | 6.2× |
| std  | 10M | 252 | **37.6 ms** | 277.8 ms | 89.8 ms | 116.7 ms | 7.4× |
| min  | 10M | 252 | **33.2 ms** | 287.3 ms | 103.0 ms | 135.3 ms | 8.7× |
| max  | 10M | 252 | **31.8 ms** | 290.1 ms | 102.4 ms | 130.2 ms | 9.1× |

fastwindow matches bottleneck on `mean` and is 2.4–3.2× faster on
`std`/`min`/`max` (blocked van Herk AVX2 kernels vs bottleneck's scalar
loops).

## Regression and correlation matrix (1M elements, window 252)

bottleneck has no equivalent for these operations.

| Operation | n | window | fastwindow | pandas | bottleneck | speedup |
|---|---|---|---|---|---|---|
| simple regression (slope+intercept+R²) | 1M | 252 | **7.7 ms** | ~28,300 ms (`.apply` + polyfit, extrapolated from 20k) | N/A — no equivalent | ~3,700× |
| correlation matrix p=10 (4 threads) | 1M | 252 | **254.5 ms** | ~3,860 ms (`.rolling().corr()`, extrapolated from 100k) | N/A — no equivalent | ~15× |

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
