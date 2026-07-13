# Changelog

All notable changes to the **fastwindow** (Python) / **fastroll** (R)
packages are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-07-13

### Added

- Public `rolling_corr_pairs` output for memory-efficient upper-triangle correlations.
- `expanding_quantile_approx`, the explicitly named P-squared stream estimator.
- Artifact, property-based, ASan and UBSan checks in CI; PyPI Trusted Publishing.

### Changed

- `rolling_corr_matrix` no longer allocates a full pair-major triangle before expansion.
- `rolling_zscore` uses no n-sized temporary in its default fused path and one in the parallel path.
- Benchmarks were re-recorded for 0.3.0 with deterministic inputs and no extrapolated timings.
- pandas pair operations require identical indexes instead of silently combining positions.
- R integer arguments are validated before conversion; fractional and out-of-range values fail.
- Python distribution metadata includes the canonical MIT text only.

### Fixed

- Reject partially overlapping NumPy `out` and input views, which could corrupt results.
- Exclude development metadata from the R source package and stale `tests` from wheels.
- Align the Spearman complexity description and document all non-finite handling.

### Deprecated

- `rolling_quantile(exact=False)`; use `expanding_quantile_approx` for the same stream estimator.

## [0.2.0] — 2026-07-12

### Added

- **Runtime AVX2 dispatch.**  The AVX2 kernels are now compiled into every
  x86-64 build (via per-function target attributes) and selected at runtime
  with a CPUID check.  Distributable wheels and default `R CMD INSTALL`
  builds run the fast blocked SIMD paths — previously they silently fell
  back to scalar code.  `has_avx2()` now reports the runtime state.
- `rolling_skew` and `rolling_kurt` — bias-corrected rolling skewness and
  excess kurtosis matching `pandas .rolling().skew()/.kurt()`
  (scipy `bias=False` conventions).  Implemented with mean-anchored shifted
  power sums, so precision holds even when `|mean| >> stddev`.
- `rolling_zscore` — `(x − rolling mean) / rolling std` with the shared
  NaN gating; composes the existing mean/std kernels (inherits the AVX2
  and `n_threads` paths).
- `rolling_min` / `rolling_max` (and the matrix/2-D variants) accept
  `min_periods` and `skip_nan`.  `skip_nan=True` gives pandas semantics
  (NaN ignored, `min_periods` counts valid observations).
- **pandas support** (Python): every function accepts `pandas.Series` /
  `pandas.DataFrame` and mirrors the container on output with the index
  (and columns) preserved.  pandas remains a soft dependency — fastwindow
  imports and works without it.  Plain arrays stay plain arrays.
- List and other array-like inputs are converted via `np.asarray`.

### Changed

- `has_avx2()` semantics: reports whether the AVX2 paths are *active on
  the current CPU*, not merely compiled in.

## [0.1.0] — 2026-06-12

Initial release.

- Rolling mean / var / std / sum (sliding Welford + blocked AVX2 scans),
  min / max (monotonic deque + blocked van Herk AVX2), with `min_periods`
  and `skip_nan` on the moment kernels.
- Rolling simple OLS regression on time or an explicit regressor;
  multiple regression (up to 16 regressors, Sherman–Morrison with
  Cholesky fallback).
- Rolling Pearson correlation / covariance, pairwise correlation
  matrices (OpenMP over pairs), Spearman rank correlation with average
  ranks.
- Rolling quantile: exact two-heap order statistics (default) or P²
  streaming approximation.
- Expanding mean / var / std / sum / regression.
- Column-wise 2-D variants parallelised with OpenMP; `n_threads` and
  `out=` on the hot 1-D kernels.
- Python (pybind11) and R (Rcpp) bindings over the shared C++17 core;
  py.typed stubs, roxygen2 man pages; CI on Linux/Windows/macOS,
  Python 3.10–3.14.
