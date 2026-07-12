# fastroll 0.2.0 (2026-07-12)

- The AVX2 kernels are now compiled into every x86-64 build and selected
  at runtime with a CPUID check, so a default `R CMD INSTALL` (portable
  compiler flags, CRAN-compliant) runs the fast SIMD paths.  `has_avx2()`
  reports the runtime state.
- New functions: `rolling_skew()`, `rolling_kurt()` (bias-corrected,
  excess kurtosis; conventions of `e1071::skewness(type = 2)` /
  `kurtosis(type = 2)` and pandas), and `rolling_zscore()`.
- `rolling_min()`, `rolling_max()`, `rolling_min_matrix()` and
  `rolling_max_matrix()` gain `min_periods` and `skip_nan` arguments;
  `skip_nan = TRUE` ignores missing values like
  `min(x, na.rm = TRUE)` per window.

# fastroll 0.1.0 (2026-06-12)

- Initial release: rolling mean/var/std/sum/min/max, simple and multiple
  OLS regression, Pearson/Spearman correlation, covariance, correlation
  matrices, exact and approximate quantiles, expanding windows, and
  OpenMP multi-column dispatch over a shared C++17 core (the Python
  sibling package is published as `fastwindow`).
