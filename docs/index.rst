fastwindow / fastroll
=====================

Rolling-window statistics that nothing else computes fast — rolling OLS
regression, correlation matrices, Spearman rank correlation, exact
quantiles, skewness/kurtosis — plus the basics (mean/std/min/max) at
bottleneck-or-better speed.  One C++17 core (runtime-dispatched AVX2 +
OpenMP), exposed to Python as ``fastwindow`` and to R as ``fastroll``
with identical semantics.

Install
-------

.. code-block:: bash

   pip install fastwindow

Wheels select the AVX2 kernels at runtime (CPUID), so a plain
``pip install`` gets full speed on any x86-64 machine made this decade.

Quick start
-----------

.. code-block:: python

   import numpy as np, fastwindow as fw

   x = np.random.randn(1_000_000)
   fw.rolling_mean(x, window=252)
   fw.rolling_regression(np.cumsum(x), window=252)["slope"]
   fw.expanding_quantile_approx(x, q=0.95)

   # pandas objects preserve metadata; paired indexes must match
   import pandas as pd
   fw.rolling_zscore(pd.Series(x), window=252)

Every function returns a NaN-padded array of the same length as the
input; ``min_periods`` emits values earlier, ``skip_nan`` switches to
pandas-style handling where available. NaN and infinities are treated as non-finite.

.. toctree::
   :maxdepth: 1

   api
   benchmarks
