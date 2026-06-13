"""
Tests for fastwindow basic rolling statistics

Coverage:
  1. Hand-computed small cases (exact values)
  2. Comparison with pandas.rolling() - tol 1e-9 (mean), 1e-7 (std)
  3. NaN propagation (default) and skip_nan behaviour
  4. Edge cases: short arrays, window > len, all-NaN, single-element window
"""

import math
import numpy as np
import pandas as pd
import pytest
import fastwindow as fw

NaN = float("nan")

def assert_allclose_nan(actual, expected, rtol=1e-9, atol=1e-12, label=""):
    """Compare arrays element-wise, treating NaN == NaN."""
    actual   = np.asarray(actual,   dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{label}: shape mismatch"
    nan_a = np.isnan(actual)
    nan_e = np.isnan(expected)
    assert np.array_equal(nan_a, nan_e), \
        f"{label}: NaN pattern differs\n  actual  NaN mask: {nan_a}\n  expected NaN mask: {nan_e}"
    finite = ~nan_e
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=atol, err_msg=f"{label}: finite values differ")

#rolling_mean

class TestRollingMeanHandComputed:
    def test_basic_window3(self):
        x   = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = fw.rolling_mean(x, window=3)
        exp = np.array([NaN, NaN, 2.0, 3.0, 4.0])
        assert_allclose_nan(out, exp, label="mean window=3")

    def test_window_equals_len(self):
        x   = np.array([1.0, 2.0, 3.0])
        out = fw.rolling_mean(x, window=3)
        exp = np.array([NaN, NaN, 2.0])
        assert_allclose_nan(out, exp, label="mean window=len")

    def test_window1(self):
        x   = np.array([10.0, 20.0, 30.0])
        out = fw.rolling_mean(x, window=1)
        assert_allclose_nan(out, x, label="mean window=1")

    def test_min_periods(self):
        x   = np.array([1.0, 2.0, 3.0, 4.0])
        out = fw.rolling_mean(x, window=3, min_periods=2)
        exp = np.array([NaN, 1.5, 2.0, 3.0])
        assert_allclose_nan(out, exp, label="mean min_periods=2")

    def test_min_periods_1(self):
        x   = np.array([5.0, 6.0, 7.0])
        out = fw.rolling_mean(x, window=3, min_periods=1)
        exp = np.array([5.0, 5.5, 6.0])
        assert_allclose_nan(out, exp, label="mean min_periods=1")

    def test_constant_array(self):
        x   = np.ones(10) * 7.0
        out = fw.rolling_mean(x, window=4)
        exp = np.array([NaN, NaN, NaN] + [7.0] * 7)
        assert_allclose_nan(out, exp, label="mean constant")

    def test_negative_values(self):
        x   = np.array([-3.0, -1.0, 1.0, 3.0])
        out = fw.rolling_mean(x, window=2)
        exp = np.array([NaN, -2.0, 0.0, 2.0])
        assert_allclose_nan(out, exp, label="mean negative")

    def test_window_larger_than_array(self):
        x   = np.array([1.0, 2.0])
        out = fw.rolling_mean(x, window=5)
        exp = np.array([NaN, NaN])
        assert_allclose_nan(out, exp, label="mean window>len")

#rolling_std / rolling_var

class TestRollingStdHandComputed:
    def test_std_window2(self):
        x   = np.array([1.0, 3.0, 5.0, 7.0])
        out = fw.rolling_std(x, window=2, ddof=1)
        #std([1,3])=sqrt(2), std([3,5])=sqrt(2), std([5,7])=sqrt(2)
        v   = math.sqrt(2.0)
        exp = np.array([NaN, v, v, v])
        assert_allclose_nan(out, exp, rtol=1e-10, label="std window=2")

    def test_std_window3_sample(self):
        x   = np.array([2.0, 4.0, 6.0, 8.0])
        out = fw.rolling_std(x, window=3, ddof=1)
        #std([2,4,6], ddof=1) = std([4,6,8], ddof=1) = 2.0
        exp = np.array([NaN, NaN, 2.0, 2.0])
        assert_allclose_nan(out, exp, rtol=1e-10, label="std window=3 sample")

    def test_var_population(self):
        x   = np.array([1.0, 2.0, 3.0])
        out = fw.rolling_var(x, window=3, ddof=0)
        #var([1,2,3], ddof=0) = 2/3
        exp = np.array([NaN, NaN, 2.0 / 3.0])
        assert_allclose_nan(out, exp, rtol=1e-10, label="var population")

    def test_std_constant(self):
        x   = np.ones(6) * 5.0
        out = fw.rolling_std(x, window=3)
        exp = np.array([NaN, NaN, 0.0, 0.0, 0.0, 0.0])
        assert_allclose_nan(out, exp, atol=1e-14, label="std constant")

#rolling_sum

class TestRollingSumHandComputed:
    def test_basic(self):
        x   = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = fw.rolling_sum(x, window=3)
        exp = np.array([NaN, NaN, 6.0, 9.0, 12.0])
        assert_allclose_nan(out, exp, label="sum basic")

    def test_min_periods(self):
        x   = np.array([1.0, 2.0, 3.0])
        out = fw.rolling_sum(x, window=4, min_periods=2)
        exp = np.array([NaN, 3.0, 6.0])
        assert_allclose_nan(out, exp, label="sum min_periods")

    def test_single_element_window(self):
        x   = np.array([10.0, 20.0, 30.0])
        out = fw.rolling_sum(x, window=1)
        assert_allclose_nan(out, x, label="sum window=1")

#rolling_min / rolling_max

class TestRollingMinMaxHandComputed:
    def test_min_basic(self):
        x   = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        out = fw.rolling_min(x, window=3)
        exp = np.array([NaN, NaN, 1.0, 1.0, 1.0, 1.0])
        assert_allclose_nan(out, exp, label="min basic")

    def test_max_basic(self):
        x   = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        out = fw.rolling_max(x, window=3)
        exp = np.array([NaN, NaN, 4.0, 4.0, 5.0, 9.0])
        assert_allclose_nan(out, exp, label="max basic")

    def test_min_window1(self):
        x   = np.array([5.0, 3.0, 8.0])
        out = fw.rolling_min(x, window=1)
        assert_allclose_nan(out, x, label="min window=1")

    def test_max_window1(self):
        x   = np.array([5.0, 3.0, 8.0])
        out = fw.rolling_max(x, window=1)
        assert_allclose_nan(out, x, label="max window=1")

    def test_min_decreasing(self):
        x   = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        out = fw.rolling_min(x, window=3)
        exp = np.array([NaN, NaN, 3.0, 2.0, 1.0])
        assert_allclose_nan(out, exp, label="min decreasing")

    def test_max_increasing(self):
        x   = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = fw.rolling_max(x, window=3)
        exp = np.array([NaN, NaN, 3.0, 4.0, 5.0])
        assert_allclose_nan(out, exp, label="max increasing")

#Comparison with pandas.rolling()

RTOL_MEAN = 1e-9
RTOL_STD  = 1e-7

class TestVsPandas:
    @pytest.fixture
    def rng_data(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(1000)

    def _pd_rolling(self, x, window, method, **kwargs):
        s = pd.Series(x)
        return getattr(s.rolling(window, **kwargs), method)().to_numpy()

    def test_mean_vs_pandas(self, rng_data):
        x   = rng_data
        out = fw.rolling_mean(x, window=50)
        exp = self._pd_rolling(x, 50, "mean")
        assert_allclose_nan(out, exp, rtol=RTOL_MEAN, label="mean vs pandas")

    def test_mean_min_periods_vs_pandas(self, rng_data):
        x   = rng_data
        out = fw.rolling_mean(x, window=50, min_periods=25)
        exp = self._pd_rolling(x, 50, "mean", min_periods=25)
        assert_allclose_nan(out, exp, rtol=RTOL_MEAN, label="mean min_periods vs pandas")

    def test_std_vs_pandas(self, rng_data):
        x   = rng_data
        out = fw.rolling_std(x, window=50, ddof=1)
        exp = self._pd_rolling(x, 50, "std")
        assert_allclose_nan(out, exp, rtol=RTOL_STD, label="std vs pandas")

    def test_var_vs_pandas(self, rng_data):
        x   = rng_data
        out = fw.rolling_var(x, window=50, ddof=1)
        exp = self._pd_rolling(x, 50, "var")
        assert_allclose_nan(out, exp, rtol=RTOL_STD, label="var vs pandas")

    def test_sum_vs_pandas(self, rng_data):
        x   = rng_data
        out = fw.rolling_sum(x, window=50)
        exp = self._pd_rolling(x, 50, "sum")
        assert_allclose_nan(out, exp, rtol=RTOL_MEAN, label="sum vs pandas")

    def test_min_vs_pandas(self, rng_data):
        x   = rng_data
        out = fw.rolling_min(x, window=50)
        exp = self._pd_rolling(x, 50, "min")
        assert_allclose_nan(out, exp, rtol=RTOL_MEAN, label="min vs pandas")

    def test_max_vs_pandas(self, rng_data):
        x   = rng_data
        out = fw.rolling_max(x, window=50)
        exp = self._pd_rolling(x, 50, "max")
        assert_allclose_nan(out, exp, rtol=RTOL_MEAN, label="max vs pandas")

    def test_large_n_mean(self):
        """Numerical stability over 10k elements with reinit cycles."""
        rng = np.random.default_rng(7)
        x   = rng.standard_normal(10_000)
        out = fw.rolling_mean(x, window=252)
        exp = pd.Series(x).rolling(252).mean().to_numpy()
        assert_allclose_nan(out, exp, rtol=RTOL_MEAN, label="mean large n")

    def test_large_n_std(self):
        rng = np.random.default_rng(7)
        x   = rng.standard_normal(10_000)
        out = fw.rolling_std(x, window=252)
        exp = pd.Series(x).rolling(252).std().to_numpy()
        assert_allclose_nan(out, exp, rtol=RTOL_STD, label="std large n")

#NaN propagation (skip_nan=False, default)

class TestNanPropagation:
    def test_mean_nan_in_window(self):
        x   = np.array([1.0, NaN, 3.0, 4.0, 5.0])
        out = fw.rolling_mean(x, window=3)
        #Window [1,NaN,3] and [NaN,3,4] contain NaN → NaN output
        exp = np.array([NaN, NaN, NaN, NaN, 4.0])
        assert_allclose_nan(out, exp, label="mean nan propagation")

    def test_std_nan_in_window(self):
        #NaN at position 0: window [NaN,2,3] is tainted; windows [2,3,4] and [3,4,5] are clean.
        x   = np.array([NaN, 2.0, 3.0, 4.0, 5.0])
        out = fw.rolling_std(x, window=3)
        exp = np.array([NaN, NaN, NaN, 1.0, 1.0])
        assert_allclose_nan(out, exp, rtol=1e-10, label="std nan propagation")

    def test_sum_nan_in_window(self):
        x   = np.array([1.0, NaN, 3.0])
        out = fw.rolling_sum(x, window=2)
        exp = np.array([NaN, NaN, NaN])
        assert_allclose_nan(out, exp, label="sum nan propagation")

    def test_min_nan_propagation(self):
        x   = np.array([1.0, NaN, 3.0, 4.0])
        out = fw.rolling_min(x, window=2)
        exp = np.array([NaN, NaN, NaN, 3.0])
        assert_allclose_nan(out, exp, label="min nan propagation")

    def test_max_nan_propagation(self):
        x   = np.array([1.0, NaN, 3.0, 4.0])
        out = fw.rolling_max(x, window=2)
        exp = np.array([NaN, NaN, NaN, 4.0])
        assert_allclose_nan(out, exp, label="max nan propagation")

    def test_all_nan(self):
        x   = np.array([NaN, NaN, NaN])
        exp = np.array([NaN, NaN, NaN])
        assert_allclose_nan(fw.rolling_mean(x, window=2), exp, label="all nan mean")
        assert_allclose_nan(fw.rolling_min(x, window=2), exp,  label="all nan min")
        assert_allclose_nan(fw.rolling_max(x, window=2), exp,  label="all nan max")

    def test_nan_exits_window(self):
        """Once the NaN slides out of the window, outputs should resume."""
        x   = np.array([1.0, 2.0, NaN, 4.0, 5.0, 6.0])
        out = fw.rolling_mean(x, window=2)
        exp = np.array([NaN, 1.5, NaN, NaN, 4.5, 5.5])
        assert_allclose_nan(out, exp, label="nan exits window")

#skip_nan=True behaviour

class TestSkipNan:
    def test_mean_skip_nan(self):
        x   = np.array([1.0, NaN, 3.0, 4.0, 5.0])
        #min_periods=2: output once at least 2 valid values are in window.
        out = fw.rolling_mean(x, window=3, skip_nan=True, min_periods=2)
        #Window [1,NaN,3]  valid=[1,3]    mean=2.0
        #Window [NaN,3,4]  valid=[3,4]    mean=3.5
        #Window [3,4,5]    valid=[3,4,5]  mean=4.0
        exp = np.array([NaN, NaN, 2.0, 3.5, 4.0])
        assert_allclose_nan(out, exp, label="mean skip_nan")

    def test_mean_skip_nan_min_periods(self):
        x   = np.array([NaN, NaN, 3.0, 4.0])
        out = fw.rolling_mean(x, window=3, skip_nan=True, min_periods=1)
        exp = np.array([NaN, NaN, 3.0, 3.5])
        assert_allclose_nan(out, exp, label="mean skip_nan min_periods=1")

    def test_sum_skip_nan(self):
        x   = np.array([1.0, NaN, 3.0])
        out = fw.rolling_sum(x, window=3, skip_nan=True, min_periods=1)
        exp = np.array([1.0, 1.0, 4.0])
        assert_allclose_nan(out, exp, label="sum skip_nan")

    def test_std_skip_nan(self):
        x   = np.array([1.0, NaN, 3.0, 5.0])
        #min_periods=2: compute std once at least 2 valid values available.
        out = fw.rolling_std(x, window=3, skip_nan=True, ddof=1, min_periods=2)
        #Window [1,NaN,3]: valid=[1,3], std=sqrt(2) ≈1.4142
        #Window [NaN,3,5]: valid=[3,5], std=sqrt(2)
        exp = np.array([NaN, NaN, math.sqrt(2), math.sqrt(2)])
        assert_allclose_nan(out, exp, rtol=1e-10, label="std skip_nan")

    def test_var_skip_nan_vs_pandas(self):
        rng = np.random.default_rng(99)
        x   = rng.standard_normal(500)
        #Inject ~10 % NaN
        mask = rng.random(500) < 0.1
        x[mask] = NaN
        #Use min_periods=2 in both (sample var requires >= 2 values, ddof=1)
        out = fw.rolling_var(x, window=20, skip_nan=True, min_periods=2)
        exp = pd.Series(x).rolling(20, min_periods=2).var().to_numpy()
        assert_allclose_nan(out, exp, rtol=1e-7, label="var skip_nan vs pandas")

#Error handling

class TestErrorHandling:
    def test_window_zero_mean(self):
        with pytest.raises((ValueError, Exception)):
            fw.rolling_mean(np.array([1.0, 2.0]), window=0)

    def test_window_zero_std(self):
        with pytest.raises((ValueError, Exception)):
            fw.rolling_std(np.array([1.0, 2.0]), window=0)

    def test_window_zero_min(self):
        with pytest.raises((ValueError, Exception)):
            fw.rolling_min(np.array([1.0, 2.0]), window=0)

    def test_empty_array(self):
        x = np.array([], dtype=float)
        assert fw.rolling_mean(x, window=3).size == 0
        assert fw.rolling_min(x, window=3).size == 0

    def test_has_avx2_returns_bool(self):
        assert isinstance(fw.has_avx2(), bool)

#Numerical stability

class TestNumericalStability:
    def test_mean_stability_over_reinit(self):
        """Mean on 1M elements must match pandas within 1e-9."""
        rng = np.random.default_rng(0)
        x   = rng.standard_normal(1_000_000)
        out = fw.rolling_mean(x, window=252)
        exp = pd.Series(x).rolling(252).mean().to_numpy()
        finite = ~np.isnan(exp)
        max_err = np.max(np.abs(out[finite] - exp[finite]))
        assert max_err < 1e-9, f"max mean error {max_err} exceeds 1e-9"

    def test_std_stability_over_reinit(self):
        """Std on 1M elements must match pandas within 1e-7."""
        rng = np.random.default_rng(0)
        x   = rng.standard_normal(1_000_000)
        out = fw.rolling_std(x, window=252)
        exp = pd.Series(x).rolling(252).std().to_numpy()
        finite = ~np.isnan(exp)
        max_err = np.max(np.abs(out[finite] - exp[finite]))
        assert max_err < 1e-7, f"max std error {max_err} exceeds 1e-7"
