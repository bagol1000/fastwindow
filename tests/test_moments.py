"""
Tests for rolling_skew, rolling_kurt and rolling_zscore.

skew/kurt match pandas .rolling().skew()/.kurt() (scipy bias=False
conventions: kurt is excess kurtosis).  zscore is
(x - rolling_mean) / rolling_std with the same NaN gating.
"""

import numpy as np
import pandas as pd
import pytest
import fastwindow as fw

NaN = float("nan")


def assert_allclose_nan(actual, expected, rtol=1e-7, atol=1e-10, label=""):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{label}: shape mismatch"
    nan_a, nan_e = np.isnan(actual), np.isnan(expected)
    assert np.array_equal(nan_a, nan_e), f"{label}: NaN pattern differs"
    np.testing.assert_allclose(actual[~nan_e], expected[~nan_e],
                               rtol=rtol, atol=atol, err_msg=label)


class TestSkewKurtVsPandas:
    @pytest.mark.parametrize("window", [5, 30, 252])
    def test_skew_normal_data(self, window):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000)
        out = fw.rolling_skew(x, window=window)
        exp = pd.Series(x).rolling(window).skew().to_numpy()
        assert_allclose_nan(out, exp, label=f"skew w={window}")

    @pytest.mark.parametrize("window", [5, 30, 252])
    def test_kurt_normal_data(self, window):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(5000)
        out = fw.rolling_kurt(x, window=window)
        exp = pd.Series(x).rolling(window).kurt().to_numpy()
        assert_allclose_nan(out, exp, label=f"kurt w={window}")

    def test_skew_large_offset(self):
        """Conditioning: mean ≈ 1e6, fluctuations O(1)."""
        rng = np.random.default_rng(2)
        x = 1e6 + rng.standard_normal(3000)
        out = fw.rolling_skew(x, window=50)
        exp = pd.Series(x).rolling(50).skew().to_numpy()
        #shifted power sums must not lose the signal to cancellation
        finite = ~np.isnan(exp)
        assert np.max(np.abs(out[finite] - exp[finite])) < 1e-4

    def test_kurt_large_offset(self):
        rng = np.random.default_rng(3)
        x = 1e6 + rng.standard_normal(3000)
        out = fw.rolling_kurt(x, window=50)
        exp = pd.Series(x).rolling(50).kurt().to_numpy()
        finite = ~np.isnan(exp)
        assert np.max(np.abs(out[finite] - exp[finite])) < 1e-3

    def test_skew_min_window(self):
        #skew needs >= 3 observations, kurt >= 4
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        assert np.isnan(fw.rolling_skew(x, window=3)[:2]).all()
        assert not np.isnan(fw.rolling_skew(x, window=3)[2:]).any()
        assert np.isnan(fw.rolling_kurt(x, window=4)[:3]).all()
        assert not np.isnan(fw.rolling_kurt(x, window=4)[3:]).any()

    def test_constant_window_nan(self):
        x = np.full(20, 3.14)
        assert np.isnan(fw.rolling_skew(x, window=5)[4:]).all()
        assert np.isnan(fw.rolling_kurt(x, window=6)[5:]).all()

    def test_nan_propagation_default(self):
        x = np.array([1.0, 2.0, 3.0, NaN, 5.0, 6.0, 7.0, 8.0, 9.0])
        out = fw.rolling_skew(x, window=4)
        #windows containing index 3 are NaN
        assert np.isnan(out[:7]).all()
        assert not np.isnan(out[7:]).any()

    def test_skip_nan_vs_pandas(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(500)
        x[rng.choice(500, 30, replace=False)] = np.nan
        out = fw.rolling_skew(x, window=20, min_periods=5, skip_nan=True)
        exp = pd.Series(x).rolling(20, min_periods=5).skew().to_numpy()
        assert_allclose_nan(out, exp, label="skew skip_nan")

    def test_kurt_skip_nan_vs_pandas(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(500)
        x[rng.choice(500, 30, replace=False)] = np.nan
        out = fw.rolling_kurt(x, window=20, min_periods=6, skip_nan=True)
        exp = pd.Series(x).rolling(20, min_periods=6).kurt().to_numpy()
        assert_allclose_nan(out, exp, label="kurt skip_nan")

    def test_min_periods(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(100)
        out = fw.rolling_skew(x, window=10, min_periods=4)
        exp = pd.Series(x).rolling(10, min_periods=4).skew().to_numpy()
        assert_allclose_nan(out, exp, label="skew min_periods")

    def test_long_series_reinit_stability(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(20_000)
        out = fw.rolling_kurt(x, window=100)
        exp = pd.Series(x).rolling(100).kurt().to_numpy()
        assert_allclose_nan(out, exp, rtol=1e-6, label="kurt 20k reinit")


class TestZscore:
    def test_matches_mean_std_composition(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal(1000)
        out = fw.rolling_zscore(x, window=50)
        mu = fw.rolling_mean(x, window=50)
        sd = fw.rolling_std(x, window=50)
        exp = (x - mu) / sd
        assert_allclose_nan(out, exp, label="zscore composition")

    def test_constant_window_nan(self):
        x = np.full(20, 2.0)
        out = fw.rolling_zscore(x, window=5)
        assert np.isnan(out).all()   #std == 0 -> NaN, no inf

    def test_nan_input_position_is_nan(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, NaN, 6.0, 7.0, 8.0, 9.0, 10.0])
        out = fw.rolling_zscore(x, window=3, skip_nan=True, min_periods=2)
        assert np.isnan(out[4])      #NaN input -> NaN z-score

    def test_ddof0(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal(300)
        out = fw.rolling_zscore(x, window=20, ddof=0)
        sd = fw.rolling_std(x, window=20, ddof=0)
        mu = fw.rolling_mean(x, window=20)
        assert_allclose_nan(out, (x - mu) / sd, label="zscore ddof0")

    def test_out_param(self):
        x = np.random.default_rng(10).standard_normal(100)
        buf = np.empty(100)
        res = fw.rolling_zscore(x, window=10, out=buf)
        assert res is not None and np.shares_memory(res, buf)
