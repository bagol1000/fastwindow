"""
Tests for rolling Pearson correlation and covariance
"""

import numpy as np
import pandas as pd
import pytest

import fastwindow as fw

NaN = float("nan")

def assert_allclose_nan(actual, expected, rtol=1e-9, atol=1e-12, label=""):
    actual   = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{label}: shape mismatch"
    nan_a, nan_e = np.isnan(actual), np.isnan(expected)
    assert np.array_equal(nan_a, nan_e), f"{label}: NaN pattern differs"
    finite = ~nan_e
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=atol, err_msg=f"{label}: values differ")


class TestVsPandas:
    @pytest.fixture
    def xy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2000)
        y = 0.7 * x + rng.standard_normal(2000)
        return x, y

    def test_corr_vs_pandas(self, xy):
        x, y = xy
        out = fw.rolling_corr(x, y, window=50)
        exp = pd.Series(x).rolling(50).corr(pd.Series(y)).to_numpy()
        assert_allclose_nan(out, exp, rtol=1e-9, label="corr vs pandas")

    def test_cov_vs_pandas(self, xy):
        x, y = xy
        out = fw.rolling_cov(x, y, window=50)
        exp = pd.Series(x).rolling(50).cov(pd.Series(y)).to_numpy()
        assert_allclose_nan(out, exp, rtol=1e-9, label="cov vs pandas")

    def test_return_cov_tuple(self, xy):
        #The dual-output and single-output paths compile to differently
        #scheduled FMA sequences under -ffast-math → up to 1 ULP apart.
        x, y = xy
        corr, cov = fw.rolling_corr(x, y, window=50, return_cov=True)
        assert_allclose_nan(corr, fw.rolling_corr(x, y, window=50), rtol=1e-14, label="tuple corr vs standalone")
        assert_allclose_nan(cov, fw.rolling_cov(x, y, window=50), rtol=1e-14, label="tuple cov vs standalone")

    def test_min_periods_vs_pandas(self, xy):
        x, y = xy
        out = fw.rolling_corr(x, y, window=50, min_periods=10)
        exp = pd.Series(x).rolling(50, min_periods=10).corr(pd.Series(y)).to_numpy()
        assert_allclose_nan(out, exp, rtol=1e-9, label="corr min_periods vs pandas")

    def test_corr_large_n_reinit(self):
        """Accuracy across the 4096-step reinit boundary on 100k points."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(100_000)
        y = -0.3 * x + rng.standard_normal(100_000)
        out = fw.rolling_corr(x, y, window=252)
        exp = pd.Series(x).rolling(252).corr(pd.Series(y)).to_numpy()
        assert_allclose_nan(out, exp, rtol=1e-9, atol=1e-11, label="corr large n")


class TestPerfectCorrelation:
    def test_perfectly_correlated(self):
        """y = 2x + 1 → corr = 1.0."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(200)
        y = 2.0 * x + 1.0
        out = fw.rolling_corr(x, y, window=20)
        finite = ~np.isnan(out)
        assert finite.sum() == 181
        np.testing.assert_allclose(out[finite], 1.0, atol=1e-12)

    def test_perfectly_anticorrelated(self):
        """y = -3x + 5 → corr = -1.0."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(200)
        y = -3.0 * x + 5.0
        out = fw.rolling_corr(x, y, window=20)
        finite = ~np.isnan(out)
        np.testing.assert_allclose(out[finite], -1.0, atol=1e-12)

    def test_clamped_to_unit_interval(self):
        """No output may ever exceed [-1, 1]."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(5000)
        y = x + 1e-9 * rng.standard_normal(5000)   #near-perfect correlation
        out = fw.rolling_corr(x, y, window=30)
        finite = ~np.isnan(out)
        assert np.all(out[finite] <= 1.0)
        assert np.all(out[finite] >= -1.0)


class TestDegenerateWindows:
    def test_constant_x_gives_nan(self):
        x = np.ones(50)
        y = np.arange(50, dtype=float)
        out = fw.rolling_corr(x, y, window=10)
        assert np.all(np.isnan(out))

    def test_constant_y_gives_nan(self):
        x = np.arange(50, dtype=float)
        y = np.full(50, 3.14)
        out = fw.rolling_corr(x, y, window=10)
        assert np.all(np.isnan(out))

    def test_locally_constant_window(self):
        """x constant only inside some windows → NaN exactly there."""
        x = np.concatenate([np.ones(10), np.arange(10, dtype=float)])
        rng = np.random.default_rng(4)
        y = rng.standard_normal(20)
        out = fw.rolling_corr(x, y, window=5)
        assert np.all(np.isnan(out[:9]))      #windows fully inside the 1s
        assert not np.isnan(out[-1])          #ramp part is fine

    def test_constant_cov_is_zero_not_nan(self):
        """cov with one constant series is 0, not NaN (variance not needed)."""
        x = np.ones(30)
        y = np.arange(30, dtype=float)
        out = fw.rolling_cov(x, y, window=10)
        finite = ~np.isnan(out)
        np.testing.assert_allclose(out[finite], 0.0, atol=1e-12)


class TestNanHandling:
    def test_propagate_default(self):
        x = np.array([1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
        out = fw.rolling_corr(x, y, window=3)
        for i in (2, 3, 4):
            assert np.isnan(out[i]), f"expected NaN at i={i}"
        assert out[5] == pytest.approx(1.0, abs=1e-12)   #window [4,5,6] clean
        assert out[6] == pytest.approx(1.0, abs=1e-12)

    def test_nan_in_y_propagates(self):
        x = np.arange(7, dtype=float)
        y = np.array([0.0, 1.0, 2.0, NaN, 4.0, 5.0, 6.0])
        out = fw.rolling_corr(x, y, window=3)
        for i in (3, 4, 5):
            assert np.isnan(out[i])
        assert out[2] == pytest.approx(1.0, abs=1e-12)

    def test_skip_nan_vs_pandas(self):
        """pandas rolling corr uses pairwise-complete obs == our skip_nan."""
        rng = np.random.default_rng(8)
        x = rng.standard_normal(500)
        y = 0.5 * x + rng.standard_normal(500)
        x[rng.random(500) < 0.08] = NaN
        y[rng.random(500) < 0.08] = NaN
        out = fw.rolling_corr(x, y, window=30, min_periods=5, skip_nan=True)
        exp = (pd.Series(x).rolling(30, min_periods=5).corr(pd.Series(y)).to_numpy())
        assert_allclose_nan(out, exp, rtol=1e-8, atol=1e-10, label="skip_nan corr vs pandas")

    def test_skip_nan_cov_vs_pandas(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)
        x[rng.random(500) < 0.08] = NaN
        out = fw.rolling_cov(x, y, window=30, min_periods=5, skip_nan=True)
        exp = (pd.Series(x).rolling(30, min_periods=5).cov(pd.Series(y)).to_numpy())
        assert_allclose_nan(out, exp, rtol=1e-8, atol=1e-10, label="skip_nan cov vs pandas")

    def test_all_nan(self):
        x = np.full(10, NaN)
        y = np.arange(10, dtype=float)
        assert np.all(np.isnan(fw.rolling_corr(x, y, window=3)))
        assert np.all(np.isnan(fw.rolling_cov(x, y, window=3)))


class TestDdof:
    def test_ddof0_vs_ddof1(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        w = 20
        c1 = fw.rolling_cov(x, y, window=w, ddof=1)
        c0 = fw.rolling_cov(x, y, window=w, ddof=0)
        finite = ~np.isnan(c1)
        #population = sample × (n-1)/n
        np.testing.assert_allclose(c0[finite], c1[finite] * (w - 1) / w,
                                   rtol=1e-12)

    def test_ddof0_vs_numpy(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(60)
        y = rng.standard_normal(60)
        w = 15
        out = fw.rolling_cov(x, y, window=w, ddof=0)
        for i in range(w - 1, 60, 7):
            xw, yw = x[i - w + 1:i + 1], y[i - w + 1:i + 1]
            exp = np.cov(xw, yw, ddof=0)[0, 1]
            assert out[i] == pytest.approx(exp, rel=1e-10), f"i={i}"

    def test_invalid_ddof_raises(self):
        with pytest.raises(ValueError):
            fw.rolling_cov(np.arange(10.0), np.arange(10.0), window=3, ddof=2)


class TestErrors:
    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            fw.rolling_corr(np.arange(10.0), np.arange(5.0), window=3)
        with pytest.raises(ValueError):
            fw.rolling_cov(np.arange(10.0), np.arange(5.0), window=3)

    def test_window_zero(self):
        with pytest.raises(ValueError):
            fw.rolling_corr(np.arange(10.0), np.arange(10.0), window=0)
