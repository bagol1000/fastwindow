"""
Tests for rolling simple linear regression

Reference: numpy.polyfit / numpy.linalg.lstsq applied per window.
"""

import numpy as np
import pytest

import fastwindow as fw

NaN = float("nan")

def reference_regression(y_win, x_win):
    """OLS via lstsq; returns (intercept, slope, r2)."""
    A = np.column_stack([np.ones_like(x_win), x_win])
    coef, *_ = np.linalg.lstsq(A, y_win, rcond=None)
    b0, b1 = coef
    resid = y_win - (b0 + b1 * x_win)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_win - y_win.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else NaN
    return b0, b1, r2


class TestRollingRegressionTimeAsX:
    def test_perfect_line(self):
        """y = 2 + 3t globally → every window recovers slope 3."""
        n = 50
        y = 2.0 + 3.0 * np.arange(n, dtype=float)
        out = fw.rolling_regression(y, window=10)
        #Window ending at i covers t = i-9..i; local x = 0..9, so the local
        #intercept equals the value at the window start: y[i-9].
        for i in range(9, n):
            assert out["slope"][i] == pytest.approx(3.0, abs=1e-9)
            assert out["intercept"][i] == pytest.approx(y[i - 9], abs=1e-8)
            assert out["r2"][i] == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.isnan(out["slope"][:9]))

    def test_vs_lstsq_random(self):
        rng = np.random.default_rng(123)
        n, w = 300, 20
        y = np.cumsum(rng.standard_normal(n))
        out = fw.rolling_regression(y, window=w)
        x_win = np.arange(w, dtype=float)
        for i in range(w - 1, n):
            b0, b1, r2 = reference_regression(y[i - w + 1:i + 1], x_win)
            assert out["intercept"][i] == pytest.approx(b0, rel=1e-8, abs=1e-8), f"i={i}"
            assert out["slope"][i]     == pytest.approx(b1, rel=1e-8, abs=1e-8), f"i={i}"
            assert out["r2"][i]        == pytest.approx(r2, rel=1e-7, abs=1e-8), f"i={i}"

    def test_min_periods_warmup(self):
        """With min_periods=2, partial windows regress on x = 0..m-1."""
        y = np.array([1.0, 3.0, 5.0, 7.0])
        out = fw.rolling_regression(y, window=4, min_periods=2)
        #m=2: points (0,1),(1,3) → slope 2, intercept 1
        assert np.isnan(out["slope"][0])
        assert out["slope"][1] == pytest.approx(2.0)
        assert out["intercept"][1] == pytest.approx(1.0)
        assert out["slope"][3] == pytest.approx(2.0)
        assert out["r2"][3] == pytest.approx(1.0)

    def test_constant_y_r2_nan(self):
        """Constant y: slope 0, R² undefined (NaN)."""
        y = np.ones(30) * 4.0
        out = fw.rolling_regression(y, window=10)
        assert out["slope"][15] == pytest.approx(0.0, abs=1e-12)
        assert out["intercept"][15] == pytest.approx(4.0, abs=1e-12)
        assert np.isnan(out["r2"][15])

    def test_nan_propagation(self):
        y = np.array([1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0])
        out = fw.rolling_regression(y, window=3)
        #windows containing index 2 → NaN
        for i in (2, 3, 4):
            assert np.isnan(out["slope"][i])
        assert out["slope"][5] == pytest.approx(1.0, abs=1e-10)
        assert out["slope"][6] == pytest.approx(1.0, abs=1e-10)

    def test_reinit_stability(self):
        """Beyond the 4096-step reinit boundary, results stay accurate."""
        rng = np.random.default_rng(5)
        n, w = 10_000, 50
        y = np.cumsum(rng.standard_normal(n))
        out = fw.rolling_regression(y, window=w)
        x_win = np.arange(w, dtype=float)
        #check a sample of windows on both sides of the 4096 boundary
        for i in [49, 1000, 4095, 4096, 4097, 8191, 9999]:
            b0, b1, r2 = reference_regression(y[i - w + 1:i + 1], x_win)
            assert out["intercept"][i] == pytest.approx(b0, rel=1e-7, abs=1e-7), f"i={i}"
            assert out["slope"][i]     == pytest.approx(b1, rel=1e-7, abs=1e-7), f"i={i}"

    def test_time_as_x_false_raises(self):
        with pytest.raises(ValueError):
            fw.rolling_regression(np.arange(10.0), window=3, time_as_x=False)


class TestRollingRegressionXY:
    def test_perfect_linear_relation(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(100)
        y = 1.5 + 2.5 * x
        out = fw.rolling_regression_xy(y, x, window=10)
        for i in range(9, 100):
            assert out["slope"][i] == pytest.approx(2.5, rel=1e-9)
            assert out["intercept"][i] == pytest.approx(1.5, rel=1e-9)
            assert out["r2"][i] == pytest.approx(1.0, abs=1e-9)

    def test_vs_lstsq_random(self):
        rng = np.random.default_rng(11)
        n, w = 300, 25
        x = rng.standard_normal(n)
        y = 0.5 * x + rng.standard_normal(n)
        out = fw.rolling_regression_xy(y, x, window=w)
        for i in range(w - 1, n):
            b0, b1, r2 = reference_regression(y[i - w + 1:i + 1], x[i - w + 1:i + 1])
            assert out["intercept"][i] == pytest.approx(b0, rel=1e-7, abs=1e-8), f"i={i}"
            assert out["slope"][i]     == pytest.approx(b1, rel=1e-7, abs=1e-8), f"i={i}"
            assert out["r2"][i]        == pytest.approx(r2, rel=1e-6, abs=1e-8), f"i={i}"

    def test_nan_in_either_series(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([1.0, NaN, 3.0, 4.0, 5.0, 6.0])
        out = fw.rolling_regression_xy(y, x, window=3)
        assert np.isnan(out["slope"][2])   #window contains y[1]=NaN
        assert np.isnan(out["slope"][3])
        assert out["slope"][5] == pytest.approx(1.0, rel=1e-9)

        x2 = np.array([1.0, 2.0, NaN, 4.0, 5.0, 6.0])
        y2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out2 = fw.rolling_regression_xy(y2, x2, window=3)
        assert np.isnan(out2["slope"][3])  #window contains x[2]=NaN
        assert out2["slope"][5] == pytest.approx(1.0, rel=1e-9)

    def test_constant_x_degenerate(self):
        """Constant x → singular design → NaN coefficients."""
        x = np.ones(20)
        y = np.arange(20, dtype=float)
        out = fw.rolling_regression_xy(y, x, window=5)
        assert np.all(np.isnan(out["slope"][4:]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            fw.rolling_regression_xy(np.arange(10.0), np.arange(5.0), window=3)

    def test_reinit_stability(self):
        rng = np.random.default_rng(13)
        n, w = 10_000, 40
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n) * 0.5
        out = fw.rolling_regression_xy(y, x, window=w)
        for i in [4095, 4096, 4097, 9999]:
            b0, b1, r2 = reference_regression(y[i - w + 1:i + 1], x[i - w + 1:i + 1])
            assert out["intercept"][i] == pytest.approx(b0, rel=1e-7, abs=1e-7), f"i={i}"
            assert out["slope"][i]     == pytest.approx(b1, rel=1e-7, abs=1e-7), f"i={i}"
