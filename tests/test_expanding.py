"""
Tests for expanding-window variants
"""

import numpy as np
import pandas as pd
import pytest

import fastwindow as fw

NaN = float("nan")

class TestExpandingBasic:
    @pytest.fixture
    def x(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(1000)

    def test_mean_matches_cumsum(self, x):
        out = fw.expanding_mean(x)
        exp = np.cumsum(x) / np.arange(1, len(x) + 1)
        np.testing.assert_allclose(out, exp, rtol=1e-12)

    def test_sum_matches_cumsum(self, x):
        out = fw.expanding_sum(x)
        np.testing.assert_allclose(out, np.cumsum(x), rtol=1e-12)

    def test_std_vs_pandas(self, x):
        out = fw.expanding_std(x)
        exp = pd.Series(x).expanding(min_periods=2).std().to_numpy()
        nan_match = np.isnan(out) == np.isnan(exp)
        assert nan_match.all()
        m = ~np.isnan(exp)
        np.testing.assert_allclose(out[m], exp[m], rtol=1e-9)

    def test_var_vs_pandas(self, x):
        out = fw.expanding_var(x)
        exp = pd.Series(x).expanding(min_periods=2).var().to_numpy()
        m = ~np.isnan(exp)
        np.testing.assert_allclose(out[m], exp[m], rtol=1e-9)

    def test_var_ddof0(self, x):
        out = fw.expanding_var(x, ddof=0)
        for i in (10, 100, 999):
            assert out[i] == pytest.approx(np.var(x[:i + 1], ddof=0), rel=1e-9), f"i={i}"

    def test_min_periods(self, x):
        out = fw.expanding_mean(x, min_periods=10)
        assert np.all(np.isnan(out[:9]))
        assert not np.isnan(out[9])

    def test_nan_skipped(self):
        x = np.array([1.0, NaN, 3.0, 5.0])
        out = fw.expanding_mean(x)
        exp = np.array([1.0, 1.0, 2.0, 3.0])   #NaN excluded, count unchanged
        np.testing.assert_allclose(out, exp)


class TestExpandingRegression:
    def test_last_step_matches_lstsq(self):
        rng = np.random.default_rng(7)
        n = 500
        y = 2.0 + 0.3 * np.arange(n) + rng.standard_normal(n)
        out = fw.expanding_regression(y)
        t = np.arange(n, dtype=float)
        A = np.column_stack([np.ones(n), t])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        assert out["intercept"][-1] == pytest.approx(coef[0], rel=1e-9)
        assert out["slope"][-1] == pytest.approx(coef[1], rel=1e-9)

    def test_every_step_matches_lstsq(self):
        rng = np.random.default_rng(8)
        n = 100
        y = np.cumsum(rng.standard_normal(n))
        out = fw.expanding_regression(y)
        for i in range(2, n, 11):
            t = np.arange(i + 1, dtype=float)
            A = np.column_stack([np.ones(i + 1), t])
            coef, *_ = np.linalg.lstsq(A, y[:i + 1], rcond=None)
            assert out["intercept"][i] == pytest.approx(coef[0], rel=1e-8, abs=1e-8), f"i={i}"
            assert out["slope"][i] == pytest.approx(coef[1], rel=1e-8, abs=1e-8), f"i={i}"

    def test_perfect_line_r2(self):
        y = 1.0 + 2.0 * np.arange(50, dtype=float)
        out = fw.expanding_regression(y)
        assert out["slope"][-1] == pytest.approx(2.0, abs=1e-10)
        assert out["r2"][-1] == pytest.approx(1.0, abs=1e-12)

    def test_warmup_nan(self):
        y = np.arange(10, dtype=float)
        out = fw.expanding_regression(y)
        assert np.isnan(out["slope"][0])       #one point: undefined
        assert not np.isnan(out["slope"][1])   #two points: defined

    def test_time_as_x_false_raises(self):
        with pytest.raises(ValueError):
            fw.expanding_regression(np.arange(10.0), time_as_x=False)
