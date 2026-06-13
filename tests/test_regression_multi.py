"""
Tests for rolling multiple linear regression

Reference: per-window numpy.linalg.lstsq with an explicit intercept column.
"""

import numpy as np
import pytest

import fastwindow as fw

NaN = float("nan")

def reference_multi(y_win, X_win):
    """OLS with intercept via lstsq; returns (coef[k+1], r2, residual_std)."""
    m, k = X_win.shape
    A = np.column_stack([np.ones(m), X_win])
    coef, *_ = np.linalg.lstsq(A, y_win, rcond=None)
    resid = y_win - A @ coef
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_win - y_win.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else NaN
    res_std = np.sqrt(ss_res / (m - (k + 1))) if m > k + 1 else NaN
    return coef, r2, res_std


class TestMultipleRegressionCore:
    def test_exact_linear_model(self):
        """Noise-free y = 1 + 2x1 - 3x2 → exact recovery everywhere."""
        rng = np.random.default_rng(0)
        n, w = 200, 30
        X = rng.standard_normal((n, 2))
        y = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
        out = fw.rolling_multiple_regression(y, X, window=w)
        for i in range(w - 1, n):
            assert out["coef"][i, 0] == pytest.approx(1.0,  rel=1e-8, abs=1e-8)
            assert out["coef"][i, 1] == pytest.approx(2.0,  rel=1e-8, abs=1e-8)
            assert out["coef"][i, 2] == pytest.approx(-3.0, rel=1e-8, abs=1e-8)
            assert out["r2"][i] == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.isnan(out["coef"][:w - 1, 0]))

    def test_vs_lstsq_k3(self):
        rng = np.random.default_rng(42)
        n, w, k = 400, 50, 3
        X = rng.standard_normal((n, k))
        y = X @ np.array([0.5, -1.0, 2.0]) + rng.standard_normal(n)
        out = fw.rolling_multiple_regression(y, X, window=w)
        for i in range(w - 1, n, 7):
            coef, r2, res_std = reference_multi(y[i - w + 1:i + 1], X[i - w + 1:i + 1])
            np.testing.assert_allclose(out["coef"][i], coef, rtol=1e-7, atol=1e-8, err_msg=f"coef mismatch at i={i}")
            assert out["r2"][i] == pytest.approx(r2, rel=1e-6, abs=1e-8), f"i={i}"
            assert out["residual_std"][i] == pytest.approx(res_std, rel=1e-6), f"i={i}"

    def test_vs_lstsq_k5_generic_path(self):
        """k=5 exercises the generic (non-unrolled) Sherman-Morrison path."""
        rng = np.random.default_rng(43)
        n, w, k = 300, 40, 5
        X = rng.standard_normal((n, k))
        y = X @ rng.standard_normal(k) + rng.standard_normal(n)
        out = fw.rolling_multiple_regression(y, X, window=w)
        for i in range(w - 1, n, 11):
            coef, r2, _ = reference_multi(y[i - w + 1:i + 1], X[i - w + 1:i + 1])
            np.testing.assert_allclose(out["coef"][i], coef, rtol=1e-6, atol=1e-7, err_msg=f"coef mismatch at i={i}")

    def test_k1_matches_simple_regression_xy(self):
        rng = np.random.default_rng(7)
        n, w = 500, 25
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n) * 0.3
        multi  = fw.rolling_multiple_regression(y, x.reshape(-1, 1), window=w)
        simple = fw.rolling_regression_xy(y, x, window=w)
        finite = ~np.isnan(simple["slope"])
        np.testing.assert_allclose(multi["coef"][finite, 0], simple["intercept"][finite], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(multi["coef"][finite, 1], simple["slope"][finite], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(multi["r2"][finite], simple["r2"][finite], rtol=1e-8, atol=1e-10)

    def test_reinit_stability(self):
        """Accuracy holds across the 4096-step recompute boundary."""
        rng = np.random.default_rng(3)
        n, w, k = 10_000, 60, 3
        X = rng.standard_normal((n, k))
        y = X @ np.array([1.0, -0.5, 0.25]) + rng.standard_normal(n)
        out = fw.rolling_multiple_regression(y, X, window=w)
        for i in [59, 4095, 4096, 4097, 8191, 9999]:
            coef, _, _ = reference_multi(y[i - w + 1:i + 1], X[i - w + 1:i + 1])
            np.testing.assert_allclose(out["coef"][i], coef, rtol=1e-6, atol=1e-7, err_msg=f"coef mismatch at i={i}")


class TestMultipleRegressionEdgeCases:
    def test_near_collinear_no_crash(self):
        """Condition number ~1e6: must not crash; outputs finite or NaN."""
        rng = np.random.default_rng(99)
        n, w = 500, 50
        x1 = rng.standard_normal(n)
        x2 = x1 + 1e-6 * rng.standard_normal(n)   #nearly identical column
        X = np.column_stack([x1, x2])
        y = x1 + rng.standard_normal(n)
        out = fw.rolling_multiple_regression(y, X, window=w)   #must not raise
        assert out["coef"].shape == (n, 3)
        #Where coefficients are produced the fit must still be sane:
        #predicted values track y reasonably (residual_std bounded).
        finite = ~np.isnan(out["residual_std"])
        assert np.all(out["residual_std"][finite] < 10.0)

    def test_exactly_collinear_gives_nan(self):
        """A constant regressor duplicates the intercept → singular → NaN."""
        n, w = 100, 20
        X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        y = np.arange(n, dtype=float)
        out = fw.rolling_multiple_regression(y, X, window=w)
        assert np.all(np.isnan(out["coef"][:, 0]))

    def test_k16_works(self):
        rng = np.random.default_rng(16)
        n, w, k = 600, 100, 16
        X = rng.standard_normal((n, k))
        y = X @ rng.standard_normal(k) + rng.standard_normal(n)
        out = fw.rolling_multiple_regression(y, X, window=w)
        for i in [w - 1, 300, 599]:
            coef, _, _ = reference_multi(y[i - w + 1:i + 1], X[i - w + 1:i + 1])
            np.testing.assert_allclose(out["coef"][i], coef, rtol=1e-4, atol=1e-5, err_msg=f"k=16 coef mismatch at i={i}")

    def test_k17_raises(self):
        rng = np.random.default_rng(17)
        X = rng.standard_normal((100, 17))
        y = rng.standard_normal(100)
        with pytest.raises(ValueError):
            fw.rolling_multiple_regression(y, X, window=30)

    def test_nan_propagation(self):
        rng = np.random.default_rng(8)
        n, w = 60, 10
        X = rng.standard_normal((n, 2))
        y = X @ np.array([1.0, 1.0]) + rng.standard_normal(n) * 0.1
        y[25] = NaN
        X[40, 1] = NaN
        out = fw.rolling_multiple_regression(y, X, window=w)
        #Windows containing a NaN row → NaN output
        for i in range(25, 25 + w):
            assert np.isnan(out["coef"][i, 0]), f"expected NaN at i={i}"
        for i in range(40, 40 + w):
            assert np.isnan(out["coef"][i, 0]), f"expected NaN at i={i}"
        #Outputs resume once the NaN row exits
        assert not np.isnan(out["coef"][25 + w, 0])
        assert not np.isnan(out["coef"][min(40 + w, n - 1), 0]) or 40 + w >= n

    def test_min_periods_warmup(self):
        rng = np.random.default_rng(9)
        n, w, k = 50, 20, 2
        X = rng.standard_normal((n, k))
        y = X @ np.array([1.0, -1.0]) + 0.5
        out = fw.rolling_multiple_regression(y, X, window=w, min_periods=5)
        #First possible output: m >= max(min_periods, k+1) = 5 → index 4
        assert np.all(np.isnan(out["coef"][:4, 0]))
        assert not np.isnan(out["coef"][4, 0])
        coef, _, _ = reference_multi(y[:5], X[:5])
        np.testing.assert_allclose(out["coef"][4], coef, rtol=1e-7, atol=1e-8)

    def test_x_dimension_validation(self):
        y = np.arange(10.0)
        with pytest.raises(ValueError):
            fw.rolling_multiple_regression(y, np.arange(10.0), window=3)   #1-D X
        with pytest.raises(ValueError):
            fw.rolling_multiple_regression(y, np.zeros((5, 2)), window=3)  #row mismatch
