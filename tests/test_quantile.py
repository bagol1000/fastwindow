"""
Tests for rolling_quantile
"""

import numpy as np
import pytest
import fastwindow as fw

NaN = float("nan")

class TestExactQuantile:
    @pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_exact_vs_numpy_percentile(self, q):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(300)
        w = 25
        out = fw.rolling_quantile(x, window=w, q=q, exact=True)
        for i in range(w - 1, 300, 7):
            exp = np.percentile(x[i - w + 1:i + 1], q * 100)
            assert out[i] == pytest.approx(exp, abs=1e-10), f"i={i}, q={q}"
        assert np.all(np.isnan(out[:w - 1]))

    def test_exact_min_periods(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(50)
        out = fw.rolling_quantile(x, window=20, q=0.5, min_periods=5, exact=True)
        assert np.all(np.isnan(out[:4]))
        for i in range(4, 19):
            exp = np.percentile(x[:i + 1], 50)
            assert out[i] == pytest.approx(exp, abs=1e-10), f"i={i}"

    def test_exact_median_odd_window(self):
        x = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        out = fw.rolling_quantile(x, window=5, q=0.5, exact=True)
        assert out[4] == pytest.approx(3.0)

    def test_exact_nan_propagation(self):
        x = np.array([1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0])
        out = fw.rolling_quantile(x, window=3, q=0.5, exact=True)
        for i in (2, 3, 4):
            assert np.isnan(out[i])
        assert out[5] == pytest.approx(5.0)
        assert out[6] == pytest.approx(6.0)


class TestP2Quantile:
    @pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
    def test_p2_vs_exact_within_5pct(self, q):
        """P² is approximate: 5% relative tolerance on stationary data."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(5000) + 10.0   #shift away from 0 so
        w = 200                                 #relative error is meaningful
        approx = fw.rolling_quantile(x, window=w, q=q, exact=False)
        exact  = fw.rolling_quantile(x, window=w, q=q, exact=True)
        #Compare in the steady-state region (P² needs a burn-in)
        sl = slice(1000, 5000)
        rel = np.abs(approx[sl] - exact[sl]) / np.abs(exact[sl])
        assert np.nanmax(rel) < 0.05, \
            f"q={q}: max relative error {np.nanmax(rel):.4f}"

    def test_p2_uniform_median(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 1, 20_000)
        out = fw.rolling_quantile(x, window=500, q=0.5, exact=False)
        #Median of U(0,1) is 0.5; P² should land close after burn-in
        assert abs(out[-1] - 0.5) < 0.02

    def test_p2_warmup_small_counts(self):
        """Fewer than 5 observations: exact interpolated quantile."""
        x = np.array([3.0, 1.0, 2.0])
        out = fw.rolling_quantile(x, window=10, q=0.5, min_periods=1, exact=False)
        assert out[0] == pytest.approx(3.0)
        assert out[1] == pytest.approx(2.0)   #median of {1,3}
        assert out[2] == pytest.approx(2.0)   #median of {1,2,3}

    def test_p2_nan_propagation(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(100)
        x[50] = NaN
        out = fw.rolling_quantile(x, window=10, q=0.5, exact=False)
        for i in range(50, 60):
            assert np.isnan(out[i]), f"expected NaN at i={i}"
        assert not np.isnan(out[60])


class TestQuantileErrors:
    def test_q_zero_raises(self):
        with pytest.raises(ValueError):
            fw.rolling_quantile(np.arange(10.0), window=3, q=0.0)

    def test_q_one_raises(self):
        with pytest.raises(ValueError):
            fw.rolling_quantile(np.arange(10.0), window=3, q=1.0)

    def test_exact_large_window_works(self):
        """window > 500 was rejected by the old sorted-buffer mode."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal(3000)
        out = fw.rolling_quantile(x, window=1000, q=0.5, exact=True)
        for i in (999, 1500, 2999):
            exp = np.percentile(x[i - 999:i + 1], 50)
            assert out[i] == pytest.approx(exp, abs=1e-10), f"i={i}"

    def test_exact_monotone_input(self):
        """Monotone data strands lazy-deleted entries → exercises rebuild."""
        x = np.arange(20000.0)
        out = fw.rolling_quantile(x, window=100, q=0.25, exact=True)
        for i in (99, 9999, 19999):
            exp = np.percentile(x[i - 99:i + 1], 25)
            assert out[i] == pytest.approx(exp, abs=1e-10), f"i={i}"

    def test_exact_random_nan_torture(self):
        """Random NaN bursts stress the lazy-deletion bookkeeping."""
        rng = np.random.default_rng(11)
        x = rng.standard_normal(4000)
        x[rng.random(4000) < 0.07] = NaN
        w = 64
        out = fw.rolling_quantile(x, window=w, q=0.4, exact=True)
        for i in range(w - 1, 4000, 37):
            win = x[i - w + 1:i + 1]
            if np.isnan(win).any():
                assert np.isnan(out[i]), f"i={i}"
            else:
                exp = np.percentile(win, 40)
                assert out[i] == pytest.approx(exp, abs=1e-10), f"i={i}"
