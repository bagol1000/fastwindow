"""
Tests for rolling Spearman rank correlation
Reference: scipy.stats.spearmanr per window.
"""

import numpy as np
import pytest
from scipy import stats

import fastwindow as fw

NaN = float("nan")


class TestSpearmanVsScipy:
    @pytest.mark.parametrize("w", [10, 25, 50])
    def test_random_data(self, w):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.standard_normal(n)
        y = 0.5 * x + rng.standard_normal(n)
        out = fw.rolling_spearman(x, y, window=w)
        for i in range(w - 1, n, 7):
            exp = stats.spearmanr(x[i - w + 1:i + 1], y[i - w + 1:i + 1]).statistic
            assert out[i] == pytest.approx(exp, abs=1e-10), f"w={w}, i={i}"
        assert np.all(np.isnan(out[:w - 1]))

    def test_with_ties(self):
        """Repeated values → average ranks, must still match scipy."""
        rng = np.random.default_rng(7)
        n, w = 200, 20
        x = np.round(rng.standard_normal(n), 1)   #heavy ties
        y = np.round(rng.standard_normal(n), 1)
        out = fw.rolling_spearman(x, y, window=w)
        for i in range(w - 1, n, 11):
            exp = stats.spearmanr(x[i - w + 1:i + 1], y[i - w + 1:i + 1]).statistic
            if np.isnan(exp):
                assert np.isnan(out[i]), f"i={i}"
            else:
                assert out[i] == pytest.approx(exp, abs=1e-10), f"i={i}"

    def test_min_periods(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal(60)
        y = rng.standard_normal(60)
        out = fw.rolling_spearman(x, y, window=30, min_periods=5)
        assert np.all(np.isnan(out[:4]))
        for i in (4, 10, 20):
            exp = stats.spearmanr(x[:i + 1], y[:i + 1]).statistic
            assert out[i] == pytest.approx(exp, abs=1e-10), f"i={i}"


class TestSpearmanProperties:
    def test_monotone_relation_gives_one(self):
        """y = exp(x) is monotone → Spearman exactly 1 (ranks identical)."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100)
        y = np.exp(x)
        out = fw.rolling_spearman(x, y, window=15)
        finite = ~np.isnan(out)
        np.testing.assert_allclose(out[finite], 1.0, atol=1e-12)

    def test_antitone_relation_gives_minus_one(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(100)
        y = -x ** 3
        out = fw.rolling_spearman(x, y, window=15)
        finite = ~np.isnan(out)
        np.testing.assert_allclose(out[finite], -1.0, atol=1e-12)

    def test_constant_series_nan(self):
        x = np.ones(50)
        y = np.arange(50, dtype=float)
        out = fw.rolling_spearman(x, y, window=10)
        assert np.all(np.isnan(out))

    def test_nan_propagation(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(60)
        y = rng.standard_normal(60)
        x[30] = NaN
        out = fw.rolling_spearman(x, y, window=10)
        for i in range(30, 40):
            assert np.isnan(out[i]), f"expected NaN at i={i}"
        assert not np.isnan(out[40])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            fw.rolling_spearman(np.arange(10.0), np.arange(5.0), window=3)
