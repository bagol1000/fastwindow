"""
Tests for multi-column _2d dispatch
"""

import numpy as np
import pytest
import fastwindow as fw

NaN = float("nan")

@pytest.fixture
def X():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 7))
    X[100, 2] = NaN  #one NaN
    return X

def nan_equal(a, b):
    return np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99))

PAIRS = [
    ("rolling_mean_2d", "rolling_mean", {}),
    ("rolling_std_2d",  "rolling_std",  {}),
    ("rolling_sum_2d",  "rolling_sum",  {}),
    ("rolling_min_2d",  "rolling_min",  None),   #None: no min_periods kwarg
    ("rolling_max_2d",  "rolling_max",  None),
]

class TestColumnwiseEquivalence:
    @pytest.mark.parametrize("fn2d,fn1d,kw", PAIRS)
    def test_columns_match_1d(self, X, fn2d, fn1d, kw):
        f2, f1 = getattr(fw, fn2d), getattr(fw, fn1d)
        out = f2(X, window=30) if kw is None else f2(X, window=30, **kw)
        assert out.shape == X.shape
        for j in range(X.shape[1]):
            ref = f1(X[:, j], window=30)
            assert nan_equal(out[:, j], ref), f"{fn2d} column {j}"

    def test_std_ddof0(self, X):
        out = fw.rolling_std_2d(X, window=30, ddof=0)
        for j in range(X.shape[1]):
            ref = fw.rolling_std(X[:, j], window=30, ddof=0)
            assert nan_equal(out[:, j], ref), f"column {j}"

    def test_min_periods(self, X):
        out = fw.rolling_mean_2d(X, window=30, min_periods=5)
        for j in range(X.shape[1]):
            ref = fw.rolling_mean(X[:, j], window=30, min_periods=5)
            assert nan_equal(out[:, j], ref), f"column {j}"


class TestThreads:
    @pytest.mark.parametrize("fn2d,_,kw", PAIRS)
    def test_threads_identical(self, X, fn2d, _, kw):
        f2 = getattr(fw, fn2d)
        out1 = f2(X, window=30, n_threads=1)
        out4 = f2(X, window=30, n_threads=4)
        assert nan_equal(out1, out4), fn2d


class TestInputs:
    def test_row_major_accepted(self, X):
        Xc = np.ascontiguousarray(X)
        out_c = fw.rolling_mean_2d(Xc, window=30)
        out_f = fw.rolling_mean_2d(np.asfortranarray(X), window=30)
        assert nan_equal(out_c, out_f)

    def test_1d_rejected(self):
        with pytest.raises(ValueError):
            fw.rolling_mean_2d(np.arange(10.0), window=3)

    def test_window_zero(self, X):
        with pytest.raises(ValueError):
            fw.rolling_mean_2d(X, window=0)


class TestValidation:
    def test_negative_n_threads_rejected(self, X):
        with pytest.raises(ValueError, match="n_threads"):
            fw.rolling_mean_2d(X, window=30, n_threads=-1)
