"""
Tests for the out= buffer-reuse parameter and the n_threads parameter of the 1-D kernels.
"""

import numpy as np
import pytest
import fastwindow as fw

@pytest.fixture
def x():
    rng = np.random.default_rng(42)
    v = rng.standard_normal(5000)
    v[1234] = np.nan
    return v


def nan_equal(a, b):
    return np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99))

OUT_FUNCS = [
    lambda x, **kw: fw.rolling_mean(x, 100, **kw),
    lambda x, **kw: fw.rolling_std(x, 100, **kw),
    lambda x, **kw: fw.rolling_var(x, 100, **kw),
    lambda x, **kw: fw.rolling_sum(x, 100, **kw),
    lambda x, **kw: fw.rolling_min(x, 100, **kw),
    lambda x, **kw: fw.rolling_max(x, 100, **kw),
    lambda x, **kw: fw.rolling_quantile(x, 100, q=0.3, **kw),
]

class TestOutParam:
    @pytest.mark.parametrize("fn", OUT_FUNCS)
    def test_identical_to_allocating_version(self, x, fn):
        buf = np.empty_like(x)
        ref = fn(x)
        res = fn(x, out=buf)
        assert res is buf
        assert nan_equal(res, ref)

    def test_two_series_functions(self, x):
        rng = np.random.default_rng(7)
        y = rng.standard_normal(len(x))
        buf = np.empty_like(x)
        assert nan_equal(fw.rolling_corr(x, y, 50, out=buf), fw.rolling_corr(x, y, 50))
        assert nan_equal(fw.rolling_cov(x, y, 50, out=buf), fw.rolling_cov(x, y, 50))
        assert nan_equal(fw.rolling_spearman(x[:500], y[:500], 30, out=buf[:500]), fw.rolling_spearman(x[:500], y[:500], 30))

    def test_wrong_length_raises(self, x):
        with pytest.raises(ValueError):
            fw.rolling_mean(x, 100, out=np.empty(len(x) - 1))

    def test_wrong_dtype_raises(self, x):
        with pytest.raises(ValueError):
            fw.rolling_mean(x, 100, out=np.empty(len(x), dtype=np.float32))

    def test_non_contiguous_raises(self, x):
        big = np.empty(2 * len(x))
        with pytest.raises(ValueError):
            fw.rolling_mean(x, 100, out=big[::2])

    def test_out_with_return_cov_raises(self, x):
        with pytest.raises(ValueError):
            fw.rolling_corr(x, x, 50, return_cov=True, out=np.empty_like(x))


class TestOneDThreads:
    @pytest.mark.parametrize("fn", [fw.rolling_mean, fw.rolling_std, fw.rolling_min, fw.rolling_max, fw.rolling_sum, fw.rolling_var])
    def test_threads_bitwise_identical(self, x, fn):
        for w in (16, 100, 1000):
            a = fn(x, w, n_threads=1)
            b = fn(x, w, n_threads=4)
            assert nan_equal(a, b), f"{fn.__name__} w={w}"
