"""
pandas Series/DataFrame support: containers in -> containers out,
index/columns preserved.  pandas stays a soft dependency (fastwindow
must import without it).
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import fastwindow as fw


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(rng.standard_normal(100), index=idx, name="px")


class TestSeriesRoundTrip:
    def test_series_in_series_out(self, series):
        out = fw.rolling_mean(series, window=10)
        assert isinstance(out, pd.Series)
        assert out.index.equals(series.index)
        assert out.name == "px"
        np.testing.assert_array_equal(
            out.to_numpy(), fw.rolling_mean(series.to_numpy(), window=10))

    @pytest.mark.parametrize("fname,kwargs", [
        ("rolling_std", {}), ("rolling_sum", {}), ("rolling_min", {}),
        ("rolling_max", {}), ("rolling_quantile", {"q": 0.7}),
        ("rolling_skew", {}), ("rolling_kurt", {}), ("rolling_zscore", {}),
        ("expanding_mean", {}), ("expanding_std", {}),
    ])
    def test_all_1d_functions(self, series, fname, kwargs):
        f = getattr(fw, fname)
        if fname.startswith("expanding"):
            out = f(series, **kwargs)
        else:
            out = f(series, window=10, **kwargs)
        assert isinstance(out, pd.Series)
        assert out.index.equals(series.index)

    def test_ndarray_still_ndarray(self):
        x = np.arange(20.0)
        out = fw.rolling_mean(x, window=3)
        assert type(out) is np.ndarray

    def test_list_input_still_ndarray(self):
        out = fw.rolling_mean([1.0, 2.0, 3.0, 4.0], window=2)
        assert type(out) is np.ndarray

    def test_int_series(self):
        s = pd.Series([1, 2, 3, 4, 5])
        out = fw.rolling_mean(s, window=2)
        assert isinstance(out, pd.Series)
        assert out.iloc[1] == 1.5


class TestTwoSeriesFunctions:
    def test_corr(self, series):
        y = series * 0.5 + 1.0
        out = fw.rolling_corr(series, y, window=20)
        assert isinstance(out, pd.Series)
        assert out.index.equals(series.index)

    def test_corr_return_cov_tuple(self, series):
        y = series * 2.0
        corr, cov = fw.rolling_corr(series, y, window=20, return_cov=True)
        assert isinstance(corr, pd.Series) and isinstance(cov, pd.Series)

    def test_cov_and_spearman(self, series):
        y = -series
        assert isinstance(fw.rolling_cov(series, y, window=15), pd.Series)
        assert isinstance(fw.rolling_spearman(series, y, window=15),
                          pd.Series)

    def test_mixed_series_ndarray(self, series):
        #box follows the first Series argument
        out = fw.rolling_corr(series, series.to_numpy(), window=20)
        assert isinstance(out, pd.Series)


class TestDictReturningFunctions:
    def test_rolling_regression(self, series):
        res = fw.rolling_regression(series, window=20)
        for key in ("intercept", "slope", "r2"):
            assert isinstance(res[key], pd.Series)
            assert res[key].index.equals(series.index)

    def test_expanding_regression(self, series):
        res = fw.expanding_regression(series)
        assert isinstance(res["slope"], pd.Series)

    def test_multiple_regression_dataframe(self, series):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((100, 3)),
                         index=series.index, columns=["a", "b", "c"])
        res = fw.rolling_multiple_regression(series, X, window=30)
        assert isinstance(res["coef"], pd.DataFrame)
        assert list(res["coef"].columns) == ["intercept", "a", "b", "c"]
        assert res["coef"].index.equals(series.index)
        assert isinstance(res["r2"], pd.Series)


class TestDataFrame2D:
    def test_rolling_mean_2d(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.standard_normal((50, 4)),
                          columns=list("abcd"))
        out = fw.rolling_mean_2d(df, window=10)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == list("abcd")
        assert out.index.equals(df.index)

    def test_corr_matrix_returns_ndarray(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.standard_normal((50, 3)))
        out = fw.rolling_corr_matrix(df, window=10)
        assert type(out) is np.ndarray and out.shape == (50, 3, 3)


class TestSoftDependency:
    def test_import_without_pandas(self):
        code = (
            "import sys; sys.modules['pandas'] = None\n"
            "import numpy as np, fastwindow as fw\n"
            "r = fw.rolling_mean(np.arange(10.0), 3)\n"
            "assert type(r) is np.ndarray\n"
            "print('OK')\n"
        )
        res = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == "OK"


class TestOutParam:
    def test_out_with_series(self, series):
        buf = np.empty(len(series))
        out = fw.rolling_mean(series, window=10, out=buf)
        assert isinstance(out, pd.Series)
        assert np.shares_memory(out.to_numpy(), buf)
