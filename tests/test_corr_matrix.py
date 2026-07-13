"""
Tests for the rolling correlation matrix
"""

import numpy as np
import pytest

import fastwindow as fw

NaN = float("nan")

@pytest.fixture
def X5():
    rng = np.random.default_rng(42)
    base = rng.standard_normal((300, 5))
    base[:, 1] += 0.5 * base[:, 0]          #introduce some correlation
    base[:, 3] -= 0.8 * base[:, 2]
    return base


class TestCorrMatrixCore:
    def test_p2_matches_rolling_corr(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 2))
        out = fw.rolling_corr_matrix(X, window=20)
        ref = fw.rolling_corr(X[:, 0], X[:, 1], window=20)
        np.testing.assert_array_equal(out[:, 0, 1], ref)   #same kernel → exact
        np.testing.assert_array_equal(out[:, 1, 0], ref)

    def test_p5_every_pair_matches(self, X5):
        out = fw.rolling_corr_matrix(X5, window=30)
        assert out.shape == (300, 5, 5)
        for i in range(5):
            for j in range(i + 1, 5):
                ref = fw.rolling_corr(X5[:, i], X5[:, j], window=30)
                finite = ~np.isnan(ref)
                np.testing.assert_allclose(
                    out[finite, i, j], ref[finite], rtol=1e-9,
                    err_msg=f"pair ({i},{j})")
                nan_match = np.isnan(out[:, i, j]) == np.isnan(ref)
                assert nan_match.all(), f"NaN pattern differs for pair ({i},{j})"

    def test_symmetry(self, X5):
        out = fw.rolling_corr_matrix(X5, window=30)
        np.testing.assert_array_equal(out, np.swapaxes(out, 1, 2))

    def test_diagonal_is_one(self, X5):
        out = fw.rolling_corr_matrix(X5, window=30)
        for d in range(5):
            np.testing.assert_array_equal(out[:, d, d], np.ones(300))

    def test_threads_identical(self, X5):
        out1 = fw.rolling_corr_matrix(X5, window=30, n_threads=1)
        out4 = fw.rolling_corr_matrix(X5, window=30, n_threads=4)
        np.testing.assert_array_equal(
            np.nan_to_num(out1, nan=-999.0),
            np.nan_to_num(out4, nan=-999.0))

    def test_min_periods(self, X5):
        out = fw.rolling_corr_matrix(X5, window=30, min_periods=10)
        ref = fw.rolling_corr(X5[:, 0], X5[:, 1], window=30, min_periods=10)
        finite = ~np.isnan(ref)
        np.testing.assert_allclose(out[finite, 0, 1], ref[finite], rtol=1e-9)
        assert np.array_equal(np.isnan(out[:, 0, 1]), np.isnan(ref))


class TestCorrMatrixNan:
    def test_nan_isolated_to_affected_pairs(self):
        """NaN in column 2 poisons only pairs involving column 2."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 4))
        X[50, 2] = NaN
        out = fw.rolling_corr_matrix(X, window=10)
        w = 10
        #Pairs with column 2 → NaN while index 50 is inside the window
        for t in range(50, 50 + w):
            for other in (0, 1, 3):
                assert np.isnan(out[t, 2, other]), f"t={t}, pair (2,{other})"
                assert np.isnan(out[t, other, 2])
        #Pair (0,1) unaffected at t=50
        assert not np.isnan(out[50, 0, 1])
        #Diagonal still 1.0 even at poisoned positions
        assert out[50, 2, 2] == 1.0
        #Recovery after the NaN leaves the window
        assert not np.isnan(out[50 + w, 2, 0])

    def test_nan_consistent_with_rolling_corr(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((150, 3))
        X[rng.random(150) < 0.05, 1] = NaN
        out = fw.rolling_corr_matrix(X, window=15)
        for i in range(3):
            for j in range(i + 1, 3):
                ref = fw.rolling_corr(X[:, i], X[:, j], window=15)
                assert np.array_equal(np.isnan(out[:, i, j]), np.isnan(ref)), \
                    f"NaN pattern differs for pair ({i},{j})"


class TestCorrMatrixErrors:
    def test_1d_input_rejected(self):
        with pytest.raises(ValueError):
            fw.rolling_corr_matrix(np.arange(10.0), window=3)

    def test_single_column_rejected(self):
        with pytest.raises(ValueError):
            fw.rolling_corr_matrix(np.zeros((10, 1)), window=3)

    def test_too_many_columns_rejected(self):
        with pytest.raises(ValueError):
            fw.rolling_corr_matrix(np.zeros((10, 51)), window=3)

    def test_window_zero(self):
        with pytest.raises(ValueError):
            fw.rolling_corr_matrix(np.zeros((10, 3)), window=0)

    def test_row_major_input_accepted(self):
        """C-ordered input must give the same result as Fortran-ordered."""
        rng = np.random.default_rng(9)
        Xc = np.ascontiguousarray(rng.standard_normal((100, 3)))
        Xf = np.asfortranarray(Xc)
        out_c = fw.rolling_corr_matrix(Xc, window=10)
        out_f = fw.rolling_corr_matrix(Xf, window=10)
        np.testing.assert_array_equal(
            np.nan_to_num(out_c, nan=-999.0),
            np.nan_to_num(out_f, nan=-999.0))


def test_public_pair_output_matches_full_cube():
    rng = np.random.default_rng(91)
    X = rng.standard_normal((120, 4))
    pairs = fw.rolling_corr_pairs(X, 15)
    cube = fw.rolling_corr_matrix(X, 15)
    expected = np.column_stack([
        cube[:, 0, 1], cube[:, 0, 2], cube[:, 0, 3],
        cube[:, 1, 2], cube[:, 1, 3], cube[:, 2, 3],
    ])
    np.testing.assert_allclose(pairs, expected, equal_nan=True)
