test_that("rolling_corr matches cor() per window", {
  set.seed(42)
  x <- rnorm(200)
  y <- 0.6 * x + rnorm(200)
  w <- 30
  out <- rolling_corr(x, y, w)
  for (i in seq(w, 200, by = 19)) {
    idx <- (i - w + 1):i
    expect_equal(out[i], cor(x[idx], y[idx]), tolerance = 1e-10)
  }
  expect_true(all(is.na(out[1:(w - 1)])))
})

test_that("rolling_cov matches cov() per window", {
  set.seed(43)
  x <- rnorm(150)
  y <- rnorm(150)
  w <- 20
  out <- rolling_cov(x, y, w)
  for (i in seq(w, 150, by = 13)) {
    idx <- (i - w + 1):i
    expect_equal(out[i], cov(x[idx], y[idx]), tolerance = 1e-10)
  }
})

test_that("perfectly correlated series gives 1", {
  set.seed(44)
  x <- rnorm(100)
  out <- rolling_corr(x, 2 * x + 1, 20)
  expect_equal(out[!is.na(out)], rep(1, sum(!is.na(out))), tolerance = 1e-12)
})

test_that("constant series gives NA, not Inf", {
  out <- rolling_corr(rep(1, 50), rnorm(50), 10)
  expect_true(all(is.na(out)))
})

test_that("rolling_corr_matrix is symmetric with unit diagonal", {
  set.seed(45)
  X <- matrix(rnorm(400), ncol = 4)
  cm <- rolling_corr_matrix(X, 25)
  expect_equal(dim(cm), c(100L, 4L, 4L))
  expect_equal(cm[, 1, 1], rep(1, 100))
  s <- cm[60, , ]
  expect_equal(s, t(s))
  expect_equal(cm[60, 1, 2], cor(X[36:60, 1], X[36:60, 2]), tolerance = 1e-10)
})

test_that("rolling_spearman matches cor(method='spearman')", {
  set.seed(46)
  x <- rnorm(100)
  y <- rnorm(100)
  w <- 20
  out <- rolling_spearman(x, y, w)
  for (i in seq(w, 100, by = 17)) {
    idx <- (i - w + 1):i
    expect_equal(out[i], cor(x[idx], y[idx], method = "spearman"),
                 tolerance = 1e-10)
  }
})
