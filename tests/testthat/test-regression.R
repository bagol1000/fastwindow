test_that("rolling_regression matches lm() on time index", {
  set.seed(7)
  y <- cumsum(rnorm(120))
  w <- 20
  fit <- rolling_regression(y, w)
  for (i in seq(w, 120, by = 17)) {
    win <- y[(i - w + 1):i]
    ref <- lm(win ~ seq(0, w - 1))
    expect_equal(fit$intercept[i], unname(coef(ref)[1]), tolerance = 1e-8)
    expect_equal(fit$slope[i],     unname(coef(ref)[2]), tolerance = 1e-8)
    expect_equal(fit$r2[i], summary(ref)$r.squared,      tolerance = 1e-7)
  }
})

test_that("rolling_regression_xy matches lm()", {
  set.seed(8)
  x <- rnorm(100)
  y <- 2 * x + rnorm(100, sd = 0.5)
  w <- 25
  fit <- rolling_regression_xy(y, x, w)
  for (i in seq(w, 100, by = 11)) {
    idx <- (i - w + 1):i
    ref <- lm(y[idx] ~ x[idx])
    expect_equal(fit$intercept[i], unname(coef(ref)[1]), tolerance = 1e-8)
    expect_equal(fit$slope[i],     unname(coef(ref)[2]), tolerance = 1e-8)
  }
})

test_that("rolling_multiple_regression matches lm()", {
  set.seed(9)
  X <- matrix(rnorm(300), ncol = 3)
  y <- as.numeric(X %*% c(1, -2, 0.5) + rnorm(100))
  w <- 30
  fit <- rolling_multiple_regression(y, X, w)
  for (i in seq(w, 100, by = 23)) {
    idx <- (i - w + 1):i
    ref <- lm(y[idx] ~ X[idx, ])
    expect_equal(unname(fit$coef[i, ]), unname(coef(ref)), tolerance = 1e-7)
    expect_equal(fit$r2[i], summary(ref)$r.squared, tolerance = 1e-7)
  }
})

test_that("k = 17 regressors are rejected", {
  expect_error(
    rolling_multiple_regression(rnorm(50), matrix(rnorm(50 * 17), ncol = 17), 30),
    "columns")
})
