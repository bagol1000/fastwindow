test_that("expanding_mean matches cumsum / seq_along", {
  set.seed(42)
  x <- rnorm(500)
  expect_equal(expanding_mean(x), cumsum(x) / seq_along(x),
               tolerance = 1e-12)
})

test_that("expanding_sum matches cumsum", {
  set.seed(43)
  x <- rnorm(500)
  expect_equal(expanding_sum(x), cumsum(x), tolerance = 1e-12)
})

test_that("expanding_std matches sd() on prefixes", {
  set.seed(44)
  x <- rnorm(100)
  out <- expanding_std(x)
  expect_true(is.na(out[1]))
  for (i in seq(2, 100, by = 13)) {
    expect_equal(out[i], sd(x[1:i]), tolerance = 1e-10)
  }
})

test_that("expanding_regression final step matches lm()", {
  set.seed(45)
  n <- 200
  y <- 2 + 0.3 * (0:(n - 1)) + rnorm(n)
  fit <- expanding_regression(y)
  ref <- lm(y ~ seq(0, n - 1))
  expect_equal(fit$intercept[n], unname(coef(ref)[1]), tolerance = 1e-9)
  expect_equal(fit$slope[n],     unname(coef(ref)[2]), tolerance = 1e-9)
})

test_that("expanding NA values are skipped, not propagated", {
  out <- expanding_mean(c(1, NA, 3, 5))
  expect_equal(out, c(1, 1, 2, 3))
})
