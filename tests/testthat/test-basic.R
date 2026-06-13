test_that("rolling_mean matches known values", {
  expect_equal(rolling_mean(c(1, 2, 3, 4, 5), 3),
               c(NA, NA, 2, 3, 4))
  expect_equal(rolling_mean(c(10, 20, 30), 1), c(10, 20, 30))
  expect_equal(rolling_mean(c(1, 2, 3, 4), 3, min_periods = 2),
               c(NA, 1.5, 2, 3))
})

test_that("rolling_std matches known values", {
  expect_equal(rolling_std(c(1, 3, 5, 7), 2),
               c(NA, sqrt(2), sqrt(2), sqrt(2)))
  expect_equal(rolling_var(c(1, 2, 3), 3, ddof = 0L)[3], 2 / 3)
})

test_that("rolling stats match base R per window", {
  set.seed(42)
  x <- rnorm(300)
  w <- 25
  m  <- rolling_mean(x, w)
  s  <- rolling_std(x, w)
  mn <- rolling_min(x, w)
  mx <- rolling_max(x, w)
  for (i in seq(w, 300, by = 13)) {
    win <- x[(i - w + 1):i]
    expect_equal(m[i],  mean(win), tolerance = 1e-12)
    expect_equal(s[i],  sd(win),   tolerance = 1e-10)
    expect_equal(mn[i], min(win))
    expect_equal(mx[i], max(win))
  }
})

test_that("NaN propagates through the window", {
  x <- c(1, 2, NA, 4, 5, 6)
  out <- rolling_mean(x, 2)
  expect_true(is.na(out[3]) && is.na(out[4]))
  expect_equal(out[5], 4.5)
})

test_that("input validation gives clean R errors", {
  expect_error(rolling_mean(c(1, 2), 0), "window")
  expect_error(rolling_std(c(1, 2), 2, ddof = 2L), "ddof")
})
