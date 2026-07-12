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

test_that("rolling_min/max support min_periods", {
  x <- c(3, 1, 4, 1, 5, 9, 2, 6)
  mn <- rolling_min(x, window = 3, min_periods = 1)
  expect_equal(mn[1], 3)                #partial window of 1
  expect_equal(mn[2], 1)                #partial window of 2
  expect_equal(mn[3], 1)
  mx <- rolling_max(x, window = 3, min_periods = 2)
  expect_true(is.na(mx[1]))
  expect_equal(mx[2], 3)
})

test_that("rolling_min/max skip_nan matches per-window na.rm reference", {
  set.seed(42)
  x <- rnorm(200)
  x[sample(200, 12)] <- NA
  w  <- 7
  mp <- 2
  ref <- function(op) {
    sapply(seq_along(x), function(i) {
      win <- x[max(1, i - w + 1):i]
      v   <- win[is.finite(win)]
      if (length(v) < mp) return(NA_real_)
      op(v)
    })
  }
  expect_equal(rolling_min(x, w, min_periods = mp, skip_nan = TRUE),
               ref(min))
  expect_equal(rolling_max(x, w, min_periods = mp, skip_nan = TRUE),
               ref(max))
})

test_that("rolling_min/max default NaN propagation is unchanged", {
  x <- c(1, 2, NA, 4, 5, 6)
  mn <- rolling_min(x, window = 3)
  expect_true(all(is.na(mn[1:5])))
  expect_equal(mn[6], 4)
})

test_that("rolling_min/max blocked path (window >= 16) honours min_periods", {
  set.seed(7)
  x <- rnorm(500)
  w <- 32
  out <- rolling_min(x, w, min_periods = 5)
  ref <- sapply(seq_along(x), function(i) {
    win <- x[max(1, i - w + 1):i]
    if (length(win) < 5) NA_real_ else min(win)
  })
  expect_equal(out, ref)
})

test_that("rolling_min_matrix/max_matrix pass min_periods and skip_nan", {
  set.seed(1)
  x <- rnorm(100)
  x[c(5, 20, 50)] <- NA
  X <- cbind(x, -x)
  out <- rolling_min_matrix(X, window = 8, min_periods = 2, skip_nan = TRUE)
  expect_equal(out[, 1],
               rolling_min(x, 8, min_periods = 2, skip_nan = TRUE))
  expect_equal(out[, 2],
               rolling_min(-x, 8, min_periods = 2, skip_nan = TRUE))
})
