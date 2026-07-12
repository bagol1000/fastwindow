#Tests for rolling_skew, rolling_kurt, rolling_zscore.
#Reference implementations in pure R follow the scipy bias=FALSE
#conventions (identical to pandas .rolling().skew()/.kurt()).

ref_skew <- function(v) {
  n <- length(v)
  if (n < 3) return(NA_real_)
  m  <- mean(v)
  M2 <- sum((v - m)^2)
  M3 <- sum((v - m)^3)
  if (M2 <= 0) return(NA_real_)
  g1 <- (M3 / n) / (M2 / n)^1.5
  sqrt(n * (n - 1)) / (n - 2) * g1
}

ref_kurt <- function(v) {
  n <- length(v)
  if (n < 4) return(NA_real_)
  m  <- mean(v)
  M2 <- sum((v - m)^2)
  M4 <- sum((v - m)^4)
  if (M2 <= 0) return(NA_real_)
  g2 <- n * M4 / M2^2 - 3
  (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
}

roll_ref <- function(x, w, f, need) {
  sapply(seq_along(x), function(i) {
    if (i < w) return(NA_real_)
    win <- x[(i - w + 1):i]
    if (anyNA(win)) return(NA_real_)
    f(win)
  })
}

test_that("rolling_skew matches the bias-corrected reference", {
  set.seed(1)
  x <- rnorm(300)
  expect_equal(rolling_skew(x, 20), roll_ref(x, 20, ref_skew),
               tolerance = 1e-9)
})

test_that("rolling_kurt matches the bias-corrected reference", {
  set.seed(2)
  x <- rnorm(300)
  expect_equal(rolling_kurt(x, 20), roll_ref(x, 20, ref_kurt),
               tolerance = 1e-9)
})

test_that("skew/kurt keep precision under a large mean offset", {
  set.seed(3)
  x <- 1e6 + rnorm(300)
  expect_equal(rolling_skew(x, 50), roll_ref(x, 50, ref_skew),
               tolerance = 1e-4)
  expect_equal(rolling_kurt(x, 50), roll_ref(x, 50, ref_kurt),
               tolerance = 1e-3)
})

test_that("skew needs >= 3 and kurt >= 4 observations", {
  x <- c(1, 2, 4, 8, 16)
  sk <- rolling_skew(x, window = 3)
  expect_true(all(is.na(sk[1:2])))
  expect_false(anyNA(sk[3:5]))
  ku <- rolling_kurt(x, window = 4)
  expect_true(all(is.na(ku[1:3])))
  expect_false(anyNA(ku[4:5]))
})

test_that("constant windows give NA", {
  x <- rep(3.14, 20)
  expect_true(all(is.na(rolling_skew(x, 5))))
  expect_true(all(is.na(rolling_kurt(x, 6))))
})

test_that("skip_nan excludes missing values", {
  set.seed(4)
  x <- rnorm(100)
  x[c(10, 30, 55)] <- NA
  out <- rolling_skew(x, 15, min_periods = 5, skip_nan = TRUE)
  ref <- sapply(seq_along(x), function(i) {
    win <- x[max(1, i - 14):i]
    v <- win[is.finite(win)]
    if (length(v) < 5) return(NA_real_)
    ref_skew(v)
  })
  expect_equal(out, ref, tolerance = 1e-9)
})

test_that("rolling_zscore equals (x - mean) / sd", {
  set.seed(5)
  x <- rnorm(200)
  mu <- rolling_mean(x, 20)
  sd <- rolling_std(x, 20)
  expect_equal(rolling_zscore(x, 20), (x - mu) / sd, tolerance = 1e-12)
})

test_that("rolling_zscore gives NA on constant windows and NA inputs", {
  x <- rep(2, 20)
  expect_true(all(is.na(rolling_zscore(x, 5))))
  y <- c(1, 2, 3, 4, NA, 6, 7, 8, 9, 10)
  out <- rolling_zscore(y, 3, min_periods = 2, skip_nan = TRUE)
  expect_true(is.na(out[5]))
})
