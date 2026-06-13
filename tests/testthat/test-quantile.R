test_that("exact rolling_quantile matches quantile(type = 7)", {
  set.seed(42)
  x <- rnorm(200)
  w <- 25
  for (q in c(0.25, 0.5, 0.75)) {
    out <- rolling_quantile(x, w, q = q, exact = TRUE)
    for (i in seq(w, 200, by = 21)) {
      win <- x[(i - w + 1):i]
      expect_equal(out[i], unname(quantile(win, q, type = 7)),
                   tolerance = 1e-10)
    }
  }
})

test_that("P2 approximation stays within 5% of exact", {
  set.seed(7)
  x <- rnorm(5000) + 10
  w <- 200
  approx <- rolling_quantile(x, w, q = 0.5, exact = FALSE)
  exact  <- rolling_quantile(x, w, q = 0.5, exact = TRUE)
  idx <- 1000:5000
  rel <- abs(approx[idx] - exact[idx]) / abs(exact[idx])
  expect_lt(max(rel, na.rm = TRUE), 0.05)
})

test_that("invalid q is rejected and large exact windows work", {
  expect_error(rolling_quantile(rnorm(10), 3, q = 0), "q")
  expect_error(rolling_quantile(rnorm(10), 3, q = 1), "q")
  set.seed(1)
  x <- rnorm(1500)
  out <- rolling_quantile(x, 600, q = 0.5, exact = TRUE)
  expect_equal(out[1500], unname(quantile(x[901:1500], 0.5, type = 7)),
               tolerance = 1e-10)
})
