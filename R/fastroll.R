#fastroll — R wrappers over the fastwindow C++17 core.
#All inputs are validated here so users get clean R error messages
#before any .Call into compiled code.

.check_basic_args <- function(x, window, min_periods) {
    if (!is.numeric(x)) stop("`x` must be a numeric vector")
    if (length(window) != 1L || !is.finite(window) || window < 1)
        stop("`window` must be a positive integer scalar")
    if (length(min_periods) != 1L || !is.finite(min_periods) || min_periods < 0)
        stop("`min_periods` must be a non-negative integer scalar")
    invisible(TRUE)
}

#' Rolling mean
#'
#' @description Computes the mean over a sliding window of fixed length.
#'   The first \code{window - 1} elements of the result are \code{NA}
#'   unless \code{min_periods} is smaller than \code{window}.
#'
#' @param x numeric vector.
#' @param window window length, a positive integer.
#' @param min_periods minimum number of valid observations required to
#'   emit a value; defaults to \code{window}.
#' @param skip_nan if \code{TRUE}, \code{NA}/\code{NaN} values are excluded
#'   from the window instead of propagating.
#' @param n_threads OpenMP threads over blocks (AVX2 builds only);
#'   \code{0L} (default) means single-threaded.
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' rolling_mean(c(1, 2, 3, 4, 5), window = 3)
#'
#' @seealso \code{\link{rolling_std}}, \code{\link{rolling_sum}}
#' @export
rolling_mean <- function(x, window, min_periods = window, skip_nan = FALSE,
                         n_threads = 0L) {
    .check_basic_args(x, window, min_periods)
    cpp_rolling_mean(as.double(x), as.integer(window),
                     as.integer(min_periods), isTRUE(skip_nan),
                     as.integer(n_threads))
}

#' Rolling standard deviation
#'
#' @description Computes the standard deviation over a sliding window using
#'   a numerically stable sliding Welford algorithm.
#'
#' @inheritParams rolling_mean
#' @param ddof delta degrees of freedom: \code{1L} (default) for the sample
#'   estimator, \code{0L} for the population estimator.
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' rolling_std(c(1, 3, 5, 7), window = 2)
#'
#' @seealso \code{\link{rolling_var}}
#' @export
rolling_std <- function(x, window, min_periods = window,
                        ddof = 1L, skip_nan = FALSE, n_threads = 0L) {
    .check_basic_args(x, window, min_periods)
    if (!ddof %in% c(0L, 1L)) stop("`ddof` must be 0 or 1")
    cpp_rolling_std(as.double(x), as.integer(window),
                    as.integer(min_periods), as.integer(ddof),
                    isTRUE(skip_nan), as.integer(n_threads))
}

#' Rolling variance
#'
#' @description Computes the variance over a sliding window using a
#'   numerically stable sliding Welford algorithm.
#'
#' @inheritParams rolling_std
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' rolling_var(c(1, 2, 3, 4), window = 3)
#'
#' @seealso \code{\link{rolling_std}}
#' @export
rolling_var <- function(x, window, min_periods = window,
                        ddof = 1L, skip_nan = FALSE, n_threads = 0L) {
    .check_basic_args(x, window, min_periods)
    if (!ddof %in% c(0L, 1L)) stop("`ddof` must be 0 or 1")
    cpp_rolling_var(as.double(x), as.integer(window),
                    as.integer(min_periods), as.integer(ddof),
                    isTRUE(skip_nan), as.integer(n_threads))
}

#' Rolling sum
#'
#' @description Computes the sum over a sliding window with periodic
#'   compensated (Kahan) re-summation for numerical stability.
#'
#' @inheritParams rolling_mean
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' rolling_sum(c(1, 2, 3, 4, 5), window = 3)
#'
#' @seealso \code{\link{rolling_mean}}
#' @export
rolling_sum <- function(x, window, min_periods = window, skip_nan = FALSE,
                        n_threads = 0L) {
    .check_basic_args(x, window, min_periods)
    cpp_rolling_sum(as.double(x), as.integer(window),
                    as.integer(min_periods), isTRUE(skip_nan),
                    as.integer(n_threads))
}

#' Rolling minimum
#'
#' @description Computes the minimum over a sliding window in O(n) total
#'   time via a monotonic deque.  Any \code{NA}/\code{NaN} inside the
#'   window yields \code{NA} for that position.
#'
#' @param x numeric vector.
#' @param window window length, a positive integer.
#' @param n_threads OpenMP threads over blocks (AVX2 builds only);
#'   \code{0L} (default) means single-threaded.
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' rolling_min(c(3, 1, 4, 1, 5), window = 3)
#'
#' @seealso \code{\link{rolling_max}}
#' @export
rolling_min <- function(x, window, n_threads = 0L) {
    .check_basic_args(x, window, window)
    cpp_rolling_min(as.double(x), as.integer(window), as.integer(n_threads))
}

#' Rolling maximum
#'
#' @description Computes the maximum over a sliding window in O(n) total
#'   time via a monotonic deque.  Any \code{NA}/\code{NaN} inside the
#'   window yields \code{NA} for that position.
#'
#' @inheritParams rolling_min
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' rolling_max(c(3, 1, 4, 1, 5), window = 3)
#'
#' @seealso \code{\link{rolling_min}}
#' @export
rolling_max <- function(x, window, n_threads = 0L) {
    .check_basic_args(x, window, window)
    cpp_rolling_max(as.double(x), as.integer(window), as.integer(n_threads))
}

#' Rolling simple linear regression on time
#'
#' @description Ordinary least squares of \code{y} on the within-window time
#'   index \code{x = 0, 1, ..., window - 1}, updated in O(1) per step.
#'   Common in finance for rolling trend estimation.
#'
#' @param y numeric vector (dependent variable).
#' @param window window length, a positive integer.
#' @param min_periods minimum number of observations required; defaults to
#'   \code{window}.
#' @param time_as_x must be \code{TRUE}; pass an explicit regressor via
#'   \code{\link{rolling_regression_xy}} instead.
#'
#' @return A list with elements \code{intercept}, \code{slope} and \code{r2},
#'   each a numeric vector of the same length as \code{y}.
#'
#' @examples
#' fit <- rolling_regression(cumsum(rnorm(100)), window = 20)
#' head(fit$slope, 25)
#'
#' @seealso \code{\link{rolling_regression_xy}}
#' @export
rolling_regression <- function(y, window, min_periods = window,
                               time_as_x = TRUE) {
    .check_basic_args(y, window, min_periods)
    if (!isTRUE(time_as_x))
        stop("`time_as_x = FALSE` requires an explicit regressor; ",
             "use rolling_regression_xy(y, x, window) instead")
    cpp_rolling_regression(as.double(y), as.integer(window),
                           as.integer(min_periods))
}

#' Rolling simple linear regression on an explicit regressor
#'
#' @description Ordinary least squares of \code{y} on \code{x} over a
#'   sliding window, updated in O(1) per step via running sums.
#'
#' @param y numeric vector (dependent variable).
#' @param x numeric vector (regressor), same length as \code{y}.
#' @param window window length, a positive integer.
#' @param min_periods minimum number of observations required; defaults to
#'   \code{window}.
#'
#' @return A list with elements \code{intercept}, \code{slope} and \code{r2},
#'   each a numeric vector of the same length as \code{y}.
#'
#' @examples
#' x <- rnorm(100)
#' y <- 2 * x + rnorm(100, sd = 0.1)
#' fit <- rolling_regression_xy(y, x, window = 20)
#'
#' @seealso \code{\link{rolling_regression}}
#' @export
rolling_regression_xy <- function(y, x, window, min_periods = window) {
    .check_basic_args(y, window, min_periods)
    if (!is.numeric(x)) stop("`x` must be a numeric vector")
    if (length(x) != length(y)) stop("`x` and `y` must have the same length")
    cpp_rolling_regression_xy(as.double(y), as.double(x),
                              as.integer(window), as.integer(min_periods))
}

#' Rolling multiple linear regression
#'
#' @description Ordinary least squares of \code{y} on \code{k} regressor
#'   columns over a sliding window.  The inverse Gram matrix is maintained
#'   incrementally via Sherman-Morrison rank-1 updates (O(k^2) per step)
#'   with a Cholesky fallback on near-singular steps.  An intercept is
#'   added internally: pass \code{k} regressors, receive \code{k + 1}
#'   coefficients with the intercept first.
#'
#' @param y numeric vector (dependent variable).
#' @param X numeric matrix with \code{length(y)} rows and \code{k} columns
#'   (\code{1 <= k <= 16}); do not include an intercept column.
#' @param window window length, a positive integer.
#' @param min_periods minimum number of observations required; defaults to
#'   \code{window}.
#'
#' @return A list with elements \code{coef} (matrix \code{n x (k+1)},
#'   intercept first), \code{r2} and \code{residual_std} (numeric vectors).
#'
#' @examples
#' X <- matrix(rnorm(300), ncol = 3)
#' y <- X %*% c(1, 2, 3) + rnorm(100, sd = 0.1)
#' fit <- rolling_multiple_regression(as.numeric(y), X, window = 30)
#'
#' @seealso \code{\link{rolling_regression_xy}}
#' @export
rolling_multiple_regression <- function(y, X, window, min_periods = window) {
    .check_basic_args(y, window, min_periods)
    if (!is.matrix(X) || !is.numeric(X)) stop("`X` must be a numeric matrix")
    if (nrow(X) != length(y)) stop("`X` must have length(y) rows")
    k <- ncol(X)
    if (k < 1 || k > 16) stop("`X` must have between 1 and 16 columns")
    storage.mode(X) <- "double"
    cpp_rolling_multiple_regression(as.double(y), X,
                                    as.integer(window),
                                    as.integer(min_periods))
}

#' Rolling Pearson correlation
#'
#' @description Computes the Pearson correlation between two series over a
#'   sliding window via O(1) running-sum updates.  A window in which either
#'   series is constant yields \code{NA} (never \code{Inf}).
#'
#' @param x,y numeric vectors of equal length.
#' @param window window length, a positive integer.
#' @param min_periods minimum number of valid pairs required; defaults to
#'   \code{window}.
#' @param return_cov if \code{TRUE}, also return the sample covariance.
#' @param skip_nan if \code{TRUE}, pairs with \code{NA} in either series are
#'   excluded from the window instead of propagating.
#'
#' @return If \code{return_cov} is \code{FALSE}, a numeric vector of
#'   correlations; otherwise a list with elements \code{corr} and \code{cov}.
#'
#' @examples
#' x <- rnorm(100)
#' rolling_corr(x, 2 * x + rnorm(100), window = 20)
#'
#' @seealso \code{\link{rolling_cov}}
#' @export
rolling_corr <- function(x, y, window, min_periods = window,
                         return_cov = FALSE, skip_nan = FALSE) {
    .check_basic_args(x, window, min_periods)
    if (!is.numeric(y)) stop("`y` must be a numeric vector")
    if (length(x) != length(y)) stop("`x` and `y` must have the same length")
    res <- cpp_rolling_corr(as.double(x), as.double(y), as.integer(window),
                            as.integer(min_periods), isTRUE(return_cov),
                            isTRUE(skip_nan))
    if (isTRUE(return_cov)) res else res$corr
}

#' Rolling covariance
#'
#' @description Computes the covariance between two series over a sliding
#'   window via O(1) running-sum updates.
#'
#' @inheritParams rolling_corr
#' @param ddof delta degrees of freedom: \code{1L} (default) for the sample
#'   estimator, \code{0L} for the population estimator.
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' x <- rnorm(100)
#' rolling_cov(x, 2 * x + rnorm(100), window = 20)
#'
#' @seealso \code{\link{rolling_corr}}
#' @export
rolling_cov <- function(x, y, window, min_periods = window,
                        ddof = 1L, skip_nan = FALSE) {
    .check_basic_args(x, window, min_periods)
    if (!is.numeric(y)) stop("`y` must be a numeric vector")
    if (length(x) != length(y)) stop("`x` and `y` must have the same length")
    if (!ddof %in% c(0L, 1L)) stop("`ddof` must be 0 or 1")
    cpp_rolling_cov(as.double(x), as.double(y), as.integer(window),
                    as.integer(min_periods), as.integer(ddof),
                    isTRUE(skip_nan))
}

#' Rolling correlation matrix
#'
#' @description Computes the rolling Pearson correlation matrix across the
#'   columns of \code{X}.  The \code{p(p-1)/2} column pairs are computed
#'   independently and in parallel with OpenMP.
#'
#' @param X numeric matrix with \code{n} rows and \code{p} columns
#'   (\code{2 <= p <= 50}).
#' @param window window length, a positive integer.
#' @param min_periods minimum number of observations required; defaults to
#'   \code{window}.
#' @param n_threads number of OpenMP threads; \code{0L} (default) uses all
#'   available.
#'
#' @return An array of dimension \code{c(n, p, p)}; each slice
#'   \code{[t, , ]} is symmetric with unit diagonal.
#'
#' @examples
#' X <- matrix(rnorm(300), ncol = 3)
#' cm <- rolling_corr_matrix(X, window = 20)
#' cm[100, , ]
#'
#' @seealso \code{\link{rolling_corr}}
#' @export
rolling_corr_matrix <- function(X, window, min_periods = window,
                                n_threads = 0L) {
    if (!is.matrix(X) || !is.numeric(X)) stop("`X` must be a numeric matrix")
    p <- ncol(X)
    if (p < 2 || p > 50) stop("`X` must have between 2 and 50 columns")
    .check_basic_args(X[, 1], window, min_periods)
    if (length(n_threads) != 1L || !is.finite(n_threads) || n_threads < 0)
        stop("`n_threads` must be a non-negative integer scalar")
    storage.mode(X) <- "double"
    cpp_rolling_corr_matrix(X, as.integer(window),
                            as.integer(min_periods), as.integer(n_threads))
}

#' Rolling quantile
#'
#' @description Computes a rolling quantile.  With \code{exact = FALSE}
#'   (default) the P-squared streaming approximation (Jain & Chlamtac, 1985)
#'   is used — O(1) per step, suitable for stationary series.  With
#'   \code{exact = TRUE} an exact two-heap order-statistic structure is used
#'   (O(log window) amortised per step) and the result matches
#'   \code{quantile(type = 7)} / \code{numpy.percentile}.
#'
#' @inheritParams rolling_mean
#' @param q the quantile, strictly between 0 and 1.
#' @param exact use the exact sorted-buffer mode instead of P-squared.
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' rolling_quantile(rnorm(100), window = 20, q = 0.5, exact = TRUE)
#'
#' @export
rolling_quantile <- function(x, window, q = 0.5, min_periods = window,
                             exact = FALSE) {
    .check_basic_args(x, window, min_periods)
    if (length(q) != 1L || !is.finite(q) || q <= 0 || q >= 1)
        stop("`q` must be strictly between 0 and 1")
    cpp_rolling_quantile(as.double(x), as.integer(window), as.double(q),
                         as.integer(min_periods), isTRUE(exact))
}

#' Expanding mean
#'
#' @description Mean over a growing window (from the first observation to
#'   the current one).  \code{NA} values are skipped, not counted.
#'
#' @param x numeric vector.
#' @param min_periods minimum number of valid observations required.
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' expanding_mean(c(1, 2, 3, 4))
#'
#' @seealso \code{\link{expanding_std}}, \code{\link{expanding_sum}}
#' @export
expanding_mean <- function(x, min_periods = 1L) {
    .check_basic_args(x, 1L, min_periods)
    cpp_expanding_mean(as.double(x), as.integer(min_periods))
}

#' Expanding variance
#'
#' @inheritParams expanding_mean
#' @param ddof delta degrees of freedom (\code{1L} sample, \code{0L}
#'   population).
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' expanding_var(rnorm(50))
#'
#' @seealso \code{\link{expanding_std}}
#' @export
expanding_var <- function(x, min_periods = 2L, ddof = 1L) {
    .check_basic_args(x, 1L, min_periods)
    if (!ddof %in% c(0L, 1L)) stop("`ddof` must be 0 or 1")
    cpp_expanding_var(as.double(x), as.integer(min_periods),
                      as.integer(ddof))
}

#' Expanding standard deviation
#'
#' @inheritParams expanding_var
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' expanding_std(rnorm(50))
#'
#' @seealso \code{\link{expanding_var}}
#' @export
expanding_std <- function(x, min_periods = 2L, ddof = 1L) {
    .check_basic_args(x, 1L, min_periods)
    if (!ddof %in% c(0L, 1L)) stop("`ddof` must be 0 or 1")
    cpp_expanding_std(as.double(x), as.integer(min_periods),
                      as.integer(ddof))
}

#' Expanding sum
#'
#' @inheritParams expanding_mean
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' expanding_sum(c(1, 2, 3))
#'
#' @seealso \code{\link{expanding_mean}}
#' @export
expanding_sum <- function(x, min_periods = 1L) {
    .check_basic_args(x, 1L, min_periods)
    cpp_expanding_sum(as.double(x), as.integer(min_periods))
}

#' Expanding simple linear regression on time
#'
#' @description OLS of \code{y} on the time index \code{0, 1, ..., i} over a
#'   growing window.
#'
#' @param y numeric vector (dependent variable).
#' @param min_periods minimum number of observations required (at least 2).
#'
#' @return A list with elements \code{intercept}, \code{slope} and \code{r2}.
#'
#' @examples
#' expanding_regression(cumsum(rnorm(100)))
#'
#' @seealso \code{\link{rolling_regression}}
#' @export
expanding_regression <- function(y, min_periods = 2L) {
    .check_basic_args(y, 1L, min_periods)
    cpp_expanding_regression(as.double(y), as.integer(min_periods))
}

#' Column-wise rolling mean
#'
#' @description Applies \code{\link{rolling_mean}} to each column of a
#'   matrix; columns are processed in parallel with OpenMP.
#'
#' @param X numeric matrix (n rows, p columns).
#' @param window window length, a positive integer.
#' @param min_periods minimum number of observations; defaults to
#'   \code{window}.
#' @param n_threads OpenMP thread count; \code{0L} uses all available.
#'
#' @return A numeric matrix of the same dimensions as \code{X}.
#'
#' @examples
#' rolling_mean_matrix(matrix(rnorm(300), ncol = 3), window = 20)
#'
#' @seealso \code{\link{rolling_mean}}
#' @export
rolling_mean_matrix <- function(X, window, min_periods = window,
                                n_threads = 0L) {
    if (!is.matrix(X) || !is.numeric(X)) stop("`X` must be a numeric matrix")
    .check_basic_args(X[, 1], window, min_periods)
    storage.mode(X) <- "double"
    cpp_rolling_mean_matrix(X, as.integer(window),
                            as.integer(min_periods), as.integer(n_threads))
}

#' Column-wise rolling standard deviation
#'
#' @inheritParams rolling_mean_matrix
#' @param ddof delta degrees of freedom (\code{1L} sample, \code{0L}
#'   population).
#'
#' @return A numeric matrix of the same dimensions as \code{X}.
#'
#' @examples
#' rolling_std_matrix(matrix(rnorm(300), ncol = 3), window = 20)
#'
#' @seealso \code{\link{rolling_std}}
#' @export
rolling_std_matrix <- function(X, window, min_periods = window,
                               ddof = 1L, n_threads = 0L) {
    if (!is.matrix(X) || !is.numeric(X)) stop("`X` must be a numeric matrix")
    .check_basic_args(X[, 1], window, min_periods)
    if (!ddof %in% c(0L, 1L)) stop("`ddof` must be 0 or 1")
    storage.mode(X) <- "double"
    cpp_rolling_std_matrix(X, as.integer(window), as.integer(min_periods),
                           as.integer(ddof), as.integer(n_threads))
}

#' Column-wise rolling sum
#'
#' @inheritParams rolling_mean_matrix
#'
#' @return A numeric matrix of the same dimensions as \code{X}.
#'
#' @examples
#' rolling_sum_matrix(matrix(rnorm(300), ncol = 3), window = 20)
#'
#' @seealso \code{\link{rolling_sum}}
#' @export
rolling_sum_matrix <- function(X, window, min_periods = window,
                               n_threads = 0L) {
    if (!is.matrix(X) || !is.numeric(X)) stop("`X` must be a numeric matrix")
    .check_basic_args(X[, 1], window, min_periods)
    storage.mode(X) <- "double"
    cpp_rolling_sum_matrix(X, as.integer(window),
                           as.integer(min_periods), as.integer(n_threads))
}

#' Column-wise rolling minimum
#'
#' @inheritParams rolling_mean_matrix
#'
#' @return A numeric matrix of the same dimensions as \code{X}.
#'
#' @examples
#' rolling_min_matrix(matrix(rnorm(300), ncol = 3), window = 20)
#'
#' @seealso \code{\link{rolling_min}}
#' @export
rolling_min_matrix <- function(X, window, n_threads = 0L) {
    if (!is.matrix(X) || !is.numeric(X)) stop("`X` must be a numeric matrix")
    .check_basic_args(X[, 1], window, window)
    storage.mode(X) <- "double"
    cpp_rolling_min_matrix(X, as.integer(window), as.integer(n_threads))
}

#' Column-wise rolling maximum
#'
#' @inheritParams rolling_mean_matrix
#'
#' @return A numeric matrix of the same dimensions as \code{X}.
#'
#' @examples
#' rolling_max_matrix(matrix(rnorm(300), ncol = 3), window = 20)
#'
#' @seealso \code{\link{rolling_max}}
#' @export
rolling_max_matrix <- function(X, window, n_threads = 0L) {
    if (!is.matrix(X) || !is.numeric(X)) stop("`X` must be a numeric matrix")
    .check_basic_args(X[, 1], window, window)
    storage.mode(X) <- "double"
    cpp_rolling_max_matrix(X, as.integer(window), as.integer(n_threads))
}

#' Rolling Spearman rank correlation
#'
#' @description Spearman rank correlation between two series over a sliding
#'   window, with average ranks for ties (matching
#'   \code{cor(method = "spearman")}).
#'
#'   Each step costs O(window) via linear walks over sorted window copies,
#'   so windows up to a few thousand elements are practical.
#'
#' @param x,y numeric vectors of equal length.
#' @param window window length, a positive integer.
#' @param min_periods minimum number of valid pairs required; defaults to
#'   \code{window}.
#'
#' @return A numeric vector of the same length as \code{x}.
#'
#' @examples
#' x <- rnorm(100)
#' rolling_spearman(x, exp(x), window = 20)
#'
#' @seealso \code{\link{rolling_corr}}
#' @export
rolling_spearman <- function(x, y, window, min_periods = window) {
    .check_basic_args(x, window, min_periods)
    if (!is.numeric(y)) stop("`y` must be a numeric vector")
    if (length(x) != length(y)) stop("`x` and `y` must have the same length")
    cpp_rolling_spearman(as.double(x), as.double(y), as.integer(window),
                         as.integer(min_periods))
}

#' Set the default OpenMP thread count
#'
#' @description Sets the number of OpenMP threads used by parallel functions
#'   when their \code{n_threads} argument is \code{0}.
#'
#' @param n a positive integer.
#'
#' @return Invisibly, \code{NULL}.
#'
#' @examples
#' set_num_threads(2)
#' get_num_threads()
#'
#' @seealso \code{\link{get_num_threads}}
#' @export
set_num_threads <- function(n) {
    if (length(n) != 1L || !is.finite(n) || n < 1)
        stop("`n` must be a positive integer scalar")
    invisible(cpp_set_num_threads(as.integer(n)))
}

#' Get the default OpenMP thread count
#'
#' @description Returns the number of OpenMP threads used by parallel
#'   functions when their \code{n_threads} argument is \code{0}.
#'
#' @return An integer.
#'
#' @examples
#' get_num_threads()
#'
#' @seealso \code{\link{set_num_threads}}
#' @export
get_num_threads <- function() {
    cpp_get_num_threads()
}

#' AVX2 availability
#'
#' @description Reports whether the compiled core was built with AVX2
#'   vectorisation enabled.
#'
#' @return \code{TRUE} if AVX2 paths are compiled in, \code{FALSE} otherwise.
#'
#' @examples
#' has_avx2()
#'
#' @export
has_avx2 <- function() {
    cpp_has_avx2()
}
