#' Fit a rational quadratic kriging model
#'
#' @description
#' Fits the rational quadratic kriging (RQK) model using compiled C++ routines
#' for factor-weight estimation, leave-one-out tuning, and conformal prediction.
#'
#' The implementation is based on \pkg{RcppArmadillo}. It is recommended to use a
#' high performance BLAS library for optimized speed.
#'
#' @param X Numeric input matrix with `n` rows and `p` columns.
#' @param y Numeric response vector of length `n`.
#' @param theta Optional positive kernel length scale. Must be supplied when `theta_fit = FALSE`. The supplied value is used as the initial value when `theta_fit = TRUE`.
#' @param theta_fit Logical; if `TRUE`, optimize `theta` by leave-one-out loss.
#' @param noise Logical; if `TRUE`, fit the noisy RQK variant with an estimated
#'   nugget term.
#' @param nugget Optional nugget term. If supplied, it is kept fixed in both
#'   noiseless and noisy fits. If `NULL`, noiseless fits use `1e-8`, whereas
#'   noisy fits estimate the nugget by line search and local optimization.
#' @param q Optional positive shape parameter of the rational quadratic kernel.
#'   If `NULL`, the implementation uses `(p + 1) / 2` for noiseless fits and
#'   `p / 2` for noisy fits, where `p = ncol(X)`.
#' @param theta_bound Optional numeric vector `c(theta_lb, theta_ub)` giving
#'   positive lower and upper bounds for `theta`.
#' @param scale Logical; if `TRUE`, each input variable is scaled to `[0, 1]`
#'   before fitting. The training minima and maxima are saved in the returned
#'   object and reused by [predict.RQK()]. This is recommended when the input
#'   data are not already scaled to `[0, 1]`.
#'
#' @details
#' The model uses a rational quadratic kernel on a `p`-dimensional input space:
#' \deqn{
#'   R(\mathbf{x}_i, \mathbf{x}_j) =
#'   \left(
#'     1 + \frac{\|\mathbf{x}_i - \mathbf{x}_j\|_w^2}{\theta^2}
#'   \right)^{-q}
#'   + \mathrm{nugget}\,\mathbb{I}\{\mathbf{x}_i = \mathbf{x}_j\}
#' }{
#'   R(x_i, x_j) = (1 + ||x_i - x_j||_w^2 / theta^2)^(-q)
#'   + nugget * I{x_i = x_j}
#' }
#' and estimates anisotropic factor weights over the `p` coordinates. The
#' returned fit stores both user-facing quantities such as `theta`, `q`, and
#' `w`, and cached matrices used by [predict.RQK()] to evaluate point
#' predictions and conformal prediction intervals. When `scale = TRUE`, fitting
#' is performed on the scaled inputs and the returned `X_train` cache is also on
#' the scaled input scale.
#'
#' @returns An object of class `"RQK"` with elements including:
#' - `w`: estimated nonnegative factor weights summing to one.
#' - `theta`: fitted kernel length scale.
#' - `nugget`: fixed or estimated nugget term.
#' - `q`: kernel shape parameter.
#' - `LOO_pred`: leave-one-out point predictions for the training responses.
#' - `scores`: training conformity scores used by [predict.RQK()] to construct
#'   interval types 1--4.
#' - `X_train`, `y_train`: training inputs and responses.
#' - `scale`: logical flag indicating whether the inputs were scaled.
#' - `scale_min`, `scale_max`: training minima and maxima for each input
#'   variable, present when `scale = TRUE`.
#' - additional cached matrices used internally by [predict.RQK()].
#'
#' @references Song, D., and Joseph, V. R. (2026). Rational Quadratic Kriging.
#'   arXiv preprint arXiv:2610.18077. \doi{10.48550/arXiv.2610.18077}
#'
#' @examples
#' borehole <- function(xx) {
#'   lower <- c(0.05, 100, 63070, 990, 63.1, 700, 1120, 9855)
#'   upper <- c(0.15, 50000, 115600, 1110, 116, 820, 1680, 12045)
#'   xx <- xx * (upper - lower) + lower
#'   rw <- xx[1]
#'   r <- xx[2]
#'   Tu <- xx[3]
#'   Hu <- xx[4]
#'   Tl <- xx[5]
#'   Hl <- xx[6]
#'   L <- xx[7]
#'   Kw <- xx[8]
#'   frac1 <- 2 * pi * Tu * (Hu - Hl)
#'   frac2a <- 2 * L * Tu / (log(r / rw) * rw^2 * Kw)
#'   frac2b <- Tu / Tl
#'   frac2 <- log(r / rw) * (1 + frac2a + frac2b)
#'   frac1 / frac2
#' }
#'
#' set.seed(1)
#' X_train <- matrix(runif(200 * 8), ncol = 8)
#' y_train <- apply(X_train, 1, borehole)
#'
#' fit <- RQK(X_train, y_train, theta = 0.2, theta_fit = FALSE)
#' fit$theta
#' fit$q
#' head(fit$w)
#'
#' X_test <- matrix(runif(1000 * 8), ncol = 8)
#' pred <- predict(fit, X_test)
#' head(data.frame(mean = pred$mean, lower = pred$lower, upper = pred$upper))
#'
#' @export
RQK <- function(
  X,
  y,
  theta = NULL,
  theta_fit = TRUE,
  noise = FALSE,
  nugget = NULL,
  q = NULL,
  theta_bound = NULL,
  scale = FALSE
) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"

  y <- as.numeric(y)

  if (!is.numeric(X) || any(!is.finite(X))) {
    stop("X must be a finite numeric matrix.", call. = FALSE)
  }
  if (!is.numeric(y) || any(!is.finite(y))) {
    stop("y must be a finite numeric vector.", call. = FALSE)
  }
  if (nrow(X) != length(y)) {
    stop("nrow(X) must equal length(y).", call. = FALSE)
  }
  if (ncol(X) < 1L) {
    stop("X must have at least one column.", call. = FALSE)
  }
  if (!is.logical(scale) || length(scale) != 1L || is.na(scale)) {
    stop("scale must be TRUE or FALSE.", call. = FALSE)
  }
  scale_flag <- isTRUE(scale)

  if (is.null(theta)) {
    theta_val <- NA_real_
  } else {
    theta_val <- as.numeric(theta)
    if (length(theta_val) != 1L || !is.finite(theta_val) || theta_val <= 0) {
      stop(
        "theta must be a finite positive scalar when provided.",
        call. = FALSE
      )
    }
  }

  if (is.null(q)) {
    q_val <- NA_real_
  } else {
    q_val <- as.numeric(q)
    if (length(q_val) != 1L || !is.finite(q_val) || q_val <= 0) {
      stop("q must be a finite positive scalar when provided.", call. = FALSE)
    }
  }

  if (is.null(theta_bound)) {
    theta_lb_val <- NA_real_
    theta_ub_val <- NA_real_
  } else {
    theta_bound_val <- as.numeric(theta_bound)
    if (
      length(theta_bound_val) != 2L ||
        any(!is.finite(theta_bound_val)) ||
        any(theta_bound_val <= 0)
    ) {
      stop(
        "theta_bound must be a numeric vector of two finite positive values.",
        call. = FALSE
      )
    }
    if (theta_bound_val[1] > theta_bound_val[2]) {
      stop(
        "theta_bound must satisfy theta_bound[1] <= theta_bound[2].",
        call. = FALSE
      )
    }
    theta_lb_val <- theta_bound_val[1]
    theta_ub_val <- theta_bound_val[2]
  }

  if (is.null(nugget)) {
    nugget_val <- NA_real_
  } else {
    nugget_val <- as.numeric(nugget)
    if (length(nugget_val) != 1L || !is.finite(nugget_val) || nugget_val < 0) {
      stop(
        "nugget must be a finite nonnegative scalar when provided.",
        call. = FALSE
      )
    }
    if (isTRUE(noise) && nugget_val <= 0) {
      stop("nugget must be positive when noise = TRUE.", call. = FALSE)
    }
    if (isTRUE(noise) && nugget_val > 1) {
      stop("nugget must be <= 1 when noise = TRUE.", call. = FALSE)
    }
  }

  scale_min <- NULL
  scale_max <- NULL
  if (scale_flag) {
    scale_min <- apply(X, 2L, min)
    scale_max <- apply(X, 2L, max)
    scale_range <- scale_max - scale_min
    if (any(!is.finite(scale_range)) || any(scale_range <= 0)) {
      stop(
        "scale = TRUE requires every column of X to have positive range.",
        call. = FALSE
      )
    }
    X <- .rqk_scale_to_unit(X, scale_min, scale_max)
  }

  fit <- rqk_fit_cpp(
    X = X,
    y = y,
    theta = theta_val,
    theta_fit = isTRUE(theta_fit),
    noise = isTRUE(noise),
    nugget = nugget_val,
    q = q_val,
    theta_lb = theta_lb_val,
    theta_ub = theta_ub_val
  )

  fit$scale <- scale_flag
  if (scale_flag) {
    fit$scale_min <- scale_min
    fit$scale_max <- scale_max
  }

  class(fit) <- "RQK"
  fit
}

#' Obtain an initial length scale and factor weights
#'
#' Computes a starting point for RQK fitting, including the kernel length scale
#' and the anisotropic factor weights over the `p` input coordinates.
#'
#' @param X Numeric input matrix with `n` rows and `p` columns.
#' @param y Numeric response vector of length `n`.
#' @param theta Optional positive kernel length scale. Must be supplied when
#'   `theta_fit = FALSE`.
#' @param theta_fit Logical; if `TRUE`, optimize `theta` while constructing the
#'   initial point.
#' @param noise Logical; if `TRUE`, use the noisy-case default for `q`.
#' @param q Optional positive shape parameter of the rational quadratic kernel.
#'   If `NULL`, the implementation uses `(p + 1) / 2` for noiseless fits and
#'   `p / 2` for noisy fits, where `p = ncol(X)`.
#'
#' @returns A list with elements:
#' - `theta`: initial kernel length scale on the original scale.
#' - `logtheta`: the same value on the log scale.
#' - `q`: kernel shape parameter used for the initialization.
#' - `u`: unconstrained optimization variables for the factor weights.
#' - `w`: factor weights on the simplex.
#' - `par`: full optimization vector returned by the initializer.
#' - `logtheta_lb`, `logtheta_ub`: log-scale bounds used for `theta`.
#'
#' @references Song, D., and Joseph, V. R. (2026). Rational Quadratic Kriging.
#'   arXiv preprint arXiv:2610.18077. \doi{10.48550/arXiv.2610.18077}
#'
#' @examples
#' borehole <- function(xx) {
#'   lower <- c(0.05, 100, 63070, 990, 63.1, 700, 1120, 9855)
#'   upper <- c(0.15, 50000, 115600, 1110, 116, 820, 1680, 12045)
#'   xx <- xx * (upper - lower) + lower
#'   rw <- xx[1]
#'   r <- xx[2]
#'   Tu <- xx[3]
#'   Hu <- xx[4]
#'   Tl <- xx[5]
#'   Hl <- xx[6]
#'   L <- xx[7]
#'   Kw <- xx[8]
#'   frac1 <- 2 * pi * Tu * (Hu - Hl)
#'   frac2a <- 2 * L * Tu / (log(r / rw) * rw^2 * Kw)
#'   frac2b <- Tu / Tl
#'   frac2 <- log(r / rw) * (1 + frac2a + frac2b)
#'   frac1 / frac2
#' }
#'
#' set.seed(1)
#' X_train <- matrix(runif(200 * 8), ncol = 8)
#' y_train <- apply(X_train, 1, borehole)
#'
#' init <- initial_lengthscale(X_train, y_train, noise = FALSE)
#' init$theta
#' head(init$w)
#'
#' @export
initial_lengthscale <- function(
  X,
  y,
  theta = NULL,
  theta_fit = TRUE,
  noise = FALSE,
  q = NULL
) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"

  y <- as.numeric(y)

  if (!is.numeric(X) || any(!is.finite(X))) {
    stop("X must be a finite numeric matrix.", call. = FALSE)
  }
  if (!is.numeric(y) || any(!is.finite(y))) {
    stop("y must be a finite numeric vector.", call. = FALSE)
  }
  if (nrow(X) != length(y)) {
    stop("nrow(X) must equal length(y).", call. = FALSE)
  }
  if (ncol(X) < 1L) {
    stop("X must have at least one column.", call. = FALSE)
  }

  if (is.null(theta)) {
    theta_val <- NA_real_
  } else {
    theta_val <- as.numeric(theta)
    if (length(theta_val) != 1L || !is.finite(theta_val) || theta_val <= 0) {
      stop(
        "theta must be a finite positive scalar when provided.",
        call. = FALSE
      )
    }
  }

  if (is.null(q)) {
    q_val <- NA_real_
  } else {
    q_val <- as.numeric(q)
    if (length(q_val) != 1L || !is.finite(q_val) || q_val <= 0) {
      stop("q must be a finite positive scalar when provided.", call. = FALSE)
    }
  }

  initial_lengthscale_cpp(
    X = X,
    y = y,
    theta = theta_val,
    theta_fit = isTRUE(theta_fit),
    noise = isTRUE(noise),
    q = q_val
  )
}

#' Predict from a fitted RQK model
#'
#' Computes point predictions and, unless disabled, conformal prediction intervals
#' from an object returned by [RQK()].
#'
#' @param object An object returned by [RQK()].
#' @param newdata Numeric matrix or data frame of new inputs with `p` columns.
#' @param alpha Miscoverage tuning parameter in `(0, 1)`.
#' @param interval_type Integer in `1:4` selecting the construction of the prediction interval,
#'   or `"none"` to return point predictions without interval components.
#'   `1` is the smooth symmetric score-quantile interval.
#'   `2` is the conformal set obtained by taking the union of possibly disjoint intervals.
#'   `3` is the convex hull of type 2.
#'   `4` is a conservative interval guaranteed to be a superset of type 3.
#' @param ... Ignored. Included for generic compatibility.
#'
#' @details
#' If `object` was fitted with `scale = TRUE`, `newdata` is scaled to `[0, 1]`
#' using the training minima and maxima saved in the fitted object before
#' predictions are computed.
#'
#' @references Song, D., and Joseph, V. R. (2026). Rational Quadratic Kriging.
#'   arXiv preprint arXiv:2610.18077. \doi{10.48550/arXiv.2610.18077}
#'
#' @returns A list with:
#' - `mean`: point predictions.
#' If `interval_type = "none"`, this is the only returned component. Otherwise
#' the list also contains:
#' - `lower`, `upper`: numeric prediction bounds for interval types 1, 3, and 4.
#'   For type 2 they are `NA_real_` because the prediction set may be disjoint.
#' - `intervals`: a list with one element per prediction point. Each element is
#'   a two-column matrix with columns `lower` and `upper`. Type 2 may contain
#'   multiple rows, whereas types 1, 3, and 4 contain a single row.
#'
#' @examples
#' borehole <- function(xx) {
#'   lower <- c(0.05, 100, 63070, 990, 63.1, 700, 1120, 9855)
#'   upper <- c(0.15, 50000, 115600, 1110, 116, 820, 1680, 12045)
#'   xx <- xx * (upper - lower) + lower
#'   rw <- xx[1]
#'   r <- xx[2]
#'   Tu <- xx[3]
#'   Hu <- xx[4]
#'   Tl <- xx[5]
#'   Hl <- xx[6]
#'   L <- xx[7]
#'   Kw <- xx[8]
#'   frac1 <- 2 * pi * Tu * (Hu - Hl)
#'   frac2a <- 2 * L * Tu / (log(r / rw) * rw^2 * Kw)
#'   frac2b <- Tu / Tl
#'   frac2 <- log(r / rw) * (1 + frac2a + frac2b)
#'   frac1 / frac2
#' }
#'
#' set.seed(1)
#' X_train <- matrix(runif(200 * 8), ncol = 8)
#' y_train <- apply(X_train, 1, borehole)
#' fit <- RQK(X_train, y_train, theta = 0.2, theta_fit = FALSE)
#'
#' X_test <- matrix(runif(1000 * 8), ncol = 8)
#' pred <- predict(fit, X_test, alpha = 0.1)
#' head(pred$mean)
#'
#' @export
predict.RQK <- function(
  object,
  newdata,
  alpha = 0.05,
  interval_type = 4,
  ...
) {
  Xnew <- as.matrix(newdata)
  storage.mode(Xnew) <- "double"

  normalize_interval_type <- function(interval_type) {
    if (
      is.character(interval_type) &&
        length(interval_type) == 1L &&
        !is.na(interval_type) &&
        identical(unname(tolower(interval_type)), "none")
    ) {
      return(0L)
    }

    if (
      !is.numeric(interval_type) ||
        length(interval_type) != 1L ||
        !is.finite(interval_type) ||
        interval_type != as.integer(interval_type) ||
        !(as.integer(interval_type) %in% 1:4)
    ) {
      stop(
        "interval_type must be \"none\" or an integer in 1:4.",
        call. = FALSE
      )
    }

    unname(as.integer(interval_type))
  }

  if (!is.numeric(Xnew) || any(!is.finite(Xnew))) {
    stop("newdata must be a finite numeric matrix.", call. = FALSE)
  }
  fit_has_scale_bounds <- all(c("scale_min", "scale_max") %in% names(object))
  fit_uses_scale <- isTRUE(object[["scale"]]) || fit_has_scale_bounds
  if (fit_uses_scale) {
    if (!fit_has_scale_bounds) {
      stop(
        "scaled RQK objects must contain scale_min and scale_max.",
        call. = FALSE
      )
    }
    scale_min <- object[["scale_min"]]
    scale_max <- object[["scale_max"]]
    if (
      !is.numeric(scale_min) ||
        !is.numeric(scale_max) ||
        length(scale_min) != ncol(Xnew) ||
        length(scale_max) != ncol(Xnew) ||
        any(!is.finite(scale_min)) ||
        any(!is.finite(scale_max)) ||
        any(scale_max <= scale_min)
    ) {
      stop(
        "object$scale_min and object$scale_max must be finite numeric vectors ",
        "matching newdata columns, with scale_max > scale_min.",
        call. = FALSE
      )
    }
    Xnew <- .rqk_scale_to_unit(Xnew, scale_min, scale_max)
  }
  if (
    !is.numeric(alpha) ||
      length(alpha) != 1L ||
      !is.finite(alpha) ||
      alpha <= 0 ||
      alpha >= 1
  ) {
    stop("alpha must be a scalar in (0, 1).", call. = FALSE)
  }
  interval_type <- normalize_interval_type(interval_type)

  score_quantile <- NA_real_
  if (identical(interval_type, 1L)) {
    score_values <- object[["scores"]]
    if (
      !is.numeric(score_values) ||
        !length(score_values) ||
        any(!is.finite(score_values))
    ) {
      stop("object$scores must be a finite numeric vector.", call. = FALSE)
    }
    score_quantile <- as.numeric(
      stats::quantile(
        score_values,
        probs = 1 - alpha,
        type = 5,
        names = FALSE
      )
    )
  }

  rqk_predict_cpp(
    fit = object,
    X = Xnew,
    alpha = alpha,
    interval_type = interval_type,
    score_quantile = score_quantile
  )
}

.rqk_scale_to_unit <- function(X, scale_min, scale_max) {
  scale_range <- scale_max - scale_min
  sweep(sweep(X, 2L, scale_min, "-"), 2L, scale_range, "/")
}
