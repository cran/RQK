test_that("RQK returns expected structure and predict dispatch works", {
  set.seed(1)
  X <- matrix(runif(120), ncol = 4)
  y <- sin(X[, 1]) + X[, 2]^2

  fit <- RQK(X, y, theta_fit = FALSE, theta = 0.3, noise = FALSE)

  expect_s3_class(fit, "RQK")
  expect_true(all(c("w", "theta", "LOO_pred", "scores", "X_train", "y_train", "p") %in% names(fit)))
  expect_equal(length(fit$w), ncol(X))
  expect_null(dim(fit$w))
  expect_equal(length(fit$LOO_pred), nrow(X))
  expect_true(all(is.finite(fit$LOO_pred)))

  pred <- predict(fit, X[1:5, , drop = FALSE], alpha = 0.1)

  expect_true(all(c("mean", "lower", "upper", "intervals") %in% names(pred)))
  expect_equal(length(pred$mean), 5)
  expect_equal(length(pred$lower), 5)
  expect_equal(length(pred$upper), 5)
  expect_length(pred$intervals, 5)
  expect_true(all(is.finite(pred$mean)))
})

test_that("initial_lengthscale returns u and w as vectors", {
  set.seed(12)
  X <- matrix(runif(120), ncol = 4)
  y <- sin(X[, 1]) + X[, 2]^2

  init <- initial_lengthscale(X, y, theta_fit = FALSE, theta = 0.3)

  expect_equal(length(init$u), ncol(X))
  expect_equal(length(init$w), ncol(X))
  expect_null(dim(init$u))
  expect_null(dim(init$w))
})

test_that("LOO_pred matches the stored LOO score construction", {
  set.seed(11)
  X <- matrix(runif(120), ncol = 4)
  y <- sin(X[, 1]) + X[, 2]^2 - X[, 3]

  fit <- RQK(X, y, theta_fit = FALSE, theta = 0.2, noise = FALSE)

  theta2q <- fit$theta^(2 * fit$q)
  theta4q <- theta2q^2
  Rd_loo <- fit$Rd - diag(fit$R) * fit$d
  s <- sqrt(pmax(0, theta4q / ((Rd_loo^2) * fit$dInv)))

  expect_equal(
    fit$scores,
    abs(y - fit$LOO_pred) / s,
    tolerance = 1e-10
  )
})

test_that("input validation errors", {
  X <- matrix(runif(20), ncol = 2)
  y <- runif(10)

  expect_error(RQK(X, y, theta_fit = FALSE), "theta must be provided")
  expect_error(RQK(X, y[-1]), "nrow\\(X\\) must equal length\\(y\\)")
  expect_error(RQK(X, y, theta_bound = c(1, 0.5)), "theta_bound\\[1\\] <= theta_bound\\[2\\]")
  expect_error(RQK(X, y, theta_bound = c(0.1, NA_real_)), "two finite positive values")

  fit <- RQK(X, y, theta_fit = FALSE, theta = 0.2)
  expect_error(predict(fit, X, alpha = 1), "alpha")
})

test_that("theta_bound is respected when fitting theta", {
  set.seed(2)
  X <- matrix(runif(160), ncol = 4)
  y <- cos(X[, 1]) + X[, 2]

  fit <- RQK(X, y, theta_fit = TRUE, theta_bound = c(0.05, 0.2), noise = FALSE)
  expect_gte(fit$theta, 0.05 - 1e-12)
  expect_lte(fit$theta, 0.2 + 1e-12)

  expect_error(
    RQK(X, y, theta = 0.5, theta_fit = TRUE, theta_bound = c(0.05, 0.2), noise = FALSE),
    "theta must lie within theta_bound"
  )
})

test_that("noisy RQK clamps oversized kNN requests to available rows", {
  set.seed(3)
  X <- matrix(runif(32), ncol = 4)
  y <- sin(X[, 1]) + X[, 2]^2

  fit <- RQK(X, y, theta_fit = FALSE, theta = 0.2, noise = TRUE, nugget = 1e-4)
  pred <- predict(fit, X[1:3, , drop = FALSE])

  expect_s3_class(fit, "RQK")
  expect_equal(length(fit$w), ncol(X))
  expect_equal(length(pred$mean), 3)
  expect_true(all(is.finite(pred$mean)))
})

test_that("nugget defaults and fixed values behave as expected", {
  set.seed(4)
  X <- matrix(runif(80), ncol = 4)
  y <- sin(X[, 1]) + X[, 2]^2

  fit_default_noiseless <- RQK(X, y, theta_fit = FALSE, theta = 0.2, noise = FALSE)
  expect_equal(fit_default_noiseless$nugget, 1e-8)

  fit_fixed_noisy <- RQK(X, y, theta_fit = TRUE, noise = TRUE, nugget = 1e-4)
  expect_equal(fit_fixed_noisy$nugget, 1e-4, tolerance = 1e-12)

  fit_estimated_noisy <- RQK(X, y, theta_fit = TRUE, noise = TRUE, nugget = NULL)
  expect_true(is.finite(fit_estimated_noisy$nugget))
  expect_gt(fit_estimated_noisy$nugget, 0)
})

test_that("theta fitting tolerates roundoff-limited local optimization", {
  func <- function(x) {
    as.numeric(
      2 * sin(0.9 * x) +
        0.15 * (x - 5)^2 -
        1 +
        (x > 2.5 & x < 3.5) * 0.5 * sin(20 * x)
    )
  }

  X <- matrix(
    c(0.6, 1.5, 4, 5, 6.5, seq(2.5, 3.6, length.out = 10)),
    ncol = 1
  )
  y <- func(X)

  fit <- RQK(X, y, theta_fit = TRUE, noise = FALSE)
  pred <- predict(fit, X)

  expect_s3_class(fit, "RQK")
  expect_true(is.finite(fit$theta))
  expect_true(all(is.finite(pred$mean)))
  expect_equal(length(pred$mean), nrow(X))
})

test_that("interval_type exposes all four variants", {
  set.seed(5)
  X <- matrix(runif(120), ncol = 4)
  y <- sin(X[, 1]) + X[, 2]^2 - X[, 3] * X[, 4]

  fit <- RQK(X, y, theta_fit = FALSE, theta = 0.25, noise = FALSE)
  X_new <- X[1:6, , drop = FALSE]

  pred_default <- predict(fit, X_new, alpha = 0.1)
  pred_type1 <- predict(fit, X_new, alpha = 0.1, interval_type = 1L)
  pred_type2 <- predict(fit, X_new, alpha = 0.1, interval_type = 2L)
  pred_type3 <- predict(fit, X_new, alpha = 0.1, interval_type = 3L)
  pred_type4 <- predict(fit, X_new, alpha = 0.1, interval_type = 4L)
  pred_none <- predict(fit, X_new, alpha = 0.1, interval_type = "none")

  expect_equal(pred_default$lower, pred_type4$lower)
  expect_equal(pred_default$upper, pred_type4$upper)
  expect_named(pred_none, "mean")
  expect_equal(pred_none$mean, pred_default$mean)

  expect_true(all(vapply(pred_type1$intervals, nrow, integer(1)) == 1L))
  expect_true(all(vapply(pred_type3$intervals, nrow, integer(1)) == 1L))
  expect_true(all(vapply(pred_type4$intervals, nrow, integer(1)) == 1L))

  expect_true(all(is.na(pred_type2$lower)))
  expect_true(all(is.na(pred_type2$upper)))
  expect_true(all(vapply(pred_type2$intervals, is.matrix, logical(1))))
  expect_true(all(vapply(pred_type2$intervals, ncol, integer(1)) == 2L))
  expect_true(all(vapply(pred_type2$intervals, function(interval_matrix) {
    if (!nrow(interval_matrix)) {
      return(TRUE)
    }
    all(interval_matrix[, 1] <= interval_matrix[, 2]) &&
      all(diff(interval_matrix[, 1]) >= 0) &&
      all(diff(interval_matrix[, 2]) >= 0) &&
      all(interval_matrix[-1, 1] > interval_matrix[-nrow(interval_matrix), 2])
  }, logical(1))))

  expect_true(all(vapply(seq_along(pred_type2$intervals), function(i) {
    interval_matrix <- pred_type2$intervals[[i]]
    hull_matrix <- pred_type3$intervals[[i]]
    if (!nrow(interval_matrix)) {
      return(!nrow(hull_matrix))
    }
    nrow(hull_matrix) == 1L &&
      isTRUE(all.equal(hull_matrix[1, 1], interval_matrix[1, 1], tolerance = 1e-10)) &&
      isTRUE(all.equal(
        hull_matrix[1, 2],
        interval_matrix[nrow(interval_matrix), 2],
        tolerance = 1e-10
      ))
  }, logical(1))))

  expect_true(all(vapply(seq_along(pred_type2$intervals), function(i) {
    interval_matrix <- pred_type2$intervals[[i]]
    if (!nrow(interval_matrix)) {
      return(TRUE)
    }
    pred_type4$lower[i] <= interval_matrix[1, 1] + 1e-10 &&
      pred_type4$upper[i] >= interval_matrix[nrow(interval_matrix), 2] - 1e-10
  }, logical(1))))
})
