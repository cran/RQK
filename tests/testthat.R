if (!requireNamespace("testthat", quietly = TRUE)) {
  quit(save = "no", status = 0)
}

library(testthat)
library(RQK)

test_check("RQK")
