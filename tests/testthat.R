library(testthat)
library(fastroll)

if (dir.exists("tests/testthat")) {
  test_dir("tests/testthat")
} else {
  test_check("fastroll")
}
