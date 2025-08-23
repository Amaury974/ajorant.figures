
.onLoad <- function(libname, pkgname) {
  if (getOption("ggplot.change.default.options", TRUE)) {
    if (requireNamespace("ggplot2", quietly = TRUE)) {
      ggplot2::theme_set(ggplot2::theme_light())
      options(ggplot2.continuous.colour="viridis", ggplot2.continuous.fill = "viridis")
    }
  }

  if (getOption("dplyr.change.default.options", TRUE)) {
    if (requireNamespace("dplyr", quietly = TRUE)) {
      options(dplyr.summarise.inform = FALSE)
    }
  }


}
