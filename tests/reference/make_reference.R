##########################################################################################
# REFERENCE OUTPUTS FROM rwf FOR pwf PARITY TESTS
##########################################################################################
# Runs rwf functions on the same CSVs pwf uses (pwf/data) and writes their results
# next to this script as CSV. tests/test_*.py compare the Python ports against them.
#
# Run from anywhere (needs rwf installed in R):
#   Rscript tests/reference/make_reference.R
#   make reference RSCRIPT="C:/Program Files/R/R-4.6.2/bin/Rscript.exe"
##########################################################################################
suppressPackageStartupMessages(library(rwf))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
out_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
data_dir <- normalizePath(file.path(out_dir, "..", "..", "data"))

load_data <- function(name) {
  utils::read.csv(file.path(data_dir, paste0(name, ".csv")), stringsAsFactors = FALSE)
}
ref <- function(name, x) {
  # binary connection: on Windows a text connection turns newlines inside values into \r\n
  con <- file(file.path(out_dir, paste0(name, ".csv")), open = "wb")
  on.exit(close(con))
  utils::write.csv(x, con, row.names = FALSE)
  cat("wrote", name, "\n")
}

df_blood_pressure <- load_data("blood_pressure")
df_insurance <- load_data("insurance")
# blood pressure with a few missing values, to check NA handling
df_blood_pressure_na <- df_blood_pressure
df_blood_pressure_na$bp_before[c(3, 50, 99)] <- NA
df_blood_pressure_na$agegrp[c(10, 70)] <- NA
##########################################################################################
# EXPLORE_DESCRIPTIVES
##########################################################################################
df_ocean <- utils::read.delim(file.path(data_dir, "ocean.csv"), stringsAsFactors = FALSE)
df_ocean_N <- df_ocean[, grep("^N", names(df_ocean))]
df_responses <- load_data("responses")
ref("compute_descriptives", compute_descriptives(df = df_insurance, dv = c(1, 3, 7)))
ref("compute_descriptives_iv", compute_descriptives(df = df_insurance, dv = c(1, 7), iv = c(2, 4)))
ref("compute_descriptives_na", compute_descriptives(df = df_blood_pressure_na, dv = 4:5, iv = 3))
ref("compute_aggregate", compute_aggregate(df = df_insurance, iv = c(2, 5)))
ref("compute_aggregate_na", compute_aggregate(df = df_blood_pressure_na[, -1], iv = 2))
ref("compute_aggregate_numeric_iv", compute_aggregate(df = df_insurance[, c("age", "children", "charges")], iv = 2))
ref("compute_frequencies", compute_frequencies(df = df_insurance[, c(2, 4, 5, 6)]))
ref("compute_frequencies_na", compute_frequencies(df = df_blood_pressure_na[, c(2, 3)]))
ref("response_frequency", response_frequency(df_ocean_N))
for (type in c("frequency", "proportion", "percent", "all")) {
  ref(paste0("response_frequency_", type), response_frequency(df_ocean_N, uniqueitems = 1:5, type = type))
}
ref("response_frequency_responses_na", response_frequency(df_responses[, 2:12], uniqueitems = 1:5, type = "all"))
ref("compute_crosstable", compute_crosstable(df = df_insurance, factor_index = c(2, 5, 6)))
ref("compute_crosstable_combinations", compute_crosstable(
  df = df_insurance, combinations = data.frame(index1 = c("sex", "smoker"), index2 = c("region", "children"))))
ref("compute_crosstable_na", compute_crosstable(df = df_blood_pressure_na, factor_index = 2:3))
crosstable_plots <- plot_crosstable(df = df_insurance, factor_index = c(2, 5, 6))
ref("plot_crosstable_names", data.frame(name = names(crosstable_plots)))
ref("plot_crosstable_region_sex", crosstable_plots[["region_sex"]]$data)
mosaic_plots <- plot_mosaic(df = df_insurance, factor_index = c(2, 5, 6))
ref("plot_mosaic_names", data.frame(name = names(mosaic_plots)))
ref("plot_mosaic_region_smoker", mosaic_plots[["region smoker"]]$data)
mosaic_na <- plot_mosaic(df = df_blood_pressure_na, factor_index = 2:3)
ref("plot_mosaic_na_agegrp_sex", mosaic_na[["agegrp sex"]]$data)
response_plots <- plot_response_frequencies(df = df_ocean_N, factor_index = 1:2)
ref("plot_response_frequencies_N1", response_plots[["N1"]]$data)
ref("plot_response_frequencies_titles", data.frame(name = names(response_plots),
  title = sapply(response_plots, function(p) p$labels$title),
  caption = sapply(response_plots, function(p) p$labels$caption)))
grDevices::pdf(NULL)  # plot_* functions print their plots
##########################################################################################
# EXPLORE_ASSUMPTIONS
##########################################################################################
for (method in c("mean", "median", "boxplot")) {
  outlier_plots <- plot_outlier(df = df_blood_pressure_na[, c("bp_before", "bp_after")], method = method, title = "bp")
  ref(paste0("plot_outlier_", method), outlier_plots[["bp_before"]]$data)
  ref(paste0("plot_outlier_", method, "_labels"), data.frame(
    name = names(outlier_plots),
    x = sapply(outlier_plots, function(p) p$labels$x),
    y = sapply(outlier_plots, function(p) p$labels$y),
    caption = sapply(outlier_plots, function(p) p$labels$caption),
    hlines = sapply(outlier_plots, function(p) paste(sapply(p$layers[3:5], function(l) format(l$data$yintercept, digits = 15)), collapse = ";"))))
}
# report_normality_tests returns NULL; a copy that returns its result table gives the exact values
report_normality_tests_table <- rwf::report_normality_tests
body(report_normality_tests_table) <- as.call(c(as.list(body(report_normality_tests_table)), quote(result_df)))
environment(report_normality_tests_table) <- asNamespace("rwf")
df_normality <- data.frame(df_insurance[, c("age", "bmi", "charges")],
                           bp = c(df_blood_pressure$bp_before, rep(NA, 1218)),
                           few = c(1:5, rep(NA, 1333)), constant = 1)
invisible(capture.output(normality <- suppressWarnings(report_normality_tests_table(df = df_normality))))
ref("report_normality_tests", normality)
##########################################################################################
# FUNCTIONS_CHECK_DATAFRAME
##########################################################################################
# titanic with one column of every type; rebuilt identically in tests/test_functions_cdf.py
df_titanic <- load_data("titanic")
df_types <- df_titanic[, c("survived", "pclass", "name", "sex", "age", "fare", "cabin", "body")]
df_types$sex <- factor(df_types$sex)
df_types$class <- factor(c("first", "second", "third")[df_types$pclass],
                         levels = c("first", "second", "third"), ordered = TRUE)
df_types$alive <- df_types$survived == 1
df_types$alive[c(5, 9)] <- NA
df_types$date <- as.Date("2020-01-01") + seq_len(nrow(df_types)) - 1
df_types$fare[1:2] <- c(Inf, -Inf)
for (case in list(list(name = "cdff_types", args = list()),
                  list(name = "cdff_types_nuniques", args = list(nuniques = 3)),
                  list(name = "cdff_head_nuniques", args = list(nuniques = 25, name_length = 5), rows = 20))) {
  data <- if (is.null(case$rows)) df_types else df_types[seq_len(case$rows), ]
  result <- do.call(cdff, c(list(df = data), case$args))
  ref(paste0(case$name, "_check"), result$check)
  ref(paste0(case$name, "_summary"), result$summary)
}
##########################################################################################
# FUNCTIONS_TRAIN_TEST
##########################################################################################
confusion_cases <- list(
  perfect = list(observed = c(0, 0, 0, 1, 1, 1, 1), predicted = c(0, 0, 0, 1, 1, 1, 1)),
  positive_0 = list(observed = c(0, 0, 0, 1, 1, 1, 1), predicted = c(0, 0, 0, 1, 1, 1, 0), positive = 0),
  positive_1 = list(observed = c(0, 0, 0, 1, 1, 1, 1), predicted = c(0, 0, 0, 1, 1, 1, 0), positive = 1),
  majority = list(observed = c(rep("no", 40), rep("yes", 10)),
                  predicted = c(rep("no", 35), rep("yes", 5), rep("no", 3), rep("yes", 7))),
  no_positive_predictions = list(observed = c("a", "a", "b", "b"), predicted = c("a", "a", "a", "a"))
)
for (case in names(confusion_cases)) {
  p <- do.call(plot_confusion_extended, confusion_cases[[case]])
  ref(paste0("plot_confusion_extended_", case), p$data)
}
##########################################################################################
# GLM_ANOVA
##########################################################################################
ref("compute_kruskal_wallis_test",
    compute_kruskal_wallis_test(formula = bp_before ~ agegrp, df = df_blood_pressure))
ref("compute_kruskal_wallis_test_na",
    compute_kruskal_wallis_test(formula = bp_before ~ agegrp, df = df_blood_pressure_na))
set.seed(1)
ref("compute_kruskal_wallis_test_ci",
    compute_kruskal_wallis_test(formula = bp_before ~ agegrp, df = df_blood_pressure, ci = TRUE, nboot = 5000))
##########################################################################################
# GLM_MEANS
##########################################################################################
ref("compute_cohens_d", data.frame(
  formula = c("bp_before ~ sex", "bp_after ~ sex", "charges ~ smoker"),
  d = c(compute_cohens_d(bp_before ~ sex, df_blood_pressure),
        compute_cohens_d(bp_after ~ sex, df_blood_pressure),
        compute_cohens_d(charges ~ smoker, df_insurance))))
bp_two_groups <- df_blood_pressure[df_blood_pressure$agegrp %in% c("30-45", "46-59"), ]
children_4_5 <- df_insurance[df_insurance$children %in% c(4, 5), ]
ref("compute_wilcoxon_effect_size", data.frame(
  case = c("bp_before_agegrp", "bp_before_agegrp_nocorrect", "charges_children_4_5"),
  r = c(compute_wilcoxon_effect_size(bp_before ~ agegrp, bp_two_groups),
        compute_wilcoxon_effect_size(bp_before ~ agegrp, bp_two_groups, correct = FALSE),
        compute_wilcoxon_effect_size(charges ~ children, children_4_5))))
for (alt in c("two.sided", "less", "greater")) {
  ref(paste0("report_ttests_", alt),
      report_ttests(df = df_blood_pressure, dv = 4:5, iv = 2:3, alternative = alt))
  ref(paste0("report_wtests_", alt),
      report_wtests(df = df_blood_pressure, dv = 4:5, iv = 2:3, alternative = alt))
}
ref("report_ttests_var_equal",
    report_ttests(df = df_blood_pressure, dv = 4, iv = 2:3, var.equal = TRUE))
ref("report_ttests_na", report_ttests(df = df_blood_pressure_na, dv = 4, iv = 3))
ref("report_wtests_na", report_wtests(df = df_blood_pressure_na, dv = 4, iv = 3))
ref("report_ttests_insurance", report_ttests(df = df_insurance, dv = 7, iv = c(2, 4)))
ref("report_wtests_insurance", report_wtests(df = df_insurance, dv = 7, iv = c(2, 4)))
ref("report_wtests_insurance_less", report_wtests(df = df_insurance, dv = 7, iv = 4, alternative = "less"))
