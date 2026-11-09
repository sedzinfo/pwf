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
  utils::write.csv(x, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
  cat("wrote", name, "\n")
}

df_blood_pressure <- load_data("blood_pressure")
df_insurance <- load_data("insurance")
# blood pressure with a few missing values, to check NA handling
df_blood_pressure_na <- df_blood_pressure
df_blood_pressure_na$bp_before[c(3, 50, 99)] <- NA
df_blood_pressure_na$agegrp[c(10, 70)] <- NA
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
