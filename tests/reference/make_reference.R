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
grDevices::pdf(NULL)  # rwf's plot_* functions print their plots; draw them nowhere, not into Rplots.pdf

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
out_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
data_dir <- normalizePath(file.path(out_dir, "..", "..", "src", "pwf", "data"))

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
df_co2 <- load_data("co2")
for (var_equal in c(TRUE, FALSE)) {
  ref(paste0("compute_one_way_test_", var_equal),
      compute_one_way_test(bp_before ~ agegrp, df = df_blood_pressure, var.equal = var_equal))
  ref(paste0("compute_one_way_test_na_", var_equal),
      compute_one_way_test(bp_before ~ agegrp, df = df_blood_pressure_na, var.equal = var_equal))
  ref(paste0("compute_one_way_test_region_", var_equal),
      compute_one_way_test(charges ~ region, df = df_insurance, var.equal = var_equal))
}
for (case in list(list("bp", df_blood_pressure$bp_before, df_blood_pressure$agegrp),
                  list("region", df_insurance$charges, df_insurance$region))) {
  post_hoc <- compute_posthoc(case[[2]], case[[3]])$output
  ref(paste0("compute_posthoc_", case[[1]], "_tukey"), data.frame(pair = rownames(post_hoc$tukey), post_hoc$tukey))
  ref(paste0("compute_posthoc_", case[[1]], "_games_howell"),
      data.frame(pair = rownames(post_hoc$games.howell), post_hoc$games.howell))
}
ref("compute_friedman_test", compute_friedman_test(uptake ~ conc | Plant, df = df_co2))
df_co2_missing <- df_co2[-c(3, 40), ]
ref("compute_friedman_test_incomplete", compute_friedman_test(uptake ~ conc | Plant, df = df_co2_missing))
set.seed(1)
ref("compute_friedman_test_ci", compute_friedman_test(uptake ~ conc | Plant, df = df_co2, ci = TRUE, nboot = 5000))
# factorial designs: 36 participants, B1 and B2 between (4 per cell), W1 and W2 within
set.seed(12345)
df_factorial <- expand.grid(W1 = c("A", "B", "C"), W2 = c("D", "E", "F"), id = 1:36, stringsAsFactors = FALSE)
df_factorial$B1 <- c("G", "H", "I")[(df_factorial$id - 1) %% 3 + 1]
df_factorial$B2 <- c("J", "K", "L")[(df_factorial$id - 1) %/% 3 %% 3 + 1]
df_factorial <- df_factorial[, c("id", "B1", "B2", "W1", "W2")]
df_factorial$DV1 <- round(rnorm(nrow(df_factorial), 10), 3) + match(df_factorial$W1, c("A", "B", "C")) +
  match(df_factorial$B1, c("G", "H", "I"))
df_factorial$DV2 <- round(rnorm(nrow(df_factorial), 10), 3) + (df_factorial$W2 == "F") * 0.8
ref("factorial_data", df_factorial)
factorial_cases <- list(
  between = list(between = c("B1", "B2")),
  between_unadjusted = list(between = c("B1", "B2"), white.adjust = FALSE),
  within = list(within = c("W1", "W2"), within_full = c("W1", "W2")),
  within_unadjusted = list(within = c("W1", "W2"), white.adjust = FALSE),
  mixed = list(within = "W1", between = "B1"),
  mixed_2x2 = list(within = c("W1", "W2"), between = c("B1", "B2")),
  mixed_observed = list(within = "W1", between = "B1", white.adjust = FALSE, observed = "B1")
)
for (case in names(factorial_cases)) {
  result <- suppressMessages(suppressWarnings(do.call(report_factorial_anova,
    c(list(df = df_factorial, wid = "id", dv = c("DV1", "DV2")), factorial_cases[[case]]))))
  ref(paste0("report_factorial_anova_", case, "_omnibus"), result$omnibus)
  ref(paste0("report_factorial_anova_", case, "_effect_size"), result$omnibus_effect_size)
  ref(paste0("report_factorial_anova_", case, "_post_hoc"), result$post_hoc)
}
unbalanced <- df_factorial[!df_factorial$id %in% c(1, 2, 14), ]
result <- suppressMessages(suppressWarnings(report_factorial_anova(df = unbalanced, wid = "id", dv = "DV1",
                                                                    within = "W1", between = "B1")))
ref("report_factorial_anova_unbalanced_omnibus", result$omnibus)
ref("report_factorial_anova_unbalanced_effect_size", result$omnibus_effect_size)
ref("report_factorial_anova_unbalanced_post_hoc", result$post_hoc)
old_contrasts <- options(contrasts = c("contr.sum", "contr.poly"))
manova_cases <- list(
  bp = manova(cbind(bp_before, bp_after) ~ sex * agegrp, data = df_blood_pressure),
  insurance = manova(cbind(age, bmi, charges) ~ smoker * region, data = df_insurance)
)
for (case in names(manova_cases)) {
  invisible(capture.output(manova_result <- report_manova(model = manova_cases[[case]])))
  ref(paste0("report_manova_", case, "_multivariate"), manova_result$multivariate)
  type_three <- manova_result$type_three
  pillai <- t(sapply(type_three$terms, function(term) {
    eig <- Re(eigen(solve(type_three$SSPE) %*% type_three$SSP[[term]])$values)
    stats:::Pillai(eig, type_three$df[term], type_three$error.df)
  }))
  colnames(pillai) <- c("test stat", "approx F", "num Df", "den Df")
  ref(paste0("report_manova_", case, "_type_three"), data.frame(term = type_three$terms, Df = type_three$df, pillai,
    "Pr(>F)" = stats::pf(pillai[, 2], pillai[, 3], pillai[, 4], lower.tail = FALSE), check.names = FALSE))
}
for (ss in c("I", "II", "III")) {
  ref(paste0("compute_aov_es_oneway_", ss), compute_aov_es(aov(uptake ~ Treatment, data = df_co2), ss = ss))
  ref(paste0("compute_aov_es_factorial_", ss), compute_aov_es(aov(uptake ~ Treatment * Type, data = df_co2), ss = ss))
  ref(paste0("compute_aov_es_unbalanced_", ss), compute_aov_es(aov(charges ~ smoker * region, data = df_insurance), ss = ss))
}
options(old_contrasts)
oneway <- report_oneway(df = df_blood_pressure_na, dv = 4:5, iv = 2:3)
for (table in c("fisher", "welch", "kruskal_wallis", "games_howell", "tukey", "homogeneity")) {
  ref(paste0("report_oneway_", table), oneway[[table]])
}
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
##########################################################################################
# ML_NLP
##########################################################################################
nlp_texts <- c(
  "word_one word_two word_three",
  "word_three word_four word_six",
  "All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet.",
  "It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable.",
  "The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc.",
  "I'm sure it's   2024 -- they don't know (yet) what a b c means; café & naïve — résumé!",
  "There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable."
)
ref("nlp_texts", data.frame(text = nlp_texts))
ref("clear_text", data.frame(text = clear_text(nlp_texts)))
ref("clear_stopwords", data.frame(text = clear_stopwords(nlp_texts)))
nlp_words <- strsplit(nlp_texts, split = " ")
similarity <- do.call(rbind, lapply(seq_along(nlp_words), function(j) {
  data.frame(i = 1, j = j, compute_text_similarity(nlp_words[[1]], nlp_words[[j]]))
}))
similarity <- rbind(similarity, data.frame(i = 3, j = 4, compute_text_similarity(nlp_words[[3]], nlp_words[[4]])),
                    data.frame(i = 5, j = 7, compute_text_similarity(nlp_words[[5]], nlp_words[[7]])))
ref("compute_text_similarity", similarity)
tversky_cases <- list(list(c("a", "b", "c", "d"), c("b", "c", "d", "e"), 0.5, 0.5),
                      list(c("a", "b", "c", "d"), c("b", "c", "d", "e"), 1, 1),
                      list(c("a", "b", "c", "d"), c("b", "c", "d", "e"), 0.9, 0.1),
                      list(c("a", "a", "b"), c("b", "b", "c"), 0.5, 0.5),
                      list(c(1, 2, 3), c("2", "3", "4"), 0.5, 0.5),
                      list(c("a", "b"), c("c", "d"), 0.5, 0.5))
ref("compute_tversky_index", data.frame(case = seq_along(tversky_cases), value = sapply(tversky_cases, function(k)
  compute_tversky_index(k[[1]], k[[2]], alpha = k[[3]], beta = k[[4]]))))
ref("stat_word_char", stat_word_char(nlp_texts))
tagged <- tag_pos(nlp_texts[3:5])
ref("tag_pos", data.frame(tag = tagged$POStags))
##########################################################################################
# ML_XGBOOST
##########################################################################################
multi_trees_tables <- function(model, features_keep) {
  `:=` <- data.table::`:=`
  tree.matrix <- xgboost::xgb.model.dt.tree(model = model)
  data.table::setDT(tree.matrix)
  root.nodes <- tree.matrix[tree.matrix$Node == 0, ]$ID
  tree.matrix$abs.node.position <- NA_character_
  tree.matrix$abs.node.position[tree.matrix$ID %in% root.nodes] <- tree.matrix$ID[tree.matrix$ID %in% root.nodes]
  precedent.nodes <- root.nodes
  while (sum(is.na(tree.matrix$abs.node.position)) > 0) {
    yes.rows <- tree.matrix[tree.matrix$abs.node.position %in% precedent.nodes & !is.na(tree.matrix$Yes), ]
    no.rows <- tree.matrix[tree.matrix$abs.node.position %in% precedent.nodes & !is.na(tree.matrix$No), ]
    yes.pos <- paste0(yes.rows$abs.node.position, "_0")
    no.pos <- paste0(no.rows$abs.node.position, "_1")
    tree.matrix$abs.node.position[match(yes.rows$Yes, tree.matrix$ID)] <- yes.pos
    tree.matrix$abs.node.position[match(no.rows$No, tree.matrix$ID)] <- no.pos
    precedent.nodes <- c(yes.pos, no.pos)
  }
  tree.matrix$Yes[!is.na(tree.matrix$Yes)] <- paste0(tree.matrix$abs.node.position[!is.na(tree.matrix$Yes)], "_0")
  tree.matrix$No[!is.na(tree.matrix$No)] <- paste0(tree.matrix$abs.node.position[!is.na(tree.matrix$No)], "_1")
  for (nm in c("abs.node.position", "Yes", "No")) tree.matrix[[nm]] <- sub("^[0-9]+-", "", tree.matrix[[nm]])
  gains <- tree.matrix[, list(Gain = sum(Gain)), by = list(abs.node.position, Feature)]
  nodes <- gains[, list(Text = paste0(paste0(Feature[seq_len(min(length(Feature), features_keep))], " (",
    format(Gain[seq_len(min(length(Gain), features_keep))], digits = 5), ")"), collapse = "\n")),
    by = abs.node.position]
  splits <- tree.matrix[tree.matrix$Feature != "Leaf", ]
  edges <- unique(rbind(data.frame(From = splits$abs.node.position, To = splits$Yes),
                        data.frame(From = splits$abs.node.position, To = splits$No)))
  list(nodes = as.data.frame(nodes), edges = edges)
}
# models trained here and saved as xgboost JSON, so pwf analyses the very same trees
df_admission <- load_data("admission")
xgb_cases <- list(
  classification = list(data = df_admission, label = "admit", features = c("gre", "gpa", "rank"),
                        params = list(objective = "binary:logistic", nthread = 1, max_depth = 3, eta = 0.3)),
  regression = list(data = df_insurance, label = "charges", features = c("age", "bmi", "children"),
                    params = list(objective = "reg:squarederror", nthread = 1, max_depth = 3, eta = 0.3))
)
for (case in names(xgb_cases)) {
  k <- xgb_cases[[case]]
  train <- k$data[1:300, ]
  test <- k$data[301:400, ]
  dtrain <- xgboost::xgb.DMatrix(as.matrix(train[, k$features]), label = train[[k$label]])
  dtest <- xgboost::xgb.DMatrix(as.matrix(test[, k$features]), label = test[[k$label]])
  set.seed(1)
  model <- xgboost::xgb.train(params = k$params, data = dtrain, nrounds = 15,
                              evals = list(train = dtrain, eval = dtest), verbose = 0)
  xgboost::xgb.save(model, file.path(out_dir, paste0("xgb_", case, ".json")))
  report <- report_xgboost(model = model, validation_data = test, label = k$label, file = NULL)
  ref(paste0("report_xgboost_", case, "_importance"), report$result$importance)
  ref(paste0("report_xgboost_", case, "_model_summary"), report$result$model_summary)
  ref(paste0("report_xgboost_", case, "_validation_metrics"), report$result$validation_metrics)
  ref(paste0("report_xgboost_", case, "_evaluation_log"), report$result$evaluation_log)
  ref(paste0("report_xgboost_", case, "_predicted"), data.frame(observed = report$observed, predicted = report$predicted))
  ref(paste0("report_xgboost_", case, "_depth"), as.data.frame(report$plots$depth$data))
  # xgb.plot.multi.trees needs DiagrammeR only to draw; this is its own code up to that point
  multi <- multi_trees_tables(model, features_keep = 10)
  ref(paste0("plot_trees_xgboost_", case, "_nodes"), multi$nodes)
  ref(paste0("plot_trees_xgboost_", case, "_edges"), multi$edges)
}
##########################################################################################
# LAVAAN ENGINE (pwf's Python CFA/SEM estimator is checked against lavaan itself)
##########################################################################################
df_ocean_500 <- df_ocean[1:500, c(paste0("E", 1:5), paste0("N", 1:5), paste0("A", 1:3))]
ref("lavaan_data", df_ocean_500)
lavaan_cases <- list(
  cfa2 = list(model = "E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", std.lv = FALSE),
  cfa2_stdlv = list(model = "E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", std.lv = TRUE),
  sem = list(model = "E =~ E1 + E2 + 0.8*E3 + E4; N =~ N1 + N2 + N3 + N4; A =~ A1 + A2 + A3; A ~ E + N; E1 ~~ E2",
             std.lv = FALSE)
)
for (case in names(lavaan_cases)) {
  k <- lavaan_cases[[case]]
  fit <- lavaan::sem(k$model, data = df_ocean_500, std.lv = k$std.lv)
  pe <- lavaan::parameterEstimates(fit, standardized = TRUE, rsquare = TRUE)
  ref(paste0("lavaan_", case, "_parameters"), pe)
  fm <- lavaan::fitMeasures(fit)
  ref(paste0("lavaan_", case, "_fit"), data.frame(measure = names(fm), value = as.numeric(fm)))
}
for (case in c("cfa2", "sem")) {
  k <- lavaan_cases[[case]]
  fit <- lavaan::sem(k$model, data = df_ocean_500, std.lv = k$std.lv)
  mi <- lavaan::modificationIndices(fit, standardized = TRUE, cov.std = TRUE, information = "expected", power = TRUE,
                                    delta = 0.1, alpha = 0.05, high.power = 0.75, sort. = TRUE, minimum.value = 0,
                                    free.remove = FALSE, na.remove = TRUE, op = NULL)
  ref(paste0("lavaan_", case, "_modindices"), data.frame(mi))
  ref(paste0("lavaan_", case, "_predict"), data.frame(lavaan::lavPredict(fit)))
}
fit <- lavaan::sem(lavaan_cases$sem$model, data = df_ocean_500)
for (what in c("est", "std")) {
  matrices <- lavaan::inspect(fit, what)
  for (name in c("lambda", "theta", "psi", "beta")) {
    ref(paste0("lavaan_sem_inspect_", what, "_", name), data.frame(unclass(matrices[[name]]), check.names = FALSE))
  }
}
# FIML: the same data with about 5% of the values removed at random
set.seed(2024)
df_ocean_missing <- df_ocean_500[1:300, c(paste0("E", 1:5), paste0("N", 1:5))]
holes <- matrix(runif(prod(dim(df_ocean_missing))) < 0.05, nrow(df_ocean_missing))
df_ocean_missing[holes] <- NA
df_ocean_missing[7, ] <- NA  # a respondent without any answer
ref("lavaan_fiml_data", df_ocean_missing)
fit <- lavaan::cfa("E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", data = df_ocean_missing,
                   missing = "ML", meanstructure = TRUE)
ref("lavaan_fiml_parameters", lavaan::parameterEstimates(fit, standardized = TRUE, rsquare = TRUE))
fm <- lavaan::fitMeasures(fit)
ref("lavaan_fiml_fit", data.frame(measure = names(fm), value = as.numeric(fm)))
scores <- matrix(NA_real_, nrow(df_ocean_missing), 2)
scores[lavaan::lavInspect(fit, "case.idx"), ] <- lavaan::lavPredict(fit)
ref("lavaan_fiml_predict", data.frame(E = scores[, 1], N = scores[, 2]))
ref("lavaan_fiml_info", data.frame(information = fit@Options$information[1], h1 = fit@Options$h1.information[1],
                                   ntotal = lavaan::lavInspect(fit, "ntotal"), npar = lavaan::lavInspect(fit, "npar")))
mi <- lavaan::modificationIndices(fit, standardized = TRUE, cov.std = TRUE, information = "expected", power = TRUE,
                                  delta = 0.1, alpha = 0.05, high.power = 0.75, sort. = TRUE, minimum.value = 0,
                                  free.remove = FALSE, na.remove = TRUE, op = NULL)
ref("lavaan_fiml_modindices", data.frame(mi))
##########################################################################################
# PSYCH EFA ENGINE (pwf's Python EFA is checked against psych::fa itself)
##########################################################################################
efa_data <- list(complete = df_ocean_500[, c(paste0("E", 1:5), paste0("N", 1:5))], missing = df_ocean_missing)
for (data_case in names(efa_data)) {
  for (nfactors in 1:3) {
    model <- psych::fa(efa_data[[data_case]], nfactors = nfactors, rotate = "oblimin", fm = "ml", residuals = TRUE)
    prefix <- paste0("psych_fa_", data_case, "_", nfactors)
    ref(paste0(prefix, "_loadings"), data.frame(variable = rownames(model$loadings), unclass(model$loadings), check.names = FALSE))
    ref(paste0(prefix, "_structure"), data.frame(variable = rownames(model$Structure), unclass(model$Structure), check.names = FALSE))
    if (!is.null(model$Phi)) ref(paste0(prefix, "_phi"), data.frame(unclass(model$Phi), check.names = FALSE))
    ref(paste0(prefix, "_residual"), data.frame(unclass(model$residual), check.names = FALSE))
    stats_out <- c(RMSEA = unname(model$RMSEA[1]), RMSEA.lower = unname(model$RMSEA[2]), RMSEA.upper = unname(model$RMSEA[3]),
                   RMSEA.conf = unname(model$RMSEA[4]), rms = model$rms, CFI = if (is.null(model$CFI)) NA else model$CFI,
                   TLI = model$TLI, STATISTIC = model$STATISTIC, dof = model$dof, objective = model$objective,
                   null.chisq = model$null.chisq, null.dof = model$null.dof, n.obs = model$n.obs)
    ref(paste0(prefix, "_stats"), data.frame(stat = names(stats_out), value = unname(stats_out)))
    ref(paste0(prefix, "_model_loadings"), model_loadings(model, matrix_type = "pattern", cut = 0.001, sort = TRUE))
  }
}
##########################################################################################
# GLM_VALIDITY
##########################################################################################
efa_items <- c(paste0("E", 1:10), paste0("N", 1:10))
efa_reverse <- c("E2", "E4", "E6", "E8", "E10", "N2", "N4")
df_efa <- df_ocean[1:400, efa_items]
df_efa[df_efa == 0] <- NA
ref("efa_cfa_data", df_efa)
efa_dimensions <- data.frame(dimension = substr(efa_items, 1, 1), subquestion_text = efa_items,
                             key = ifelse(efa_items %in% efa_reverse, 1, -1))
efa_result <- suppressMessages(suppressWarnings(compute_efa_cfa(dimensions = efa_dimensions, df_response = df_efa,
                                                                filename = file.path(tempdir(), "efa_cfa.xlsx"))))
ref("compute_efa_cfa_models", efa_result$efa_cfa_models)
ref("compute_efa_cfa_scores", efa_result$cfa_scores)
ref("compute_efa_cfa_coefficients", efa_result$cfa_coefficients)
##########################################################################################
# GLM_RELIABILITY (variance components)
##########################################################################################
set.seed(42)
vc_design <- expand.grid(time = 1:3, item = 1:3, person = 1:10)
vc_design$response <- vc_design$time + vc_design$item + rnorm(10, 0, 1)[vc_design$person] + rnorm(90, 0, 0.5)
ref("vc_design", vc_design)
vc_factors <- change_data_type(vc_design, type = "factor")
vc_factors$response <- vc_design$response
vc_model <- mixlm::lm(response ~ r(time) * r(person) + r(item) * r(person), data = vc_factors)
vc_result <- extract_components(vc_model)
ref("extract_components", vc_result$components)
vc_anova <- mixlm::Anova(vc_model, type = "III")
ref("extract_components_anova", data.frame(term = rownames(vc_anova$anova), as.data.frame(vc_anova$anova), check.names = FALSE))
##########################################################################################
# GLM_EFA (psych::fa with minres and ml, and rwf::report_efa)
##########################################################################################
for (data_case in names(efa_data)) {
  for (fm in c("minres", "ml")) {
    for (nfactors in 1:3) {
      model <- psych::fa(efa_data[[data_case]], nfactors = nfactors, rotate = "oblimin", fm = fm)
      prefix <- paste0("psych_fa_full_", data_case, "_", fm, "_", nfactors)
      ref(paste0(prefix, "_loadings"), data.frame(unclass(model$loadings), check.names = FALSE))
      ref(paste0(prefix, "_items"), data.frame(communality = model$communality, communalities = model$communalities,
                                               uniquenesses = model$uniquenesses, complexity = model$complexity,
                                               values = model$values, e.values = model$e.values))
      ref(paste0(prefix, "_weights"), data.frame(unclass(model$weights), check.names = FALSE))
      ref(paste0(prefix, "_scores"), data.frame(model$scores, check.names = FALSE))
      ref(paste0(prefix, "_vaccounted"), data.frame(unclass(model$Vaccounted), check.names = FALSE))
      stats_out <- c(crms = model$crms, BIC = model$BIC, SABIC = model$SABIC, EBIC = model$EBIC, ESABIC = model$ESABIC,
                     fit = model$fit, fit.off = model$fit.off, rms = model$rms, TLI = model$TLI, CFI = model$CFI,
                     RMSEA = unname(model$RMSEA[1]), STATISTIC = model$STATISTIC)
      ref(paste0(prefix, "_stats"), data.frame(stat = names(stats_out), value = unname(stats_out)))
    }
  }
}
for (data_case in names(efa_data)) {
  model <- psych::fa(efa_data[[data_case]], nfactors = 2, rotate = "oblimin", fm = "minres")
  report <- report_efa(model, efa_data[[data_case]])
  for (part in c("residual_stats", "determinant_test", "bartlett_test", "kmo_test", "loadings", "correlations")) {
    ref(paste0("report_efa_", data_case, "_", part), report[[part]])
  }
}
##########################################################################################
# TETRACHORIC POLYCHORIC POLYSERIAL BISERIAL (psych, through rwf::report_choric_serial)
##########################################################################################
choric_items <- c(paste0("E", 1:5), paste0("N", 1:5))
choric_data <- list(
  complete = df_ocean_500[, choric_items],
  missing = df_ocean_missing,
  small = df_ocean_500[1:40, choric_items],
  unequal = within(df_ocean_500[, choric_items], N5 <- pmin(N5, 3))
)
for (case in names(choric_data)) {
  items <- choric_data[[case]]
  tet <- report_choric_serial((items > 3) * 1, type = "tetrachoric")
  ref(paste0("tetrachoric_", case, "_rho"), data.frame(tet$rho, check.names = FALSE))
  ref(paste0("tetrachoric_", case, "_tau"), data.frame(item = names(tet$tau), tau = as.numeric(tet$tau)))
  pol <- suppressWarnings(report_choric_serial(items, type = "polychoric"))
  ref(paste0("polychoric_", case, "_rho"), data.frame(pol$rho, check.names = FALSE))
  ref(paste0("polychoric_", case, "_tau"), data.frame(pol$tau, check.names = FALSE))
}
serial_x <- data.frame(E = rowSums(df_ocean_500[, paste0("E", 1:5)]), A = rowSums(df_ocean_500[, paste0("A", 1:3)]))
ref("polyserial_data", cbind(serial_x, df_ocean_500[, paste0("N", 1:3)]))
ref("polyserial", report_choric_serial(serial_x, df_ocean_500[, paste0("N", 1:3)], type = "polyserial"))
ref("biserial", report_choric_serial(serial_x, (df_ocean_500[, paste0("N", 1:3)] > 3) * 1, type = "biserial"))
##########################################################################################
# REPORT CORRELATION (psych::corr.test through rwf)
##########################################################################################
correlation_cases <- list(
  complete = list(x = df_insurance[, c("age", "bmi", "children", "charges")], y = NULL, use = "pairwise",
                  method = "pearson", adjust = "holm"),
  missing = list(x = df_ocean_missing[, 1:5], y = NULL, use = "pairwise", method = "pearson", adjust = "holm"),
  spearman = list(x = df_ocean_missing[, 1:5], y = NULL, use = "pairwise", method = "spearman", adjust = "BH"),
  kendall = list(x = df_ocean_missing[, 1:5], y = NULL, use = "complete", method = "kendall", adjust = "none"),
  xy = list(x = df_ocean_missing[, 1:3], y = df_ocean_missing[, 6:8], use = "pairwise", method = "pearson",
            adjust = "hommel")
)
for (case in names(correlation_cases)) {
  k <- correlation_cases[[case]]
  result <- report_correlation(k$x, k$y, use = k$use, method = k$method, adjust = k$adjust, scatterplot = FALSE)
  for (part in setdiff(names(result), "ci")) ref(paste0("report_correlation_", case, "_", part), result[[part]])
  ref(paste0("report_correlation_", case, "_ci"), data.frame(pair = rownames(result$ci), result$ci))
}
##########################################################################################
# REPORT REGRESSION
##########################################################################################
regression_cases <- list(simple = charges ~ age, mixed = charges ~ age * bmi + smoker + region,
                         additive = charges ~ age + bmi + children)
for (case in names(regression_cases)) {
  model <- eval(bquote(lm(.(regression_cases[[case]]), data = df_insurance)))
  set.seed(1)
  result <- report_regression(model, plot_diagnostics = FALSE)
  for (part in c("r", "coeficients", "deviance", "durbin_watson", "vif", "call")) {
    ref(paste0("report_regression_", case, "_", part), result[[part]])
  }
  ref(paste0("report_regression_", case, "_anova"), data.frame(term = rownames(result$anova), result$anova,
                                                                 check.names = FALSE))
  ref(paste0("report_regression_", case, "_variance_covariance"), result$variance_covariance)
  ref(paste0("report_regression_", case, "_outlier_test"), data.frame(observation = rownames(result$outlier_test),
                                                                        result$outlier_test))
  ref(paste0("report_regression_", case, "_diagnostics"), result$diagnostics)
}
##########################################################################################
# REPORT LOGISTIC (rwf::report_logistic returns NULL: its body is rerun returning its result)
##########################################################################################
infert_data <- datasets::infert[, c("education", "age", "parity", "induced", "case", "spontaneous")]
infert_data$education <- as.character(infert_data$education)
ref("infert", infert_data)
infert_data <- utils::read.csv(file.path(out_dir, "infert.csv"), stringsAsFactors = FALSE)
report_logistic_result <- rwf::report_logistic
body(report_logistic_result) <- as.call(c(as.list(body(report_logistic_result)), quote(
  return(list(result = result, cmatrix = cmatrix, confusion_performance = confusion_performance)))))
environment(report_logistic_result) <- asNamespace("rwf")
logistic_cases <- list(
  education = list(formula = case ~ education, family = "binomial", validation = FALSE),
  numeric = list(formula = case ~ age + parity + spontaneous, family = "binomial", validation = FALSE),
  mixed = list(formula = case ~ education + age, family = "binomial", validation = FALSE),
  validation = list(formula = case ~ age + parity + spontaneous, family = "binomial", validation = TRUE),
  poisson = list(formula = spontaneous ~ age + parity, family = "poisson", validation = FALSE),
  gaussian = list(formula = age ~ parity + induced, family = "gaussian", validation = FALSE)
)
for (case in names(logistic_cases)) {
  k <- logistic_cases[[case]]
  model <- eval(bquote(glm(.(k$formula), family = .(as.name(k$family)), data = infert_data)))
  out <- report_logistic_result(model, validation_data = if (k$validation) infert_data else NULL)
  result <- out$result
  for (part in c("result_R2_logistic", "result_X2_logistic", "logistic_output", "scores", "model_call")) {
    ref(paste0("report_logistic_", case, "_", part), result[[part]])
  }
  ref(paste0("report_logistic_", case, "_coefficients"), data.frame(term = rownames(result$coefficients),
                                                                    result$coefficients, check.names = FALSE))
  ref(paste0("report_logistic_", case, "_VIF"), data.frame(term = rownames(result$VIF), result$VIF,
                                                           check.names = FALSE))
  ref(paste0("report_logistic_", case, "_score_descriptives"), result$score_descriptives)
  ref(paste0("report_logistic_", case, "_cmatrix"), data.frame(out$cmatrix, check.names = FALSE))
  ref(paste0("report_logistic_", case, "_cut_performance"), out$confusion_performance$cut_performance)
  ref(paste0("report_logistic_", case, "_cut"), data.frame(out$confusion_performance$cut))
}
##########################################################################################
# TRAIN TEST (confusion tables, cut search, kappas, dummy coding, ROC)
##########################################################################################
train_test_cases <- list(
  numbers = list(observed = c(1, 2, 3, 4, 5, 10), predicted = c(1, 2, 3, 4, 5, 11)),
  ordinal = list(observed = c(1, 2, 2, 2, 2, 3, 3, 1, 2), predicted = c(1, 1, 2, 2, 2, 3, 2, 2, 3)),
  labels = list(observed = c("no", "no", "yes", "yes", "no"), predicted = c("no", "yes", "yes", "no", "no"))
)
for (case in names(train_test_cases)) {
  k <- train_test_cases[[case]]
  ref(paste0("confusion_matrix_percent_", case), data.frame(row = rownames(confusion_matrix_percent(k$observed, k$predicted)),
                                                            confusion_matrix_percent(k$observed, k$predicted),
                                                            check.names = FALSE))
  ref(paste0("proportion_accurate_", case), proportion_accurate(k$observed, k$predicted))
}
set.seed(11)
cut_observed <- rbinom(150, 1, 0.4)
cut_predicted <- round(pmin(1, cut_observed * 0.3 + runif(150, 0, 0.7)), 6)
ref("confusion_performance_data", data.frame(observed = cut_observed, predicted = cut_predicted))
for (step in c(0.1, 0.05)) {
  cp <- result_confusion_performance(cut_observed, cut_predicted, step = step)
  ref(paste0("confusion_performance_", step * 100, "_cut_performance"), cp$cut_performance)
  ref(paste0("confusion_performance_", step * 100, "_cut"), data.frame(cut = cp$cut))
}
roc_rows <- list()
for (levels in list(c(1, 0), c(0, 1))) {
  roc <- pROC::roc(cut_observed, cut_predicted, levels = levels, quiet = TRUE)
  roc_rows[[length(roc_rows) + 1]] <- data.frame(control = levels[1], auc = as.numeric(roc$auc), direction = roc$direction)
}
roc_rows[[length(roc_rows) + 1]] <- with(list(roc = pROC::roc(cut_observed, 1 - cut_predicted, levels = c(0, 1), quiet = TRUE)),
                                         data.frame(control = -1, auc = as.numeric(roc$auc), direction = roc$direction))
ref("plot_roc_auc", do.call(rbind, roc_rows))
dummy_data <- data.frame(numeric_var = c(1, 2, 3, 4, 5), n2 = c(5, 3, 3, 9, 1),
                         factor_var = c("B", "A", "A", "B", "C"), other = c("x", "y", "x", "x", "x"))
ref("recode_scale_dummy_data", dummy_data)
ref("recode_scale_dummy_two", recode_scale_dummy(dummy_data))
ref("recode_scale_dummy_one", recode_scale_dummy(dummy_data[, 1:3]))
##########################################################################################
# REPORT HLR
##########################################################################################
sleepstudy <- data.frame(lme4::sleepstudy)
sleepstudy$Subject <- as.character(sleepstudy$Subject)
sleepstudy$Fatigue <- round(sleepstudy$Reaction / 50 + sin(seq_len(nrow(sleepstudy))), 4)
ref("sleepstudy", sleepstudy)
ref("report_hlr_sleepstudy", report_hlr(df = sleepstudy, corlist = c(1, 4), factorlist = 3, predictor = "Days",
                                        random_effect = "Subject"))
# as rwf's example, grouping by the predictor: the random-effect variances are 0 at the maximum
ref("report_hlr_infert", report_hlr(df = utils::read.csv(file.path(out_dir, "infert.csv"), stringsAsFactors = FALSE),
                                    corlist = c(2, 3, 4, 6), factorlist = 1, predictor = "case",
                                    random_effect = "case"))
##########################################################################################
# REPORT LDA (MASS::lda)
##########################################################################################
ref("iris", datasets::iris)
iris_data <- utils::read.csv(file.path(out_dir, "iris.csv"), stringsAsFactors = FALSE)
infert_data <- utils::read.csv(file.path(out_dir, "infert.csv"), stringsAsFactors = FALSE)
lda_models <- list(infert = MASS::lda(case ~ ., data = infert_data),
                   iris = MASS::lda(Species ~ ., data = iris_data),
                   iris_two = MASS::lda(Species ~ Sepal.Length + Petal.Width, data = iris_data))
for (case in names(lda_models)) {
  result <- report_lda(lda_models[[case]])
  ref(paste0("report_lda_", case, "_prior_counts"), data.frame(class = rownames(result$prior_counts),
                                                               result$prior_counts, check.names = FALSE))
  ref(paste0("report_lda_", case, "_coeficients"), data.frame(variable = rownames(result$coeficients),
                                                              result$coeficients, check.names = FALSE))
  ref(paste0("report_lda_", case, "_model_description"), result$model_description)
  ref(paste0("report_lda_", case, "_cmatrix"), data.frame(row = rownames(result$cmatrix), result$cmatrix,
                                                          check.names = FALSE))
  ref(paste0("report_lda_", case, "_call"), result$call)
  ref(paste0("report_lda_", case, "_posterior"), data.frame(predict(lda_models[[case]])$posterior, check.names = FALSE))
}
##########################################################################################
# TIME SERIES
##########################################################################################
lh_values <- as.numeric(datasets::lh)
ref("lh", data.frame(value = lh_values))
ref("plot_ts_lh", plot_ts(lh_values)$data)
acf_plot <- plot_acf(lh_values)
ref("plot_acf_lh", acf_plot$data)
ref("plot_acf_lh_ci", do.call(rbind, lapply(c("correlation", "covariance", "partial"), function(type) {
  values <- stats::acf(lh_values, lag.max = length(lh_values), plot = FALSE, type = type)$acf
  ci <- Rmisc::CI(values, ci = 0.95)
  data.frame(type = type, upper = ci[[1]], mean = ci[[2]], lower = ci[[3]])
})))
ref("plot_acf_lh_10", plot_acf(lh_values, lag.max = 10)$data)
moving <- data.frame(a = lh_values, b = lh_values^2)
ref("compute_moving_average_lh", compute_moving_average(moving, 2))
##########################################################################################
# STATISTICAL HELPERS
##########################################################################################
standard_vector <- c(3, 7, 1, 7, NA, 12, 5, 9, 2, 7, 15, 4)
ref("compute_standard_all", compute_standard(standard_vector, type = "all"))
ref("compute_standard_types", data.frame(sapply(c("z", "uz", "sten", "t", "stanine", "center", "center_reversed",
                                                  "percent", "percentile", "scale_zero_one", "normal_density",
                                                  "cumulative_density"),
                                                function(type) compute_standard(standard_vector, mean = 2, sd = 3,
                                                                                type = type))))
set.seed(3)
noise <- data.frame(v1 = rnorm(50), e1 = rnorm(50, sd = 0.5), v2 = rnorm(50), e2 = rnorm(50, sd = 0.3))
noise$v2 <- noise$v2 + noise$v1
ref("dissatenuation_data", noise)
ref("statistical_helpers", data.frame(
  dissatenuation = compute_dissatenuation(noise$v1, noise$e1, noise$v2, noise$e2),
  skewness = compute_skewness(standard_vector), kurtosis = compute_kurtosis(standard_vector),
  standard_error = compute_standard_error(standard_vector),
  confidence_interval = compute_confidence_inteval(standard_vector),
  sidak = compute_adjustment(0.05, 7)$sidak, bonferroni = compute_adjustment(0.05, 7)$bonferroni))
##########################################################################################
# STRING HELPERS
##########################################################################################
words <- c("_hello world_", "TES.T", "a  b\tc", "mixed-Case_text", "x", "Long sentence that needs wrapping at twelve")
ref("strings_vectorised", data.frame(
  input = words, proper = str_proper(words), aes = str_aes(words), aes_plain = str_aes(words, proper = FALSE),
  right = str_sub(words, 3, "right"), left = str_sub(words, 3, "left"), squish = str_squish(words),
  wrap = str_wrap(words, 12), pad_left = str_pad(words, 8, "left", "*"), pad_both = str_pad(words, 9, "both", "-"),
  count_e = str_count(words, "e"), count_fixed = str_count(words, fixed(".")),
  replace = str_replace(words, "[aeiou]", "#"), replace_all = str_replace_all(words, "[aeiou]", "#"),
  replace_fixed = str_replace_all(words, fixed("."), "!"),
  mgsub = str_mgsub(words, c("[.]", "_", "-"), "+")))
ref("str_split", str_split(c("a/b/c", "d/e/f"), include_original = TRUE))
ref("str_split_fixed", data.frame(str_split_fixed(c("a-b-c-d", "e-f", "g"), "-", 3)))
##########################################################################################
# CDF (the original, slower check of rwf)
##########################################################################################
df_titanic <- load_data("titanic")
df_types <- df_titanic[, c("survived", "pclass", "name", "sex", "age", "fare", "cabin", "body")]
df_types$sex <- factor(df_types$sex)
df_types$class <- factor(c("first", "second", "third")[df_types$pclass],
                         levels = c("first", "second", "third"), ordered = TRUE)
df_types$alive <- df_types$survived == 1
df_types$alive[c(5, 9)] <- NA
df_types$date <- as.Date("2020-01-01") + seq_len(nrow(df_types)) - 1
df_types$fare[1:2] <- c(Inf, -Inf)
df_types$empty <- NA_real_
for (case in list(list(name = "cdf_types", args = list()),
                  list(name = "cdf_types_nuniques", args = list(nuniques = 3)),
                  list(name = "cdf_head_nuniques", args = list(nuniques = 25, name_length = 5), rows = 20))) {
  data <- if (is.null(case$rows)) df_types else df_types[seq_len(case$rows), ]
  result <- suppressWarnings(do.call(cdf, c(list(df = data), case$args)))
  ref(paste0(case$name, "_check"), result$check)
  ref(paste0(case$name, "_summary"), result$summary)
}
##########################################################################################
# ANOVA PLOTS (data behind the plots)
##########################################################################################
oneway <- plot_oneway(df = df_insurance, dv = c(3, 7), iv = c(2, 5, 6))
ref("plot_oneway_plot_data_df", oneway$plot_data_df)
ref("plot_oneway_keys", data.frame(key = names(oneway$plot_data)))
interaction <- plot_interaction(df = df_insurance, dv = 7, iv = c(2, 5, 6))
ref("plot_interaction_plot_data_df", interaction$plot_data_df)
ref("plot_interaction_keys", data.frame(key = names(interaction$plot_data)))
##########################################################################################
# LAVAAN WLSMV (ordered indicators)
##########################################################################################
wlsmv_items <- c(paste0("E", 1:5), paste0("N", 1:5))
wlsmv_data <- list(ordinal = df_ocean_500[, wlsmv_items],
                   binary = data.frame(lapply(df_ocean_500[, wlsmv_items], function(x) as.integer(x > 3))))
ref("wlsmv_binary_data", wlsmv_data$binary)
wlsmv_model <- "E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5"
for (case in names(wlsmv_data)) {
  for (parameterization in c("delta", "theta")) {
    fit <- lavaan::cfa(wlsmv_model, data = wlsmv_data[[case]], ordered = wlsmv_items, estimator = "WLSMV",
                       parameterization = parameterization)
    prefix <- paste0("wlsmv_", case, "_", parameterization)
    ref(paste0(prefix, "_parameters"), lavaan::parameterEstimates(fit, standardized = TRUE))
    fm <- lavaan::fitMeasures(fit)
    ref(paste0(prefix, "_fit"), data.frame(measure = names(fm), value = as.numeric(fm)))
    if (parameterization == "delta") {
      sampstat <- lavaan::lavInspect(fit, "sampstat")
      ref(paste0("wlsmv_", case, "_th"), data.frame(th = as.numeric(sampstat$th), name = names(sampstat$th)))
      ref(paste0("wlsmv_", case, "_cor"), data.frame(unclass(sampstat$cov), check.names = FALSE))
      ref(paste0("wlsmv_", case, "_gamma"), data.frame(unclass(lavaan::lavInspect(fit, "gamma")), check.names = FALSE))
      ref(paste0("wlsmv_", case, "_wls_v"), data.frame(w = diag(lavaan::lavInspect(fit, "wls.v"))))
    }
  }
}
##########################################################################################
# THURSTONIAN IRT (thurstonianIRT::fit_TIRT_lavaan on its triplets data, rwf's examples)
##########################################################################################
suppressPackageStartupMessages(library(thurstonianIRT))
tirt_env <- new.env()
utils::data("triplets", package = "thurstonianIRT", envir = tirt_env)
triplets <- tirt_env$triplets
ref("tirt_triplets", triplets)
tirt_blocks <- set_block(c("i1", "i2", "i3"), traits = c("t1", "t2", "t3"), signs = c(1, 1, 1)) +
  set_block(c("i4", "i5", "i6"), traits = c("t1", "t2", "t3"), signs = c(-1, 1, 1)) +
  set_block(c("i7", "i8", "i9"), traits = c("t1", "t2", "t3"), signs = c(1, 1, -1)) +
  set_block(c("i10", "i11", "i12"), traits = c("t1", "t2", "t3"), signs = c(1, -1, 1))
triplets_long <- make_TIRT_data(data = triplets, blocks = tirt_blocks, direction = "larger", format = "pairwise",
                                family = "bernoulli", range = c(0, 1))
writeLines(thurstonianIRT:::make_lavaan_code(triplets_long), file.path(out_dir, "tirt_lavaan_model.txt"))
for (estimator in c("ULSMV", "WLSMV")) {
  tirt_fit <- fit_TIRT_lavaan(triplets_long, estimator = estimator)
  prefix <- paste0("tirt_", tolower(estimator))
  ref(paste0(prefix, "_parameters"), lavaan::parameterEstimates(tirt_fit$fit))
  fm <- lavaan::fitMeasures(tirt_fit$fit)
  ref(paste0(prefix, "_fit"), data.frame(measure = names(fm), value = as.numeric(fm)))
  pars <- extract_tirt_params(tirt_fit)
  ref(paste0(prefix, "_lambda"), data.frame(indicator = rownames(pars$lambda), unclass(pars$lambda), check.names = FALSE))
  ref(paste0(prefix, "_vectors"), data.frame(indicator = names(pars$theta_diag), theta_diag = as.numeric(pars$theta_diag),
                                             tau = as.numeric(pars$tau), nu = as.numeric(pars$nu)))
  ref(paste0(prefix, "_psi"), data.frame(unclass(pars$Psi), check.names = FALSE))
  heywood <- check_heywood(tirt_fit$fit, verbose = FALSE)
  ref(paste0(prefix, "_heywood"), data.frame(item = names(heywood), value = vapply(heywood, function(x)
    if (is.data.frame(x)) nrow(x) else if (is.logical(x)) as.numeric(x) else NA_real_, numeric(1))))
}
##########################################################################################
# DATASETS (how R reads pwf's CSVs)
##########################################################################################
dataset_summary <- do.call(rbind, lapply(c("admission", "automotive_data", "blood_pressure", "co2", "crop_yield",
                                           "difficile", "insurance", "ocean", "personality", "responses",
                                           "responses_state", "sexual_comp", "titanic"), function(name) {
  path <- file.path(data_dir, paste0(name, ".csv"))
  df <- if (name == "ocean") utils::read.delim(path, stringsAsFactors = FALSE) else utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  data.frame(dataset = name, column = names(df), numeric = vapply(df, is.numeric, logical(1)),
             missing = vapply(df, function(x) sum(is.na(x)), numeric(1)),
             total = vapply(df, function(x) if (is.numeric(x)) sum(x, na.rm = TRUE) else NA_real_, numeric(1)),
             row.names = NULL)
}))
ref("datasets_summary", dataset_summary)
##########################################################################################
# UTILITY MODULES (recode, keys, unix time, matrix, mathematical, generate data, IRT U)
##########################################################################################

matrix_df <- function(m) {
  # a matrix as a data frame: rownames (if any) in a "row" column, then the cells
  d <- as.data.frame(m, stringsAsFactors = FALSE)
  names(d) <- if (is.null(colnames(m))) paste0("V", seq_len(ncol(m))) else colnames(m)
  if (!is.null(rownames(m))) d <- cbind(row = rownames(m), d)
  d
}
vector_df <- function(...) {
  data.frame(..., stringsAsFactors = FALSE, check.names = FALSE)
}

##########################################################################################
# FUNCTIONS_RECODE
##########################################################################################
ref("recode_dummy_arrange_text", dummy_arrange(c("Agree,Hi", "All", "Hi,All,Agree", "Hi")))
ref("recode_dummy_arrange_numeric", dummy_arrange(c(1, 3, 2, 4, 1)))
ref("recode_dummy_arrange_case", dummy_arrange(c("b", "B", "a", "A")))
ref("recode_dummy_arrange_spaces", dummy_arrange(c("A, B", "B", "")))
ref("recode_dummy_arrange_na", dummy_arrange(c("A,B", NA, "C")))
ref("recode_dummy_arrange_sort", dummy_arrange(c("2,10", "1", "10,3")))
# one response value: R drops the one-column data frame to a vector
ref("recode_dummy_arrange_single", vector_df(value = dummy_arrange(c("x", "x"))))

ref("recode_flatten_list_named", flatten_list(list(a = list(x = 1:2, y = c("p", "q")), b = list(x = 3:4))))
ref("recode_flatten_list_unnamed", flatten_list(list(list(x = 1:2), list(x = 3:4, z = 5))))
ref("recode_flatten_list_vectors", flatten_list(list(a = 1:2, b = 3:4)))
ref("recode_flatten_list_frames", flatten_list(list(a = data.frame(m = 1, n = "z"), b = data.frame(m = 2))))

ref("recode_swap_numeric", vector_df(input = c(1:10, 1, 2, 3), output = swap(c(1:10, 1, 2, 3))))
ref("recode_swap_na", vector_df(input = c(1L, 3L, 3L, NA, 2L), output = swap(c(1L, 3L, 3L, NA, 2L))))
ref("recode_swap_text", vector_df(input = c("b", "B", "a", "A", "b"), output = swap(c("b", "B", "a", "A", "b"))))
ref("recode_swap_decimal", vector_df(input = c(1.5, 10, 2), output = swap(c(1.5, 10, 2))))

drop_levels_df <- function(d) {
  # factors as text plus their levels joined by "|"
  out <- list()
  for (n in names(d)) {
    out[[n]] <- as.character(d[[n]])
    if (is.factor(d[[n]])) out[[paste0(n, "_levels")]] <- rep(paste(levels(d[[n]]), collapse = "|"), nrow(d))
  }
  data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}
df_levels <- data.frame(
  numeric1 = 1:20,
  factor1 = factor(c(rep("A", 10), rep("B", 10)), levels = c("A", "B", "C", "D")),
  factor2 = factor(c(rep("A", 3), rep("B", 17)))
)
ref("recode_drop_levels_9", drop_levels_df(drop_levels(df_levels, minimum_frequency = 9)))
ref("recode_drop_levels_10", drop_levels_df(drop_levels(df_levels, minimum_frequency = 10)))
ref("recode_drop_levels_index", drop_levels_df(drop_levels(df_levels, factor_index = 3, minimum_frequency = 3)))
df_levels_other <- data.frame(g = factor(c("Other", "x", "x", "y")), h = c("a", "a", "b", "c"))
ref("recode_drop_levels_other", drop_levels_df(drop_levels(df_levels_other, minimum_frequency = 1)))
ref("recode_drop_levels_character", drop_levels_df(drop_levels(df_levels_other, factor_index = 2, minimum_frequency = 1)))

##########################################################################################
# FUNCTIONS_KEYS
##########################################################################################
keys_df <- function(key_list) {
  do.call(rbind, lapply(seq_along(key_list), function(i) {
    data.frame(dimension = rep(i, length(key_list[[i]])), question = key_list[[i]])
  }))
}
ref("keys_questions_by_keys", keys_df(questions_by_keys(c(1, 2, 3, 4, 5, 1, 2, 3, 4, 5))))
ref("keys_questions_by_keys_uneven", keys_df(questions_by_keys(c(3, 1, 1, 2, 3, 3))))
ref("keys_questions_by_keys_no_one", data.frame(lengths = lengths(questions_by_keys(c(2, 2, 3)))))
ref("keys_questions_dimensions_dataframe",
    questions_dimensions_dataframe(c(3, 1, 1, 2, 3, 3), paste0("Dimension", 1:3),
                                   paste0("Elaborated_Dimension", 1:3), paste0("Question", 1:6)))

##########################################################################################
# FUNCTIONS_UNIX_TIME
##########################################################################################
unix_df <- function(d) {
  # julian() is a difftime: store its number of days
  if ("JULIAN" %in% names(d)) d$JULIAN <- as.numeric(d$JULIAN)
  d
}
d1 <- as.Date("2026-01-15")
d2 <- as.POSIXct("2026-01-15 14:30:00", tz = "GMT")
ref("unix_time_date", unix_df(decompose_datetime(x = d1)))
ref("unix_time_date_extended", unix_df(decompose_datetime(x = d1, extended = TRUE)))
ref("unix_time_datetime", unix_df(decompose_datetime(x = d2)))
ref("unix_time_datetime_extended", unix_df(decompose_datetime(x = d2, extended = TRUE)))
ref("unix_time_text_date", unix_df(decompose_datetime(x = "01/15/1900", format = "%m/%e/%Y", extended = TRUE)))
times <- c("2026-03-01 23:00:00", "2026-03-01 05:05:09", "2026-07-04 13:00:01", "2026-12-31 00:30:00")
ref("unix_time_text_datetime",
    unix_df(decompose_datetime(x = times, format = "%Y-%m-%d %H:%M:%S", extended = TRUE)))
ref("unix_time_text_breaks",
    unix_df(decompose_datetime(x = times, format = "%Y-%m-%d %H:%M:%S", extended = TRUE, breaks = c(-1, 6, 12, 18, 22, 24))))
ref("unix_time_text_tz",
    unix_df(decompose_datetime(x = times, format = "%Y-%m-%d %H:%M:%S", tz = "America/New_York", extended = TRUE)))
ref("unix_time_fraction", unix_df(decompose_datetime(x = c(d2 + 0.5, d2 + 3600.25), extended = TRUE)))
ref("unix_time_excel", data.frame(convert_excel_unix_timestamp(c(1, 25569, 45000.5))))

##########################################################################################
# FUNCTIONS_MATRIX
##########################################################################################
m1 <- matrix(1:9, nrow = 3, ncol = 3)
m2 <- matrix(c(11:18, 0.5), nrow = 3, ncol = 3)
ref("matrix_triangle_lower", matrix_df(matrix_triangle(m = m1)))
ref("matrix_triangle_lower_na", matrix_df(matrix_triangle(m = m1, diagonal = NA, type = "lower")))
ref("matrix_triangle_upper_na", matrix_df(matrix_triangle(m = m1, diagonal = NA, type = "upper")))
ref("matrix_triangle_upper", matrix_df(matrix_triangle(m = m1, diagonal = NULL, type = "upper")))
df_matrix <- data.frame(a = c(1, 2, 3), b = c(4, 5, 6), c = c(7, 8, 9), row.names = c("r1", "r2", "r3"))
ref("matrix_triangle_frame", matrix_df(matrix_triangle(df_matrix, off_diagonal = 0, diagonal = c(-1, -2, -3), type = "upper")))
ref("matrix_triangle_wide", matrix_df(matrix_triangle(matrix(1:6, 2))))
ref("matrix_triangle_tall", matrix_df(matrix_triangle(matrix(1:6, 3), type = "upper")))
ref("matrix_display_upper", matrix_df(display_upper_lower_triangle(m_upper = m1, m_lower = m2, diagonal = "upper")))
ref("matrix_display_lower", matrix_df(display_upper_lower_triangle(m_upper = m1, m_lower = m2, diagonal = "lower")))
ref("matrix_display_na", matrix_df(display_upper_lower_triangle(m_upper = m1, m_lower = m2, diagonal = NA)))
ref("matrix_display_value", matrix_df(display_upper_lower_triangle(m_upper = m1, m_lower = m2, diagonal = 1)))
ref("matrix_display_values", matrix_df(display_upper_lower_triangle(m_upper = m1, m_lower = m2, diagonal = c(1, 2, 3))))
ref("matrix_display_text", matrix_df(display_upper_lower_triangle(m_upper = m1, m_lower = m2, diagonal = c("X1", "X2", "X3"))))
m_lower <- matrix_triangle(matrix(1:9, nrow = 3, ncol = 3), type = "lower", diagonal = NA)
m_upper <- matrix_triangle(matrix(11:19, nrow = 3, ncol = 3), type = "upper", diagonal = NA)
ref("matrix_symmetric_lower", matrix_df(symmetric_matrix(matrix = m_lower, duplicate = "lower", diagonal = NA)))
ref("matrix_symmetric_upper", matrix_df(symmetric_matrix(matrix = m_upper, duplicate = "upper", diagonal = NA)))
ref("matrix_symmetric_frame", matrix_df(symmetric_matrix(as.matrix(df_matrix), duplicate = "upper")))
ref("matrix_off_diagonal_index", off_diagonal_index(length = 6))

##########################################################################################
# FUNCTIONS_MATHEMATICAL
##########################################################################################
angles <- c(-pi, 0, 0.5, pi / 2, pi, 10)
ref("mathematical_angles", data.frame(x = angles, rad2deg = rad2deg(angles), deg2rad = deg2rad(angles)))

##########################################################################################
# FUNCTIONS_GENERATE_DATA (random: structure and deterministic parts only)
##########################################################################################
set.seed(12345)
structure_row <- function(call, x) {
  x <- as.data.frame(x)
  data.frame(call = call, nrow = nrow(x), ncol = ncol(x), names = paste(names(x), collapse = "|"),
             classes = paste(sapply(x, function(v) class(v)[1]), collapse = "|"),
             levels = paste(sapply(x, function(v) paste(levels(v), collapse = ",")), collapse = "|"),
             na = sum(is.na(x)), stringsAsFactors = FALSE)
}
ref("generate_data_structure", rbind(
  structure_row("generate_data_normal", generate_data(nrows = 10, ncols = 5)),
  structure_row("generate_data_uniform", generate_data(nrows = 10, ncols = 5, min = 1, max = 5, type = "uniform")),
  structure_row("generate_data_decimal", generate_data(nrows = 10, ncols = 2, min = 1.5, max = 4, type = "uniform")),
  structure_row("generate_data_other", generate_data(nrows = 3, ncols = 2, type = "other")),
  structure_row("generate_factor_random", generate_factor(vector = LETTERS[1:5], ncols = 5, nrows = 10, type = "random")),
  structure_row("generate_factor_balanced", generate_factor(vector = LETTERS[1:5], ncols = 5, nrows = 10, type = "balanced")),
  structure_row("generate_factor_numeric", generate_factor(vector = c(3, 1, 2), ncols = 2, nrows = 4)),
  structure_row("generate_correlation_matrix_default", generate_correlation_matrix(nrows = 4)),
  structure_row("generate_correlation_matrix_named",
                generate_correlation_matrix(matrix(c(1, .5, .5, 1), 2, dimnames = list(c("p", "q"), c("p", "q"))), nrows = 3)),
  structure_row("simulate_correlation_from_sample",
                simulate_correlation_from_sample(data.frame(u = c(1, 2, 3, 5), v = c(2, 1, 4, 4)), nrows = 3)),
  structure_row("generate_missing_frame", generate_missing(data.frame(a = 1:4, b = letters[1:4]), missing = 2))
))
ref("generate_data_balanced", generate_factor(vector = LETTERS[1:5], ncols = 3, nrows = 10, type = "balanced"))
ref("generate_data_balanced_single", data.frame(x = generate_factor(vector = LETTERS[1:5], ncols = 1, nrows = 10, type = "balanced")))
uniform_pool <- function(min, max) {
  paste(sort(unique(unlist(generate_data(nrows = 200, ncols = 5, min = min, max = max, type = "uniform")))), collapse = "|")
}
ref("generate_data_uniform_pool", data.frame(
  min = c(1, 1.5, 3, 0.1), max = c(5, 4, 3, 1),
  values = c(uniform_pool(1, 5), uniform_pool(1.5, 4), uniform_pool(3, 3), uniform_pool(0.1, 1))
))
# the default target correlation matrix of generate_correlation_matrix(nrows = 4), built the way rwf builds it
default_target <- symmetric_matrix(as.matrix(generate_data(ncols = 4, nrows = 4, min = 0.1, max = 1, type = "uniform")))
diag(default_target) <- 1
ref("generate_data_default_target", matrix_df(unname(default_target)))
mr <- generate_multiple_response_vector(responses = c("a", "b", "c"), responded = 2, length = 200)
ref("generate_data_multiple_response", data.frame(
  value = sort(unique(unlist(strsplit(mr, ", ", fixed = TRUE)))), stringsAsFactors = FALSE))
ref("generate_data_multiple_response_counts", data.frame(counts = sort(unique(lengths(strsplit(mr, ", ", fixed = TRUE))))))

##########################################################################################
# GLM_IRT_U
##########################################################################################
theta <- seq(-3, 3, by = 0.5)
ref("irt_u_theta", data.frame(
  theta = theta,
  a5_b0 = compute_unidimensional_theta(a = 5, b = 0, theta = theta),
  a1_bm1 = compute_unidimensional_theta(a = 1, b = -1, theta = theta),
  a10_g1 = compute_unidimensional_theta(a = 10, b = 0, g = .1, theta = theta),
  a2_g2_i9 = compute_unidimensional_theta(a = 2, b = .5, g = .2, i = .9, theta = theta),
  a1_d1 = compute_unidimensional_theta(a = 1, b = 0, d = 1, theta = theta),
  info_1pl = compute_info_1pl(b = 1, theta = theta),
  info_2pl = compute_info_2pl(a = 1.5, b = 1, theta = theta),
  info_3pl = compute_info_3pl(a = 1.5, b = 1, g = .2, theta = theta),
  se_2pl = compute_se_theta(compute_info_2pl(a = 1.5, b = 1, theta = theta))
))
ref("irt_u_theta_vectorised", data.frame(
  value = compute_unidimensional_theta(a = c(1, 2, 0.5), b = c(0, 1, -1), g = c(0, .1, .2), i = c(1, .9, .8), theta = 0.5)
))
ref("irt_u_ability", data.frame(
  example = 1:4,
  theta = c(
    compute_unidimensional_ability(a = c(0.39, 0.45, 0.52, 0.3, 0.35, 0.43, 0.42, 0.44, 0.34, 0.42),
                                   b = c(-1.96, -1.9, -1.38, -0.58, 0.48, -0.81, -0.35, 1.59, 1.33, 2.93),
                                   u = c(1, 1, 1, 1, 0, 0, 1, 0, 1, 0), d = 1.7, g = NULL),
    compute_unidimensional_ability(a = c(1.27, 0.9, 0.94, 0.95, 0.55, 0.6, 0.44, 0.4),
                                   b = c(-0.54, 0.18, 0.21, 1.26, 1.73, -0.87, 1.72, 2.67),
                                   u = c(1, 1, 1, 1, 0, 0, 0, 0), d = 1.7, g = NULL),
    compute_unidimensional_ability(a = c(0.41, 0.32, 0.33, 1.2, 0.63, 0.62, 0.7, 0.61, 0.38, 0.53, 0.6, 1.16),
                                   b = c(-1.4, -1.3, -1.17, 0.2, 0.71, 0.86, -0.12, 0.12, 2.06, 1.38, 1.18, -0.33),
                                   u = c(1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0), d = 1.7, g = NULL),
    compute_unidimensional_ability(a = c(1, 1.5, 2, 0.8), b = c(-1, 0, 1, 2), g = c(.2, .1, .25, .15),
                                   u = c(1, 1, 0, 1))
  )
))
ref("irt_u_ability_limits", data.frame(
  case = c("all_correct", "all_wrong"),
  theta = c(compute_unidimensional_ability(a = c(1, 1, 1), b = c(0, 0, 0), u = c(1, 1, 1)),
            compute_unidimensional_ability(a = c(1, 1, 1), b = c(0, 0, 0), u = c(0, 0, 0), lim_theta = c(-4, 4)))
))
##########################################################################################
# FUNCTIONS (data frame utilities)
##########################################################################################
df_round <- data.frame(x = c(0.15, 2.675, 1.005, -0.5, 2.5, NA, 0.285, -0.05), y = 1:8, label = letters[1:8])
for (type in c("round", "ceiling", "floor", "tenth")) {
  ref(paste0("round_dataframe_", type), round_dataframe(df_round, digits = 1, type = type))
}
df_types <- data.frame(f = c(1, 0.1 + 0.2, NA, 1e-20, 123456789.123), i = c(1L, NA, 3L, 4L, 5L),
                       s = c("\t a \n", "b", NA, " c", "1.5"), l = c(TRUE, FALSE, NA, TRUE, TRUE),
                       stringsAsFactors = FALSE)
ref("change_data_type_character", change_data_type(df_types, "character"))
ref("change_data_type_numeric", suppressWarnings(change_data_type(df_types, "numeric")))
# levels of text with leading whitespace follow the platform's collation, so this column has none
factors <- change_data_type(data.frame(df_types[, c("f", "i", "l")], s = c("b", "B", "a", NA, "A10")), "factor")
ref("change_data_type_factor_levels", data.frame(column = names(factors),
                                                 levels = vapply(factors, function(x) paste(levels(x), collapse = "|"), "")))
combinations <- comparison_combinations(data.frame(b = 1, a = 2, C = 3, `d e` = 4, check.names = FALSE))
ref("comparison_combinations", data.frame(row = row.names(combinations), combinations))
bound <- rbind_all(data.frame(A = 1:2, B = c("x", "y")), data.frame(B = c("z", "w"), C = c(7.5, 8.5), row.names = c("r3", "r4")))
ref("rbind_all", data.frame(row = row.names(bound), bound))
df_nc <- data.frame(a = c(1, NA, Inf, 4), b = c(NA, NA, NA, NA), c = c(5, 5, 5, 5), d = c(-Inf, NaN, 2, 3))
ref("remove_nc_value", remove_nc(df_nc, value = 0))
for (aggressive in c(FALSE, TRUE)) {
  kept <- remove_nc(df_nc[, c("a", "d")], remove_rows = TRUE, aggressive = aggressive)
  ref(paste0("remove_nc_rows_", tolower(aggressive)), data.frame(row = row.names(kept), kept))
}
ref("remove_nc_cols", remove_nc(df_nc, remove_cols = TRUE, remove_zero_variance = TRUE))
ref("replace_na_with_previous", data.frame(value = replace_na_with_previous(c(NA, NA, 3, NA, 5, NA))))
ref("padNA", rwf:::padNA(data.frame(A = 1:2, B = c("x", "y")), 4, first = FALSE))
v1 <- 1:3
v2 <- c(10.5, 20.5)
m <- data.frame(p = 1:4, q = letters[1:4])
ref("c_bind", c_bind(v1, v2, m))
extremes <- min_max_index(c(3, 1, 3, 2, 1))
ref("min_max_index", data.frame(which = rep(c("max", "min"), lengths(extremes)), index = unlist(extremes)))
##########################################################################################
# POWER OF A CORRELATION (pwr::pwr.r.test through rwf)
##########################################################################################
power_cases <- list(two_sided = list(n = 30, r = 0.3, sig.level = 0.05, alternative = "two.sided"),
                    less = list(n = 20, r = -0.25, sig.level = 0.01, alternative = "less"),
                    greater = list(n = 25, r = 0.2, sig.level = 0.05, alternative = "greater"))
for (case in names(power_cases)) {
  ref(paste0("compute_power_r_", case), do.call(compute_power_r, power_cases[[case]])$power_table)
}
power_matrix <- matrix(c(1, 0.3, -0.5, 0.3, 1, 0.1, -0.5, 0.1, 1), nrow = 3)
ref("compute_power_r_matrix", compute_power_r_matrix(power_matrix, n = 15)$power_table)
##########################################################################################
# LOGISTIC MODEL COMPARISON
##########################################################################################
infert_logistic <- utils::read.csv(file.path(out_dir, "infert.csv"), stringsAsFactors = FALSE)
ref("output_compare_model_logistic", output_compare_model_logistic(
  glm(case ~ 1, family = binomial, data = infert_logistic),
  glm(case ~ age + parity + spontaneous, family = binomial, data = infert_logistic)))
ref("compute_y_logistic", data.frame(x = -2:2, y = compute_y_logistic(intercept = 0.5, coefficient = -1.2, x = -2:2)))
##########################################################################################
# CRONBACH'S ALPHA (rwf::report_alpha through psych::alpha, and the alpha helpers)
##########################################################################################
alpha_data <- list(complete = utils::read.csv(file.path(out_dir, "lavaan_data.csv")),
                   missing = utils::read.csv(file.path(out_dir, "lavaan_fiml_data.csv")))
alpha_key <- list(E = paste0("E", 1:5), N = paste0("N", 1:5), single = "E1")
alpha_parts <- c("result_total", "result_item_statistics", "result_dropped")
for (case in names(alpha_data)) {
  invisible(capture.output(alpha <- suppressWarnings(report_alpha(alpha_data[[case]], key = alpha_key))))
  for (part in alpha_parts) ref(paste0("report_alpha_", case, "_", sub("result_", "", part)), alpha[[part]])
}
invisible(capture.output(alpha <- suppressWarnings(report_alpha(
  alpha_data$complete, key = alpha_key[1:2], reverse = list(E = c(1, -1, 1, -1, 1), N = c(1, 1, 1, 1, -1)),
  mini = 1, maxi = 5, cumulative = TRUE))))
for (part in alpha_parts) ref(paste0("report_alpha_reverse_", sub("result_", "", part)), alpha[[part]])
alpha_items <- alpha_data$missing[, paste0("E", 1:5)]
ref("compute_raw_alpha", data.frame(alpha = compute_raw_alpha(alpha_items)))
ref("compute_alpha_diagnostics", compute_alpha_diagnostics(alpha_items))
ref("compute_mean_sd_alpha", compute_mean_sd_alpha(alpha_items))
ref("compute_mean_sd_alpha_divisor", compute_mean_sd_alpha(alpha_items, divisor = 5))
ref("key_to_cfa_model", data.frame(model = key_to_cfa_model(alpha_key[1:2])))
ref("compute_shrout", compute_shrout(sperson = 0.5, spersonitem = 0.2, stime = 0.1, spersontime = 0.15, serror = 0.3,
                                     m = 4, k = 3))
##########################################################################################
# THURSTONIAN IRT HELPERS (scoring with the ULSMV parameters above, and the data helpers)
##########################################################################################
tirt_lambda <- utils::read.csv(file.path(out_dir, "tirt_ulsmv_lambda.csv"), check.names = FALSE)
tirt_lambda <- as.matrix(data.frame(tirt_lambda[, -1], row.names = tirt_lambda$indicator, check.names = FALSE))
tirt_vectors <- utils::read.csv(file.path(out_dir, "tirt_ulsmv_vectors.csv"))
tirt_psi <- as.matrix(utils::read.csv(file.path(out_dir, "tirt_ulsmv_psi.csv"), check.names = FALSE))
tirt_patterns <- utils::read.csv(file.path(out_dir, "tirt_triplets.csv"))[1:25, rownames(tirt_lambda)]
tirt_patterns[3, 2] <- NA
tirt_patterns[5, c(1, 7)] <- NA
ref("score_tirt", data.frame(score_tirt(tirt_patterns, tirt_lambda, tirt_vectors$theta_diag, tirt_vectors$tau,
                                        tirt_psi, tirt_vectors$nu)))
tirt_ranks <- data.frame(a = c(1, 3, 2, 2, 1, 3), b = c(2, 1, 3, 1, 3, 2), c = c(3, 2, 1, 3, 2, 1),
                         d = c(2, 3, 1, 1, 2, 3), e = c(1, 2, 3, 3, 1, 2), f = c(3, 1, 2, 2, 3, 1))
ref("rank_to_binary", data.frame(rank_to_binary(tirt_ranks[, 1:3], 3)))
ref("rank_df_to_binary", rank_df_to_binary(tirt_ranks, 3))
ref("rank3_to_triplets", rank3_to_triplets(rank_to_binary(tirt_ranks[, 1:3], 3)))
ref("generate_matrix_A", data.frame(generate_matrix_A(2, 4)))
ref("generate_matrix_lambda_hat", data.frame(generate_matrix_lambda_hat(2, 3)))
ref("name_triplet_pairs", data.frame(pair = name_triplet_pairs(7, strict = FALSE)))
tirt_ability <- compute_ability(c(1, 0, 1), seq(-3, 3, 0.25), gamma = c(0.2, -0.3, 0.1), lambda = c(1, 1.5, 0.8),
                                psi = c(0.5, 0.7, 0.6))
ref("compute_ability_icc", tirt_ability$icc)
ref("compute_ability", data.frame(ability_ml = tirt_ability$ability_ml, ability_map = tirt_ability$ability_map))
ref("compute_solve", data.frame(compute_solve(matrix(c(4, 1, 2, 1, 3, 0.5, 2, 0.5, 5), 3), matrix(1:6, 3))))
##########################################################################################
# OUTLIERS AND EFA RESIDUAL STATISTICS
##########################################################################################
set.seed(11)
outlier_vector <- c(rnorm(300), 4.2, -3.9, 6, NA, NA)
ref("outlier_vector", data.frame(v = outlier_vector))
outlier_vector <- utils::read.csv(file.path(out_dir, "outlier_vector.csv"))$v
ref("outlier_summary", rbind(outlier_summary(outlier_vector), outlier_summary(c(1:19, 100))))
ref("remove_outliers", data.frame(default = remove_outliers(outlier_vector),
                                  type6 = remove_outliers(outlier_vector, probs = c(0.1, 0.9), type = 6)))
residual_a <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.2, 0.3, 0.2, 1), 3)
residual_b <- matrix(c(1, 0.42, 0.36, 0.42, 1, 0.27, 0.36, 0.27, 1.1), 3)
ref("compute_residual_stats_matrices", compute_residual_stats(residual_a, residual_b))
##########################################################################################
# EXPLORE PLOT CAPTIONS (rwf::plot_histogram, plot_qq, plot_boxplot)
##########################################################################################
caption_data <- utils::read.csv(file.path(out_dir, "..", "..", "src", "pwf", "data", "insurance.csv"))
caption_data$bmi[c(3, 9)] <- NA
caption_data <- caption_data[, c("age", "sex", "bmi", "charges")]
caption_histograms <- plot_histogram(caption_data)
caption_qq <- plot_qq(caption_data)
ref("explore_plot_captions", data.frame(
  plot = c(paste0("histogram_", names(caption_histograms)), paste0("qq_", names(caption_qq)), "boxplot"),
  caption = c(vapply(caption_histograms, function(p) p$labels$caption, ""), vapply(caption_qq, function(p) p$labels$caption, ""),
              plot_boxplot(caption_data)$labels$caption)))
##########################################################################################
# GLM_MEANS OPTIONS (mu, conf.level, exact, correct)
##########################################################################################
means_bp <- utils::read.csv(file.path(out_dir, "..", "..", "src", "pwf", "data", "blood_pressure.csv"),
                            stringsAsFactors = FALSE)
ref("report_ttests_mu_conf", report_ttests(df = means_bp, dv = 4:5, iv = 2:3, mu = 2, conf.level = 0.9))
ref("report_wtests_asymptotic_uncorrected", suppressWarnings(
  report_wtests(df = means_bp, dv = 4:5, iv = 2:3, exact = FALSE, correct = FALSE, conf.level = 0.9)))
ref("report_wtests_mu", suppressWarnings(report_wtests(df = means_bp, dv = 4, iv = 2, mu = 3)))
ref("report_wtests_exact", suppressWarnings(report_wtests(df = means_bp[c(1:12, 61:72), ], dv = 4:5, iv = 2,
                                                          exact = TRUE)))
##########################################################################################
# REPORT REGRESSION (polynomial, log outcome, no intercept, missing values)
##########################################################################################
regression_data <- utils::read.csv(file.path(out_dir, "..", "..", "src", "pwf", "data", "insurance.csv"),
                                   stringsAsFactors = FALSE)
regression_na <- regression_data
regression_na$bmi[c(5, 50, 500)] <- NA
regression_na$children[c(7, 70)] <- NA
regression_more <- list(poly = lm(charges ~ age + I(age^2) + sex, data = regression_data),
                        log = lm(log(charges) ~ age + smoker * sex, data = regression_data),
                        noint = lm(charges ~ 0 + age + bmi, data = regression_data),
                        na = lm(charges ~ bmi + children + smoker, data = regression_na))
for (case in names(regression_more)) {
  set.seed(1)
  invisible(capture.output(result <- suppressWarnings(report_regression(regression_more[[case]],
                                                                         plot_diagnostics = FALSE))))
  for (part in c("r", "coeficients", "deviance", "vif")) {
    ref(paste0("report_regression_", case, "_", part), result[[part]])
  }
  ref(paste0("report_regression_", case, "_anova"), data.frame(term = rownames(result$anova), result$anova,
                                                                 check.names = FALSE))
}
##########################################################################################
# FLATTEN LIST (nested lists, R's data.frame() naming)
##########################################################################################
ref("flatten_list_nested", flatten_list(list(a = list(c = 2, d = list(e = 3)), b = list(c = 4, d = list(e = 5, f = 6)))))
##########################################################################################
# LAVAAN CALL TEXT AND MODIFICATION INDEX ROW NUMBERS (rwf::report_cfa)
##########################################################################################
SEM_MODEL <- "E =~ E1 + E2 + 0.8*E3 + E4; N =~ N1 + N2 + N3 + N4; A =~ A1 + A2 + A3; A ~ E + N; E1 ~~ E2"
data <- utils::read.csv(file.path(out_dir, "lavaan_data.csv"))
# direct calls: lavaan keeps the names the arguments were given with (do.call would pass the values)
for (fit in list(sem = lavaan::sem(SEM_MODEL, data = data), cfa = lavaan::cfa(SEM_MODEL, data = data))) {
  invisible(capture.output(suppressWarnings(result <- report_cfa(fit))))
  ref(paste0("report_cfa_call_", fit@Options$model.type), result$call)
}
fit <- lavaan::cfa(SEM_MODEL, data = data, std.lv = TRUE)
invisible(capture.output(suppressWarnings(result <- report_cfa(fit))))
ref("report_cfa_call_cfa_std_lv", result$call)
ref("report_cfa_modification_rows", data.frame(row = as.integer(rownames(result$modification_indices)),
                                               result$modification_indices[, c("lhs", "op", "rhs")]))
# std.lv = TRUE with an endogenous latent variable: lavaan fixes its residual variance to 1 too
fit <- lavaan::sem(SEM_MODEL, data = data, std.lv = TRUE)
ref("lavaan_sem_std_lv_parameters", lavaan::parameterEstimates(fit)[, c("lhs", "op", "rhs", "est", "se")])
ref("lavaan_sem_std_lv_fit", data.frame(measure = c("chisq", "df"), value = as.numeric(lavaan::fitMeasures(fit, c("chisq", "df")))))
##########################################################################################
# R GRAPHICS HELPERS AND PLOT DATA (density, pretty, hist, boxplot.stats; rwf plot data)
##########################################################################################
# self-contained, so the section can be generated alone
graphics_data_dir <- normalizePath(file.path(out_dir, "..", "..", "src", "pwf", "data"))
graphics_insurance <- utils::read.csv(file.path(graphics_data_dir, "insurance.csv"), stringsAsFactors = FALSE)
d <- stats::density(graphics_insurance$bmi)
ref("r_density_bmi", data.frame(x = d$x, y = d$y, bw = d$bw))
d <- stats::density(graphics_insurance$age, n = 100, from = 10, to = 70)
ref("r_density_age_from_to", data.frame(x = d$x, y = d$y, bw = d$bw))
pretty_ranges <- list(c(0, 1e-13), c(-15, 0), c(18, 64), c(15.96, 53.13), c(1121.8739, 63770.42801), c(-2.3, 7.9),
                      c(0.001, 0.0011), c(5, 5), c(0, 0), c(-1e6, 3))
pretty_rows <- list()
for (i in seq_along(pretty_ranges)) for (n in c(1, 5, 12, 25)) for (min_n in unique(c(0, n %/% 3, n))) {
  pretty_rows[[length(pretty_rows) + 1]] <- data.frame(case = paste(i, n, min_n), lo = pretty_ranges[[i]][1],
                                                       hi = pretty_ranges[[i]][2], n = n, min_n = min_n,
                                                       value = pretty(pretty_ranges[[i]], n = n, min.n = min_n))
}
ref("r_pretty", do.call(rbind, pretty_rows))
hist_rows <- list()
for (v in c("age", "bmi", "charges", "children")) for (br in list("Sturges", "Scott", "FD", 7)) {
  h <- graphics::hist(graphics_insurance[[v]], breaks = br, plot = FALSE)
  hist_rows[[length(hist_rows) + 1]] <- data.frame(variable = v, breaks = as.character(br), left = utils::head(h$breaks, -1),
                                                   right = h$breaks[-1], count = h$counts)
}
ref("r_hist", do.call(rbind, hist_rows))
box_rows <- list()
for (v in c("bmi", "charges", "children")) {
  b <- grDevices::boxplot.stats(graphics_insurance[[v]])
  box_rows[[v]] <- data.frame(variable = v, statistic = c("whisker_low", "hinge_low", "median", "hinge_high", "whisker_high",
                                                          "conf_low", "conf_high", "outliers"),
                              value = c(b$stats, b$conf, length(b$out)))
}
ref("r_boxplot_stats", do.call(rbind, box_rows))
# rwf::plot_qq: ggplot2's stat_qq puts the sorted values at qnorm(ppoints(n))
p <- plot_qq(graphics_insurance[, "bmi", drop = FALSE])
ref("plot_qq_bmi_points", ggplot2::ggplot_build(p$bmi)$data[[1]][, c("x", "y")])
# rwf::plot_separability: as.factor() levels and ggplot2's density of each class
set.seed(11)
separability <- data.frame(observed = rep(c(-1, 1, 10, 2), each = 30),
                           predicted = round(c(stats::rbeta(30, 2, 5), stats::rbeta(30, 5, 2), stats::rbeta(30, 3, 3),
                                               stats::rbeta(30, 1, 1)), 3))
ref("plot_separability_data", separability)
p <- plot_separability(separability$observed, separability$predicted, title = "T")
b <- ggplot2::ggplot_build(p)$data[[1]]
ref("plot_separability_density", data.frame(observed = levels(p$data$observed)[b$group], x = b$x, density = b$density))
ref("plot_separability_labels", data.frame(title = p$labels$title, caption = p$labels$caption))
# rwf::plot_logistic_model: the title argument, melt()'s predictor order and the binomial smooth of a predictor
logistic_plot <- data.frame(outcome = c(rep(1, 10), rep(0, 10)), z2 = c(rep(1, 11), rep(0, 9)), b1 = c(rep(1, 9), rep(0, 11)),
                            a3 = c(5.2, 4.1, 6.3, 3.9, 5.5, 7.1, 4.8, 6.0, 5.9, 3.5, 8.2, 9.1, 7.7, 10.4, 8.8, 9.9, 7.2,
                                   11.0, 8.5, 6.6))
p <- plot_logistic_model(logistic_plot, title = "T")
b <- suppressWarnings(suppressMessages(ggplot2::ggplot_build(p)))$data[[1]]
ref("plot_logistic_model_labels", data.frame(title = p$labels$title, caption = p$labels$caption,
                                             predictors = paste(levels(p$data$Predictor), collapse = ",")))
ref("plot_logistic_model_smooth_a3", b[b$group == 3, c("x", "y")])
# rwf::plot_icc_thurstonian: items keep their column order (item1, ..., item10, ...)
icc <- compute_icc_thurstonian(eta = seq(-3, 3, by = 1), gamma = rep(0.5, 12), lambda = rep(1, 12), psi = rep(1, 12))$icc
ref("plot_icc_thurstonian_levels", data.frame(level = levels(plot_icc_thurstonian(icc)$data$variable)))
# rwf::plot_loadings: the variables are text, in alphabetical order on the axis; without sort rows and factors sorted
ocean <- utils::read.delim(file.path(graphics_data_dir, "ocean.csv"), stringsAsFactors = FALSE)
set.seed(3)  # psych labels the factors after random starts; in this state E1-E5 load on MR1, as in pwf
model <- psych::fa(ocean[1:500, c(paste0("E", 1:5), paste0("N", 1:5))], nfactors = 2, rotate = "oblimin", fm = "minres")
for (s in c(TRUE, FALSE)) {
  p <- plot_loadings(model, matrix_type = "structure", sort = s, title = "T")
  ref(paste0("plot_loadings_barplot_", tolower(s)), data.frame(p$plot_barplot$data, check.names = FALSE))
  ref(paste0("plot_loadings_axis_", tolower(s)), data.frame(variable = ggplot2::layer_scales(p$plot_barplot)$x$get_limits()))
}
# rwf::plot_mtmm: R's collation for the methods and traits (m1, m2, M3), lower-case names, labels
set.seed(7)
mtmm_base <- matrix(stats::rnorm(60 * 9), 60, 9)
mtmm_method <- function(shift) {
  d <- as.data.frame(round(mtmm_base + matrix(stats::rnorm(60 * 9, sd = shift), 60, 9) + 3, 1))
  names(d) <- paste0("x", 1:9)
  d
}
mtmm_data <- rbind(mtmm_method(0.5), mtmm_method(1), mtmm_method(1.5))
mtmm_data$method <- rep(c("m1", "m2", "M3"), each = 60)
mtmm_data$id <- rep(1:60, 3)
mtmm_data[c(5, 70, 150), "x2"] <- NA
ref("plot_mtmm_data", mtmm_data)
g <- plot_mtmm(mtmm_data, key = list(b = paste0("x", 1:3), A = paste0("x", 4:6), c = paste0("x", 7:9)),
               method = "method", subject = "id", title = "T")
mtmm_plot <- g$data
mtmm_plot[] <- lapply(mtmm_plot, function(v) if (is.factor(v)) as.character(v) else v)
mtmm_plot$label <- ggplot2::ggplot_build(g)$data[[2]]$label
ref("plot_mtmm_plot_data", mtmm_plot)
ref("plot_mtmm_levels", data.frame(trait_x = paste(levels(g$data$trait_x), collapse = ","),
                                   trait_y = paste(levels(g$data$trait_y), collapse = ","),
                                   method_x = paste(levels(g$data$method_x), collapse = ",")))
##########################################################################################
# PATH DIAGRAMS, DIAGNOSTIC PANELS AND IRT CURVES (igraph layouts, grid's curve, lowess; plot data)
##########################################################################################
# self-contained, so the section can be generated alone
plots_data_dir <- normalizePath(file.path(out_dir, "..", "..", "src", "pwf", "data"))
plots_insurance <- utils::read.csv(file.path(plots_data_dir, "insurance.csv"), stringsAsFactors = FALSE)
plots_blood_pressure <- utils::read.csv(file.path(plots_data_dir, "blood_pressure.csv"), stringsAsFactors = FALSE)
# igraph::layout_as_tree (Reingold-Tilford, several roots, unreachable vertices) and layout_in_circle
set.seed(41)
tree_rows <- list()
for (case in 1:60) {
  n <- sample(2:16, 1)
  if (case %% 2 == 0) {  # path-diagram shaped: latent variables point at indicators and at each other
    nl <- sample(1:min(4, n - 1), 1)
    from <- c(sample(nl, n - nl, replace = TRUE), sample(nl, sample(0:2, 1), replace = TRUE))
    to <- c((nl + 1):n, sample(nl, length(from) - (n - nl), replace = TRUE))
    roots <- seq_len(nl)
  } else {
    m <- sample(0:(2 * n), 1)
    from <- sample(n, m, replace = TRUE)
    to <- sample(n, m, replace = TRUE)
    roots <- sort(sample(n, sample(1:max(1, n %/% 3), 1)))
  }
  g <- igraph::add_edges(igraph::make_empty_graph(n, directed = TRUE), as.vector(rbind(from, to)))
  l <- igraph::layout_as_tree(g, root = roots, flip.y = TRUE)
  tree_rows[[case]] <- data.frame(case = case, n = n, edges = paste(paste0(from - 1, ">", to - 1), collapse = ";"),
                                  roots = paste(roots - 1, collapse = ";"), vertex = seq_len(n) - 1, x = l[, 1], y = l[, 2])
}
ref("r_layout_as_tree", do.call(rbind, tree_rows))
ref("r_layout_in_circle", do.call(rbind, lapply(1:12, function(n) {
  l <- igraph::layout_in_circle(igraph::make_empty_graph(n))
  data.frame(n = n, vertex = seq_len(n) - 1, x = l[, 1], y = l[, 2])
})))
# grid's curveGrob (ggplot2::geom_curve): the X-spline's points in inches on the 7 x 7 inch pdf(NULL) device
curve_rows <- list()
set.seed(43)
for (case in 1:12) {
  p <- round(stats::runif(4, 0, 7), 3)
  if (case == 6) p[3] <- p[1]
  if (case == 9) p[4] <- p[2]
  curvature <- c(0.35, -0.35, 0.5, 1, -1.2, 0.1)[(case %% 6) + 1]
  g <- grid::curveGrob(p[1], p[2], p[3], p[4], default.units = "inches", curvature = curvature, angle = 90, ncp = 5,
                       square = FALSE, squareShape = 1, inflect = FALSE, open = TRUE)
  points <- grid::xsplinePoints(grid:::calcCurveGrob(g, FALSE)$children[[1]])
  curve_rows[[case]] <- data.frame(case = case, x1 = p[1], y1 = p[2], x2 = p[3], y2 = p[4], curvature = curvature,
                                   x = as.numeric(points$x), y = as.numeric(points$y))
}
ref("r_curve_points", do.call(rbind, curve_rows))
# stats::lowess (clowess): ties, robustness iterations, delta
set.seed(47)
lowess_rows <- list()
for (case in 1:12) {
  n <- c(1, 2, 3, 7, 12, 30, 60, 120, 200, 45, 80, 25)[case]
  x <- switch((case %% 4) + 1, stats::rnorm(n), sample(1:3, n, TRUE) + 0, round(stats::runif(n, 0, 10), 1), stats::rexp(n))
  y <- if (case %% 3 == 0) ifelse(stats::runif(n) < 0.1, 50, stats::rnorm(n)) else x^2 + stats::rnorm(n, sd = 0.3)
  f <- c(2/3, 0.3, 0.9)[(case %% 3) + 1]
  iter <- c(3, 0, 1)[(case %% 3) + 1]
  l <- stats::lowess(x, y, f = f, iter = iter)
  lowess_rows[[case]] <- data.frame(case = case, f = f, iter = iter, x = x, y = y, lowess_x = l$x, lowess_y = l$y)
}
ref("r_lowess", do.call(rbind, lowess_rows))
# rwf::plot_cfa_gg and plot_cfa: igraph layouts, R's merge() order, formatC labels; "eq" fails without labels
plots_layer_rows <- function(p) {
  b <- ggplot2::ggplot_build(p)
  do.call(rbind, lapply(seq_along(b$data), function(i) {
    d <- b$data[[i]]
    data.frame(layer = i, geom = class(p$layers[[i]]$geom)[1],
               x = if (is.null(d$x)) (d$xmin + d$xmax) / 2 else d$x, y = if (is.null(d$y)) (d$ymin + d$ymax) / 2 else d$y,
               xend = if (is.null(d$xend)) NA else d$xend, yend = if (is.null(d$yend)) NA else d$yend,
               label = if (is.null(d$label)) NA else as.character(d$label))
  }))
}
cfa_data <- utils::read.csv(file.path(out_dir, "lavaan_data.csv"))
cfa_fit <- lavaan::sem("E =~ E1 + E2 + 0.8*E3 + E4; N =~ N1 + N2 + N3 + N4; A =~ A1 + A2 + A3; A ~ E + N; E1 ~~ E2",
                       data = cfa_data)
for (layout in c("tree", "circle")) for (what in c("std", "est")) {
  p <- plot_cfa_gg(cfa_fit, what = what, layout = layout)
  ref(paste0("plot_cfa_gg_", layout, "_", what), data.frame(title = p$labels$title, plots_layer_rows(p)))
}
cross_fit <- lavaan::cfa("b =~ E1 + E2 + E3 + N1; a =~ N1 + N2 + N3 + E3; C =~ A1 + A2 + A3", data = cfa_data)
ref("plot_cfa_gg_cross_tree_std", data.frame(title = "", plots_layer_rows(plot_cfa_gg(cross_fit, what = "std", layout = "tree"))))
ref("plot_cfa_keys", data.frame(key = names(suppressMessages(plot_cfa(cfa_fit)))))
# rwf::plot_oneway_diagnostics and the autoplot() of rwf::report_regression and report_logistic (ggfortify)
autoplot_rows <- function(mp, name, drop_panels = integer()) {
  # one row per point of each layer of each panel, and one row of labels (title, caption, x range) per panel
  layers <- list()
  labels <- list()
  for (k in setdiff(seq_along(mp@plots), drop_panels)) {
    p <- mp@plots[[k]]
    b <- suppressWarnings(ggplot2::ggplot_build(p))
    layers[[k]] <- do.call(rbind, lapply(seq_along(b$data), function(i) {
      d <- b$data[[i]]
      column <- function(column_name) if (is.null(d[[column_name]])) NA else d[[column_name]]
      data.frame(panel = k, layer = i, geom = class(p$layers[[i]]$geom)[1], x = column("x"), y = column("y"),
                 ymax = column("ymax"), slope = column("slope"), intercept = column("intercept"),
                 yintercept = column("yintercept"), label = if (is.null(d$label)) NA else as.character(d$label))
    }))
    labels[[k]] <- data.frame(panel = k, title = if (is.null(p$labels$title)) NA else p$labels$title,
                              caption = p$labels$caption, x_min = b$layout$panel_params[[1]]$x.range[1],
                              x_max = b$layout$panel_params[[1]]$x.range[2])
  }
  ref(paste0(name, "_layers"), do.call(rbind, layers))
  ref(paste0(name, "_labels"), do.call(rbind, labels))
}
diagnostics <- plot_oneway_diagnostics(plots_insurance, dv = c(3, 7), iv = c(5, 6, 4))
ref("plot_oneway_diagnostics_keys", data.frame(key = names(diagnostics)))
autoplot_rows(diagnostics[["smoker_charges"]], "plot_oneway_diagnostics_smoker_charges")
# equal group sizes: panel 5 plots the residuals against the factor levels (panel 6's lowess over equal leverages
# depends on the last bits of R's leverages, so it is left out)
autoplot_rows(plot_oneway_diagnostics(plots_blood_pressure, dv = 4, iv = 3)[[1]], "plot_oneway_diagnostics_balanced",
              drop_panels = 6)
infert_data <- utils::read.csv(file.path(out_dir, "infert.csv"))
model <- stats::lm(age ~ parity + induced, data = infert_data)
plot <- ggfortify:::autoplot.lm(model, which = 1:6, ncol = 2) +
  ggplot2::labs(title = "", caption = paste0(deparse(model$terms), "\nobservations=", nrow(model$model))) +
  ggplot2::theme_bw(base_size = 10)
autoplot_rows(plot, "report_regression_diagnostics")
model <- stats::glm(case ~ age + parity + spontaneous, family = binomial, data = infert_data)
plot <- ggfortify:::autoplot.lm(model) +
  ggplot2::labs(title = "T", caption = paste0(deparse(model$terms), "\nobservations=", nrow(model$model))) +
  ggplot2::theme_bw(base_size = 10)
autoplot_rows(plot, "report_logistic_diagnostics")
# rwf::plot_irt_onefactor (mirt): factor levels of the series (their colours) and the curves; responses saved
set.seed(53)
irt_items <- as.data.frame(psych::sim.rasch(nvar = 11, n = 1000, low = -3, high = 3, d = NULL, a = 1, mu = 0, sd = 1)$items)
ref("plot_irt_onefactor_items", irt_items)
p <- plot_irt_onefactor(mirt::mirt(irt_items, 1, verbose = FALSE), title = "T")
ref("plot_irt_onefactor_data", data.frame(type = p$data$type, theta = p$data$theta,
                                          variable = as.character(p$data$variable), value = p$data$value))
ref("plot_irt_onefactor_levels", data.frame(level = levels(p$data$variable), title = p$labels$title))
##########################################################################################
# REPORT XGBOOST WITH EARLY STOPPING (best iteration and score, predictions up to the best iteration)
##########################################################################################
# self-contained, so the section can be generated alone; the model is saved as xgboost JSON as above
early_admission <- utils::read.csv(file.path(out_dir, "..", "..", "src", "pwf", "data", "admission.csv"))
early_train <- early_admission[1:300, ]
early_test <- early_admission[301:400, ]
early_features <- c("gre", "gpa", "rank")
early_dtrain <- xgboost::xgb.DMatrix(as.matrix(early_train[, early_features]), label = early_train$admit)
early_dtest <- xgboost::xgb.DMatrix(as.matrix(early_test[, early_features]), label = early_test$admit)
set.seed(1)
model <- xgboost::xgb.train(params = list(objective = "binary:logistic", nthread = 1, max_depth = 2, eta = 0.3),
                            data = early_dtrain, nrounds = 50, evals = list(train = early_dtrain, test = early_dtest),
                            early_stopping_rounds = 3, verbose = 0)
xgboost::xgb.save(model, file.path(out_dir, "xgb_early_stop.json"))
report <- report_xgboost(model = model, validation_data = early_test, label = "admit", file = NULL)
ref("report_xgboost_early_stop_model_summary", report$result$model_summary)
ref("report_xgboost_early_stop_validation_metrics", report$result$validation_metrics)
ref("report_xgboost_early_stop_evaluation_log", report$result$evaluation_log)
##########################################################################################
# THURSTONIAN IRT INDEX AND SCORING HELPERS (R's 2:1 loops, recycling, row names, optim's controls)
##########################################################################################
# self-contained, so the section can be generated alone
ref("compute_dummy_comparisons", data.frame(items = 0:6, comparisons = compute_dummy_comparisons(0:6)))
tirt_index <- lapply(c(0, 2, 3, 4), generate_unique_comparisons_index)
ref("generate_unique_comparisons_index", data.frame(items = rep(c(0, 2, 3, 4), sapply(tirt_index, nrow)),
                                                    row = unlist(lapply(tirt_index, rownames)),
                                                    do.call(rbind, tirt_index)))
ref("generate_comparisons_matrix", data.frame(generate_comparisons_matrix(4)))
# one block gives three rows: rwf's for (i in 2:blocks) runs over 2:1
ref("increase_index", do.call(rbind, lapply(list(c(1, 3), c(2, 3), c(3, 2), c(0, 2), c(2.5, 2)), function(a) {
  m <- increase_index(a[1], a[2])
  data.frame(blocks = a[1], items = a[2], row = as.vector(row(m)), col = as.vector(col(m)), value = as.vector(m))
})))
tirt_dimension_cases <- list(list(1:18, 3, c(1, 2)), list(1:12, 3, c(2, 3)), list(1:8, 3, c(1, 3)), list(1:2, 3, c(1, 2)),
                             list(1:9, 3, c(1, 4)), list(c(0.5, 2, 3.5, 4, 5, 6), 2, 2))
ref("response_dimension", do.call(rbind, lapply(seq_along(tirt_dimension_cases), function(i) {
  a <- tirt_dimension_cases[[i]]
  data.frame(case = i, value = response_dimension(a[[1]], a[[2]], a[[3]]))
})))
ref("cfa_icc_index", do.call(rbind, lapply(list(c(18, 3), c(12, 4), c(6, 2), c(6, 1), c(7, 3)), function(a) {
  x <- cfa_icc_index(a[1], a[2])
  m <- x$index_matrix
  data.frame(nitems = a[1], nfactors = a[2], row = as.vector(row(m)), col = as.vector(col(m)), matrix = as.vector(m),
             vector = x$index_vector)
})))
ref("icc_cfa", data.frame(case = rep(1:3, c(5, 3, 4)), value = c(
  icc_cfa(seq(-2, 2, 1), 1, 1, 1),
  icc_cfa(c(-1, 0, 1), c(0.5, -0.2, 0.1), c(1, 2, -1), c(1, 2, 0.5)),
  icc_cfa(c(-1, 0, 1, 2), c(0.5, -0.2), 1, 1))))  # gamma recycled, as R's arithmetic
ref("compute_map", data.frame(eta = seq(-2, 2, 0.5), standard = compute_map(seq(-2, 2, 0.5)),
                              shifted = compute_map(seq(-2, 2, 0.5), 0.5, 2)))
# rwf's examples of compute_ability and compute_scores, which pass eta, gamma, lambda and psi by position
tirt_gamma <- c(0.556, -1.253, -1.729, 0.618, 0.937, 0.295, -0.672, -1.127, -0.446, 0.632, 1.147, 0.498)
tirt_item_psi <- c(2.172, 1.883, 2.055, 1.869, 2.231, 2.100, 1.762, 1.803, 1.565, 1.892, 1.794, 1.686)
tirt_item_lambda <- c(1.082, 1.082, -1.297, -1.297, 0.802, 0.802, 1.083, 1.083)
tirt_gamma <- tirt_gamma[response_dimension(c(1:12), 3, c(1, 2))]
tirt_item_psi <- tirt_item_psi[response_dimension(c(1:12), 3, c(1, 2))]
tirt_eta <- seq(-6, 6, by = 0.1)
tirt_responses <- data.frame(rbind(c(0, 0, 0, 0, 0, 0, 0, 0), c(1, 1, 1, 1, 1, 1, 1, 1), c(1, 0, 1, 0, 1, 0, 1, 0),
                                   c(0, 1, 0, 1, 0, 1, 0, 1), c(1, 1, 0, 0, 1, 1, 0, 0)))
ref("compute_scores", data.frame(ability = compute_scores(tirt_responses, tirt_eta, tirt_gamma, tirt_item_lambda,
                                                          tirt_item_psi, map = compute_map(tirt_eta), plot = FALSE)))
# with plot = TRUE rwf prints the curves and returns the likelihood and posterior scaled up by tens
tirt_plotted <- compute_ability(c(1, 1, 0, 0, 1, 1, 0, 0), tirt_eta, tirt_gamma, tirt_item_lambda, tirt_item_psi,
                                plot = TRUE)
ref("compute_ability_plot", tirt_plotted$icc)
ref("compute_ability_plot_title", data.frame(title = ggplot2::last_plot()$labels$title))
# score_tirt_pattern with optim's controls, a starting point and missing responses (ULSMV parameters above)
tirt_lambda <- utils::read.csv(file.path(out_dir, "tirt_ulsmv_lambda.csv"), check.names = FALSE)
tirt_lambda <- as.matrix(data.frame(tirt_lambda[, -1], row.names = tirt_lambda$indicator, check.names = FALSE))
tirt_vectors <- utils::read.csv(file.path(out_dir, "tirt_ulsmv_vectors.csv"))
tirt_psi <- as.matrix(utils::read.csv(file.path(out_dir, "tirt_ulsmv_psi.csv"), check.names = FALSE))
tirt_pattern <- as.numeric(utils::read.csv(file.path(out_dir, "tirt_triplets.csv"))[1, rownames(tirt_lambda)])
tirt_score_cases <- list(
  default = list(pattern = tirt_pattern),
  maxit = list(pattern = tirt_pattern, control = list(maxit = 3)),
  init = list(pattern = tirt_pattern, init = c(0.5, -0.5, 0.2)),
  reltol = list(pattern = tirt_pattern, control = list(reltol = 1e-4)),
  parscale = list(pattern = tirt_pattern, control = list(parscale = c(2, 1, 0.5), maxit = 5)),
  fnscale = list(pattern = tirt_pattern, control = list(fnscale = 10, maxit = 5)),
  missing = list(pattern = replace(tirt_pattern, c(2, 5, 9), NA)),
  all_missing = list(pattern = rep(NA_real_, length(tirt_pattern))))
ref("score_tirt_pattern", do.call(rbind, lapply(names(tirt_score_cases), function(case) {
  estimate <- do.call(score_tirt_pattern, c(tirt_score_cases[[case]], list(
    lambda = tirt_lambda, theta_diag = tirt_vectors$theta_diag, tau = tirt_vectors$tau, Psi = tirt_psi,
    nu = tirt_vectors$nu)))
  data.frame(case = case, trait = names(estimate), value = as.numeric(estimate))
})))
# R's errors and warnings that pwf repeats
tirt_condition <- function(expr) tryCatch(expr, error = conditionMessage, warning = conditionMessage)
ref("tirt_helper_conditions", data.frame(
  call = c("generate_comparisons_matrix(1)", "rank_to_binary(one item)", "increase_index(2, 0)",
           "score_tirt_pattern(control = list(foo = 1))"),
  message = c(tirt_condition(generate_comparisons_matrix(1)),
              tirt_condition(rank_to_binary(data.frame(a = 1:3)[, 1, drop = FALSE], 1)),
              tirt_condition(increase_index(2, 0)),
              tirt_condition(score_tirt_pattern(tirt_pattern, tirt_lambda, tirt_vectors$theta_diag, tirt_vectors$tau,
                                                tirt_psi, control = list(foo = 1))))))
##########################################################################################
# STRING HELPER EDGE CASES (strsplit's trailing separator, data.frame()'s names, strwrap, call_to_string)
##########################################################################################
# self-contained, so the section can be generated alone
string_warning <- function(expr) {
  message <- NA_character_
  value <- withCallingHandlers(expr, warning = function(w) {
    message <<- conditionMessage(w)
    invokeRestart("muffleWarning")
  })
  list(value = value, warning = message)
}
ref("str_split_trailing", str_split(c("a/b/", "c/d/")))  # strsplit() drops a trailing empty piece
string_ragged <- string_warning(str_split(c("a/b", "c/d/e")))  # matrix() refills the rows, with a warning
ref("str_split_ragged", string_ragged$value)
ref("str_split_one_piece", str_split(c("a", "b"), include_original = TRUE))  # data.frame() names the matrix
string_df <- data.frame(X1 = c(10, 20), b = c("u", "v"), row.names = c("1/aa", "2/bb"))
ref("str_split_df_row", data.frame(row = row.names(str_split_df(string_df)), str_split_df(string_df)))
string_ids <- data.frame(id = c("x-1-p", "y-2-q"), v = 1:2)
ref("str_split_df_collumn", str_split_df(string_ids, split = "-", type = "collumn", index = 1, include_original = TRUE))
ref("str_split_df_automatic", str_split_df(data.frame(v = 1:3)))
ref("str_trim_df", str_trim_df(data.frame(s = c("  a   b  ", " c", "x.  y", "p\tq", "end?  Yes"),
                                          t = c("u", " v ", "w  ", "z", ""))))
string_insurance <- load_data("insurance")
string_infert <- load_data("infert")
ref("call_to_string", data.frame(case = c("lm", "glm_long", "poisson", "Call", "none"), call = c(
  call_to_string(lm(charges ~ age + bmi, data = string_insurance)),
  call_to_string(glm(case ~ education + age + parity + induced + spontaneous + age:parity + age:induced,
                     family = binomial, data = string_infert)),
  call_to_string(glm(spontaneous ~ age + parity, family = poisson, data = string_infert)),
  call_to_string(list(Call = quote(f(x = 1, y = "a b")))),
  call_to_string(list(a = 1)))))
ref("output_separator", data.frame(line = c(
  utils::capture.output(output_separator("TITLE", output = "OUT", instruction = "Read me", length = 25)),
  utils::capture.output(output_separator("TITLE2", length = 22.5)))))
ref("string_conditions", data.frame(call = c("str_split(ragged)", "str_split_df(type = column)"), message = c(
  string_ragged$warning, tryCatch(str_split_df(string_ids, type = "column", index = 1), error = conditionMessage))))
##########################################################################################
# EXCEL HELPERS (rwf's workbooks themselves: excel_helpers.xlsx and report_dataframe.xlsx)
##########################################################################################
# self-contained, so the section can be generated alone; tests read the cells, styles, comments and
# conditional formats of these files and of pwf's
excel_df <- data.frame(id = 1:6, score = c(1.25, 2.5, NA, 4.75, 5, 6.125), count = c(3, NA, 5, 6, 7, 8),
                       group = c("a", "b", "", "a", "b", "c"), p = c(0.01, 0.2, 0.049, 0.5, NA, 0.03),
                       z = c(-2.5, 0.3, 1.97, -Inf, 2.4, NA), stringsAsFactors = FALSE)
excel_insurance <- load_data("insurance")
excel_r <- stats::cor(excel_insurance[, c("age", "bmi", "children", "charges")])
excel_observed <- c(1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3)
excel_predicted <- c(1, 2, 2, 2, 3, 1, 1, 2, 3, 3, 2, 3)
excel_wb <- openxlsx::createWorkbook()
excel_critical_value(excel_df, excel_wb, sheet = "critical", title = "Critical values",
                     comment = list(score = "Score note", group = "Group note", nothere = "ignored"),
                     critical = list(p = "<0.05", z = c("<-1.96", ">1.96"), group = "=\"a\""))
excel_matrix(excel_r, excel_wb, sheet = "matrix", title = "Correlations", comment = list(age = "Age note"),
             conditional_formatting = TRUE, diagonal = TRUE)
excel_matrix(excel_insurance[1:3, c("age", "bmi", "charges", "children")], excel_wb, sheet = "rectangle",
             numFmt = "#0.000", diagonal = TRUE)
excel_confusion_matrix(confusion_matrix_percent(excel_observed, excel_predicted), excel_wb)
openxlsx::saveWorkbook(excel_wb, file.path(out_dir, "excel_helpers.xlsx"), overwrite = TRUE)
report_dataframe(excel_df[1:3, 1:4], file = file.path(out_dir, "report_dataframe"), type = "matrix",
                 sheet = "data", conditional_formatting = TRUE)
ref("report_dataframe_no_file", report_dataframe(excel_df, file = NULL))
ref("data_frame_index", data.frame(data_frame_index(3, 2)))
##########################################################################################
# ENVIRONMENT HELPERS (combine_files: list.files() order, readLines() lines; print() under environment_options)
##########################################################################################
# self-contained, so the section can be generated alone; the test writes the same files
environment_dir <- file.path(tempfile("combine_files"))
dir.create(file.path(environment_dir, "in", "sub"), recursive = TRUE)
dir.create(file.path(environment_dir, "in", "dir.R"))  # a directory matching the pattern
environment_files <- list("a.R" = "a <- 1\nx <- 2", "Z.R" = "Z <- 1\n", "sub/c.r" = "c <- 3\r\nd <- 4\r\n",
                          "note.txt" = "not r\n", "ff.R" = "f\fg\n", "m.R" = "old\r\nmac\rline\n", "empty.R" = "")
for (name in names(environment_files)) {
  writeBin(charToRaw(environment_files[[name]]), file.path(environment_dir, "in", name))
}
environment_wd <- setwd(environment_dir)
environment_combined <- list(recursive = combine_files(input_dir = "in", output_file = "out_rec.R"),
                             flat = combine_files(input_dir = "in", output_file = "in/out_flat.R", recursive = FALSE))
ref("combine_files_list", data.frame(recursive = rep(names(environment_combined), lengths(environment_combined)),
                                     file = unlist(environment_combined, use.names = FALSE)))
ref("combine_files_lines", rbind(data.frame(recursive = "recursive", line = readLines("out_rec.R")),
                                 data.frame(recursive = "flat", line = readLines("in/out_flat.R"))))
setwd(environment_wd)
environment_numbers <- c(16884.924, 0.000012345, 1.5, 1 / 3, 123456789.123, -2.5e-7, 100, 0.1 + 0.2, 9.99995, 0)
environment_old <- options(digits = 4, scipen = 999)  # what environment_options() sets for printing numbers
ref("environment_options_print", data.frame(x = environment_numbers, printed = vapply(environment_numbers, function(x)
  sub("^\\[1\\] ", "", utils::capture.output(print(x))), character(1))))
options(environment_old)
##########################################################################################
# K FOLD, K SAMPLE AND PLOT CONFUSION (fold sizes and printed lines; tiles, colours and caption)
##########################################################################################
# self-contained, so the section can be generated alone. The folds are random, their sizes are not: the
# printed lines of k_fold and of k_sample's halving into train, test and validation are compared
ref("cut_labels", do.call(rbind, lapply(list(c(10, 3), c(8, 3), c(1338, 10), c(7, 7), c(5, 2)), function(a)
  data.frame(n = a[1], k = a[2], label = cut(seq_len(a[1]), breaks = a[2], labels = FALSE)))))
fold_insurance <- load_data("insurance")
set.seed(1)
ref("k_fold_printed", data.frame(line = utils::capture.output(
  fold_result <- k_fold(fold_insurance, charges ~ age + bmi + smoker, k = 7))))
ref("k_fold_parts", data.frame(variables = paste(fold_result$variables, collapse = ","),
                               predictors = paste(fold_result$predictors, collapse = ","),
                               outcome = fold_result$outcome, n_index = length(fold_result$index)))
ref("k_sample_printed", do.call(rbind, lapply(1:12, function(n) {
  frame <- data.frame(y = seq_len(n), x = seq_len(n) * 2)
  line <- tryCatch(utils::capture.output(invisible(k_sample(frame, y ~ x, k = 1))),
                   error = function(e) paste("ERROR:", conditionMessage(e)))
  data.frame(n = n, line = line)
})))
tiles_cases <- list(numbers = list(observed = c(1, 2, 3, 1, 2, 3, 10, 2), predicted = c(1, 2, 3, 1, 1, 3, 10, 2)),
                    text = list(observed = c(rep("male", 10), rep("female", 10), "male", "male"),
                                predicted = c(rep("male", 10), rep("female", 10), "female", "female")),
                    unequal = list(observed = c("a", "b", "c", "a", "B"), predicted = c("a", "b", "b", "a", "a")))
ref("plot_confusion_tiles", do.call(rbind, lapply(names(tiles_cases), function(case) {
  p <- do.call(plot_confusion, c(tiles_cases[[case]], title = "T"))
  built <- ggplot2::ggplot_build(p)
  tiles <- merge(built$data[[1]][, c("x", "y", "fill")], built$data[[2]][, c("x", "y", "label", "size")])
  data.frame(case = case, tiles[order(tiles$x, tiles$y), ],
             observed = paste(levels(p$data$observed), collapse = ","),
             predicted = paste(levels(p$data$predicted), collapse = ","),
             title = p$labels$title, caption = p$labels$caption)
})))
##########################################################################################
# PLOT HELPERS (plot_scree, plot_scatterplot's captions, plot_normality_diagnostics' panels, excel_generic_format)
##########################################################################################
# self-contained, so the section can be generated alone
plots_insurance <- load_data("insurance")
plots_insurance$bmi[c(2, 5)] <- NA
plots_insurance$charges[7] <- NA
# plot_scree: the eigenvalues of the pairwise correlations, the criteria note and the labels
plots_scree <- plot_scree(plots_insurance[, c("age", "bmi", "children", "charges")], base_size = 12, title = "T")
plots_scree_built <- ggplot2::ggplot_build(plots_scree)
ref("plot_scree_line", plots_scree_built$data[[3]][, c("x", "y")])
ref("plot_scree_labels", data.frame(title = plots_scree$labels$title, x = plots_scree$labels$x,
                                    y = plots_scree$labels$y, note = plots_scree_built$data[[5]]$label,
                                    note_x = plots_scree_built$data[[5]]$x, note_y = plots_scree_built$data[[5]]$y))
# plot_scatterplot: rwf draws ggExtra::ggMarginal's grob, so its caption is read from the grob's texts
plots_texts <- function(grob) {
  out <- character(0)
  if (inherits(grob, "text") && !is.null(grob$label)) out <- c(out, paste(grob$label, collapse = " | "))
  for (child in c(if (!is.null(grob$children)) as.list(grob$children), if (!is.null(grob$grobs)) grob$grobs)) {
    out <- c(out, plots_texts(child))
  }
  out
}
plots_scatter <- list(
  default = plot_scatterplot(plots_insurance[, c("age", "bmi", "children")], title = "T"),
  poly = plot_scatterplot(plots_insurance[, c("age", "bmi")], formula = y ~ poly(x, 2), str_aes = FALSE),
  too_few = plot_scatterplot(data.frame(a = c(1, NA, 3), b = c(NA, 2, NA))))
ref("plot_scatterplot_captions", do.call(rbind, lapply(names(plots_scatter), function(case) {
  do.call(rbind, lapply(names(plots_scatter[[case]]), function(name) {
    p <- plots_scatter[[case]][[name]]
    texts <- if (is.null(p)) character(0) else plots_texts(p$layers[[1]]$geom_params$grob)
    data.frame(case = case, name = name, drawn = !is.null(p),
               caption = if (is.null(p)) NA_character_ else texts[grepl("^Pairwise n", texts)])
  }))
})))
# plot_normality_diagnostics' four panels are R's hist(), density(), boxplot() and qqnorm() with qqline()
plots_vector <- stats::na.omit(plots_insurance$bmi)
for (plots_breaks in list("Sturges", 5)) {
  plots_hist <- graphics::hist(plots_vector, breaks = plots_breaks, plot = FALSE)
  ref(paste0("plot_normality_hist_", plots_breaks), data.frame(left = utils::head(plots_hist$breaks, -1),
                                                               right = plots_hist$breaks[-1],
                                                               count = plots_hist$counts))
}
plots_density <- stats::density(plots_vector)
ref("plot_normality_density", data.frame(x = plots_density$x, y = plots_density$y, bw = plots_density$bw))
plots_box <- grDevices::boxplot.stats(plots_vector)
ref("plot_normality_box", data.frame(part = c(rep("stats", 5), rep("out", length(plots_box$out))),
                                     value = c(plots_box$stats, plots_box$out)))
plots_qq <- stats::qqnorm(plots_vector, plot.it = FALSE)
plots_quartiles <- stats::quantile(plots_vector, c(0.25, 0.75), names = FALSE)
plots_slope <- diff(plots_quartiles) / diff(stats::qnorm(c(0.25, 0.75)))
ref("plot_normality_qq", data.frame(x = plots_qq$x, y = plots_qq$y, slope = plots_slope,
                                    intercept = plots_quartiles[1] - plots_slope * stats::qnorm(0.25)))
ref("plot_normality_caption", data.frame(caption = paste0(  # rwf's title(sub = ...)
  "\nVariable=", str_aes("bmi"), "\nObservations=", length(plots_vector), "\nMean=", round(mean(plots_vector), 2),
  "\nSD=", round(stats::sd(plots_vector), 2), "\nMedian=", round(stats::median(plots_vector), 2))))
# excel_generic_format on a sheet written with the row names, as its example
plots_wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(plots_wb, "sheet")
openxlsx::writeData(plots_wb, sheet = "sheet", x = plots_insurance[1:6, ], colNames = TRUE, rowNames = TRUE)
excel_generic_format(plots_insurance[1:6, ], plots_wb, sheet = "sheet", title = "Generic",
                     comment = list(age = "Age note", region = "Region note"), numFmt = "#0.0")
openxlsx::saveWorkbook(plots_wb, file.path(out_dir, "excel_generic.xlsx"), overwrite = TRUE)
##########################################################################################
# REPORT IRT (rwf::report_irt on a mirt model of the plot_irt_onefactor items; pwf needs R through rpy2)
##########################################################################################
# self-contained, so the section can be generated alone
irt_report_items <- utils::read.csv(file.path(out_dir, "plot_irt_onefactor_items.csv"))
irt_report <- report_irt(mirt::mirt(irt_report_items, 1, verbose = FALSE))
for (part in c("model_coefficients", "model_coefficients_oblimin", "model_options", "q3_matrix", "exp_residuals",
               "item_fit", "g2_fit", "m2_fit")) {
  ref(paste0("report_irt_", part), data.frame(row = row.names(irt_report[[part]]), irt_report[[part]],
                                              check.names = FALSE))
}
ref("report_irt_model_call", data.frame(call = irt_report$model_call))
##########################################################################################
# GRADIENT FILLS (ggplot2's scale_fill_gradient2: colours interpolated in Lab, censored before rescaling)
##########################################################################################
# self-contained, so the section can be generated alone
gradient_cases <- list(
  limits = list(values = seq(-1, 1, 0.1), args = list(midpoint = 0, limit = c(-1, 1))),
  asymmetric = list(values = c(-0.8, -0.5, -0.2, 0, 0.3, 0.6, 1, NA), args = list(midpoint = 0, limit = c(-0.5, 1))),
  colours = list(values = c(-0.37, -0.1, 0, 0.25, 0.5, 0.81), args = list(low = "#5E912C", mid = "white",
                                                                        high = "#5F2C91", midpoint = 0)),
  names = list(values = c(0, 1, 2, 3, 4), args = list(low = "green", mid = "gray", high = "purple", midpoint = 1)))
ref("scale_fill_gradient2_fills", do.call(rbind, lapply(names(gradient_cases), function(case) {
  data <- data.frame(x = seq_along(gradient_cases[[case]]$values), value = gradient_cases[[case]]$values)
  p <- ggplot2::ggplot(data, ggplot2::aes(x, 1, fill = value)) + ggplot2::geom_tile() +
    do.call(ggplot2::scale_fill_gradient2, gradient_cases[[case]]$args)
  data.frame(case = case, value = data$value, fill = ggplot2::ggplot_build(p)$data[[1]]$fill)
})))
# rwf's plots with scale_fill_gradient2: plot_corrplot, and plot_loadings' correlation tiles and loading bars
gradient_insurance <- load_data("insurance")
gradient_corrplot <- ggplot2::ggplot_build(plot_corrplot(stats::cor(gradient_insurance[, c("age", "bmi", "children",
                                                                                           "charges")]), title = "T"))
ref("plot_corrplot_fills", gradient_corrplot$data[[1]][, c("x", "y", "fill")])
gradient_ocean <- utils::read.delim(file.path(data_dir, "ocean.csv"), stringsAsFactors = FALSE)
set.seed(3)  # psych labels the factors after random starts; in this state E1-E5 load on MR1, as in pwf
gradient_model <- psych::fa(gradient_ocean[1:500, c(paste0("E", 1:5), paste0("N", 1:5))], nfactors = 2,
                            rotate = "oblimin", fm = "minres")
gradient_loadings <- plot_loadings(gradient_model, matrix_type = "structure", title = "T")
# the correlation tiles are drawn inside ggpubr::ggarrange's first grob: its rects' fills and its texts' labels
gradient_grobs <- function(grob, kind) {
  out <- list()
  if (inherits(grob, kind) && length(if (kind == "rect") grob$gp$fill else grob$label) > 1) {
    out <- list(if (kind == "rect") grob$gp$fill else grob$label)
  }
  for (child in c(if (!is.null(grob$children)) as.list(grob$children), if (!is.null(grob$grobs)) grob$grobs)) {
    out <- c(out, gradient_grobs(child, kind))
  }
  out
}
gradient_tiles <- gradient_loadings$correlation_loadings$layers[[1]]$geom_params$grob
gradient_fills <- gradient_grobs(gradient_tiles, "rect")[[1]]
gradient_labels <- Filter(function(x) length(x) == length(gradient_fills), gradient_grobs(gradient_tiles, "text"))[[1]]
ref("plot_loadings_correlation_fills", data.frame(label = gradient_labels, fill = gradient_fills))
gradient_bars <- ggplot2::ggplot_build(gradient_loadings$plot_barplot)
ref("plot_loadings_barplot_fills", data.frame(gradient_loadings$plot_barplot$data[, c("variable", "Factor",
                                                                                      "Loading")],
                                              fill = gradient_bars$data[[1]]$fill))
##########################################################################################
# LAVAAN LABELS AND CONSTRAINTS (shared labels, == constraints, := definitions; ML and FIML)
##########################################################################################
# self-contained, so the section can be generated alone; the data are the LAVAAN ENGINE section's
labels_data <- utils::read.csv(file.path(out_dir, "lavaan_data.csv"))
labels_fiml_data <- utils::read.csv(file.path(out_dir, "lavaan_fiml_data.csv"))
labels_cases <- list(
  equal = list(model = "E =~ E1 + a*E2 + a*E3 + E4 + E5; N =~ N1 + b*N2 + b*N3 + N4 + N5"),
  constraint = list(model = "E =~ E1 + l2*E2 + l3*E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5; l2 == l3 + 0.2"),
  variances = list(model = "E =~ E1 + E2 + E3; E1 ~~ v*E1; E2 ~~ v*E2; E3 ~~ v*E3"),
  combined = list(model = paste("E =~ E1 + a*E2 + a*E3 + l4*E4; N =~ N1 + N2 + l3*N3 + N4; A =~ A1 + A2 + A3;",
                                "def1 := l4*2; N ~ r*E; l4 == -l3 - 0.4; A ~ s*N; ind := r*s; N2 ~~ v*N2;",
                                "N4 ~~ v*N4")),
  stdlv = list(model = "E =~ a*E1 + a*E2 + a*E3 + E4 + E5", std.lv = TRUE),
  fixed_label = list(model = "E =~ a*E1 + a*E3 + E2 + E4 + E5"),  # E3 is fixed at E1's 1 by the label
  fiml = list(model = "E =~ E1 + a*E2 + a*E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", missing = "ml"))
for (case in names(labels_cases)) {
  k <- labels_cases[[case]]
  fit <- if (identical(k$missing, "ml")) {
    lavaan::sem(k$model, data = labels_fiml_data, missing = "ML", meanstructure = TRUE)
  } else {
    lavaan::sem(k$model, data = labels_data, std.lv = isTRUE(k$std.lv))
  }
  pe <- lavaan::parameterEstimates(fit, standardized = TRUE, rsquare = TRUE)
  ref(paste0("labels_", case, "_parameters"), data.frame(row = row.names(pe), pe, check.names = FALSE))
  fm <- lavaan::fitMeasures(fit)
  ref(paste0("labels_", case, "_fit"), data.frame(measure = names(fm), value = as.numeric(fm)))
  mi <- suppressWarnings(lavaan::modificationIndices(fit, standardized = TRUE, cov.std = TRUE,
                                                     information = "expected", power = TRUE, delta = 0.1,
                                                     alpha = 0.05, high.power = 0.75, sort. = TRUE,
                                                     minimum.value = 0, free.remove = FALSE, na.remove = TRUE,
                                                     op = NULL))
  ref(paste0("labels_", case, "_modindices"), data.frame(row = row.names(mi), mi, check.names = FALSE))
}
# rwf's report_cfa asks for the user's == rows too (remove.eq = FALSE, remove.system.eq = TRUE)
labels_fit <- lavaan::sem(labels_cases$combined$model, data = labels_data)
labels_report <- lavaan::parameterEstimates(labels_fit, se = TRUE, zstat = TRUE, pvalue = TRUE, ci = TRUE,
                                            level = 0.95, boot.ci.type = "perc", standardized = TRUE, fmi = FALSE,
                                            remove.system.eq = TRUE, remove.eq = FALSE, remove.ineq = FALSE,
                                            remove.def = FALSE, rsquare = TRUE, add.attributes = TRUE)
ref("labels_combined_report_parameters", data.frame(row = row.names(labels_report), labels_report,
                                                    check.names = FALSE))
# rwf's "eq" diagrams, which need labels: a labelled covariance arc and a constraint; plot_cfa's keys
labels_layer_rows <- function(p) {  # as plots_layer_rows() above
  b <- ggplot2::ggplot_build(p)
  do.call(rbind, lapply(seq_along(b$data), function(i) {
    d <- b$data[[i]]
    data.frame(layer = i, geom = class(p$layers[[i]]$geom)[1],
               x = if (is.null(d$x)) (d$xmin + d$xmax) / 2 else d$x,
               y = if (is.null(d$y)) (d$ymin + d$ymax) / 2 else d$y,
               xend = if (is.null(d$xend)) NA else d$xend, yend = if (is.null(d$yend)) NA else d$yend,
               label = if (is.null(d$label)) NA else as.character(d$label))
  }))
}
labels_diagram_fit <- lavaan::sem("E =~ E1 + a*E3 + a*E5 + E2; N =~ N1 + b*N3 + l5*N5 + N2; E ~~ c*N; l5 == b - 0.2",
                                  data = labels_data)
for (layout in c("tree", "circle")) {
  p <- plot_cfa_gg(labels_diagram_fit, what = "eq", layout = layout)
  ref(paste0("labels_plot_cfa_gg_", layout, "_eq"), data.frame(title = p$labels$title, labels_layer_rows(p)))
}
ref("labels_plot_cfa_keys", data.frame(key = names(suppressMessages(plot_cfa(labels_diagram_fit)))))
##########################################################################################
# EFA METHODS AND ROTATIONS (psych::fa's fm, rotate and oblique.scores; rwf::report_efa with them)
##########################################################################################
# self-contained; the data are the LAVAAN ENGINE section's. psych::fa rotates from the identity and 19 random
# orthogonal starts (n.rotations = 20, R's random numbers) and keeps the start with most loadings below .15;
# pwf rotates from the identity, so GPArotation's rotations are made with n.rotations = 1. varimax, promax and
# equamax have no random starts in psych and use its defaults.
methods_data <- list(
  complete = utils::read.csv(file.path(out_dir, "lavaan_data.csv"))[, c(paste0("E", 1:5), paste0("N", 1:5))],
  missing = utils::read.csv(file.path(out_dir, "lavaan_fiml_data.csv")))
fa_long <- function(model, key) {
  block <- function(part, x) {
    if (is.null(x)) return(NULL)
    x <- as.matrix(unclass(x))
    data.frame(model = key, part = part, row = rep(seq_len(nrow(x)), ncol(x)),
               column = rep(seq_len(ncol(x)), each = nrow(x)), value = as.vector(x))
  }
  stats <- c(rms = model$rms, crms = model$crms, RMSEA = unname(model$RMSEA[1]), RMSEA.lower = unname(model$RMSEA[2]),
             RMSEA.upper = unname(model$RMSEA[3]), TLI = model$TLI, CFI = model$CFI, STATISTIC = model$STATISTIC,
             dof = model$dof, objective = model$objective, BIC = model$BIC, SABIC = model$SABIC, EBIC = model$EBIC,
             ESABIC = model$ESABIC, fit = model$fit, fit.off = model$fit.off, null.chisq = model$null.chisq)
  rbind(block("loadings", model$loadings), block("Phi", model$Phi), block("weights", model$weights),
        block("Vaccounted", model$Vaccounted), block("communality", model$communality),
        block("communalities", model$communalities), block("uniquenesses", model$uniquenesses),
        block("complexity", model$complexity), block("values", model$values), block("e.values", model$e.values),
        data.frame(model = key, part = "stats", row = seq_along(stats), column = 1, value = unname(stats)))
}
fa_methods <- list()
fa_columns <- list()
fa_errors <- list()
fit_case <- function(key, ...) {
  model <- tryCatch(suppressMessages(suppressWarnings(psych::fa(...))), error = function(e) e)
  if (inherits(model, "error")) {
    fa_errors[[key]] <<- data.frame(model = key, error = conditionMessage(model))
    return(NULL)
  }
  fa_methods[[key]] <<- fa_long(model, key)
  fa_columns[[key]] <<- data.frame(model = key, columns = paste(colnames(model$loadings), collapse = ","))
  model
}
# every extraction method, oblimin from the identity, rwf's oblique.scores = TRUE
for (data_case in names(methods_data)) {
  for (fm in c("minres", "uls", "ols", "wls", "gls", "ml", "minchi", "old.min", "pa", "alpha")) {
    for (nfactors in 1:3) {
      fit_case(paste(data_case, fm, nfactors, "oblimin", sep = "|"), methods_data[[data_case]], nfactors = nfactors,
               fm = fm, rotate = "oblimin", n.rotations = 1, oblique.scores = TRUE)
    }
  }
}
# every rotation of the complete data, minres (psych's default extraction), psych's default scores
for (rotate in c("none", "varimax", "Varimax", "quartimax", "bentlerT", "geominT", "equamax", "promax", "Promax",
                 "oblimin", "quartimin", "simplimax", "geominQ", "bentlerQ")) {
  for (nfactors in 2:3) {
    n_rotations <- if (rotate %in% c("varimax", "promax", "equamax")) 20 else 1
    fit_case(paste("rotation", "minres", nfactors, rotate, sep = "|"), methods_data$complete, nfactors = nfactors,
             fm = "minres", rotate = rotate, n.rotations = n_rotations)
  }
}
fit_case("rotation|minres|2|foo", methods_data$complete, nfactors = 2, rotate = "foo")
fit_case("complete|foo|2|oblimin", methods_data$complete, nfactors = 2, fm = "foo", rotate = "oblimin",
         n.rotations = 1, oblique.scores = TRUE)
ref("psych_fa_methods", do.call(rbind, unname(fa_methods)))
ref("psych_fa_methods_columns", do.call(rbind, unname(fa_columns)))
ref("psych_fa_methods_errors", do.call(rbind, unname(fa_errors)))
# rwf::report_efa on models of other methods (rwf's examples use oblique.scores = TRUE); scores of two models
for (fm in c("pa", "ols", "wls")) {
  model <- suppressMessages(psych::fa(methods_data$complete, nfactors = 2, fm = fm, rotate = "oblimin",
                                      n.rotations = 1, oblique.scores = TRUE))
  report <- report_efa(model, methods_data$complete, scores = TRUE)
  for (part in c("loadings", "correlations", "residual_stats", "weights", "scores")) {
    ref(paste0("report_efa_methods_", fm, "_", part), data.frame(report[[part]], check.names = FALSE))
  }
}
##########################################################################################
# R'S OPTIM L-BFGS-B (pwf._lbfgsb, which pwf's EFA uses: par, value, counts, convergence and message)
##########################################################################################
# self-contained. Generated on the reference BLAS, as R on Windows: an R linked to OpenBLAS sums some dot
# products in another order, which changes the last bits of the iterates (on Linux: LD_PRELOAD of
# libblas.so.3 and liblapack.so.3)
lbfgsb_rosen <- function(x) {
  n <- length(x)
  sum(100 * (x[-1] - x[-n]^2)^2 + (1 - x[-n])^2)
}
lbfgsb_rosen_gr <- function(x) {
  n <- length(x)
  g <- numeric(n)
  g[-n] <- -400 * x[-n] * (x[-1] - x[-n]^2) - 2 * (1 - x[-n])
  g[-1] <- g[-1] + 200 * (x[-1] - x[-n]^2)
  g
}
lbfgsb_quad <- function(x) sum(seq_along(x) * (x - cos(seq_along(x)))^2)
lbfgsb_quad_gr <- function(x) 2 * seq_along(x) * (x - cos(seq_along(x)))
lbfgsb_cases <- list(
  rosen2 = list(par = c(-1.2, 1), fn = lbfgsb_rosen, gr = lbfgsb_rosen_gr),
  rosen5_bounds = list(par = rep(-1, 5), fn = lbfgsb_rosen, gr = lbfgsb_rosen_gr, lower = -0.5, upper = 0.8),
  rosen4_numerical = list(par = c(0.5, 0.5, 0.5, 0.5), fn = lbfgsb_rosen, lower = 0),
  rosen6_maxit = list(par = rep(-1, 6), fn = lbfgsb_rosen, gr = lbfgsb_rosen_gr, maxit = 5),
  quad8_scaled = list(par = rep(0, 8), fn = lbfgsb_quad, gr = lbfgsb_quad_gr,
                      lower = c(-Inf, 0, 0.5, -Inf, -1, 0, -Inf, 0.2), upper = c(Inf, Inf, 1, 0, 1, 0.1, Inf, 0.3),
                      parscale = c(1, 2, 0.5, 1, 1, 10, 1, 0.01), fnscale = 2),
  quad3_maximize = list(par = c(1, 1, 1), fn = function(x) -lbfgsb_quad(x), gr = function(x) -lbfgsb_quad_gr(x),
                        fnscale = -1, upper = 0.4),
  quad4_numerical_on_bound = list(par = c(0, 1, 0, 0), fn = lbfgsb_quad, lower = 0, upper = 1, ndeps = rep(1e-4, 4)),
  rosen3_lmm_factr = list(par = c(-1, 2, -1), fn = lbfgsb_rosen, gr = lbfgsb_rosen_gr, lmm = 3, factr = 1e3,
                          pgtol = 1e-8))
lbfgsb_results <- lbfgsb_par <- list()
for (case in names(lbfgsb_cases)) {
  x <- lbfgsb_cases[[case]]
  control <- x[intersect(names(x), c("maxit", "parscale", "fnscale", "ndeps", "lmm", "factr", "pgtol"))]
  fit <- stats::optim(x$par, x$fn, x$gr, method = "L-BFGS-B", lower = if (is.null(x$lower)) -Inf else x$lower,
                      upper = if (is.null(x$upper)) Inf else x$upper, control = control)
  lbfgsb_results[[case]] <- data.frame(case = case, value = fit$value, fn_count = unname(fit$counts[1]),
                                       gr_count = unname(fit$counts[2]), convergence = fit$convergence,
                                       message = fit$message)
  lbfgsb_par[[case]] <- data.frame(case = case, index = seq_along(fit$par), par = fit$par)
}
ref("optim_lbfgsb_results", do.call(rbind, unname(lbfgsb_results)))
ref("optim_lbfgsb_par", do.call(rbind, unname(lbfgsb_par)))
##########################################################################################
# LAVAAN OBSERVED PREDICTORS (fixed.x, lavaan's dummy latent variables, auto.cov.y; rwf::report_cfa, plot_cfa_gg)
##########################################################################################
# self-contained: respondents of the ocean data with a plausible age and gender 1 or 2, and a copy with values
# removed (age too: lavaan then drops those cases)
x_ocean <- utils::read.delim(file.path(data_dir, "ocean.csv"), stringsAsFactors = FALSE)
x_data <- x_ocean[x_ocean$age < 100 & x_ocean$gender %in% 1:2, ][1:500, c("age", "gender", paste0("E", 1:5),
                                                                        paste0("N", 1:5))]
rownames(x_data) <- NULL
ref("lavaan_x_data", x_data)
set.seed(7)
x_missing <- x_data
x_missing$age[sample(500, 20)] <- NA
x_missing$E2[sample(500, 30)] <- NA
x_missing$N1[sample(500, 25)] <- NA
ref("lavaan_x_missing_data", x_missing)
x_cases <- list(
  mimic = list(model = "E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender"),
  mimic_direct = list(model = "E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender; E5 ~ age"),
  two_outcomes = list(model = "E =~ E1 + E2 + E3; N =~ N1 + N2 + N3; E ~ age; N ~ age + gender"),
  mediation = list(model = "E =~ E1 + E2 + E3; N =~ N1 + N2 + N3; N ~ E + age; E ~ gender"),
  path = list(model = "E1 ~ age + gender; E2 ~ E1 + age"),
  ov_outcome = list(model = "E =~ E2 + E3 + E4; E1 ~ E + age"),
  ov_mediator = list(model = "E =~ E1 + E2 + E3 + E4 + E5; N1 ~ age; E ~ N1"),
  random_x = list(model = "E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender", fixed.x = FALSE),
  user_x = list(model = "E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender; age ~~ gender"),
  fiml = list(model = "E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3; E ~ age + gender; N ~ E", missing = "ml"),
  stdlv = list(model = "E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender", std.lv = TRUE),
  labels = list(model = "E =~ E1 + E2 + E3; N =~ N1 + N2 + N3; N ~ b*E + age; E ~ a*gender; ab := a*b"))
x_rows <- list(parameters = list(), fit = list(), rsquare = list(), mi = list(), inspect = list(), predict = list())
for (case in names(x_cases)) {
  x <- x_cases[[case]]
  data <- if (identical(x$missing, "ml")) x_missing else x_data
  fit <- suppressWarnings(do.call(lavaan::sem, c(list(model = x$model, data = data), x[setdiff(names(x), "model")])))
  # rwf::report_cfa's arguments
  pe <- lavaan::parameterEstimates(fit, se = TRUE, zstat = TRUE, pvalue = TRUE, ci = TRUE, level = 0.95,
                                   boot.ci.type = "perc", standardized = TRUE, fmi = FALSE, remove.system.eq = TRUE,
                                   remove.eq = FALSE, remove.ineq = FALSE, remove.def = FALSE, rsquare = TRUE,
                                   add.attributes = TRUE)
  x_rows$parameters[[case]] <- data.frame(case = case, row = rownames(pe), data.frame(pe, check.names = FALSE),
                                          check.names = FALSE)
  fm <- lavaan::fitMeasures(fit)
  x_rows$fit[[case]] <- data.frame(case = case, measure = names(fm), value = unname(as.numeric(fm)))
  r2 <- lavaan::lavInspect(fit, "rsquare")
  x_rows$rsquare[[case]] <- data.frame(case = case, name = names(r2), value = unname(r2))
  mi <- lavaan::modificationIndices(fit, standardized = TRUE, cov.std = TRUE, information = "expected", power = TRUE,
                                    delta = 0.1, alpha = 0.05, high.power = 0.75, sort. = TRUE, minimum.value = 0,
                                    free.remove = FALSE, na.remove = TRUE, op = NULL)
  x_rows$mi[[case]] <- data.frame(case = case, row = as.integer(rownames(mi)), data.frame(mi))
  for (what in c("est", "std")) {
    matrices <- lavaan::lavInspect(fit, what)
    for (name in names(matrices)) {
      a <- unclass(matrices[[name]])
      x_rows$inspect[[paste(case, what, name)]] <- data.frame(
        case = case, what = what, matrix = name, row = rep(rownames(a), ncol(a)),
        column = rep(colnames(a), each = nrow(a)), value = as.vector(a))
    }
  }
  scores <- lavaan::lavPredict(fit)
  if (ncol(scores) > 0) {
    x_rows$predict[[case]] <- data.frame(case = case, row = rep(seq_len(nrow(scores)), ncol(scores)),
                                         column = rep(colnames(scores), each = nrow(scores)),
                                         value = as.vector(scores), nobs = lavaan::lavInspect(fit, "nobs"))
  }
}
for (kind in names(x_rows)) ref(paste0("lavaan_x_", kind), do.call(plyr::rbind.fill, unname(x_rows[[kind]])))
# rwf::report_cfa and its path diagrams of a model with a direct effect of a predictor on an indicator
x_layer_rows <- function(p) {  # as plots_layer_rows() of the PATH DIAGRAMS section
  b <- ggplot2::ggplot_build(p)
  do.call(rbind, lapply(seq_along(b$data), function(i) {
    d <- b$data[[i]]
    data.frame(layer = i, geom = class(p$layers[[i]]$geom)[1],
               x = if (is.null(d$x)) (d$xmin + d$xmax) / 2 else d$x,
               y = if (is.null(d$y)) (d$ymin + d$ymax) / 2 else d$y,
               xend = if (is.null(d$xend)) NA else d$xend, yend = if (is.null(d$yend)) NA else d$yend,
               label = if (is.null(d$label)) NA else as.character(d$label))
  }))
}
x_fit <- lavaan::sem(x_cases$mimic_direct$model, data = x_data)
invisible(capture.output(suppressWarnings(x_report <- report_cfa(x_fit))))
ref("lavaan_x_report_parameters", data.frame(row = rownames(x_report$parameters), x_report$parameters,
                                             check.names = FALSE))
ref("lavaan_x_report_r_squared", data.frame(name = rownames(x_report$r_squared), x_report$r_squared))
for (layout in c("tree", "circle")) for (what in c("std", "est")) {
  p <- plot_cfa_gg(x_fit, what = what, layout = layout)
  ref(paste0("lavaan_x_plot_cfa_gg_", layout, "_", what), data.frame(title = p$labels$title, x_layer_rows(p)))
}
ref("lavaan_x_plot_cfa_keys", data.frame(key = names(suppressMessages(plot_cfa(x_fit)))))
##########################################################################################
# LAVAAN MEAN STRUCTURES (meanstructure = TRUE with complete data, intercepts in the syntax, df = 0)
##########################################################################################
# self-contained; the data are the LAVAAN ENGINE and LAVAAN OBSERVED PREDICTORS sections'
ms_data <- list(ld = utils::read.csv(file.path(out_dir, "lavaan_data.csv")),
                lm = utils::read.csv(file.path(out_dir, "lavaan_fiml_data.csv")),
                xd = utils::read.csv(file.path(out_dir, "lavaan_x_data.csv")))
ms_cases <- list(
  cfa = list(data = "ld", model = "E =~ E1 + E2 + E3 + E4 + E5; N =~ N1 + N2 + N3 + N4 + N5", meanstructure = TRUE),
  sem = list(data = "ld", model = paste("E =~ E1 + E2 + 0.8*E3 + E4; N =~ N1 + N2 + N3 + N4; A =~ A1 + A2 + A3;",
                                        "A ~ E + N; E1 ~~ E2"), meanstructure = TRUE),
  mimic = list(data = "xd", model = "E =~ E1 + E2 + E3 + E4 + E5; E ~ age + gender; E5 ~ age", meanstructure = TRUE),
  path = list(data = "xd", model = "E1 ~ age + gender; E2 ~ E1 + age", meanstructure = TRUE),
  stdlv = list(data = "ld", model = "E =~ E1 + E2 + E3 + E4 + E5", meanstructure = TRUE, std.lv = TRUE),
  latent_mean = list(data = "ld", model = "E =~ E1 + E2 + E3 + E4 + E5; E1 ~ 0*1; E ~ 1"),
  equal_intercepts = list(data = "ld", model = "E =~ E1 + E2 + E3 + E4 + E5; E2 ~ a*1; E4 ~ a*1; E5 ~ 3*1",
                          meanstructure = TRUE),
  intercept_only = list(data = "ld", model = "E =~ E1 + E2 + E3; E1 ~ 1"),
  fiml_latent_mean = list(data = "lm", model = "E =~ E1 + E2 + E3 + E4 + E5; E1 ~ 0*1; E ~ 1", missing = "ml"))
ms_rows <- list(parameters = list(), fit = list(), rsquare = list(), mi = list(), inspect = list(), predict = list())
for (case in names(ms_cases)) {
  x <- ms_cases[[case]]
  fit <- suppressWarnings(do.call(lavaan::sem, c(list(model = x$model, data = ms_data[[x$data]]),
                                                 x[setdiff(names(x), c("model", "data"))])))
  # rwf::report_cfa's arguments
  pe <- lavaan::parameterEstimates(fit, se = TRUE, zstat = TRUE, pvalue = TRUE, ci = TRUE, level = 0.95,
                                   boot.ci.type = "perc", standardized = TRUE, fmi = FALSE, remove.system.eq = TRUE,
                                   remove.eq = FALSE, remove.ineq = FALSE, remove.def = FALSE, rsquare = TRUE,
                                   add.attributes = TRUE)
  ms_rows$parameters[[case]] <- data.frame(case = case, row = rownames(pe), data.frame(pe, check.names = FALSE),
                                           check.names = FALSE)
  fm <- lavaan::fitMeasures(fit)
  ms_rows$fit[[case]] <- data.frame(case = case, measure = names(fm), value = unname(as.numeric(fm)))
  r2 <- lavaan::lavInspect(fit, "rsquare")
  ms_rows$rsquare[[case]] <- data.frame(case = case, name = names(r2), value = unname(r2))
  mi <- suppressWarnings(lavaan::modificationIndices(fit, standardized = TRUE, cov.std = TRUE,
                                                     information = "expected", power = TRUE, delta = 0.1,
                                                     alpha = 0.05, high.power = 0.75, sort. = TRUE,
                                                     minimum.value = 0, free.remove = FALSE, na.remove = TRUE,
                                                     op = NULL))
  ms_rows$mi[[case]] <- data.frame(case = case, row = as.integer(rownames(mi)), data.frame(mi))
  for (what in c("est", "std")) {
    matrices <- lavaan::lavInspect(fit, what)
    for (name in names(matrices)) {
      a <- unclass(matrices[[name]])
      ms_rows$inspect[[paste(case, what, name)]] <- data.frame(
        case = case, what = what, matrix = name, row = rep(rownames(a), ncol(a)),
        column = rep(colnames(a), each = nrow(a)), value = as.vector(a))
    }
  }
  scores <- suppressWarnings(lavaan::lavPredict(fit))
  if (ncol(scores) > 0) {  # with missing data, a case without answers is a row of NA
    ms_rows$predict[[case]] <- data.frame(case = case, row = rep(seq_len(nrow(scores)), ncol(scores)),
                                          column = rep(colnames(scores), each = nrow(scores)),
                                          value = as.vector(scores), nobs = lavaan::lavInspect(fit, "nobs"))
  }
}
for (kind in names(ms_rows)) ref(paste0("lavaan_ms_", kind), do.call(plyr::rbind.fill, unname(ms_rows[[kind]])))
# rwf::report_cfa of a model with a free latent mean
ms_fit <- lavaan::sem(ms_cases$latent_mean$model, data = ms_data$ld)
invisible(capture.output(suppressWarnings(ms_report <- report_cfa(ms_fit))))
ref("lavaan_ms_report_parameters", data.frame(row = rownames(ms_report$parameters), ms_report$parameters,
                                              check.names = FALSE))
