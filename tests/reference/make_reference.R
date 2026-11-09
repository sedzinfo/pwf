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
