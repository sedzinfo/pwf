##########################################################################################
# REFERENCE OUTPUTS FROM rwf FOR pwf PARITY TESTS: RECODE, KEYS, UNIX TIME, MATRIX, MATHEMATICAL, GENERATE DATA, GLM_IRT_U
##########################################################################################
# Runs rwf functions on the same CSVs pwf uses (pwf/data) and writes their results
# next to this script as CSV. tests/test_*.py compare the Python ports against them.
#
# Run from anywhere (needs rwf installed in R):
#   Rscript tests/reference/make_reference_utilities.R
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
