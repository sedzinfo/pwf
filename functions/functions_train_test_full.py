# -*- coding: utf-8 -*-
"""
Full, self-contained port of R rwf::FUNCTIONS_TRAIN_TEST.R (train/test
splitting, confusion-matrix analysis, and related plots), independent of
the simpler versions already in functions_train_test.py.

Differences from the R originals, by design:
  - k_fold/k_sample take explicit `predictors`/`outcome` column-name lists
    instead of an R model formula (Python has no native formula object).
  - All xgboost::xgb.DMatrix preparation is dropped (xgboost unavailable
    in this environment) — the train/test/validation split logic itself
    is preserved in full.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score
from plotnine import (
    ggplot, aes, geom_line, geom_abline, geom_tile, geom_text, geom_density,
    geom_vline, labs, theme_bw, theme, element_blank, scale_fill_gradient,
    coord_fixed,
)
##########################################################################################
# CONFUSION MATRIX
##########################################################################################
def confusion(observed, predicted):
    """
    Create a confusion matrix from observed and predicted vectors.

    Parameters:
    observed (list): List of observed variables.
    predicted (list): List of predicted variables.

    Returns:
    pd.DataFrame: Confusion matrix with observed values as columns and
    predicted values as rows.

    Examples:
    >>> confusion(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11])
    >>> confusion(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2])
    """
    levels = sorted(set(list(observed) + list(predicted)))
    data_observed = pd.Categorical(observed, categories=levels, ordered=True)
    data_predicted = pd.Categorical(predicted, categories=levels, ordered=True)
    result = pd.crosstab(index=data_predicted, columns=data_observed,
                          rownames=['predicted'], colnames=['observed'], dropna=False)
    return result
##########################################################################################
# CONFUSION MATRIX PERCENT
##########################################################################################
def confusion_matrix_percent(observed, predicted):
    """
    Create a confusion matrix with row and column percentages from
    observed and predicted vectors.

    Parameters:
    observed (list): List of observed variables.
    predicted (list): List of predicted variables.

    Returns:
    pd.DataFrame: Confusion matrix with an appended "sum" row/column
    (totals) and "p" row/column (per-class recall / precision, with the
    bottom-right cell holding overall accuracy).

    Notes:
    - Total measures:
      - Accuracy: (TP + TN) / total
      - Prevalence: (TP + FN) / total
      - Proportion Incorrectly Classified: (FN + FP) / total
    - Horizontal measures:
      - True Positive Rate (Sensitivity): TP / (TP + FN)
      - True Negative Rate (Specificity): TN / (FP + TN)
      - False Negative Rate (Miss Rate): FN / (TP + FN)
      - False Positive Rate (Fall-out): FP / (FP + TN)
    - Vertical measures:
      - Positive Predictive Value (Precision): TP / (TP + FP)
      - Negative Predictive Value: TN / (FN + TN)
      - False Omission Rate: FN / (FN + TN)
      - False Discovery Rate: FP / (TP + FP)

    Examples:
    >>> confusion_matrix_percent(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11])
    >>> confusion_matrix_percent(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2])
    """
    levels = sorted(set(list(observed) + list(predicted)))
    data_observed = pd.Categorical(observed, categories=levels, ordered=True)
    data_predicted = pd.Categorical(predicted, categories=levels, ordered=True)
    cmatrix = pd.crosstab(index=data_predicted, columns=data_observed,
                           rownames=['predicted'], colnames=['observed'], dropna=False)

    overall_accuracy = cmatrix.values.diagonal().sum() / cmatrix.sum().sum()
    cmatrix['sum'] = cmatrix.sum(axis=1)
    cmatrix.loc['sum'] = cmatrix.sum()

    dimensions = cmatrix.shape
    pr1, pr2 = [], []
    for i in range(len(levels)):
        p1 = cmatrix.iloc[i, dimensions[1]-1] and cmatrix.iloc[i, i] / cmatrix.iloc[i, dimensions[1]-1] or 0
        p2 = cmatrix.iloc[dimensions[0]-1, i] and cmatrix.iloc[i, i] / cmatrix.iloc[dimensions[0]-1, i] or 0
        pr1.append(p1)
        pr2.append(p2)

    pr1.append(np.array(pr1).sum() / len(pr1))
    pr2.append(np.array(pr2).sum() / len(pr2))
    pr2.append(overall_accuracy)

    cmatrix['p'] = np.array(pr1)
    cmatrix.loc['p'] = np.array(pr2)

    cmatrix = cmatrix.fillna(0)
    cmatrix = cmatrix.round(2)
    return cmatrix
##########################################################################################
# PROPORTION ACCURATE
##########################################################################################
def proportion_accurate(observed, predicted):
    """
    Calculate the proportion overall accuracy of a confusion matrix and
    Cohen's kappa statistics.

    Parameters:
    observed (list): List of observed variables.
    predicted (list): List of predicted variables.

    Returns:
    pd.DataFrame: One row with cm_diagonal (overall accuracy),
    cm_off_diagonal (accuracy allowing off-by-one misclassification),
    kappa_unweighted, kappa_linear, kappa_squared.

    Examples:
    >>> proportion_accurate(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11])
    """
    cmatrix = confusion_matrix(observed, predicted)
    cm_diagonal = np.trace(cmatrix) / np.sum(cmatrix)

    data = []
    for row in range(len(cmatrix)):
        nonzero_cols = np.nonzero(cmatrix[row])[0]
        for col in nonzero_cols:
            if 0 <= col < len(cmatrix):
                data.append(cmatrix[row, col])
            if 0 <= col + 1 < len(cmatrix):
                data.append(cmatrix[row, col + 1])
            if 0 <= col - 1 < len(cmatrix):
                data.append(cmatrix[row, col - 1])
    cm_off_diagonal = sum(data) / np.sum(cmatrix)

    kappa_unweighted = cohen_kappa_score(observed, predicted, weights=None)
    kappa_linear = cohen_kappa_score(observed, predicted, weights='linear')
    kappa_squared = cohen_kappa_score(observed, predicted, weights='quadratic')

    return pd.DataFrame({
        'cm_diagonal': [cm_diagonal],
        'cm_off_diagonal': [cm_off_diagonal],
        'kappa_unweighted': [kappa_unweighted],
        'kappa_linear': [kappa_linear],
        'kappa_squared': [kappa_squared],
    })
##########################################################################################
# PLOT ROC
##########################################################################################
def plot_roc(observed, predicted, base_size=10, title=""):
    """
    Plot ROC curves from observed outcomes and predicted probabilities,
    for both choices of which class is treated as "positive" — the
    Python equivalent of R's reversed/non-reversed pROC::roc() orderings.

    Parameters:
    observed (array-like): True binary class labels (exactly 2 unique values).
    predicted (array-like): Predicted probabilities for the positive class.
    base_size (int, optional): Base font size for the plot. Defaults to 10.
    title (str, optional): Plot title suffix. Defaults to "".

    Returns:
    dict: {str(positive_level): plotnine.ggplot}, one entry per class
    treated as positive, each captioned with AUC, control level, and
    positive level.

    Examples:
    >>> observed = [0, 0, 1, 1]
    >>> predicted = [0.1, 0.4, 0.35, 0.8]
    >>> plots = plot_roc(observed, predicted)
    >>> list(plots.values())[0].show()
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    levels = sorted(pd.unique(observed))
    if len(levels) != 2:
        raise ValueError("plot_roc requires exactly two unique observed classes")

    plotlist = {}
    for pos_label in (levels[1], levels[0]):
        control_level = levels[0] if pos_label == levels[1] else levels[1]
        fpr, tpr, _ = roc_curve(observed, predicted, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        p = (ggplot(df, aes(x='FPR', y='TPR'))
             + geom_line(color='darkorange')
             + geom_abline(linetype='--', color='navy')
             + labs(title=f"ROC {title}",
                    x="False Positive Rate", y="True Positive Rate",
                    caption=(f"Observations:{len(observed)}"
                             f"\nAUC:{roc_auc*100:.2f}%"
                             f"\nControl Level:{control_level}"
                             f"\nPositive Level:{pos_label}"))
             + theme_bw(base_size=base_size)
             + coord_fixed())
        plotlist[str(pos_label)] = p
    return plotlist
##########################################################################################
# PLOT CONFUSION
##########################################################################################
def plot_confusion(observed, predicted, base_size=10, title=""):
    """
    Plot a confusion matrix heatmap with observed (x, descending order)
    against predicted (y, ascending order), counts annotated on each
    tile, and overall/off-diagonal accuracy plus Cohen's kappa
    (unweighted/linear/squared) in the caption.

    Parameters:
    observed (list or array-like): True class labels.
    predicted (list or array-like): Predicted class labels.
    base_size (int, optional): Base font size for the plot. Defaults to 10.
    title (str, optional): Plot title suffix. Defaults to "".

    Returns:
    plotnine.ggplot: The confusion matrix heatmap.

    Examples:
    >>> plot_confusion(observed=[1,2,3,1,2,3], predicted=[1,2,3,1,2,3])
    """
    cmatrix = confusion(observed=observed, predicted=predicted)
    pa = proportion_accurate(observed=observed, predicted=predicted)
    observations = int(cmatrix.values.sum())

    melted = cmatrix.reset_index().melt(id_vars='predicted', var_name='observed', value_name='value')
    obs_levels = sorted(melted['observed'].unique(), reverse=True)
    pred_levels = sorted(melted['predicted'].unique())
    melted['observed'] = pd.Categorical(melted['observed'], categories=obs_levels, ordered=True)
    melted['predicted'] = pd.Categorical(melted['predicted'], categories=pred_levels, ordered=True)

    p = (ggplot(melted, aes(x='observed', y='predicted', fill='value'))
         + geom_tile(color='white')
         + geom_text(aes(x='observed', y='predicted', label='value'), color='black', size=base_size/2)
         + theme_bw(base_size=base_size)
         + theme(axis_ticks_x=element_blank(),
                 axis_ticks_y=element_blank(),
                 panel_grid_major=element_blank(),
                 panel_grid_minor=element_blank(),
                 panel_border=element_blank(),
                 panel_background=element_blank(),
                 legend_position='none')
         + scale_fill_gradient(low='#fafaff', high='#132B43')
         + labs(title=f"Confusion Matrix {title}",
                caption=(f"Observations:{observations}"
                         f"\nAccuracy:{round(float(pa['cm_diagonal'].iloc[0]), 2)}"
                         f"\nAccuracy with off diagonals:{round(float(pa['cm_off_diagonal'].iloc[0]), 2)}"
                         f"\nKappa unweighted:{round(float(pa['kappa_unweighted'].iloc[0]), 2)}"
                         f"\nKappa linear:{round(float(pa['kappa_linear'].iloc[0]), 2)}"
                         f"\nKappa squared:{round(float(pa['kappa_squared'].iloc[0]), 2)}"))
         + coord_fixed())
    return p
##########################################################################################
# PLOT SEPARABILITY
##########################################################################################
def plot_separability(observed, predicted, base_size=10, title=""):
    """
    Plot the density distribution of predicted probabilities for each
    observed category, to visualize how well predictions separate the
    classes.

    Parameters:
    observed (list or array-like): True class labels.
    predicted (list or array-like): Predicted probabilities.
    base_size (int, optional): Base font size for the plot. Defaults to 10.
    title (str, optional): Plot title suffix. Defaults to "".

    Returns:
    plotnine.ggplot: The separability density plot.

    Examples:
    >>> plot_separability(observed=[0,0,1,1], predicted=[0.1,0.2,0.8,0.9])
    """
    df = pd.DataFrame({'observed': pd.Categorical(observed), 'predicted': predicted})
    p = (ggplot(df, aes(x='predicted', color='observed'))
         + geom_density(size=1)
         + labs(title=f"Predicted proportion vs Observed category {title}",
                color="observed",
                caption=f"Observations:{len(df)}")
         + theme_bw(base_size=base_size))
    return p
##########################################################################################
# CONFUSION MATRIX PERFORMANCE
##########################################################################################
def result_confusion_performance(observed, predicted, step=0.1, base_size=10, title=""):
    """
    Evaluate confusion-matrix performance across a range of cut-off
    points on `predicted`, find the cut-off that maximizes the mean of
    the per-class row/column proportions, and plot the performance curve
    with a vertical line at the optimum.

    Parameters:
    observed (array-like): True binary class labels.
    predicted (array-like): Predicted probabilities/scores to threshold.
    step (float, optional): Step size between tested cut-off points. Defaults to 0.1.
    base_size (int, optional): Base font size for the plot. Defaults to 10.
    title (str, optional): Plot title suffix. Defaults to "".

    Returns:
    dict: {
        'plot_performance': plotnine.ggplot,
        'cut_performance': pd.DataFrame (one row per tested cut point),
        'cut': float or list of float (optimal cut point(s)),
        'confusion_matrix': pd.DataFrame (confusion_matrix_percent at the
            optimal cut point),
    }

    Examples:
    >>> result_confusion_performance(observed=[0,0,1,1,0,1], predicted=[0.1,0.2,0.9,0.8,0.3,0.7], step=0.1)
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted, dtype=float)
    min_predicted = np.nanmin(predicted)
    max_predicted = np.nanmax(predicted)

    rows = []
    cutpoints = np.arange(min_predicted, max_predicted + step, step)
    for cut in cutpoints:
        thresholded = np.where(predicted > cut, 1, 0)
        cmatrix = confusion_matrix_percent(observed=list(observed), predicted=list(thresholded))
        k = cmatrix.shape[0] - 2
        column_observed = cmatrix.iloc[0:k, -1].to_numpy()
        row_predicted = cmatrix.iloc[-1, 0:k].to_numpy()
        overall = cmatrix.iloc[-1, -1]
        row = {'cut_point': cut, 'Overall': overall}
        for j, val in enumerate(column_observed):
            row[f'Column_Observed_{j}'] = val
        for j, val in enumerate(row_predicted):
            row[f'Row_Predicted_{j}'] = val
        rows.append(row)

    df_cut_performance = pd.DataFrame(rows)
    value_cols = [c for c in df_cut_performance.columns if c != 'cut_point']
    cp = df_cut_performance.melt(id_vars='cut_point', value_vars=value_cols)

    df_cut_performance['Mean_proportion'] = df_cut_performance[value_cols].mean(axis=1, skipna=True)
    best_mean = df_cut_performance['Mean_proportion'].max()
    mcp = df_cut_performance.loc[df_cut_performance['Mean_proportion'] == best_mean, 'cut_point'].tolist()

    optimal_cut = float(np.mean(mcp))
    thresholded_optimal = np.where(predicted > optimal_cut, 1, 0)
    cmatrixp = confusion_matrix_percent(observed=list(observed), predicted=list(thresholded_optimal))

    plot_performance = (
        ggplot(cp, aes(x='cut_point', y='value', color='variable'))
        + geom_line(size=1)
        + geom_vline(xintercept=mcp, size=1)
        + labs(title=f"Confusion Matrix Performance {title}",
               x="Cut Point", y="Proportion Correct",
               caption=f"Observations:{len(df_cut_performance)}\nCut point:{round(optimal_cut, 4)}")
        + theme(legend_title=element_blank())
        + theme_bw(base_size=base_size)
    )

    return {
        'plot_performance': plot_performance,
        'cut_performance': df_cut_performance,
        'cut': mcp if len(mcp) > 1 else mcp[0],
        'confusion_matrix': cmatrixp,
    }
##########################################################################################
# K-FOLD TRAIN TEST SPLIT
##########################################################################################
def k_fold(df, predictors, outcome, k=10):
    """
    Split a dataframe into k folds for cross-validation. Each fold serves
    as a test set once, with the remaining k-1 folds as the training set.

    Unlike R's k_fold, this takes explicit `predictors`/`outcome` column
    names instead of a model formula (Python has no formula object), and
    does not build xgboost DMatrix objects (xgboost unavailable here) —
    the fold index/train/test split logic is otherwise unchanged.

    Parameters:
    df (pd.DataFrame): Data to split.
    predictors (list of str): Predictor column names.
    outcome (str): Outcome column name.
    k (int, optional): Number of folds. Defaults to 10.

    Returns:
    dict: {
        'f': {'index': {...}, 'train': {...}, 'test': {...},
              'x_test': {...}, 'y_test': {...}}, one entry per fold
              keyed "f1".."fk",
        'index': np.ndarray of fold assignment per row,
        'predictors': list of str,
        'outcome': str,
        'variables': list of str (predictors + outcome),
    }

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': range(20), 'y': [0,1]*10})
    >>> result = k_fold(df, predictors=['x'], outcome='y', k=5)
    """
    variables = predictors + [outcome]
    df = df[variables].reset_index(drop=True)
    n = len(df)
    index = np.random.permutation(pd.cut(np.arange(1, n + 1), bins=k, labels=False))
    rows = np.arange(n)

    fold = {'index': {}, 'train': {}, 'test': {}, 'x_test': {}, 'y_test': {}}
    for i in range(k):
        k_index = f"f{i+1}"
        iteration_index = rows[index == i]
        fold['index'][k_index] = iteration_index
        train = df.iloc[np.setdiff1d(rows, iteration_index)]
        test = df.iloc[iteration_index]
        fold['train'][k_index] = train
        fold['test'][k_index] = test
        fold['x_test'][k_index] = test[predictors]
        fold['y_test'][k_index] = test[outcome]
        print(f"Fold Cases: {i+1} Train: {len(train)} Test: {len(test)} "
              f"Total: {len(train)+len(test)} Unique Train: {train.index.nunique()} "
              f"Unique Test: {test.index.nunique()}")

    return {
        'f': fold,
        'index': index,
        'predictors': predictors,
        'outcome': outcome,
        'variables': variables,
    }
##########################################################################################
# TRAIN TEST VALIDATION SAMPLE
##########################################################################################
def k_sample(df, predictors, outcome, k=1):
    """
    Split a dataframe into train, test, and validation sets, optionally
    across k folds. With k=1, this is a plain train/test/validation split.

    Unlike R's k_sample, this takes explicit `predictors`/`outcome` column
    names instead of a model formula, and does not build xgboost DMatrix
    objects (xgboost unavailable here) — the split logic is otherwise
    unchanged: each fold's non-train rows are split in half again into
    test and validation.

    Parameters:
    df (pd.DataFrame): Data to split.
    predictors (list of str): Predictor column names.
    outcome (str): Outcome column name.
    k (int, optional): Number of folds. Defaults to 1.

    Returns:
    dict: {
        'f': {'index': {'train':{},'test':{},'validation':{}},
              'train': {...}, 'test': {...}, 'validation': {...},
              'x_test': {...}, 'y_test': {...},
              'x_validation': {...}, 'y_validation': {...}},
        'index': np.ndarray of fold assignment per row,
        'predictors': list of str,
        'outcome': str,
        'variables': list of str (predictors + outcome),
    }

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': range(20), 'y': [0,1]*10})
    >>> result = k_sample(df, predictors=['x'], outcome='y', k=1)
    """
    def sv(arr):
        arr = np.asarray(arr)
        mid = int(np.ceil(len(arr) / 2))
        return arr[:mid], arr[mid:]

    variables = predictors + [outcome]
    df = df[variables].reset_index(drop=True)
    n = len(df)
    if k > 1:
        index = np.random.permutation(pd.cut(np.arange(1, n + 1), bins=k, labels=False))
    else:
        index = np.zeros(n, dtype=int)
    rows = np.arange(n)

    fold = {
        'index': {'train': {}, 'test': {}, 'validation': {}},
        'train': {}, 'test': {}, 'validation': {},
        'x_test': {}, 'y_test': {}, 'x_validation': {}, 'y_validation': {},
    }
    for i in range(k):
        k_index = f"fold{i+1}"
        iteration_index = np.random.permutation(rows[index == i])
        train_index, test_validation_index = sv(iteration_index)
        test_index, validation_index = sv(test_validation_index)

        fold['index']['train'][k_index] = rows_train = train_index
        fold['index']['test'][k_index] = test_index
        fold['index']['validation'][k_index] = validation_index

        train = df.iloc[train_index]
        test = df.iloc[test_index]
        validation = df.iloc[validation_index]

        fold['train'][k_index] = train
        fold['test'][k_index] = test
        fold['validation'][k_index] = validation
        fold['x_test'][k_index] = test[predictors]
        fold['y_test'][k_index] = test[outcome]
        fold['x_validation'][k_index] = validation[predictors]
        fold['y_validation'][k_index] = validation[outcome]

        print(f"Fold Cases: {i+1} Train: {len(train)} Test: {len(test)} "
              f"Validation: {len(validation)} Total: {len(train)+len(test)+len(validation)} "
              f"Unique Train: {train.index.nunique()} Unique Test: {test.index.nunique()} "
              f"Unique Validation: {validation.index.nunique()}")

    return {
        'f': fold,
        'index': index,
        'predictors': predictors,
        'outcome': outcome,
        'variables': variables,
    }
##########################################################################################
# SCALE AND DUMMY CODE
##########################################################################################
def recode_scale_dummy(df, categories=10):
    """
    Scale numeric columns to [0, 1] (min-max) and one-hot ("dummy") code
    character/categorical columns that have fewer than `categories`
    unique values.

    Parameters:
    df (pd.DataFrame): Data to process.
    categories (int, optional): A character/categorical column is dummy
        coded only if it has fewer than this many unique values. Defaults
        to 10.

    Returns:
    pd.DataFrame: Dummy-coded columns (if any) concatenated with the
    scaled numeric columns, preserving df's original row index.

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'numeric_var': [1,2,3,4,5], 'factor_var': ['A','B','A','B','C']})
    >>> recode_scale_dummy(df)
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = [
        c for c in df.columns
        if c not in numeric_cols and df[c].nunique(dropna=True) < categories
    ]

    scaled = pd.DataFrame(index=df.index)
    if numeric_cols:
        mins = df[numeric_cols].min()
        maxs = df[numeric_cols].max()
        scaled = (df[numeric_cols] - mins) / (maxs - mins)

    if cat_cols:
        dummy = pd.get_dummies(df[cat_cols].astype('category'), columns=cat_cols)
        result = pd.concat([dummy, scaled], axis=1)
    else:
        result = scaled

    result.index = df.index
    return result
