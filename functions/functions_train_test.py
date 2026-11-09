# -*- coding: utf-8 -*-
"""
Classification evaluation plots (ROC, confusion matrix heatmap,
separability) and confusion-matrix/accuracy statistics.

Note: confusion() and confusion_matrix_percent() take `observed`/
`predicted` as Python lists specifically, not numpy arrays -- `levels =
sorted(set(observed + predicted))` concatenates the two lists, which
silently does elementwise addition instead if either argument is an
ndarray (wrong, and no error is raised).

Note: this file's confusion()/confusion_matrix_percent()/
plot_separability() overlap with newer, independently-built versions in
functions_train_test_full.py from this project's R-port work. Both
files currently coexist.

Fixed a real bug in proportion_accurate(): its "off-diagonal accuracy"
computation indexed into np.nonzero(cmatrix) positionally assuming at
least as many nonzero cells as matrix rows, which isn't guaranteed --
its own docstring example (observed with an 11 category predicted
values don't share) crashed with an IndexError. Rewritten to directly
sum the diagonal and its immediate neighbor cells (correct-or-off-by-
one-category), matching the apparent intent without the fragile assumption.
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
    scale_fill_gradient, labs, theme, theme_bw, element_text, element_blank,
    scale_fill_identity, scale_color_identity, scale_x_continuous, scale_y_continuous,
    coord_fixed, theme_minimal,
)
##########################################################################################
# PLOT ROC
##########################################################################################
def plot_roc(observed, predicted, base_size=10, title=""):
    """
    Plot the Receiver Operating Characteristic (ROC) curve from observed and predicted values.

    Parameters:
    observed (array-like): List or array of true binary labels.
    predicted (array-like): List or array of predicted scores or probabilities.
    base_size (int, optional): Base size for the plot. Defaults to 10.
    title (str, optional): Title of the plot. Defaults to an empty string.

    Returns:
    ggplot: A ggplot object representing the ROC curve.

    Notes:
    - The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
    - The Area Under the Curve (AUC) is calculated and displayed in the plot caption.
    - The plot includes a reference diagonal line indicating the performance of a random classifier.

    Examples:
    >>> observed = [0, 0, 1, 1]
    >>> predicted = [0.1, 0.4, 0.35, 0.8]
    >>> plot = plot_roc(observed, predicted)
    >>> print(plot)
    """
    fpr, tpr, _ = roc_curve(observed, predicted)
    roc_auc = auc(fpr, tpr)
    # Create a DataFrame with false positive rate and true positive rate
    df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    # Create the ROC plot
    p = (ggplot(df, aes(x='FPR', y='TPR'))+ 
         geom_line(color='darkorange')+
         geom_abline(linetype='--', color='navy')+
         labs(title=f'ROC {title}',
                x='False Positive Rate', 
                y='True Positive Rate',
                caption=f'AUC = {roc_auc * 100:.2f}%\nObservations = {len(observed)}')+
         theme_bw(base_size=base_size))
    return p
##########################################################################################
# PLOT CONFUSION
##########################################################################################
def plot_confusion(observed, predicted, base_size=10, title=""):
    """
    Plot a confusion matrix heatmap using ggplot (plotnine) in Python.

    Parameters:
    observed (list or array-like): Vector of observed (true) class labels.
    predicted (list or array-like): Vector of predicted class labels.
    base_size (int, optional): Integer value representing the base font size for the plot. Defaults to 10.
    title (str, optional): String representing the title of the plot. Defaults to an empty string.

    Returns:
    plotnine.ggplot.ggplot: A ggplot object representing the confusion matrix heatmap.

    Example usage:
    observed = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    predicted = [1, 2, 3, 1, 1, 3, 1, 2, 2]
    p = plot_confusion(observed, predicted, title="Confusion Matrix")
    p.show()
    """
    cm=confusion(observed=observed,predicted=predicted)
    pa=proportion_accurate(observed=observed,predicted=predicted)
    labels = np.unique(np.concatenate((observed, predicted)))
    
    # Create DataFrame
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_melted = cm_df.reset_index().melt(id_vars='index')
    cm_df.reset_index().melt(id_vars='index')

    total=cm_melted['value'].sum()
    diagonal=cm_df.values.diagonal().sum()
    
    # Generate Plot
    p = (ggplot(cm_melted, aes(x='index', y='variable', fill='value')) +
         geom_tile(color="gray") +
         geom_text(aes(x="index",y="variable",label="value"),color="black",size=base_size)+
         scale_fill_gradient(low='#FFFFFF', high='#4169E1') +
         labs(title=f"{title}", x="Observed", y="Predicted", fill="Count",
              caption=f"Total={total}\nDiagonal={diagonal}") +
         theme_bw(base_size=base_size) +
         theme(axis_text_x=element_text(rotation=45, hjust=1),
               axis_ticks_x=element_blank(),
               axis_ticks_y=element_blank(),
               panel_grid_major=element_blank(),
               panel_grid_minor=element_blank(),
               panel_border=element_blank(),
               panel_background=element_blank(),
               legend_position="none"))

    return p
##########################################################################################
# PLOT SEPARABILITY
##########################################################################################
def plot_separability(observed, predicted, base_size=10, title=""):
    """
    Plot separability showing the density distribution of predicted probabilities for different observed categories.

    Parameters:
    observed (list or array-like): Vector of observed (true) class labels.
    predicted (list or array-like): Vector of predicted outcome probabilities.
    base_size (int, optional): Integer value representing the base font size for the plot. Defaults to 10.
    title (str, optional): String representing the title of the plot. Defaults to an empty string.

    Returns:
    plotnine.ggplot.ggplot: A ggplot object representing the separability plot.

    Example usage:
    df1 = pd.DataFrame(np.random.rand(1000, 2), columns=['X1', 'X2'])
    df1['X1'] = np.where(np.abs(df1['X1']) < 0.5, 0, 1)
    df1['X2'] = (df1['X2'] - df1['X2'].min()) / (df1['X2'].max() - df1['X2'].min())
    p = plot_separability(observed=round(np.abs(df1['X1']), 0), predicted=np.abs(df1['X2']))
    p.show()
    """
    
    df = pd.DataFrame({'observed': pd.Categorical(observed), 'predicted': predicted})

    p = (ggplot(df, aes(x='predicted', color='observed'))
         + geom_density(size=1)
         + labs(title=f"{title}",
                color="Observed",
                caption=f"Observations: {len(df)}")
         + theme_bw(base_size=base_size))
    
    return p
##########################################################################################
# CONFUSION
##########################################################################################
def confusion(observed, predicted):
    """
    Create a confusion matrix from observed and predicted vectors.

    Parameters:
    observed (list): List of observed variables.
    predicted (list): List of predicted variables.

    Returns:
    pd.DataFrame: Confusion matrix with observed values as columns and predicted values as rows.

    Examples:
    >>> confusion(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11])

    >>> confusion(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2])
    """
    levels = sorted(set(observed + predicted))
    cat_type = CategoricalDtype(categories=levels, ordered=True)
    
    data_observed = pd.Categorical(observed, categories=levels, ordered=True)
    data_predicted = pd.Categorical(predicted, categories=levels, ordered=True)
    
    result = pd.crosstab(index=data_predicted, columns=data_observed, rownames=['predicted'], colnames=['observed'], dropna=False)
    return result
##########################################################################################
# CONFUSION MATRIX PERCENT
##########################################################################################
def confusion_matrix_percent(observed, predicted):
    """
    Create a confusion matrix with row and column percentages from observed and predicted vectors.

    Parameters:
    observed (list): List of observed variables.
    predicted (list): List of predicted variables.

    Returns:
    pd.DataFrame: Confusion matrix with observed values as columns and predicted values as rows, including row and column percentages.

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
    >>> observed = [1, 2, 3, 4, 5, 10]
    >>> predicted = [1, 2, 3, 4, 5, 11]
    >>> confusion_matrix_percent(observed, predicted)
    
    >>> observed = [1, 2, 2, 2, 2]
    >>> predicted = [1, 1, 2, 2, 2]
    >>> confusion_matrix_percent(observed, predicted)
    """

    levels = sorted(set(observed + predicted))
    cat_type = CategoricalDtype(categories=levels, ordered=True)
    
    data_observed = pd.Categorical(observed, categories=levels, ordered=True)
    data_predicted = pd.Categorical(predicted, categories=levels, ordered=True)
    
    cmatrix = pd.crosstab(index=data_predicted, columns=data_observed, rownames=['predicted'], colnames=['observed'], dropna=False)
    
    overall_accuracy = cmatrix.values.diagonal().sum() / cmatrix.sum().sum()
    cmatrix['sum'] = cmatrix.sum(axis=1)
    cmatrix.loc['sum'] = cmatrix.sum()
    
    dimensions = cmatrix.shape

    pr1 = []
    pr2 = []
    for i in range(len(levels)):
        if cmatrix.iloc[i, dimensions[1]-1] != 0:
            p1 = cmatrix.iloc[i, i] / cmatrix.iloc[i, dimensions[1]-1]
        else:
            p1 = 0
        if cmatrix.iloc[dimensions[0]-1, i] != 0:
            p2 = cmatrix.iloc[i, i] / cmatrix.iloc[dimensions[0]-1, i]
        else:
            p2 = 0
        pr1.append(p1)
        pr2.append(p2)
     
    
    pr1.append(np.array(pr1).sum()/len(np.array(pr1)))
    pr2.append(np.array(pr2).sum()/len(np.array(pr2)))
    pr2.append(overall_accuracy)
    
    cmatrix['p']=np.array(pr1)
    cmatrix.loc['p']=np.array(pr2)
    
    cmatrix = cmatrix.fillna(0)
    cmatrix = cmatrix.round(2)
    return cmatrix
##########################################################################################
# PROPORTION ACCURATE
##########################################################################################
def proportion_accurate(observed, predicted):
    """
    Calculate the proportion overall accuracy of a confusion matrix and Cohen's kappa statistics.

    Parameters:
    observed (list): List of observed variables.
    predicted (list): List of predicted variables.

    Returns:
    pd.DataFrame: DataFrame with overall accuracy, off-diagonal accuracy, and Cohen's kappa statistics.

    Examples:
    >>> observed = [1, 2, 3, 4, 5, 10]
    >>> predicted = [1, 2, 3, 4, 5, 11]
    >>> result = proportion_accurate(observed, predicted)
    >>> print(result)
    """
    
    # Create confusion matrix
    cmatrix = confusion_matrix(observed, predicted)
    train_test = pd.DataFrame({'observed': observed, 'predicted': predicted})
    
    # Calculate overall accuracy (diagonal proportion)
    cm_diagonal = np.trace(cmatrix) / np.sum(cmatrix)
    
    # Calculate off-diagonal accuracy (diagonal plus its immediate neighbors,
    # i.e. "correct or off-by-one category" -- meaningful for ordinal labels)
    n = len(cmatrix)
    data = []
    for i in range(n):
        data.append(cmatrix[i, i])
        if i + 1 < n:
            data.append(cmatrix[i, i + 1])
        if i - 1 >= 0:
            data.append(cmatrix[i, i - 1])
    cm_off_diagonal = sum(data) / np.sum(cmatrix)
    
    # Calculate Cohen's kappa statistics
    kappa_unweighted = cohen_kappa_score(observed, predicted, weights=None)
    kappa_linear = cohen_kappa_score(observed, predicted, weights='linear')
    kappa_squared = cohen_kappa_score(observed, predicted, weights='quadratic')
    
    # Create result DataFrame
    result = pd.DataFrame({
        'cm_diagonal': [cm_diagonal],
        'cm_off_diagonal': [cm_off_diagonal],
        'kappa_unweighted': [kappa_unweighted],
        'kappa_linear': [kappa_linear],
        'kappa_squared': [kappa_squared]
    })

    return result
##########################################################################################
# PLOT CONFUSION EXTENDED
##########################################################################################
def _r_label(value):
    """R's as.character() of a class label: 1.0 -> "1"."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    return str(value)


def _r_color_ramp(low, high, n):
    """
    grDevices::colorRampPalette(c(low, high))(n): linear RGB steps as upper-case
    hex. R's rgb(..., maxColorValue = 255) truncates fractional channel values.
    """
    low = np.array([int(low[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    high = np.array([int(high[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    steps = np.linspace(0, 1, n)[:, None]
    rgb = np.floor(low + (high - low) * steps + 1e-9).astype(int)
    return ["#{:02X}{:02X}{:02X}".format(*row) for row in rgb]


def plot_confusion_extended(observed, predicted, positive=None, base_size=10, title=""):
    """
    2x2 confusion matrix with its row, column and overall measures (port of
    rwf::plot_confusion_extended): counts shaded blue by size, precision /
    false discovery rate and NPV / false omission rate beside the predicted
    rows, sensitivity / miss rate and specificity / fall-out under the
    observed columns, and accuracy, prevalence and the proportion
    incorrectly classified in the corner.

    plotnine cannot put the x axis on top, so the "Observed" labels sit
    below the grid (above it in rwf). The plot's data (p.data) holds the
    same cells, labels and colours as rwf's.

    Parameters:
    observed (array-like): True class labels, 2 distinct values.
    predicted (array-like): Predicted class labels, same 2 values.
    positive (optional): Label of the positive class. Defaults to the second
        sorted label, as in rwf::confusion().
    base_size (float, optional): Base font size. Defaults to 10.
    title (str, optional): Added after "Confusion Matrix" in the title.

    Returns:
    plotnine.ggplot: The plot.

    Examples:
    >>> plot_confusion_extended(observed=[0, 0, 0, 1, 1, 1, 1], predicted=[0, 0, 0, 1, 1, 1, 0])
    >>> plot_confusion_extended(observed=[0, 0, 0, 1, 1, 1, 1], predicted=[0, 0, 0, 1, 1, 1, 0], positive=0)
    """
    observed = np.array([_r_label(v) for v in observed])
    predicted = np.array([_r_label(v) for v in predicted])
    lvls = sorted(set(observed) | set(predicted))
    if len(lvls) != 2:
        raise ValueError("plot_confusion_extended() only supports a binary (2-class) confusion matrix")
    positive = lvls[1] if positive is None else _r_label(positive)
    negative = next(level for level in lvls if level != positive)

    TP = int(np.sum((predicted == positive) & (observed == positive)))
    TN = int(np.sum((predicted == negative) & (observed == negative)))
    FP = int(np.sum((predicted == positive) & (observed == negative)))
    FN = int(np.sum((predicted == negative) & (observed == positive)))
    total = TP + TN + FP + FN

    def ratio(a, b):
        return a / b if b else np.nan

    accuracy, prevalence, picc = (TP + TN) / total, (TP + FN) / total, (FN + FP) / total
    sensitivity, miss_rate = ratio(TP, TP + FN), ratio(FN, TP + FN)
    specificity, fallout = ratio(TN, FP + TN), ratio(FP, FP + TN)
    precision, fdr = ratio(TP, TP + FP), ratio(FP, TP + FP)
    npv, for_rate = ratio(TN, FN + TN), ratio(FN, FN + TN)

    def pct(x):
        return "NaN%" if np.isnan(x) else f"{x * 100:.1f}%"

    seq_ramp = _r_color_ramp("#cde2fb", "#0d366b", 100)

    def fill_for(n):
        return seq_ramp[int(max(1, min(100, round(n / total * 99) + 1))) - 1]

    ink_primary, ink_secondary, margin_fill = "#0b0b0b", "#52514e", "#f0efec"
    cells = pd.DataFrame({
        "col": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "row": [3, 3, 3, 2, 2, 2, 1, 1, 1],
        "kind": ["main", "main", "margin", "main", "main", "margin", "margin", "margin", "corner"],
        "n": [TP, FP, np.nan, FN, TN, np.nan, np.nan, np.nan, np.nan],
        "label": [
            f"TP\n{TP}\n({pct(TP / total)})",
            f"FP\n{FP}\n({pct(FP / total)})",
            f"Precision\n{pct(precision)}\nFDR {pct(fdr)}",
            f"FN\n{FN}\n({pct(FN / total)})",
            f"TN\n{TN}\n({pct(TN / total)})",
            f"NPV\n{pct(npv)}\nFOR {pct(for_rate)}",
            f"Sensitivity\n{pct(sensitivity)}\nMiss Rate {pct(miss_rate)}",
            f"Specificity\n{pct(specificity)}\nFall-out {pct(fallout)}",
            f"Accuracy {pct(accuracy)}\nPrevalence {pct(prevalence)}\nIncorrect {pct(picc)}",
        ],
    })
    main = cells["kind"] == "main"
    cells["fill"] = [fill_for(n) if is_main else margin_fill for n, is_main in zip(cells["n"], main)]
    cells["text_color"] = np.where(main & cells["n"].notna() & (cells["n"] / total > 0.5), "#ffffff", ink_primary)

    col_labels = [f"Observed: {positive}", f"Observed: {negative}", ""]
    row_labels = ["", f"Predicted: {negative}", f"Predicted: {positive}"]
    return (ggplot(cells, aes(x="col", y="row"))
            + geom_tile(aes(fill="fill"), color="#fcfcfb", size=2, width=0.96, height=0.96)
            + geom_text(aes(label="label", color="text_color"), size=base_size * 0.95, fontweight="bold")
            + scale_fill_identity()
            + scale_color_identity()
            + scale_x_continuous(breaks=[1, 2, 3], labels=col_labels, expand=(0.02, 0.02))
            + scale_y_continuous(breaks=[1, 2, 3], labels=row_labels, expand=(0.02, 0.02))
            + coord_fixed()
            + labs(title=f"Confusion Matrix {title}", x="", y="", caption=f"Observations: {total}")
            + theme_minimal(base_size=base_size)
            + theme(axis_text=element_text(color=ink_secondary, weight="bold"),
                    axis_ticks=element_blank(),
                    panel_grid=element_blank(),
                    plot_title=element_text(color=ink_primary, weight="bold"),
                    plot_caption=element_text(color=ink_secondary)))
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    np.random.seed(0)

    print("=" * 80, "\nplot_roc\n", "=" * 80, sep="")
    observed_roc = [0, 0, 1, 1]
    predicted_roc = [0.1, 0.4, 0.35, 0.8]
    plot_roc(observed_roc, predicted_roc).save("plot_roc_example.png", verbose=False)
    print("saved plot_roc_example.png")

    print("\n" + "=" * 80, "\nplot_confusion\n", "=" * 80, sep="")
    observed_cm = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    predicted_cm = [1, 2, 3, 1, 1, 3, 1, 2, 2]
    plot_confusion(observed_cm, predicted_cm, title="Confusion Matrix").save(
        "plot_confusion_example.png", verbose=False)
    print("saved plot_confusion_example.png")

    print("\n" + "=" * 80, "\nplot_separability\n", "=" * 80, sep="")
    df1 = pd.DataFrame(np.random.rand(1000, 2), columns=['X1', 'X2'])
    df1['X1'] = np.where(np.abs(df1['X1']) < 0.5, 0, 1)
    df1['X2'] = (df1['X2'] - df1['X2'].min()) / (df1['X2'].max() - df1['X2'].min())
    plot_separability(observed=round(df1['X1'], 0), predicted=df1['X2']).save(
        "plot_separability_example.png", verbose=False)
    print("saved plot_separability_example.png")

    print("\n" + "=" * 80, "\nconfusion / confusion_matrix_percent\n", "=" * 80, sep="")
    print(confusion(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11]))
    print(confusion_matrix_percent(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2]))

    print("\n" + "=" * 80, "\nproportion_accurate\n", "=" * 80, sep="")
    print(proportion_accurate(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11]))
    print(proportion_accurate(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2]))

