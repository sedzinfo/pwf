##########################################################################################
# PLOT ROC
##########################################################################################
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from plotnine import ggplot, aes, geom_line, geom_abline, labs, theme_bw

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

# Example usage:
observed = np.round(np.abs(np.random.normal(0, 0.5, 100)))
predicted = np.abs(np.random.normal(0, 0.5, 100))
res=plot_roc(observed, predicted)
res.show()

df1=generate_normal(ncols=1,nrows=1000,mean=0,sd=1)
df1.describe()
df1[1] = np.where(np.abs(df1[1]) < 1, 0, 1
df1[0] = np.abs(df1[0])
df1[0] = (df1[0]-df1[0].min())/(df1[0].max()-df1[0].min())
observed=np.round(df1[1].values)
predicted=df1[0].values.round()

res=plot_roc(observed,predicted)
res.show()
##########################################################################################
# PLOT CONFUSION
##########################################################################################
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_tile, scale_fill_gradient, labs, theme, element_text, theme_bw, element_blank, geom_text
from sklearn.metrics import confusion_matrix

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

# Example usage
observed = [1, 2, 3, 1, 2, 3, 1, 2, 3]
predicted = [1, 2, 3, 1, 1, 3, 1, 2, 2]
p = plot_confusion(observed, predicted, title="Confusion Matrix")
p.show()
##########################################################################################
# PLOT SEPARABILITY
##########################################################################################
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_density, labs, theme_bw

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
    
    df = pd.DataFrame({'observed': observed, 'predicted': predicted})
    
    p = (ggplot(df, aes(x='predicted', color='factor(observed)'))
         + geom_density(size=1)
         + labs(title=f"{title}",
                color="Observed",
                caption=f"Observations: {len(df)}")
         + theme_bw(base_size=base_size))
    
    return p

# Example usage
df1 = pd.DataFrame(np.random.rand(1000, 2), columns=['X1', 'X2'])
df1['X1'] = np.where(np.abs(df1['X1']) < 0.5, 0, 1)
df1['X2'] = (df1['X2'] - df1['X2'].min()) / (df1['X2'].max() - df1['X2'].min())
p = plot_separability(observed=round(np.abs(df1['X1']), 0), predicted=np.abs(df1['X2']))
p.show()
##########################################################################################
# CONFUSION
##########################################################################################
import pandas as pd
from pandas.api.types import CategoricalDtype

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

# Examples
confusion(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11])
confusion(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2])
##########################################################################################
# CONFUSION MATRIX PERCENT
##########################################################################################
import pandas as pd
from pandas.api.types import CategoricalDtype

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

# Examples
confusion_matrix_percent(observed=[1, 2, 3, 4, 5, 10], predicted=[1, 2, 3, 4, 5, 11])
confusion_matrix_percent(observed=[1, 2, 2, 2, 2], predicted=[1, 1, 2, 2, 2])
##########################################################################################
# CONFUSION MATRIX PERCENT
##########################################################################################
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score

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
    
    # Calculate off-diagonal accuracy
    index = np.transpose(np.nonzero(cmatrix))
    data = []
    for i in range(len(cmatrix)):
        row, col = index[i]
        if 0 <= col < len(cmatrix):
            data.append(cmatrix[row, col])
        if 0 <= col + 1 < len(cmatrix):
            data.append(cmatrix[row, col + 1])
        if 0 <= col - 1 < len(cmatrix):
            data.append(cmatrix[row, col - 1])
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

# Example usage
observed = [1, 2, 3, 4, 5, 10]
predicted = [1, 2, 3, 4, 5, 11]
proportion_accurate(observed, predicted)

