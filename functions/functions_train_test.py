##########################################################################################
# PLOT ROC
##########################################################################################
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from plotnine import ggplot, aes, geom_line, geom_abline, labs, theme_bw

def plot_roc(observed, predicted, base_size=10, title=""):
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



