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

