# eda_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# The figure for Numerical variables
def create_fig_num(title,data):
    plt.close('all')
    fig = plt.figure(figsize = (6.5,2.5))
    plt.subplot(1, 2, 1)
    data.hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data)
    plt.title('Univaries Analysis of '+ title,loc='left')
    return fig

# The figure for Categorical variables
def create_fig_cat(title,data):
    plt.close('all')
    fig =plt.figure(figsize = (6,2))
    plt.title('Bar plot for categorical variables in the dataset')
    sns.countplot(x = title, data = data,order = data[title].value_counts().index)
    return fig

# The heatmap for the correlation matrix
def corr_heatmap(corr):
    plt.close('all')
    fig = plt.figure(figsize = (4,4))
    mask = np.zeros_like(corr,dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.palette="vlag"

    sns.heatmap(corr,annot=True,  mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    return fig