import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def summary_statistics(df):
    desc = pd.DataFrame(index=list(df))
    desc['type'] = df.dtypes
    desc['count'] = df.count()
    desc['nunique'] = df.nunique()
    desc['%unique'] = desc['nunique'] / len(df) * 100
    desc['null'] = df.isnull().sum()
    desc['%null'] = desc['null'] / len(df) * 100
    desc['min'] = df.min()
    desc['max'] = df.max()
    return desc

def heatmap(dataset, method, label = None):
    corr = dataset.corr(method = method)
    plt.figure(figsize = (18, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask = mask, cmap = 'viridis', annot = True, annot_kws = {'size' : 10})
    plt.title(f'{label}', fontsize = 18, weight = 'bold')
    plt.show()

def feature_distance(dataset, method, label):
    #thanks to @sergiosaharovsky for the fix
    corr = dataset.corr(method = method)
    dist_linkage = linkage(squareform(1 - abs(corr)), 'complete')
    
    plt.figure(figsize = (15, 8))
    dendro = dendrogram(dist_linkage, labels=dataset.columns, leaf_rotation=90)
    plt.title(f'Feature Distance in {label} Dataset', weight = 'bold', size = 20)
    plt.show()