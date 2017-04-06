# -*- coding: utf-8 -*-
"""
@author: plkovac & Jiayi
"""

import numpy as np
from scipy.sparse import csr_matrix
from functions import get_dataset_2 as start_2
import matplotlib.pyplot as plt
from functions import func

#==============================================================================
# Data retrieval
#==============================================================================

wdir='/Users/paulvernhet/Desktop/3A/SEMINAIRE 1000mercis/week_4'

first = False
if first:
    X, y, l = start_2.get_dataset()
    X_csr = csr_matrix(X)
    func.save_sparse_csr('Cookies', X_csr )
    np.savez('labels', y=y)
else :
    X_csr = func.load_sparse_csr('Cookies.npz')
    X = X_csr.toarray() # pour retrouver la matrice dense
    dataL = np.load('labels.npz')
    y = dataL['y']

#==============================================================================
# I - FEATURES CLUSTERING
# Histogram division
#==============================================================================

ind_useful = X.sum(axis = 0) != 0 # remove useless features
X = X[:,ind_useful]

mask_cspp = y.astype(bool) # csp+ mask

n_cspp = X[mask_cspp,:].sum(axis = 0) # number of csp+ visits
n_cspm = X[~mask_cspp,:].sum(axis = 0) # number of csp- visits
site_contrast = (n_cspp - n_cspm) * 1.0 / (n_cspp + n_cspm) # score

num_inters = 25 # number of intervals - 1 in histogram
percentiles = False
if percentiles :
    inter = np.percentile(site_contrast, np.linspace(0,100,num_inters+1) )
else :
    inter = np.linspace(-1,1,num_inters+1)
plt.hist(site_contrast, bins = num_inters)
plt.xlabel('Intervals')
plt.ylabel('Scores')
plt.title('Histogram of scores according to sites clusters')

#classification of each site into each interval 
count = 0
site_clf = np.zeros((1,X.shape[1]))
for low,high in zip(inter[:num_inters],inter[1:]):
    if low==-1 :
        site_clf = site_clf + count * np.logical_and(site_contrast <= high, site_contrast >= low)
        
    else:
        site_clf = site_clf + count * np.logical_and(site_contrast <= high, site_contrast > low)
    count = count + 1
    site_clf = site_clf.astype(int)
    
#representation of data : n_samples x n_intervals    
X_new = np.zeros((X.shape[0],num_inters))
for feat in np.arange(num_inters):
    ind = (site_clf == feat)
    X_new[:,feat] = X[:,ind[0,:]].sum(axis = 1)
    
plt.bar(np.arange(num_inters),X_new.sum(axis = 0) )
plt.xlabel('Intervals')
plt.ylabel('# of visits')
plt.title('Histogram of visits according to sites clusters')

Nbr = X_new.sum(axis = 0)
#==============================================================================
# Results for algorithms
#==============================================================================

repeat_time = 6
C = 0.01
score_XGB = func.XGboost_crossvalid(X_new, y, repeat_time = repeat_time)
score_LR = func.logreg_crossvalid(X_new, y, repeat_time = repeat_time)
score_LSVM = func.LinSVM_crossvalid(X_new, y, C_best = C, repeat_time = repeat_time)
score_NBM = func.naiveBayesM_crossvalid(X_new, y, repeat_time = repeat_time)
score_RF = func.RandomForest_crossvalid(X_new, y, repeat_time = repeat_time)
 
#==============================================================================
# Results for algorithms
#==============================================================================
#XGboost : Mean Accuracy: 0.677 +/- 0.004
#XGboost : Mean sensibility: 0.653 +/- 0.019
#XGboost : Mean specificity: 0.700 +/- 0.016
#LogistRegress : Mean Accuracy: 0.660 +/- 0.002
#LogistRegress : Mean sensibility: 0.732 +/- 0.017
#LogistRegress : Mean specificity: 0.592 +/- 0.015
#LinSVM : Mean Accuracy: 0.654 +/- 0.007
#LinSVM : Mean sensibility: 0.597 +/- 0.079
#LinSVM : Mean specificity: 0.709 +/- 0.071
#Multinomial Naives Bayes : Mean Accuracy: 0.654 +/- 0.006
#Multinomial Naives Bayes : Mean sensibility: 0.579 +/- 0.009
#Multinomial Naives Bayes : Mean specificity: 0.725 +/- 0.007
#RandomForest : Mean Accuracy: 0.658 +/- 0.007
#RandomForest : Mean sensibility: 0.630 +/- 0.010
#RandomForest : Mean specificity: 0.685 +/- 0.007