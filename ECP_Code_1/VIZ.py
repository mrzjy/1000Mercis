# -*- coding: utf-8 -*-
"""
@author: plkovac & Jiayi
March 2017
Visualizations on datasets (agglomeration, features selection, nonlinear PCA)
"""

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from functions import func
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, normalize
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import chi2, f_classif
from scipy.stats import ttest_ind
import time
import colorsys

#==============================================================================
# Data retrieval
#==============================================================================

def Chi2_stat(X,y, log = True, fig = False):

    Chi2, pval = chi2(X, y)
    inter = [np.percentile(Chi2, qq) for qq in np.linspace(0, 100, 6)]
    count = 0
    ch_features = np.zeros((1, len(Chi2)))
    for low, high in zip(inter[:len(inter)], inter[1:]):
        if low == -1:
            ch_features += count * np.logical_and(Chi2 <= high, Chi2 >= low)
        else:
            ch_features += count * np.logical_and(Chi2 <= high, Chi2 > low)
        count += 1
    ch_features = ch_features.astype(int)

    if fig:
        fig, ax1 = plt.subplots(1)
        t = np.arange(len(pval))
        Chi2.sort()
        pval.sort()
        ax1.plot(t,Chi2[::-1], 'b-')
        ax1.set_xlabel('features')
        ax1.set_ylabel('Chi2', color='b')
        ax1.tick_params('y', colors='b')
        if log:
            ax1.set_yscale('log')
        ax2 = ax1.twinx()
        ax2.plot(t, pval, 'r-')
        ax2.set_ylabel('pvalue', color='r')
        ax2.tick_params('y', colors='r')
        if log:
            ax2.set_yscale('log')
        fig.tight_layout()
        plt.title(' Chi2 - test on cleaned dataset ')
        plt.show()

    return ch_features

def F_stat(X,y, log = True, fig = False):

    F, pval = f_classif(X, y)
    inter = [np.percentile(F, qq) for qq in np.linspace(0, 100, 6)]
    count = 0
    f_features = np.zeros((1, len(F)))
    for low, high in zip(inter[:len(inter)], inter[1:]):
        if low == -1:
            f_features += count * np.logical_and(F <= high, F >= low)
        else:
            f_features += count * np.logical_and(F <= high, F > low)
        count += 1
    f_features = f_features.astype(int)

    if fig:
        fig, ax1 = plt.subplots(1)
        t = np.arange(len(pval))
        F.sort()
        pval.sort()
        ax1.plot(t,F[::-1], 'b-')
        ax1.set_xlabel('features')
        ax1.set_ylabel('F', color='b')
        ax1.tick_params('y', colors='b')
        if log:
            ax1.set_yscale('log')
        ax2 = ax1.twinx()
        ax2.plot(t, pval, 'r-')
        ax2.set_ylabel('pvalue', color='r')
        ax2.tick_params('y', colors='r')
        if log:
            ax2.set_yscale('log')
        fig.tight_layout()
        plt.title(' F - test on cleaned dataset ')
        plt.show()

    return f_features

def data_cleaning(X, preprocess = 'std'):
    # X is in sparse csr format
    ### 1/ remove empty rows
    num_nonzeros = np.diff(X.indptr)
    X_t = X[num_nonzeros != 0]
    ### 2/ remove empty columns
    X_t = X_t[:, np.unique(X_t.nonzero()[1])]

    ### Standardizations
    if preprocess == 'std':
        preprocessing = StandardScaler(with_mean=False)
        X_t = preprocessing.fit_transform(X_t)
    elif preprocess == 'max':
        preprocessing = MaxAbsScaler()
        X_t = preprocessing.fit_transform(X_t)
    elif preprocess == 'full_std':
        preprocessing = StandardScaler()
        X_t = X_t.toarray()
        X_t = preprocessing.fit_transform(X_t)
    elif preprocess == 'norm':
        X_t = normalize(X_t.toarray(), axis=0, norm='l1')
        print 'norm'
    return X_t

def global_aggregation_viz(X, mask_cspp, ch_features):

    n_cspp = X[mask_cspp, :].sum(axis=0)   # number of csp+ visits
    n_cspm = X[~mask_cspp, :].sum(axis=0)  # number of csp- visits
    site_contrast = (n_cspp - n_cspm) * 1.0 / (n_cspp + n_cspm)  # score

    # Color generation = 5 for simplicity's sake
    N = 5
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    q1 = ch_features == 0
    q2 = ch_features == 1
    q3 = ch_features == 2
    q4 = ch_features == 3
    q5 = ch_features == 4

    p1 = plt.scatter(np.asarray(n_cspm[0, q1[0]]), np.asarray(n_cspp[0, q1[0]]), marker='o', color=colors[0])
    p2 = plt.scatter(np.asarray(n_cspm[0, q2[0]]), np.asarray(n_cspp[0, q2[0]]), marker='o', color=colors[1])
    p3 = plt.scatter(np.asarray(n_cspm[0, q3[0]]), np.asarray(n_cspp[0, q3[0]]), marker='o', color=colors[2])
    p4 = plt.scatter(np.asarray(n_cspm[0, q4[0]]), np.asarray(n_cspp[0, q4[0]]), marker='o', color=colors[3])
    p5 = plt.scatter(np.asarray(n_cspm[0, q5[0]]), np.asarray(n_cspp[0, q5[0]]), marker='o', color=colors[4])
    plt.legend((p1, p2, p3, p4, p5),
              ('Q1','Q2','Q3','Q4','Q5'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("Projection of features in score plan")
    plt.xlabel("Negative contrast")
    plt.ylabel("Positive contrast")
    plt.show()

def global_aggregation_viz_blank(X, mask_cspp,ch_features):

    n_cspp = X[mask_cspp, :].sum(axis=0)  # number of csp+ visits
    n_cspm = X[~mask_cspp, :].sum(axis=0)  # number of csp- visits
    site_contrast = (n_cspp - n_cspm) * 1.0 / (n_cspp + n_cspm)  # score

    # Color generation = 5 for simplicity's sake
    N = 5
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    q1 = ch_features == 0
    q2 = ch_features == 1
    q3 = ch_features == 2
    q4 = ch_features == 3
    q5 = ch_features == 4

    p1 = plt.scatter(np.asarray(n_cspm[q1[0]]), np.asarray(n_cspp[q1[0]]), marker='o', color=colors[0])
    p2 = plt.scatter(np.asarray(n_cspm[q2[0]]), np.asarray(n_cspp[q2[0]]), marker='o', color=colors[1])
    p3 = plt.scatter(np.asarray(n_cspm[q3[0]]), np.asarray(n_cspp[q3[0]]), marker='o', color=colors[2])
    p4 = plt.scatter(np.asarray(n_cspm[q4[0]]), np.asarray(n_cspp[q4[0]]), marker='o', color=colors[3])
    p5 = plt.scatter(np.asarray(n_cspm[q5[0]]), np.asarray(n_cspp[q5[0]]), marker='o', color=colors[4])
    plt.legend((p1, p2, p3, p4, p5),
              ('Q1','Q2','Q3','Q4','Q5'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title("Projection of features in score plan")
    plt.xlabel("Negative contrast")
    plt.ylabel("Positive contrast")
    plt.show()

def global_aggregation_viz_black(X, mask_cspp):

    n_cspp = X[mask_cspp, :].sum(axis=0)  # number of csp+ visits
    n_cspm = X[~mask_cspp, :].sum(axis=0)  # number of csp- visits

    plt.scatter(np.asarray(n_cspm), np.asarray(n_cspp), marker='o')
    plt.title("Projection of features in score plan")
    plt.xlabel("Negative contrast")
    plt.ylabel("Positive contrast")
    plt.show()

if __name__ == '__main__':
    wdir = 'path_to_dataset'

    ###########################################################################
    # Creating/loading matrix data
    ###########################################################################

    print 'loading data ...'
    X_csr = func.load_sparse_csr('Cookies.npz')
    dataL = np.load('labels.npz')
    y = dataL['y']
    print 'Cleaning and normalizing ...'
    X = data_cleaning(X_csr, preprocess='max')
    print 'Data scaled and treated : %d cookies / %d features'%(X.shape[0],X.shape[1])

    mask_cspp = y.astype(bool)  # selection des csp+
    clean_mask = X.sum( axis=0).astype(bool)

    ###########################################################################
    # FEATURE SELECTION
    ###########################################################################
    ch_features = Chi2_stat(X, y, log = True, fig = True)

    ###########################################################################
    # CUSTOM AGGREGATION
    ###########################################################################
    custom = True
    if custom:
        global_aggregation_viz(X, mask_cspp, ch_features )
        global_aggregation_viz_blank(X, mask_cspp , ch_features)
        global_aggregation_viz_black(X, mask_cspp)
