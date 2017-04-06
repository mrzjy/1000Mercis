# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 20:22:06 2017

@author: Kevin
"""

import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from math import floor
from sklearn.metrics import confusion_matrix
from sklearn import linear_model, svm
import os
import pandas as pd
import time
import scipy.sparse as sps
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, Binarizer, normalize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import colorsys
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

#==============================================================================
# SAVING / LOADING
#==============================================================================
# ========
def save_sparse_csr(path, filename, array):
    np.savez(os.path.join(path, filename), data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def save_full(path, filename, vector):
    np.savez(os.path.join(path, filename), data=vector)

# ========
def load_sparse_csr(path_to):
    loader = np.load(path_to)
    matrix = sps.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                            shape=loader['shape'])
    return matrix

def load_full(path_to):
    loader = np.load(path_to)
    array = loader['data']
    return array

# ========
def static_load_csr(path_to):
    X = load_sparse_csr(path_to + '_X.npz')
    y = load_full(path_to + '_y.npz')
    return X, y

# ========
def hyperParamSearch(X_train, y_train, X_test, y_test, clf_name="logistic", preprocess = 'Std', metric = 'euclidean'):

    # PREPROCESSING = FEATURES SCALING

    if preprocess == 'MaxMin':
        preprocessing = MaxAbsScaler()
        #preprocessing.fit(X_train)
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.fit_transform(X_test)

    if preprocess == 'Binarization':
        preprocessing = Binarizer()
        #preprocessing.fit(X_train)
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.fit_transform(X_test)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'Std':
        preprocessing = StandardScaler(with_mean=False)
        #preprocessing.fit(X_train)
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.fit_transform(X_test)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'full_std':
        preprocessing = StandardScaler()
        X_train = preprocessing.fit_transform(X_train.toarray())
        X_test = preprocessing.fit_transform(X_test.toarray())
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'norm':
        X_train = normalize(X_train.toarray(), axis=0, norm='l1')
        print 'preprocess %s completed'%(preprocess)

    if clf_name == "xgb":
        cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
        ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic'}
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring='accuracy', cv=5, n_jobs=-1)
        optimized_GBM.fit(X_train, y_train)
        print(optimized_GBM.grid_scores_)
        print('Cross-validation !')

def masked_randomForest(X_train, y_train, preprocess = 'Std'):

    # PREPROCESSING = FEATURES SCALING
    if preprocess == 'MaxMin':
        preprocessing = MaxAbsScaler()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'Binarization':
        preprocessing = Binarizer()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'Std':
        preprocessing = StandardScaler(with_mean=False)
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'full_std':
        preprocessing = StandardScaler()
        X_train = preprocessing.fit_transform(X_train.toarray())
        print 'preprocess %s completed'%(preprocess)

    if preprocess == 'norm':
        X_train = normalize(X_train.toarray(), axis=0, norm='l1')
        print 'preprocess %s completed'%(preprocess)

    clf = RandomForestClassifier(n_jobs=-1, n_estimators = 50)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_

    inter = [np.percentile(importances, qq) for qq in np.linspace(0, 100, 6)]
    count = 0
    RF_features = np.zeros((1, len(importances)))
    for low, high in zip(inter[:len(inter)], inter[1:]):
        if low == -1:
            RF_features += count * np.logical_and(importances <= high, importances >= low)
        else:
            RF_features += count * np.logical_and(importances <= high, importances > low)
        count += 1
        RF_features = RF_features.astype(int)

    importances.sort()
    fig, ax1 = plt.subplots(1)
    x = np.arange(len(importances))
    ax1.plot(x, importances[::-1], 'b-')
    ax1.set_xlabel('features')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Random Forest importance', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_yscale('log')
    fig.tight_layout()
    plt.title(' RandomForest importance ')
    plt.show()

    return RF_features