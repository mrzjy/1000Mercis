# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 20:23:17 2017

@author: Kevin & Paul
"""
#==============================================================================
# Libraries
#==============================================================================
import sys
import os
import time
import numpy as np
import pandas as pd
from math import ceil
from fonctions import f_script_final as fs
import scipy.sparse as sps
import matplotlib
matplotlib.use('Agg')
#==============================================================================
# Data retrieval
#==============================================================================
def load_sparse_csr(path_to):
    loader = np.load(path_to)
    matrix = sps.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    return matrix

def static_load_csr(path_to, kind):
    X = load_sparse_csr(path_to + '_X_' + kind + '.npz')
    loader = np.load(path_to + '_y_' + kind + '.npz')
    y = loader['y']
    return X,y

#==============================================================================
# Script launch
#==============================================================================
 
if __name__ == '__main__':

    sourcePath = '/home/pl_vernhet/cookies_data/v1/final_FILTERS'
    destinationPath = '/home/pl_vernhet/git/look-alike-cookies/'

    X_train,y_train = static_load_csr(sourcePath, 'trainVal')
    X_test,y_test = static_load_csr(sourcePath, 'test')
    print 'Data loaded'

    fs.naive_hyperParamSearch(X_train, y_train, X_test, y_test,figPath = destinationPath, clf_name="logistic", preprocess=None)
    fs.naive_hyperParamSearch(X_train, y_train, X_test, y_test,figPath = destinationPath, clf_name="logistic", preprocess='MaxMin')
    fs.naive_hyperParamSearch(X_train, y_train, X_test, y_test,figPath = destinationPath, clf_name="logistic", preprocess='Std')
    fs.naive_hyperParamSearch(X_train, y_train, X_test, y_test,figPath = destinationPath, clf_name="logistic", preprocess='Binarization')

    fs.naive_hyperParamSearch(X_train, y_train, X_test, y_test,figPath = destinationPath, clf_name="randomForest", preprocess=None)
    fs.naive_hyperParamSearch(X_train, y_train, X_test, y_test,figPath = destinationPath, clf_name="randomForest", preprocess='MaxMin')
    fs.naive_hyperParamSearch(X_train, y_train, X_test, y_test,figPath = destinationPath, clf_name="randomForest", preprocess='Std')
    fs.naive_hyperParamSearch(X_train, y_train, X_test, y_test,figPath = destinationPath, clf_name="randomForest", preprocess='Binarization')
    
    
    
