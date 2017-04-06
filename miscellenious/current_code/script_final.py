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
    X = load_sparse_csr(path_to + 'final_X_' + kind + '.npz')
    loader = np.load(path_to + 'final_y_' + kind + '.npz')
    y = loader['y']
    return X,y

#==============================================================================
# Script launch
#==============================================================================
 
if __name__ == '__main__':

    sourcePath = '/home/pl_vernhet/cookies_data/v1/'
    savePath = '/home/pl_vernhet/cookies_data/v1/'
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

    #orig_stdout = sys.stdout
    #f = open('out.txt', 'w')
    #sys.stdout = f

    # the input is in sparse format
    #preprocess = ['MaxMin','Binarization','Std']
    #scoring = ['accuracy','f1_micro','recall_micro','roc_auc']
    #clf = ['randomForest','logistic']
    #print 'SMOTE SAMPLING ADDED'
    #for clfs in clf:
    #    f_script_final.hyperParamSearch( X_trainVal, y_trainVal,X_test, y_test, clf = clfs, scoring = scoring, preprocess = preprocess[2])
    #print '\n'
    #print 'Features selection with Chi2'
    #methods = ['feat_select','reduce_dim']
    #methods = ['feat_select']
    #for m in methods:
    #    f_script_final.hyperParamSearch_v2( X_trainVal, y_trainVal,X_test, y_test, clf = 'logistic', scoring = scoring, preprocess = preprocess[0], method = m )
    #    f_script_final.hyperParamSearch_v2( X_trainVal, y_trainVal,X_test, y_test, clf = 'randomForest', scoring = scoring, preprocess = preprocess[0], method = m )

    #sys.stdout = orig_stdout
    #f.close()
    
    
    
