# -*- coding: utf-8 -*-
"""
@author: plkovac
"""
"""
Script_3 = Random Forest and Logistic regression
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
from fonctions import f_script_3
import dateutil
import scipy.sparse as sps
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#==============================================================================
# Data retrieval
#==============================================================================
def load_sparse_csr(path_to):
    loader = np.load(path_to)
    matrix = sps.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    return matrix

def static_load_csr(path_to, pref):
    X = load_sparse_csr(path_to + '_X_' + pref + '.npz')
    loader = np.load(path_to + '_y_' + pref + '.npz')
    y = loader['y']
    return X, y

#==============================================================================
# I - Algorithms on static-like data
#==============================================================================

def SMOTE_based(sourcePath):
    print '************************************************'
    print '            SMOTE ALGOS                         '
    print '************************************************'
    print '\n'

    X_train, y_train = static_load_csr(sourcePath, 'train')
    X_val, y_val = static_load_csr(sourcePath, 'val')
    X_test, y_test = static_load_csr(sourcePath, 'test')

    start = time.time()
    sm = SMOTE(random_state=40, n_jobs = -1)
    X, y_train = sm.fit_sample(X_train.toarray(), y_train)
    X_train = sps.csr_matrix(X)
    stop = time.time()
    print 'SMOTE sampling done in %f'%(stop-start)
    print '\n'

    print '************************************************'
    print '              RAW ALGORITHMS                    '
    print '\n'
    print 'Logistic Regressions starting ....'
    t1 = time.time()
    f_script_3.logreg_ttv(X_train, X_val, X_test, y_train, y_val, y_test)
    t2 = time.time()
    print 'Job finished in %f s'%(t2-t1)
    print 'RandomForest starting ...'
    f_script_3.RandomForest_ttv(X_train, X_val, X_test, y_train, y_val, y_test)
    t3 = time.time()
    print 'Job finished in %f s'%(t3-t2)
    print '\n'

    print '************************************************'
    print '             FEATURES CLUSTERING                '
    print '\n'
    powers = np.arange(2,4)
    K = 10**powers
    
    for num_inters in K:
        print 'Algos for %d clusters'%(num_inters)
        grouping_features = f_script_3.feature_cluster_ttv(X_train.toarray(), y_train, weights = False, num_inters = num_inters)
        X1, X2, X3 = f_script_3.combine_ttv(X_train.toarray(), X_val.toarray(), X_test.toarray(), grouping_features)
        X_train_c = sps.csr_matrix(X1)
        X_val_c   = sps.csr_matrix(X2)
        X_test_c  = sps.csr_matrix(X3)
        print '\n'

        print '************************************************'
        print '                  ALGORITHMS                    '
        print '\n'
        print 'Logistic Regressions on clustered data starting ....'
        t4 = time.time()
        f_script_3.logreg_ttv(X_train_c, X_val_c, X_test_c, y_train, y_val, y_test)
        t5 = time.time()
        print 'Job finished in %f s'%(t5-t4)
        print 'RandomForest starting ...'
        f_script_3.RandomForest_ttv(X_train_c, X_val_c, X_test_c, y_train, y_val, y_test)
        t6 = time.time()
        print 'Job finished in %f s'%(t6-t5)
        print 'balanced RandomForest starting ...'
        f_script_3.RandomForest_ttv(X_train_c, X_val_c, X_test_c, y_train, y_val, y_test, class_weight = "balanced")
        t7 = time.time()
        print 'Job finished in %f s'%(t7-t6)
        print 'RandomForest min_samples_leaf = 50 starting ...'
        f_script_3.RandomForest_ttv(X_train_c, X_val_c, X_test_c, y_train, y_val, y_test, min_samples_leaf = 50)
        t8 = time.time()
        print 'Job finished in %f s'%(t8-t7)
        print 'balanced RandomForest min_samples_leaf = 50 starting ...'
        f_script_3.RandomForest_ttv(X_train_c, X_val_c, X_test_c, y_train, y_val, y_test, class_weight = "balanced" , min_samples_leaf = 50)
        t9 = time.time()
        print 'Job finished in %f s'%(t9-t8)

    return 0

def Weighted_based(sourcePath):
    print '************************************************'
    print '         WEIGHTED FEATURE CLUSTERING            '
    print '************************************************'
    print '\n'

    X_train, y_train = static_load_csr(sourcePath, 'train')
    X_val, y_val = static_load_csr(sourcePath, 'val')
    X_test, y_test = static_load_csr(sourcePath, 'test')


    print '************************************************'
    print '             FEATURES CLUSTERING                '
    print '\n'
    powers = np.arange(2,4)
    K = 10**powers
    
    for num_inters in K:
        print 'Algos for %d clusters'%(num_inters)
        grouping_features = f_script_3.feature_cluster_ttv(X_train.toarray(), y_train, weights = True, num_inters = num_inters)
        X1, X2, X3 = f_script_3.combine_ttv(X_train.toarray(), X_val.toarray(), X_test.toarray(), grouping_features)
        X_train_c = sps.csr_matrix(X1)
        X_val_c   = sps.csr_matrix(X2)
        X_test_c  = sps.csr_matrix(X3)
        print '\n'

        print '************************************************'
        print '                  ALGORITHMS                    '
        print '\n'
        print 'Logistic Regressions on clustered data starting ....'
        t4 = time.time()
        f_script_3.logreg_ttv(X_train_c, X_val_c, X_test_c, y_train, y_val, y_test)
        t5 = time.time()
        print 'Job finished in %f s'%(t5-t4)
        print 'RandomForest starting ...'
        f_script_3.RandomForest_ttv(X_train_c, X_val_c, X_test_c, y_train, y_val, y_test)
        t6 = time.time()
        print 'Job finished in %f s'%(t6-t5)
        print 'balanced RandomForest min_samples_leaf = 50 starting ...'
        f_script_3.RandomForest_ttv(X_train_c, X_val_c, X_test_c, y_train, y_val, y_test, class_weight = "balanced", min_samples_leaf = 50)
        t7 = time.time()
        print 'Job finished in %f s'%(t7-t6)

    print '************************************************'
    print '   BASIC ALGORITHMS FOR COMPARISONS             '
    print '\n'
    print 'Logistic Regressions starting ....'
    t1 = time.time()
    f_script_3.logreg_ttv(X_train, X_val, X_test, y_train, y_val, y_test)
    t2 = time.time()
    print 'Job finished in %f s'%(t2-t1)
    print 'RandomForest starting ...'
    f_script_3.RandomForest_ttv(X_train, X_val, X_test, y_train, y_val, y_test)
    t3 = time.time()
    print 'Job finished in %f s'%(t3-t2)
    print 'balanced RandomForest starting ...'
    f_script_3.RandomForest_ttv(X_train, X_val, X_test, y_train, y_val, y_test, class_weight = "balanced")
    t4 = time.time()
    print 'Job finished in %f s'%(t4-t3)
    print 'RandomForest min_samples_leaf = 50 starting ...'
    f_script_3.RandomForest_ttv(X_train, X_val, X_test, y_train, y_val, y_test, min_samples_leaf = 50)
    t5 = time.time()
    print 'Job finished in %f s'%(t5-t4)
    print 'balanced RandomForest min_samples_leaf = 50 starting ...'
    f_script_3.RandomForest_ttv(X_train, X_val, X_test, y_train, y_val, y_test, class_weight = "balanced" , min_samples_leaf = 50)
    t6 = time.time()
    print 'Job finished in %f s'%(t6-t5)
    print 'SVM test starting ...'
    f_script_3.SVM_ttv(X_train, X_val, X_test, y_train, y_val, y_test)
    t7 = time.time()
    print 'Job finished in %f s'%(t7-t6)

    print '\n'
    

    return 0
#==============================================================================
# Script launch
#==============================================================================

if __name__ == '__main__':
    sourcePath = '/home/pl_vernhet/cookies_data/v1/ttv'
    outfile = 'outPaul.txt'

    orig_stdout = sys.stdout
    f = open(outfile, 'w')
    sys.stdout = f

    SMOTE_based(sourcePath)
    Weighted_based(sourcePath)

    sys.stdout = orig_stdout
    f.close()
