# -*- coding: utf-8 -*-
"""
@author: plkovac
"""

"""
Script_1 = Classification with Logistic regression on clustered features
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
from fonctions import f_script_2
import dateutil
import scipy.sparse as sps
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix
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

def static_load_csr(path_to):
    X = load_sparse_csr(path_to + '_X.npz')
    loader = np.load(path_to + '_y.npz')
    y = loader['y']
    return X,y

def SMOTE_load(path_to):
    loader = np.load(path_to + 'SMOTE_X.npz')
    X = loader['X']
    loader = np.load(path_to + 'SMOTE_y.npz')
    y = loader['y']
    return X,y

def save_full_SMOTE(path, filename, matrix):
    np.savez(os.path.join(path, filename), X=matrix)

def save_labels(path, filename, vector):
    np.savez(os.path.join(path, filename), y=vector)

#==============================================================================
# I - Algorithms on static-like data
#==============================================================================

def SMOTE_save(X,y, path):
    print '************************************************'
    print '           SMOTE OVERSAMPLING                   '
    print '************************************************'
    print '\n'
    t1 = time.time()
    X = X.todense()
    sm = SMOTE(random_state=42)
    X0, y0 = sm.fit_sample(X, y)
    print 'New samples generated = %d '%(len(y))
    save_full_SMOTE(path, 'SMOTE_X', X1)
    save_labels(path,'SMOTE_y', y0)
    t2 = time.time()
    print "Elasped time for SMOTE oversampling : %f"%(t2-t1)
    return X0, y0

def Feature_selections_SMOTE(X, y, train_size = 0.5, percentiles = True):
    print '************************************************'
    print '           FEATURE CLUSTERING                   '
    print '************************************************'
    print '\n'

    #==============================================================================
        # SMOTE version
    #==============================================================================
    #==============================================================================
    t2 = time.time()
    train_size = train_size
    split = int(ceil( train_size*X.shape[0] ))
    i_cluster = split
    i_test = X.shape[0] - split
    print 'X1 = X[:i_cluster,:]'
    X1 = X[:i_cluster,:]
    #==============================================================================
    #==============================================================================
    n = X.shape[1]
    print 'mask_cspp = y[:i_cluster].astype(bool)'
    mask_cspp = y[:i_cluster].astype(bool) # csp+ mask
    print 'n_cspp = X1[mask_cspp,:].sum(axis = 0)'
    n_cspp = X1[mask_cspp,:].sum(axis = 0) # number of csp+ visits
    n_cspm = X1[~mask_cspp,:].sum(axis = 0) # number of csp- visits
    site_contrast = [ (n_cspp[i] - n_cspm[i]) * 1.0 / (n_cspp[i] + n_cspm[i]) if n_cspp[i] + n_cspm[i] != 0 else -1/float(3) for i in np.arange(n) ]
    print 'site_contrast calculated'

    num_inters = 25 # number of intervals - 1 in histogram
    percentiles = False # checking unbalanced effects
    if percentiles :
        inter = np.percentile(site_contrast, np.linspace(0,100,num_inters+1) )
        print inter
    else :
        inter = np.linspace(-1,1,num_inters+1)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.hist(site_contrast, bins = num_inters)
    #plt.title('Histogram for domaines-score repartition')
    #plt.xlabel('Domains scores by percentiles')
    #plt.ylabel('Domains frequency')
    #plt.yscale('log', nonposy='clip')
    #name = 'SMOT_scores_1'
    #plt.savefig('../img/'+name)

    t3 = time.time()
    print "Elasped time for score computation: %f"%(t3-t2)
    #==============================================================================
    #classification of each site into each interval
    count = 0
    site_clf = np.zeros((1,X1.shape[1]))
    for low,high in zip(inter[:num_inters],inter[1:]):
        if low==-1 :
            site_clf +=  count * np.logical_and(site_contrast <= high, site_contrast >= low)
        else:
            site_clf +=  count * np.logical_and(site_contrast <= high, site_contrast > low)
        count += 1
        site_clf = site_clf.astype(int)
    t4 = time.time()
    print "Elasped time for features clustering: %f"%(t4-t3)
    return site_clf, num_inters, i_test

def Completion(X ,y , site_clf, num_inters, i_test):
    t1 = time.time()
    X0 = np.zeros((i_test,num_inters))
    for feat in np.arange(num_inters):
        ind = (site_clf == feat)
        X0[:,feat] = X[:-i_test,ind[0,:]].sum(axis = 1).ravel()

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.hist(np.arange(num_inters),X0.sum(axis = 0))
    #plt.title('Histogram of visits according to sites clusters')
    #plt.xlabel('Domaines clusters (score-oriented)')
    #plt.ylabel('number of visits')
    #plt.yscale('log', nonposy='clip')
    #name = 'SMOT_scores_2'
    #plt.savefig('../img/'+name)

    y0 = y[:-i_test]
    t2 = time.time()
    print "Elasped time for matrix reshaping: %f"%(t2-t1)
    return X0, y0

#==============================================================================
# Script launch
#==============================================================================

if __name__ == '__main__':
    path_to = '/home/pl_vernhet/cookies_data/v1/'
    print 'BEGINNING OF SMOTE PHASE'

    orig_stdout = sys.stdout
    f = open('out_SMOTE.txt', 'w')
    sys.stdout = f

    first_gen = True
    if first_gen :
        X , y = static_load_csr('/home/pl_vernhet/cookies_data/v1/sparse')
        X0 , y0 = SMOTE_save(X,y, path_to)
    else:
        X0 , y0 = SMOTE_load(path_to)
        print 'Data loaded : %d cookies and %d features'%(X0.shape[0],X0.shape[1])
    site_clf, num_inters, i_test = Feature_selections_SMOTE(X0,y0)
    X0, y0 = Completion(X0, y0, site_clf, num_inters, i_test)
    f_script_2.logreg_crossvalid(X0, y0, repeat_time = 5, penalty = 'l2')

    sys.stdout = orig_stdout
    f.close()
