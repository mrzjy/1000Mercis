# -*- coding: utf-8 -*-
"""
@author: plkovac
"""

"""
data_gen = data generation to pandas / sparse csr formats
"""

#==============================================================================
# Libraries
#==============================================================================

import os
import numpy as np
import pandas as pd
from math import ceil
import dateutil
import time
import scipy.sparse as sps
from sklearn.model_selection import StratifiedKFold

#==============================================================================
# Data Generation and saving
#==============================================================================
def save_sparse_csr(path, filename,array):
    np.savez(os.path.join(path, filename),data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    
def save_labels(path, filename,vector):
    np.savez(os.path.join(path, filename), y=vector)
    
def load_raw(path):
    # wdir=''
    path_neg = path + 'negative_tuid_centrale_finale_with_header.csv'
    path_pos = path + 'positive_tuid_centrale_finale_with_header.csv'

    data_neg = pd.read_csv(path_neg, encoding = 'utf-8', sep = '|')
    data_pos = pd.read_csv(path_pos, encoding = 'utf-8', sep = '|')
    data_neg["label"]=0
    data_pos["label"]=1    
    data = pd.concat( [data_neg, data_pos] )
    data['visits']=1

    Np = data_pos['tuid'].nunique()
    Nm = data_neg['tuid'].nunique()
    W = data['domaine'].nunique()
    
    print('Data imported in pandas \n \n')
    print('data_pos : %d lines' %( data_pos.shape[0] ) )
    print(' cookies csp+ : {} \n'.format(Np))
    print('data_neg : %d lines' %( data_neg.shape[0] )) 
    print(' cookies csp- : {} \n'.format(Nm))
    print(' Total domains visited : {} \n'.format(W))
    return data
    
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

#==============================================================================
    
def static_generation_csr(path, filename, shuffle=True):
    print ('************************************************')
    print ('            STATIC GENERATION                   ')
    print ('************************************************')
    start = time.time()
    data = load_raw(path)
    data.drop('date',axis=1,inplace=True)
    
    aggregation = {'label': lambda x: 1 if x.all() else 0, 'visits' :'sum'}
    data = data.groupby(['tuid','domaine']).agg(aggregation).reset_index().set_index(['tuid','domaine'])

    X = sps.coo_matrix((data.visits, (data.index.labels[0], data.index.labels[1])))
    X = X.tocsr()
    y = data['label'].as_matrix()
    if shuffle:
        np.random.seed(0)
        index = np.arange(np.shape(X)[0])
        np.random.shuffle(index)
        X = X[index, :]
        y = y[index]

    save_sparse_csr(path, filename+'_X', X)
    save_labels(path, filename+'_y', y)
    print('Job finished in {} sec'.format(time.time()-start))
    return 0

def ttv_generation_csr(path, filename, test_size=0.15, train_size=0.7, stratify=True, random_state = 50):
    print ('************************************************')
    print ('          TRAIN TEST VALIDATION                 ')
    print ('************************************************')
    start = time.time()
    sparse_path = '/home/pl_vernhet/cookies_data/v1/sparse'
    X,y = static_load_csr(sparse_path)
    X = X.todense() 
    
    X_wait, X_test, y_wait, y_test = train_test_split( X, y, test_size=test_size, stratify=stratify,random_state=random_state)
    print 'test contains %d elements, with ratio %f of class 1'%(len(y_test), y_test.sum()/float(len(y_test)) )
    
    relative_train_size = train_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split( X_wait, y_wait, train_size=relative_train_size, stratify=stratify,random_state=random_state)
    print 'train contains %d elements, with ratio %f of class 1'%(len(y_train), y_train.sum()/float(len(y_train)) )
    print 'val contains %d elements, with ratio %f of class 1'%(len(y_val), y_val.sum()/float(len(y_val)) )
    
    X_train = sps.csr_matrix(X_train)
    X_val = sps.csr_matrix(X_val)
    X_test = sps.csr_matrix(X_test)
    
    save_sparse_csr(path, filename+'_X_train', X_train)
    save_labels(path, filename+'_y_train', y_train)
    save_sparse_csr(path, filename+'_X_val', X_val)
    save_labels(path, filename+'_y_val', y_val)
    save_sparse_csr(path, filename+'_X_test', X_test)
    save_labels(path, filename+'_y_test', y_test)
    
    print('Job finished in {} sec'.format(time.time()-start))
    return 0

#==============================================================================
# Script launch
#==============================================================================
 
if __name__ == '__main__':
    path = '/home/pl_vernhet/cookies_data/v1/'
    static_generation_csr(path, 'sparse')
    ttv_generation_csr(path, 'ttv')
    
    
