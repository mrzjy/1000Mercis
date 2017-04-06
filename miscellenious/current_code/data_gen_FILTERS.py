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
from sklearn.cross_validation import train_test_split

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

# ==============================================================================
# ========
def filters_generation(sourcePath, savePath, filename, shuffle=True):
    print ('************************************************')
    print ('            LOCAL TESTS                         ')
    print ('************************************************')
    start = time.time()
    data = load_raw(sourcePath)

    # YEAR-MONTH-DAY
    def process(R):
        limit_time = 2016
        r = R.split()
        r = r[0].split('-')
        if float(r[0]) < limit_time :
            return -1
        else :
            return 1

    # Preprocessing dates by seasons since 2016
    data['date'] = data['date'].apply(lambda x: process(x))
    data.drop(data[data.date == -1].index, inplace=True)
    print 'Number of remaining rows %d' % (data.shape[0])

    data.drop('date', axis=1, inplace=True)

    # Removing uninteresting features ? NOT DONE YET !!

    aggregation = {'label': lambda x: 1 if x.all() else 0, 'visits': 'sum'}
    data = data.groupby(['tuid', 'domaine']).agg(aggregation).reset_index().set_index(['tuid', 'domaine'])

    X = sps.coo_matrix((data.visits, (data.index.labels[0], data.index.labels[1])))
    X = X.tocsr()
    y = data['label'].as_matrix()
    if shuffle:
        np.random.seed(0)
        index = np.arange(np.shape(X)[0])
        np.random.shuffle(index)
        X = X[index, :]
        y = y[index]

    save_sparse_csr(savePath, filename + '_X', X)
    save_labels(savePath, filename + '_y', y)
    print('Job finished in {} sec'.format(time.time() - start))
    return 0

# ========
def filters_trainVal(sourcePath, savePath, filename, test_size=0.15, stratify=None, random_state=50):
    print ('************************************************')
    print ('          TRAIN TEST                            ')
    print ('************************************************')
    start = time.time()
    X, y = static_load_csr(sourcePath)
    X_trainVal, X_test, y_trainVal, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify,
                                                              random_state=random_state)
    print 'train contains %d elements, with ratio %f of class 1' % (len(y_trainVal), y_trainVal.sum() / float(len(y_trainVal)))
    print 'test contains %d elements, with ratio %f of class 1' % (len(y_test), y_test.sum() / float(len(y_test)))

    X_trainVal = sps.csr_matrix(X_trainVal)
    X_test = sps.csr_matrix(X_test)

    save_sparse_csr(savePath, filename + '_X_trainVal', X_trainVal)
    save_labels(savePath, filename + '_y_trainVal', y_trainVal)
    save_sparse_csr(savePath, filename + '_X_test', X_test)
    save_labels(savePath, filename + '_y_test', y_test)

    print('Job finished in {} sec'.format(time.time() - start))
    return 0

#==============================================================================
# Script launch
#==============================================================================
 
if __name__ == '__main__':

    filename = 'final_FILTERS'
    sourcePath1 = savePath1 = savePath2 = '/home/pl_vernhet/cookies_data/v1/'
    sourcePath2 = sourcePath1+filename

    filters_generation(sourcePath1, savePath1, filename, shuffle=True)
    filters_trainVal(sourcePath2, savePath2, filename, test_size=0.15)
