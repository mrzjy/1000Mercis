# -*- coding: utf-8 -*-
"""
@author: Kevin & Paul
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
import time
import scipy.sparse as sps

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Binarizer
#==============================================================================
# Data Generation and saving
#==============================================================================
def save_sparse_csr(path, filename,array):
    np.savez(os.path.join(path, filename),data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    
def save_labels(path, filename,vector):
    np.savez(os.path.join(path, filename), y=vector)
    
def load_sparse_csr(path_to):
    loader = np.load(path_to)
    matrix = sps.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    return matrix

def static_load_csr(path_to):
    X = load_sparse_csr(path_to + 'X.npz')
    loader = np.load(path_to + 'y.npz')
    y = loader['y']
    return X,y
    
def load_raw(path):
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

#==============================================================================
# Data filtering
# =============================================================================
def static_filtre_generation_csr(sourcePath, savePath, prefix, shuffle=True):
    print ('************************************************')
    print ('            filtre_generation                   ')
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
    aggregation = {'label': lambda x: 1 if x.all() else 0, 'visits' :'sum'}
    data = data.groupby(['tuid','domaine']).agg(aggregation)
    #.reset_index().set_index(['tuid','domaine'])
    X = sps.coo_matrix((data.visits, (data.index.labels[0], data.index.labels[1])))
    X = X.tocsr()
    y = data['label'].as_matrix()
    if shuffle:
        np.random.seed(50)
        index = np.arange(np.shape(X)[0])
        np.random.shuffle(index)
        X = X[index, :]
        y = y[index]

    save_sparse_csr(savePath, prefix +'X', X)
    save_labels(savePath, prefix +'y', y)
    
    print('Job finished in {} sec'.format(time.time() - start))
    return 0

#==============================================================================
# Data spliting
# =============================================================================
def cv_generation_equilibre_csr(loadPath, savePath, prefix, n_samples, test_size=0.15, save_test_size = 2000, stratify=None, random_state = 50):
    print ('************************************************')
    print ('          TRAIN TEST VALIDATION                 ')
    print ('************************************************')
    start = time.time()
#    sparse_path = '/home/pl_vernhet/cookies_data/v1/sparse'
    X, y = static_load_csr(loadPath)
    X_pos, y_pos = X[y == 1,:], y[y == 1]
    X_neg, y_neg = X[y == 0,:], y[y == 0]
     
    X_pos = X_pos.todense() 
    X_neg = X_neg.todense() 
    print ('************************************************')
    print 'y_pos: '+str(y_pos.shape)
    print 'X_pos: '+str(X_pos.shape)
    print 'y_neg: '+str(y_neg.shape)
    print 'X_neg: '+str(X_neg.shape)
    print ('************************************************')    
    X_pos_trainVal, X_pos_test, y_pos_trainVal, y_pos_test = train_test_split(X_pos, y_pos, 
                                                                              test_size=test_size, 
                                                                              stratify=stratify,
                                                                              random_state=random_state)
    numTest = len(y_pos_test)
    numTrainVal = len(y_pos_trainVal)
    '''*********************
    test-set equilibre
    *********************'''    
    np.random.seed(random_state)
    index = np.arange(np.shape(X_neg)[0])
    np.random.shuffle(index)
    X_neg_test = X_neg[index[:numTest], :]
    y_neg_test = y_neg[index[:numTest]]
    
    X_test = np.vstack((X_pos_test, X_neg_test))
    y_test = np.concatenate((y_pos_test, y_neg_test))
    print X_test.shape
    print y_test.shape
    '''*********************
    trainVal-set equilibre
    *********************'''   
    
    X_neg_allRest = X_neg[index[numTest:],:]
    y_neg_allRest = y_neg[index[numTest:]]

    X_neg_trainVal = X_neg_allRest[:numTrainVal,:]
    y_neg_trainVal = y_neg_allRest[:numTrainVal]
    
    X_trainVal = np.vstack((X_pos_trainVal, X_neg_trainVal))
    y_trainVal = np.concatenate((y_pos_trainVal, y_neg_trainVal))
    print X_trainVal.shape
    print y_trainVal.shape
    '''*********************
    save sets
    *********************'''      
    X_trainVal = sps.csr_matrix(X_trainVal)
    X_test = sps.csr_matrix(X_test)
    for n_sample in n_samples:
        np.random.seed(random_state)
        index = np.arange(X_trainVal.shape[0])
        np.random.shuffle(index)
        save_sparse_csr(savePath, prefix + str(n_sample) + '_sampled_X_trainVal', X_trainVal[index[:n_sample],:])
        save_labels(savePath, prefix + str(n_sample) + '_sampled_y_trainVal', y_trainVal[index[:n_sample]])
    
    np.random.seed(random_state)
    index = np.arange(X_test.shape[0])
    np.random.shuffle(index)
    save_sparse_csr(savePath, prefix + str(save_test_size) + '_sampled_X_test', X_test[index[:save_test_size],:])
    save_labels(savePath, prefix + str(save_test_size) + '_sampled_y_test', y_test[index[:save_test_size]])  
    
    print('Job finished in {} sec'.format(time.time()-start))
    return 0
    
#==============================================================================
# sparsity Measurement
#==============================================================================
def sparsityMeasure(loadPath, prefix):
    X, y = static_load_csr(loadPath)
    X_pos = X[y == 1,:]
    X_neg = X[y == 0,:]
    
    mean_traffic_pos = np.sum( np.sum(X_pos, axis = 1) ) *1.0/ X_pos.shape[0]
    mean_traffic_neg = np.sum( np.sum(X_neg, axis = 1) ) *1.0/ X_neg.shape[0]
    
    binarizer = Binarizer()
    X_pos = binarizer.fit_transform(X_pos)
    X_neg = binarizer.fit_transform(X_neg)
    
    mean_domains_pos = np.sum( np.sum(X_pos, axis = 1) ) *1.0/ X_pos.shape[0]
    mean_domains_neg = np.sum( np.sum(X_neg, axis = 1) ) *1.0/ X_neg.shape[0]
    
    print 'mean_traffic_pos : ' + str(mean_traffic_pos)
    print 'mean_traffic_neg : ' + str(mean_traffic_neg)
    print 'mean_domains_pos : ' + str(mean_domains_pos)
    print 'mean_domains_neg : ' + str(mean_domains_neg)
    
    overall_traffic = (mean_traffic_pos * X_pos.shape[0] + mean_traffic_neg * X_neg.shape[0])*1.0/ X.shape[0]
    overall_domains = (mean_domains_pos * X_pos.shape[0] + mean_domains_neg * X_neg.shape[0])*1.0/ X.shape[0]
    print ' overall_traffic : ' + str( overall_traffic)
    print 'overall_domains : ' + str(overall_domains)
#==============================================================================
# Script launch
#==============================================================================
 
if __name__ == '__main__':
    localenv = 0

    if localenv == 1:    
        sourcePath1 = savePath1 = savePath2 = 'D:/Python anaconda/Worksapce/google_python/current_code/'
        sourcePath2 = 'D:/Python anaconda/Worksapce/google_python/current_code/'
    else:
        sourcePath =  '/home/Kevin/'
        savePath = savePath1 = savePath2 = '/home/jiayi_zhang9312/look-alike-cookies/current_code2/'
    
    filename = 'local'
    n_samples = [500, 1000, 2000, 3500, 5000, 8000, 10000, 18000, 27178]
    random_state = 35
    prefix = 'filtre_equi_'
    loadPath = savePath + prefix
#    Generate filtered data
    static_filtre_generation_csr(sourcePath, savePath, prefix)
#    Split the generated data
    cv_generation_equilibre_csr(loadPath, savePath, prefix, n_samples)
#    (with save_name = prefix + str(n_sample) + '_sampled_y_test')
    sparsityMeasure(loadPath, prefix)