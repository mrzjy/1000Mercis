# -*- coding: utf-8 -*-
"""
@author: plkovac
"""

"""
Script_1 = Random Forest and Logistic regression
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
from fonctions import f_script_1
import dateutil
import scipy.sparse as sps

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

#==============================================================================
# I - Algorithms on static-like data
#==============================================================================

def first_algos(X, y, repeat_time = 6):
    print '************************************************'
    print '            ALGOS RUNNING                       '
    print '************************************************'
    print '\n'
    X = X.todense()
    #score_LR = f_script_1.logreg_crossvalid(X, y, repeat_time = repeat_time)
    score_RF = f_script_1.RandomForest_crossvalid(X, y, repeat_time = repeat_time)
    return 0
    
#==============================================================================
# Script launch
#==============================================================================
 
if __name__ == '__main__':
    path_to = '/home/pl_vernhet/cookies_data/v1/sparse'
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    X,y = static_load_csr(path_to)
    first_algos(X,y)
    sys.stdout = orig_stdout
    f.close()
    
    
    
    
