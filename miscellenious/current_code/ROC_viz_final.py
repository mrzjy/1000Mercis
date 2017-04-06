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
from fonctions import f_script_final
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
    print 'Generating ROC Curves ...'
    sourcePathModels = '/home/pl_vernhet/cookies_data/v1/final/r_6/'
    sourcePath = '/home/pl_vernhet/cookies_data/v1/'
    destinationPath = '/home/pl_vernhet/git/look-alike-cookies/'
    figname = 'Std_r_6'
    X_test,y_test = static_load_csr(sourcePath, 'test')
    
    #orig_stdout = sys.stdout
    #f = open('out_ROC.txt', 'w')
    #sys.stdout = f

    # the input is in sparse format
    f_script_final.ROC_curvesCV(X_test,y_test, sourcePathModels, figPath = destinationPath, figname = figname)
    
    #sys.stdout = orig_stdout
    #f.close()
    
    
    
