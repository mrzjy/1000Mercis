# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 20:23:17 2017

@author: Kevin & Paul
"""
#==============================================================================
# Libraries
#==============================================================================
import sys
import numpy as np
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
    #  path of data
    localenv = 0
    if localenv == 1:    
        source_path = 'D:/Python anaconda/Worksapce/google_python/current_code/'
        sourceModelPath = 'D:/Python anaconda/Worksapce/google_python/current_code/models/'
        sourceDataPath = 'D:/Python anaconda/Worksapce/google_python/current_code/'
    else:
        destinationPath = '/home/jiayi_zhang9312/look-alike-cookies/img/'
        sourceModelPath_RFsampling = '/home/jiayi_zhang9312/look-alike-cookies/current_code2/models/sampling/randomForest/'
        sourceModelPath_LRsampling = '/home/jiayi_zhang9312/look-alike-cookies/current_code2/models/sampling/logistic/'
        sourceModelPath_rawNone = '/home/jiayi_zhang9312/look-alike-cookies/current_code2/'
        sourceDataPath = '/home/jiayi_zhang9312/look-alike-cookies/current_code2/models/'
    
    X_test,y_test = static_load_csr(sourceDataPath, 'test')
    
    orig_stdout = sys.stdout
    f = open('csv_test.txt', 'a')
    sys.stdout = f

    # the input is in sparse format
    f_script_final.ROC_curvesCV(X_test,y_test, sourceModelPath_rawNone, figPath = destinationPath, figname = 'MaxMin_ROC_LRagglo')
    # show test result
    f_script_final.performTest(X_test,y_test, sourceModelPath_rawNone)
    
    sys.stdout = orig_stdout
    f.close()
    
    
    
