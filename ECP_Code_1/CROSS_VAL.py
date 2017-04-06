# -*- coding: utf-8 -*-
"""
@author: Paul Vernhet & Zhang Jiayi
Cross-validation on dataset using func functions
November 2016
"""

import numpy as np
from scipy.sparse import csr_matrix
from functions import generate_set as f
import matplotlib.pyplot as plt
from functions import func

#==============================================================================
# Data retrieval
#==============================================================================

if __name__ == '__main__':
    wdir='path_to_data'
    first = True # True = data generated for the first time / False = data already existing on local path
    if first:
        X, y, l = f.generate_set()
        # Working with python sparse type
        X_csr = csr_matrix(X)
        func.save_sparse_csr('Cookies', X_csr )
        np.savez('labels', y=y)
    else :
        X_csr = func.load_sparse_csr('Cookies.npz')
        X = X_csr.toarray() # pour retrouver la matrice dense
        dataL = np.load('labels.npz')
        y = dataL['y']

    #--------------------------------------------
    # Excerpt of cross-valdiation on first datasets
    # NB: Better cross-validation code was used afterward on Google Cloud Engines
    #--------------------------------------------
    repeat_time = 6
    C = 0.01
    score_XGB = func.XGboost_crossvalid(X_new, y, repeat_time = repeat_time)
    score_LR = func.logreg_crossvalid(X_new, y, repeat_time = repeat_time)
    score_LSVM = func.LinSVM_crossvalid(X_new, y, C_best = C, repeat_time = repeat_time)
    score_NBM = func.naiveBayesM_crossvalid(X_new, y, repeat_time = repeat_time)
    score_RF = func.RandomForest_crossvalid(X_new, y, repeat_time = repeat_time)
