# -*- coding: utf-8 -*-
"""
@author: plkovac & Jiayi
February 2017
Quick statistics on datasets
"""

import numpy as np
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt
from functions import func
from fonctions import f_global2 as fg
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, normalize
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
import colorsys

#==============================================================================
# Functions definition
#==============================================================================
# ========
# cosine PCA 2D plot
# ========

def getviz_cosinus(X_train, y_train):
    preprocessing = MaxAbsScaler()
    X_train = preprocessing.fit_transform(X_train)
    reds = y_train == 0
    blues = y_train == 1
    plt.figure()
    kpca = KernelPCA(kernel='cosine', n_components=2, n_jobs=-1)
    X_kpca = kpca.fit_transform(X_train)
    plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro", label = 'csp-')
    plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo", label = 'csp+')
    plt.title("Projection by cosine PCA")
    plt.xlabel("1st principal component")
    plt.ylabel("2nd component")
    plt.legend(loc="lower right", prop={'size': 6})
    plt.show()

# ========
# train-test split
# ========
def trainVal(X, y, savePath, test_size=0.15, stratify=None, random_state=50):
    print ('************************************************')
    print ('          TRAIN TEST                            ')
    print ('************************************************')
    start = time.time()
    X_trainVal, X_test, y_trainVal, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify,
                                                              random_state=random_state)
    print 'train contains %d elements, with ratio %f of class 1' % (len(y_trainVal), y_trainVal.sum() / float(len(y_trainVal)))
    print 'test contains %d elements, with ratio %f of class 1' % (len(y_test), y_test.sum() / float(len(y_test)))

    X_trainVal = csr_matrix(X_trainVal)
    X_test = csr_matrix(X_test)

    fg.save_sparse_csr(savePath, 'trainVal_X', X_trainVal)
    fg.save_full(savePath, 'trainVal_y', y_trainVal)
    fg.save_sparse_csr(savePath, 'test_X', X_test)
    fg.save_full(savePath, 'test_y', y_test)

    print('Job finished in {} sec'.format(time.time() - start))
    return 0

if __name__ == '__main__':
    wdir = '/Users/paulvernhet/Desktop/3A/SEMINAIRE 1000mercis/week_4/POSTERIOR'

    ###########################################################################
    # TRAIN - TEST SPLITS
    ###########################################################################

    first = True
    if first:
        X_csr = func.load_sparse_csr('Cookies.npz')
        dataL = np.load('labels.npz')
        y = dataL['y']
        savePath = 'data/'
        trainVal(X_csr, y, savePath, test_size=0.15)

    ###########################################################################
    # Counting basic statistics
    ###########################################################################
    # mean stats for visits
    sec = False
    if sec:
        m = y.astype(bool)  # selection des csp+
        n = len(y)
        nf = X_csr.shape[1]
        traffic_P = np.mean(X_csr[m, :].sum(axis=1))
        traffic_N = np.mean(X_csr[~m, :].sum(axis=1))
        traffic_T = np.mean(X_csr.sum(axis=1))
        visits_P = np.mean(X_csr[m, :].astype(bool).sum(axis=1))
        visits_N = np.mean(X_csr[~m, :].astype(bool).sum(axis=1))
        visits_T = np.mean(X_csr.astype(bool).sum(axis=1))
        print 'csp+ traffic :  %f and visits :  %f ==> sparsity = %f'%(traffic_P,visits_P, visits_P/float(nf))
        print 'csp- traffic :  %f and visits :  %f ==> sparsity = %f' % (traffic_N, visits_N, visits_N / float(nf))
        print 'all traffic :  %f and visits :  %f ==> sparsity = %f' % (traffic_T, visits_T, visits_T / float(nf))


    ###########################################################################
    # Grid-Search and ROC curves
    ###########################################################################
    # 0/ Loading data
    sourcePath = 'data/'
    X_train, y_train = fg.static_load_csr(sourcePath + 'trainVal')
    X_test, y_test = fg.static_load_csr(sourcePath + 'test')
    mask_cspp = y_train.astype(bool)  # selection des csp+

    # 1/ GridSearch without cross-validation + plotting all ROCS
    rocs = True
    #clfs = ['voting','ada', 'logistic','randomforest', 'kNN', 'naiveBayesM']
    clfs = ['xgb']
    metric = 'euclidean'
    preprocess = 'Std'
    if rocs:
        for clf in clfs:
            start = time.time()
            print 'Beginning for %s with %s processing'%(clf, preprocess)
            fg.hyperParamSearch(X_train, y_train, X_test, y_test, clf_name=clf, preprocess = preprocess, metric = metric)
            stop = time.time()
            print 'Models generated in %f s'%(stop-start)
