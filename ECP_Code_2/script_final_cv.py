# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:21:52 2017

@author: Kevin & Paul
"""
#==============================================================================
# Libraries
#==============================================================================
import numpy as np

from fonctions import f_script_final_cv as CV
#from fonctions import f_script_final as CV_sklearn
import scipy.sparse as sps
from sklearn.metrics import roc_curve, auc
import colorsys
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

def static_load_csr(path_to, prefix, n_sample, kind):
#    save_name = prefix + str(n_sample) + '_sample_y_test'
    X = load_sparse_csr(path_to + prefix + str(n_sample) + '_sampled_X_' + kind +  '.npz')
    loader = np.load(path_to + prefix + str(n_sample) + '_sampled_y_' + kind +  '.npz')
    y = loader['y']
    return X,y

#==============================================================================
# Script launch
#==============================================================================
if __name__ == '__main__':
    localenv = 0

    if localenv == 1:    
        source_path = 'D:/Python anaconda/Worksapce/google_python/current_code/'
        save_path = 'D:/Python anaconda/Worksapce/google_python/current_code/'
    else:
        source_path = '/home/Kevin/'
        save_path = '/home/jiayi_zhang9312/look-alike-cookies/current_code2/'
        fig_path = '/home/jiayi_zhang9312/look-alike-cookies/img/'
    
    prefix = 'filtre_equi_'    
    n_samples = [500, 1000, 2000, 3500, 5000, 8000, 10000, 18000, 27178]
    save_test_size = 2000
    X_TrainVal, y_TrainVal, X_Test, y_Test = list(), list(), list(), list()
    for n_sample in n_samples:
        X_trainVal, y_trainVal = static_load_csr(save_path, prefix, n_sample,'trainVal')        
        
        X_TrainVal.append(X_trainVal)
        y_TrainVal.append(y_trainVal)
    
        print 'Data loaded - check balance'
        print str(n_sample)
        print y_trainVal.sum()/float(len(y_trainVal))
        
    X_test, y_test = static_load_csr(save_path, prefix, save_test_size, 'test')

    # the input is in sparse format
    preprocess = ['MaxMin','Std']
    # samplings = ['SMOTE','ENN','SMOTEENN','None']
    clfs = ['logistic','randomForest','KNN','Voting','MLP']
    featurings = ['agglo_custom','reduce_dim','feat_select','None']
    preprocess = preprocess[0]
    featuring = featurings[0]
    clf = clfs[2]

    fprs = list()    
    tprs = list()  
    names = list() 
    N = len(n_samples)
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    lineWidth = 2
        
    epoch = 0
    for n_sample in n_samples:   
        print '          n_sample : %s'%(n_sample)
        name_best, preprocessing, clf_best, featureMethod = CV.crossValid(X_TrainVal[epoch], y_TrainVal[epoch],
                                            preprocess = preprocess,
                                            clf_name = clf,
                                            n_sample = n_sample,
                                            fig_path = fig_path,
                                            featuring = featuring,
                                            cv = 3, random_state = 50)

        X_test_preprocess = preprocessing.transform(X_test)
        if featureMethod != 'None':
            X_test_preprocess = featureMethod.transform(X_test_preprocess)
#        y_pred = clf_best.predict(X_test_preprocess)
#        print '*****CV manual****'
#        confus = confusion_matrix(y_test, y_pred)
#        print confus        
        
        proba = clf_best.predict_proba(X_test_preprocess)
        ROC_scores = proba[:, 1]  # thresholds for ROC
        fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
        fprs.append(fpr)
        tprs.append(tpr)
        names.append(name_best)
#        CV_sklearn.hyperParamSearch(X_TrainVal[epoch], y_TrainVal[epoch],
#                                   X_test, y_test, clf = clf,
#                                   scoring = ['roc_auc'],
#                                   preprocess = preprocess)
        epoch += 1
    plt.figure()
    epoch = 0
    for n_sample in n_samples:  
        
        plt.plot(fprs[epoch], tprs[epoch], color= RGB_tuples[epoch],
                 lw = lineWidth, 
                 label = 'n_sample = %s '%(n_sample) + names[epoch] + ' (area = %0.2f)' % auc(fprs[epoch], tprs[epoch]))
        epoch += 1
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for %s best roc_auc'%(clf))
    plt.legend(loc="lower right", prop={'size': 6})
    plt.savefig(fig_path + 'Test ROC(n_samples)_%s_%s_%s' % (clf, preprocess, featuring))
    
    
