# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 20:13:07 2016

@author: Kevin & Paul
"""

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

#==============================================================================
# function : logreg_crossvalid
#==============================================================================
def logreg_crossvalid(X, y, repeat_time = 5, penalty = 'l2'):                      
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
    fmeasure = np.zeros((1,repeat_time))
    matrix_confus = list()
    importances = list()
    kf = KFold(n_splits = repeat_time, shuffle=True) # K-fold
    i = 0
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf = linear_model.LogisticRegression()
        clf.fit(X_train,y_train)
        y_test_pred = clf.predict(X_test)
        
        confus = confusion_matrix(y_test, y_test_pred)
        True_neg[0,i] = confus[0,0]
        True_pos[0,i] = confus[1,1]
        # capacite de detecter les csp+
        sensibility[0,i] = True_pos[0,i] * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity[0,i] = True_neg[0,i] * 1.0 / sum(confus[0,::]) 
        accuracy[0,i] = (True_pos[0,i] + True_neg[0,i]) * 1.0 / sum(sum(confus))
        precision = confus[1,1]* 1.0/(confus[1,1]+confus[0,1])
        fmeasure[0,i] = 2*(sensibility[0,i] * precision) * 1.0 / (sensibility[0,i] + precision)
        matrix_confus.append( confus )
        importances.append( clf.coef_ )
        i = i+1
        
    performance_dict = {'True_neg': True_neg, 
                        'True_pos': True_pos, 
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'F1' : fmeasure,
                        'matrix_confus': matrix_confus,
                        'feat_importance': importances};
                        
    print "LogistRegress : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std())
    print "LogistRegress : Mean sensitivity: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std())
    print "LogistRegress : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std())
    print "LogistRegress : F1 measure : %0.3f +/- %0.3f" % (fmeasure.mean(), fmeasure.std())
    return performance_dict
        
#==============================================================================
#  function : RandomForest_crossvalid
#==============================================================================
def RandomForest_crossvalid(X, y, repeat_time = 5, N = 50):                      
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
    fmeasure = np.zeros((1,repeat_time))
    importances = list()
    matrix_confus = list()
    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
    i = 0

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #sample_weight = np.array([3 if i else 1 for i in y_train])
        clf = RandomForestClassifier(n_estimators = N, n_jobs = -1)
        #clf.fit(X_train,y_train, sample_weight)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        
        confus = confusion_matrix(y_test, y_test_pred)
        True_neg[0,i] = confus[0,0]
        True_pos[0,i] = confus[1,1]
        # capacite de detecter les csp+
        sensibility[0,i] = True_pos[0,i] * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity[0,i] = True_neg[0,i] * 1.0 / sum(confus[0,::]) 
        accuracy[0,i] = (True_pos[0,i] + True_neg[0,i]) * 1.0 / sum(sum(confus)) 
        precision = confus[1,1]* 1.0/(confus[1,1]+confus[0,1])
        fmeasure[0,i] = 2*(sensibility[0,i] * precision) * 1.0 / (sensibility[0,i] + precision)
        importances.append( clf.feature_importances_ )
        matrix_confus.append( confus )
        i= i+1
    performance_dict = {'True_neg': True_neg, 
                        'True_pos': True_pos, 
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'F1' : fmeasure,
                        'matrix_confus': matrix_confus,
                        'feat_importance': importances};  
    print "RandomForest : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std())
    print "RandomForest : Mean sensitivity: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std())
    print "RandomForest : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std())
    print "RandomForest : F1 measure : %0.3f +/- %0.3f" % (fmeasure.mean(), fmeasure.std())
    return performance_dict
    
