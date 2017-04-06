# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 20:22:06 2017

@author: Kevin
"""
import numpy as np
import time
#from scipy.sparse import csr_matrix

from sklearn.metrics import confusion_matrix
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import csr_matrix
from sklearn.externals import joblib

#==============================================================================
# function : logreg_ttv
#==============================================================================
def logreg_ttv(X_train, X_val, X_test, y_train, y_val, y_test):                      
    param = ["l1","l2"]
    
    print '************************************************'
    print '                  Logregression                 '
    print '************************************************'
    True_neg = np.zeros((1,len(param)))
    sensibility = np.zeros((1,len(param)))
    True_pos = np.zeros((1,len(param)))
    specificity = np.zeros((1,len(param)))
    accuracy = np.zeros((1,len(param)))
    fmeasure = np.zeros((1,len(param)))
    
    clfs = []
    for p in param:
        clfs += [linear_model.LogisticRegression(penalty = p, n_jobs = -1)]    

# train + valid 
    for i in range(0,len(param)):
        clfs[i].fit(X_train,y_train)
        y_val_pred = clfs[i].predict(X_val)
        confus = confusion_matrix(y_val, y_val_pred)
        True_neg = confus[0,0]
        True_pos = confus[1,1]
        # capacite de detecter les csp+
        sensibility = True_pos * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity = True_neg * 1.0 / sum(confus[0,::]) 
        accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
        precision = confus[1,1]*1.0/(confus[1,1]+confus[0,1])
        fmeasure = 2*(sensibility * precision) * 1.0 / (sensibility + precision)
        #clf_name = "model_logreg_%s.pkl" % (param[i])        
        #joblib.dump(clfs[i], clf_name, compress=True) 
        print "LogistRegress (penalty = %s): Validation Accuracy: %0.3f" % (param[i],accuracy)
        print "LogistRegress (penalty = %s): Validation sensitivity: %0.3f" % (param[i],sensibility)
        print "LogistRegress (penalty = %s): Validation specificity: %0.3f" % (param[i],specificity)
        print "LogistRegress(penalty = %s): Validation F1 measure : %0.3f" % (param[i],fmeasure)
# test    
    for i in range(0,len(param)):
        y_test_pred = clfs[i].predict(X_test)
        confus = confusion_matrix(y_test, y_test_pred)
        True_neg = confus[0,0]
        True_pos = confus[1,1]
        # capacite de detecter les csp+
        sensibility = True_pos * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity = True_neg * 1.0 / sum(confus[0,::]) 
        accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
        precision = confus[1,1]*1.0/(confus[1,1]+confus[0,1])
        fmeasure = 2*(sensibility * precision) * 1.0 / (sensibility + precision)
        print "LogistRegress (penalty = %s): Test Accuracy: %0.3f" % (param[i],accuracy)
        print "LogistRegress (penalty = %s): Test sensitivity: %0.3f" % (param[i],sensibility)
        print "LogistRegress(penalty = %s): Test specificity: %0.3f" % (param[i],specificity)
        print "LogistRegress (penalty = %s): Test F1 measure : %0.3f" % (param[i],fmeasure)
        
#==============================================================================
#  function : RandomForest_ttv
#==============================================================================
def RandomForest_ttv(X_train, X_val, X_test, y_train, y_val, y_test, class_weight = None, min_samples_leaf = 1):                      
    param = [50, 100, 200, 500]
    print '************************************************'
    print '                 randomForest                   '
    print '************************************************'   
    True_neg = np.zeros((1,len(param)))
    sensibility = np.zeros((1,len(param)))
    True_pos = np.zeros((1,len(param)))
    specificity = np.zeros((1,len(param)))
    accuracy = np.zeros((1,len(param)))
    fmeasure = np.zeros((1,len(param)))
    
    clfs = []
    for p in param:
        clfs += [RandomForestClassifier(n_estimators = p, class_weight = class_weight, min_samples_leaf = min_samples_leaf, n_jobs = -1)]
    
# train + valid 
    for i in range(0,len(param)):
        clfs[i].fit(X_train,y_train)
        y_val_pred = clfs[i].predict(X_val)
        confus = confusion_matrix(y_val, y_val_pred)
        True_neg = confus[0,0]
        True_pos = confus[1,1]
        print confus
        # capacite de detecter les csp+
        sensibility = True_pos * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity = True_neg * 1.0 / sum(confus[0,::]) 
        accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
        precision = confus[1,1]*1.0/(confus[1,1]+confus[0,1])
        fmeasure = 2*(sensibility * precision) * 1.0 / (sensibility + precision)
        #clf_name = "model_RF_%s.pkl" % (param[i])        
        #joblib.dump(clfs[i], clf_name,compress=True)
        print "randomForest %s: Validation Accuracy: %0.3f" % (param[i],accuracy)
        print "randomForest %s: Validation sensitivity: %0.3f" % (param[i],sensibility)
        print "randomForest %s: Validation specificity: %0.3f" % (param[i],specificity)
        print "randomForest %s: Validation F1 measure : %0.3f" % (param[i],fmeasure)
# test    
    for i in range(0,len(param)):
        y_test_pred = clfs[i].predict(X_test)
        confus = confusion_matrix(y_test, y_test_pred)
        True_neg = confus[0,0]
        True_pos = confus[1,1]
        # capacite de detecter les csp+
        sensibility = True_pos * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity = True_neg * 1.0 / sum(confus[0,::]) 
        accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
        precision = confus[1,1]*1.0/(confus[1,1]+confus[0,1])
        fmeasure = 2*(sensibility * precision) * 1.0 / (sensibility + precision)
        print "randomForest (n_estimator = %s): Test Accuracy: %0.3f" % (param[i],accuracy)
        print "randomForest (n_estimator = %s): Test sensitivity: %0.3f" % (param[i],sensibility)
        print "randomForest (n_estimator = %s): Test specificity: %0.3f" % (param[i],specificity)
        print "randomForest (n_estimator = %s): Test F1 measure : %0.3f" % (param[i],fmeasure)

#==============================================================================
#  function : SVM_ttv
#==============================================================================
def SVM_ttv(X_train, X_val, X_test, y_train, y_val, y_test):                      
    param = [0.1,1,100]
    print '************************************************'
    print '                 SVM test                       '
    print '************************************************'   
    True_neg = np.zeros((1,len(param)))
    sensibility = np.zeros((1,len(param)))
    True_pos = np.zeros((1,len(param)))
    specificity = np.zeros((1,len(param)))
    accuracy = np.zeros((1,len(param)))
    fmeasure = np.zeros((1,len(param)))
    
    clfs = []
    for p in param:
        clfs += [svm.SVC(kernel='linear', C=p)]
    
# train + valid 
    for i in range(0,len(param)):
        clfs[i].fit(X_train,y_train)
        y_val_pred = clfs[i].predict(X_val)
        confus = confusion_matrix(y_val, y_val_pred)
        True_neg = confus[0,0]
        True_pos = confus[1,1]
        print confus
        # capacite de detecter les csp+
        sensibility = True_pos * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity = True_neg * 1.0 / sum(confus[0,::]) 
        accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
        precision = confus[1,1]*1.0/(confus[1,1]+confus[0,1])
        fmeasure = 2*(sensibility * precision) * 1.0 / (sensibility + precision)
        #clf_name = "model_SVM_%s.pkl" % (param[i])        
        #joblib.dump(clfs[i], clf_name,compress=True)
        print "SVM %s: Validation Accuracy: %0.3f" % (param[i],accuracy)
        print "SVM %s: Validation sensitivity: %0.3f" % (param[i],sensibility)
        print "SVM %s: Validation specificity: %0.3f" % (param[i],specificity)
        print "SVM %s: Validation F1 measure : %0.3f" % (param[i],fmeasure)
# test    
    for i in range(0,len(param)):
        y_test_pred = clfs[i].predict(X_test)
        confus = confusion_matrix(y_test, y_test_pred)
        True_neg = confus[0,0]
        True_pos = confus[1,1]
        # capacite de detecter les csp+
        sensibility = True_pos * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity = True_neg * 1.0 / sum(confus[0,::]) 
        accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
        precision = confus[1,1]*1.0/(confus[1,1]+confus[0,1])
        fmeasure = 2*(sensibility * precision) * 1.0 / (sensibility + precision)
        print "SVM (C = %s): Test Accuracy: %0.3f" % (param[i],accuracy)
        print "SVM (C = %s): Test sensitivity: %0.3f" % (param[i],sensibility)
        print "SVM (C = %s): Test specificity: %0.3f" % (param[i],specificity)
        print "SVM (C = %s): Test F1 measure : %0.3f" % (param[i],fmeasure)

#==============================================================================
#  function : feature_cluster_ttv
#==============================================================================
def feature_cluster_ttv(X, y, weights = False, num_inters = 100, percentiles = False):
    print '************************************************'
    print '\n'
    #===============
    t1 = time.time()
    n = X.shape[1]
    print 'step 1 : score computation'
    mask_cspp = y.astype(bool) # csp+ mask
    n_cspp = X[mask_cspp,:].sum(axis = 0) # number of csp+ visits
    n_cspm = X[~mask_cspp,:].sum(axis = 0) # number of csp- visits
    if weights:
        w = [2/float(3),1/float(3)]
        mean = 0
    else:
        w = [1,1]
        mean = -1/float(3)
    site_contrast = [ (w[0]*n_cspp[i] - w[1]*n_cspm[i]) * 1.0 / (w[0]*n_cspp[i] + w[1]*n_cspm[i]) if n_cspp[i] + n_cspm[i] != 0 else mean for i in np.arange(n) ]
    print '........................... : site_contrast calculated'
    print 'step 2 : intervals creation'
    num_inters = num_inters # number of intervals - 1 in histogram
    percentiles = percentiles # type of intervals
    if percentiles :
        inter = np.percentile(site_contrast, np.linspace(0,100,num_inters+1) )
    else :
        inter = np.linspace(-1,1,num_inters+1)
    print 'check : %d intervals '%(len(inter))
    t2 = time.time()
    print "............................: Elasped time for score computation: %f"%(t2-t1)
   #===============
    print 'step 3 : building mask'
    #classification of each site into each interval
    count = 0
    site_clf = np.zeros((1,X.shape[1]))
    for low,high in zip(inter[:num_inters],inter[1:]):
        if low==-1 :
            site_clf +=  count * np.logical_and(site_contrast <= high, site_contrast >= low)
        else:
            site_clf +=  count * np.logical_and(site_contrast <= high, site_contrast > low)
        count += 1
        site_clf = site_clf.astype(int)
    t3 = time.time()
    print "............................: Elasped time for features clustering mask : %f"%(t3-t2)
    return site_clf

def combine_ttv(X_train, X_val, X_test, grouping_features):
    t1 = time.time()
    print 'step 1 : matrix creation'
    num_inters = np.max(grouping_features)+1
    print 'check : %d intervals '%(num_inters)
    X_train_c = np.zeros((X_train.shape[0],num_inters))
    X_val_c   = np.zeros((X_val.shape[0],num_inters))
    X_test_c  = np.zeros((X_test.shape[0],num_inters))
    
    for feat in np.arange(num_inters):
        ind = (grouping_features == feat)
        X_train_c[:,feat] = X_train[:,ind[0,:]].sum(axis = 1).ravel()
        X_val_c[:,feat] = X_val[:,ind[0,:]].sum(axis = 1).ravel()
        X_test_c[:,feat] = X_test[:,ind[0,:]].sum(axis = 1).ravel()
        #print '......................................: assigning %d/%d'%(feat,num_inters)
    t2 = time.time()
    print "....................................: Elasped time for matrices reshaping : %f"%(t2-t1)
    return X_train_c, X_val_c, X_test_c
