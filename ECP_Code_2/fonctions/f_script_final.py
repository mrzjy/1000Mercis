# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 20:22:06 2017

@author: Kevin & Paul - Sklearn version of hyperParamSearch
"""
import numpy as np
#from scipy.sparse import csr_matrix
#from imblearn.over_sampling import SMOTE
#from imblearn.combine import SMOTEENN 
#from imblearn.under_sampling import EditedNearestNeighbours 
#from scipy.sparse import csr_matrix
from sklearn.externals import joblib
import os
import colorsys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
# preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Binarizer
# sampling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from imblearn.under_sampling import EditedNearestNeighbours 
# clfs
from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV #(sklearn lower ver.)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.base import BaseEstimator

from sklearn.cross_validation import train_test_split
#==============================================================================
# function : get models .pkl
#==============================================================================
def get_files(namedir, model = 'final', extension = ".pkl", printOut = True):
    names = []
    models = []
    os.chdir(namedir)
    if printOut == True:
        print 'getting files ...'
        if model is not None:
            print 'Model is not None : model = %s'%(model)
            print 'Directory of models : %s'%(namedir)
            for file in os.listdir(namedir):
                if file.endswith(extension) and file.startswith(model):
                    print 'chosen file : %s'%(file)
                    names += [file[:-len(extension)]]
                    models += [ joblib.load(file) ]
        else:
            print 'is model None ? '
            print model
            for file in os.listdir(namedir):
                if file.endswith(extension):
                    names += [file[:-len(extension)]]
                    models += [ joblib.load(file) ]
    else:
        if model is not None:
            for file in os.listdir(namedir):
                if file.endswith(extension) and file.startswith(model):
                    names += [file[:-len(extension)]]
                    models += [ joblib.load(file) ]
        else:
            for file in os.listdir(namedir):
                if file.endswith(extension):
                    names += [file[:-len(extension)]]
                    models += [ joblib.load(file) ]
    return names, models

#==============================================================================
# function : Test performance - csv
#==============================================================================
def performTest(X_test,y_test, sourcePathModels, models = 'final'):
    names, clf = get_files(sourcePathModels, model = models, printOut = False)
    print 'model,accuracy,sensitivity,specificity,fmeasure' 
    N = len(names)
    for i in range(N):
        y_pred = clf[i].predict(X_test)
        confus = confusion_matrix(y_test, y_pred)
        True_neg = confus[0,0]
        True_pos = confus[1,1]
        # capacite de detecter les csp+
        sensibility = True_pos * 1.0 / sum(confus[1,::]) 
        # capacite de detecter les csp-
        specificity = True_neg * 1.0 / sum(confus[0,::]) 
        accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
        precision = confus[1,1]*1.0/(confus[1,1]+confus[0,1])
        fmeasure = 2*(sensibility * precision) * 1.0 / (sensibility + precision)  
        print( '%s,%s,%s,%s,%s'%(names[i],str(accuracy),str(sensibility),str(specificity),str(fmeasure)) )

#==============================================================================
# function : ROC Curve
#==============================================================================
def ROC_curvesCV(X_test,y_test, sourcePathModels,figPath, figname = 'not_mentionned', models = 'final'):
    print 'ROC curves for best models from CV'
#   loading best cross-val models
    names, clf = get_files(sourcePathModels, model = models)
    print 'Models loaded'
#   colors
    N = len(names)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    print 'colors generation'
    os.chdir(figPath)
    i = 0
    plt.figure()
    lineWidth = 2
    
    
    for i in range(N):
#   predict_proba method is needed for estimator //otherwise : predict
        proba = clf[i].predict_proba(X_test)
#   proba.shape = [n_samples, n_classes]
        ROC_scores = proba[:,1] # thresholds for ROC
        fpr, tpr, threshold = roc_curve(y_test, ROC_scores, pos_label=1)
        print threshold
        ROC_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=RGB_tuples[i],
                 lw=lineWidth, label=names[i]+'(area = %0.2f)' % ROC_auc)
                 
    plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label = 'Monkey')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for CV best results')
    plt.legend(loc="lower right", prop={'size':6})
    plt.savefig(figname)
    
    return 0
#==============================================================================
# function : PRC Curve
#==============================================================================
def PRC_curvesCV(X_test,y_test, sourcePathModels,figPath, figname = 'not_mentionned', models = 'final'):
    print 'PRC curves for best models from CV'
#   loading best cross-val models
    names, clf = get_files(sourcePathModels, model = models)
    print 'Models loaded'
#   colors
    N = len(names)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    print 'colors generation'
    os.chdir(figPath)
    i = 0
    plt.figure()
    lineWidth = 2
    
    
    for i in range(N):
#   predict_proba method is needed for estimator //otherwise : predict
        proba = clf[i].predict_proba(X_test)
#   proba.shape = [n_samples, n_classes]
        y_scores = proba[:,1] 
        precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=1)
        PRC_auc = average_precision_score(y_test, y_scores, average="micro")
        plt.plot(recall, precision, color=RGB_tuples[i],
                 lw=lineWidth, label=names[i]+'(area = %0.2f)' % PRC_auc)
                 
    plt.plot([0, 1], [1, 0], color='navy', lw=lineWidth, linestyle='--', label = 'Monkey')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PRC Curves for CV best results')
    plt.legend(loc="upper right", prop={'size':6})
    plt.savefig(figname)
    
    return 0

#==============================================================================
# function : hyperParamSearch
#==============================================================================
def hyperParamSearch( X_train, y_train, X_test, y_test, clf = "logistic",
                     scoring = 'accuracy',preprocess = 'MaxMin', sampling ="None"  ):
    tuned_parameters = dict()
# sampling
    if sampling == "SMOTE":
        sm = SMOTE(random_state=42, n_jobs = -1)
        X, y_train = sm.fit_sample(X_train.toarray(), y_train)
        X_train = csr_matrix(X)
    elif sampling == "ENN":
        enn = EditedNearestNeighbours(random_state=42, n_jobs = -1)
        X, y_train = enn.fit_sample(X_train.toarray(), y_train)    
        X_train = csr_matrix(X)
    elif sampling == "SMOTEENN":
        sme = SMOTEENN(random_state=42, n_jobs = -1)
        X, y_train = sme.fit_sample(X_train.toarray(), y_train) 
        X_train = csr_matrix(X)    
# preprocessing
    if preprocess == 'MaxMin':
        preprocessing = ( 'MaxMin',MaxAbsScaler() )
    if preprocess == 'Binarization':
        preprocessing = ( 'Bin',Binarizer() )
        
    if clf == "logistic":
        #Parameters of pipelines can be set using ‘__’ separated parameter names:
        tuned_parameters = [{'logistic__penalty':['l2'],
                             'logistic__C': [0.001, 0.1, 1, 10, 100],
                             'logistic__class_weight' : [None] }]
        pipe = Pipeline(steps=[preprocessing, ('logistic', LogisticRegression(n_jobs = -1))])
    if clf == "randomForest":
        tuned_parameters = [{'randomForest__n_estimators':[100, 500],
                             'randomForest__min_samples_leaf':[1,10,25],
                             'randomForest__class_weight' : [None,'balanced']
                            }]
        pipe = Pipeline(steps=[preprocessing, ('randomForest', RandomForestClassifier(n_jobs = -1))])
    if clf == "KNN":
        tuned_parameters = [{'KNN__n_neighbors':[5,10,20,40],
                             'KNN__weights': ['distance','uniform'],
                             'KNN__metric':['euclidean','manhattan']
                            }]
        pipe = Pipeline(steps=[preprocessing, ('KNN', KNeighborsClassifier(n_jobs = -1))])
    for score in scoring:
        estimator = GridSearchCV(pipe,
                                 tuned_parameters,
                                 cv = 3,
                                 scoring = score, 
                                 error_score = -1,
                                 n_jobs = -1)
        estimator.fit(X_train, y_train)
        
        save_name = "final_%s(%s based_%s preprocessed_%s sampling).pkl" % (clf, score, preprocess,sampling) 
        joblib.dump(estimator, save_name, compress=True)
        # print information
        print("************************* GENERAL INFO ***********************")
        print(" - classifier : %s"%(clf) )
        print(" - sampling : %s"%(sampling) )        
        print(" - preprocessing : %s"%(preprocess) )
        print(" - hyperParam based on : %s"%(score) )
        print("**************************************************************")        
        print("Best parameters set found on development set:")
        print(estimator.best_params_)

        print("%s scores on development set:"%(score))
        means = estimator.cv_results_['mean_test_score']
        stds = estimator.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, estimator.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, estimator.predict(X_test)
#        print(classification_report(y_true, y_pred))
        
        confus = confusion_matrix(y_true, y_pred)
        print '*****CV python****'
        print confus
        
def hyperParamSearch_v2(X_train, y_train, X_test, y_test, clf="logistic", scoring='accuracy', preprocess='MaxMin', method = 'feat_select'):
    tuned_parameters = dict()
    if preprocess == 'MaxMin':
        preprocessing = ('MaxMin', MaxAbsScaler())
    if preprocess == 'Binarization':
        preprocessing = ('Bin', Binarizer())

    if clf == "logistic":
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        tuned_parameters = [{'logistic__penalty': ['l1', 'l2'],
                             'logistic__C': [0.0001, 0.1, 1, 10],
                             'logistic__class_weight': [None, 'balanced']}]
        if method == 'agglo_custom':
            tuned_parameters[0]['featuresaggregationscore__clusters'] = [40,100,250,500]
            pipe = Pipeline(steps=[('featuresaggregationscore', FeaturesAggregationScore()),preprocessing, ('logistic', LogisticRegression(n_jobs=-1))])
        if method == 'reduce_dim':
            tuned_parameters[0]['kernelpca__n_components'] = [50]
            pipe = Pipeline(steps=[preprocessing, ('kernelpca', KernelPCA(kernel = 'cosine', n_jobs=-1)),('logistic', LogisticRegression(n_jobs=-1))])
        elif method == 'feat_select':
            fselect = SelectPercentile(chi2)
            tuned_parameters[0]['fselect__percentile'] = [20]
            pipe = Pipeline(steps=[preprocessing, ('fselect', fselect),('logistic', LogisticRegression(n_jobs=-1))])

    if clf == "randomForest":
        tuned_parameters = [{'randomForest__n_estimators': [100, 500],
                             'randomForest__min_samples_leaf': [1, 10, 25],
                             'randomForest__class_weight': [None, 'balanced']
                             }]
        if method == 'agglo_custom':
            tuned_parameters[0]['featuresaggregationscore__clusters'] = [50, 100, 200, 500]
            pipe = Pipeline(steps=[preprocessing, ('featuresaggregationscore', FeaturesAggregationScore()), ('randomForest', RandomForestClassifier(n_jobs=-1))])
        if method == 'reduce_dim':
            tuned_parameters[0]['kernelpca__n_components'] = [50, 100, 200, 500]
            pipe = Pipeline(steps=[preprocessing, ('kernelpca', KernelPCA(kernel = 'cosine', n_jobs=-1)),('randomForest', RandomForestClassifier(n_jobs=-1))])
        elif method == 'feat_select':
            fselect = SelectPercentile(chi2)
            tuned_parameters[0]['fselect__percentile'] = [20, 40, 60, 80]
            pipe = Pipeline(steps=[preprocessing, ('fselect', fselect), ('randomForest', RandomForestClassifier(n_jobs=-1)) ])
    print tuned_parameters
    for score in scoring:
        estimator = GridSearchCV(pipe,
                                 tuned_parameters,
                                 cv=3,
                                 scoring=score,
                                 error_score=-1,
                                 n_jobs=-1)
        estimator.fit(X_train, y_train)
        save_name = "final_%s(%s based_%s preprocessed_%s).pkl" % (clf, score, preprocess, method)
        # print information
        print("INFO: %s model (preprocessed by %s crossvalid based on %s)" % (clf, preprocess, score))
        print("Best parameters set found on development set:")
        print(estimator.best_params_)

        print("%s scores on development set:" % (score))
        means = estimator.cv_results_['mean_test_score']
        stds = estimator.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, estimator.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, estimator.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        joblib.dump(estimator, save_name, compress=True)
    
#==============================================================================
# local behaviours
#==============================================================================    
def local(X_trainVal, y_trainVal, X_test, y_test,clf = 'logistic', cv = 3, train_size = 0.7):
    param = ["l1","l2"]  
    clf1 = LogisticRegression(penalty = param[0])    
    clf2 = LogisticRegression(penalty = param[1])
    clfs = [clf1,clf2]     
    for epoch in range(0,cv):
        X_train, X_val, y_train, y_val = train_test_split( X_trainVal, y_trainVal, train_size = train_size, stratify = None, random_state = 50)
        print 'train contains %d elements, with ratio %f of class 1'%(len(y_train), y_train.sum()/float(len(y_train)) )
        print 'val contains %d elements, with ratio %f of class 1'%(len(y_val), y_val.sum()/float(len(y_val)) ) 
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
            clf_name = "Localmodel_logreg_%s.pkl" % (param[i])        
            joblib.dump(clfs[i], clf_name, compress=True)
            print "%s (penalty = %s): Validation Accuracy: %0.3f" % (clf,param[i],accuracy)
            print "%s (penalty = %s): Validation sensitivity: %0.3f" % (clf,param[i],sensibility)
            print "%s (penalty = %s): Validation specificity: %0.3f" % (clf,param[i],specificity)
            print "%s (penalty = %s): Validation F1 measure : %0.3f" % (clf,param[i],fmeasure)

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
        clf_name = "model_%s_%s.pkl" % (clf, param[i])        
        joblib.dump(clfs[i], clf_name, compress=True)
    

#==============================================================================
# class : custom features aggregation
#==============================================================================
class FeaturesAggregationScore(BaseEstimator):
    
    def __init__(self, clusters = 100, method = 'iso', weights = False):
        self.clusters = clusters
        self.method = method
        self.weights = weights
        self.mask = None
        
    def transform(self, X):
        X = X.toarray() # ou utiliser deep_copy ?
        X_loc = np.zeros((X.shape[0], self.clusters))
        for feat in np.arange(self.clusters):
            ind = (self.mask == feat)
            X_loc[:, feat] = X[:, ind[0, :]].sum(axis=1).ravel()
        return X_loc
    
    def fit(self, X, y):
        #-----------------------------#
        # Score calculation
        # ----------------------------#
        n = X.shape[1]
        mask_cspp = y.astype(bool)            # csp+ mask
        n_cspp = X[mask_cspp, :].sum(axis=0)  # number of csp+ visits
        n_cspm = X[~mask_cspp, :].sum(axis=0) # number of csp- visits
        if self.weights:
            w = [2 / float(3), 1 / float(3)]
            mean = 0
        else:
            w = [1, 1]
            mean = -1 / float(3)
#        print'************************************'
#        print n_cspm.shape
#        print'************************************'
        site_contrast = [(w[0] * n_cspp[0,i] - w[1] * n_cspm[0,i]) * 1.0 / (w[0] * n_cspp[0,i] + w[1] * n_cspm[0,i]) if n_cspp[0,i] + n_cspm[0,i] != 0 else mean for i in np.arange(n)]
        
        #----------------------------#
        #  Features aggregation
        #----------------------------#
        if self.method == 'percentiles':
            inter = [np.percentile(site_contrast, qq) for qq in np.linspace(0, 100, self.clusters + 1)]
        elif self.method == 'iso':
            inter = np.linspace(-1, 1, self.clusters + 1)

        count = 0
        site_clf = np.zeros((1, X.shape[1]))
        for low, high in zip(inter[:self.clusters], inter[1:]):
            if low == -1:
                site_clf += count * np.logical_and(site_contrast <= high, site_contrast >= low)
            else:
                site_clf += count * np.logical_and(site_contrast <= high, site_contrast > low)
            count += 1
        site_clf = site_clf.astype(int)
        self.mask = site_clf
        return self

