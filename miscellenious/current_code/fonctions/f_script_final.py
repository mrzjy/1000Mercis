# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 20:22:06 2017

@author: Kevin
"""
import numpy as np
from scipy.sparse import csr_matrix
from imblearn.over_sampling import SMOTE
#from imblearn.combine import SMOTEENN 
#from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.externals import joblib
import os
# visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import colorsys
# preprocessing
from sklearn.preprocessing import MaxAbsScaler, StandardScaler,Binarizer
from sklearn.feature_selection import chi2, f_classif, SelectPercentile
from sklearn.decomposition import PCA, KernelPCA
# clfs
from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV #(sklearn lower ver.)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, classification_report

#==============================================================================
# function : get models .pkl
#==============================================================================

def get_files(namedir, model = 'final', extension = ".pkl"):
    names = []
    models = []
    os.chdir(namedir)
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
    return names, models

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
        fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
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
    plt.savefig('img/'+figname)
    
    return 0

#==============================================================================
# functions : hyperParamSearch with variants in pipeline
#==============================================================================
def hyperParamSearch( X_train, y_train, X_test, y_test, clf = "logistic",scoring = 'accuracy',preprocess = 'MaxMin'):
    tuned_parameters = dict()
    if preprocess == 'MaxMin':
        preprocessing = ( 'MaxMin',MaxAbsScaler() )
    if preprocess == 'Binarization':
        preprocessing = ( 'Bin',Binarizer() )
    if preprocess == 'Std':
        preprocessing = ( 'Std',StandardScaler(with_mean=False) )    
    if clf == "logistic":
        #Parameters of pipelines can be set using ‘__’ separated parameter names:
        tuned_parameters = [{'logistic__penalty':['l1','l2'],
                             'logistic__C': [0.000001, 0.00001, 0.0001, 0.005, 0.001, 0.05, 0.01],
                             'logistic__class_weight' : [None,'balanced'] }]
        pipe = Pipeline(steps=[preprocessing, ('logistic', LogisticRegression(n_jobs = -1))])
    if clf == "randomForest":
        tuned_parameters = [{'randomForest__n_estimators':[100, 300, 500],
                             'randomForest__min_samples_leaf':[1,2,5,10,25,50],
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
        save_name = "final_%s(%s based_%s preprocessed).pkl" % (clf, score, preprocess)        
        # print information
        print("INFO: %s model (preprocessed by %s crossvalid based on %s)"%(clf, preprocess, score))        
        print("Best parameters set found on development set:")
        print(estimator.best_params_)
        
        print("%s scores on development set:"%(score))
        means = estimator.cv_results_['mean_test_score']
        stds = estimator.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, estimator.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, estimator.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        joblib.dump(estimator, save_name, compress=True)


def hyperParamSearch_v2(X_train, y_train, X_test, y_test, clf="logistic", scoring='accuracy', preprocess='MaxMin', method = 'feat_select'):
    tuned_parameters = dict()
    if preprocess == 'MaxMin':
        preprocessing = ('MaxMin', MaxAbsScaler())
    if preprocess == 'Binarization':
        preprocessing = ('Bin', Binarizer())

    if clf == "logistic":
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        tuned_parameters = [{'logistic__penalty': ['l1', 'l2'],
                             'logistic__C': [0.0001, 0.001, 0.1, 1, 10],
                             'logistic__class_weight': [None, 'balanced']}]
        #if method == 'agglo_custom':
        #    tuned_parameters[0]['featuresaggregationscore__clusters'] = [50,100,200,500]
        #    pipe = Pipeline(steps=[preprocessing, ('featuresaggregationscore', FeaturesAggregationScore()), ('logistic', LogisticRegression(n_jobs=-1))])
        if method == 'reduce_dim':
            tuned_parameters[0]['kernelpca__n_components'] = [50, 100, 200, 500]
            pipe = Pipeline(steps=[preprocessing, ('kernelpca', KernelPCA(kernel = 'cosine', n_jobs=-1)),('logistic', LogisticRegression(n_jobs=-1))])
        elif method == 'feat_select':
            fselect = SelectPercentile(chi2)
            tuned_parameters[0]['fselect__percentile'] = [20,40,60,80]
            pipe = Pipeline(steps=[preprocessing, ('fselect', fselect),('logistic', LogisticRegression(n_jobs=-1))])

    if clf == "randomForest":
        tuned_parameters = [{'randomForest__n_estimators': [100, 500],
                             'randomForest__min_samples_leaf': [1, 10, 25],
                             'randomForest__class_weight': [None, 'balanced']
                             }]
        #if method == 'agglo_custom':
        #    tuned_parameters[0]['featuresaggregationscore__clusters'] = [50, 100, 200, 500]
        #    pipe = Pipeline(steps=[preprocessing, ('featuresaggregationscore', FeaturesAggregationScore()), ('randomForest', RandomForestClassifier(n_jobs=-1))])
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

def hyperParamSearch_SMOTE(X_train, y_train, X_test, y_test, clf="logistic", scoring='accuracy', preprocess='MaxMin'):
    sm = SMOTE(random_state=1, n_jobs = -1)
    X, y_train = sm.fit_sample(X_train.toarray(), y_train)
    X_train = csr_matrix(X)
    tuned_parameters = dict()
    if preprocess == 'MaxMin':
        preprocessing = ('MaxMin', MaxAbsScaler())
    if preprocess == 'Binarization':
        preprocessing = ('Bin', Binarizer())

    if clf == "logistic":
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        tuned_parameters = [{'logistic__penalty': ['l1', 'l2'],
                            'logistic__C': [1e-5,1e-4, 1e-3, 1e-2,1e-1, 1, 10],
                            'logistic__class_weight': [None, 'balanced']}]
        pipe = Pipeline(steps=[preprocessing, ('logistic', LogisticRegression(n_jobs=-1))])

    if clf == "randomForest":
        tuned_parameters = [{'randomForest__n_estimators': [100,300, 500],
                        'randomForest__min_samples_leaf': [1,2,5, 10, 25],
                        'randomForest__class_weight': [None, 'balanced']
                        }]
        pipe = Pipeline(steps=[preprocessing, ('randomForest', RandomForestClassifier(n_jobs=-1)) ])

    for score in scoring:
        estimator = GridSearchCV(pipe,
                                 tuned_parameters,
                                 cv=3,
                                 scoring=score,
                                 error_score=-1,
                                 n_jobs=-1)
        estimator.fit(X_train, y_train)
        save_name = "final_%s(%s based_%s preprocessed_%s).pkl" % (clf, score, preprocess, 'SMOTE')
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
# class : custom features aggregation
#==============================================================================

class FeaturesAggregationScore(object):
    
    def __init__(self, K = 100, method = 'iso', weights = False):
        self.clusters = K
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
        site_contrast = [(w[0] * n_cspp[i] - w[1] * n_cspm[i]) * 1.0 / (w[0] * n_cspp[i] + w[1] * n_cspm[i]) if n_cspp[i] + n_cspm[i] != 0 else mean for i in np.arange(n)]
                             
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

# ========
def naive_hyperParamSearch(X_train, y_train, X_test, y_test, figPath, clf_name="logistic", preprocess = None):
    os.chdir(figPath)
    if preprocess == 'MaxMin':
        preprocessing = MaxAbsScaler()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_test = preprocessing.transform(X_test)

    if preprocess == 'Binarization':
        preprocessing = Binarizer()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_test = preprocessing.transform(X_test)

    if preprocess == 'Std':
        preprocessing = StandardScaler(with_mean=False)
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_test = preprocessing.transform(X_test)

    if clf_name == "logistic":
        params = 10**np.linspace(-5,-1,5)
        N = len(params)
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        print 'colors generation'
        i = 0
        plt.figure()
        lineWidth = 2

        for C in params:
            clf = LogisticRegression(n_jobs=-1, penalty = 'l2', C = C)
            clf.fit(X_train, y_train)
            print 'Logistic regression fitted : %d/%d'%(i+1,N)
            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label= 'parameters : C=' + str(C) + ' (area = %0.2f)' % ROC_auc)
            i += 1
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing'%(clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.savefig('img/naive_%s_%s' % (clf_name, preprocess))
    print 'ROC curves done'

    if clf_name == "randomForest":
        params = [50,100]
        N = len(params)
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        print 'colors generation'
        i = 0
        plt.figure()
        lineWidth = 2

        for nb in params:
            clf = RandomForestClassifier(n_jobs=-1, n_estimators = nb)
            clf.fit(X_train, y_train)
            print 'randomForest fitted : %d/%d'%(i+1,N)
            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label= 'parameters : N=' + str(nb) + ' (area = %0.2f)' % ROC_auc)
            i += 1
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing'%(clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.savefig('img/naive_%s_%s'%(clf_name, preprocess))
    print 'ROC curves done'
