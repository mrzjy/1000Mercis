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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Binarizer
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


def hyperParamSearch_v2(X_train, y_train, X_test, y_test, clf="logistic", scoring='accuracy', preprocess='MaxMin', method = 'agglo_custom'):
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
        if method == 'agglo_custom':
            tuned_parameters[0]['featuresaggregationscore__clusters'] = [50,100,200,500]
            pipe = Pipeline(steps=[preprocessing, ('featuresaggregationscore', FeaturesAggregationScore()), ('logistic', LogisticRegression(n_jobs=-1))])
        elif method == 'reduce_dim':
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
        if method == 'agglo_custom':
            tuned_parameters[0]['featuresaggregationscore__clusters'] = [50, 100, 200, 500]
            pipe = Pipeline(steps=[preprocessing, ('featuresaggregationscore', FeaturesAggregationScore()), ('randomForest', RandomForestClassifier(n_jobs=-1))])
        elif method == 'reduce_dim':
            tuned_parameters[0]['kernelpca__n_components'] = [50, 100, 200, 500]
            pipe = Pipeline(steps=[preprocessing, ('kernelpca', KernelPCA(kernel = 'cosine', n_jobs=-1)),('randomForest', RandomForestClassifier(n_jobs=-1))])
        elif method == 'feat_select':
            fselect = SelectPercentile(chi2)
            tuned_parameters[0]['fselect__percentile'] = [20, 40, 60, 80]
            pipe = Pipeline(steps=[preprocessing, ('fselect', fselect), ('randomForest', RandomForestClassifier(n_jobs=-1)) ])

    for score in scoring:
        estimator = GridSearchCV(pipe,
                                 tuned_parameters,
                                 cv=3,
                                 scoring=score,
                                 error_score=-1,
                                 n_jobs=-1)
        estimator.fit(X_train, y_train)
        save_name = "final_%s(%s based_%s preprocessed).pkl" % (clf, score, preprocess)
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

def hyperParamSearch_SMOTE(X_train, y_train, X_test, y_test, clf="logistic", scoring='accuracy', preprocess='MaxMin', method = 'agglo_custom'):
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
                            'logistic__C': [0.0001, 0.001, 0.1, 1, 10],
                            'logistic__class_weight': [None, 'balanced']}]
        if method == 'agglo_custom':
            tuned_parameters[0]['featuresaggregationscore__clusters'] = [50,100,200,500]
            pipe = Pipeline(steps=[preprocessing, ('featuresaggregationscore', FeaturesAggregationScore()), ('logistic', LogisticRegression(n_jobs=-1))])
        elif method == 'reduce_dim':
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
        if method == 'agglo_custom':
            tuned_parameters[0]['featuresaggregationscore__clusters'] = [50, 100, 200, 500]
            pipe = Pipeline(steps=[preprocessing, ('featuresaggregationscore', FeaturesAggregationScore()), ('randomForest', RandomForestClassifier(n_jobs=-1))])
        elif method == 'reduce_dim':
            tuned_parameters[0]['kernelpca__n_components'] = [50, 100, 200, 500]
            pipe = Pipeline(steps=[preprocessing, ('kernelpca', KernelPCA(kernel = 'cosine', n_jobs=-1)),('randomForest', RandomForestClassifier(n_jobs=-1))])
        elif method == 'feat_select':
            fselect = SelectPercentile(chi2)
            tuned_parameters[0]['fselect__percentile'] = [20, 40, 60, 80]
            pipe = Pipeline(steps=[preprocessing, ('fselect', fselect), ('randomForest', RandomForestClassifier(n_jobs=-1)) ])

    for score in scoring:
        estimator = GridSearchCV(pipe,
                                 tuned_parameters,
                                 cv=3,
                                 scoring=score,
                                 error_score=-1,
                                 n_jobs=-1)
        estimator.fit(X_train, y_train)
        save_name = "final_%s(%s based_%s preprocessed).pkl" % (clf, score, preprocess)
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
# getviz_kpca function : KPCA - visualization [NOT-OPTIMIZED]
#==============================================================================

def getviz_kpca(X,y, figPath, fig_prefix = 'KPCA_viz'):

    # this is a non optimized visualization ! Just for thoughts
    preprocessing = MaxAbsScaler()
    X_train = preprocessing.fit_transform(X)
    print 'preprocessing MaxAbs done'

    os.chdir(figPath)

    reds = y == 0
    blues = y == 1
    kernels = ['cosine','rbf','regular']
    gammas = [1e-4,1e-3,1e-2]

    for k in kernels:
        if k == 'rbf':
            for g in gammas:
                plt.figure()
                kpca = KernelPCA(kernel=k, gamma=g, n_components=2, n_jobs=-1)
                X_kpca = kpca.fit_transform(X_train)
                plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro", label='csp-')
                plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo", label='csp+')
                plt.title("Projection by PCA with %s kernel, gamma = %f" % (k,g))
                plt.xlabel("1st principal component")
                plt.ylabel("2nd component")
                plt.legend(loc="lower right", prop={'size': 6})
                plt.savefig('img/' + fig_prefix + k + 'gamma_'+str(g))
            print 'rbf PCA done'

        elif k == 'regular':
            plt.figure()
            kpca = PCA()
            X_kpca = kpca.fit_transform(X_train)
            plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro", label='csp-')
            plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo", label='csp+')
            plt.title("Projection by PCA")
            plt.xlabel("1st principal component")
            plt.ylabel("2nd component")
            plt.legend(loc="lower right", prop={'size': 6})
            plt.savefig('img/' + fig_prefix+k)

            plt.figure()
            plt.plot(kpca.explained_variance_, linewidth=2)
            plt.xlabel('n_components')
            plt.ylabel('explained_variance_')
            plt.title("Projection by PCA")
            plt.savefig('img/' + fig_prefix + k+'explained_variance')

            print 'PCA done'

        elif k == 'cosine':
            plt.figure()
            kpca = KernelPCA(kernel=k, n_components = 2, n_jobs=-1)
            X_kpca = kpca.fit_transform(X_train)
            plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro", label = 'csp-')
            plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo", label = 'csp+')
            plt.title("Projection by PCA with %s kernel"%(k))
            plt.xlabel("1st principal component")
            plt.ylabel("2nd component")
            plt.legend(loc="lower right", prop={'size': 6})
            plt.savefig('img/' + fig_prefix + k)

            print 'consine PCA done'


