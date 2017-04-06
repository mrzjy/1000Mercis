# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 20:22:06 2017

@author: Kevin
"""

import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from math import floor
from sklearn.metrics import confusion_matrix
from sklearn import linear_model, svm
import os
import pandas as pd
import time
import scipy.sparse as sps
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, Binarizer, normalize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import colorsys
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

#==============================================================================
# SAVING / LOADING
#==============================================================================
# ========
def save_sparse_csr(path, filename, array):
    np.savez(os.path.join(path, filename), data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def save_full(path, filename, vector):
    np.savez(os.path.join(path, filename), data=vector)

# ========
def load_sparse_csr(path_to):
    loader = np.load(path_to)
    matrix = sps.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                            shape=loader['shape'])
    return matrix

def load_full(path_to):
    loader = np.load(path_to)
    array = loader['data']
    return array

# ========
def static_load_csr(path_to):
    X = load_sparse_csr(path_to + '_X.npz')
    y = load_full(path_to + '_y.npz')
    return X, y

# ========
def hyperParamSearch(X_train, y_train, X_test, y_test, clf_name="logistic", preprocess = 'Std', metric = 'euclidean'):

    # PREPROCESSING = FEATURES SCALING

    if preprocess == 'MaxMin':
        preprocessing = MaxAbsScaler()
        #preprocessing.fit(X_train)
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.fit_transform(X_test)

    if preprocess == 'Binarization':
        preprocessing = Binarizer()
        #preprocessing.fit(X_train)
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.fit_transform(X_test)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'Std':
        preprocessing = StandardScaler(with_mean=False)
        #preprocessing.fit(X_train)
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.fit_transform(X_test)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'full_std':
        preprocessing = StandardScaler()
        X_train = preprocessing.fit_transform(X_train.toarray())
        X_test = preprocessing.fit_transform(X_test.toarray())
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'norm':
        X_train = normalize(X_train.toarray(), axis=0, norm='l1')
        print 'preprocess %s completed'%(preprocess)

    if clf_name == "logistic":
        params = 10**np.linspace(-8,2,13)
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

            ### Accuracy Part ###
            y_test_pred = clf.predict(X_test)
            confus = confusion_matrix(y_test, y_test_pred)
            print confus
            True_neg = confus[0, 0]
            True_pos = confus[1, 1]
            # capacite de detecter les csp+
            sensitivity = True_pos * 1.0 / sum(confus[1, ::])
            # capacite de detecter les csp-
            specificity = True_neg * 1.0 / sum(confus[0, ::])
            accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
            print 'accuracy : %s'%(accuracy)
            print 'sensitivity : %s' % (sensitivity)
            print 'specificity : %s' % (specificity)

            ### ROC Part ###
            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label= 'parameters : C=' + str(C) + ' (area = %0.2f)' % ROC_auc)
            i += 1
            joblib.dump(clf, clf_name + '_' + str(C), compress=True)
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing'%(clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.show()
        print 'ROC curves done'

    if clf_name == "randomforest":
        params = [50,100,200]
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
            ### Accuracy Part ###
            y_test_pred = clf.predict(X_test)
            confus = confusion_matrix(y_test, y_test_pred)
            print confus
            True_neg = confus[0, 0]
            True_pos = confus[1, 1]
            # capacite de detecter les csp+
            sensitivity = True_pos * 1.0 / sum(confus[1, ::])
            # capacite de detecter les csp-
            specificity = True_neg * 1.0 / sum(confus[0, ::])
            accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
            print 'accuracy : %s' % (accuracy)
            print 'sensitivity : %s' % (sensitivity)
            print 'specificity : %s' % (specificity)

            ### ROC Part ###
            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label= 'parameters : N=' + str(nb) + ' (area = %0.2f)' % ROC_auc)
            i += 1
            joblib.dump(clf, clf_name+'_'+str(nb), compress=True)
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing'%(clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.show()
        print 'ROC curves done'

    if clf_name == "kNN":
        params = [10, 50, 75,  100]
        N = len(params)
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        print 'colors generation'
        i = 0
        plt.figure()
        lineWidth = 2

        for k in params:
            clf = KNeighborsClassifier(n_jobs=-1, n_neighbors = k, metric = metric)
            clf.fit(X_train, y_train)
            print 'kNN fitted : %d/%d'%(i+1,N)
            ### Accuracy Part ###
            y_test_pred = clf.predict(X_test)
            confus = confusion_matrix(y_test, y_test_pred)
            print confus
            True_neg = confus[0, 0]
            True_pos = confus[1, 1]
            # capacite de detecter les csp+
            sensitivity = True_pos * 1.0 / sum(confus[1, ::])
            # capacite de detecter les csp-
            specificity = True_neg * 1.0 / sum(confus[0, ::])
            accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
            print 'accuracy : %s' % (accuracy)
            print 'sensitivity : %s' % (sensitivity)
            print 'specificity : %s' % (specificity)

            ### ROC Part ###
            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label= 'parameters : k=' + str(k) + ' (area = %0.2f)' % ROC_auc)
            i += 1
            joblib.dump(clf, clf_name + '_' + str(k), compress=True)
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing'%(clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.show()
        print 'ROC curves done'

    if clf_name == "naiveBayesM":
        params = [1.]
        N = len(params)
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        print 'colors generation'
        i = 0
        plt.figure()
        lineWidth = 2

        for a in params:
            clf = MultinomialNB(alpha = a)
            clf.fit(X_train, y_train)
            print 'naiveBayes fitted : %d/%d' % (i + 1, N)
            ### Accuracy Part ###
            y_test_pred = clf.predict(X_test)
            confus = confusion_matrix(y_test, y_test_pred)
            print confus
            True_neg = confus[0, 0]
            True_pos = confus[1, 1]
            # capacite de detecter les csp+
            sensitivity = True_pos * 1.0 / sum(confus[1, ::])
            # capacite de detecter les csp-
            specificity = True_neg * 1.0 / sum(confus[0, ::])
            accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
            print 'accuracy : %s' % (accuracy)
            print 'sensitivity : %s' % (sensitivity)
            print 'specificity : %s' % (specificity)

            ### ROC Part ###
            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label='parameters : a=' + str(a) + ' (area = %0.2f)' % ROC_auc)
            i += 1
            joblib.dump(clf, clf_name + '_' + str(a), compress=True)
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing' % (clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.show()
        print 'ROC curves done'

    if clf_name == "linSVM":
        params = 10 ** np.linspace(-5, 1, 6)
        N = len(params)
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        print 'colors generation'
        i = 0
        plt.figure()
        lineWidth = 2

        for c in params:
            clf = LinearSVC(C = c, penalty='l2', loss='squared_hinge')
            clf.fit(X_train, y_train)
            print 'linSVM fitted : %d/%d' % (i + 1, N)
            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label='parameters : C=' + str(c) + ' (area = %0.2f)' % ROC_auc)
            i += 1
            joblib.dump(clf, clf_name + '_' + str(c), compress=True)
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing' % (clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.show()
        print 'ROC curves done'

    if clf_name == "voting":
        params = ['soft','hard']
        N = len(params)
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        print 'colors generation'
        i = 0
        plt.figure()
        lineWidth = 2

        for v in params:
            clf1 = LogisticRegression(n_jobs=-1, penalty = 'l2', C = 1e-6)
            clf2 = RandomForestClassifier(n_jobs=-1, n_estimators=50)
            #clf4 = MultinomialNB()
            #clf5 = KNeighborsClassifier(n_jobs=-1, n_neighbors = 10, metric = 'euclidean')

            #clf = VotingClassifier(estimators=[('lr', clf1), ('RF', clf2), ('mnb', clf4), ('kNN', clf5)],
            #                        voting=v)
            clf = VotingClassifier(estimators=[('lr', clf1), ('RF', clf2)],
                                    voting=v)
            clf.fit(X_train, y_train)
            print 'Voting fitted : %d/%d' % (i + 1, N)

            ### Accuracy Part ###
            y_test_pred = clf.predict(X_test)
            confus = confusion_matrix(y_test, y_test_pred)
            print confus
            True_neg = confus[0, 0]
            True_pos = confus[1, 1]
            # capacite de detecter les csp+
            sensitivity = True_pos * 1.0 / sum(confus[1, ::])
            # capacite de detecter les csp-
            specificity = True_neg * 1.0 / sum(confus[0, ::])
            accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
            print 'accuracy : %s' % (accuracy)
            print 'sensitivity : %s' % (sensitivity)
            print 'specificity : %s' % (specificity)

            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label='parameters : v=' + str(v) + ' (area = %0.2f)' % ROC_auc)
            i += 1
            joblib.dump(clf, clf_name + '_' + str(v), compress=True)
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing' % (clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.show()
        print 'ROC curves done'

    if clf_name == "AdaBoost":
        params = [1,2,3,4,5]
        N = len(params)
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        print 'colors generation'
        i = 0
        plt.figure()
        lineWidth = 2

        for m in params:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=m),
                               n_estimators=20)
            clf.fit(X_train, y_train)
            print 'Voting fitted : %d/%d' % (i + 1, N)

            ### Accuracy Part ###
            y_test_pred = clf.predict(X_test)
            confus = confusion_matrix(y_test, y_test_pred)
            print confus
            True_neg = confus[0, 0]
            True_pos = confus[1, 1]
            # capacite de detecter les csp+
            sensitivity = True_pos * 1.0 / sum(confus[1, ::])
            # capacite de detecter les csp-
            specificity = True_neg * 1.0 / sum(confus[0, ::])
            accuracy = (True_pos + True_neg) * 1.0 / sum(sum(confus))
            print 'accuracy : %s' % (accuracy)
            print 'sensitivity : %s' % (sensitivity)
            print 'specificity : %s' % (specificity)

            proba = clf.predict_proba(X_test)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_test, ROC_scores, pos_label=1)
            ROC_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=RGB_tuples[i],
                     lw=lineWidth, label='parameters : m=' + str(m) + ' (area = %0.2f)' % ROC_auc)
            i += 1
            joblib.dump(clf, clf_name + '_' + str(m), compress=True)
        plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for %s classifier with %s preprocessing' % (clf_name, preprocess))
        plt.legend(loc="lower right", prop={'size': 6})
        plt.show()
        print 'ROC curves done'

    if clf_name == "xgb":
        cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
        ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic'}
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring='accuracy', cv=5, n_jobs=-1)
        optimized_GBM.fit(X_train, y_train)
        print optimized_GBM.grid_scores_
        print 'Cross-validation !'

def masked_randomForest(X_train, y_train, preprocess = 'Std'):

    # PREPROCESSING = FEATURES SCALING
    if preprocess == 'MaxMin':
        preprocessing = MaxAbsScaler()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'Binarization':
        preprocessing = Binarizer()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'Std':
        preprocessing = StandardScaler(with_mean=False)
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        print 'preprocess %s completed' % (preprocess)

    if preprocess == 'full_std':
        preprocessing = StandardScaler()
        X_train = preprocessing.fit_transform(X_train.toarray())
        print 'preprocess %s completed'%(preprocess)

    if preprocess == 'norm':
        X_train = normalize(X_train.toarray(), axis=0, norm='l1')
        print 'preprocess %s completed'%(preprocess)

    clf = RandomForestClassifier(n_jobs=-1, n_estimators = 50)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_

    inter = [np.percentile(importances, qq) for qq in np.linspace(0, 100, 6)]
    count = 0
    RF_features = np.zeros((1, len(importances)))
    for low, high in zip(inter[:len(inter)], inter[1:]):
        if low == -1:
            RF_features += count * np.logical_and(importances <= high, importances >= low)
        else:
            RF_features += count * np.logical_and(importances <= high, importances > low)
        count += 1
        RF_features = RF_features.astype(int)

    importances.sort()
    fig, ax1 = plt.subplots(1)
    x = np.arange(len(importances))
    ax1.plot(x, importances[::-1], 'b-')
    ax1.set_xlabel('features')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Random Forest importance', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_yscale('log')
    fig.tight_layout()
    plt.title(' RandomForest importance ')
    plt.show()

    return RF_features