# -*- coding: utf-8 -*-
"""
@author: Kevin & Paul
Part 1 :
Functions used for cross-validation
Functions used for loading and saving data
"""

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

#==============================================================================
# function : save and load sparse data into .npz file
#==============================================================================

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

#==============================================================================
# Cross-validation specific functions
#==============================================================================

#==============================================================================
# function : logreg_crossvalid
#==============================================================================

def logreg_crossvalid(X, y, repeat_time = 5, penalty = 'l2'):
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
    matrix_confus = list()
    importances = list()
    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
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
        matrix_confus.append( confus )
        importances.append( clf.coef_ )
        i = i+1

    performance_dict = {'True_neg': True_neg,
                        'True_pos': True_pos,
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'matrix_confus': matrix_confus,
                        'feat_importance': importances};
    print("LogistRegress : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std()))
    print("LogistRegress : Mean sensibility: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std()))
    print("LogistRegress : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std()))
    return performance_dict

#==============================================================================
# function : naiveBayesG_crossvalid (Gaussian)
#==============================================================================
def naiveBayesG_crossvalid(X, y, repeat_time = 5):
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
    matrix_confus = list()
    X = X.toarray() #ne fonctionne qu'avec matrice dense !
    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = GaussianNB()
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
        matrix_confus.append( confus )
        i = i+1

    performance_dict = {'True_neg': True_neg,
                        'True_pos': True_pos,
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'matrix_confus': matrix_confus};
    print("Gaussien Naives Bayes : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std()))
    print("Gaussian Naives Bayes : Mean sensibility: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std()))
    print("Gaussian Naives Bayes : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std()))
    return performance_dict
#==============================================================================
# function : naiveBayesM_crossvalid (Multinomial)
#==============================================================================
def naiveBayesM_crossvalid(X, y, repeat_time = 5):
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
    matrix_confus = list()
    importances = list()
    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = MultinomialNB()
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
        matrix_confus.append( confus )
        importances.append( clf.feature_log_prob_ )
        i = i+1

    performance_dict = {'True_neg': True_neg,
                        'True_pos': True_pos,
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'matrix_confus': matrix_confus,
                        'feat_importance': importances};
    print("Multinomial Naives Bayes : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std()))
    print("Multinomial Naives Bayes : Mean sensibility: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std()))
    print("Multinomial Naives Bayes : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std()))
    return performance_dict

#==============================================================================
# function : naiveBayesB_crossvalid (Bernoulli)
#==============================================================================
def naiveBayesB_crossvalid(X, y, repeat_time = 5):
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
    matrix_confus = list()
    importances = list()
    X = X.astype(bool)
    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = BernoulliNB()
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
        matrix_confus.append( confus )
        importances.append( clf.feature_log_prob_ )
        i = i+1

    performance_dict = {'True_neg': True_neg,
                        'True_pos': True_pos,
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'matrix_confus': matrix_confus,
                        'feat_importance': importances};
    print("Bernoulli Naives Bayes : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std()))
    print("Bernoulli Naives Bayes : Mean sensibility: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std()))
    print("Bernoulli Naives Bayes : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std()))
    return performance_dict

#==============================================================================
# function : LinSVM_crossvalid
#==============================================================================
def LinSVM_crossvalid(X, y, C_best = 1, repeat_time = 5 ):
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
    matrix_confus = list()
    importances = list()

    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = LinearSVC(C = C_best)
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
        matrix_confus.append( confus )
        importances.append( clf.coef_ )
        i = i+1

    performance_dict = {'True_neg': True_neg,
                        'True_pos': True_pos,
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'matrix_confus': matrix_confus,
                        'feat_importance': importances};
    print("LinSVM : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std()))
    print("LinSVM : Mean sensibility: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std()))
    print("LinSVM : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std()))
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
    importances = list()
    matrix_confus = list()
    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators = N)
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
        importances.append( clf.feature_importances_ )
        matrix_confus.append( confus )
        i= i+1
    performance_dict = {'True_neg': True_neg,
                        'True_pos': True_pos,
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'matrix_confus': matrix_confus,
                        'feat_importance': importances};
    print("RandomForest : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std()))
    print("RandomForest : Mean sensibility: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std()))
    print("RandomForest : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std()))
    return performance_dict

#==============================================================================
# function : MLP_crossvalid
#==============================================================================
def MLP_crossvalid(X, y, repeat_time = 5, hidden_layer_sizes = (100,), activation = 'relu' ):
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
#    importances = list()
    matrix_confus = list()
    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = MLPClassifier( hidden_layer_sizes = hidden_layer_sizes, activation = activation )
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
#        importances.append( clf.feature_importances_ )
        matrix_confus.append( confus )
        i = i+1
    performance_dict = {'True_neg': True_neg,
                        'True_pos': True_pos,
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'matrix_confus': matrix_confus};
    print("MLP : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std()))
    print("MLP : Mean sensibility: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std()))
    print("MLP : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std()))
    return performance_dict

#==============================================================================
# function : XGboost_crossvalid
#==============================================================================
def XGboost_crossvalid(X, y, repeat_time = 6):
    True_neg = np.zeros((1,repeat_time))
    sensibility = np.zeros((1,repeat_time))
    True_pos = np.zeros((1,repeat_time))
    specificity = np.zeros((1,repeat_time))
    accuracy = np.zeros((1,repeat_time))
    matrix_confus = list()
    importances = list()
    kf = KFold(n_splits = repeat_time, shuffle=True) #K-fold
    i = 0
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = xgb.XGBClassifier()
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
        matrix_confus.append( confus )
        importances.append( clf.feature_importances_ )
        i = i+1

    performance_dict = {'True_neg': True_neg,
                        'True_pos': True_pos,
                        'sensibility': sensibility,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'matrix_confus': matrix_confus,
                        'feat_importance': importances};
    print("XGboost : Mean Accuracy: %0.3f +/- %0.3f" % (accuracy.mean(), accuracy.std()))
    print("XGboost : Mean sensibility: %0.3f +/- %0.3f" % (sensibility.mean(), sensibility.std()))
    print("XGboost : Mean specificity: %0.3f +/- %0.3f" % (specificity.mean(), specificity.std()))
    return performance_dict
