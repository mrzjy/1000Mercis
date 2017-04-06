# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:37:54 2017

@author: Kevin & Paul - own version of hyperParamSearch 
"""
import numpy as np
from scipy.sparse import csr_matrix
#from sklearn.externals import joblib
import colorsys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# preprocessing
from sklearn.preprocessing import MaxAbsScaler,StandardScaler
from sklearn.preprocessing import Binarizer
# sampling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from imblearn.under_sampling import EditedNearestNeighbours 
# featuring
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.base import BaseEstimator
#from sklearn.grid_search import GridSearchCV, KFold #(sklearn lower ver.)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

from scipy import interp
from sklearn.model_selection import train_test_split, KFold
'''****************************************************************************
preprocessing
****************************************************************************'''
def preprocessMethod(X_train, X_val, preprocess = 'MaxMin' ):
    if preprocess == 'MaxMin':
        preprocessing = MaxAbsScaler()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_val = preprocessing.transform(X_val)

    if preprocess == 'Binarization':
        preprocessing = Binarizer()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_val = preprocessing.transform(X_val)

    if preprocess == 'Std':
        preprocessing = StandardScaler(with_mean=False)
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_val = preprocessing.transform(X_val)
    return X_train, X_val, preprocessing
'''***************************************************************************
sampling
****************************************************************************'''   
def samplingMethod(X_train, y_train, sampling = "None" ):
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
    return X_train, y_train
'''***************************************************************************
featuring
****************************************************************************'''
def featuringMethod(X_train, X_val, y_train, featuring = 'None' ):
    feature = 'None'
    if featuring == 'agglo_custom':
        feature = FeaturesAggregationScore()
    elif featuring == 'reduce_dim':
        feature = KernelPCA(n_components = 100, kernel = 'cosine', n_jobs=-1)
    elif featuring == 'feat_select':
        feature = SelectPercentile(chi2, percentile = 20)
        
    feature.fit(X_train, y_train)
    X_train = feature.transform(X_train)
    X_val = feature.transform(X_val)
    
    return X_train, X_val, feature
'''***************************************************************************
crossValidation
****************************************************************************'''
def crossValid(X_trainVal, y_trainVal, preprocess = 'MaxMin', featuring = 'None', clf_name = 'logistic', sampling = 'None', n_sample = 2000, fig_path = None, cv = 3, random_state = 50):     
    perform = dict()
    perform['AUC'] = list()
    
    clfs = list()
    epoch, lineWidth = 0, 2

    if clf_name == "logistic":
        param = 'C'
        params = {param : [0.001]}
    elif clf_name == "randomForest":
        param = 'n_estimators'
        params = {param : [150]}
    elif clf_name == "KNN":
        param = 'n_neighbors'
        params = {param : [200]}
    elif clf_name == "NBMulti":
        param = 'alpha'
        params = {param : [1,5,10,25]}
    elif clf_name == "Voting":
        param = 'voting'
        params = {param : ['soft']}
    elif clf_name == "MLP":
        param = 'hidden_layer_sizes'
        params = {param : [100,200,400]}    
    N = len(params[param])
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    
    plt.figure()
    
    for param_value in params[param]:
        kf = KFold(n_splits = cv, shuffle=True) #K-fold
        cv, AUC, clf_cv = 0, list(), list()
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, n_sample)
        for train_index, val_index in kf.split(X_trainVal):
            if clf_name == "logistic":
                clf = LogisticRegression(n_jobs=-1, penalty ='l2', C = param_value)
            elif clf_name == "randomForest":
                clf = RandomForestClassifier(n_jobs = -1, n_estimators = param_value)
            elif clf_name == "KNN":
                clf = KNeighborsClassifier(n_jobs = -1, n_neighbors = param_value)
            elif clf_name == "NBMulti":    
                clf = MultinomialNB(alpha = param_value)
            elif clf_name == "MLP":
                clf = MLPClassifier(hidden_layer_sizes = param_value)
            elif clf_name == "Voting":   
                clf1 = LogisticRegression(n_jobs=-1, C = 0.001)
                clf2 = RandomForestClassifier(n_jobs = -1, n_estimators = 150)
                clf3 = MultinomialNB(alpha = 5)
                clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                        voting='soft', n_jobs = -1)
            X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size = 0.15, random_state = random_state)
            # sampling
            if sampling != 'None':
                X_train, y_train = samplingMethod(X_train, y_train, sampling = sampling )            
            
            # preprocessing
            X_train, X_val, preprocessing = preprocessMethod(X_train,X_val, preprocess = preprocess )
                
            # featuring
            if featuring != 'None':
                X_train, X_val, featureMethod = featuringMethod(X_train, X_val,y_train, featuring = featuring )
            clf.fit(X_train, y_train)   
            proba = clf.predict_proba(X_val)
            ROC_scores = proba[:, 1]  # thresholds for ROC
            fpr, tpr, _ = roc_curve(y_val, ROC_scores, pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            AUC.append(auc(fpr, tpr))
            clf_cv.append(clf)
            cv += 1
        mean_tpr /= cv
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        best_cv = np.array(AUC).argmax()            

        perform['AUC'].append( mean_auc )
        plt.plot(fpr, tpr, color=RGB_tuples[epoch],
                 lw=lineWidth, 
                 label= 'parameters : %s ='%(param) + str(param_value) + ' (mean area = %0.2f)' % mean_auc)
        clfs.append(clf_cv[best_cv])
        epoch += 1
        
        
    plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--', label='Monkey')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(n_sample) + ' : ROC Curves for %s classifier with %s preprocessing'%(clf_name, preprocess))
    plt.legend(loc="lower right", prop={'size': 6})
    plt.savefig(fig_path + 'n_samples = ' + str(n_sample) + ' : Mean ROC(params)_%s_%s_%s with %s fold cv' % (clf_name, preprocess,featuring, cv))
        
    best = np.array(perform['AUC']).argmax()
    param_best = params[param][best]
    clf_best = clfs[best]
    name_best = '%s preprocessed_%s_%s (%s = %s)'%(preprocess, clf_name, featuring, param, param_best)
    return name_best, preprocessing, clf_best, featureMethod
            
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
            