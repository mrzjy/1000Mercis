# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:39:44 2016

@author: paulvernhet
"""
#==============================================================================
# Naïve dictionnary-like approach : using new concepts from LSA on each class
#==============================================================================

#==============================================================================
# Libraries import
#==============================================================================
wdir='/Users/paulvernhet/Desktop/3A/SEMINAIRE 1000mercis/week_1'
from sklearn.svm import SVC, LinearSVC
from math import ceil
from scipy import sparse
import numpy as np
from functions import get_dataset as start
from functions import tops as tops
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score

#==============================================================================
# Data loading & normalization :
#   X,y : Raw data
#   no normalization (first attempt)
#==============================================================================

print("Le chargement des données commence ...\n")
#X, y, l = start.get_dataset()
s_lsa_p = np.load("s_svd_lsa_p.npy")
#U_lsa_p = np.load("U_svd_lsa_p.npy")
V_lsa_p = np.load("V_svd_lsa_p.npy")
s_lsa_m = np.load("s_svd_lsa_m.npy")
#U_lsa_m = np.load("U_svd_lsa_m.npy")
V_lsa_m = np.load("V_svd_lsa_m.npy")
print("Chargement des données complété !\n")
#print("Normalisation des données ... ")
#scaler = pp.StandardScaler()
#X_sc = scaler.fit_transform( X)
#print("Normalisation des données terminée \n")
#Xs = sparse.csr_matrix(X)
#indicx = np.arange(X.shape[0])
mask_cspp = y.astype(bool) # selection des csp+

#==============================================================================
# Number of meaningfull topics and words (energy based)
#==============================================================================
energy = 0.05 # 99% of square norm
topics_p = tops.meaning_top(s_lsa_p,energy)
topics_m = tops.meaning_top(s_lsa_m,energy)
t = 0.01 # 99% of square norm
av_p = tops.meaning_word(V_lsa_p[:topics_p], t)
av_m = tops.meaning_word(V_lsa_m[:topics_m], t)

ax_p = np.arange(topics_p)
ax_m = np.arange(topics_m)
plt.figure(figsize=(10, 5))
lw = 2
plt.plot(ax_p, av_p, 'k', color='turquoise', lw = lw,
         label='csp+ LSA-reduced')
plt.plot(ax_p, np.mean(av_p)*np.ones((topics_p, 1 )) , 'k--', color='turquoise', lw = lw+1,
         label='csp+ LSA-reduced mean')
plt.plot(ax_m, av_m, 'k',  color='darkorange',  lw=lw,
         label='csp- LSA-reduced')
plt.plot(ax_m, np.mean(av_m)*np.ones((topics_m,1 )), 'k--', color='darkorange', lw=lw+1,
         label='csp- LSA-reduced mean')                
plt.xlabel('Ranked explicative concepts/topics')
plt.ylabel('Number of meaningful words')
plt.xlim(0, max(ax_p[-1],ax_m[-1]) )
#plt.ylim(0.4, 0.6)
plt.grid()
plt.title('Meaningful words along meaningful topics for {} of square norm'.format( 1-t))
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()

words = ceil (max(  np.mean(av_p), np.mean(av_m) ) ) 
vocab_p = tops.vocabulaire(V_lsa_p[:topics_p], words)
vocab_m = tops.vocabulaire(V_lsa_m[:topics_m], words)
vocab = np.unique ( np.concatenate( ( vocab_p.T, vocab_m.T )) ) # vecteur colonne

print("Words caracteristic of csp+ {}:".format( np.setdiff1d(vocab_p,vocab_m)) )
print("Words caracteristic of csp- {}:".format( np.setdiff1d(vocab_m,vocab_p)) )

V_dic = np.concatenate( (V_lsa_p[:topics_p,vocab] ,V_lsa_m[:topics_m,vocab]) , axis = 0)
np.save('V_dic_lsa2.npy', V_dic)
np.save('vocab_dic_lsa2.npy', vocab)
#==============================================================================
# We can choose : 
#       #topics = 61 csp+ / 49 csp- = 110 topics
#       #vocabulary is of size 124
# Exemple de caractéristiques pour energy = 0.05 et t=0.1
# Caracteristic words for csp+ : 98   706   731  1762  2000  2179  2228  2485  2753  2945  3375  4250
#  4463  4481  4858  5166  5330  6748  7968  9705  9863 10184 10750 10796
# 11506
# Caracteristic words for csp- : 413  1277  1708  3805  4233  4912  5368  5519  6280  7134  7764  8786
#  8788  9201  9342  9515  9566  9983 10417 12107 12450 13081
#==============================================================================

#==============================================================================
# We can choose (2): 
#       #topics = 1107 csp+ / 1228 csp- = 2335 topics
#       #vocabulary is of size 7712
# Exemple de caractéristiques pour energy = 0.01 et t=0.01
# Caracteristic words for csp+ : (...)
# Caracteristic words for csp- : (...)
#==============================================================================

#==============================================================================
# How to construct new vectors :
#       if X is containing observations on features (ordered as they are now...)
#       then X_reduct = np.dot( tf-idf.fit(X[::, vocab]).toarray() , V_dic.T ) cf data-generation for syntax
#       Notice that X_reduct is mimiting tf-idf(X) and not X itself
#==============================================================================
