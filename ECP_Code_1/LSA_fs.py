# -*- coding: utf-8 -*-
"""
@author: plkovac & Jiayi
Octobre 2016
LSA decomposition for visualization (done on each label population. For analysis, change to SVD on total population = unsupervised)
"""

from sklearn.feature_extraction.text import TfidfTransformer
from numpy.linalg import norm
import numpy as np
from functions import get_dataset as start
from functions import SVD as sv
from functions import radar as rad
import matplotlib.pyplot as plt
from math import ceil
import time

if __name__ == '__main__':
    wdir='path_to_dataset'
    first = False # True = SVD is not in memory and needs to be done / False = SVD is in memory (.npz)
    im=0
    print("Le chargement des données commence ...\n")
    X, y, l = start.get_dataset()
    print("Chargement des données complété !\n")
    mask_cspp = y.astype(bool)

    print("Démarrage de tf-idf \n")
    start = time.time()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X).toarray()
    end = time.time()
    print("Fin de tf-idf ... \n")
    print("temps de tf-idf sur les données : {}".format( end - start) )

    if first :
        print("Démarrage de SVD ... \n")
        start = time.time()
        U_lsa_p, s_lsa_p, V_lsa_p = sv.svd_set(tfidf[mask_cspp,::], string="lsa_p")
        U_lsa_m, s_lsa_m, V_lsa_m = sv.svd_set(tfidf[~mask_cspp,::], string="lsa_m")
        end = time.time()
        print("Fin de tf-idf ... \n")
        print("temps des SVD sur les données : {}".format( end - start) )

    "Chargement des données "

    s_lsa_p = np.load("s_svd_lsa_p.npy")
    U_lsa_p = np.load("U_svd_lsa_p.npy")
    V_lsa_p = np.load("V_svd_lsa_p.npy")
    s_lsa_m = np.load("s_svd_lsa_m.npy")
    U_lsa_m = np.load("U_svd_lsa_m.npy")
    V_lsa_m = np.load("V_svd_lsa_m.npy")

    " Plot des graphes et images "
    # Radar charts
    pl_1 = False
    if pl_1:
        threshold = 500
        plt.figure(im)
        im+1
        plt.plot(s_lsa_p[:threshold], 'k--', label=' csp+')
        plt.plot(s_lsa_m[:threshold], 'k:', label=' csp-')
        legend = plt.legend(loc='upper center', shadow=False)
        legend.get_frame().set_facecolor('#00FFCC')
        plt.xlabel('sv rank')
        plt.ylabel('sv values')
        plt.title('Supervised model - Latent Semantic Analysis')
        plt.show()

        sglval = 8 #nbre de singular values d'intérêt
        feat = 20 #rang des features les plus importantes
        explicative_features_lsap = sv.svd_naive_cluster(V_lsa_p, sglval, feat)
        explicative_features_lsam = sv.svd_naive_cluster(V_lsa_m, sglval, feat)

        # Radar chart pour chaque singular value (-> premiers clusters)
        V_pond_lsap = sv.svd_to_plot(V_lsa_p, explicative_features_lsap)
        V_pond_lsam = sv.svd_to_plot(V_lsa_m, explicative_features_lsam)
        rad.radar_sglv(V_pond_lsap, explicative_features_lsap, "SVD Supervised model - LSA : features clustering csp+")
        rad.radar_sglv(V_pond_lsam, explicative_features_lsam, "SVD Supervised model - LSA : features clustering csp-")
        im= im+2*ceil(sglval/4)+1

        t =0.05 #taux de conservation des features importantes (essayer bas)
        nn = 200 #étude sur les nn premiers clusters (pour vérifier la pertinence)
        Idelsap, Scorelsap, llsap = sv.clust_ress_fixin(V_lsa_p,V_lsa_m,t,nn,nn,0)
        sv.clust_pl1(V_lsa_p,V_lsa_m,t,nn,nn,0," Relation csp+/scp- : LSA ")

    # energy-based graphs
    pl_2 = False
    if pl_2:
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
