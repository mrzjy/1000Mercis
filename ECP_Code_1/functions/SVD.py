# -*- coding: utf-8 -*-
"""
@author: pPaul Vernhet & Zhang Jiayi
October
SVD plots and selection of energy
"""

import numpy as np
import matplotlib.pyplot as plt

def svd_set(data, string="00"):

    #calculate SVD
    U, s, V = np.linalg.svd(data, full_matrices=False )

    np.save("U_svd_"+string,U) #attention aux réécritures !!!
    np.save("s_svd_"+string,s)
    np.save("V_svd_"+string,V)

    return U, s, V

def svd_reduce(U,s, n=2):
    return np.dot(U[::,:n], np.diag(s[:n]) )

def svd_naive_cluster(V, sglval=100, feat=50):
    nrow ,_ = V.shape
    index = min(sglval, nrow)
    explicative_features = np.zeros((index,feat))
    for k in range(index):
        explicative_features[k,::] = np.argsort(abs(V[k,::]))[::-1][:feat]
    return explicative_features.astype(int)

def svd_to_plot(V, explicative_features):
    a,b = explicative_features.shape
    V_pond = np.zeros((a,b))
    count=0;
    for row in explicative_features:
        V_pond[count,::] = abs(V[count,row])/abs(V[count,row[0]])
        count=count+1
    return V_pond

def svd_thresh(line,t):
    linea = abs(line)
    iord = np.argsort(linea)[::-1]
    n_max = linea[iord[0]] #ou alors iord[0,0], ou autre format ...
    arret = np.sum( (linea/n_max > t).astype(int) )
    print("Nombre de variables d'intéret de ligne {}:".format(arret))
    return iord[:arret] #répéter le même format que prcédemment

def score_calc(a,b):
    liste = np.setdiff1d(a,b)
    return liste, 1-len(liste)/len(a)

def clust_ress_E(a, B, t):
    c = 0
    score = 0
    b = svd_thresh(a,t)
    for s in B:
        u = svd_thresh(s,t)
        if (score < score_calc(b,u)[1]):
            liste, score = score_calc(b,u)
            ide = c
        c=c+1
    return ide, score, liste


def clust_ress(A,B,t):
    n,_ = A.shape
    Ide= np.zeros((1,n))
    Score= np.zeros((1,n))
    Liste = list()
    ite=0
    for rowA in A:
        Ide[::,ite], Score[::,ite], l = clust_ress_E(rowA, B, t)
        Liste.append(l)
        ite=ite+1

    return Ide, Score, Liste

def clust_ress_fixin(A,B,t, sga=20, sgb=20, meth=0):
    A = A[:sga,::]
    B = B[:sgb,::]
    n,m = A.shape
    Ide= np.zeros((1,n))
    Score= np.zeros((1,n))
    Liste = list()
    ite=0
    for rowA in A:

        c = 0
        score = 0
        l =0

        linea = abs(rowA)
        iord = np.argsort(linea)[::-1]
        n_max = linea[iord[0]]
        arret = np.sum( (linea/n_max > t).astype(int) )
        b = iord[:arret]

        for s in B:

            lineb = abs(s)
            iord2 = np.argsort(lineb)[::-1]
            n_max2 = lineb[iord2[0]]
            arret2 = np.sum( (lineb/n_max2 > t).astype(int) )
            u = iord[:arret2]

            listeC = np.setdiff1d(b,u)
            if meth ==1 :
                scoreC = (len(b)-len(listeC))/(len(b)*len(u)) #score pénalisant L1
            elif meth ==0 :
                scoreC = (len(b)-len(listeC))**2/(len(b)*len(u)) #score pénalisant L2
            else :
                scoreC = (len(b)-len(listeC))/len(b) #score naïf

            if (score < scoreC):
                score = scoreC
                l=listeC
                ide = c
            c=c+1

        Ide[0,ite] = ide
        Score[0,ite] = score
        Liste.append(l)
        ite=ite+1

    return Ide, Score, Liste

def clust_pl1(A,B,t,sga=20, sgb=20, meth=0, title="ø"):

    Ide, Score, _ = clust_ress_fixin(A,B,t,sga, sgb, meth)
    _,n = Ide.shape
    x = np.arange(n)
    #area = 0.1*np.pi*(1/Score)**2
    area = 10*np.pi*np.ones((1,n))
    colors = np.random.rand(n)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0,n+1,1))
    #ax.set_yticks(np.arange(0,10,1))

    plt.scatter(x,Ide , s=area, c=colors, alpha=0.5)
    plt.grid(True)
    ax.set_title(title + " pénalité " +str(meth), weight='bold', size='medium', position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')

    plt.show()
