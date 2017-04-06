# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 08:08:55 2016

@author: paulvernhet
"""

"First Observations "

wdir='/Users/paulvernhet/Desktop/3A/SEMINAIRE 1000mercis/week_1'
import numpy as np
from functions import get_dataset as start
from functions import SVD as sv
from functions import radar as rad
from functions import easy_stats as es
from functions import data_viz as dv
import matplotlib.pyplot as plt
from sklearn import manifold
from time import time
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

"Chargement des données "
print("Le chargement des données commence ...\n")
X, y, l = start.get_dataset()
mask_cspp = y.astype(bool) # selection des csp+
im=0
print("Chargement des données complété !\n")

print("Chargement des données en cours ...")
s_lsa = np.load("s_svd_lsa.npy")
U_lsa = np.load("U_svd_lsa.npy")
V_lsa = np.load("V_svd_lsa.npy")
#s_lsa_p = np.load("s_svd_lsa_p.npy")
#U_lsa_p = np.load("U_svd_lsa_p.npy")
#V_lsa_p = np.load("V_svd_lsa_p.npy")
#s_lsa_m = np.load("s_svd_lsa_m.npy")
#U_lsa_m = np.load("U_svd_lsa_m.npy")
#V_lsa_m = np.load("V_svd_lsa_m.npy")
print("LSA - done")
#s_r = np.load("s_svd_r.npy")
#U_r = np.load("U_svd_r.npy")
#V_r = np.load("V_svd_r.npy")
#s_r_p = np.load("s_svd_r_p.npy")
#U_r_p = np.load("U_svd_r_p.npy")
#V_r_p = np.load("V_svd_r_p.npy")
#s_r_m = np.load("s_svd_r_m.npy")
#U_r_m = np.load("U_svd_r_m.npy")
#V_r_m = np.load("V_svd_r_m.npy")
#print("SVD - done")
#print("Chargement de SVD et LSA terminés")

"Graphe visites de sites webs selon csp+ / csp- "
Xb=X.astype(bool)
Fp = Xb[mask_cspp,::]
Fm = Xb[~mask_cspp,::]
f_cspp = np.sum(Fp, axis = 0)  # vecteur des features visités par cookies + (nombre de cookies différents)
f_cspm = np.sum(Fm, axis = 0)  # features visités par cookies - (nombre de cookies différents)
features_visited_cspp = np.argsort(f_cspp)[::-1]
features_visited_cspm = np.argsort(f_cspm)[::-1] 

n = 30 #n premiers sites les plus visités par chaque population
x = np.arange(n)
Y1 = f_cspp[features_visited_cspp[x]]
Y2 = f_cspm[features_visited_cspm[x]]
plt.figure(im)
im=im+1
plt.bar(x, Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(x, -Y2, facecolor='#ff9999', edgecolor='white')
for a,b in zip(x,Y1):
    plt.text(a+0.4, b+0.05, '%.2d' % features_visited_cspp[a], ha='center', va= 'bottom')
for a,b in zip(x,Y2):
    plt.text(a+0.4, -b-0.2, '%.2d' % features_visited_cspm[a], ha='center', va= 'bottom')
plt.show()

es.mstat(Xb)

dv.most_visited(X, y, 30, 0)

jcut = 50
n_points = 100
X_red = np.dot(U_lsa[::,:jcut], np.diag(s_lsa[:jcut]))
dv.comp_data_viz_2D(X_red, mask_cspp,  n_points, 10)

"SVD and LSA"
"Decaying values "
threshold = 1000
plt.figure(im)
im+1
plt.plot(s_lsa[:threshold], 'k', label=' global')
#plt.plot(s_lsa_p[:threshold], 'k--', label=' csp+')
#plt.plot(s_lsa_m[:threshold], 'k:', label=' csp-')
legend = plt.legend(loc='upper center', shadow=False)
legend.get_frame().set_facecolor('#00FFCC')
plt.xlabel('sv rank')
plt.ylabel('sv values')
plt.title('Decaying of singular values for Latent Semantic Analysis')
plt.show()

#plt.figure(im)
#im+1
#plt.plot(s_r[:threshold], 'k', label=' global')
#plt.plot(s_r_p[:threshold], 'k--', label=' csp+')
#plt.plot(s_r_m[:threshold], 'k:', label=' csp-')
#legend = plt.legend(loc='upper center', shadow=False)
#legend.get_frame().set_facecolor('#00FFCC')
#plt.xlabel('sv rank')
#plt.ylabel('sv values')
#plt.title('Decaying of singular values for Singular Value Decomposition')
#plt.show()

"Raw projection comparison : 2 first componants "
Xc_lsa = np.dot(U_lsa[::,:3],np.diag(s_lsa[:3]))
#Xc_r = np.dot(U_r[::,:2],np.diag(s_r[:2]))
fig = plt.figure(im)
im=im+1
ax = fig.gca()
plt.scatter(Xc_lsa[mask_cspp,0],Xc_lsa[mask_cspp,1],color="red")
plt.scatter(Xc_lsa[~mask_cspp,0],Xc_lsa[~mask_cspp,1],color="blue")
plt.xlabel('First concept')
plt.ylabel('Second concept')
ax.set_title("Décomposition LSA", weight='bold', size='medium', position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')
#fig = plt.figure(im)
#im=im+1
#ax = fig.gca()
#plt.scatter(Xc_r[mask_cspp,0],Xc_r[mask_cspp,1],color="red")
#plt.scatter(Xc_r[~mask_cspp,0],Xc_r[~mask_cspp,1],color="blue")
#plt.xlabel('First concept')
#plt.ylabel('Second concept')
#ax.set_title("Décomposition SVD", weight='bold', size='medium', position=(0.5, 1.1),
#                         horizontalalignment='center', verticalalignment='center')

fig = plt.figure(im) ## Pour les sites les plus visités par exemple !!
im=im+1
ax = fig.gca()
plt.scatter(X[mask_cspp,6526],X[mask_cspp,2918],color="red")
plt.scatter(X[~mask_cspp,6526],X[~mask_cspp,2918],color="blue")
plt.xlabel('feature 6526')
plt.ylabel('feature 2918')
ax.set_title("Décomposition Ssur premiers features", weight='bold', size='medium', position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(Xc_lsa[mask_cspp,0], Xc_lsa[mask_cspp,1], Xc_lsa[mask_cspp,2], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(Xc_lsa[~mask_cspp,0], Xc_lsa[~mask_cspp,1], Xc_lsa[~mask_cspp,2], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('LSA for 3 first components')
ax.legend(loc='upper right')

plt.show()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(X[mask_cspp,6526],X[mask_cspp,2918], X[mask_cspp,4309], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(X[~mask_cspp,6526],X[~mask_cspp,2918], X[~mask_cspp,4309], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Raw for 3 most viewed websites')
ax.legend(loc='upper right')

plt.show()





"Using IsoMap/t-SNE/ another mapping on reduced matrixes ???"
"Isomaps pour 50 et 200 features + 100 pts ?"
t0 = time.time()
Y = manifold.Isomap(10, 2).fit_transform(X[:1000,::])
t1 = time.time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[mask_cspp[:1000], 0], Y[mask_cspp[:1000], 1], c='red', cmap=plt.cm.Spectral)
plt.scatter(Y[~mask_cspp[:1000], 0], Y[~mask_cspp[:1000], 1], c='blue', cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))

#Xrr = np.dot( U_lsa[:500,:1000], np.dot( np.diag(s_lsa[:1000]), V_lsa[:1000,:1000] ) )
Xrr = np.dot( U_lsa[:1000,:5000], np.diag(s_lsa[:5000]) )
t0 = time.time()
Y2 = manifold.Isomap(10, 2).fit_transform(Xrr)
t1 = time.time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y2[mask_cspp[:1000], 0], Y2[mask_cspp[:1000], 1], c='red', cmap=plt.cm.Spectral)
plt.scatter(Y2[~mask_cspp[:1000], 0], Y2[~mask_cspp[:1000], 1], c='blue', cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))

"Radar-charts for concept features"
sglval = 4 #n first concept features (singular vectors)
feat = 15 #n most represented real features (=websites)
ef_lsa = sv.svd_naive_cluster(V_lsa, sglval, feat) #explicative features for global LSA 
#ef_lsap = sv.svd_naive_cluster(V_lsa_p, sglval, feat) #explicative features for csp+ LSA
#ef_lsam = sv.svd_naive_cluster(V_lsa_m, sglval, feat) #explicative features for csp- LSA
#ef_r = sv.svd_naive_cluster(V_r, sglval, feat) #explicative features for global SVD
#ef_rp = sv.svd_naive_cluster(V_r_p, sglval, feat) #explicative features for csp+ SVD
#ef_rm = sv.svd_naive_cluster(V_r_m, sglval, feat) #explicative features for csp- SVD
VP_lsa = sv.svd_to_plot(V_lsa, ef_lsa) #Norm for max feature
#VP_lsap = sv.svd_to_plot(V_lsa_p, ef_lsap)
#VP_lsam = sv.svd_to_plot(V_lsa_m, ef_lsam)
#VP_r = sv.svd_to_plot(V_r, ef_r)
#VP_rp = sv.svd_to_plot(V_r_p, ef_rp)
#VP_rm = sv.svd_to_plot(V_r_m, ef_rm)
rad.radar_sglv(VP_lsa, ef_lsa, "Global LSA : concept clusters")
#rad.radar_sglv(VP_lsap, ef_lsap, "csp+ LSA : concept clusters")
#rad.radar_sglv(VP_lsam, ef_lsam, "csp- LSA : concept clusters")
#rad.radar_sglv(VP_r, ef_r, "Global SVD : concept clusters")
#rad.radar_sglv(VP_rp, ef_rp, "csp+ SVD : concept clusters")
#rad.radar_sglv(VP_rm, ef_rm, "csp- SVD : concept clusters")

" A measure for dissimilarities between populations "

"NB: pas le même nombre de concepts entre csp+ et csp- --> détection de la différence ???"
