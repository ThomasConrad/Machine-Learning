# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:36:56 2019

@author: thoma
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import KFold

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

max_display_levels = 200
data = np.loadtxt("prostate.data",delimiter="\t",skiprows=1,usecols=[1,2,3,4,5,6,7,8,9])

m = np.mean(data,axis=0)
sd = np.std(data,axis=0)

norm_data = (data)/sd
X = data/sd
X = np.delete(X,4,1)

truelabel = data[:,4]

###### TREES ######


treesingle = linkage(X, method='single', metric='euclidean')
dendrogram(treesingle, truncate_mode='level', p=max_display_levels)
plt.title("Minimum",fontdict=font)
plt.figure()
treeaverage = linkage(X, method='average', metric='euclidean')
dendrogram(treeaverage, truncate_mode='level', p=max_display_levels)
plt.title("Average",fontdict=font)
plt.figure()
treecomplete = linkage(X, method='complete', metric='euclidean')
dendrogram(treecomplete, truncate_mode='level', p=max_display_levels)
plt.title("Maximum",fontdict=font)
plt.show()


print("rand score for singletree:")
cls = fcluster(treesingle, criterion='maxclust', t=2)
singlelabels = []
for i in range(len(cls)):
    singlelabels.append(cls[i]-1)
print(adjusted_rand_score(singlelabels,truelabel))

print("rand score for averagetree:")
cls = fcluster(treeaverage, criterion='maxclust', t=2)
averagelabels = []
for i in range(len(cls)):
    averagelabels.append(cls[i]-1)
print(adjusted_rand_score(averagelabels,truelabel))

print("rand score for completetree:")
cls = fcluster(treecomplete, criterion='maxclust', t=2)
completelabels = []
for i in range(len(cls)):
    completelabels.append(cls[i]-1)
print(adjusted_rand_score(completelabels,truelabel))
'''

"""
Rand value = 0.05440210949673045
Center1 = [[ 0.41973702  0.13532949  0.14330872  0.05205968  0.66246289  0.47092803
   0.54685503  0.38874262]
Center2 = [-0.54974704 -0.17724667 -0.18769739 -0.06818473 -0.86765522 -0.61679404
  -0.7162388  -0.50915239]]
"""


###### GMM ######

'''
scores = []

K = 5
kf = KFold(n_splits=K)
n = 1
for i in range(1,4):
    sumscore = 0
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]

        
        gmm = GMM(n_components=n, n_init = 4)
        gmm.fit(x_train)
        sumscore += gmm.score(x_test)
    scores.append(sumscore/K)
    n += 1

gmm = GMM(n_components=2, n_init = 4)

gmm.fit(X)
predlabel = gmm.predict(X)

print(adjusted_rand_score(predlabel,truelabel))

print(gmm.means_)
plt.plot(range(1,4),scores)
plt.show()



###### KDE ######


















