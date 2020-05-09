# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:36:56 2019

@author: thoma
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import gaussian_kde
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

################ SIMILARITY SHIT
import numpy as np
from scipy.stats import zscore


def similarity(X, Y, method):
    '''
    SIMILARITY Computes similarity matrices

    Usage:
        sim = similarity(X, Y, method)

    Input:
    X   N1 x M matrix
    Y   N2 x M matrix 
    method   string defining one of the following similarity measure
           'SMC', 'smc'             : Simple Matching Coefficient
           'Jaccard', 'jac'         : Jaccard coefficient 
           'ExtendedJaccard', 'ext' : The Extended Jaccard coefficient
           'Cosine', 'cos'          : Cosine Similarity
           'Correlation', 'cor'     : Correlation coefficient

    Output:
    sim Estimated similarity matrix between X and Y
        If input is not binary, SMC and Jaccard will make each
        attribute binary according to x>median(x)

    Copyright, Morten Morup and Mikkel N. Schmidt
    Technical University of Denmark '''

    X = np.mat(X)
    Y = np.mat(Y)
    N1, M = np.shape(X)
    N2, M = np.shape(Y)
    
    method = method[:3].lower()
    if method=='smc': # SMC
        X,Y = binarize(X,Y);
        sim = ((X*Y.T)+((1-X)*(1-Y).T))/M
    elif method=='jac': # Jaccard
        X,Y = binarize(X,Y);
        sim = (X*Y.T)/(M-(1-X)*(1-Y).T)        
    elif method=='ext': # Extended Jaccard
        XYt = X*Y.T
        sim = XYt / (np.log( np.exp(sum(np.power(X.T,2))).T * np.exp(sum(np.power(Y.T,2))) ) - XYt)
    elif method=='cos': # Cosine
        sim = (X*Y.T)/(np.sqrt(sum(np.power(X.T,2))).T * np.sqrt(sum(np.power(Y.T,2))))
    elif method=='cor': # Correlation
        X_ = zscore(X,axis=1,ddof=1)
        Y_ = zscore(Y,axis=1,ddof=1)
        sim = (X_*Y_.T)/(M-1)
    return sim
        
def binarize(X,Y=None):
    ''' Force binary representation of the matrix, according to X>median(X) '''
    x_was_transposed = False
    if Y is None:
        if X.shape[0] == 1:
            x_was_transposed = True
            X = X.T;
        
        Xmedians = np.ones((np.shape(X)[0],1)) * np.mean(X,0)
        Xflags = X>Xmedians
        X[Xflags] = 1; X[~Xflags] = 0

        if x_was_transposed:
            return X.T
        return X
    else:
        #X = np.matrix(X); Y = np.matrix(Y);
        #XYmedian= np.median(np.bmat('X; Y'),0)
        #Xmedians = np.ones((np.shape(X)[0],1)) * XYmedian
        #Xflags = X>Xmedians
        #X[Xflags] = 1; X[~Xflags] = 0
        #Ymedians = np.ones((np.shape(Y)[0],1)) * XYmedian
        #Yflags = Y>Ymedians
        #Y[Yflags] = 1; Y[~Yflags] = 0
        return [binarize(X,None),binarize(Y,None)]
        

## Example
#import numpy as np
#from similarity import binarize2
#A = np.asarray([[1,2,3,4,5],[6,7,8,9,10],[1,2,3,4,5],[6,7,8,9,10]]).T
#binarize2(A,['a','b','c','d'])
def binarize2(X,columnnames):
    X = np.concatenate((binarize(X),1-binarize(X)),axis=1)

    new_column_names = []
    [new_column_names.append(elm) for elm in [name+' 50th-100th percentile' for name in columnnames]]
    [new_column_names.append(elm) for elm in [name+' 0th-50th percentile' for name in columnnames]]

    return X, new_column_names




############





font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

max_display_levels = 20
data = np.loadtxt("prostate.data",delimiter="\t",skiprows=1,usecols=[1,2,3,4,5,6,7,8,9])

m = np.mean(data,axis=0)
sd = np.std(data,axis=0)

norm_data = (data-m)/sd
X = (data-m)/sd
X = np.delete(X,4,1)

truelabel = data[:,4]

###### TREES ######
"""

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
'''


###### KDE ######
'''
widths = [0.1,0.5,1,2,4]
scores = []

for width in widths: #not efficient LOO, but it's short and works
    score = 0
    for k in range(97):
        x_train = np.delete(X,k,axis=0) #delete i'th row
        x_test = X[k]

        KDE = gaussian_kde(x_train.T,bw_method=width) #tranposed because expects data as matrix of size (features, datapoints)
        score += np.log(KDE.evaluate(x_test))
    score = score/97
    scores.append(score)
print(scores)
#plt.plot(widths,scores)
#plt.show()


KDE = gaussian_kde(X.T,bw_method=1)

logprop = np.zeros(97)
for n, i in enumerate(X):
    logprop[n] = np.log(KDE.evaluate(i))
i = np.arange(0,97)
logprop = np.array(logprop).squeeze()
i = logprop.argsort()
logprop = logprop[i]

plt.plot(range(20),logprop[:20])
plt.title('Log probability of KDE, lowest 20')
plt.show()

###### KNN & ARD ######
K=5
KNN = NearestNeighbors(n_neighbors=K).fit(X)
D, i = KNN.kneighbors(X)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i = density.argsort()
density = density[i]

plt.plot(range(20),density[:20])
plt.title('KNN density, lowest 20')
plt.show()

KNN = NearestNeighbors(n_neighbors=K).fit(X)
D, i = KNN.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avgreldensity = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avgrel = avgreldensity.argsort()
avgreldensity = avgreldensity[i_avgrel]

plt.plot(range(20),avgreldensity[:20])
plt.title('KNN ARD density, lowest 20')
plt.show()
'''

###### Association Learning ######

#Binarizing data and apriori algorigm
from apyori import apriori

def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T
def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules

labels = ["lcavol","lweight","age","lbph","svi","lcp","gleason","pgg45","lpsa"]

binary_data, binary_names = binarize2(norm_data,labels)



T = mat2transactions(binary_data,labels=binary_names)
rules = apriori(T, min_support=0.5, min_confidence=0)
print_apriori_rules(rules)





