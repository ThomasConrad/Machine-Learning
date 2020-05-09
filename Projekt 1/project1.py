import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


data = np.loadtxt("prostate.data",delimiter="\t",skiprows=1,usecols=[1,2,3,4,5,6,7,8,9])

m = np.mean(data,axis=0)
sd = np.std(data,axis=0)

norm_data = (data-m)/sd
X = (data-m)/sd


U, s, Vt = linalg.svd(norm_data)
V = np.transpose(Vt)

#test = np.diag(s)
#test = np.concatenate((test,np.zeros((88,9))),axis=0)

test = np.concatenate((np.diag(s),np.zeros((88,9))),axis=0)

#print(norm_data[0])
#print((U@test@Vt)[0])

V2 = V[:,0:2]

B2 = X@V2

V3 = V[:,0:3]

B3 = X@V3

#W = np.trace(np.transpose(X)@X)/np.trace(np.transpose(norm_data)@norm_data)
W = s**2/np.sum(s**2)

plt.figure(1)

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

a1 = plt.subplot(121)
plt.scatter(np.arange(1,10),W,c='r',edgecolors='b')
plt.plot(np.arange(1,10),W,'b-')
a1.set_ylabel('Fraction of variance explained', fontdict=font)
a1.set_xlabel('Principal direction', fontdict=font)
plt.xticks(np.arange(1, 10, 1))
plt.grid()

a2 = plt.subplot(122)
plt.plot(np.arange(1,10),np.cumsum(W),'b-')
plt.scatter(np.arange(1,10),np.cumsum(W),c='r',edgecolors='b')
a2.set_ylabel('Fraction of variance explained', fontdict=font)
a2.set_xlabel('Principal directions', fontdict=font)
plt.xticks(np.arange(1, 10, 1))
plt.grid()

plt.figure(2)
plt.scatter(B2[:,1],B2[:,0],edgecolors='b')
plt.ylabel('Data projection onto $v_1$', fontdict=font)
plt.xlabel('Data projection onto $v_2$', fontdict=font)

plt.figure(3)
ax = plt.axes(projection="3d")
ax.scatter3D(B3[:,0], B3[:,1], B3[:,2], c=B3[:,2],edgecolors='b');

plt.figure(4)
p1 = plt.subplot(231)
plt.hist(data[:,0],bins=12)
p1.set_title("lcavol", fontdict=font)
p2 = plt.subplot(232)
plt.hist(data[:,1],bins=12)
p2.set_title("lweight", fontdict=font)
p3 = plt.subplot(233)
plt.hist(data[:,2],bins=12)
p3.set_title("age", fontdict=font)
p4 = plt.subplot(234)
plt.hist(data[:,3],bins=12)
p4.set_title("lbph", fontdict=font)
p5 = plt.subplot(235)
plt.hist(data[:,7],bins=12)
p5.set_title("lcp", fontdict=font)
p6 = plt.subplot(236)
plt.hist(data[:,6],bins=3)
p6.set_title("lpsa", fontdict=font)



plt.figure(5)
ax = plt.axes(projection="3d")
ax.scatter3D(data[:,1], data[:,2], data[:,3], c=data[:,3],edgecolors='b');


plt.figure(6)
plt.scatter(data[:,2],np.exp(data[:,3]),c='r',edgecolors='b')

a= np.corrcoef(data.T)

plt.figure(7)
labels = 'Present', 'Not present'
sizes = [21,76]
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


plt.figure(9)

plt.boxplot(data)

plt.show()







