# In[]:
import sklearn.datasets as sk
import numpy as np
import matplotlib.pyplot as plt

moonsdata , moonsclass = sk.make_moons(1000)
moonsdata , moonsclass = sk.make_circles(1000)
# In[]:
def euclideandist(X):
    shape=np.shape(X)
    N = shape[0]
    k = np.zeros([N,N])
    for i in range(N):
        for j in  range(N):
            k[i,j] = np.linalg.norm(X[i,:]-X[j,:])       
    return k
def guassian(x,gamma):
    shap=np.shape(x)
    N = shap[0]
    k = np.zeros([N,N])
    for i in range(N):
        for j in  range(N):
            k[i,j] = np.exp(-(gamma)*np.linalg.norm(x[i,:]-x[j,:])**2)       
    return k
def poly(x, degree, c):
    shap=np.shape(x)
    N = shap[0]
    k = np.zeros([N,N])
    for i in range(N):
        for j in  range(N):
            k[i,j] = (x[i].dot(x[j])+c)**degree      
    return k
def sigmoid(x):
    shap=np.shape(x)
    N = shap[0]
    k = np.zeros([N,N])
    for i in range(N):
        for j in  range(N):
            k[i,j] = 1/(1+np.exp(-1*np.linalg.norm(x[i,:]-x[j,:])))     
    return k 
# In[]: A and D and L
A = euclideandist(moonsdata)
A = guassian(moonsdata,10)
A = guassian(moonsdata,10)
A_Sum = np.sum(A, axis=1)
A_Sum_shape = np.shape(A_Sum)
D = HalfD = HalfDminus = np.zeros([A_Sum_shape[0],A_Sum_shape[0]]) 
for i in range(A_Sum_shape[0]):
                D[i,i] = A_Sum[i]
                HalfD[i,i] = np.sqrt(A_Sum[i])
                HalfDminus[i,i] = A_Sum[i]**(-0.5) 
L = D - A
# In[]: Normalized L
Lnorm = np.identity(A_Sum_shape[0]) - HalfDminus.dot(A).dot(HalfD)
# In[] : eigvec and eigval
eigval, eigvec =np.linalg.eig(Lnorm)
plt.figure()
plt.scatter(eigvec[moonsclass==0,1],eigvec[moonsclass==0,2], color='r')
plt.scatter(eigvec[moonsclass==1,1],eigvec[moonsclass==1,2], color='b')
# In[]:
from sklearn.cluster import k_means
Kmeans = k_means(eigvec[:,0:2], 2)
plt.figure()
plt.scatter(moonsdata[Kmeans[1]==0,0],moonsdata[Kmeans[1]==0,1], color='r',marker='x')
plt.scatter(moonsdata[Kmeans[1]==1,0],moonsdata[Kmeans[1]==1,1], color='b',marker='x')
plt.scatter(moons[0][moons[1]==0,0],moons[0][moons[1]==0,1], color='r',marker='o')
plt.scatter(moons[0][moons[1]==1,0],moons[0][moons[1]==1,1], color='b',marker='o')
plt.scatter(Kpca[Kmeans[1]==0,0],Kpca[Kmeans[1]==0,1], color='r',marker='s')
plt.scatter(Kpca[Kmeans[1]==1,0],Kpca[Kmeans[1]==1,1], color='b',marker='s')
# In[]:
from sklearn.cluster import spectral_clustering