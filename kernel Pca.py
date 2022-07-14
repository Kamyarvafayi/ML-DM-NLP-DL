import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sk
def circle(r,size):
    points = np.zeros([size,2])
    counter = 0
    while counter<size:
        x = np.random.uniform(-r,r)
        y = np.random.uniform(-r,r)
        z = np.random.uniform(-r,r)
        if x**2 + y**2 < r**2 and x**2 + y**2  > (r**2/1.2):
            points[counter,0]=x
            points[counter,1]=y
            counter += 1
    return points
points = circle(5,300)
points2= circle(10,300)
point = circle(15,300)
finalpoints = np.append(points,points2,axis=0)
finalpoints3 = np.append(finalpoints,points,axis=0)
plt.figure(figsize=[5,5])
plt.scatter(points2[:,0],points2[:,1])
plt.scatter(points[:,0],points[:,1])
plt.scatter(point[:,0],point[:,1])
# In[]:
moons = sk.make_moons(1000,noise=0)
plt.scatter(moons[0][moons[1]==0,0],moons[0][moons[1]==0,1],color='r')
plt.scatter(moons[0][moons[1]==1,0],moons[0][moons[1]==1,1],color='b')
blobs = sk.make_blobs(1000, centers = 3)
plt.scatter(blobs[0][blobs[1]==0,0],blobs[0][blobs[1]==0,1],color='r')
plt.scatter(blobs[0][blobs[1]==1,0],blobs[0][blobs[1]==1,1],color='b')
plt.scatter(blobs[0][blobs[1]==2,0],blobs[0][blobs[1]==2,1],color='y')
circles = sk.make_circles(1000, noise=0.2, factor = 0.3)
plt.scatter(circles[0][circles[1]==0,0],circles[0][circles[1]==0,1],color='r')
plt.scatter(circles[0][circles[1]==1,0],circles[0][circles[1]==1,1],color='b')
# In[]:
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
# In[]:
k5 = guassian(moons[0],10)
k5 = guassian(circles[0],10)
k5 = poly(moons[0],3,2)
k5 = poly(circles[0],3,1)
k5 = sigmoid(circles[0])
# k5 = sigmoid(moons[0])
oneN = 1/1000* np.ones([1000,1000])
oneN2 = 1/900* np.ones([900,900])
centeredK = k5-oneN.dot(k5)-k5.dot(oneN)+oneN.dot(k5).dot(oneN)
eigval, eigvec =np.linalg.eig(centeredK)
finaldata = [eigvec[:,i].reshape(1,1000).dot(centeredK) for i in range(2)]
eigg = eigvec[:,0]
plt.figure()
plt.scatter(eigvec[moons[1]==0,0],eigvec[moons[1]==0,1], color='r')
plt.scatter(eigvec[moons[1]==1,0],eigvec[moons[1]==1,1], color='b')
plt.figure()
plt.scatter(eigvec[circles[1]==0,0],eigvec[circles[1]==0,1], color='r')
plt.scatter(eigvec[circles[1]==1,0],eigvec[circles[1]==1,1], color='b')
from sklearn.decomposition import PCA, KernelPCA
Kpca = KernelPCA(n_components=2,kernel="rbf",gamma=10)
Kpca = KernelPCA(n_components=2,kernel="sigmoid",degree=3)
#Kpca = KernelPCA(n_components=2,kernel="sigmoid")
#Kpca = Kpca.fit_transform(moons[0]) 
Kpca = Kpca.fit_transform(circles[0]) 
plt.figure()
plt.scatter(Kpca[circles[1]==0,0],Kpca[circles[1]==0,1], color='r')
plt.scatter(Kpca[circles[1]==1,0],Kpca[circles[1]==1,1], color='b')
# In[]:
from sklearn.cluster import k_means
# In[]:
Kmeans = k_means(Kpca, 2)
plt.figure()
plt.scatter(moons[0][Kmeans[1]==0,0],moons[0][Kmeans[1]==0,1], color='r',marker='x')
plt.scatter(moons[0][Kmeans[1]==1,0],moons[0][Kmeans[1]==1,1], color='b',marker='x')

plt.scatter(circles[0][Kmeans[1]==0,0],circles[0][Kmeans[1]==0,1], color='r',marker='x')
plt.scatter(circles[0][Kmeans[1]==1,0],circles[0][Kmeans[1]==1,1], color='b',marker='x')

plt.scatter(moons[0][moons[1]==0,0],moons[0][moons[1]==0,1], color='r',marker='o')
plt.scatter(moons[0][moons[1]==1,0],moons[0][moons[1]==1,1], color='b',marker='o')

plt.scatter(Kpca[Kmeans[1]==0,0],Kpca[Kmeans[1]==0,1], color='r',marker='s')
plt.scatter(Kpca[Kmeans[1]==1,0],Kpca[Kmeans[1]==1,1], color='b',marker='s')
