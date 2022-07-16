import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cl

# In[]:
iris = sns.load_dataset("iris") 
iris.drop('species',axis=1,inplace=True)
data = np.array(iris[0:10])

# In[]:
kmeans = cl.KMeans(2)
kmeans = kmeans.fit(data)

kmeans3 = cl.KMeans(3)
kmeans3 = kmeans3.fit(data)

Centers1 = np.array([[5.125, 3.6, 1.5 , 0.25],
       [4.69, 3.12, 1.42, 0.2]])

Centers2 = np.array([[4.7, 3.12, 1.42, 0.2],
       [5.03, 3.5, 1.43, 0.2],
       [5.4, 3.9, 1.7, 0.41]])

Membership1 = np.array([[0.01,0.99],[0.99,0.01],[0.99,0.01],[0.99,0.01],[0.01,0.95], [0.01,0.99],[0.99,0.01],[0.01,0.99],[0.99,0.01],[0.99,0.01]])

Membership2 = np.array([[0.01,0.98,0.01],[0.98,0.01,0.01],[0.98,0.01,0.01],[0.98,0.01,0.1],[0.01,0.98,0.01], [0.01,0.01,0.98],[0.98,0.01,0.01],[0.01,0.98,0.01],[0.98, 0.01,0.01],[0.98,0.01,0.01]])

# In[]:
plt.scatter(data[Membership1[:,0]>0.5,0],data[Membership1[:,0]>0.5,1],color = 'b')
plt.scatter(data[Membership1[:,1]>0.5,0],data[Membership1[:,1]>0.5,1],color = 'r')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(["First Cluster", "Second Cluster"])

plt.figure(figsize=[6,5])
plt.scatter(data[Membership1[:,0]>0.5,2],data[Membership1[:,0]>0.5,3],color = 'b')
sct = plt.scatter(data[Membership1[:,1]>0.5,2],data[Membership1[:,1]>0.5,3],color = 'r')
plt.xlabel("x3")
plt.ylabel("x4")
plt.legend(["First Cluster", "Second Cluster"])

Colors = ['b','r',[0.5,1,0.5]]
Legend = ["First Cluster", "FirstCenter", "Second Cluster", "Second Center", "third Cluster", "Third Center"]
plt.figure()
for i in range(3):
    plt.scatter(data[Membership2[:,i]>0.5,0],data[Membership2[:,i]>0.5,1],edgecolors = Colors[i], facecolors='none')
    plt.scatter(Centers2[i,0],Centers2[i,1],marker = 'x', color = Colors[i])
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(Legend)

plt.figure()
for i in range(3):
    plt.scatter(data[Membership2[:,i]>0.5,2],data[Membership2[:,i]>0.5,3],color = Colors[i])
    plt.scatter(Centers2[i,2],Centers2[i,3], marker = 'x', color = Colors[i])
plt.xlabel("x3")
plt.ylabel("x4")
plt.legend(Legend)
# In[]:
def PC(Membership):
    bank = 0
    Shape = np.shape(Membership)
    N = Shape[0]
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            bank = bank + Membership[i,j]**2
    ### OR
    #bank = sum(Membership**2)
    return bank/N

# In[]:
def CE(Membership):
    bank = 0
    Shape = np.shape(Membership)
    N = Shape[0]
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            bank = bank + Membership[i,j]*np.log(Membership[i,j])
    return -bank/N

# In[]:
def xlogx(x):
    return x*np.log(x)

def x2(x):
    return x**2

plt.figure()
plt.plot(np.linspace(0.001,1,num=999),xlogx(np.linspace(0.001,1,num=999)))
plt.plot(np.linspace(0.001,1,num=999),-xlogx(np.linspace(0.001,1,num=999)))
plt.legend(["xlogx","-xlogx"])
plt.figure()
plt.plot(np.linspace(0.01,1,num=999),x2(np.linspace(0.01,1,num=999)))
plt.legend(["x^2"] , loc = 1)

# In[]:
def SC(Membership,Data,Centers,m):
    Shape = np.shape(Membership)
    N = Shape[0]
    K = Shape[1]
    FinalSc = 0
    for j in range(K):
        bank = 0
        bank2 = 0
        for i in range(N):
            bank = bank + Membership[i,j]**m *np.linalg.norm(Data[i] - Centers[j],ord = 2)**2
        for j2 in range(K):
            bank2 = bank2 + np.linalg.norm(Centers[j2]-Centers[j], ord=2)**2
        print(sum(Membership[:,j]))
        FinalSc = FinalSc + bank/(sum(Membership[:,j])*bank2)
    return FinalSc

# In[]:
def S(Membership,Data,Centers):
    Shape = np.shape(Membership)
    N = Shape[0]
    K = Shape[1]
    Centers_Dist = np.ones([K,K])*10**10
    for j1 in range(K):
        for j2 in range(K-j1):
            if j1 != j2:
                Centers_Dist[j1,j2] =  np.linalg.norm(Centers[j1] - Centers[j2],ord = 2)**2
    Min = np.min(Centers_Dist)
    bank = 0
    for j in range(K):
        for i in range(N):
            bank = bank + Membership[i,j]**2 *np.linalg.norm(Data[i] - Centers[j],ord = 2)**2
    FinalS = bank/(N*Min)
    return FinalS, Centers_Dist, Min,bank, N

# In[]:
def XB(Membership,Data,Centers, m):
    Shape = np.shape(Membership)
    N = Shape[0]
    K = Shape[1]
    Dist = np.ones([N,K])
    for i1 in range(N):
        for j1 in range(K):
            Dist[i1,j1] =  np.linalg.norm(Data[i1] - Centers[j1],ord = 2)**2
    Min = np.min(Dist)
    bank = 0
    for j in range(K):
        for i in range(N):
            bank = bank + Membership[i,j]**m *np.linalg.norm(Data[i] - Centers[j],ord = 2)**2
    FinalXB = bank/(N*Min)
    return FinalXB, Dist, Min