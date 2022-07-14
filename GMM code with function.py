import seaborn as sns
import matplotlib.pylab as plt
import sklearn.cluster as cl
import numpy as np
from scipy.stats import *
import math
import pandas as pd
# In[]:
def LoadingData():
    data = sns.load_dataset("iris") 
    data.drop('species',axis=1,inplace=True)
    data = np.array(data)
    return data
# In[]:    
def KmeansResult(data , NumberofClusters):
    # finding initial mu using Kmeans
    KmeansModel = cl.KMeans(n_clusters = NumberofClusters).fit(data)
    Centers = KmeansModel.cluster_centers_
    return Centers
# In[]:
## Initialization
def GmmInitialization(NumberofClusters , InitialCenters,data):
    GmmCenters = InitialCenters
    
    CovMatrix = []   
    for i in range(NumberofClusters):
        CovMatrix.append(np.identity(data.shape[1]))
    pi =1/NumberofClusters*np.ones([NumberofClusters,1])
    return GmmCenters , CovMatrix , pi
# In[]:
## Membership
def findmembership(data , NumberofClusters,GmmCenters,CovMatrix , pi):
    Membership = np.zeros([NumberofClusters,data.shape[0]])
    for k in range(data.shape[0]):
        bank = 0.0
        for i in range(NumberofClusters):
            bank = bank + pi[i]*multivariate_normal.pdf(data[k],mean = GmmCenters[i],cov =CovMatrix[i])
        for i in range(NumberofClusters):
            Membership[i][k]=(pi[i]*multivariate_normal.pdf(data[k],mean=GmmCenters[i],cov=CovMatrix[i]))/(bank)
    return Membership
# In[]:
## Pi
def findingPi(NumberofClusters,Membership):
    pi = np.zeros([NumberofClusters,1])
    for i in range(NumberofClusters):
        pi[i] = sum(Membership[i])
    return pi
# In[]:
## mean and covmatrix
def findingCentersandCovmtx(data,NumberofClusters,Membership,pi):
    GmmCenters=np.zeros([NumberofClusters,data.shape[1]])
    for i in range(NumberofClusters):
        for t in range(data.shape[1]):
            for k in range(data.shape[0]):
                GmmCenters[i][t]=GmmCenters[i][t]+Membership[i][k]*data[k][t]/pi[i]      
    CovMatrix = [[]]*NumberofClusters
    data_mtx=np.matrix(data)
    Centers_mtx=np.matrix(GmmCenters)
    for i in range(NumberofClusters):
        CovMatrix[i]=np.zeros([data.shape[1],data.shape[1]])
        for k in range(data.shape[0]):
            CovMatrix[i]= CovMatrix[i] + Membership[i][k]*((data_mtx[k][:]-Centers_mtx[i][:]).transpose()*(data_mtx[k][:]-Centers_mtx[i][:]))/pi[i]
    return GmmCenters , CovMatrix
# In[]:
## Objective Function
def findobjectivefunction(data,NumberofClusters,pi,GmmCenters,CovMatrix):
    PiPercentage=pi/data.shape[0]
    # finding objective function(log liklihood)
    objectivefunction = 0
    for k in range(data.shape[0]):
        bank=0.0
        for i in range(NumberofClusters):
            bank=bank+PiPercentage[i]*multivariate_normal.pdf(data[k],mean=GmmCenters[i],cov=CovMatrix[i])
        objectivefunction = objectivefunction + math.log(bank) 
    return objectivefunction
# In[]:
## Main GMM Function
def GMM(Data = LoadingData(), NumberofClusters = 3,Criteria = 0.01):
    data = Data;
    KmeansCenter = KmeansResult(data,NumberofClusters)
    GmmCenters , CovMatrix ,pi = GmmInitialization(NumberofClusters , KmeansCenter,data)
    # main loop
    NumberofIterations = 0
    ObjectiveFunctions = []
    while True:
        Membership = findmembership(data , NumberofClusters,GmmCenters,CovMatrix,pi)
        pi = findingPi(NumberofClusters,Membership)
        GmmCenters , CovMatrix = findingCentersandCovmtx(data,NumberofClusters,Membership,pi)
        ObjectiveFunctions.append(findobjectivefunction(data,NumberofClusters,pi,GmmCenters,CovMatrix))
        if NumberofIterations > 1:
            if ObjectiveFunctions[-1]- ObjectiveFunctions[-2] < Criteria:
                break
        NumberofIterations = NumberofIterations + 1
    # the outputs of the model are stored in a dictionary
    model = {
            "data" : data ,
            "Membership" : Membership ,
            "pi" : pi ,
            "GmmCenters" : GmmCenters , 
            "CovMatrix" : CovMatrix ,
            "ObjectiveFunctions" : ObjectiveFunctions , 
            "NumberofClusters" : NumberofClusters ,
            "Criteria" : Criteria
            }
    return model
# In[]:
def drawingplot(model):
    plt.plot(model["ObjectiveFunctions"],'g-o')
    plt.xlabel('iterations')
    plt.ylabel('log liklihood')
    plt.title('objective function for K={}'.format(model["NumberofClusters"]))
    plt.axis([0,len(model["ObjectiveFunctions"]),model["ObjectiveFunctions"][0]-10,model["ObjectiveFunctions"][-1]+10])    
# In[]:       
NumberofClusters = int(input('Enter the number of clusters: '))
EndingCriteria = float(input('Enter the Ending Criteria: '))

Data = LoadingData()
# First Model (Data = Iris, K = 3, Criteria = 0.01 )
GmmModel = GMM(Data)
# Second Model (Data = Iris, K = 10, Criteria = 0.1 )
GmmModel2 = GMM(NumberofClusters=10,Criteria=0.1)
# Third Model (Data = Iris, K = given by user, Criteria = given by user )
newData = Data
GmmModel3 = GMM(newData,NumberofClusters, EndingCriteria)

plt.subplot(1,3,1)
drawingplot(GmmModel)
plt.subplot(1,3,2)
drawingplot(GmmModel2)
plt.subplot(1,3,3)
drawingplot(GmmModel3)

