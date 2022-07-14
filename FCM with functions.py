# In[]:
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import random as random
import pandas as pd
# In[]:
def LoadingData():
    data = sns.load_dataset("iris") 
    data.drop('species',axis=1,inplace=True)
    data = np.array(data)
    return data
# In[]: First Initialization
# You can try both of these Initializations and check whether the initial membership influences the final results or not
def FcmInitialization(data,NumberofClusters,FuzzyDegree):
    Membership = np.zeros([data.shape[0],NumberofClusters])
    for i in range(data.shape[0]):
        RandNumber = random.random()
        for k in range(NumberofClusters):
            if (RandNumber <=(k+1)/NumberofClusters) & (RandNumber >= (k)/NumberofClusters):
                Membership[i][k]=1
    MembershipInThePowerofR = Membership**FuzzyDegree
    return Membership , MembershipInThePowerofR
# In[]: Second Initialization
def FcmInitialization2(data,NumberofClusters,FuzzyDegree):
    Membership = np.zeros([data.shape[0],NumberofClusters])
    for i in range(data.shape[0]):
        for k in range(NumberofClusters):
            if (i/data.shape[0] <(k+1)/NumberofClusters) & (i/data.shape[0] >= (k)/NumberofClusters):
                Membership[i,k]=1
    MembershipInThePowerofR = Membership**FuzzyDegree
    return Membership , MembershipInThePowerofR
# In[]:
def FindingClustersCenters(data , MembershipInThePowerofR  ,NumberofClusters ):
     ClustersCenters = np.zeros([NumberofClusters,data.shape[1]])
     for k in range(NumberofClusters):
        bank = 0
        for i in range(data.shape[0]):
            bank = bank + (MembershipInThePowerofR [i,k])*data[i]/sum(MembershipInThePowerofR [:,k])  
        ClustersCenters[k,:]=bank
     return ClustersCenters 
# In[]:
def FindDistances(data,NumberofClusters,ClustersCenters):
    Distances = np.zeros([data.shape[0],NumberofClusters])
    for k in range(NumberofClusters):
        for i in range(data.shape[0]):
            Distances[i,k] = np.linalg.norm(data[i]-ClustersCenters[k],ord=2)**1  
        
    return Distances
# In[]:
def FindMembership(data, NumberofClusters ,FuzzyDegree , Distances):
    Membership = np.zeros([data.shape[0],NumberofClusters])
    for i in range(data.shape[0]):
        if np.where(Distances[i]==0,True,False).all()==False:
            for k in range(NumberofClusters):
                bank = 0;
                for t in range(NumberofClusters):
                    bank = bank + (Distances[i,k]/Distances[i,t])**(2/(FuzzyDegree-1))
                Membership[i,k] = 1/bank
        else:
            Membership[i]=np.where(Distances[i]==0,1,0) 
            
    MembershipInThePowerofR = Membership**FuzzyDegree
    
    return Membership , MembershipInThePowerofR    
# In[]:   
def ObjectiveFunction(MembershipInThePowerofR,Distances):
    Objective=0;
    for i in range(MembershipInThePowerofR.shape[0]):
        for k in range(MembershipInThePowerofR.shape[1]):
            Objective = Objective + MembershipInThePowerofR[i,k]* Distances[i,k]**2;
    return Objective
# In[]:        
def FCM(Data=LoadingData(),NumberofClusters = 3,FuzzyDegree = 2,Criteria = 0.01,InitializationMethod = "Random"):
    data = Data
    if InitializationMethod=="Random":
        Membership , MembershipInThePowerofR = FcmInitialization(data,NumberofClusters,FuzzyDegree)
    elif InitializationMethod=="Arbitrary":
        Membership , MembershipInThePowerofR = FcmInitialization2(data,NumberofClusters,FuzzyDegree)
    else: 
        print("Your Initialization Method is Invalid. Please choose between Random or Arbitrary")
        return "ERROR"
    ObjectiveFunctions = []
    NumberofIterations = 0
    while True:   
        ClustersCenters = FindingClustersCenters(data , MembershipInThePowerofR  ,NumberofClusters)
        Distances = FindDistances(data,NumberofClusters,ClustersCenters)
        ObjectiveFunctions.append(ObjectiveFunction(MembershipInThePowerofR,Distances))
        Membership , MembershipInThePowerofR = FindMembership(data, NumberofClusters ,FuzzyDegree , Distances)
        if NumberofIterations > 1:
            if ObjectiveFunctions[-2]- ObjectiveFunctions[-1] < Criteria:
                break
        NumberofIterations = NumberofIterations + 1
    Model = {
            "Data" : data ,
            "NumberofClusters" : NumberofClusters ,
            "FuzzyDegree" : FuzzyDegree ,
            "Membership" : Membership ,
            "MembershipInthePowerofR" : MembershipInThePowerofR ,
            "ClustersCenters" : ClustersCenters ,
            "Distances" : Distances ,
            "ObjectiveFunctions" : ObjectiveFunctions
            }
    return Model
# In[]:
def drawingplot(model):
    plt.plot(model["ObjectiveFunctions"],'g-o')
    plt.xlabel('iterations')
    plt.ylabel('Sum of distances from centers')
    plt.title('objective function for K={}'.format(model["NumberofClusters"] )+' and FuzzyDegree={}'.format(model["FuzzyDegree"] ))
    plt.axis([0,len(model["ObjectiveFunctions"]),model["ObjectiveFunctions"][-1]-10 ,model["ObjectiveFunctions"][0]+10])    

Model1 = FCM()
Model2 = FCM(NumberofClusters = 3 , FuzzyDegree = 3)
Model3 = FCM(NumberofClusters = 10,InitializationMethod="Arbitrary")

plt.subplot(1,3,1)
drawingplot(Model1)
plt.subplot(1,3,2)
drawingplot(Model2)
plt.subplot(1,3,3)
drawingplot(Model3)