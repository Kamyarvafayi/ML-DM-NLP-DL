# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import random as rnd
import pandas as pd

def LoadingData():
    data = sns.load_dataset("iris") # the format of this data is dataframe
    # we just need the features, that is why the species must be removed
    data.drop('species',axis=1,inplace=True)
    # converting the type of the data to array
    data = np.array(data)
    return data

def Initialization(Data, DataShape,NumberofClusters):
    InitialCentersIndices = []
    InitialCenters = np.zeros([NumberofClusters,DataShape[1]])
    for i in range(NumberofClusters):
        Status = True
        while Status:
            RandomIndex = rnd.randint(0,DataShape[0]-1)
            if RandomIndex not in InitialCentersIndices:
                InitialCentersIndices.append(RandomIndex)
                InitialCenters[i] = Data[RandomIndex]
                Status = False
    return InitialCenters , InitialCentersIndices

def FindMembership(Data,DataShape,NumberofClusters,CentersMtx):
    Membership = np.zeros([DataShape[0],NumberofClusters])
    for i in range(DataShape[0]):
        MinIndex = 0
        MinDist =  np.linalg.norm(Data[i]-CentersMtx[0],ord = 2)
        for j in range(NumberofClusters):
            if np.linalg.norm(Data[i]-CentersMtx[j],ord = 2)<MinDist:
                MinIndex = j
                MinDist = np.linalg.norm(Data[i]-CentersMtx[j],ord = 2)
        Membership[i][MinIndex] = 1
    return Membership

def Objective(Data,DataShape,NumberofClusters,CentersMtx,Membership):
    ObjectiveFun = 0
    for j in range(NumberofClusters):
        for i in range(DataShape[0]):
            if Membership[i,j] == 1:
               ObjectiveFun = ObjectiveFun + np.linalg.norm(Data[i]-CentersMtx[j],ord = 2)
    return ObjectiveFun
    
def centers(Data,Membership,DataShape,NumberofClusters):
    CentersMtx = np.zeros([NumberofClusters,DataShape[1]])
    for j in range(NumberofClusters):
        Bank = 0
        for i in range(DataShape[0]):
            if Membership[i,j] == 1:
               Bank = Bank + Data[i]
        CentersMtx [j] = Bank/sum(Membership[:,j])
    return CentersMtx
           
def MyKmeans(Data = LoadingData(),NumberofClusters = 3, EndingCriteria = 0.01):
    #Data shape and number of samples and features are found
    DataShape = np.shape(Data)
    NumSamples = DataShape[0]
    NumFeatures = DataShape[1]
    AllObjectives = []
    Iteration = 0
    CentersMtx , InitialCentersIndices = Initialization(Data, DataShape,NumberofClusters)
    while True:
        Membership = FindMembership(Data,DataShape,NumberofClusters,CentersMtx)
        Z = Objective(Data,DataShape,NumberofClusters,CentersMtx,Membership)
        AllObjectives.append(Z)
        if Iteration >=1:
            if AllObjectives[Iteration-1]-AllObjectives[Iteration] < EndingCriteria:
                break
        CentersMtx = centers(Data,Membership,DataShape,NumberofClusters)
        Iteration = Iteration + 1
        
    Model = {
            "Data" : Data ,
            "NumberofClusters" : NumberofClusters ,
            "Membership" : Membership ,
            "InitialCentersIndices" : InitialCentersIndices ,
            "ClustersCenters" : CentersMtx ,
            "ObjectiveFunctions" : AllObjectives,
            "EndingCriteria" : EndingCriteria
            }
    return Model

def drawingplot1(model):
    plt.figure()
    plt.plot(model["ObjectiveFunctions"],'g-o')
    plt.xlabel('iterations')
    plt.ylabel('Sum of distances from centers')
    plt.title('objective function for K={}'.format(model["NumberofClusters"] ) )
    plt.axis([0,len(model["ObjectiveFunctions"]),model["ObjectiveFunctions"][-1]-10 ,model["ObjectiveFunctions"][0]+10])
    
Model1 = MyKmeans()
Model2 = MyKmeans(Data = LoadingData(),NumberofClusters = 10,EndingCriteria = 0.0001)
drawingplot1(Model1)
drawingplot1(Model2)