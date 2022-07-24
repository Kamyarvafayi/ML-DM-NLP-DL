import numpy as np
import random
# In[]: Competitive Learning
class SelfOrganizingMap_Clustering:
    def __init__ (self, Iter = 1000, Grid = [3,3]):
        self.Iter = Iter
        self.Grid_Rows = Grid[0]
        self.Grid_Columns = Grid[1]
    def Initialization (self):
        random_indices = random.sample(range(self.Input.shape[0]),self.Grid_Rows*self.Grid_Columns)
        #random_indices = np.array([0,51,101])
        self.Weights = np.array([self.Input[i] for i in random_indices])
        #self.Weights= np.array([[np.random.uniform(np.min(self.Input[i]), np.max(self.Input[i])) for i in range(self.Input.shape[1])] for j in range(self.Grid_Columns*self.Grid_Rows)])
        from sklearn.cluster import KMeans
        Model = KMeans(n_clusters=self.Grid_Rows*self.Grid_Columns)
        Model = Model.fit(self.Input)
        self.Weights = Model.cluster_centers_
    def PCA_Initialization(self):
        Cov_MTX = np.cov(self.Input.T)
        eig_VAl , Eig_Vec = np.linalg.eig(Cov_MTX)
        Random_Coeff = np.array([[np.random.randn() for i in range(2)] for j in range(self.Grid_Rows*self.Grid_Columns)])
        self.Weights = np.array([Random_Coeff[i,0] * Eig_Vec[0] + Random_Coeff[i,1]*Eig_Vec[1] for i in range(self.Grid_Rows*self.Grid_Columns)])
    def Find_Distance(self):
        self.Distance = np.array([[np.linalg.norm(self.Input[i]-self.Weights[j]) for j in range(self.Grid_Rows*self.Grid_Columns)]
                                 for i in range(self.Input.shape[0])])
    def Create_Neuron_Position_array(self):
        self.NeuronsPositions = []
        for i in range(self.Grid_Rows):
            for j in range(self.Grid_Columns):
                self.NeuronsPositions.append(np.array([i,j]))
    def Find_NeuronDistance(self,Neuron1Position, Neuron2Position):
        return np.linalg.norm(Neuron1Position-Neuron2Position)
    def Find_Winner(self):
        self.Winner = np.zeros([self.Input.shape[0], self.Grid_Rows*self.Grid_Columns])
        for i in range(self.Input.shape[0]):
            self.Winner[i, np.argmin(self.Distance[i])] = 1
    def Update_Weights(self):
        for i in range(self.Input.shape[0]):
            for j in range(self.Grid_Rows*self.Grid_Columns):
                step = self.Learning_Rate*np.exp(-1*np.linalg.norm(self.NeuronsPositions[np.argmax(self.Winner[i])]-self.NeuronsPositions[j])**2/(2*self.Sigma**2)) * (self.Input[i]-self.Weights[j])
                self.Weights[j] = self.Weights[j] + step
    def SOM_Clustering_Fit(self, Input, Initial_Learning_Rate = 1, Initial_Sigma = 1, Initialization = "Kmeans"):
        self.Input = Input
        self.Learning_Rate = Initial_Learning_Rate
        self.Sigma = Initial_Sigma
        self.Create_Neuron_Position_array()
        if Initialization =="Kmeans":
            self.Initialization()
        elif Initialization=="PCA":    
            self.PCA_Initialization()
        print(self.Weights)
        for i in range(self.Iter):
            self.Find_Distance()
            self.Find_Winner()
            self.Update_Weights() 
            self.Learning_Rate *= (1-i/self.Iter)
            self.Sigma *= np.exp(-i/self.Iter)
# In[]:
import sklearn.datasets as dataset
Iris_Data = dataset.load_iris()
Input = Iris_Data["data"]
Grid = [2,1]
SOM = SelfOrganizingMap_Clustering(50,Grid = Grid)
model = SOM.SOM_Clustering_Fit(Input, Initial_Learning_Rate = 1, Initial_Sigma = 1, Initialization="PCA")
Weights =SOM.Weights
Cluster = SOM.Winner
print(Cluster)
print(np.sum(SOM.Winner,axis=0))
# In[]:
import matplotlib.pyplot as plt
color = ['red','blue', 'green', 'black', 'cyan','yellow','brown','purple','grey','orange']
for i in range(Grid[0]*Grid[1]):
    if i >=10:
        color.append([np.random.rand() for counter in range(3) ])
    plt.scatter(Input[Cluster[:,i]==1,0], Input[Cluster[:,i]==1,1],color = color[i])
    plt.scatter(Weights[i,0],Weights[i,1],marker= 'x', color = color[i])
    plt.xlabel('X1')
    plt.ylabel('x2')
plt.figure()
for i in range(Grid[0]*Grid[1]):
    plt.scatter(Input[Cluster[:,i]==1,2], Input[Cluster[:,i]==1,3],color = color[i])
    plt.scatter(Weights[i,2],Weights[i,3],marker= 'x', color = color[i])
    plt.xlabel('X3')
    plt.ylabel('x4')
