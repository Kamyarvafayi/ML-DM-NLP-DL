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
                step = self.Learning_Rate*np.exp(-1*np.linalg.norm(self.NeuronsPositions[np.argmax(self.Winner[i])]-self.NeuronsPositions[j])**2/(2*self.Sigma**2)) * np.linalg.norm(self.Input[i]-self.Weights[j])
                self.Weights[j] = self.Weights[j] + step/self.Input.shape[0]
    def SOM_Clustering_Fit(self, Input, Initial_Learning_Rate = 0.1, Initial_Sigma = 1):
        self.Input = Input
        self.Learning_Rate = Initial_Learning_Rate
        self.Sigma = Initial_Sigma
        self.Create_Neuron_Position_array()
        self.Initialization()
        for i in range(self.Iter):
            self.Find_Distance()
            self.Find_Winner()
            self.Update_Weights() 
            #print(self.Winner)
            self.Learning_Rate *= 0.63
            self.Sigma *= np.exp(i/self.Iter)
# In[]:
import sklearn.datasets as dataset
Iris_Data = dataset.load_iris()
Input = Iris_Data["data"]
SOM = SelfOrganizingMap_Clustering(10,Grid = [2,3])
model = SOM.SOM_Clustering_Fit(Input)
Weights =SOM.Weights
Cluster = SOM.Winner
print(Cluster)
print(np.sum(SOM.Winner,axis=0))

