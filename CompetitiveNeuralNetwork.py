import numpy as np
import random
# In[]: Competitive Learning
class Competitive_Clustering:
    def __init__ (self, Iter = 1000, Neurons = 3):
        self.Iter = Iter
        self.Neurons_Num = Neurons
    def Initialization (self):
        #random_indices = random.sample(range(self.Input.shape[0]),self.Neurons_Num)
        random_indices = np.array([0,51,101])
        self.Weights = np.array([self.Input[i] for i in random_indices])
        #self.Weights= np.array([[np.random.uniform(np.min(self.Input[i]), np.max(self.Input[i])) for i in range(self.Input.shape[1])] for j in range(self.Neurons_Num)])
    def Find_Distance(self):
        self.Distance = np.array([[np.linalg.norm(self.Input[i]-self.Weights[j]) for j in range(self.Neurons_Num)]
                                 for i in range(self.Input.shape[0])])
    def Find_Winner(self):
        self.Winner = np.zeros([self.Input.shape[0], self.Neurons_Num])
        for i in range(self.Input.shape[0]):
            self.Winner[i, np.argmin(self.Distance[i])] = 1
    def Update_Weights(self):
        direction = []
        for i in range(self.Neurons_Num):
            step = np.array([np.average(self.Input[self.Winner[:,i]==1],axis = 0)])
            self.Weights[i] = self.Weights[i] + self.Learning_Rate*step/self.Input.shape[0]
            direction.append(step)
    def Competitive_Clustering_Fit(self, Input, Initial_Learning_Rate = 1):
        self.Input = Input
        self.Learning_Rate = Initial_Learning_Rate
        self.Initialization()
        for i in range(self.Iter):
            self.Find_Distance()
            self.Find_Winner()
            self.Update_Weights() 
            print(self.Winner)
            self.Learning_Rate *= 0.8
# In[]:
import sklearn.datasets as dataset
Iris_Data = dataset.load_iris()
Input = Iris_Data["data"]
Competitive = Competitive_Clustering(100,3)
Competitive.Competitive_Clustering_Fit(Input)
print(Competitive.Weights)
print(np.sum(Competitive.Winner,axis=0))
# In[]: Kmeans
from sklearn.cluster import KMeans
Model = KMeans(n_clusters=Competitive.Neurons_Num)
Model = Model.fit(Competitive.Input)
print(Model.cluster_centers_)
