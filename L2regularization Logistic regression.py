import numpy as np
import sklearn.datasets as sk
import matplotlib.pyplot as plt

def sigmoid(X, W):
#    return (np.exp(-1*W.T.dot(X)))/(1+np.exp(-1*W.T.dot(X)))
    return (1)/(1+np.exp(-1*W.T.dot(X)))
Num = 100
moonsdata, moonsclass = sk.make_moons(Num)
def initialization(X):
    shape = np.shape(X)
    P = shape[1]
    W = np.random.randint(-1,2, size = [3,1]) * np.random.rand(P + 1,1)
    return W
def derivative(X, Y, W, i,c):
    shape = np.shape(X)
    delta = 0
    delta = delta + (Y-sigmoid(X,W))*(1-sigmoid(X,W))*sigmoid(X,W)*X[i,0]-np.array(c*np.sign(W[i,0])/Num)
    return delta
def learning(W, delta, LearningRate):
    return W + delta*LearningRate

w = initialization(moonsdata)
shape = np.shape(moonsdata)
N = shape[0]
moonsdataadded = np.append(moonsdata,np.ones([N,1]),axis=1)
shape = np.shape(moonsdataadded)
P = shape[1]
initiallearningrate = 10
decay = 0.01
counter = 0
Lambda = 0.1
while counter < 1500:
    learningrate = initiallearningrate/(1+decay*counter)
    for i in range(N):
        delta = np.zeros([P,1])
        for j in range(P):
            delta[j,0] = derivative(np.reshape(moonsdataadded[i],[P,1]), moonsclass[i], w, j,Lambda)
        for j in range(P):
            w[j,0] = learning(w[j,0], delta[j,0], learningrate)
    counter = counter + 1
#print(derivative(moonsdataadded[0:2],moonsclass[0:2], w,0))


import sklearn.linear_model as lm
LR  =  lm.LogisticRegression(penalty = 'l1', tol=0.001)
LRmodel = LR.fit(moonsdata,moonsclass)
print(LRmodel.coef_,LRmodel.intercept_)




