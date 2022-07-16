# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
print(1)
x= 10
from math import *


Member2 = np.array([[0.1,0.9],[0.7,0.3],[0.5,0.5],[0.05,0.95],[0.95,0.05]])
def PC(Membership):
    bank = 0
    Shape = np.shape(Membership)
    N = Shape[0]
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            bank = bank + Membership[i,j]**2
    
    return bank/N


print(PC(Member2))



def CE(Membership):
    bank = 0
    Shape = np.shape(Membership)
    N = Shape[0]
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            bank = bank + Membership[i,j]*np.log(Membership[i,j])
    return -bank/N
print(SC(Member2))



def xlogx(x):
    return x*np.log(x)

def x2(x):
    return x**2

plt.plot(np.linspace(0.001,1,num=999),xlogx(np.linspace(0.001,1,num=999)))
plt.plot(np.linspace(0.001,1,num=999),-xlogx(np.linspace(0.001,1,num=999)))
plt.figure()
plt.plot(np.linspace(0.01,1,num=999),x2(np.linspace(0.01,1,num=999)))





def SC2(Membership,Data,Centers,m = 2):
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
        FinalSc = FinalSc + bank/(sum(Membership[:,j]*bank2))
    return FinalSc



def SC2(Membership,Data,Centers,m = 2):
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
        FinalSc = FinalSc + bank/(sum(Membership[:,j]*bank2))
    return FinalSc