# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:00:24 2019

@author: ASUS
"""

import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import sklearn.cluster as cl
import numpy as np
from scipy.stats import *
import math

z=[]
# preparing data and charts
data = sns.load_dataset("iris")
data.drop('species',axis=1,inplace=True)
#pd.plotting.scatter_matrix(data,figsize=(14,8),diagonal='kde')
#plt.show()
#pd.plotting.boxplot(data)
#plt.show()
dataarr=np.array(data)
# taking the number of clusters from user
for n in range(2,11):
    zscore=[]
    ## finding first means with k_means
    Kmeans=cl.KMeans(n_clusters=n)
    kmeans=Kmeans.fit_predict(data)
# initial miu
    miu=Kmeans.cluster_centers_
# initial pi
    pi=np.zeros([1,n])
    for i in range(n):
        pi[0,i]=1/n
# first covariance matrix
    co=np.zeros([4,4])
    diag=np.ones([1,4])
    covmtx=[]
    np.fill_diagonal(co,1)
    covmtx=[co]
    covmtx=covmtx*n
    numerator=0
    gamma=np.zeros([n,150])
    nk=np.zeros([1,n])
# main loop
    while numerator<=30:
        # finding gamma for all K and N
        #upgrading gamma
        gamma=np.zeros([n,150],dtype=float)
        for k in range (150):
            normal=0.0
            for i in range(n):
                normal=normal+pi[0][i]*multivariate_normal.pdf(dataarr[k],mean=miu[i],cov=covmtx[i])
            for i in range(n):
                gamma[i][k]=(pi[0][i]*multivariate_normal.pdf(dataarr[k],mean=miu[i],cov=covmtx[i]))/(normal)
        # finding Nk
        for i  in range(n):
            nk[0][i]=sum(gamma[i])
        # finding new centers(miu)
        for i in range(n):
            for t in range(4):
                miu[i][t]=0
                for k in range(150):
                    miu[i][t]=miu[i][t]+gamma[i][k]*dataarr[k][t]/nk[0][i]  
        # upgrading cov matrix
        data_mtx=np.matrix(dataarr)
        miu_mtx=np.matrix(miu)
        for i in range(n):
            covmtx[i]=np.zeros([4,4])
            for k in range(150):
                covmtx[i]=covmtx[i]+gamma[i][k]*((data_mtx[k][:]-miu_mtx[i][:]).transpose()*(data_mtx[k][:]-miu_mtx[i][:]))/nk[0][i]
        #upgrading pi
        pi=nk/150
        # finding objective function(log liklihood)
        zscore.extend([0.0])
        for k in range(150):
            bank=0.0
            for i in range(n):
                bank=bank+pi[0][i]*multivariate_normal.pdf(dataarr[k],mean=miu[i],cov=covmtx[i])
            zscore[numerator]=zscore[numerator]+math.log(bank) 
        # convergence criteria
        if numerator>1:
            if zscore[numerator]-zscore[numerator-1]<0.01:
                break
        numerator=numerator+1
    z.append(zscore[numerator-1])
plt.plot(z,'r-o')
plt.xlabel('K')
plt.ylabel('log liklihood')
plt.title('objective function for K between 2 and 10'.format(n))
plt.axis([2,10,z[0]-10,z[-1]+10])