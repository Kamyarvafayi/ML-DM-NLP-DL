# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math

# preparing data 
#data=pd.read_csv (r"C:\Users\ASUS\Desktop\customers.csv")
#data.drop(['Region','Channel'],axis=1,inplace=True)# axis=1 column axis=0 row
data=pd.read_csv (r"C:\Users\ASUS\Desktop\iris.data")
data.drop(['class'],axis=1,inplace=True)# axis=1 column axis=0 row
dataarr=np.array(data,dtype=float)
dataarr2=np.matrix(dataarr).transpose()
# normalizing the data
s=np.shape(dataarr2)
miu=[]
var=[]
for i in range(s[0]):
    miu.append(np.mean(dataarr2[:][i]))
    var.append(np.var(dataarr2[:][i]))
    dataarr2[:][i]=(dataarr2[:][i]-miu[i]*np.ones([1,s[1]]))/math.sqrt(var[i])
## finding outliers
outlierindex=[]
for i in range(s[1]):
    isoutlier='no';
    for j in range(s[0]):
        if (dataarr2[j,i]>3 or dataarr2[j,i]<-3):
            isoutlier='yes';
    if isoutlier=='yes':
        outlierindex.append(i)
## removing outliers
dataarr3=np.delete(dataarr2,np.array(outlierindex),axis=1)# axis=1 column axis=0 row
dataarr3=np.matrix(dataarr3,dtype=float)
#####################1.PCA
## finding cov matrix
covmtx=np.cov(np.matrix(dataarr3,dtype=float))
## finding the value of eigenvectors and eigenvalues
eigen=np.linalg.eig(covmtx)
## eigen values
eigenval=eigen[0]
eigenval=np.sort(eigenval)
## eigen vectors
eigenvec=np.matrix(eigen[1]) ## now , all eigenvectors are in the row of this matrix
## finding necessary eigen values
threshold=0.9
bank=0
necessarynumberofeigen=0
eshape=np.shape(eigenval)
for i in range(eshape[0]):
    bank=bank+eigenval[eshape[0]-i-1]/sum(eigenval)
    necessarynumberofeigen+=1
    if bank>threshold:
        break
    
newdataPCA=dataarr3.transpose()*eigenvec[:,0:necessarynumberofeigen]
####################2.SVD
svd=np.linalg.svd(dataarr3.transpose())
u=svd[0]
d=svd[1]
v=np.matrix(svd[2]).transpose()
## finding necessary eigen values
threshold=0.9
bank2=0
necessarynumberofd=0
dshape=np.shape(d)
for i in range(dshape[0]):
    bank2=bank2+d[i]/sum(d)
    necessarynumberofd+=1
    if bank2>threshold:
        break
newdataSVD=dataarr3.transpose()*v[:,0:necessarynumberofd]