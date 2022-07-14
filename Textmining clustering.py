# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:51:49 2019

@author: ASUS
"""
import random
d1='data mining is a rather new field of study for discovering knowledge from data.this data can be numeric data or even a text,so data mining is of paramount importance.'
d2='physical activities helps all people to remain healthy.another way they have for remaining healthy is eating food which is healthy.'
d3="all people have access to a computer.computer was devised about 50 years ago by some American scientists.since then,it has affected all parts of our lives."
#d3='data mining is a rather new field of study.this data can be numeric data or even a text,so data mining is importance.'
docs=list();
docs.append(d1)
docs.append(d2)
docs.append(d3)
## making list of commonly used words
commonwords=['a','an','the','all','people','it','is','this','of','for','from','can','be','or','even','so','to','another','other','they','have','which','was','about','by','some','since','then','has','our','way']
vocablibrary=[]
# finding the length of all documents
length=list()
for i in range(len(docs)):
    length.append(len(docs[i]))
word=[]
# spliting words of all docs and creating  vocablibrary
for j in range(3):
    count=0
    count2=0
    s=[]
    for i in range(length[j]):
        if (docs[j][i]!=' ' and docs[j][i]!='.' and docs[j][i]!=','):
            count2=count2+1
        else:
            s.append(docs[j][count:(count+count2)])
            vocablibrary.append(docs[j][count:(count+count2)])
            count=count+count2+1
            count2=0
    word.append(s)
## removing duplicates in the word list
wordwithoutduplicates=[[1],[2],[3]]
for j in range(len(word)):
    for i in range(len(word[j])):
        if word[j][i] not in wordwithoutduplicates[j]:
            if i==0:
                wordwithoutduplicates[j][i]=word[j][i]
            else:
                wordwithoutduplicates[j].append(word[j][i])
## removing duplicates from vocab library
vocablibrarywithoutduplicates=[]
for i in range(len(vocablibrary)):
    if vocablibrary[i] not in vocablibrarywithoutduplicates:
        vocablibrarywithoutduplicates.append(vocablibrary[i])
## counting the repetition of words in all documents
countofwords=[[0],[0],[0]]                
for j in range(len(word)):
    for i in range(len(vocablibrarywithoutduplicates)):
        countofrepword=0
        for k in range(len(word[j])):
            if word[j][k]==vocablibrarywithoutduplicates[i]:
                countofrepword+=1
        if i==0:
            countofwords[j][i]=countofrepword
        else:
            countofwords[j].append(countofrepword)
############################################
############################ 1.CLUSTERING
############# initialization 
#p(w|theta)            
PWordsinclusters=[]
numberofclusters=2
for k in range(numberofclusters):
    temp=[1]
    sumofrand=0;
    for j in range(len(vocablibrarywithoutduplicates)):
        if j==0:
            temp[j]=((random.uniform(0,1)))
        else:
            temp.append((random.uniform(0,1)))
    sumofrand=sumofrand+sum(temp)
    for j in range(len(vocablibrarywithoutduplicates)):
        temp[j]=temp[j]/sumofrand
    if k==0:
            PWordsinclusters=[temp]
    else:
            PWordsinclusters.append(temp)
                      
#p(clusters)
clusterprobobility=[]
for i in range(numberofclusters):
    clusterprobobility.append(1/numberofclusters)    
############### while loop
##########################
numerator=0
Z=[]       
while numerator<=3:
## finding membership(P(zd=0 or 1 if a doc is chosen))
    membership=[[1],[1]]
    for i in range(3):
        Bank=[]
        for l in range(numberofclusters):
            bank=1;
            for j in range(len(vocablibrarywithoutduplicates)):
                bank=bank*PWordsinclusters[l][j]**countofwords[i][j]
            Bank.append(bank*clusterprobobility[l])
        if i==0:
            membership[0]=[Bank[0]/sum(Bank)]
            membership[1]=[Bank[1]/sum(Bank)]
        else:
            membership[0].append(Bank[0]/sum(Bank))
            membership[1].append(Bank[1]/sum(Bank))
## finding clusters probobility according to memberships
    clusterprobobility=[]
    for l in range(numberofclusters):
        clusterprobobility.append(sum(membership[l])/3)             
## finding new P(w if theta)
    PWordsinclusters=[[1],[1]]
    for j in range(len(vocablibrarywithoutduplicates)):
        for l in range(numberofclusters):
            bank=0;
            for i in range(len(docs)):
                bank+=membership[l][i]*countofwords[i][j]
            if j==0:
                PWordsinclusters[l][j]=bank
            else:
                PWordsinclusters[l].append(bank)
## normalizing PWordsinclusters
    for l in range(numberofclusters):
        sumall=sum(PWordsinclusters[l])
        for j in range(len(vocablibrarywithoutduplicates)):
            PWordsinclusters[l][j]=PWordsinclusters[l][j]/sumall                           
## objective function(maximum liklihood)
    zbank=[];
    for l in range(numberofclusters):
        zscore=1
        for j in range(len(vocablibrarywithoutduplicates)):
            for i in range(3):
                zscore=zscore*PWordsinclusters[l][j]**countofwords[i][j] 
        zbank.append(clusterprobobility[l]*zscore)
    Z.append(sum(zbank))
## convergence criteria
    if numerator>=1:
        if Z[numerator-1]-Z[numerator]<0.00001 :
            break;       
    numerator+=1                    

                       
        
        