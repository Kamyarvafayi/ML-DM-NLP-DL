# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:57:01 2022

@author: Admin
"""

class Nlp_Prep:
    def __init__(self,string = "Hello world"):
        self.string = string.lower()
# In[]: Tokenization
    def Tokenize(self,*args:str):
        import nltk.tokenize as token
        if len(args) == 1:
            return token.casual_tokenize(args)
        else:
            return token.casual_tokenize(self.string)
# In[]: removing punctuations and stop words
    def Remove_Punc_Stopwords(self):
        from string import punctuation
        from nltk.corpus import stopwords
        punctuation = list(punctuation)
        stop_words = stopwords.words("english")
        cleanedToken = [word for word in self.Tokenize() if word not in punctuation and word not in stop_words]
        return cleanedToken
# In[]: Stem and spelling check
    def Stem(self):
        import nltk.stem as stem
        from autocorrect import spell
        from nltk import WordNetLemmatizer
        Ps = stem.PorterStemmer()
        Ss = stem.SnowballStemmer('english')
        lemmatizer = WordNetLemmatizer()
        Cleaned_Stem = [Ss.stem(spell(word)) for word in self.Remove_Punc_Stopwords()]
        #Cleaned_Stem = [lemmatizer.lemmatize(spell(word),"a") for word in self.Remove_Punc_Stopwords()]
        return Cleaned_Stem  
# In[]: Creating Bag of words
class Bag_Words:
    def __init__(self,*args):
        self.Bag_of_Words = []
        for i in range(len(args)):
            for Word in args[i]:
                if Word not in self.Bag_of_Words:
                    self.Bag_of_Words.append(Word)    
    def Add_Words(self,New_Words = []):
        for Word in New_Words:
            if Word not in self.Bag_of_Words:
                self.Bag_of_Words.append(Word)
    def Find_Word2Vec(self, words, Normalized = False):
        #self.Add_Words(words)
        Vector = []
        for i in range(len(self.Bag_of_Words)):
            Vector.append(words.count(self.Bag_of_Words[i]))
        if Normalized:
            Normalized_Vec = [Vector[i]/len(words) for i in range(len(self.Bag_of_Words))]
            return Normalized_Vec
        else:
            return Vector
# In[]: Importing test and Train Data
import glob
import os
my_files = []
os.chdir(r'C:\Users\Admin\Desktop\DM\sentiment analysis\aclImdb\train\pos')
my_files.append(glob.glob('*.txt'))
print(my_files)

Train_Pos = []
for i in range(len(my_files[0])): 
    f = open(my_files[0][i], 'r')
    Train_Pos.append(f.read())
    
os.chdir(r'C:\Users\Admin\Desktop\DM\sentiment analysis\aclImdb\train\neg')
my_files.append(glob.glob('*.txt'))
print(my_files)

Train_Neg = []
for i in range(len(my_files[1])): 
    f = open(my_files[1][i], 'r')
    Train_Neg.append(f.read())
    
os.chdir(r'C:\Users\Admin\Desktop\DM\sentiment analysis\aclImdb\test\neg')
my_files.append(glob.glob('*.txt'))
print(my_files)

Test_Neg = []
for i in range(len(my_files[2])): 
    f = open(my_files[2][i], 'r')
    Test_Neg.append(f.read())

os.chdir(r'C:\Users\Admin\Desktop\DM\sentiment analysis\aclImdb\test\pos')
my_files.append(glob.glob('*.txt'))
print(my_files)

Test_Pos = []
for i in range(len(my_files[3])): 
    f = open(my_files[3][i], 'r')
    Test_Pos.append(f.read())
# In[]: preparing the texts for classification
Data_Number = 500
Train_Data_Pos = []
Test_Data_Pos = []
Train_Data_Neg = []
Test_Data_Neg = []

for i in range(Data_Number):
    TrainPosobj = Nlp_Prep(Train_Pos[i])
    Train_Data_Pos.append(TrainPosobj.Stem())
    TrainNegobj = Nlp_Prep(Train_Neg[i])
    Train_Data_Neg.append(TrainNegobj.Stem())
    TestNegobj = Nlp_Prep(Test_Neg[i])
    Test_Data_Neg.append(TestNegobj.Stem())
    TestPosobj = Nlp_Prep(Test_Pos[i])
    Test_Data_Pos.append(TestPosobj.Stem())
# In[]: Bag_Of_Words
Bag = Bag_Words()
for i in range(Data_Number):
    Bag.Add_Words(Train_Data_Pos[i])
    Bag.Add_Words(Train_Data_Neg[i])
    Bag.Add_Words(Test_Data_Pos[i])
    Bag.Add_Words(Test_Data_Neg[i])
Bag_of_Words = Bag.Bag_of_Words
# In[]: Word2Vec
Train_Pos_Vec = []
Train_Neg_Vec = []
Test_Pos_Vec = []
Test_Neg_Vec = []

for i in range(Data_Number):
    Train_Pos_Vec.append(Bag.Find_Word2Vec(Train_Data_Pos[i]))
    Train_Neg_Vec.append(Bag.Find_Word2Vec(Train_Data_Neg[i]))
    Test_Pos_Vec.append(Bag.Find_Word2Vec(Test_Data_Pos[i]))
    Test_Neg_Vec.append(Bag.Find_Word2Vec(Test_Data_Neg[i]))

# In[]: Arrays
import numpy as np
Train_Pos_array = np.array(Train_Pos_Vec)
Train_Neg_array = np.array(Train_Neg_Vec)
Test_Pos_array = np.array(Test_Pos_Vec)
Test_Neg_array = np.array(Test_Neg_Vec)

Train_Data = np.append(Train_Pos_array,Train_Neg_array,axis = 0)
Test_Data = np.append(Test_Pos_array,Test_Neg_array,axis = 0)
Final_Train_Data = Train_Data[:,np.sum(Train_Data,axis=0)>50]
Final_Test_Data = Test_Data[:,np.sum(Train_Data,axis=0)>50]
# Targets
Train_Class = np.zeros([2*Data_Number,1])
Train_Class[Data_Number:] = Train_Class[Data_Number:] + 1
Test_Class = np.zeros([Data_Number,1])
Test_Class[Data_Number:] = Test_Class[Data_Number:] + 1
# In[]: Pca
import sklearn.decomposition as decompose
PCA = decompose.PCA(n_components = 50)
Pca = PCA.fit(Train_Data)
Pca_TrainData = Pca.transform(Train_Data)

Pca = PCA.fit(Test_Data)
Pca_TestData = Pca.transform(Test_Data)
# In[]: KernelPca
KernelPca = decompose.KernelPCA(kernel = "rbf", n_components=20)
KPca = KernelPca.fit(Train_Data)
Kpca_Train_Data = KPca.transform(Train_Data)
# In[]: KNN classifier
import sklearn.neighbors as cl
KNN = cl.KNeighborsClassifier(n_neighbors = 5)
Knn = KNN.fit(Pca_TrainData,Train_Class.reshape(900,))
PredictKnn = Knn.predict(Pca_TestData)
# In[]: SVM
import sklearn.svm as svm
SVM = svm.SVC(kernel="rbf")
SVM2 = svm.SVC(kernel="rbf")
Svm = SVM.fit(Pca_TrainData,Train_Class.reshape(900,))
PredictSvm = Svm.predict(Pca_TestData)

Svm2 = SVM2.fit(Final_Train_Data,Train_Class.reshape(1000,))
PredictSvm2 = Svm2.predict(Final_Test_Data)
print(np.sum(PredictSvm2[0:500]))
print(np.sum(PredictSvm2[500:1000]))
# In[]: Naive Bayes
import sklearn.naive_bayes as naive
NB = naive.GaussianNB()
NBClassifier = NB.fit(Train_Data,Train_Class.reshape(1000,))
NBClassifier2 = NB.fit(Final_Train_Data,Train_Class.reshape(1000,))
PredictNB = NBClassifier.predict(Test_Data)
PredictNB2 = NBClassifier2.predict(Final_Test_Data)
print(np.sum(PredictNB2[0:500]))
print(np.sum(PredictNB2[500:1000]))
# In[]: Kmeans for clustering
from sklearn.cluster import KMeans
Kmeans1 = KMeans(n_clusters = 2)
Kmeans2 = KMeans(n_clusters = 2)
Kmeans3 = KMeans(n_clusters = 5)
kmeans1 = Kmeans1.fit(Final_Train_Data)
kmeans2 = Kmeans2.fit(Pca_TrainData)
kmeans3 = Kmeans3.fit(Kpca_Train_Data)
print( kmeans1.labels_ )
print( kmeans2.labels_ )
print( kmeans3.labels_ )
