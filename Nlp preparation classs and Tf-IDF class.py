# In[]: Class TF_IDF
class TF_IDF:   
    def __init__(self,Docs= { "Doc1":["hi", "my","dear","brother"], "Doc2": ["hello","world", "hello", "this", "is"]}, doc=["hello","World", "Hello", "this", "is"], word="Hello"):
        self.Docs = Docs 
        self.word = word.lower()
        self.doc = doc
# TF : frequency of a word in a document        
    def Cal_TF(self, *args):
        if len(args)==2:
            TF= args[0].count(args[1])
        elif len(args)==0:
            TF = self.doc.count(self.word)
        print("There are {1} words in the Document".format('1',TF))
        return TF
# IDF
    def Cal_IDF(self, *args):
        import numpy as np
        if len(args)==0:
            Docs = self.Docs
            word = self.word
        else:
            Docs = args[0]
            word = args[1]
        Count = 0
        for Doc in Docs:
            if Doc.count(word)>0:
                Count = Count + 1
        IDF = np.log(len(Docs)/(Count+1))
        return(IDF)
# Calculating TF_IDF
    def Cal_TF_IDF(self, *args):
        if len(args)==2:
            return args[0]*args[1]
        else:
            return self.Cal_IDF()*self.Cal_TF()
#ob = TF_IDF()
#print(ob.Cal_TF())        
#print(ob.Cal_TF(["hi","my","brother"],"hi"))        
#print(ob.Cal_IDF())
#print(ob.Cal_TF_IDF())
#ob2 = TF_IDF(doc = TextStem, word = "fresh")
#ob2.Cal_TF()
#ob2.Cal_IDF()

# In[]: Class data prerparation for NLP
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
        Ps = stem.PorterStemmer()
        Ss = stem.SnowballStemmer('english')
        Cleaned_Stem = [Ps.stem(spell(word)) for word in self.Remove_Punc_Stopwords()]
        return Cleaned_Stem
# In[]: Checking the performance of the class with an object      
object1 = Nlp_Prep("This is a test For checking the classes!")
print(object1.Tokenize())
print(object1.Remove_Punc_Stopwords())
print(object1.Stem())
 
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
# In[]: Checking the performance of the class with an object
object2 = Bag_Words(["hi", "my","dear","brother"],["This","is"])
print(object2.Bag_of_Words)
object2.Add_Words(["hello","world", "hello", "dear", "brother"])
print(object2.Bag_of_Words) 
print(object2.Find_Word2Vec(["hello","world","world"]))     
print(object2.Find_Word2Vec(["hello","world","world"],Normalized=True)) 
# In[]: Mixing the two classes
MixedObject =  Nlp_Prep("This is a test For checking the classes!")
MixedObject =  Bag_Words(MixedObject.Stem())       
print(MixedObject.Bag_of_Words)

# In[]: Reading all text files from a directory
import glob
import os

os.chdir(r'C:\Users\Admin\Desktop\DM')
my_files = glob.glob('*.txt')
print(my_files)

text = []
for i in range(len(my_files)): 
    f = open(my_files[i], 'r')
    text.append(f.read())
# In[]: Using the classes for our text files
Stems = []
for i in range(len(text)):
    Preparation = Nlp_Prep(text[i])
    Stems.append(Preparation.Stem())
Bag = Bag_Words()    
for i in range(len(Stems)):
    Bag.Add_Words(Stems[i]) 
    
print(Bag.Bag_of_Words)   

Vec = []
for i in range(len(Stems)):
    Vec.append(Bag.Find_Word2Vec(Stems[i]))
VecNormalized = []
for i in range(len(Stems)):
    VecNormalized.append(Bag.Find_Word2Vec(Stems[i],Normalized = True))
