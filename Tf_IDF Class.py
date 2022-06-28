# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:48:19 2022

@author: Admin
"""
class Nlp_Prep:
    def __init__(self,string = "Hello world"):
        self.string = string.lower()
# In[]: Tokenization
    def Tokenize(self,*args):
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
        
# In[]: TF-IDF Algorithm
class Tf_idf:
    def __init__(self, *Docs):
        self.docs = []
        self.bag = []
        for i in range(len(Docs)):
            self.docs.append(Docs[i])
    def Bag_Word(self, *text):
        for i in range(len(text)):
            self.docs.append(text[i])
        BagObject = Bag_Words()
        for document in range(len(self.docs)):        
            BagObject = BagObject.Add_Words(self.docs[document])    
        
        self.bag = BagObject.Bag_of_Words        
    def Prepration(self,text):
        PreObject = Nlp_Prep(text)
        stem = PreObject.Stem()
        return stem
            
        
object1 = Tf_idf("Hello World!","Hi python.")
object1.Prepration(object1.docs[1])           
object1.Bag_Word()
object1.bag
