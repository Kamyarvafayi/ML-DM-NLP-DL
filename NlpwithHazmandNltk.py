# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:45:37 2022

@author: ASUS
"""
import nltk.tokenize as token
import nltk
from nltk.corpus import treebank
nltk.download('wordnet')
nltk. download('stopwords')
nltk.download('treebank')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
# In[]:
paragraph = "Hi! this is a test for textt mining tokenization. be patient and ambitious and just continue the courses for 3.5 hours. your abulity is beyond people's expectations. lets play"
# In[]: sentence tokenization
sentoken = token.sent_tokenize(paragraph)
# In[]: Word Tokenization
wordtoken = token.word_tokenize(paragraph, language='english')
wordpunct = token.wordpunct_tokenize(paragraph)
wordcasual = token.casual_tokenize(paragraph)
wordtree = token.TreebankWordTokenizer().tokenize(paragraph)
# In[]: removing punctuations (like . ! ? ...) and stopwords (like am is are and ...)
from nltk.corpus import stopwords
stop_words2 = stopwords.words()
stop_words = stopwords.words('english')

from string import punctuation
punctuation = list(punctuation)

tokenwithoutstopandpunc = [token for token in wordcasual  if token not in stop_words 

                  and token not in punctuation]
tokenwithoutstopandpunc2 = [token for token in wordtree  if token not in stop_words 

                  and token not in punctuation]
# In[]: spell correction
from autocorrect import spell
# in the paragraph text and ability are misspelled
# In[]: stemming and lemmazation
import nltk.stem as stem
from nltk import WordNetLemmatizer
ps = stem.PorterStemmer()
ss = stem.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
wordstem1 = [ps.stem(spell(word)) for word in tokenwithoutstopandpunc]
wordstem2 = [ss.stem(spell(word)) for word in tokenwithoutstopandpunc]
# lemmatizer can also be used for finding other forms of a word
wordstem3 = [lemmatizer.lemmatize(spell(word), 'a') for word in tokenwithoutstopandpunc]
print(wordstem3)

# lemmatizer can also be used for finding other forms of a word
print(lemmatizer.lemmatize('ran', 'v'))
print(lemmatizer.lemmatize('better', 'a'))

# In[]: checking the spell of a word using pyspellchecker library install this package using pip install pyspellchecker
from spellchecker import SpellChecker
spellcheck = SpellChecker()
print(spellcheck.correction("boook"))
print(spellcheck.candidates("boook"))

# In[]: Tagging words in a text (find nouns, verbs, adj, adv and etc)
tagged_text=nltk.pos_tag(wordcasual)
print(tagged_text)
tagged_text2=nltk.pos_tag(wordstem3)
print(tagged_text2)
tagged_text3=nltk.pos_tag(tokenwithoutstopandpunc)
print(tagged_text3)

# In[]: Hazm
from __future__ import unicode_literals
from hazm import *
import hazm
print(hazm.word_tokenize("سلام بر تو اي برادر! از كجا آمده اي؟"))



normalizer = Normalizer()
normalizer.normalize('اصلاح نويسه ها و استفاده از نیم‌فاصله پردازش را آسان مي كند')

sent_tokenize('ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟')

word_tokenize('ولی برای پردازش، جدا بهتر نیست؟')


stemmer = Stemmer()
stemmer.stem('برويم')

lemmatizer = Lemmatizer()
lemmatizer.lemmatize('می‌روم')
