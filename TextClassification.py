import nltk.tokenize as token
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
# In[]:
Doc1 = "If you were to ask me what my favourite sport is, my answer would be swimming. I started learning how to swim when I was five years old and I have been swimming ever since. There are many reasons why I love swimming but I will just share a few with you, and maybe I can even encourage you to go for a swim. The first reason that I love swimming is that it can be extremely relaxing. I love the feeling of floating on the water and feeling almost weightless. I find that whenever I leave the pool I feel totally relaxed. On the other hand, it can also be a fast-paced competitive sport which raises your heartbeat. It is amazing pushing yourself to the challenge of beating someone else to the finish line and its great fun racing across the pool as fast as you can! Trying little tricks like underwater handstands and flips also puts a big smile on my face. Another reason that I love swimming is that it has more variety than other sports. Swimming offers many different strokes, so it doesn’t feel like you are stuck doing the same thing over and over again. Adding swimming equipment like flippers, a snorkel or a noodle (a flexible cylindrical piece of foam) changes the experience yet again and can help you develop your swimming technique."
Doc1 = Doc1.lower()
Doc2 = "Swimming is the self-propulsion of a person through water, or other liquid, usually for recreation, sport, exercise, or survival. Locomotion is achieved through coordinated movement of the limbs and the body to achieve hydrodynamic thrust which results in directional motion. Humans can hold their breath underwater and undertake rudimentary locomotive swimming within weeks of birth, as a survival response. Swimming is consistently among the top public recreational activities, and in some countries, swimming lessons are a compulsory part of the educational curriculum.[6] As a formalized sport, swimming is featured in a range of local, national, and international competitions, including every modern Summer Olympics. Swimming involves repeated motions known as strokes in order to propel the body forward. While the front crawl is widely regarded as the fastest out of four primary strokes, other strokes are practiced for special purposes, such as for training."
Doc2 = Doc2.lower()
Doc3 = "Basketball is a team sport in which two teams, most commonly of five players each, opposing one another on a rectangular court, compete with the primary objective of shooting a basketball (approximately 9.4 inches (24 cm) in diameter) through the defender's hoop (a basket 18 inches (46 cm) in diameter mounted 10 feet (3.048 m) high to a backboard at each end of the court, while preventing the opposing team from shooting through their own hoop. A field goal is worth two points, unless made from behind the three-point line, when it is worth three. After a foul, timed play stops and the player fouled or designated to shoot a technical foul is given one, two or three one-point free throws. The team with the most points at the end of the game wins, but if regulation play expires with the score tied, an additional period of play (overtime) is mandated. Players advance the ball by bouncing it while walking or running (dribbling) or by passing it to a teammate, both of which require considerable skill. On offense, players may use a variety of shots – the layup, the jump shot, or a dunk; on defense, they may steal the ball from a dribbler, intercept passes, or block shots; either offense or defense may collect a rebound, that is, a missed shot that bounces from rim or backboard. It is a violation to lift or drag one's pivot foot without dribbling the ball, to carry it, or to hold the ball with both hands then resume dribbling."
Doc3 = Doc3.lower()
Doc4 = "A basketball ball is a spherical ball used in basketball games by basketball players. Basketballs usually range in size from very small promotional items that are only a few inches (some centimeters) in diameter to extra large balls nearly 2 feet (60 cm) in diameter used in training exercises. For example, a youth basketball could be 27 inches (69 cm) in circumference, while a National Collegiate Athletic Association (NCAA) men's ball would be a maximum of 30 inches (76 cm) and an NCAA women's ball would be a maximum of 29 inches (74 cm). The standard for a basketball in the National Basketball Association (NBA) is 29.5 inches (75 cm) in circumference and for the Women's National Basketball Association (WNBA), a maximum circumference of 29 inches (74 cm). High school and junior leagues normally use NCAA, NBA or WNBA sized balls."
Doc4 = Doc4.lower()

Doc5test = "Basketball is a team sport.Basketball has a  Two teams of five players each try to score by shooting a ball through a hoop elevated 10 feet above the ground. The game is played on a rectangular floor called the court, and there is a hoop at each end. The court is divided into two main sections by the mid-court line. basketball is not easy."
Doc5test.lower()

Doc1Token = token.casual_tokenize(Doc1)
Doc2Token = token.casual_tokenize(Doc2)
Doc3Token = token.casual_tokenize(Doc3)
Doc4Token = token.casual_tokenize(Doc4)
Doc5Token = token.casual_tokenize(Doc5test)
def remove_punc_stopwords(Token):
    stop_words = stopwords.words('english')
    stop_words.append("would")
    stop_words.append("also")
    stop_words.append("yet")
    Punctuation_list = list(punctuation)
    Punctuation_list.append("’")
    TokenWitoutPuncstop = [i for i in Token if i not in stop_words and i not in Punctuation_list]
    return TokenWitoutPuncstop
Doc1WordsWitoutPuncstop = remove_punc_stopwords(Doc1Token)
Doc2WordsWitoutPuncstop = remove_punc_stopwords(Doc2Token)
Doc3WordsWitoutPuncstop = remove_punc_stopwords(Doc3Token)
Doc4WordsWitoutPuncstop = remove_punc_stopwords(Doc4Token)
Doc5WordsWitoutPuncstop = remove_punc_stopwords(Doc5Token)
# In[]: Stem of words and checking spelling
import nltk.stem as stem
from nltk import WordNetLemmatizer
from autocorrect import spell
ps = stem.PorterStemmer()
ss = stem.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
Doc1stem1= [ps.stem(spell(word)) for word in Doc1WordsWitoutPuncstop]
Doc1stem2 = [ss.stem(spell(word)) for word in Doc1WordsWitoutPuncstop]
Doc2stem1= [ps.stem(spell(word)) for word in Doc2WordsWitoutPuncstop]
Doc2stem2 = [ss.stem(spell(word)) for word in Doc2WordsWitoutPuncstop]
Doc3stem1= [ps.stem(spell(word)) for word in Doc3WordsWitoutPuncstop]
Doc3stem2 = [ss.stem(spell(word)) for word in Doc3WordsWitoutPuncstop]
Doc4stem1= [ps.stem(spell(word)) for word in Doc4WordsWitoutPuncstop]
Doc4stem2 = [ss.stem(spell(word)) for word in Doc4WordsWitoutPuncstop]
Doc5stem1= [ps.stem(spell(word)) for word in Doc5WordsWitoutPuncstop]
Doc5stem2 = [ss.stem(spell(word)) for word in Doc5WordsWitoutPuncstop]
# In[]: Making Bag of words
def create_bag_of_words(Stem, bag = []):
    for i in range(len(Stem)):
        if Stem[i] not in bag:
            bag.append(Stem[i])
    return bag
bag_of_words = []
bag_of_words = create_bag_of_words(Doc1stem1, bag_of_words)
bag_of_words = create_bag_of_words(Doc2stem1, bag_of_words)
bag_of_words = create_bag_of_words(Doc3stem1, bag_of_words)
bag_of_words = create_bag_of_words(Doc4stem1, bag_of_words)
bag_of_words = create_bag_of_words(Doc5stem1, bag_of_words)
# In[]: Word2Vec
def Word2Vec(Stem, bag):
    Vector = []
    for i in range(len(bag)):
        Vector.append(Stem.count(bag[i]))
    return Vector    
Doc1Vector = Word2Vec(Doc1stem1, bag_of_words)
Doc2Vector = Word2Vec(Doc2stem1, bag_of_words)
Doc3Vector = Word2Vec(Doc3stem1, bag_of_words)
Doc4Vector = Word2Vec(Doc4stem1, bag_of_words)
Doc5Vector = Word2Vec(Doc5stem1, bag_of_words)
# In[]: normalizing the vectors
Doc1VectorNormalized = [Doc1Vector[i]/len(Doc1stem1) for i in range(len(bag_of_words))]
Doc2VectorNormalized = [Doc2Vector[i]/len(Doc2stem1) for i in range(len(bag_of_words))]
Doc3VectorNormalized = [Doc3Vector[i]/len(Doc3stem1) for i in range(len(bag_of_words))]
Doc4VectorNormalized = [Doc4Vector[i]/len(Doc4stem1) for i in range(len(bag_of_words))]
Doc5VectorNormalized = [Doc5Vector[i]/len(Doc5stem1) for i in range(len(bag_of_words))]
# In[]: Distance Between 2 vectors
print("1 and 2: ", np.linalg.norm(np.array(Doc1VectorNormalized)-np.array(Doc2VectorNormalized)))
print("1 and 3: ",np.linalg.norm(np.array(Doc1VectorNormalized)-np.array(Doc3VectorNormalized)))
print("2 and 3: ",np.linalg.norm(np.array(Doc2VectorNormalized)-np.array(Doc3VectorNormalized)))
print("1 and 4: ",np.linalg.norm(np.array(Doc1VectorNormalized)-np.array(Doc4VectorNormalized)))
print("2 and 4: ",np.linalg.norm(np.array(Doc2VectorNormalized)-np.array(Doc4VectorNormalized)))
print("3 and 4: ", np.linalg.norm(np.array(Doc3VectorNormalized)-np.array(Doc4VectorNormalized)))
# In[]: storing vectors in an array
x = np.array(Doc1VectorNormalized).reshape(len(bag_of_words),1)
x = np.append(x,np.array(Doc2VectorNormalized).reshape(len(bag_of_words),1), axis=1)
x = np.append(x,np.array(Doc3VectorNormalized).reshape(len(bag_of_words),1), axis=1)
x = np.append(x,np.array(Doc4VectorNormalized).reshape(len(bag_of_words),1), axis=1)
# In[]: Kmeans
import sklearn.cluster as cl
kmeans = cl.KMeans(n_clusters=2)
kmeans.fit(x.transpose())
Centers = kmeans.cluster_centers_
lables = kmeans.labels_
print(kmeans.cluster_centers_)
# In[]
print(np.linalg.norm(np.array(Doc5VectorNormalized)-Centers[0]))
print(np.linalg.norm(np.array(Doc5VectorNormalized)-Centers[1]))

# In[]: reading a text about freshfood
f = open('FreshFood.txt', 'r')
text = f.read()
# In[]: preparation the text
TokenizedText = token.casual_tokenize(text.lower())
TextWitoutPuncstop = remove_punc_stopwords(TokenizedText)
TextStem = [ps.stem(word) for word in TextWitoutPuncstop]


# In[]: Class TF_IDF
class TF_IDF:  
    def __init__(self,Docs= {"Doc1":["hi", "my","dear","brother"], "Doc2": ["hello","world", "hello", "this", "is"]}, doc=["hello","World", "Hello", "this", "is"], word="Hello"):
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
ob = TF_IDF()
print(ob.Cal_TF())        
print(ob.Cal_TF(["hi","my","brother"],"hi"))        
print(ob.Cal_IDF())
print(ob.Cal_TF_IDF())

ob2 = TF_IDF(doc = TextStem, word = "fresh")
ob2.Cal_TF()
ob2.Cal_IDF()