import pandas as pd
import numpy as np
import ast

from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

train_input="dataset/topic/train.csv"
test_input="dataset/topic/valid.csv"
topic_input="dataset/topic/topics.txt"
voc_input="dataset/topic/vocabulary.txt"

def load_class():
    CLASSES = set()
    with open(topic_input, 'r', encoding="utf-8") as f:
        for line in f:
            CLASSES.add(str(line[:-1]))
    return list(CLASSES)

CLASSES = load_class()


def load_voc():
    with open(voc_input, "r", encoding="utf-8") as file:
        voc = ast.literal_eval(file.readline())
    return voc


def topic_to_string(nb):
    return CLASSES[nb]

voc=load_voc()
voc = dict(map(reversed, voc.items()))

def text_to_string(text):
    text=text[1:-1]
    sentence = ""
    for word in text.split(', '):
        sentence+=voc[int(word)]
        sentence+=" "
    return sentence

train = pd.read_csv(train_input,index_col=0,nrows=100)
train.shape

test = pd.read_csv(test_input,index_col=0,nrows=10)
test.shape

train['topic_name']=train['topic'].apply(topic_to_string)
train['text']=train['text'].apply(text_to_string)
test['topic_name']=test['topic'].apply(topic_to_string)
test['text']=test['text'].apply(text_to_string)

tfidf = TfidfVectorizer(sublinear_tf=True,
                        ngram_range=(1, 2))



x_train, x_test, y_train, y_test,indices_train,indices_test = train_test_split(tfidf.fit_transform(train.text).toarray(), 
                                                               train['topic'], 
                                                               train.index, test_size=0.25, 
                                                               random_state=1)

model = LinearSVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("fin")
