import pandas as pd
#!pip install imblearn
import matplotlib.pyplot as plt
#import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold 
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences        
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
from tensorflow import keras
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
estopwords = set(stopwords.words('english')).union(["thats","weve","dont","lets","youre","im","thi","ha",\
    "wa","st","ask","want","like","thank","know","susan","ryan","say","got","ought","ive","theyre"])  #get the English stop words

#only use text to train the model. 

def prepro(typedata):
    data = pd.read_csv('./data/'+typedata+'_variants')
    data_text =pd.read_csv("./data/"+typedata+"_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
    data_text.head()


    df = pd.merge(data, data_text,on='ID', how='left')
    df = df.dropna()
    df = df.reset_index(drop=True)
    for colname in ['TEXT','Gene','Variation']:
        
        df[colname] = df[colname].apply(lambda r: BeautifulSoup(r, 'html.parser').get_text())
        if colname=='TEXT':
            df[colname].replace(to_replace ='[^a-z A-Z]', value = '', regex = True, inplace = True)
        df[colname] = df[colname].str.lower()  #convert to lower case
    return df
df_train=prepro('training')
df_train['Class'].value_counts().plot(kind="bar", rot=0)

df_train['TEXT'] = df_train['TEXT'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in set(x.split()) if word not in estopwords]))

MAX_VOCABS = 5000
tokenizer = Tokenizer(num_words = MAX_VOCABS)
tokenizer.fit_on_texts(pd.concat([df_train['TEXT']]))
x_train = tokenizer.texts_to_sequences(df_train['TEXT'])

MAX_LEN = max([len(i) for i in x_train])
vocab_size = MAX_VOCABS + 1
x_train = pad_sequences(x_train, padding='post', maxlen=MAX_LEN, value=vocab_size)

# convert integers to dummy variables (i.e. one hot encoded)
y_train = pd.get_dummies(df_train['Class']).values
dummy_columns=pd.get_dummies(df_train['Class']).columns
dummy_columns=dummy_columns.tolist
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)

embedd_dim = 256
model = keras.Sequential([keras.layers.Embedding(vocab_size + 1, embedd_dim), keras.layers.GlobalAveragePooling1D(),\
                          keras.layers.Dense(128, activation='relu'), keras.layers.Dense(9, activation='softmax')])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_crossentropy'])
history = model.fit(x_train, y_train, epochs=75,  batch_size=32, verbose = 2, validation_data = (x_val, y_val))


#read test data
df_test=prepro('test')
df_test['TEXT'] = df_test['TEXT'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in set(x.split()) if word not in estopwords]))
x_test = tokenizer.texts_to_sequences(df_test['TEXT'])
x_test = pad_sequences(x_test, padding='post', maxlen=MAX_LEN, value=vocab_size)

# convert integers to dummy variables (i.e. one hot encoded)
y_test_predict_prob=model.predict(x_test)
y_test_predict=np.zeros(len(x_test))
for i in range(0,len(x_test)):
    y_test_predict[i]=np.argmax(y_test_predict_prob[i,:])+1
print('accuracy score for test data is ',accuracy_score(y_test_predict,df_test['Class']))
#0.6566757493188011    
print('f1 score for test data is ',f1_score(y_test_predict,df_test['Class'],average=None))
#f1 score for test data is  [0.69892473 0.43589744 0.         0.61654135 0.57692308 0.66666667
# 0.77419355 0.         0.8       ]
print('f1 score for test data is ',f1_score(y_test_predict,df_test['Class'],average='weighted'))
#f1 score for test data is  0.6662866776748431



#So far, we haven't used the gene mutation information, I am going to create a new variable by combining the variant and text

df_train['TEXT1']=df_train['Gene']+' '+df_train['Variation']+df_train['TEXT']
df_test['TEXT1']=df_test['Gene']+' '+df_test['Variation']+df_test['TEXT']

MAX_VOCABS = 5000
tokenizer = Tokenizer(num_words = MAX_VOCABS)
tokenizer.fit_on_texts(pd.concat([df_train['TEXT1']]))
x_train = tokenizer.texts_to_sequences(df_train['TEXT1'])

MAX_LEN = max([len(i) for i in x_train])
vocab_size = MAX_VOCABS + 1
x_train = pad_sequences(x_train, padding='post', maxlen=MAX_LEN, value=vocab_size)

# convert integers to dummy variables (i.e. one hot encoded)
y_train = pd.get_dummies(df_train['Class']).values
dummy_columns=pd.get_dummies(df_train['Class']).columns
dummy_columns=dummy_columns.tolist
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)

embedd_dim = 128
model = keras.Sequential([keras.layers.Embedding(vocab_size + 1, embedd_dim), keras.layers.LSTM(64, recurrent_dropout=0.2, dropout=0.2),\
                           keras.layers.Dense(9, activation='softmax')])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_crossentropy'])
history = model.fit(x_train, y_train, epochs=75,  batch_size=32, verbose = 2, validation_data = (x_val, y_val))


#read test data
df_test['TEXT1'] = df_test['TEXT1'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in set(x.split()) if word not in estopwords]))
x_test = tokenizer.texts_to_sequences(df_test['TEXT1'])
x_test = pad_sequences(x_test, padding='post', maxlen=MAX_LEN, value=vocab_size)

# convert integers to dummy variables (i.e. one hot encoded)
y_test_predict_prob=model.predict(x_test)
y_test_predict=np.zeros(len(x_test))
for i in range(0,len(x_test)):
    y_test_predict[i]=np.argmax(y_test_predict_prob[i,:])+1
print('accuracy score for test data is ',accuracy_score(y_test_predict,df_test['Class']))
# 
print('f1 score for test data is ',f1_score(y_test_predict,df_test['Class'],average=None))
#
# 
print('f1 score for test data is ',f1_score(y_test_predict,df_test['Class'],average='weighted'))
#f1 score for test data is 






















