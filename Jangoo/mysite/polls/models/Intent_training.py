#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Dark-Sied/Intent_Classification/blob/master/Intent_classification_final.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint


# In[2]:


def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
  print(df.head())
  intent = df["Intent"]
  unique_intent = list(set(intent))
  sentences = list(df["Sentence"])
  
  return (intent, unique_intent, sentences)
  


# In[3]:


intent, unique_intent, sentences = load_dataset("Dataset.csv")


# In[4]:


print(sentences[:5])


# In[5]:


nltk.download("stopwords")
nltk.download("punkt")


# In[6]:


#define stemmer
stemmer = LancasterStemmer()


# In[7]:


def cleaning(sentences):
  words = []
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    #stemming
    words.append([i.lower() for i in w])
    
  return words  


# In[8]:


cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])  
  


# In[9]:


def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token


# In[10]:


def max_length(words):
  return(len(max(words, key = len)))
  


# In[11]:


word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))


# In[12]:


def encoding_doc(token, words):
  return(token.texts_to_sequences(words))


# In[13]:


encoded_doc = encoding_doc(word_tokenizer, cleaned_words)


# In[14]:


def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))


# In[15]:


padded_doc = padding_doc(encoded_doc, max_length)


# In[16]:


padded_doc[:5]


# In[17]:


print("Shape of padded docs = ",padded_doc.shape)


# In[18]:


#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')


# In[19]:


output_tokenizer.word_index


# In[20]:


encoded_output = encoding_doc(output_tokenizer, intent)


# In[21]:


encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)


# In[22]:


encoded_output.shape


# In[23]:


def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))


# In[24]:


output_one_hot = one_hot(encoded_output)


# In[25]:


output_one_hot.shape


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)


# In[28]:


print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


# In[32]:


#def create_model(vocab_size, max_length):
  #model = Sequential()
  #model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  #model.add(Bidirectional(LSTM(128)))
#   model.add(LSTM(128))
  #model.add(Dense(32, activation = "relu"))
  #model.add(Dropout(0.5))
  #model.add(Dense(22, activation = "softmax"))
  
  #return model


# In[33]:


#model = create_model(vocab_size, max_length)

#model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
#model.summary()


# In[35]:


filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])

m#odel.save("model.h5")
# In[ ]:




