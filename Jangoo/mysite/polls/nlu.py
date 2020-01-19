# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:34:28 2019

@author: HP
"""

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


intent, unique_intent, sentences = load_dataset("C:/Users/HP/Desktop/input/Dataset.csv")


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
