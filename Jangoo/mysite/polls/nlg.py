# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:36:47 2019

@author: HP
"""

import numpy as np # linear algebra
import pandas as pd 
import os
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils as ku
from keras.callbacks import EarlyStopping

# set seeds for reproducability

from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import string
import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore',  category=FutureWarning)


# In[2]:


curr_dir = "C:/Users/HP/Desktop/input/"
all_headlines = []

x=0

for filename in os.listdir(curr_dir):
    if 'generativeData' in filename:
        
        if x==0:
            print(filename)
            
        article_df = pd.read_csv(curr_dir + filename)
        
        if x==0:
            print(article_df.shape)
            print(article_df.columns)
            print(article_df.head(5))
            print(article_df.tail(5))
            
        all_headlines.extend(list(article_df.text.values))
        
        if x==0:
            print(article_df.text)
            print(article_df.text.values)
            
        x=1
        break

all_headlines = [ h for h in all_headlines if h != 'Unknown' ]
print(len(all_headlines))
print(all_headlines[:5])


# In[3]:


def clean_text(txt):
   # txt = text.tostring()
    txt = "".join(w for w in txt if w not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii","ignore")
    return txt

# In[4]:
#clean_text(series)




# In[5]:
print(clean_text("Questions for: ‘Colleges Discover the Rural St.."))


print(string.punctuation)


# In[6]:


corpus = [clean_text(x) for x in all_headlines]
print(corpus[:10])


# In[7]:


tokenizer = Tokenizer()
def get_sequence_of_tokens(corpus):
    q=0
    # tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    if q==0:
        print(total_words)
    
    # convert data into sequence of tokens
    input_sequences = []
    for line in corpus:
        
        if q==0:
            print(line)
            
        token_list = tokenizer.texts_to_sequences([line])[0]
        
        if q==0:
            print(token_list)
            print(len(token_list))
            
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            if q==0:
                print(n_gram_sequence)
                
            input_sequences.append(n_gram_sequence)
            
            if q==0:
                print(input_sequences)
                
            q=1
    
    return input_sequences, total_words


# In[8]:


input_sequence, total_words = get_sequence_of_tokens(corpus)
print(total_words)


# In[9]:


(input_sequence[:10])


# In[10]:


def generate_padded_sequences(input_sequence):
    max_sequence_len = max([len(x) for x in input_sequence])
    input_sequences = np.array(pad_sequences(input_sequence, maxlen=max_sequence_len, padding='pre'))
    predictors, labels = input_sequences[:,:-1], input_sequences[:,-1]
    labels = ku.to_categorical(labels, num_classes=total_words)
    
    return predictors, labels, max_sequence_len


# In[11]:


predictors, labels, max_seq_len = generate_padded_sequences(input_sequence)


# In[12]:


len(predictors)


# In[13]:


len(labels)


# In[14]:


(max_seq_len)


# In[15]:


print(total_words)

