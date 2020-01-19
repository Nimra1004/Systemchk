# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:24:07 2019

@author: HP
"""
# In[1] 
from keras.models import Sequential, load_model
from numpy import load
from . import nlg

# In[2]
def generate_text(seed_text, next_words, model, max_seq_len):
    w=0
    for _ in range(next_words):
        token_list = nlg.tokenizer.texts_to_sequences([seed_text])[0]
        
        if w==0:
            token_list
            
        token_list = nlg.pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        
        if w==0:
            token_list
        
        predicted = model.predict_classes(token_list, verbose=0)
        
        if w==0:
            predicted
        
        output_word = ''
        for word,index in nlg.tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
                
        seed_text = seed_text + " " + output_word
        
        w=1
        
    return seed_text.title()

# In[ ]:




