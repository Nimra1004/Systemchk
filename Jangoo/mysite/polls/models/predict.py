# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:45:00 2019

@author: HP
"""
# In[1] 
from keras.models import Sequential, load_model
from numpy import load
import Intent_training
from nltk.tokenize import word_tokenize
import re


# In[2]:
def predictions(text, model):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = Intent_training.word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = Intent_training.word_tokenizer.texts_to_sequences(test_word)
  print(test_word)
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = Intent_training.np.array(test_ls).reshape(1, len(test_ls))
 
  x = Intent_training.padding_doc(test_ls, Intent_training.max_length)
  
  pred = model.predict_proba(x)
  return pred

# In[3]:
def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = Intent_training.np.array(classes)
  ids = Intent_training.np.argsort(-predictions)
  classes = classes[ids]
  predictions = -Intent_training.np.sort(-predictions)
 
  for i in range(pred.shape[1]):
    print("%s has confidence = %s" % (classes[i], (predictions[i]*100)))

# In[4]