# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:45:00 2019

@author: HP
"""
# In[1] 
from keras.models import Sequential, load_model
from numpy import load
from . import nlu

# In[2]:
global graph
def predictions(text, model):
  clean = nlu.re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = nlu.word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = nlu.word_tokenizer.texts_to_sequences(test_word)
  print(test_word)
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = nlu.np.array(test_ls).reshape(1, len(test_ls))
 
  x = nlu.padding_doc(test_ls, nlu.max_length)
  pred = model.predict_proba(x)
  return pred

# In[3]:
def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = nlu.np.array(classes)
  ids = nlu.np.argsort(-predictions)
  classes = classes[ids]
  predictions = -nlu.np.sort(-predictions)
 
  for i in range(pred.shape[1]):
    print("%s has confidence = %s" % (classes[i], (predictions[i]*100)))

# In[4]