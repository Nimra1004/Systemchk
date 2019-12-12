# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:55:56 2019

@author: HP
"""
# In[1]
import predict
import content
import collaborative
import Generate
import nltk
from nltk.tokenize import word_tokenize
import Intent_training


# In[4]
Predictmodel = predict.load_model("model.h5")

# In[5]
GenModel = Generate.load_model("GenerativeModel.h5")

# In[6]
def showrecommendation(item_id, num):
    products = collaborative.recommend(item_id, num)
    for  p in products:
        print (Generate.generate_text(p , 5, GenModel, Generate.max_seq_len))

    return

# In[7]
text = "do uou have this cap"
pred = predict.predictions(text, Predictmodel)
predict.get_final_output(pred, Intent_training.unique_intent)

# In[8]
tags = nltk.pos_tag(word_tokenize(text))
for t in tags:
    if t[1] == 'NN':
        entity = t[0]
print(entity)

# In[9]
print(pred)
print(Intent_training.unique_intent)

# In[8]
if (Intent_training.unique_intent[11] == 'faq.recommend'):
    if (pred[1][19] *100 > 20):
        showrecommendation()