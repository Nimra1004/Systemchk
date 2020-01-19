# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:55:56 2019

@author: HP
"""
# In[1]
from . import nlu
from . import nlg
from . import predict
from . import content
from . import collaborative
from . import Generate
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf



# In[4]
global graph
graph = tf.get_default_graph()
Predictmodel = predict.load_model("polls/model.h5")

# In[5]
GenModel = Generate.load_model("polls/GenerativeModel.h5")

# In[6]
#out.append(str(Generate.generate_text(t , 5, GenModel, nlg.max_seq_len))))
def showCollaborative(num):
    products = collaborative.UBCF(num)
    out = []
    for  p in products:
        print (Generate.generate_text(p , 5, GenModel, nlg.max_seq_len))

    return

# In[7]

def showContent(item_id, num):
    out = []
    products = content.recommend(item_id, num)
    for  p in products:
         out.append(str(Generate.generate_text(p , 5, GenModel, nlg.max_seq_len)))
       

    return out

# In[8]
text_query = "do u have some shirt?"
with graph.as_default():
    pred = predict.predictions(text_query, Predictmodel)
    predict.get_final_output(pred, nlu.unique_intent)

# In[8]
tags = nltk.pos_tag(word_tokenize(text_query))
print(tags)
for t in tags:
    if t[1] == 'NN':
        entity = t[0]
    elif t[1] == 'NNS':
        entity = t[0]
print(entity)

# In[9]
#print(pred)
#print(nlu.unique_intent)

# In[10]
id1 = nlu.output_tokenizer.word_index['faq.recommend']
id2 = nlu.output_tokenizer.word_index['commonq.assist']
thresh1 = pred[0][id1] * 100
thresh2 = pred[0][id2] * 100
print(thresh1)
print(thresh2)
# In[8]
if (nlu.unique_intent[11] == 'faq.recommend'):
    if (thresh1 > 3):
        showContent(34, 5)
    elif ( thresh1 < 5 & thresh2 > 2):
        showCollaborative(5)
                
# In[90]
#showCollaborative(5)
#sadia = showContent(34, 5)
# In[91]
sadia = showCollaborative(5)
print(sadia)