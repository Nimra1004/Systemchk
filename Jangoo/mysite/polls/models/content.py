#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ds = pd.read_csv("C:/Users/HP/Desktop/Jangoo/mysite/polls/models/ProductData.csv")


# In[8]:


#With Tfidfvectorizer you compute the word counts, idf and tf-idf values all at once
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
#fit_transform(x) it does the prepocessing of the data and the NaN values can be handled through this function returns Tf-idf-weighted document-term matrix.
tfidf_matrix = tf.fit_transform(ds['description']) 


# In[13]:


#cosine_similarity (dot product of the tfidfmatrix formed with the weight of tfidf)
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}

for idx, row in ds.iterrows(): #each index with the row to access the columns
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]
    results[row['id']] = similar_items[1:]
    
print('done')


# In[10]:


def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]


# In[62]:


def recommend(item_id, num):
    recs = results[item_id][:num]
    a = []
    for rec in recs:
        Final = item(rec[1])
        a.append(Final)
    return(a)


# In[63]:



# In[ ]:




