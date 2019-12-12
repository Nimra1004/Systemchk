#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:

curr_dir = "C:/Users/HP/Desktop/Jangoo/mysite/polls/models"
import pandas as pd
products = pd.read_csv("C:/Users/HP/Desktop/Jangoo/mysite/polls/models/JokeText.csv")
data = pd.read_csv("C:/Users/HP/Desktop/Jangoo/mysite/polls/models/UserRatings1.csv",index_col="productId")
data.head()


# In[3]:


data = data.iloc[:,:5000]
sums_of_columns = data.sum(axis=1)
columns_size = len(data.columns)
value = sums_of_columns/columns_size
index_of_max = value[value == value.max()].index[0]
print("The best product is index as {} and value of product is :{}".format(index_of_max,value.max()))


# In[4]:


from sklearn.metrics.pairwise import cosine_similarity
data = data.T
Filtering_cosim = cosine_similarity(data,data)


# In[5]:


most_sim_users = sorted(list(enumerate(Filtering_cosim[8])), key=lambda x: x[1], reverse=True)
most_sim_users = most_sim_users[1:11]
sim_users = [x[0] for x in most_sim_users]
print(sim_users)


# In[6]:


candidates_products = data.iloc[sim_users,:]


# In[7]:


def UBCF(user_num):
    ### finding most similar users among matrix

    most_sim_users = sorted(list(enumerate(Filtering_cosim[user_num])), key=lambda x: x[1], reverse=True)
    most_sim_users = most_sim_users[1:11]

    ### user index and their similairity values 

    sim_users = [x[0] for x in most_sim_users]
    sim_values = [x[1] for x in most_sim_users]

    ### among users having most similar preferences, finding movies having highest average score
    ### however except the movie that original user didn't see

    candidates_products = data.iloc[sim_users,:]

    candidates_products.mean(axis=0).head()

    mean_score = pd.Series(candidates_products.mean(axis=0))
    mean_score = mean_score.sort_values(axis=0, ascending=False)
    
    recom_products = list(mean_score.iloc[0:10].keys())
    for i in recom_products:
        print("Index Number {} and product is {} :".format(i,products.iloc[i,:]))
    return(recom_products)


UBCF(1)





