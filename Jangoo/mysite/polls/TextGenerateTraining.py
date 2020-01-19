#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nlg

# In[16]:


def create_model(max_seq_len, total_words):
    input_len = max_seq_len - 1
    
    model = nlg.Sequential()
    
    # input: embedding layer
    model.add(nlg.Embedding(total_words, 10, input_length=input_len))
    
    # hidden: lstm layer
    model.add(nlg.LSTM(100))
    model.add(nlg.Dropout(0.1))
    
    # output layer
    model.add(nlg.Dense(total_words, activation='softmax'))
    
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model


# In[17]:


model = create_model(nlg.max_seq_len, nlg.total_words)
model.summary()


# In[18]:


model.fit(nlg.predictors, nlg.labels, epochs=100, verbose=5)
model.save("GenerativeModel.h5")


# In[19]:


