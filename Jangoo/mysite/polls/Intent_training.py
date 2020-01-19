#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Dark-Sied/Intent_Classification/blob/master/Intent_classification_final.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:

from sklearn.model_selection import train_test_split 

# In[26]:

import nlu


# In[27]:


train_X, val_X, train_Y, val_Y = train_test_split(nlu.padded_doc, nlu.output_one_hot, shuffle = True, test_size = 0.2)


# In[28]:


print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


# In[32]:


def create_model(vocab_size, max_length):
  model = nlu.Sequential()
  model.add(nlu.Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(nlu.Bidirectional(nlu.LSTM(128)))
#   model.add(LSTM(128))
  model.add(nlu.Dense(32, activation = "relu"))
  model.add(nlu.Dropout(0.5))
  model.add(nlu.Dense(22, activation = "softmax"))
  
  return model


# In[33]:


model = create_model(nlu.vocab_size, nlu.max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()


# In[35]:


filename = 'model.h5'
checkpoint = nlu.ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])

model.save("model.h5")
# In[ ]:




