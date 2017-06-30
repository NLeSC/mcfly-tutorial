
# coding: utf-8

# In[2]:

import sys
import os
import numpy as np
import pandas as pd
# mcfly
from mcfly import modelgen, find_architecture, storage

# Parameters
data_path = '/media/sf_VBox_Shared/timeseries/UCI_EEG_alcoholic/'
number_of_models = 10
nr_epochs = 5
subset_size = 512
batch_size = 32
early_stopping = True

# In[3]:


preprocessed_path = os.path.join(data_path, 'preprocessed')
result_path = os.path.join(data_path, 'models')


# In[4]:


X_train = np.load(os.path.join(preprocessed_path, 'X_train.npy'))
X_val = np.load(os.path.join(preprocessed_path, 'X_val.npy'))
X_test = np.load(os.path.join(preprocessed_path, 'X_test.npy'))
y_train = np.load(os.path.join(preprocessed_path, 'y_train.npy'))
y_val = np.load(os.path.join(preprocessed_path, 'y_val.npy'))
y_test = np.load(os.path.join(preprocessed_path, 'y_test.npy'))


# ## Generate models

# In[5]:

num_classes = y_train.shape[1]

models = modelgen.generate_models(X_train.shape,
                                  number_of_classes=num_classes,
                                  number_of_models = number_of_models)


# In[6]:

#what is the fraction of a vs c in the validation set?
y_val.mean(axis=0)


# In[7]:

if not os.path.exists(result_path):
        os.makedirs(result_path)


# In[ ]:

outputfile = os.path.join(result_path, 'modelcomparison.json')
histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train,
                                                                           X_val, y_val,
                                                                           models,nr_epochs=nr_epochs,
                                                                           subset_size=subset_size,
                                                                           verbose=True,
                                                                           batch_size=batch_size,
                                                                           outputfile=outputfile,
                                                                           early_stopping=early_stopping)
print('Details of the training process were stored in ',outputfile)


# In[ ]:



