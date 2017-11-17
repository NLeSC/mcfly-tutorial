
# coding: utf-8

# # Experiment PAMAP with mcfly

# ## Import required Python modules

# In[1]:

import sys
import os
import numpy as np
import pandas as pd
# mcfly
from mcfly import modelgen, find_architecture, storage
from keras.models import load_model
np.random.seed(2)


# In[2]:

sys.path.insert(0, os.path.abspath('../..'))
from utils import tutorial_pamap2


# Load the preprocessed data as stored in Numpy-files. Please note that the data has already been split up in a training (training), validation (val), and test subsets. It is common practice to call the input data X and the labels y.

# In[3]:

data_path = '/media/sf_VBox_Shared/timeseries/PAMAP_Dataset/cleaned_7act/'


# In[4]:

X_train, y_train_binary, X_val, y_val_binary, X_test, y_test_binary, labels = tutorial_pamap2.load_data(data_path)


# In[5]:

print('x shape:', X_train.shape)
print('y shape:', y_train_binary.shape)


# The data is split between train test and validation.

# In[6]:

print('train set size:', X_train.shape[0])
print('validation set size:', X_val.shape[0])
print('test set size:', X_test.shape[0])


# Let's have a look at the distribution of the labels:

# In[7]:

frequencies = y_train_binary.mean(axis=0)
frequencies_df = pd.DataFrame(frequencies, index=labels, columns=['frequency'])
frequencies_df


# ## Generate models

# In[8]:

num_classes = y_train_binary.shape[1]

models = modelgen.generate_models(X_train.shape,
                                  number_of_classes=num_classes,
                                  number_of_models = 5)


# In[10]:

models_to_print = range(len(models))
for i, item in enumerate(models):
    if i in models_to_print:
        model, params, model_types = item
        print("-------------------------------------------------------------------------------------------------------")
        print("Model " + str(i))
        print(" ")
        print("Hyperparameters:")
        print(params)
        print(" ")
        print("Model description:")
        model.summary()
        print(" ")
        print("Model type:")
        print(model_types)
        print(" ")


# ## Compare models

# In[13]:

# Define directory where the results, e.g. json file, will be stored
resultpath = os.path.join(data_path, '..', 'data/models')
if not os.path.exists(resultpath):
        os.makedirs(resultpath)


# In[14]:

outputfile = os.path.join(resultpath, 'modelcomparison_pamap.json')
histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train_binary,
                                                                           X_val, y_val_binary,
                                                                           models,nr_epochs=5,
                                                                           subset_size=1000,
                                                                           verbose=True,
                                                                           outputfile=outputfile)
print('Details of the training process were stored in ',outputfile)


# In[15]:

best_model_index = np.argmax(val_accuracies)
best_model, best_params, best_model_types = models[best_model_index]
print('Model type and parameters of the best model:')
print(best_model_types)
print(best_params)


# ## Train the best model on the full dataset

# In[16]:

#We make a copy of the model, to start training from fresh
nr_epochs = 1
datasize = X_train.shape[0]
history = best_model.fit(X_train[:datasize,:,:], y_train_binary[:datasize,:],
              epochs=nr_epochs, validation_data=(X_val, y_val_binary))


# In[17]:

modelname = 'my_bestmodel.h5'
model_path = os.path.join(resultpath,modelname)


# In[18]:

best_model.save(model_path)


# In[ ]:



