
# coding: utf-8

# # Experiment PAMAP2 with mcfly

# This experiment finds an optimal model for the PAMAP2 dataset.

# ## Import required Python modules

# In[1]:

import sys
import os
import numpy as np
import pandas as pd
# mcfly
from mcfly import modelgen, find_architecture, storage


# ## Load the data

# In[2]:

# Define directory where the results, e.g. json file, will be stored
datapath = '/data/mcfly/input/'
resultpath = '/data/mcfly/output/' 
if not os.path.exists(resultpath):
        os.makedirs(resultpath)


# In[3]:

Xs = []
ys = []

ext = '.npy'
for i in range(9):
    Xs.append(np.load(os.path.join(datapath,'X_'+str(i)+ext)))
    ys.append(np.load(os.path.join(datapath, 'y_'+str(i)+ext)))


# In[4]:

print(Xs[0].shape, ys[0].shape)


# ## Generate models

# First step is to create a model architecture. As we do not know what architecture is best for our data we will create a set of models to investigate which architecture is most suitable for our data and classification task. You will need to specificy how many models you want to create with argument 'number_of_models', the type of model which can been 'CNN' or 'DeepConvLSTM', and maximum number of layers per modeltype. See for a full overview of the optional arguments the function documentation of modelgen.generate_models

# In[5]:

num_classes = ys[0].shape[1]
np.random.seed(123)
models = modelgen.generate_models(Xs[0].shape,
                                  number_of_classes=num_classes,
                                  number_of_models = 15)


# ## Compare models
# Now that the model architectures have been generated it is time to compare the models by training them in a subset of the training data and evaluating the models in the validation subset. This will help us to choose the best candidate model. Performance results are stored in a json file.

# In[6]:

def split_train_test(X_list, y_list, j):
    X_train = np.concatenate(X_list[0:j]+X_list[j+1:])
    X_test = X_list[j]
    y_train = np.concatenate(y_list[0:j]+y_list[j+1:])
    y_test = y_list[j]
    return X_train, y_train, X_test, y_test

def split_train_small_val(X_list, y_list, j, trainsize=500, valsize=500):
    X = np.concatenate(X_list[0:j]+X_list[j+1:])
    y = np.concatenate(y_list[0:j]+y_list[j+1:])
    rand_ind = np.random.choice(X.shape[0], trainsize+valsize, replace=False)
    X_train = X[rand_ind[:trainsize]]
    y_train = y[rand_ind[:trainsize]]
    X_val = X[rand_ind[trainsize:]]
    y_val = y[rand_ind[trainsize:]]
    return X_train, y_train, X_val, y_val


# In[9]:

from keras.optimizers import Adam
from keras.models import model_from_json

def get_fresh_copy(model, lr):
    model_json = model.to_json()
    model_copy = model_from_json(model_json)
    model_copy.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    #for layer in model_copy.layers:
    #    layer.build(layer.input_shape)
    return model_copy


# In[10]:

models = [(get_fresh_copy(model, params['learning_rate']), params, model_type)  for model, params, model_type in models]


# In[11]:

trainsize = 500
valsize = 500 


# In[12]:

import time
t = time.time()
np.random.seed(123)
histories_list, val_accuracies_list, val_losses_list = [], [], []
for j in range(len(Xs)):
    print('fold '+str(j))
    models = [(get_fresh_copy(model, params['learning_rate']), params, model_type)  for model, params, model_type in models]
    X_train, y_train, X_val, y_val = split_train_small_val(Xs, ys, j, trainsize=trainsize, valsize=valsize)
    histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train,
                                                                           X_val, y_val,
                                                                           models,
                                                                           nr_epochs=10,
                                                                           subset_size=500,
                                                                           verbose=True,
                                                                           outputfile=os.path.join(resultpath, 
                                                                                  'experiment'+str(j)+'.json'),
                                                                           early_stopping=True)
    histories_list.append(histories)
    val_accuracies_list.append(val_accuracies)
    val_losses.append(val_losses)
print(time.time()-t)


# In[13]:

# Read them all back in
import json
model_jsons = []
for j in range(len(Xs)):
    with open(os.path.join(resultpath, 'experiment'+str(j)+'.json'), 'r') as outfile:
        model_jsons.append(json.load(outfile))


# In[ ]:

val_accuracies = np.array([[mod['val_acc'][-1] for mod in fold] for fold in model_jsons])


# In[ ]:

val_acc = np.array([np.array([mod['val_acc'][-1] for mod in fold], dtype='float') for fold in model_jsons])
train_acc = np.array([np.array([mod['train_acc'][-1] for mod in fold], dtype='float') for fold in model_jsons])
train_loss = np.array([np.array([mod['train_loss'][-1] for mod in fold], dtype='float') for fold in model_jsons])
val_loss = np.array([np.array([mod['val_loss'][-1] for mod in fold], dtype='float') for fold in model_jsons])


# In[ ]:

val_accuracies_avg = val_acc.mean(axis=0)
print('val_accuracies_avg:', val_accuracies_avg)


# In[ ]:

best_model_index = np.argmax(val_accuracies_avg)
best_model, best_params, best_model_types = models[best_model_index]
print('Model type and parameters of the best model:')
print(best_model_types)
print(best_params)


# In[ ]:

modelname = 'bestmodel_sample'
storage.savemodel(best_model,resultpath,modelname)


# ## Train the best model for real

# Now that we have identified the best model architecture out of our random pool of models we can continue by training the model on the full training sample. For the purpose of speeding up the example we only train the full model on the first 1000 values. You will need to replace this by 'datasize = X_train.shape[0]' in a real world example.

# In[ ]:

nr_epochs = 2

np.random.seed(123)
histories, test_accuracies_list, models = [], [], []
for j in range(len(Xs)):
    X_train, y_train, X_test, y_test = split_train_test(Xs, ys, j)
    model_copy = get_fresh_copy(best_model, best_params['learning_rate'])
    datasize = X_train.shape[0]
    
    history = model_copy.fit(X_train[:datasize,:,:], y_train[:datasize,:],
              nb_epoch=nr_epochs, validation_data=(X_test, y_test))
    
    histories.append(history)
    test_accuracies_list.append(history.history['val_acc'][-1] )
    models.append(model_copy)


# In[ ]:

print('accuracies: ', test_accuracies_list)


# In[ ]:

print(np.mean(test_accuracies_list))


# ### Saving, loading and comparing reloaded model with orignal model

# The modoel can be saved for future use. The savemodel function will save two separate files: a json file for the architecture and a npy (numpy array) file for the weights.

# In[ ]:

modelname = 'my_bestmodel'


# In[ ]:

for i, model in enumerate(models):
    storage.savemodel(model,resultpath,modelname+str(i))


# In[ ]:



