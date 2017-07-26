
# coding: utf-8

# In[1]:

import sys
import os
import numpy as np
import pandas as pd
import json
# mcfly
from mcfly import modelgen, find_architecture, storage


# In[2]:

data_path = '/media/sf_VBox_Shared/timeseries/actitiracker/WISDM_at_v2.0/'
preprocessed_path = os.path.join(data_path, 'preprocessed')
result_path = os.path.join(data_path, 'models_test')



# In[3]:

X_train = np.load(os.path.join(preprocessed_path, 'X_train.npy'))
X_val = np.load(os.path.join(preprocessed_path, 'X_val.npy'))
X_test = np.load(os.path.join(preprocessed_path, 'X_test.npy'))
y_train = np.load(os.path.join(preprocessed_path, 'y_train.npy'))
y_val = np.load(os.path.join(preprocessed_path, 'y_val.npy'))
y_test = np.load(os.path.join(preprocessed_path, 'y_test.npy'))



with open(os.path.join(preprocessed_path, 'labels.json')) as f:
    labels = json.load(f)


# ## Generate models


num_classes = y_train.shape[1]

models = modelgen.generate_models(X_train.shape,
                                  number_of_classes=num_classes,
                                  number_of_models = 15)




#what is the fraction of classes in the validation set?
pd.Series(y_val.mean(axis=0), index=labels)


if not os.path.exists(result_path):
        os.makedirs(result_path)



histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train,
                                                                           X_val, y_val,
                                                                           models,nr_epochs=5,
                                                                           subset_size=512,
                                                                           verbose=True,
                                                                           batch_size=32,
                                                                           outputpath=result_path,
                                                                           early_stopping=True)



print('Details of the training process were stored in ',os.path.join(result_path, 'models.json'))



best_model_index = np.argmax(val_accuracies)
best_model, best_params, best_model_types = models[best_model_index]
print('Model type and parameters of the best model:')
print(best_model_types)
print(best_params)


nr_epochs = 3
datasize = X_train.shape[0]
history = best_model.fit(X_train[:datasize,:,:], y_train[:datasize,:],
              epochs=nr_epochs, validation_data=(X_val, y_val))


best_model.save(os.path.join(result_path, 'best_model.h5'))



## Test on Testset
score_test = best_model.evaluate(X_test, y_test, verbose=True)
print('Score of best model: ' + str(score_test))





