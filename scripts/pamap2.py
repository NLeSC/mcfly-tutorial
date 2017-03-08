'''
Script do do a model comparison
Run this script from the root of repository:

`python scripts/pamap2.py`
'''
import sys
import os
import numpy as np
import pandas as pd
from mcfly import modelgen, find_architecture, storage

np.random.seed(2)
sys.path.insert(0, os.path.abspath('.'))
print(sys.path)
from utils import tutorial_pamap2

# ## Settings
# Specify in which directory you want to store the data:
directory_to_extract_to = 'notebooks/tutorial/'
number_of_models = 2
subset_size = 10
nr_epochs = 1

# ## Download data and pre-proces data
data_path = tutorial_pamap2.download_preprocessed_data(directory_to_extract_to)
X_train, y_train_binary, X_val, y_val_binary, X_test, y_test_binary, labels = tutorial_pamap2.load_data(data_path)

# The data is split between train test and validation.

print('train set size:', X_train.shape[0])
print('validation set size:', X_val.shape[0])
print('test set size:', X_test.shape[0])

# ## Generate models

num_classes = y_train_binary.shape[1]
models = modelgen.generate_models(X_train.shape,
                                  number_of_classes=num_classes,
                                  number_of_models=number_of_models)

# Define output path
resultpath = os.path.join(directory_to_extract_to, 'data/models')
if not os.path.exists(resultpath):
    os.makedirs(resultpath)
outputfile = os.path.join(resultpath, 'modelcomparison.json')

histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train_binary,
                                                                                  X_val, y_val_binary,
                                                                                  models, nr_epochs=nr_epochs,
                                                                                  subset_size=subset_size,
                                                                                  verbose=True,
                                                                                  outputfile=outputfile)
print('Details of the training process were stored in ', outputfile)

# # Inspect model performance (table)
modelcomparisons = pd.DataFrame({'model': [str(params) for model, params, model_types in models],
                                 'train_acc': [history.history['acc'][-1] for history in histories],
                                 'train_loss': [history.history['loss'][-1] for history in histories],
                                 'val_acc': [history.history['val_acc'][-1] for history in histories],
                                 'val_loss': [history.history['val_loss'][-1] for history in histories]
                                 })
modelcomparisons.to_csv(os.path.join(resultpath, 'modelcomparisons.csv'))

modelcomparisons

# # Choose the best model and save it


best_model_index = np.argmax(val_accuracies)
best_model, best_params, best_model_types = models[best_model_index]
print('Model type and parameters of the best model:')
print(best_model_types)
print(best_params)
modelname = 'my_bestmodel'
storage.savemodel(best_model, resultpath, modelname)
