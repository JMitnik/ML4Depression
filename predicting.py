#%%
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk
import importlib
%matplotlib inline

from sklearn.linear_model import Ridge, Lasso
from feature_engineering import patient_x, patient_y

#%%
def split_dataset(x, y, split_index):
    return (x[:split_index], y[:split_index], x[split_index:], y[split_index:])

rand_split = int(len(patient_x) * 0.66)
train_x, train_y, test_x, test_y = split_dataset(patient_x, patient_y, rand_split)

#%%
# Testing Lasso
lasso_regression = Lasso(alpha=0.1)
lasso_regression.fit(train_x, train_y)

lasso_prediction = lasso_regression.predict(test_x)

#%%
# Evaluation Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
lasso_score = mean_squared_log_error(test_y, lasso_prediction)
lasso_score
