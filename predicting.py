#%%
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
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
patient_x

#%%
# Testing Lasso
lasso_regression = Lasso(alpha=0.1)
lasso_regression.fit(train_x, train_y)

lasso_prediction = lasso_regression.predict(test_x)

#%%
# Evaluation Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
lasso_score = mean_absolute_error(test_y, lasso_prediction)
lasso_score

#%%
# Testing Ridge
ridge_regression = Ridge(alpha=0.1)
ridge_regression.fit(train_x, train_y)

ridge_prediction = ridge_regression.predict(test_x)

#%%

ridge_score = mean_absolute_error(test_y, ridge_prediction)
("Ridge MAE:"+str(ridge_score), "Lasso MAE:" +str(lasso_score))

#%%
time_index = test_y.index
pred_pandas_ridge = pd.DataFrame(ridge_prediction, index=time_index)
pred_pandas_lasso = pd.DataFrame(lasso_prediction, index=time_index)

plot.plot(pred_pandas_ridge, color='green')
plot.plot(pred_pandas_lasso, color='blue')
plot.plot(test_y, color='red')
