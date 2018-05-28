#%%
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import sklearn as sk
import importlib
%matplotlib inline

from sklearn.linear_model import RidgeCV , LassoCV
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from feature_engineering import patient_x, patient_y

#%%
def split_dataset(x, y, split_index):
    return (x[:split_index], y[:split_index], x[split_index:], y[split_index:])

rand_split = int(len(patient_x) * 0.66)
train_x, train_y, test_x, test_y = split_dataset(patient_x, patient_y, rand_split)

#%%
patient_x
# plot.plot(patient_y)
plot.plot(train_y)
plot.plot(test_y)

#%%
# Testing the algorithms
cv_alphas = (0.1, 0.2, 0.4, 0.6, 0.8)

# Training and Testing Lasso
lasso_regression = LassoCV(alphas=cv_alphas)
lasso_regression.fit(train_x, train_y)

lasso_prediction = lasso_regression.predict(test_x)

#%%
test_x

#%%


# Training and Testing Ridge
ridge_regression = RidgeCV(alphas=cv_alphas)
ridge_regression.fit(train_x, train_y)
ridge_prediction = ridge_regression.predict(test_x)

#%%
# Evaluating the algorithms
lasso_score = mean_absolute_error(test_y, lasso_prediction)
ridge_score = mean_absolute_error(test_y, ridge_prediction)

("Ridge MAE:"+str(ridge_score), "Lasso MAE:" +str(lasso_score))

# Training and Testing the different forms of SVR


#%%
time_index = test_y.index
pred_pandas_ridge = pd.DataFrame(ridge_prediction, index=time_index)
pred_pandas_lasso = pd.DataFrame(lasso_prediction, index=time_index)


#%%
# Training SVR

# Plotting the predictions vs actual
plot.plot(pred_pandas_ridge, color='green', label='ridge')
plot.plot(pred_pandas_lasso, color='blue', label='lasso')
plot.plot(test_y, color='red', label='true')
plot.legend()

#%%
# Let's get the parameters off of ridge and lasso.
ridge_regression.alpha_
lasso_regression.alpha_
