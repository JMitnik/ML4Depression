#%%
# Importing libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plot
import sklearn as sk
import importlib
%matplotlib inline

from sklearn.linear_model import RidgeCV , LassoCV
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, accuracy_score
from ema_features import patient_x, patient_y

#%%
def split_dataset(x, y, split_index):
    return (x[:split_index], y[:split_index], x[split_index:], y[split_index:])

def train_algorithm(algorithm, x, y, alphas=()):
    ml_alg = algorithm(alphas=alphas)
    ml_alg.fit(x, y)

    return ml_alg

def try_algorithms(list_algorithms, train_x, train_y, test_x, test_y, alphas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    predictions = []

    for algo in list_algorithms:
        algorithm = train_algorithm(algo, train_x, train_y, cv_alphas)
        algorithm_pred = test_algorithm(algorithm, test_x)
        predictions.append(
            {
                "algorithm": algorithm,
                "prediction": algorithm_pred
            }
        )

    eval_algorithms(predictions, test_y)

    return predictions

def test_algorithm(algorithm, x):
    return algorithm.predict(x)

def generate_color():
    color = "%06x" % random.randint(0, 0xFFFFFF)

def plot_algorithms(list_predictions, test_y):
    test_index = test_y.index

    for pred in list_predictions:
        pred = pd.DataFrame(pred['prediction'], index=test_index)
        plot.plot(pred, color=generate_color(), label='test')

    plot.plot(test_y, color='red', label='true')
    plot.legend()

def eval_algorithms(list_predictions, test_y):


data_split = int(len(patient_x) * 0.66)
train_x, train_y, test_x, test_y = split_dataset(patient_x, patient_y, data_split)

#%%
# Testing the algorithms
try_algorithms([LassoCV, RidgeCV], train_x, train_y, test_x, test_y)
