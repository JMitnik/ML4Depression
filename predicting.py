#%%
# Importing libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plot
import sklearn as sk
import importlib

from sklearn.linear_model import RidgeCV , LassoCV
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, accuracy_score

#%%
def train_algorithms(list_algorithms, train_x, train_y, alphas):
    result = []

    for algo in list_algorithms:
        algo['model'].fit(train_x, train_y)
        result.append(list_algorithms)

    return result

def test_algorithms(list_algorithms, test_x):
    result = []

    for algorithm_object in list_algorithms:
        algorithm_object['prediction'] = algorithm_object['model'].predict(test_x)
        result.append(algorithm_object)

    return result

def eval_algorithms(list_algorithms, test_y):
    plot_algorithms(list_algorithms, test_y)
    return ""

def generate_color():
    return np.random.random(size=3) * 256

def plot_algorithms(list_algorithms, test_y):
    test_index = test_y.index

    for algo in list_algorithms:
        print(algo)
        pred = pd.DataFrame(algo['prediction'], index=test_index)
        plot.plot(pred, color='blue', label='test')

    plot.plot(test_y, color='red', label='true')
    plot.legend()
