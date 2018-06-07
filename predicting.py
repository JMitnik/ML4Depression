#%%
# Importing libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plot
import sklearn as sk
import importlib

from sklearn.linear_model import RidgeCV , LassoCV
from sklearn import metrics

#%%
def train_algorithms(list_algorithms, train_x, train_y, alphas):
    result = []

    for algo in list_algorithms:
        algo['model'].fit(train_x, train_y)
        result.append(algo)

    return result

def test_algorithms(list_algorithms, test_x):
    result = []

    for algorithm_object in list_algorithms:
        algorithm_object['prediction'] = algorithm_object['model'].predict(test_x)
        result.append(algorithm_object)

    return result

def eval_algorithms(list_algorithms, test_y):
    plot_algorithms(list_algorithms, test_y)
    results = []

    for algo in list_algorithms:
        results.append(eval_algorithm(algo, test_y))

    return results

def eval_algorithm(algorithm, test_y):
    results = []

    results.append({"explained_var": metrics.explained_variance_score(algorithm['prediction'], test_y)})
    results.append({"mae": metrics.mean_absolute_error(algorithm['prediction'], test_y)})
    results.append({"mse": metrics.mean_squared_error(algorithm['prediction'], test_y)})
    results.append({"r2": metrics.r2_score(algorithm['prediction'], test_y)})

    algorithm['score'] = results
    return algorithm

def generate_color():
    return np.random.random(size=3) * 256

def plot_algorithms(list_algorithms, test_y):
    test_index = test_y.index

    for algo in list_algorithms:
        print(algo)
        pred = pd.DataFrame(algo['prediction'], index=test_index)
        plot.plot(pred, color='blue', label=algo['name'])

    plot.plot(test_y, color='red', label='true')
    plot.legend()
