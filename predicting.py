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
def train_algorithms(list_algorithms, train_x, train_y):
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
    a = hex(random.randrange(0, 256))
    b = hex(random.randrange(0,256))
    c = hex(random.randrange(0,256))
    a = a[2:]
    b = b[2:]
    c = c[2:]
    if len(a)<2:
        a = "0" + a
    if len(b)<2:
        b = "0" + b
    if len(c)<2:
        c = "0" + c
    z = a + b + c
    return "#" + z.upper()

def plot_algorithms(list_algorithms, test_y):
    test_index = test_y.index

    for algo in list_algorithms:
        print(algo)
        pred = pd.DataFrame(algo['prediction'], index=test_index)
        plot.plot(pred, color=generate_color(), label=algo['name'])

    plot.plot(test_y, color='red', label='true')
    plot.legend()
