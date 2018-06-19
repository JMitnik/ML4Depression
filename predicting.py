#%%
# Importing libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plot
import sklearn as sk
import importlib
from helpers import get_relevant_dates, convert_features_to_statistics, split_dataset, get_patient_id_by_rank

from sklearn.linear_model import RidgeCV , LassoCV
from sklearn import metrics

#%%
def train_algorithms(list_algorithms, train_x, train_y):
    result = []

    for algo in list_algorithms:
        algo['model'].fit(train_x, train_y)
        result.append(algo)

    return result

def make_algorithms(list_algorithms, patient_x, patient_y, set_split=0.66):
    split_index = int(len(patient_x) * set_split)

    train_x, train_y, test_x, test_y = split_dataset(
        patient_x, patient_y, split_index)
    trained_models = train_algorithms(sample_models, train_x, train_y)
    tested_models = test_algorithms(trained_models, test_x)
    eval_models = eval_algorithms(tested_models, test_y)

    return eval_models

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
