#%% region [cell] Importing all the tools
#region
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk
import importlib
%matplotlib inline

from sklearn import metrics, linear_model
# Importing the different features
from ema_features import sample_patient_EMA_features, sample_patient_engagement
from predicting import train_algorithms, test_algorithms, eval_algorithms
#Importing the machine learning module

from helpers import get_relevant_dates, convert_features_to_statistics, split_dataset

#endregion

#%% region [cell] Init constants
#region
SLIDING_WINDOW = 7
CV_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
#endregion

#%% region [cell] ML models defined
#region
ml_algorithms = [
    {
        "name": "Lasso",
        "model": linear_model.LassoCV(alphas=CV_ALPHAS)
    },
    {
        "name": "Ridge",
        "model": linear_model.RidgeCV(alphas=CV_ALPHAS)
    }
]

#endregion

#%% region [cell] The Main Code
#region
split_index = int(len(sample_patient_EMA_features) * 0.66)
sample_patient_ML_features = convert_features_to_statistics(sample_patient_EMA_features, SLIDING_WINDOW)

sample_patient_ML_features = get_relevant_dates(sample_patient_ML_features)
sample_patient_engagement = get_relevant_dates(sample_patient_engagement)

train_x, train_y, test_x, test_y = split_dataset(sample_patient_ML_features, sample_patient_engagement, split_index)
trained_models = train_algorithms(ml_algorithms, train_x, train_y, CV_ALPHAS)
tested_models = test_algorithms(ml_algorithms, test_x)

eval_models = eval_algorithms(tested_models, test_y)
#endregion
