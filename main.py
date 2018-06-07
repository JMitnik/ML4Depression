#%% region [cell] Importing all the tools
#region
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk

from sklearn import metrics, linear_model
from helpers import get_relevant_dates, convert_features_to_statistics, split_dataset

# Importing the different features
from ema_features import sample_patient_EMA_features, sample_patient_engagement
from module_features import sample_patient_module_features
from context_features import get_weekend_days

#Importing the machine learning module
from predicting import train_algorithms, test_algorithms, eval_algorithms

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

#%% region [cell] Combine EMA and Module features
#region
sample_patient_ML_features = convert_features_to_statistics(sample_patient_EMA_features, SLIDING_WINDOW)

#endregion

#%% region [cell] Add contextual information
#region
sample_patient_ML_features['weekendDay'] = get_weekend_days(sample_patient_ML_features.index.to_series())
#endregion

#%% region [cell] ML Predicting Modeling
#region
split_index = int(len(sample_patient_EMA_features) * 0.66)

sample_patient_ML_features = get_relevant_dates(sample_patient_ML_features)
sample_patient_engagement = get_relevant_dates(sample_patient_engagement)

train_x, train_y, test_x, test_y = split_dataset(
    sample_patient_ML_features, sample_patient_engagement, split_index)
trained_models = train_algorithms(ml_algorithms, train_x, train_y, CV_ALPHAS)
tested_models = test_algorithms(ml_algorithms, test_x)

eval_models = eval_algorithms(tested_models, test_y)
#endregion

#%% region [cell] Investigating the evaluation
#region
eval_models
#endregion
