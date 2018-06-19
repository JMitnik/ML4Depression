#%% region [cell] Importing all the tools
# region
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk
from sklearn import metrics, linear_model, ensemble, neural_network, svm, dummy
from helpers import get_relevant_dates, convert_features_to_statistics, split_dataset, get_patient_id_by_rank

# Importing the different features
from ema_features import get_EMA_features_and_target_for_patient
from module_features import get_module_features_for_patient
from context_features import get_weekend_days

# Importing the machine learning module
from predicting import train_algorithms, test_algorithms, eval_algorithms, plot_algorithms, make_algorithms
from feature_selection import backward_selection

# endregion

#%% region [cell] Init constants
# region
SLIDING_WINDOW = 7
CV_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
# endregion

#%% region [cell] Initating patient(s)
# TODO: Get all safe patients automatically.
# Safe patients: [1, 4, 5, 7, 8]
# region
sample_patient_id = get_patient_id_by_rank(1)
sample_patient_ema_features, sample_patient_engagement = get_EMA_features_and_target_for_patient(sample_patient_id)

#TODO: For some reason, this code below crashes or gets stuck.
# sample_patient_module_features = get_module_features_for_patient(sample_patient_id).transpose().fillna(0)
# endregion

#%% region [cell] ML models defined
# region
ml_algorithms = [
    {
        "name": "Lasso",
        "model": linear_model.LassoCV(alphas=CV_ALPHAS)
    },
    {
        "name": "Ridge",
        "model": linear_model.RidgeCV(alphas=CV_ALPHAS)
    },
    {
        "name": "Random Forest",
        "model": ensemble.RandomForestRegressor(n_estimators=1000, max_depth=2)
    },
    {
        "name": "Dummy Mean Regressor",
        "model": dummy.DummyRegressor()
    },
    {
        "name": "SVR RBF",
        "model": svm.SVR()
    }
]

# endregion

#%% region [cell] Combine EMA and Module features
# region
#TODO: Once fixed the above piece, remove line 68 and uncomment 67.
# sample_patient_features = sample_patient_ema_features.join(sample_patient_module_features.fillna(0)).fillna(0)
sample_patient_features = sample_patient_ema_features

sample_patient_ML_features = convert_features_to_statistics(
    sample_patient_features, SLIDING_WINDOW)

sample_patient_ML_features
# endregion

#%% region [cell] Add contextual information
# region
sample_patient_ML_features['weekendDay'] = get_weekend_days(
    sample_patient_ML_features.index.to_series())
# endregion

#%% region [cell] Feature selection
#region
patient_x = get_relevant_dates(sample_patient_ML_features)
patient_y = get_relevant_dates(sample_patient_engagement)

This is where we do the feature selection before we pass it to the ML-prediction of the next cell.
# #endregion

#%% region [cell] ML Predicting Modeling
# region

models = make_algorithms(ml_algorithms, patient_x, patient_y)
# endregion

#%% region [cell] Investigating the evaluation
# region
feature_ranking = models[2]['model'].feature_importances_
matched_feature_ranking = [(feature, ranking) for (feature, ranking) in zip(patient_x, feature_ranking)]
sorted(matched_feature_ranking, key=lambda x: x[1])
# endregion

#%% region [cell] Experimenting
#region

#endregion
