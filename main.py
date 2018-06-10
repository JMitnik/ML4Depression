#%% region [cell] Importing all the tools
# region
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk
from sklearn import metrics, linear_model, ensemble
from helpers import get_relevant_dates, convert_features_to_statistics, split_dataset, get_patient_id_by_rank

# Importing the different features
from ema_features import get_EMA_features_and_target_for_patient
from module_features import get_module_features_for_patient
from context_features import get_weekend_days

# Importing the machine learning module
from predicting import train_algorithms, test_algorithms, eval_algorithms

# endregion

#%% region [cell] Init constants
# region
SLIDING_WINDOW = 7
CV_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
# endregion

#%% region [cell] Initating patient(s)
# region
sample_patient_id = get_patient_id_by_rank(4)
sample_patient_ema_features, sample_patient_engagement = get_EMA_features_and_target_for_patient(sample_patient_id)
sample_patient_module_features = get_module_features_for_patient(sample_patient_id).transpose().fillna(0)
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
        "model": ensemble.RandomForestRegressor(n_estimators=1000)
    }
]

# endregion

#%% region [cell] Combine EMA and Module features
# region
sample_patient_features = sample_patient_ema_features.join(sample_patient_module_features.fillna(0)).fillna(0)

sample_patient_ML_features = convert_features_to_statistics(
    sample_patient_features, SLIDING_WINDOW)

# endregion

#%% region [cell] Add contextual information
# region
sample_patient_ML_features['weekendDay'] = get_weekend_days(
    sample_patient_ML_features.index.to_series())
# endregion

#%% region [cell] ML Predicting Modeling
# region
split_index = int(len(sample_patient_features) * 0.66)

patient_x = get_relevant_dates(sample_patient_ML_features)
patient_y = get_relevant_dates(sample_patient_engagement)

train_x, train_y, test_x, test_y = split_dataset(
    patient_x, patient_y, split_index)
trained_models = train_algorithms(ml_algorithms, train_x, train_y, CV_ALPHAS)
tested_models = test_algorithms(ml_algorithms, test_x)

eval_models = eval_algorithms(tested_models, test_y)
# endregion

#%% region [cell] Investigating the evaluation
# region
rank = [{'coef':x, 'rank':i} for i,x in enumerate(eval_models[1]['model'].coef_)]
rank = sorted(rank, key=lambda x: x['coef'], reverse=True)
patient_x.columns[rank[0]['rank']]
# endregion

#%% region [cell] Experimenting
#region
eval_models
#endregion
