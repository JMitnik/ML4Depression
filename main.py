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
from predicting import train_algorithms, test_algorithms, eval_algorithms, plot_algorithms
from feature_selection import backward_selection

# endregion

#%% region [cell] Init constants
# region
SLIDING_WINDOW = 7
CV_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
BE_EMA_COLS = [
    'avg_count_ema_q_2_7_days',
    'min_count_ema_q_2_7_days',
    'max_count_ema_q_2_7_days',
    'min_count_ema_q_4_7_days',
    'avg_count_ema_q_5_7_days',
    'min_count_ema_q_5_7_days',
    'max_count_ema_q_5_7_days',
    'std_count_ema_q_5_7_days',
    'avg_count_ema_q_6_7_days',
    'min_count_ema_q_6_7_days',
    'max_count_ema_q_6_7_days',
    'std_count_ema_q_6_7_days',
    'min_count_ema_q_7_7_days',
    'min_average_ema_q_1_7_days',
    'min_average_ema_q_3_7_days',
    'std_average_ema_q_3_7_days',
    'std_average_ema_q_4_7_days',
    'min_average_ema_q_6_7_days',
    'min_average_ema_q_7_7_days',
    'weekendDay'
]
# endregion

#%% region [cell] Initating patient(s)
# Safe patients: [1, 4, 5, 7, 8]
# region
sample_patient_id = get_patient_id_by_rank(1)
sample_patient_ema_features, sample_patient_engagement = get_EMA_features_and_target_for_patient(sample_patient_id)
# sample_patient_module_features = get_module_features_for_patient(sample_patient_id).transpose().fillna(0)
sample_patient_ema_features
# endregion

#%% region [cell] ML models defined
# region
ml_algorithms = [
    # {
    #     "name": "Lasso",
    #     "model": linear_model.LassoCV(alphas=CV_ALPHAS)
    # },
    # {
    #     "name": "Ridge",
    #     "model": linear_model.RidgeCV(alphas=CV_ALPHAS)
    # },
    {
        "name": "Random Forest",
        "model": ensemble.RandomForestRegressor(n_estimators=1000, max_depth=2)
    }
    # {
    #     "name": "Dummy Mean Regressor",
    #     "model": dummy.DummyRegressor()
    # },
    # {
    #     "name": "SVR RBF",
    #     "model": svm.SVR()
    # }
]

# endregion

#%% region [cell] Combine EMA and Module features
# region
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
# split_index = int(len(sample_patient_features) * 0.66)

# patient_x = get_relevant_dates(sample_patient_ML_features)
# patient_y = get_relevant_dates(sample_patient_engagement)

# test_features = backward_selection(10, ml_algorithms, patient_x, patient_y)
# #endregion

#%% region [cell] ML Predicting Modeling
# region
split_index = int(len(sample_patient_features) * 0.66)

patient_x = get_relevant_dates(sample_patient_ML_features)
patient_y = get_relevant_dates(sample_patient_engagement)

backward_selection(10, ml_algorithms, patient_x, patient_y)

train_x, train_y, test_x, test_y = split_dataset(
    patient_x, patient_y, split_index)
trained_models = train_algorithms(ml_algorithms, train_x, train_y)
tested_models = test_algorithms(ml_algorithms, train_x)

eval_models = eval_algorithms(tested_models, train_y)
# endregion

#%% region [cell] Investigating the evaluation
# region
def plot_feature_rank(feature_list, feature_ranking):
    plot.pyplot.style.use('fivethirtyeight')
    plot.pyplot.bar(feature_list, feature_ranking, orientation='vertical')
    plot.pyplot.xticks(feature_list, feature_ranking, rotation='vertical')
    plot.pyplot.ylabel('Importance')
    plot.pyplot.xlabel('Variable')
    plot.pyplot.title('Variable Importances')

feature_ranking = eval_models[2]['model'].feature_importances_
matched_feature_ranking = [(feature, ranking) for (feature, ranking) in zip(patient_x, feature_ranking)]
sorted(matched_feature_ranking, key=lambda x: x[1])

# plot_feature_rank(patient_x.columns, feature_ranking)

# endregion

#%% region [cell] Experimenting
#region-
# test_y.plot()
eval_models
# plot_algorithms(eval_models, test_y)

#endregion
