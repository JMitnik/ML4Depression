import math
import copy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
# import matplotlib.pylab as plt
from predicting import make_algorithms
import numpy as np
from helpers import get_relevant_dates, convert_features_to_statistics, split_dataset, get_patient_id_by_rank
from operator import itemgetter

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

def precalculated_feature_selection(patient_x):
    return patient_x[BE_EMA_COLS]

def backward_selection(max_features, ml_models, patient_x, patient_y):
    features = patient_x.columns.tolist()

    # We do this X times, where X is the number of columns we plan to remove.
    for i in range(0, len(patient_x.columns.tolist()) - max_features):
        best_performance = math.inf
        worst_feature = ''

        for feature in features:
            sample_models = copy.deepcopy(ml_models)
            temp_features = copy.deepcopy(features)
            temp_features.remove(feature)
            fet_patient_x = patient_x[temp_features]
            models = make_algorithms(sample_models, fet_patient_x, patient_y)
            performance = models[0]['score']
            performance = next(item for item in performance if item.get('mae'))['mae']

            if performance < best_performance:
                best_performance = performance
                worst_feature = feature

        features.remove(worst_feature)
    return features

def forward_selection(max_features, ml_models, patient_x, patient_y):
    ordered_features = []
    ordered_scores = []
    features = []

    prev_best_perf = math.inf

    all_performances_index = []
    all_performances_perf = []

    for i in range(0, max_features):
        features_left = list(set(patient_x.columns) - set(features))
        best_performance = math.inf
        best_feature = ''

        for feature in features_left:
            sample_models = copy.deepcopy(ml_models)
            temp_features = copy.deepcopy(features)
            temp_features.append(feature)
            fet_patient_x = patient_x[temp_features]
            models = make_algorithms(sample_models, fet_patient_x, patient_y)
            # performance = models[0]['score']
            performance = models[0]['score']['mae']

            if performance < best_performance:
                best_performance = performance
                best_feature = feature


        print("Feature nr: #", i)
        all_performances_index.append(i+1)
        all_performances_perf.append(best_performance)
        features.append(best_feature)
        prev_best_perf = best_performance

    return features, (all_performances_index, all_performances_perf)

def correlate_features(max_features, patient_x, patient_y):
    correlations = []
    full_columns_and_corr = []
    abs_columns_and_corr = []

    for i in range(0, len(patient_x.columns)):
        corr_scores, p = pearsonr(patient_x[patient_x.columns[i]], patient_y)
        correlations.append(abs(corr_scores))

        if np.isfinite(corr_scores):
            full_columns_and_corr.append((patient_x.columns[i], corr_scores))
            abs_columns_and_corr.append((patient_x.columns[i], abs(corr_scores)))

    sorted_attributes = sorted(abs_columns_and_corr, key=itemgetter(1), reverse=True)
    res_list = [x[0] for x in sorted_attributes[:max_features]]

    return res_list, sorted(full_columns_and_corr, key=itemgetter(1), reverse=True)
