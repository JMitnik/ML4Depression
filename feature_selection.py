import math
import copy
from predicting import make_algorithms
from helpers import get_relevant_dates, convert_features_to_statistics, split_dataset, get_patient_id_by_rank

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

def precalculated_feature_selection(ml_models):
    return ml_models

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

# TODO: Create forwards feature selection.
# # Specifies feature selection approaches for classification to identify the most important features.
# class FeatureSelectionRegression:

#     # Forward selection for classification which selects a pre-defined number of features (max_features)
#     # that show the best accuracy. We assume a decision tree learning for this purpose, but
#     # this can easily be changed. It return the best features.
#     def forward_selection(self, max_features, X_train, y_train):
#         ordered_features = []
#         ordered_scores = []

#         # Start with no features.
#         selected_features = []
#         ra = RegressionAlgorithms()
#         re = RegressionEvaluation()
#         prev_best_perf = sys.float_info.max

#         # Select the appropriate number of features.
#         for i in range(0, max_features):

#             #Determine the features left to select.
#             features_left = list(set(X_train.columns) - set(selected_features))
#             best_perf = sys.float_info.max
#             best_feature = ''

#             # For all features we can still select...
#             for f in features_left:
#                 temp_selected_features = copy.deepcopy(selected_features)
#                 temp_selected_features.append(f)

#                 # Determine the mse of a decision tree learner if we were to add
#                 # the feature.
#                 pred_y_train, pred_y_test = ra.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
#                 perf = re.mean_squared_error(y_train, pred_y_train)

#                 # If the performance is better than what we have seen so far (we aim for low mse)
#                 # we set the current feature to the best feature and the same for the best performance.
#                 if perf < best_perf:
#                     best_perf = perf
#                     best_feature = f
#             # We select the feature with the best performance.
#             selected_features.append(best_feature)
#             prev_best_perf = best_perf
#             ordered_features.append(best_feature)
#             ordered_scores.append(best_perf)
#         return selected_features, ordered_features, ordered_scores

#     # Backward selection for classification which selects a pre-defined number of features (max_features)
#     # that show the best accuracy. We assume a decision tree learning for this purpose, but
#     # this can easily be changed. It return the best features.

#     # Select features based upon the correlation through the Pearson coefficient.
#     # It return the max_features best features.
#     def pearson_selection(self, max_features, X_train, y_train):
#         correlations = []
#         full_columns_and_corr = []
#         abs_columns_and_corr = []

#         # Compute the absolute correlations per column.
#         for i in range(0, len(X_train.columns)):
#             corr, p = pearsonr(X_train[X_train.columns[i]], y_train)
#             correlations.append(abs(corr))
#             if np.isfinite(corr):
#                 full_columns_and_corr.append((X_train.columns[i], corr))
#                 abs_columns_and_corr.append((X_train.columns[i], abs(corr)))

#         sorted_attributes = sorted(abs_columns_and_corr,key=itemgetter(1), reverse=True)
#         res_list = [x[0] for x in sorted_attributes[0:max_features]]

#         # And return the most correlated ones.
#         return res_list, sorted(full_columns_and_corr,key=itemgetter(1), reverse=True)
