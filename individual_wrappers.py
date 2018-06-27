from helpers import *
from predicting import *
from ema_features import *
from module_features import *
from context_features import *
from feature_selection import *
from functools import reduce
from copy import deepcopy
from collections import Counter
FEATURE_PATH = "data/features/"

def get_features_for_patient(patient_id):
    """Returns patient_x and patient_Y for patient_id"""
    patient_id = str(patient_id)
    patient_ema_features, patient_engagement = get_EMA_features_and_target_for_patient(
        patient_id)
    patient_module_features = get_module_features_for_patient(
        patient_id).transpose().fillna(0)
    patient_features = patient_ema_features.join(
        patient_module_features.fillna(0)).fillna(0)
    patient_extended_features = convert_features_to_statistics(
        patient_features, SLIDING_WINDOW)
    patient_extended_features['weekendDay'] = get_weekend_days(
        patient_extended_features.index.to_series())
    patient_x = get_relevant_dates(patient_extended_features)
    patient_y = get_relevant_dates(patient_engagement)

    return (patient_x, patient_y)

def learn_patients_setups(list_patients_objects, ml_algorithms, max_features=10):
    """ Trains models for multiple patients, for multiple setups

        Pre:
            * All patients have already a list of top-features.
            * All patietns already have a list of top-correlations.
    """
    feature_cols = get_FS_cols(deepcopy(list_patients_objects), max_features)
    top_feature_cols = get_FS_cols(deepcopy(list_patients_objects), 20, 'top_features')

    top_feature_results = get_patients_scores(deepcopy(list_patients_objects), deepcopy(ml_algorithms), top_feature_cols)
    all_features_results = get_patients_scores(deepcopy(list_patients_objects), deepcopy(ml_algorithms))
    FS_results = get_patients_scores(deepcopy(list_patients_objects), deepcopy(ml_algorithms), feature_cols)

    return (top_feature_results, FS_results, all_features_results)

def get_FS_cols(list_patients_objects, max_features=10, technique='correlation'):
    """ Retrieves the overall feature-selection columns for all patients.

        Pre:
            * All patients already have embedded the top-X features.
            * All patients have embedded the correlation features.
    """

    if technique == 'correlation':
        features = get_patients_correlated_score(
            list_patients_objects).index.to_series().tolist()

    if technique == 'top_features':
        features = get_top_features(list_patients_objects)

    return features[:max_features]

def get_top_features(list_patients_objects, max_features=20):
    count = Counter()
    for patient in list_patients_objects:
        count += get_top_feature_ranking(patient['top_features'])

    ranked_top_features = [item[0] for item in count.most_common()]

    return ranked_top_features


def get_top_feature_ranking(top_features):
    result = []
    count = Counter()

    for index, feature in enumerate(top_features):
        count = count + Counter({feature: len(top_features) - index})

    return count

def get_patients_scores_and_features(list_patients_objects, ml_algorithms, max_features=10):
    """ Initializes a run to embed into the patients-object both the top-features, MAE and
        pearson-correlated-features.

        Pre:
            * Each patient has embedded a patient-id.
    """
    result = []
    copied_patients = deepcopy(list_patients_objects)

    for patient in copied_patients:
        patient_x, patient_y = get_features_for_patient(patient['patient_id'])
        models = make_algorithms(ml_algorithms, patient_x, patient_y)
        performance = next(item for item in models[2]['score'] if item.get('mae'))['mae']

        # TODO: Make this a setting
        top_features = forward_selection(max_features, ml_algorithms, patient_x, patient_y)

        patient['pearson_correlated_features'] = correlate_features(max_features  , patient_x, patient_y)
        patient['top_features'] = top_features

        save_patient_object(patient, '_top5_featureselection_')

        result.append(patient)

    return result


def get_patients_scores(list_patients_objects, ml_algorithms, feature_cols=None):
    """ Returns a list of patient objects.

        Pre:
            * Patients need a patient_id embedded
            * Optional: feature_columns need to be given
    """
    result = []

    for patient in list_patients_objects:
        patient_x, patient_y = get_features_for_patient(patient['patient_id'])

        if feature_cols:
            patient_x = patient_x[feature_cols]
            patient['feature_selection'] = 'true'

        models = make_algorithms(ml_algorithms, patient_x, patient_y)
        performance = next(item for item in models[2]['score'] if item.get('mae'))['mae']
        patient['MAE'] = performance

        result.append(patient)

    return result

def get_patients_mean_MAE_score(list_patients_objects):
    """ Returns an average of all MAE scores for all patients.

        Pre:
            * list_patients_objects must contain patients with embedded MAE eval scores.
    """
    # print(list_patients_objects)
    # return None
    return pd.DataFrame(list_patients_objects).mean()['MAE']

def get_patients_correlated_score(list_patients_objects):
    """ Returns a ranking in correlated score

        Pre:
            * Patients must have already embedded pearson_correlated_features, either
            by a) importing stored patient_object, or by b) getting these values.
    """

    processed_patients = []

    for patient in list_patients_objects:
        patient = pd.DataFrame(patient['pearson_correlated_features'][1]).set_index([0])
        processed_patients.append(patient)

    results = reduce(lambda x, y: x+y, processed_patients)

    return results.sort_values(by=results.columns[0], ascending=False)

def save_patient_object(patient_object, prefix='', path_to_features=FEATURE_PATH):
    """ Stores a patient object (containing the patient_id) to a npy format."""
    np.save(path_to_features + str(prefix) + str(patient_object['patient_id'])+'_'+'top_features.npy', patient_object)


def load_patient_object(patient_id, prefix='', path_to_features=FEATURE_PATH):
    """ Returns a patient object from a patient_id."""
    return np.load(path_to_features + str(prefix) + str(patient_id)+'_'+'top_features.npy')

def load_patients_object(list_patients, prefix='', path_to_features=FEATURE_PATH):
    result = []

    for patient in list_patients:
        patient_object = load_patient_object(patient['patient_id'], prefix, path_to_features).tolist()
        result.append(patient_object)

    return result
