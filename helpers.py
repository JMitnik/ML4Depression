import pandas as pd
import random
from statistics import mean
from ema_features import get_patient_features


def get_relevant_dates(patient_df):
    return patient_df[7:-7]


def convert_features_to_statistics(features, window):
    patient_ml = pd.DataFrame(index=features.index)
    for col in features.fillna(0):
        patient_ml['avg_'+col+'_'+str(window)+'_days'] = features[col].rolling(
            str(window)+'d').mean().shift(1)
        patient_ml['min_'+col+'_'+str(window)+'_days'] = features[col].rolling(
            str(window)+'d').min().shift(1)
        patient_ml['max_'+col+'_'+str(window)+'_days'] = features[col].rolling(
            str(window)+'d').max().shift(1)
        patient_ml['std_'+col+'_'+str(window)+'_days'] = features[col].rolling(
            str(window)+'d').std().shift(1)
    return patient_ml


def split_dataset(x, y, split_index):
    return (x[:split_index], y[:split_index], x[split_index:], y[split_index:])


def generate_rand_color():
    return "%06x" % random.randint(0, 0xFFFFFF)

def get_all_patients():
    meta_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678_META.csv')
    patients_sorted = meta_EMA.sort_values(by=['xEmaNRatings'], ascending=False)

    return patients_sorted

def get_all_proper_patients():
    meta_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678_META.csv')
    patients_sorted = meta_EMA.sort_values(by=['xEmaNRatings'], ascending=False)
    patients = []

    for _, patient in patients_sorted.iterrows():
        if get_patient_features(patient['ECD_ID']):

            patient_obj = {
                'patient_id': str(patient['ECD_ID'])
            }

            patients.append(patient_obj)

    return patients

def get_proper_patients(max_patients=20):
    """Returns all patients who answered queried questions at least once, and initiated at least once."""
    meta_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678_META.csv')
    patients_sorted = meta_EMA.sort_values(by=['xEmaNRatings'], ascending=False)
    patients = []

    for _, patient in patients_sorted.iterrows():
        if get_patient_features(patient['ECD_ID']):

            patient_obj = {
                'patient_id': str(patient['ECD_ID'])
            }

            patients.append(patient_obj)

    return patients[:max_patients]


def get_patient_id_by_rank(ratings_rank):
    meta_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678_META.csv')
    patients_sorted = meta_EMA.sort_values(
        by=['xEmaNRatings'], ascending=False)
    return patients_sorted['ECD_ID'][ratings_rank]


def get_average_MAE(eval_models):
    results = []

    for model in eval_models:
        results.append(model['score']['mae'])

    return mean(results)


def get_all_MAE(eval_models):
    results = []

    for model in eval_models:
        results.append({
            'model': model['name'],
            'MAE': model['score']['mae'],
        })

    return results
