import pandas as pd
import random

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
