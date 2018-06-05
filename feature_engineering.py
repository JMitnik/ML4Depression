#%%
# Importing libraries and dependencies
import pandas as pd
import numpy as np
import matplotlib as plot
import matplotlib.pylab as plt
import sklearn as sk
import seaborn as sns
import importlib
from helpers import *

# region [cell] Init variables
#%%
SLIDING_WINDOW = 7
SAMPLE_TIME = '1d'
# endregion

#%%
# Reading all of the EMA data, and fusing the date and time together.


def read_EMA_code():
    full_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678.csv',
                           parse_dates=[['xEmaDate', 'xEmaTime']])
    full_EMA['xEmaDate'] = full_EMA['xEmaDate_xEmaTime']
    full_EMA = full_EMA.drop(['xEmaDate_xEmaTime'], axis=1)
    return full_EMA


def get_patient_by_rank(ratings_rank):
    meta_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678_META.csv')
    patients_sorted = meta_EMA.sort_values(
        by=['xEmaNRatings'], ascending=False)
    return patients_sorted['ECD_ID'][ratings_rank]


def init_patient(full_EMA, patient_id):
    patient_df = full_EMA[full_EMA['ECD_ID'] == patient_id]
    patient_df.index = patient_df['xEmaDate']
    return patient_df


def split_self_init_sessions(patient_df):
    patient_self_init_df = patient_df[patient_df['xEmaSchedule'] == 4]
    patient_df = patient_df[patient_df['xEmaSchedule'] != 4]
    return (patient_df, patient_self_init_df)


def get_patient_features(full_EMA, patient_id):
    patient_df = init_patient(full_EMA, patient_id)
    patient_df = extract_ema_vals(patient_df)
    patient_df, patient_self_init_df = split_self_init_sessions(patient_df)
    patient_df = resample_patient_dataframe(patient_df, SAMPLE_TIME)
    return patient_df

def extract_ema_vals(patient_df):
    ema_q = one_hot_encode_feature(patient_df, 'xEmaQuestion', prefix='ema_q')
    ema_a = ema_q.multiply(patient_df['xEmaRating'], axis='index')
    ema_q = ema_q.add_prefix('count_')
    ema_a = ema_a.add_prefix('average_')

    patient_df = patient_df.join([ema_q, ema_a])
    return patient_df

def one_hot_encode_feature(patient_df, feature, prefix):
    return pd.get_dummies(patient_df[feature], prefix=prefix)


def resample_datetimeindex(dt_index, sample_time):
    date_min = dt_index.min()
    date_max = dt_index.max() + pd.DateOffset(days=1)
    resampled_date = pd.date_range(date_min, date_max, freq=sample_time)[:-1]
    fill_data = pd.Series(resampled_date, resampled_date.floor(sample_time))
    return fill_data.index

def resample_patient_dataframe(patient_df, sample_time):
    ema_q_resampled = count_patient_ema_questions(patient_df, sample_time)
    ema_a_resampled = avg_patient_ema_values(patient_df, sample_time)
    return ema_q_resampled.join(ema_a_resampled)

def avg_patient_ema_values(patient_df, sample_time):
    ema_columns = patient_df.filter(regex="average_ema_*.")
    patient_df = pd.DataFrame(index=resample_datetimeindex(patient_df.index, sample_time))
    patient_df = patient_df.join(ema_columns.resample(sample_time).mean())
    return patient_df

def count_patient_ema_questions(patient_df, sample_time):
    ema_columns = patient_df.filter(regex="count_ema_*.")
    # ! Left-off: For some reason, the first column is not displaying anything here
    patient_df = pd.DataFrame(index=resample_datetimeindex(patient_df.index, sample_time))
    patient_df = patient_df.join(ema_columns.resample(sample_time).sum())
    return patient_df

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

# region [cell] Initiating the code
#%%
full_EMA = read_EMA_code()
sample_patient_id = get_patient_by_rank(4)
sample_patient_features = get_patient_features(full_EMA, sample_patient_id)

sample_patient_features
# sample_patient_ML_features = convert_features_to_statistics(sample_patient_features, SLIDING_WINDOW)
# endregion


#region [todo] Defining engagement
#%%
# TODO: Get a better representation than this
patient_q_asked = pd_sample_patient['xEmaSchedule'].resample('1d').count()
patient_q_asked[:7] = 10
patient_q_asked[len(patient_q_asked) - 7:] = 10
patient_q_asked[7:len(patient_q_asked) - 7][patient_q_asked > 1] = 10
patient_q_asked[patient_q_asked <= 1] = 1

pd_engagement = pd_sample_patient['xEmaSchedule'].resample(
    '1d') / patient_q_asked
pd_engagement = pd_engagement.fillna(0)

patient_base_features = patient_base_features.join(pd_engagement).rename(
    columns=({'xEmaSchedule': 'actual_engagement'}))
# endregion
