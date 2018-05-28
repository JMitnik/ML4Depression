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

#%%
# Reading all of the EMA data, and fusing the date and time together.
full_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678.csv',
                       parse_dates=[['xEmaDate', 'xEmaTime']])
full_EMA['xEmaDate'] = full_EMA['xEmaDate_xEmaTime']
full_EMA = full_EMA.drop(['xEmaDate_xEmaTime'], axis=1)
meta_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678_META.csv')

#%%
patients_sorted = meta_EMA.sort_values(by=['xEmaNRatings'], ascending=False)

#%%
# Per-patient basis: Looking at one patient for now
sample_patient = patients_sorted['ECD_ID'][4]
pd_sample_patient = get_patient_values(full_EMA, sample_patient)
pd_sample_patient.index = pd_sample_patient['xEmaDate']

#%%
pd_sample_patient
#%%

# Get representation of each of the moods of the patient
pd_sample_patient_moods = pd.get_dummies(
    pd_sample_patient['xEmaQuestion'], prefix='ema_question')
pd_sample_patient = pd_sample_patient.join(pd_sample_patient_moods)

patient_moods_index = pd_sample_patient_moods.multiply(
    pd_sample_patient['xEmaRating'], axis='index')
patient_moods_index = patient_moods_index.rename(
    columns={c: c+'_answer' for c in patient_moods_index.columns})
pd_sample_patient = pd_sample_patient.drop(
    ['xEmaQuestion', 'xEmaRating', 'xEmaDate'], axis=1)

#%%
pd_sample_patient
#%%
pd_sample_patient_self_initiated = pd_sample_patient[pd_sample_patient['xEmaSchedule'] == 4]
pd_sample_patient = pd_sample_patient[pd_sample_patient['xEmaSchedule'] != 4]

sliding_window = 7

mood_question_columns = pd_sample_patient.filter(regex="ema_question_\d")
pd_resampled_days = pd_sample_patient.resample('1d')

patient_base_features = pd.DataFrame(index=pd_resampled_days.index)

for col in mood_question_columns:
    resampled_col = mood_question_columns[col].resample('1d').sum()
    patient_base_features[resampled_col.name] = resampled_col

# Getting a number of generic statistics for the different features
patient_ml_features = pd.DataFrame(index=patient_base_features.index)

# TODO: Get a better representation than this
patient_q_asked = pd_sample_patient['xEmaSchedule'].resample('1d').count()
patient_q_asked[:7] = 10
patient_q_asked[len(patient_q_asked) - 7:] = 10
patient_q_asked[7:len(patient_q_asked) - 7][patient_q_asked > 1] = 10
patient_q_asked[patient_q_asked <= 1] = 1

pd_engagement = pd_sample_patient['xEmaSchedule'].resample(
    '1d') / patient_q_asked
pd_engagement = pd_engagement.fillna(0)

#%%
patient_base_features = patient_base_features.join(pd_engagement).rename(
    columns=({'xEmaSchedule': 'actual_engagement'}))
#%%
for col in patient_base_features.fillna(0):
    patient_ml_features['avg_'+col+'_'+str(sliding_window)+'_days'] = patient_base_features[col].rolling(
        str(sliding_window)+'d').mean().shift(1)
    patient_ml_features['min_'+col+'_'+str(sliding_window)+'_days'] = patient_base_features[col].rolling(
        str(sliding_window)+'d').min().shift(1)
    patient_ml_features['max_'+col+'_'+str(sliding_window)+'_days'] = patient_base_features[col].rolling(
        str(sliding_window)+'d').max().shift(1)
    patient_ml_features['std_'+col+'_'+str(sliding_window)+'_days'] = patient_base_features[col].rolling(
        str(sliding_window)+'d').std().shift(1)

patient_ml_features = patient_ml_features.fillna(0)

# This is eventually what we export
patient_x = patient_ml_features
patient_y = pd_engagement

#%%
# Reading Module data
full_mod = pd.read_csv('data/v2/ema_logs/ECD_Y001.csv')
full_mod['yDateTime'] = pd.to_datetime(full_mod['yDateTime'])
patient_mod = full_mod[full_mod['ECD_ID'] == sample_patient]
patient_mod = patient_mod.set_index(['yDateTime'])

#%%
patient_mod_total_time = patient_mod['yDuration'].resample('1d').sum()

patient_mod_total_pages = (1 + patient_mod['yPage'].resample(
    '1d').max() - patient_mod['yPage'].resample('1d').min()).fillna(0)

patient_mod_total_sessions = (1 + patient_mod['ySession'].resample(
    '1d').max() - patient_mod['ySession'].resample('1d').min()).fillna(0)

#%%
