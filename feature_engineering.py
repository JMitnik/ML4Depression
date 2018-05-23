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
%matplotlib inline

#%%
# Reading all of the EMA data, and fusing the date and time together.
full_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678.csv',
                       parse_dates=[['xEmaDate', 'xEmaTime']])
full_EMA['xEmaDate'] = full_EMA['xEmaDate_xEmaTime']
full_EMA = full_EMA.drop(['xEmaDate_xEmaTime'], axis=1)

#%%
# Per-patient basis: Looking at one patient for now
sample_patient = full_EMA.loc[12000]['ECD_ID']
pd_sample_patient = get_patient_values(full_EMA, sample_patient)
pd_sample_patient.index = pd_sample_patient['xEmaDate']

# Get representation of each of the moods of the patient
pd_sample_patient_moods = pd.get_dummies(
    pd_sample_patient['xEmaQuestion'], prefix='mood_question')
pd_sample_patient = pd_sample_patient.join(pd_sample_patient_moods)

patient_moods_index = pd_sample_patient_moods.multiply(
    pd_sample_patient['xEmaRating'], axis='index')
patient_moods_index = patient_moods_index.rename(
    columns={c: c+'_answer' for c in patient_moods_index.columns})
pd_sample_patient = pd_sample_patient.drop(
    ['xEmaQuestion', 'xEmaRating', 'xEmaDate'], axis=1)

pd_sample_patient_self_initiated = pd_sample_patient[pd_sample_patient['xEmaSchedule'] == 4]
pd_sample_patient = pd_sample_patient[pd_sample_patient['xEmaSchedule'] != 4]

sliding_window = 7

mood_question_columns = pd_sample_patient.filter(regex="mood_question_\d")
pd_resampled_days = pd_sample_patient.resample('1d')

patient_base_features = pd.DataFrame(index=pd_resampled_days.index)

for col in mood_question_columns:
    resampled_col = mood_question_columns[col].resample('1d').sum()
    patient_base_features[resampled_col.name] = resampled_col

# Getting a number of generic statistics for the different features
patient_ml_features = pd.DataFrame(index=patient_base_features.index)

for col in patient_base_features.fillna(0):
    patient_ml_features['avg_'+col+'_'+sliding_window+'_days'] = patient_base_features[col].rolling(
        str(sliding_window)+'d').mean().shift(1)
    patient_ml_features['min_'+col+'_'+sliding_window+'_days'] = patient_base_features[col].rolling(
        str(sliding_window)+'d').min().shift(1)
    patient_ml_features['max_'+col+'_'+sliding_window+'_days'] = patient_base_features[col].rolling(
        str(sliding_window)+'d').max().shift(1)
    patient_ml_features['std_'+col+'_'+sliding_window+'_days'] = patient_base_features[col].rolling(
        str(sliding_window)+'d').std().shift(1)

patient_ml_features = patient_ml_features.fillna(0)

#TODO: Get a better representation than this.
patient_q_asked = pd_sample_patient['xEmaSchedule'].resample('1d').count()
patient_q_asked[:7] = 10
patient_q_asked[len(patient_q_asked) - 7:] = 10
patient_q_asked[7:len(patient_q_asked) - 7][patient_q_asked > 1] = 10
patient_q_asked[patient_q_asked <= 1] = 1

pd_engagement = pd_sample_patient['xEmaSchedule'].resample('1d') / q_asked
pd_engagement = pd_engagement.fillna(0)
pd_engagement.fillna(0)

# This is eventually what we export
patient_x = patient_ml_features
patient_y = pd_engagement
