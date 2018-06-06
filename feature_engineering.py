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
patients_sorted = meta_EMA.sort_values(by=['xEmaNRatings'], ascending=False)
sample_patient_id = patients_sorted['ECD_ID'][4]

#%%
# Per-patient basis: Looking at one patient for now
patient = Patient(sample_patient_id, full_EMA)
patient.set_feature(patient.records['xEmaQuestion'], feature_prefix='ema_question')

#TODO: For refactor, multiply this in the object or in here?
patient_moods_index = pd_sample_patient_moods.multiply(
    pd_sample_patient['xEmaRating'], axis='index')
patient_moods_index = patient_moods_index.rename(
    columns={c: c+'_answer' for c in patient_moods_index.columns})

#TODO: For refactor, drop this feature via interface
pd_sample_patient = pd_sample_patient.drop(
    ['xEmaQuestion', 'xEmaRating', 'xEmaDate'], axis=1)

#TODO: Make distinction in the class using a private method.
pd_sample_patient_self_initiated = pd_sample_patient[pd_sample_patient['xEmaSchedule'] == 4]

#TODO: Make distinction in the class using a private method.
pd_sample_patient = pd_sample_patient[pd_sample_patient['xEmaSchedule'] != 4]

#TODO: This stays here
sliding_window = 7


#TODO: We create a patient method for this, but pass the 'dataframe' or collection of columns from here.
mood_question_columns = pd_sample_patient.filter(regex="ema_question_\d")
pd_resampled_days = pd_sample_patient.resample('1d')

patient_base_features = pd.DataFrame(index=pd_resampled_days.index)

for col in mood_question_columns:
    resampled_col = mood_question_columns[col].resample('1d').sum()
    patient_base_features[resampled_col.name] = resampled_col

# Getting a number of generic statistics for the different features
patient_ml_features = pd.DataFrame(index=patient_base_features.index)

#%%
# We define our target value, the engagement, here
patient_self_init = pd_sample_patient_self_initiated.resample('1d').count()['xEmaSchedule']
patient_nr_answered_questions = patient_base_features.sum(axis=1)
self_init = patient_base_features.join(patient_self_init).fillna(0).rename(columns=({'xEmaSchedule': 'self_init'}))['self_init']
boolean_answered = patient_nr_answered_questions.apply(lambda row: min(1, row))
boolean_initiated = self_init.apply(lambda row: min(1, row))

pd_engagement = (self_init + boolean_answered + boolean_initiated).rename('actual_engagement')
#%%
patient_base_features = patient_base_features.join(pd_engagement)

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
