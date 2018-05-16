#%%
import pandas as pd
import numpy as np
import matplotlib as plot
import matplotlib.pylab as plt
import sklearn as sk
import seaborn as sns
import importlib
from helpers import *
%matplotlib inline

# TODO: Get number of times the user intiated the logging by themselves.
#%%
###
#   Reload Modules
#   Reload modules! Execute this cell if you want to reload
#   imported modules.
##

importlib.reload(helpers)

#%%
full_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678.csv',
                       parse_dates=[['xEmaDate', 'xEmaTime']])
full_EMA['xEmaDate'] = full_EMA['xEmaDate_xEmaTime']
full_EMA = full_EMA.drop(['xEmaDate_xEmaTime'], axis=1)

#%%
# Get the value of the first patient available
sample_patient = full_EMA.loc[12000]['ECD_ID']
pd_sample_patient = get_patient_values(full_EMA, sample_patient)
pd_sample_patient.index = pd_sample_patient['xEmaDate']

#%%
# Get representation of each of the moods of the patient
pd_sample_patient_moods = pd.get_dummies(
    pd_sample_patient['xEmaQuestion'], prefix='mood_question')
pd_sample_patient = pd_sample_patient.join(pd_sample_patient_moods)


#%%
test = pd_sample_patient_moods.multiply(
    pd_sample_patient['xEmaRating'], axis='index')
#%%
test2 = test.rename(
    columns={c: c+'_answer' for c in test.columns})
pd_sample_patient = pd_sample_patient.drop(
    ['xEmaQuestion', 'xEmaRating', 'xEmaDate'], axis=1)

pd_sample_patient_self_initiated = pd_sample_patient[pd_sample_patient['xEmaSchedule'] == 4]
pd_sample_patient = pd_sample_patient[pd_sample_patient['xEmaSchedule'] != 4]

#%%
# Features to create
# TODO: Mean/min/max/std/slope for each of the five moods of the last lambda-days.
# TODO: Phase 1: Engement definition of previous day.
# TODO: The engagement definitions.
# TODO: Nr. modules completed over lambda-days.
# TODO: Etc. module info
# TODO: User-initiated modules.

#%%
sliding_window = 7
# sample_patient_features = pd.Dataframe

mood_question_columns = pd_sample_patient.filter(regex="mood_question_\d")
mood_answer_columns = pd_sample_patient.filter(regex=".*answer")
pd_resampled_days = pd_sample_patient.resample('1d')

sample_patient_features = pd.DataFrame(index=pd_resampled_days.index)

for col in mood_question_columns:
    resampled_col = mood_question_columns[col].resample('1d').sum()
    sample_patient_features[resampled_col.name] = resampled_col


#%%
# Getting a number of generic statistics for the different features
sample_patient_ml_features = pd.DataFrame(index=sample_patient_features.index)
for col in sample_patient_features.fillna(0):
    sample_patient_ml_features['avg_'+col] = sample_patient_features[col].rolling(
        str(sliding_window)+'d').mean().shift(1)
    sample_patient_ml_features['min_'+col] = sample_patient_features[col].rolling(
        str(sliding_window)+'d').min().shift(1)
    sample_patient_ml_features['max_'+col] = sample_patient_features[col].rolling(
        str(sliding_window)+'d').max().shift(1)
    sample_patient_ml_features['std_'+col] = sample_patient_features[col].rolling(
        str(sliding_window)+'d').std().shift(1)
#%%
#Amin questions
#TODO: Inconsistent data-set seemingly; does not seemingly match the schedule given in the manual (look at first week, and then on the following weeks;)
#TODO:

sample_patient_ml_features = sample_patient_ml_features.fillna(0)


#%%
pd_sample_patient['xEmaSchedule'].resample('1d').count()
#%%
q_asked = pd_sample_patient['xEmaSchedule'].resample('1d').count()

q_asked[:7] = 10
q_asked[len(q_asked) - 7:] = 10

q_asked[7:len(q_asked) - 7][q_asked > 1] = 10
q_asked[q_asked <= 1] = 1

pd.DataFrame([q_asked, pd_sample_patient['xEmaSchedule'].resample('1d').count()]).transpose()
#%%

pd_engagement = pd_sample_patient['xEmaSchedule'].resample('1d') / q_asked

# MVP-first:
# 1. Get the basic features
# 2. Get some random algorithm to predict something.
# 3. That's all for the MVP!

#%%
from sklearn import linear_model

reg = linear_model.Lasso(alpha=0.1)
pd_engagement = pd_engagement.fillna(0)
pd_engagement.fillna(0)
#%%

reg.fit(sample_patient_ml_features, pd_engagement)
