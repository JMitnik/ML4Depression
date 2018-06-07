#%%
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk
import importlib

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
# Grab the biggest index of both 'mergable sets'
# TODO - Join whichever is biggest
# * Possible method: biggest_index = patient_mod.index if len(patient_mod.index) > len(patient_ml_features.index) else patient_ml_features.index
full_patient = patient_ml_features.join(
    [patient_mod_total_time, patient_mod_total_pages, patient_mod_total_sessions])

# Shave off first and last week
patient_x = full_patient[7:-7].fillna(0)
patient_y = patient_y[7:-7].fillna(0)

#%%
# Day of the week next up
weekend_days = patient_x.index.to_series().apply(lambda x: (
    x.isoweekday() == 6 or x.isoweekday() == 7)).astype(int)
patient_x['weekend_day'] = weekend_days

#%%
patient_x
