#%%
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk

def get_module_features_for_patient(patient_id):
    full_mod = read_module_data()
    patient_df = init_patient(full_mod, patient_id)
    patient_mod_features = make_patient_mod_features(patient_df)

    return patient_mod_features

def read_module_data():
    full_mod = pd.read_csv('data/v2/ema_logs/ECD_Y001.csv')
    full_mod['yDateTime'] = pd.to_datetime(full_mod['yDateTime'])
    return full_mod

def init_patient(full_mod, patient_id):
    patient_df = full_mod[full_mod['ECD_ID'] == patient_id]
    patient_df = patient_df.set_index(['yDateTime'])
    return patient_df

def make_patient_mod_features(patient_df):
    patient_mod_total_time = patient_df['yDuration'].resample('1d').sum().rename('mod_total_time')

    patient_mod_total_pages = (1 + patient_df['yPage'].resample(
        '1d').max() - patient_df['yPage'].resample('1d').min()).fillna(0).rename('mod_total_pages')

    patient_mod_sessions = (1 + patient_df['ySession'].resample(
        '1d').max() - patient_df['ySession'].resample('1d').min()).fillna(0).rename('mod_nr_sessions')

    return pd.DataFrame([patient_mod_total_time, patient_mod_total_pages, patient_mod_sessions])
