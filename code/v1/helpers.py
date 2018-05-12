#%%
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk
import importlib

def get_user_data(user_table, patient_id):
    return user_table[user_table['codPatient'] == patient_id]

def sort_dataframe_by_col(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name])
    return df.sort_values(by=[col_name])
