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
full_EMA.drop(['xEmaDate_xEmaTime'], axis=1)

#%%
# Get the value of the first patient available
sample_patient = full_EMA.loc[0]['ECD_ID']
pd_sample_patient = get_patient_values(full_EMA, sample_patient)

#%%
# Separate and resample the dates
pd_sample_patient.index = pd_sample_patient['xEmaDate']
pd_sample_patient_ts = pd_sample_patient['xEmaSchedule'].resample('1d').count()
