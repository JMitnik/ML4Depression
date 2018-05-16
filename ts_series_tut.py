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

#%%
# Start of the tutorial. Let's plot the time-series first
plt.plot(pd_sample_patient_ts)

#%%
# Let's plot the rolling average and rolling standard-deviation
test_stationarity(pd_sample_patient_ts)

#%%
# Making Test Statistics more stationary
# Method 1: Penalizing higher values using logs

ts_log = np.log(pd_sample_patient_ts)
moving_avg = pd.rolling_mean(ts_log, 7)
# plt.plot(ts_log)
# plt.plot(moving_avg, color='red')

# Now we have the rolling average, let's detract this from our punished values.
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)

# test_stationarity(ts_log_moving_avg_diff)

#%%
# Let's now assign an exponential moving weighted average to give more priority to recent values
expweighted_avg = pd.ewma(ts_log, halflife=7)

ts_log_ewma_diff_ts = ts_log - expweighted_avg
test_stationarity(ts_log_ewma_diff_ts)

#%%
# Now, let us remove trend and seasonality using shift
ts_log_shift = ts_log - ts_log.shift()
ts_log_shift.dropna(inplace=True)
test_stationarity(ts_log_shift)
