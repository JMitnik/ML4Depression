#%%
import pandas as pd
import numpy as np
import matplotlib as plot
import matplotlib.pylab as plt
import sklearn as sk
import seaborn as sns
import importlib
%matplotlib inline

#%%
#%%
full_EMA = pd.read_csv('data/v2/ema_logs/ECD_X970_12345678.csv', parse_dates=[['xEmaDate', 'xEmaTime']])
full_EMA['xEmaDate'] = full_EMA['xEmaDate_xEmaTime']
full_EMA.drop(['xEmaDate_xEmaTime'], axis=1)
#%%
# Let's Merge EMADate with EmaTime
full_EMA
