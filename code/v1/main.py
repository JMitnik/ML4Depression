#%%
import pandas as pd
import numpy as np
import matplotlib as plot
import sklearn as sk
from helpers import *

# TODO: Amin - What is relapse prevention?
# TODO: Amin - What is truePatient?

EMA_DATA_PATH = 'data/TimeEma_Mark_20180301_1404.csv'
SEQ_MOD_DATA_PATH = 'data/PatientLogs.xlsx'
USER_DATA_PATH = 'data/PatientInfo.xlsx'

#%%
###
#   Reading Data
###
users_data = pd.read_excel(USER_DATA_PATH)

module_data = pd.read_excel(SEQ_MOD_DATA_PATH)
module_data = sort_dataframe_by_col(module_data, 'serverTimeStamp')

ema_data = pd.read_csv(EMA_DATA_PATH, error_bad_lines=False, sep=';')
ema_data = sort_dataframe_by_col(ema_data, 'scheduleAt')

#%%
###
#   Reading user data
###

# Meaningful data
code_users = users_data[users_data['codPatient'].notnull()]
self_treated_users = users_data[users_data['onSelfTreatment'] == 1]
non_self_treated_users = users_data[users_data['onSelfTreatment'] == 0]
languages_users = users_data[users_data['languageId'].notnull()]
#%%
###
#   Reading module data
###
web_entries = module_data[module_data['systemType'] == 'web']
mobile_entries = module_data[module_data['systemType'] == 'mobile']

#TODO: More can be done here

#%%
###
#   Reading EMA data
###
# TODO: Filter out test-patients (fake ID's)

# Interesting data
values_with_ratings = ema_data[ema_data['rateValue'].notnull()]
values_without_ratings = ema_data[ema_data['rateValue'].isnull()]

# Mood, sleep, esteem, worrying, enjoy, social
mood_query_types = ema_data['ratingType'].unique()


def filter_ratings_only(dataframe):
    return dataframe[dataframe['rateValue'].notnull()]


#%%
values_with_ratings.to_csv('data/sorted_values_with_ratings.csv')
values_without_ratings.to_csv('data/sorted_values_without_ratings.csv')


# END OF BASIC EXPLORATION!

#%%
# App Engagement, how do we measure this?

# EMA
# EMA data gives us a number of estimates for our patients: the number of times they rated their mood.
# ServerTimeStamp is the time when it received the request. ScheduleAt is when they receive it.
# Using our 'assumed' timezone, we could do something with it.

# Let's start with simply frequencies. Let's think about it in terms of patterns.

# On which levels can we look?
# 1. Patient-level
# Let's start with getting only the relevant patients who occur at least once within the rated dataframe.

activePatients = values_with_ratings[' patientId'].unique()

# Now, for each patient, we want to do certain activities. Let's start with getting their user-data.


def get_EMA_for_user(EMA_dataframe, patient_id):
    return EMA_dataframe[EMA_dataframe[' patientId'] == patient_id]


#%%
granularity = "day"

#TODO: Undo the [1:]
#TODO: Find out about the duplicates

patient = activePatients[1]
user_info = get_user_data(patient)
# Next, now for this patient, we might want to get all their EMA values, ranked or non-ranked
user_EMA_values = get_EMA_for_user(ema_data, patient)

# Sets ScheduleAt as index
user_EMA_values['scheduleAt'] = pd.to_datetime(user_EMA_values['scheduleAt'])
user_EMA_values = user_EMA_values.set_index('scheduleAt')

# Removes the duplicates
user_EMA_values = user_EMA_values[~user_EMA_values.index.duplicated()]

#%%
upd = user_EMA_values
cols = ['scheduleAt', 'rateValue']
window_size = 5

for col in cols:
    upd[col + '_temp_freq_' + str(window_size)] = np.nan

#%%
# Let's now get the frequencies in general of upd
upd.resample('1d', how='count')
dailySchedule = upd['scheduleDate'].resample('1d', how='count')
dailyEntered = upd['rateValue'].resample('1d', how='count')

#%%
new_data_frame = pd.DataFrame([dailySchedule, dailyEntered]).transpose()

#%%
entries = new_data_frame[new_data_frame['scheduleDate'] > 0]
entries['engagement'] = entries['rateValue'] / entries['scheduleDate']

#%%
# Let's read in the estimated time-zones
est_time = pd.read_csv('data/estimated_TimeZone.csv', sep=';')
est_time

#%%
# Let's merge the timezone
merged_user_data = users_data.merge(right=est_time, how='inner', left_on='codPatient', right_on='patientID')
merged_user_data[merged_user_data['estimated_delay'] > 1]

#%%
# upd
# upd.iloc[0][' patientId']
# est_time.loc[est_time['patientID'] == 'D7F80B4895']
entered_values = user_EMA_values[user_EMA_values['serverTimeStamp'].notnull()]
entered_values['serverTimeStamp'] = pd.to_datetime(entered_values['serverTimeStamp'])
test = entered_values['serverTimeStamp'] - entered_values.index
# user_EMA_values

#%%
#TODO: Fix this better
#TODO: Remove outliers
test = test[test > pd.Timedelta(0, 's')]

#%%
def to_seconds(row):
    print(row.seconds)
    return row.seconds
test = test.apply(to_seconds)

#%%
test.plot.box()
# test.mean()
# test.iloc[4] > pd.Timedelta(0, 's')
# test.resample('1d').mean()

#%%
#TODO: Confirm we can match this?
resampled_time = test.resample('1d', how='mean')
resampled_time[resampled_time.notnull()]
entries['avg_response_time'] = resampled_time[resampled_time.notnull()]

#%%
entries.to_hdf('data/patientA.h5', 'entries')

#%%
testhd = pd.read_hdf('data/patientA.h5')
testhd.head(30)
#%%
testhd.columns = ['nr_asked', 'nr_responded', 'engagement', 'avg_response_time']

testhd

#%%
testhd.to_hdf('data/v2-patientA.h5', 'patientA')
