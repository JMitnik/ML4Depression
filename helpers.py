from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib as plot
import matplotlib.pylab as plt

def get_patient_values(dataframe, patient_id):
    return dataframe[dataframe['ECD_ID'] == patient_id]

def test_stationarity(ts):
    # Determine rolling statistics
    rol_mean = pd.rolling_mean(ts, window=7)
    rol_std = pd.rolling_std(ts, window=7)

    # Plotting these
    orig = plt.plot(ts, color='blue', label='Original')
    mean = plt.plot(rol_mean, color='red', label='Rolling Mean')
    std = plt.plot(rol_std, color='black', label='Rolling Stdev')
    plt.legend(loc='best')
    plt.title('Rolling mean & Standard Deviation')
    plt.show(block=False)
