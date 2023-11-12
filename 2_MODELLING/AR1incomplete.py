#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AR1

@author: nataliacardenasf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math
import itertools
import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import power_transform

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error as mse
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential

from tqdm import tqdm
import concurrent.futures


#Set personal path
import os 
os.chdir('/Users/nataliacardenasf/Documents/GitHub/Population-growth-forecast-using-an-ARIMA-SARIMA-ANN-LSTM')


#%% Start - Import cleaned data and check quality 
COUNTRIES =['Japan', 'France', 'Usa', 'Colombia', 'Sweden'] #Scope of our analysis
FILE_NAME= '1_DATA/population_data_bis.xlsx'                #Change if needed


def determine_differencing_order_for_all(data):
    stat={}
    for column in data.columns:
        series = data[column].dropna()
        d = 0

        while True:
            # Perform Augmented Dickey-Fuller test for stationarity
            result = adfuller(series)

            print(f"Results for column '{column}' (I({d})):")
            print(f"ADF Statistic: {result[0]}")
            print(f"P-Value: {result[1]}")
            print("Critical Values:")
            for key, value in result[4].items():
                print(f"  {key}: {value}")

            if result[1] <= 0.05:
                print(f"Series '{column}' is stationary at order I({d}) (reject null hypothesis)")
                break
            else:
                print(f"Series '{column}' is non-stationary at order I({d}) (fail to reject null hypothesis)")
                d += 1
                series = series.diff().dropna()  # Differencing the series
            print('\n')
        stat[column]=d

    return(stat)

def replace_outliers_with_moving_average(data, window_size, z_score_threshold=3):
    """
    Replaces outliers in a DataFrame with the moving average.

    :param data: A pandas DataFrame with numerical data
    :param window_size: The window size for calculating the moving average
    :param z_score_threshold: The z-score threshold to identify outliers
    :return: DataFrame with outliers replaced by moving averages
    """
    # Create a copy of the data to avoid modifying the original DataFrame
    data_cleaned = data.copy()

    # Iterate through each column
    for column in data.columns:
        # Skip non-numeric columns
        if not np.issubdtype(data[column].dtype, np.number):
            continue

        # Calculate the moving average
        moving_avg = data[column].rolling(window=window_size, min_periods=1).mean()

        # Calculate the z-score for each value
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())

        # Identify outliers
        outliers = z_scores > z_score_threshold

        # Replace outliers with moving average
        data_cleaned[column].where(~outliers, moving_avg, inplace=True)

    return data_cleaned

data = pd.read_excel(FILE_NAME)
data = data.set_index('Date')
data = np.log(data)
data.plot()

#get integration order
order = determine_differencing_order_for_all(data)
order 

#outliers in deltas, we take care of that here, from  here on we work with series in log-deltas (equivalent to a growth rate)
data = replace_outliers_with_moving_average(data, 12)
order = determine_differencing_order_for_all(data.diff().dropna())
order # we still find that series (in log deltas) for JP, FR, USA, Sweden are I(1) ie series in levels are I(2)


def decompose_data(data):
    decomposed_data = {}
    dec_trend=pd.DataFrame()

    for column in data.columns:
        # Decompose the time series
        decomposition = sm.tsa.seasonal_decompose(data[column].dropna(), model='additive')
        dec_trend[column]=decomposition.trend
        # Store the decomposition components in a dictionary
        decomposed_data[column] = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }

        # Visualize the decomposed components
        plt.figure(figsize=(12, 6))
        plt.subplot(411)
        plt.plot(decomposed_data[column]['trend'])
        plt.title(f'Trend of {column}')
        plt.subplot(412)
        plt.plot(decomposed_data[column]['seasonal'])
        plt.title(f'Seasonal of {column}')
        plt.subplot(413)
        plt.plot(decomposed_data[column]['residual'])
        plt.title(f'Residual of {column}')
        plt.subplot(414)
        plt.plot(data[column])
        plt.title(f'Original {column}')
        plt.tight_layout()
        plt.show()

    return dec_trend

decomposed_data = decompose_data(data)


#%%Get relevant data for our analysis ie before COVID
DATA_BL = data.copy() #store initial df to compare results
data2 =  data[data.index < datetime.datetime(2020,1,1)]

order2 = determine_differencing_order_for_all(data2)

dtastat = data2.copy()
for x in ['Japan', "France", "Usa", "Sweden"]:
    dtastat[x] = dtastat[x].diff().dropna()

o = determine_differencing_order_for_all(dtastat) #Everything is stationary 

decomp_stat = decompose_data(dtastat)

#%% Get lags 

# First option is to look at APC graphs, for all but FR we have only one relevant lag 
# Plot partial autocorrelation
for x in data2.columns:
    plt.rc("figure", figsize=(11,5))
    plot_pacf(data[x], method='ywm')
    plt.xlabel('Lags', fontsize=18)
    plt.ylabel('Correlation', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'Partial Autocorrelation Plot for {x}', fontsize=20)
    plt.tight_layout()
    plt.show()
    

