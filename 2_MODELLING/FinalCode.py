#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population growth forecast

Second do file. We are working with data that is already cleaned! 

@author: nataliacardenasf
"""

#Used packages in this file 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math
import itertools

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error as mse
import statsmodels.api as sm

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

#create function to open file with data and set index 
def process_data(path, countries=[]):
    df = pd.read_excel(path)
    df=df.set_index('Date')

    return df,np.log(df).dropna() #np.log(df).pct_change().dropna()

df,data = process_data(FILE_NAME,COUNTRIES)

data.columns #ok 
data #ok 


#Don't have any missing values after cleaning in previous file: we can continue 
def plot_nan_percentage(data):
    nan_percentage = (data.isnull().sum() / len(data)) * 100
    nan_percentage = nan_percentage[nan_percentage >=0]  # Filter out columns with no missing values

    # Plotting
    plt.figure(figsize=(12, 6))
    nan_percentage.sort_values(ascending=False).plot(kind='bar')
    plt.title('Percentage of Missing Values in Columns')
    plt.xlabel('Columns')
    plt.ylabel('Percentage of Missing Values')
    plt.show()

plot_nan_percentage(data) # all good, no missing values

#Common timeframe across all countries 
def nan_intime(output):
    missing_data = output.isnull()

    # Find the common time frame (min and max)
    treshold=output.dropna()
    common_min = treshold.index.min()
    common_max = treshold.index.max()

    # Create a line plot with missing data highlighted
    plt.figure(figsize=(12, 6))

    for country in output.columns:
        plt.plot(output.index, output[country], label=country)

    # Highlight missing data in red
    for country in output.columns:
        plt.scatter(output.index[missing_data[country]], output[country][missing_data[country]], c='red', s=10)
        print(f'Country : {country}, From {output[country].dropna().index.min()} to  {output[country].dropna().index.max()}')

    # Add a vertical line for the common time frame
    plt.axvline(common_min, color='green', linestyle='--', label='Common Min')
    plt.axvline(common_max, color='blue', linestyle='--', label='Common Max')

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(f'TimeFrame with no null data is from {common_min} to {common_max}')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage => there are some outliers (spikes) in the data, we prefer to replace with MA
nan_intime(data)

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
data = replace_outliers_with_moving_average(data,12)




#Plot data 
sns.pairplot(data)


#%% Stationarity checks 

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

'''
We need to check how to define the variables to get statioarity 
'''

orders =determine_differencing_order_for_all(data) # 2, 0, 2, 1, 2
orders

#1, 1, 1, 0, 1
data_level = pd.read_excel(FILE_NAME)
data_level = data_level.set_index('Date')
data_level = np.log(data_level)
data_level = data_level.diff().dropna()

test = determine_differencing_order_for_all(data_level)


#%% Decompose series and seasonality study 

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

# Example usage
decomposed_data = decompose_data(data)


#%% Study trends and forecast COVID impact on demography

#Choose order of integration of model 
def find_best_order_single_column(column, series, min_ar,min_ma,max_ar, max_ma, cutoff_date, d):
    best_order = None
    best_aic = float('inf')

    for p in range(min_ar, max_ar + 1):
        for q in range(min_ma, max_ma + 1):
            training_data = series[series.index <= cutoff_date]
            testing_data = series[series.index > cutoff_date]

            model = ARIMA(training_data, order=(p, d, q))
            arima_model = model.fit()
            aic = arima_model.aic #use AIC as information criterion of the quality of model/prediction 

            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
    return column, best_order


def find_best_orders(data, min_ar,min_ma,max_ar, max_ma, cutoff_date, diff_orders):
    best_orders = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for column in data.columns:
            series = data[column].dropna()
            d = diff_orders.get(column, 0)  # Use the pre-determined 'd' from the 'orders' dictionary
            futures.append(executor.submit(find_best_order_single_column, column, series,min_ar,min_ma, max_ar, max_ma, cutoff_date, d))

        # Use tqdm to display a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Finding best orders"):
            column, order = future.result()
            best_orders[column] = order
    return best_orders


#Forecast pop using ARIMA and data pre-COVID only 
def ARIMA_forecast(data, best_orders, cutoff=None):
    if cutoff is None:
        cutoff_date = pd.to_datetime('2020-03-01')  #work on data pre-covid only 
    else:
        cutoff_date = pd.to_datetime(cutoff)

    arima_models = {}

    for column, order in best_orders.items():
        series = data[column]

        p, d, q = order

        # Split the data into training and testing based on the cutoff date
        training_data = series[series.index <= cutoff_date]
        testing_data = series[series.index > cutoff_date]

        # Fit an ARIMA model based on the best order using the training data
        model = ARIMA(training_data, order=order)
        arima_model = model.fit()

        arima_models[column] = arima_model

    return arima_models


decomposed_data = decomposed_data.dropna()
decomposed_data

trend_orders = determine_differencing_order_for_all(decomposed_data)

cutoff_date = pd.to_datetime('2020-03-01')
trend_orders


#=== Get ARIMA orders 

def analyze_arima_orders(data, stationnarity_orders,max_lags=120, acf_pacf_lags=120 ):
    """
    Analyze and find the best ARIMA orders for each column in a DataFrame based on ACF and PACF.

    :param data: A pandas DataFrame with time series data
    :param max_lags: The maximum number of lags to consider for finding the best ARIMA order
    :param acf_pacf_lags: Number of lags to show in ACF and PACF plots
    :return: Dictionary with the best ARIMA orders for each column
    """
    best_orders = {}

    for column in data.columns:
        time_series = data[column]

        # ACF and PACF plots
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(time_series, lags=acf_pacf_lags, ax=plt.gca())
        plt.title(f'ACF for {column}')

        plt.subplot(122)
        plot_pacf(time_series, lags=acf_pacf_lags, ax=plt.gca())
        plt.title(f'PACF for {column}')
        plt.show()

        # Find the best ARIMA order
        lag_acf = acf(time_series, nlags=max_lags)
        lag_pacf = pacf(time_series, nlags=max_lags, method='ols')

        p = next((x for x, val in enumerate(lag_pacf) if val < abs(0.6)), 0)
        q = next((x for x, val in enumerate(lag_acf) if val < abs(0.6)), 0)

        best_orders[column] = (p, stationnarity_orders.get(column,0), q)  # Assuming d=0 for simplicity

    return best_orders


best_pacf_acf_orders = analyze_arima_orders(decomposed_data,trend_orders)
print("\nBest ARIMA Orders for Each Column:")
print(best_pacf_acf_orders)


#==== Determine differencing orders

# Use pre-determined differencing orders to find best ARIMA orders
# Use best_orders to forecast with ARIMA models
arima_models = ARIMA_forecast(decomposed_data, best_pacf_acf_orders, cutoff='2020-03-01')

# Plot the predictions and observed test data for every country
for column, arima_model in arima_models.items():
    # Forecast future values
    predictions = arima_model.forecast(steps=43)

    # Plotting for each country
    plt.figure(figsize=(10, 6))
    plt.plot(decomposed_data.index, decomposed_data[column], label='Observed', marker='o')
    plt.plot(predictions.index, predictions, label='Forecast', linestyle='--', marker='o')

    plt.title(f'ARIMA Forecast for {column}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
     
best_pacf_acf_orders


#%% Hybrid ARIMA: fit trend

def hybrid_arima(data,best_order):
    test_errors = pd.DataFrame()  # DataFrame to store test errors
    training_errors = pd.DataFrame()  # DataFrame to store training errors
    predictions_df = pd.DataFrame()  # DataFrame to store predictions
    test_set = pd.DataFrame()

    log_transformed_data =   data   # Apply log transformation to the entire dataset
    cutoff_date = pd.to_datetime('2020-03-01')

    for column in data.columns:
        train = log_transformed_data[column][log_transformed_data.index <= cutoff_date]
        test = log_transformed_data[column][log_transformed_data.index > cutoff_date]

        # Find the best order
        #best_order = find_best_order_single_column(train, test, column, orders, max_ar=5, max_ma=5)

        # Fit the ARIMA model
        model = ARIMA(train, order=best_order.get(column,0))
        fitted_model = model.fit()

        # Forecast all test values
        predictions = fitted_model.forecast(steps=len(test))

        # Calculate errors
        error_list = np.expm1(test) - np.expm1(predictions)
        test_error = mean_squared_error(np.expm1(test), np.expm1(predictions))
        training_error = mean_squared_error(np.expm1(train), np.expm1(fitted_model.fittedvalues))

        # Store the test errors
        test_errors[column] = error_list

        # Store the training errors
        training_errors[column] = np.expm1(train) - np.expm1(fitted_model.fittedvalues)

        # Store the predictions
        predictions_df[column] = np.expm1(predictions)

        test_set[column] = np.expm1(test)

        print(f'Test MSE for {column}: %.3f' % test_error)
        plt.plot(test.index, np.expm1(test), label='Test Data')
        plt.plot(predictions.index, np.expm1(predictions), label='Predicted Data', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Test vs. Predicted Data for {column}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return test_errors, training_errors, predictions_df, test_set

# Example usage:
test_errors, training_errors, predictions_df,test_s = hybrid_arima(decomposed_data,best_pacf_acf_orders)

# Access the DataFrames with test errors, training errors, and predictions
print("Test Errors:")
print(test_errors)

print("\nTraining Errors:")
print(training_errors)

print("\nPredictions:")
print(predictions_df)



# Assuming you have defined training_residuals as a DataFrame with one column per country
test_pred = pd.DataFrame()

# Define the model
model = Sequential()
model.add(Dense(64, activation='tanh', input_shape=(1,)))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Iterate over columns (countries) in training_residuals
for i in training_errors.columns:
    train_resid = training_errors[i].values.reshape(-1, 1)
    test_resid = test_errors[i].values.reshape(-1, 1)

    scaler = StandardScaler()
    train_resid = scaler.fit_transform(train_resid)
    test_resid = scaler.transform(test_resid)

    # Use the same model for all countries
    model.fit(train_resid, train_resid, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Predict test values
    test_predictions = model.predict(test_resid)

    # Inverse transform the predictions
    test_predictions = scaler.inverse_transform(test_predictions)

    # Store predictions in the DataFrame
    test_pred[i] = test_predictions.flatten()


# Assuming 'test' is your observed test data

# Create a DataFrame to store combined forecasts for each country
combined_forecasts = pd.DataFrame(index=test_s.index)

# Iterate over columns (countries)
for column in predictions_df.columns:
    final_forecasts = predictions_df[column].values + test_pred[column].values

    # Plotting for each country
    plt.figure(figsize=(10, 6))
    plt.plot(test_s.index, test_s[column], label='Test Data', marker='o')
    plt.plot(test_s.index, final_forecasts, label='Combined Forecast', linestyle='--', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Combined Forecast vs. Test Data for {column}')
    plt.legend()
    plt.show()

    # Store the combined forecasts in the DataFrame
    combined_forecasts[column] = final_forecasts

# Display the combined forecasts DataFrame
print(combined_forecasts)


#%% Assess model

#%%Fit AR(1) 


def calculate_forecast_accuracy_metrics(actual, forecast):
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    bias = np.mean(forecast - actual)
    return mse, mae, rmse, mape, bias

# Dictionary to store accuracy metrics for each country
accuracy_metrics = {}

# Analyzing forecasts for each country
for column in combined_forecasts.columns:
    mse, mae, rmse, mape, bias = calculate_forecast_accuracy_metrics(test_s[column], combined_forecasts[column])
    accuracy_metrics[column] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Bias': bias}

# Displaying accuracy metrics for each country
for country, metrics in accuracy_metrics.items():
    print(f"Accuracy Metrics for {country}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    print("\n")
    

#==== resample to annual data
# Assuming 'test_s' is your observed test data
# Resample the observed test data to annual frequency
test_s_annual = test_s.resample('A').sum()

# Resample the forecasted data to annual frequency
combined_forecasts_annual = combined_forecasts.resample('A').sum()

# Plot the observed vs. forecast values for each country
for column in combined_forecasts_annual.columns:
    plt.figure(figsize=(12, 8))
    plt.plot(test_s_annual.index, test_s_annual[column], label=f'Observed - {column}', marker='o')
    plt.plot(combined_forecasts_annual.index, combined_forecasts_annual[column], label=f'Forecast - {column}', linestyle='--', marker='o')

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(f'Observed vs. Forecast (Annual Resampled) for {column}')
    plt.legend()
    plt.show()
    
    
# Assuming 'test_s' is your observed test data with actual population levels
# 'predictions_df' and 'test_pred' contain predicted and observed log(growth rates), respectively
for column in combined_forecasts.columns:
    # Convert log(growth rates) back to growth rates
    growth_rates = np.exp(combined_forecasts[column].values )

    # Calculate cumulative product to get the forecasted population levels
    forecasted_population =  df[df[column].index=='2020-04-01'][column].values * growth_rates.cumprod()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[column], label='Test Data', marker='o')
    plt.plot(combined_forecasts.index, forecasted_population, label='Forecasted Population', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Population')
    plt.title(f'Forecasted Population vs. Test Data for {column}')
    plt.legend()
    plt.show()

    # Store the forecasted population in the DataFrame

# Display the combined forecasts DataFrame
print(combined_forecasts)

     
data_d_orders = determine_differencing_order_for_all(data)
data_d_orders
arima_models = ARIMA_forecast(data, best_pacf_acf_orders, cutoff='2020-03-01')
best_ar_1_orders= {}
# Plot the predictions and observed test data for every country

data_d_orders

for i in data_d_orders:
  best_ar_1_orders[i]=(1,data_d_orders.get(i,0),0)
arima_models = ARIMA_forecast(data, best_ar_1_orders, cutoff='2020-03-01')


arima_models = ARIMA_forecast(data, best_ar_1_orders, cutoff='2020-03-01')

# Plot the predictions and observed test data for every country
for column, arima_model in arima_models.items():
    # Forecast future values
    predictions = arima_model.forecast(steps=43)

    # Plotting for each country
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[column], label='Observed', marker='o')
    plt.plot(predictions.index, predictions, label='Forecast', linestyle='--', marker='o')

    plt.title(f'ARIMA Forecast for {column}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


#%% Fit ARIMA(p,d,q) #this doesn't seem to work

data_d_orders = determine_differencing_order_for_all(data)
data_d_orders
best_pacf_acf_orders = analyze_arima_orders(data,data_d_orders)
print("\nBest ARIMA Orders for Each Column:")
print(best_pacf_acf_orders)

best_pacf_acf_orders

arima_models = ARIMA_forecast(data, best_pacf_acf_orders, cutoff='2020-03-01')

# Plot the predictions and observed test data for every country
for column, arima_model in arima_models.items():
    # Forecast future values
    predictions = arima_model.forecast(steps=43)

    # Plotting for each country
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[column], label='Observed', marker='o')
    plt.plot(predictions.index, predictions, label='Forecast', linestyle='--', marker='o')

    plt.title(f'ARIMA Forecast for {column}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()





#%% Use SARIMA -> integrate seasonal component because we're using monthly data

def find_best_seasonal_params(data, orders):
    # Define the range of values for P, D, Q, and m (seasonal parameters)
    P_values = range(0, 2)  # Seasonal autoregressive order
    D_values = range(0, 3)  # Seasonal differencing order
    Q_values = range(0, 2)  # Seasonal moving average order
    m_values = [12]         # Seasonal period

    # Create all possible combinations of seasonal SARIMA parameters
    seasonal_param_combinations = list(itertools.product(P_values, D_values, Q_values, m_values))

    # Initialize a dictionary to store the best parameters for each column
    best_params_dict = {}

    # Loop through the columns of the dataset
    for column in data.columns:
        log_transformed_column = data[column].apply(lambda x: math.log(x) if x > 0 else 0)

        # Retrieve non-seasonal orders from the 'orders' dictionary
        p, d, q = orders[column]

        # Initialize AIC with a large value for each column
        best_aic = float("inf")
        best_params = None

        # Create a tqdm progress bar for parameter search
        with tqdm(total=len(seasonal_param_combinations), desc=f"Column: {column}") as pbar:
            for seasonal_params in seasonal_param_combinations:
                try:
                    model = sm.tsa.SARIMAX(log_transformed_column,
                                           order=(p, d, q),
                                           seasonal_order=seasonal_params)
                    result = model.fit(disp=False)
                    aic = result.aic

                    # Ensure the convergence of the model
                    if not math.isinf(result.zvalues.mean()):
                        if aic < best_aic:
                            best_aic = aic
                            best_params = seasonal_params

                except:
                    continue

                # Update the progress bar
                pbar.update(1)

        # Store the best parameters for this column in the dictionary
        best_params_dict[column] = best_params

    # Print the best parameters for each column
    for column, params in best_params_dict.items():
        print(f"Column: {column}, Best Seasonal Parameters: {params}")

    return best_params_dict

# Example usage:
# Assuming you have a DataFrame 'data' and a dictionary 'orders'
best_seasonal_params = find_best_seasonal_params(data, best_pacf_acf_orders)
print("\nBest Seasonal Parameters for Each Column:")
print(best_seasonal_params)

seasonal_params={'Japan': (1, 2, 0, 12), 'France': (1, 0, 0, 12), 'Usa': (1, 0, 0, 12), 'Colombia': (0, 0, 0, 12), 'Sweden': (1, 0, 0, 12)}
def combine_sarima_params(pacf_acf_orders, seasonal_params):
    combined_params = {}
    for country in pacf_acf_orders.keys():
        # Ensure the country is present in both dictionaries
        if country in seasonal_params:
            combined_params[country] = pacf_acf_orders[country] + seasonal_params[country]
        else:
            print(f"Missing seasonal parameters for {country}")
    return combined_params

combined_sarima_params=combine_sarima_params(best_pacf_acf_orders,best_seasonal_params)
     

best_pacf_acf_orders, best_seasonal_params
def sarima_forecast(data, best_params_dict, cutoff=None):
    if cutoff is None:
        cutoff_date = pd.to_datetime('2020-03-01')
    else:
        cutoff_date = pd.to_datetime(cutoff)

    sarima_models = {}

    for column, params in best_params_dict.items():
        series = data[column]

        order = params[:3]
        seasonal_order = params[3:]

        # Apply log transformation to the series
        log_transformed_series = series.apply(lambda x: math.log(x) if x > 0 else 0)

        # Split the data into training and testing based on the cutoff date
        training_data = log_transformed_series[log_transformed_series.index <= cutoff_date]
        testing_data = log_transformed_series[log_transformed_series.index > cutoff_date]

        # Fit a SARIMA model based on the best parameters using the training data
        model = sm.tsa.SARIMAX(training_data, order=order, seasonal_order=seasonal_order)
        sarima_model = model.fit()

        sarima_models[column] = sarima_model

    return sarima_models

def plot_sarima_forecast(data, sarima_models, cutoff_date):
    # Plot the forecasts versus the observations for every country
    for column, sarima_model in sarima_models.items():
        # Forecast future values
        forecast_steps = len(data) - len(data[data.index <= cutoff_date])
        predictions = sarima_model.get_forecast(steps=forecast_steps)

        # Transform the log-scale predictions back to the original scale
        predicted_values = pd.Series(predictions.predicted_mean, index=predictions.row_labels)
        predicted_values = predicted_values.apply(lambda x: math.exp(x))

        # Plotting for each country
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data[column], label='Observed', marker='o')
        plt.plot(predicted_values.index, predicted_values, label='Forecast', linestyle='--', marker='o')

        plt.title(f'SARIMA Forecast for {column}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

# Example usage:
# Load your dataset (assuming it's in a DataFrame)
sarima_models = sarima_forecast(data, combined_sarima_params, cutoff='2020-03-01')
plot_sarima_forecast(data, sarima_models, cutoff_date=pd.to_datetime('2020-03-01'))
     

#%% Hybrid ARIMA #fails 

def find_best_order_single_column(train_data, test, column, orders, d=None, max_ar=12, max_ma=12):
    d = orders.get(column, 0) if d is None else d
    best_order = None
    best_aic = float('inf')

    for p in range(0, max_ar + 1):
        for q in range(0, max_ma + 1):
            training_data = train_data
            testing_data = test

            model = ARIMA(training_data, order=(p, d, q))
            arima_model = model.fit()
            aic = arima_model.aic

            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)

    return best_order

def hybrid_arima(data ):
    test_errors = pd.DataFrame()  # DataFrame to store test errors
    training_errors = pd.DataFrame()  # DataFrame to store training errors
    predictions_df = pd.DataFrame()  # DataFrame to store predictions
    test_set = pd.DataFrame()

    log_transformed_data = np.log1p(data)  # Apply log transformation to the entire dataset
    cutoff_date = pd.to_datetime('2020-03-01')

    for column in data.columns:
        train = log_transformed_data[column][log_transformed_data.index <= cutoff_date]
        test = log_transformed_data[column][log_transformed_data.index > cutoff_date]

        # Find the best order

        # Fit the ARIMA model
        model = ARIMA(train, order=best_pacf_acf_orders.get(column))
        fitted_model = model.fit()

        # Forecast all test values
        predictions = fitted_model.forecast(steps=len(test))

        # Calculate errors
        error_list = np.expm1(test) - np.expm1(predictions)
        test_error = mean_squared_error(np.expm1(test), np.expm1(predictions))
        training_error = mean_squared_error(np.expm1(train), np.expm1(fitted_model.fittedvalues))

        # Store the test errors
        test_errors[column] = error_list

        # Store the training errors
        training_errors[column] = np.expm1(train) - np.expm1(fitted_model.fittedvalues)

        # Store the predictions
        predictions_df[column] = np.expm1(predictions)

        test_set[column] = np.expm1(test)

        print(f'Test MSE for {column}: %.3f' % test_error)
        plt.plot(test.index, np.expm1(test), label='Test Data')
        plt.plot(predictions.index, np.expm1(predictions), label='Predicted Data', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Test vs. Predicted Data for {column}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return test_errors, training_errors, predictions_df, test_set

# Example usage:
# Assuming 'data' is a DataFrame containing multiple columns of time series data
# Replace 'data' with your actual dataset

test_errors, training_errors, predictions_df,test_s = hybrid_arima(data)

# Access the DataFrames with test errors, training errors, and predictions
print("Test Errors:")
print(test_errors)

print("\nTraining Errors:")
print(training_errors)

print("\nPredictions:")
print(predictions_df)


# Assuming you have defined training_residuals as a DataFrame with one column per country
test_pred = pd.DataFrame()

# Define the model
model = Sequential()
model.add(Dense(64, activation='tanh', input_shape=(1,)))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Iterate over columns (countries) in training_residuals
for i in training_errors.columns:
    train_resid = training_errors[i].values.reshape(-1, 1)
    test_resid = test_errors[i].values.reshape(-1, 1)

    scaler = StandardScaler()
    train_resid = scaler.fit_transform(train_resid)
    test_resid = scaler.transform(test_resid)

    # Use the same model for all countries
    model.fit(train_resid, train_resid, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Predict test values
    test_predictions = model.predict(test_resid)

    # Inverse transform the predictions
    test_predictions = scaler.inverse_transform(test_predictions)

    # Store predictions in the DataFrame
    test_pred[i] = test_predictions.flatten()


test_pred


# Assuming 'test' is your observed test data

# Create a DataFrame to store combined forecasts for each country
combined_forecasts = pd.DataFrame(index=test_s.index)

# Iterate over columns (countries)
for column in predictions_df.columns:
    final_forecasts = predictions_df[column].values + test_pred[column].values

    # Plotting for each country
    plt.figure(figsize=(10, 6))
    plt.plot(test_s.index, test_s[column], label='Test Data', marker='o')
    plt.plot(test_s.index, final_forecasts, label='Combined Forecast', linestyle='--', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Combined Forecast vs. Test Data for {column}')
    plt.legend()
    plt.show()

    # Store the combined forecasts in the DataFrame
    combined_forecasts[column] = final_forecasts

# Display the combined forecasts DataFrame
print(combined_forecasts)


# Assuming 'test_s' is your observed test data
# Resample the observed test data to annual frequency
test_s_annual = test_s.resample('A').sum()
combined_forecasts_annual = combined_forecasts.resample('A').sum()

# Plot the observed vs. forecast values for each country
plt.figure(figsize=(12, 8))
for column in combined_forecasts_annual.columns:
    plt.plot(test_s_annual.index, test_s_annual[column], label=f'Observed - {column}', marker='o')
    plt.plot(combined_forecasts_annual.index, combined_forecasts_annual[column], label=f'Forecast - {column}', linestyle='--', marker='o')

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(f'Observed vs. Forecast (Annual Resampled) {column}')
    plt.legend()
    plt.show()
    
#%%Hybrid SARIMA

def hybrid_sarima(data, best_params_dict):
    test_errors = pd.DataFrame()  # DataFrame to store test errors
    training_errors = pd.DataFrame()  # DataFrame to store training errors
    predictions_df = pd.DataFrame()  # DataFrame to store predictions
    test_set = pd.DataFrame()

    log_transformed_data = np.log1p(data)  # Apply log transformation to the entire dataset
    cutoff_date = pd.to_datetime('2020-03-01')

    for column in data.columns:
        train = log_transformed_data[column][log_transformed_data.index <= cutoff_date]
        test = log_transformed_data[column][log_transformed_data.index > cutoff_date]

        # Get the best order for the current time series
        best_order = best_params_dict[column][:3]
        best_seasonal_order = best_params_dict[column][3:]

        # Fit the SARIMA model
        model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
        fitted_model = model.fit()

        # Forecast all test values
        predictions = fitted_model.forecast(steps=len(test))
        predictions.index = test.index  # Align the index of predictions with the test data

        # Calculate errors
        test_error = mean_squared_error(np.expm1(test), np.expm1(predictions))
        training_error = mean_squared_error(np.expm1(train), np.expm1(fitted_model.fittedvalues))

        # Store errors and predictions
        test_errors[column] = np.expm1(test) - np.expm1(predictions)
        training_errors[column] = np.expm1(train) - np.expm1(fitted_model.fittedvalues)
        predictions_df[column] = np.expm1(predictions)
        test_set[column] = np.expm1(test)

        print(f'Test MSE for {column}: {test_error:.3f}')
        plt.plot(test.index, np.expm1(test), label='Test Data')
        plt.plot(predictions.index, np.expm1(predictions), label='Predicted Data', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Test vs. Predicted Data for {column}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return test_errors, training_errors, predictions_df, test_set

# Example usage:
# Assuming 'data' is a DataFrame containing multiple columns of time series data
# Assuming 'best_params_dict' contains the best SARIMA parameters for each time series in 'data'
# Replace 'data' and 'best_params_dict' with your actual dataset and parameters

test_errors, training_errors, predictions_df, test_set = hybrid_sarima(data, combined_sarima_params)


# Assuming you have training_errors and test_errors DataFrames as before
test_pred = pd.DataFrame()

# Iterate over columns (countries) in training_errors
for i in training_errors.columns:
    train_resid = training_errors[i].values.reshape(-1, 1)
    test_resid = test_errors[i].values.reshape(-1, 1)

    scaler = StandardScaler()
    train_resid = scaler.fit_transform(train_resid)
    test_resid = scaler.transform(test_resid)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    train_resid = train_resid.reshape((train_resid.shape[0], 1, train_resid.shape[1]))
    test_resid = test_resid.reshape((test_resid.shape[0], 1, test_resid.shape[1]))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=(1, 1)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(train_resid, train_resid, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Predict test values
    test_predictions = model.predict(test_resid)

    # Inverse transform the predictions
    test_predictions = scaler.inverse_transform(test_predictions)

    # Store predictions in the DataFrame
    test_pred[i] = test_predictions.flatten()

# Output DataFrame with predictions
print(test_pred)


# Assuming 'test' is your observed test data

# Create a DataFrame to store combined forecasts for each country
combined_forecasts = pd.DataFrame(index=test_s.index)

# Iterate over columns (countries)
for column in predictions_df.columns:
    final_forecasts = predictions_df[column].values + test_pred[column].values

    # Plotting for each country
    plt.figure(figsize=(10, 6))
    plt.plot(test_s.index, test_s[column], label='Test Data', marker='o')
    plt.plot(test_s.index, final_forecasts[0:32], label='Combined Forecast', linestyle='--', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Combined Forecast vs. Test Data for {column}')
    plt.legend()
    plt.show()

    # Store the combined forecasts in the DataFrame
    combined_forecasts[column] = final_forecasts[0:32]

# Display the combined forecasts DataFrame
print(combined_forecasts)


def calculate_forecast_accuracy_metrics(actual, forecast):
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    bias = np.mean(forecast - actual)
    return mse, mae, rmse, mape, bias

# Dictionary to store accuracy metrics for each country
accuracy_metrics = {}

# Analyzing forecasts for each country
for column in combined_forecasts.columns:
    mse, mae, rmse, mape, bias = calculate_forecast_accuracy_metrics(test_s[column], combined_forecasts[column])
    accuracy_metrics[column] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Bias': bias}

# Displaying accuracy metrics for each country
for country, metrics in accuracy_metrics.items():
    print(f"Accuracy Metrics for {country}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    print("\n")

