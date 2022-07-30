# -*- coding: utf-8 -*-
"""Gold Prediction Code | Group-01 .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OQCtToToogap8dD9V2pKZzi1YeufEolk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller,acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pylab import rcParams
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Gold_data.csv",parse_dates=['date'], index_col='date')
data

data.shape

data.ndim

type(data)

list(data)

data.head()

data.info()

data.isnull().sum()

data.describe()

plt.figure(figsize=(15,10))
plt.plot(data)
plt.xlabel('Years')
plt.ylabel('Gold Price')
plt.title('Trend of the Time Series')

"""# Yearwise"""

data_temp = data.copy()
data_temp['Year'] = pd.DatetimeIndex(data_temp.index).year
data_temp['Month'] = pd.DatetimeIndex(data_temp.index).month
data_temp['Weeks'] = pd.DatetimeIndex(data_temp.index).week

# Stacked line plot
plt.figure(figsize=(15,10))
plt.title('Seasonality of the Time Series')
sns.boxplot(x='Year',y='price',hue='Year',data=data_temp)

"""We are having outliers in the year 2016 and 2021.

# Monthwise
"""

# To plot the seasonality we are going to create a temp dataframe and add columns for Month and Year values
data_temp = data.copy()
data_temp['Year'] = pd.DatetimeIndex(data_temp.index).year
data_temp['Month'] = pd.DatetimeIndex(data_temp.index).month
data_temp['Weeks'] = pd.DatetimeIndex(data_temp.index).week
# Stacked line plot
plt.figure(figsize=(10,10))
plt.title('Seasonality of the Time Series')
sns.pointplot(x='Month',y='price',hue='Year',data=data_temp)

"""Seasonality: pirce is maximum in the month of August and minimum in the month of march generally.
First four years price ranges between 2500-3000 whereas in last two years there is a sudden hike in the price and it ranges from 4000-4500

"""

# To plot the seasonality we are going to create a temp dataframe and add columns for Month and Year values
data_temp = data.copy()
data_temp['Year'] = pd.DatetimeIndex(data_temp.index).year
data_temp['Month'] = pd.DatetimeIndex(data_temp.index).month
#data_temp['Weeks'] = pd.DatetimeIndex(data_temp.index).week
#data_temp['Day'] = pd.DatetimeIndex(data_temp.index).day
# Stacked line plot
plt.figure(figsize=(20,10))
plt.title('Seasonality of the Time Series')
sns.barplot(x='Month',y='price',hue='Year',data=data_temp)

"""# Additive Decomposition

"""

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data.price.iloc[1900:2182], model='additive') 
fig = decomposition.plot()
plt.rcParams['figure.figsize'] = (30, 15)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data.price.iloc[:31], model='additive') 
fig = decomposition.plot()
plt.rcParams['figure.figsize'] = (20, 8)

"""# ADFuller Test for stationarity"""

adf = adfuller(data["price"])[1]
print(f"p value:{adf.round(4)}", ", Series is Stationary" if adf <0.05 else ", Series is Non-Stationary")

"""# Differencing"""

de_trended = data['price'].diff(1).dropna()
adf2 = adfuller(de_trended)[1]
print(f"p value:{adf2}", ", Series is Stationary" if adf2 <0.05 else ", Series is Non-Stationary")
de_trended.plot()

"""ACF Plots"""

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data)
plt.figure(figsize = (40,10))
plt.show()

"""ACF plots tells us how many moving average (MA) to be taken to remove autocorrelation in a stationarised series. So here we can clearly see that only one spike is out of this blue region (significance region), so we take MA/q=1

# ACF Plot after Differencing

# PACF Plots
"""

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data)
plt.figure(figsize = (40,10))
plt.show()

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(de_trended)
plt.figure(figsize = (40,10))
plt.show()

"""# ARIMA Model"""

from statsmodels.tsa.arima_model import ARIMA

# 1,1,1 ARIMA Model
model = ARIMA(data.price, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()

"""# Grid Search




"""

# grid search ARIMA parameters for time series
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# evaluate parameters
p_values = [0, 1, 2, 4,]
d_values = range(0, 3)
q_values = range(0, 3)
evaluate_models(data.price, p_values, d_values, q_values)

from statsmodels.tsa.arima_model import ARIMA

# 4,1,1 ARIMA Model
model = ARIMA(data.price, order=(4,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.figure(figsize=(12,5), dpi=100)
plt.show()

from statsmodels.tsa.stattools import acf

# Create Training and Test
train = data.price[:1745]
test = data.price[1745:]

pred = best_model.forecast(len(test))
preda = pd.Series(pred, index=test.index)
resi = test - preda

# Build Model
model = ARIMA(train, order=(4, 1, 1))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(437, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

"""# SARIMA"""

# Import
data = pd.read_csv('Gold_data.csv', parse_dates=['date'], index_col='date')

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasinal Dei
axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Price', fontsize=16)
plt.show()

!pip3 install pyramid-arima
import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(data, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()

from statsmodels.tsa.stattools import acf

# Create Training and Test
train = data.price[:1745]
test = data.price[1745:]

from statsmodels.tsa.statespace.sarimax import SARIMAX

best_model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(2, 1, 0, 12)).fit(dis=-1)
print(best_model.summary())

pred = best_model.forecast(len(test))
preda = pd.Series(pred, index=test.index)
resi = test - preda

print('RMSE:',np.sqrt(np.mean(resi**2)))

plt.figure(figsize=(18,10))

plt.plot(data)
plt.plot(preda)
plt.legend(('data','predictions'),fontsize=15)

"""# SMA - Simple Moving Averages"""

data['6-month-SMA'] = data['price'].rolling(window=6).mean()
data['12-month-SMA'] = data['price'].rolling(window=12).mean()
data.plot(title='Simple Moving Averages',figsize=(15,10));

"""# EWMA - Exponentially Weighted Moving Average"""

data['ewma12'] = data['price'].ewm(span=12,adjust=False).mean()
data[['price','ewma12']].plot(figsize=(15,10));

"""# Comparing SMA to EWMA"""

data[['price','ewma12','6-month-SMA','12-month-SMA']].plot(figsize=(15,10));

"""# Holt-Winters"""

# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

m = 52
alpha = 1/(2*m)

"""# Single Smoothening"""

data['HWES1'] = SimpleExpSmoothing(data['price']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
data[['price','HWES1']].plot(title='Holt Winters Single Exponential Smoothing',figsize=(35,12));

"""# Double Smoothening"""

import warnings
warnings.filterwarnings("ignore")
data['HWES2_ADD'] = ExponentialSmoothing(data['price'],trend='add').fit().fittedvalues
data[['price','HWES2_ADD']].plot(title='Holt Winters Double Exponential Smoothing: Additive Trend', figsize=(35,12));

"""# Triple Smoothening"""

import warnings
warnings.filterwarnings("ignore")
data['HWES3_ADD'] = ExponentialSmoothing(data['price'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
data[['price','HWES3_ADD']].plot(title='Holt Winters Triple Exponential Smoothing: Additive Seasonality',figsize=(35,12));

"""# Train & Test"""

train_data = data[:1745]
test_data = data[1745:]

fitted_model = ExponentialSmoothing(train_data['price'],trend='add',seasonal='add',seasonal_periods=31).fit()
test_predictions = fitted_model.forecast(437)
train_data['price'].plot(legend=True,label='TRAIN')
test_data['price'].plot(legend=True,label='TEST',figsize=(30,12))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters')

"""# Prediction Model"""

test_data['price'].plot(legend=True,label='TEST',figsize=(10,8))
test_predictions.plot(legend=True,label='PREDICTION',xlim=['2021-11-22','2021-12-22']);

"""# Checking Mean Square Error"""

from sklearn.metrics import mean_absolute_error,mean_squared_error
from math import sqrt

print(f'Mean Absolute Error = {mean_absolute_error(test_data,test_predictions)}')
print(f'Mean Squared Error = {mean_squared_error(test_data,test_predictions)}')
print(f'Root Mean Squared Error = {sqrt(mean_squared_error(test_data,test_predictions))}')

"""# Weekly Analysis"""

data  = pd.read_csv('/Gold_data.csv',index_col='date',parse_dates=True)
data.index

data.index = pd.to_datetime(data.index)
data = data.resample('1W').mean()
data.index
data.head()

len(data)

data[:15]
data['price'].plot(legend=True,label='data weekly',figsize=(12,8))

train_data = data[:200] 
test_data = data[199:] 

len(test_data)

fitted_model = ExponentialSmoothing(train_data['price'],trend='add',seasonal='add',seasonal_periods=52).fit()

test_predictions = fitted_model.forecast(125).rename('HW Test Forecast')

test_predictions[:15]

train_data['price'].plot(legend=True,label='TRAIN')
test_data['price'].plot(legend=True,label='TEST',figsize=(12,8))
plt.title('Train and Test Data');

train_data['price'].plot(legend=True,label='TRAIN')
test_data['price'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters');