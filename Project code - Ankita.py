# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:40:11 2023

@author: ankit
"""

from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon 
from sktime.utils.plotting import plot_series   
from sktime.utils.plotting import plot_correlations
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
  
os.chdir(r'C:\Users\ankit\Documents\Ankita\UD\Sem 3\Forecasting Methods\project 2')  

os.getcwd()

# Setting up charting formats:
    
# optional plot parameters
    
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['figure.figsize'] = [14.0, 5.0]
plt.rcParams['font.size']= 18  
    
plt.style.available   # Check what are the styles available for Chart formats

plt.style.use('fivethirtyeight')       # Assign FiveThirtyEight 

#Load the CSV file in Python as a DataFrame

df=pd.read_csv('PersonalSavings.csv',index_col=0,parse_dates=True)
df

plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=5))
df.plot(label='Personal Savings', xlabel='Year', ylabel='Millions of Dollars',title='Line Plot')

#########################################################################################################
#                    Obtaining Prediction Intervals of ETS Models for presonal Saving
#########################################################################################################
# ETS(A,A,N): Holt’s linear method with multiplicative errors

model1 = ETSModel(df['PersonalSavings'], error='additive', trend='additive', initialization_method="estimated")
fit1   = model1.fit()
fcast1 = fit1.forecast(6)

fit1.summary()

fcast1.plot(label='Fcast: Holts Linear method with mult Error')
plt.plot(fcast1)
plt.xlabel('year')
plt.ylabel('Forecasted Saving')
plt.title('Forecat after the last availabe data for Personal Saving')
plt.show()
df['PersonalSavings'].tail(1)    # figure out the last time period of your data

pred = fit1.get_prediction(start='04-01-2023', end='04-01-2026')

pred_intervals = pred.summary_frame(alpha=0.05)
pred_intervals 

# now plot with intervals 


plt.figure(figsize=(16, 8))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=5)) # Adjust the x-axis to display data for 5 years
plt.plot(df.index, df['PersonalSavings'], label='Actual')
plt.plot(pred_intervals['mean'], label='Predicted')
plt.fill_between(pred_intervals.index, pred_intervals['pi_lower'], pred_intervals['pi_upper'], alpha=0.3)
plt.title('ETS Forecast for Personal Savings (6 Quarters)')
plt.xlabel('Year')  
plt.ylabel('Millions of Dollars')
plt.legend()
plt.show()



###############################################################################          
#           Holt-Winter's method forecasts for the next 6 quarters
 ##############################################################################

model_holt_winter= ExponentialSmoothing(df, trend='additive', seasonal='additive', initialization_method="estimated")
fit_holt_winter  = model_holt_winter.fit()
fcast_holt_winter = fit_holt_winter.forecast(6)
fit_holt_winter.summary()

# Plot the original data and the forecast
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=5))
plt.plot(df['PersonalSavings'], label='Original', color='blue')
plt.plot(fit_holt_winter.fittedvalues, label='Fitted',color='orange')
plt.plot(fcast_holt_winter, label='Forecast', color='red')
plt.xlabel('Year')  
plt.ylabel('Millions of Dollars')
plt.legend()
plt.title("Holt-Winter's method forecasts for Personal Savings (6 Quarters)")
plt.show()

############################################################################### 
#               VAR method all 3 variable 
############################################################################### 

import statsmodels.api as sm
from statsmodels.tsa.api import VAR

os.chdir(r'C:\Users\ankit\Documents\Ankita\UD\Sem 3\Forecasting Methods\project 2')  

os.getcwd()

# Setting up charting formats:
       
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['figure.figsize'] = [14.0, 5.0]
plt.rcParams['font.size']= 18  
    
plt.style.available   # Check what are the styles available for Chart formats

plt.style.use('fivethirtyeight')       # Assign FiveThirtyEight 

#Load the CSV file in Python as a DataFrame
df2=pd.read_csv('income.csv',index_col=0,parse_dates=True)
df2

df2['PersonalSaving'].plot()
df2['PersonalConsumptionExpenditure'].plot()
df2['GDI'].plot()

# make a VAR model

model2 = VAR(df2)

fit_var= model2.fit(2)
fit_var.summary()

# instead of explicitly passing a lag order (i.e., 2 above), when calling the fit function, you can pass a 
# maximum number of lags and the order criterion (e.g., 'aic' to use for order selection

fit_var = model2.fit(maxlags=15, ic='aic')


best_lag = fit_var.k_ar   
best_lag                   # see what lag order was automatically selected to be optimal i.e., 14

fit_var.summary()
# obtaining just the mid forecast:
    
fcast_var=fit_var.forecast(df2.values[-best_lag:], 12)
fcast_var

# now plotting the results and forecasts

# plotting the fits

fit_var.plot()

# quickly plotting the forecast
fit_var.plot_forecast(12)

# Detailed plots with confidence intervals require obtaining the mid, lower and upper forecasts for the variables 
        # Use the code:  mid, lower, upper = results.forecast_interval(results.endog[-3:],12, alpha=0.05)

mid, lower, upper = fit_var.forecast_interval(df2.values[-best_lag:],12, alpha=0.05)

mid
lower
upper 

# Now create data frames with time index to house the mid, lower, upper values

df2.tail(6)

from datetime import datetime, timedelta

start_date = datetime(year=2023,month=3,day=1)

horizon= pd.date_range(start=start_date, periods=12, freq='Q')

fcast_mid=pd.DataFrame(mid, index = horizon ,columns = ['PersonalSaving','PersonalConsumptionExpenditure','GDI'])
fcast_mid

fcast_lower=pd.DataFrame(lower, index = horizon ,columns = ['PersonalSaving','PersonalConsumptionExpenditure','GDI'])
fcast_lower

fcast_upper=pd.DataFrame(upper, index = horizon,columns = ['PersonalSaving','PersonalConsumptionExpenditure','GDI'])
fcast_upper

# Now do forecast plot with all the prediction intervals one at a time

plt.figure(figsize=(16,8))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.plot(df2.index, df2['PersonalSaving'], label='Training')
plt.plot(horizon, fcast_mid['PersonalSaving'], label='Predicted')
plt.fill_between(horizon, fcast_lower.iloc[:,1], fcast_upper.iloc[:,1], alpha=0.3)
plt.title('Personal Saving forecast')
plt.legend()
plt.show()

plt.figure(figsize=(16,8))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.plot(df2.index, df2['PersonalConsumptionExpenditure'], label='Training')
plt.plot(horizon, fcast_mid['PersonalConsumptionExpenditure'], label='Predicted')
plt.fill_between(horizon, fcast_lower.iloc[:,1], fcast_upper.iloc[:,1], alpha=0.3)
plt.title('Personal Consumption Expenditure forecast')
plt.legend()
plt.show()

plt.figure(figsize=(16,8))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.plot(df2.index, df2['GDI'], label='Training')
plt.plot(horizon, fcast_mid['GDI'], label='Predicted')
plt.fill_between(horizon, fcast_lower.iloc[:,1], fcast_upper.iloc[:,1], alpha=0.3)
plt.title('Gross Domestic Income forecast')
plt.legend()
plt.show()






