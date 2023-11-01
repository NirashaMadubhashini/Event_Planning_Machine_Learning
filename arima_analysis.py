import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product

# Load the sales_train.csv dataset
sales_data = pd.read_csv('data/sales_train.csv')

# Convert the 'date' column to datetime format and aggregate to monthly level
sales_data['date'] = pd.to_datetime(sales_data['date'], dayfirst=True)
monthly_sales = sales_data.groupby([sales_data['date'].dt.to_period("M").astype('datetime64[ns]')])['event_cnt_day'].sum().reset_index()
monthly_sales.columns = ['ds', 'y']

# Check stationarity using Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return result[1] <= 0.05

# Make series stationary if not
if not adf_test(monthly_sales['y']):
    monthly_sales['y'] = monthly_sales['y'] - monthly_sales['y'].shift(1)
    monthly_sales.dropna(inplace=True)

# Determine best ARIMA parameters
p = d = q = range(0, 2)
pdq = list(product(p, d, q))
best_aic = np.inf
best_order = None

for order in pdq:
    try:
        model = ARIMA(monthly_sales['y'], order=order)
        arima_result = model.fit()
        if arima_result.aic < best_aic:
            best_aic = arima_result.aic
            best_order = order
    except Exception as e:
        print(f"ARIMA{order} - AIC:{np.inf} -> Error: {e}")

# Fit ARIMA model with best parameters
model = ARIMA(monthly_sales['y'], order=best_order)
arima_result = model.fit()

# Forecast future sales for the next 12 months
forecast, stderr, conf_int = arima_result.forecast(steps=12)

# Plot the historical and forecasted sales
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['ds'], monthly_sales['y'], label='Historical Sales')
plt.plot(pd.date_range(monthly_sales['ds'].iloc[-1], periods=13, closed='right'), np.append(monthly_sales['y'].iloc[-1], forecast), color='red', label='Forecasted Sales')
plt.fill_between(pd.date_range(monthly_sales['ds'].iloc[-1], periods=13, closed='right'),
                 np.append(monthly_sales['y'].iloc[-1], conf_int[:, 0]),
                 np.append(monthly_sales['y'].iloc[-1], conf_int[:, 1]), color='pink', alpha=0.3)
plt.title('Monthly Sales Forecasting with ARIMA')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
