import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Data Loading
sales_train = pd.read_csv('data/sales_train.csv')
event_service = pd.read_csv('data/event_service.csv')
event_data = pd.read_csv('data/event.csv')

# Data Merging
merged_sales_train = sales_train.merge(event_data, on='event_id', how='left')
merged_sales_train = merged_sales_train.merge(event_service, left_on='event_service_id', right_on='event_services_id', how='left')
merged_sales_train.drop(columns=['event_service_id'], inplace=True)

# Convert date to datetime
merged_sales_train['date'] = pd.to_datetime(merged_sales_train['date'], dayfirst=True)

# Monthly sales data
monthly_sales = merged_sales_train.groupby([merged_sales_train['date'].dt.year.rename("Year"),
                                           merged_sales_train['date'].dt.month.rename("Month")]).agg({'event_cnt_day': 'sum'}).reset_index()

monthly_sales.columns = ['Year', 'Month', 'Total Sales']


# Visualizing monthly sales
plt.figure(figsize=(15,7))
plt.plot(monthly_sales['Total Sales'], '-o', color='blue')
plt.title('Monthly Sales')
plt.xlabel('Time (in months)')
plt.ylabel('Total Sales')
plt.xticks(ticks=range(len(monthly_sales)), labels=[f"{row['Month']}-{row['Year']}" for _, row in monthly_sales.iterrows()], rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Feature Engineering
merged_sales_train['month'] = merged_sales_train['date'].dt.month
merged_sales_train['year'] = merged_sales_train['date'].dt.year
merged_sales_train['event_name_encoded'] = merged_sales_train['event_name'].astype('category').cat.codes
merged_sales_train['event_service_encoded'] = merged_sales_train['event_services_name'].astype('category').cat.codes

# Drop columns that are not needed for modeling
model_data = merged_sales_train.drop(columns=['date', 'event_name', 'event_services_name'])

# Splitting Data for model training
X = model_data.drop(columns=['event_cnt_day'])
y = model_data['event_cnt_day']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
gbr.fit(X_train, y_train)

# Predict future sales for events and services
next_month = merged_sales_train['month'].max() + 1
next_year = merged_sales_train['year'].max() if next_month != 1 else merged_sales_train['year'].max() + 1

# Placeholder for future event and service predictions
future_event_sales = []
for event, event_name in zip(merged_sales_train['event_name_encoded'].unique(), merged_sales_train['event_name'].unique()):
    for service, service_name in zip(merged_sales_train['event_service_encoded'].unique(), merged_sales_train['event_services_name'].unique()):
        future_data = X_train.iloc[0:1].copy()  # Use the first row of X_train as a template
        future_data['month'] = next_month
        future_data['year'] = next_year
        future_data['event_name_encoded'] = event
        future_data['event_service_encoded'] = service
        predicted_sales = gbr.predict(future_data)
        future_event_sales.append((event, event_name, service, service_name, predicted_sales[0]))

future_sales_df = pd.DataFrame(future_event_sales, columns=['Event_Code', 'Event', 'Service_Code', 'Service', 'Predicted Sales'])



# Displaying Predicted Sales for future events and services
print(future_sales_df)


# Visualizing training loss over epochs
train_score = np.zeros((gbr.n_estimators,), dtype=np.float64)
for i, y_pred_train in enumerate(gbr.staged_predict(X_train)):
    train_score[i] = mean_squared_error(y_train, y_pred_train)

plt.figure(figsize=(15, 7))
plt.plot(np.arange(gbr.n_estimators) + 1, train_score, 'b-', label='Training Set Error')
plt.title('Training Loss over Epochs')
plt.xlabel('Boosting Iterations')
plt.ylabel('Mean Squared Error')
plt.legend(loc='upper right')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()