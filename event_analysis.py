import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the data
train_data = pd.read_csv('./data/sales_train.csv')
test_data = pd.read_csv('./data/test.csv')

# Pivot the training data
train_Data = train_data.pivot_table(index=['shop_id', 'event_id'],
                                    values=['event_cnt_day'],
                                    columns=['date_block_num'],
                                    fill_value=0,
                                    aggfunc='sum')

# Flatten the multi-level columns and convert to string
train_Data.columns = ['_'.join(map(str, col)).strip() for col in train_Data.columns.values]

# Reset index
train_Data.reset_index(inplace=True)

# Merge test and training data
Combine_train_test = pd.merge(test_data, train_Data, how='left', on=['shop_id', 'event_id']).fillna(0)

# Sort and drop columns
Combine_train_test = Combine_train_test.sort_values(by='ID')
Combine_train_test = Combine_train_test.drop(columns=['ID'])

# Prepare data for LSTM
X_train = np.array(Combine_train_test.values[:, :-1]).reshape(Combine_train_test.values[:, :-1].shape[0],
                                                              Combine_train_test.values[:, :-1].shape[1], 1)
y_train = Combine_train_test.values[:, -1:]

# LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history_1 = model.fit(X_train, y_train, epochs=15, batch_size=128)

# Plotting the loss
df_his1 = pd.DataFrame(history_1.history)
plt.figure(figsize=(10, 10))
plt.plot(df_his1.index + 1, df_his1['loss'], color='r', label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# MySQL connection parameters
HOST = "localhost"
USER = "root"
PASSWORD = "1234"
DATABASE = "event_ml"


def write_df_to_sql(df, table_name):
    connection = pymysql.connect(host=HOST,
                                 user=USER,
                                 password=PASSWORD,
                                 database=DATABASE)

    cursor = connection.cursor()
    columns = ', '.join(df.columns)
    placeholders = ', '.join(['%s'] * len(df.columns))
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    try:
        cursor.executemany(query, [tuple(row) for row in df.values])
        connection.commit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()


# Writing dataframes to MySQL
write_df_to_sql(train_data, 'sales_train_table')
write_df_to_sql(test_data, 'test_table')
