# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and prepare dataset
data = {'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Usage': np.random.randint(200, 500, 100)}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# ---- ARIMA MODEL ----
model = ARIMA(df['Usage'], order=(2,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
print("ARIMA Forecast:\n", forecast)

# ---- LSTM MODEL ----
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[['Usage']])

X, y = [], []
for i in range(5, len(scaled_data)):
    X.append(scaled_data[i-5:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=1, verbose=0)

predicted = model.predict(X)
predicted_usage = scaler.inverse_transform(predicted)

# ---- Visualization ----
plt.figure(figsize=(10,5))
plt.plot(df.index[5:], df['Usage'][5:], label='Actual Usage')
plt.plot(df.index[5:], predicted_usage, label='Predicted Usage (LSTM)')
plt.xlabel('Date')
plt.ylabel('Electricity Usage (kWh)')
plt.title('Electricity Usage Prediction')
plt.legend()
plt.show()
