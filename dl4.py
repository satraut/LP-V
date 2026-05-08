# RNN (LSTM) for Google Stock Price Prediction

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load Dataset
# ---------------------------
data = pd.read_csv("GOOG.csv")   # Make sure file exists
prices = data['Close'].values.reshape(-1, 1)

# ---------------------------
# 2. Normalize Data
# ---------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(prices)

# ---------------------------
# 3. Create Time-Series Data
# ---------------------------
X = []
y = []

time_step = 60   # use last 60 days

for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# reshape for RNN [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# ---------------------------
# 4. Split Train/Test
# ---------------------------
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------
# 5. Build RNN Model
# ---------------------------
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

# ---------------------------
# 6. Compile Model
# ---------------------------
model.compile(optimizer='adam', loss='mean_squared_error')

# ---------------------------
# 7. Train Model
# ---------------------------
model.fit(X_train, y_train, epochs=5, batch_size=32)

# ---------------------------
# 8. Predict
# ---------------------------
predictions = model.predict(X_test)

# convert back to original scale
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# ---------------------------
# 9. Plot Results
# ---------------------------
plt.plot(y_test_actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()