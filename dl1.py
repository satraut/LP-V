# Boston Housing Price Prediction using Deep Neural Network

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# 1. Load Dataset
# ---------------------------
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# ---------------------------
# 2. Feature Scaling
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# 3. Build Deep Neural Network
# ---------------------------
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer (regression)

# ---------------------------
# 4. Compile Model
# ---------------------------
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ---------------------------
# 5. Train Model
# ---------------------------
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# ---------------------------
# 6. Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# ---------------------------
# 7. Predict Example
# ---------------------------
print("\nSample Predictions:")
for i in range(5):
    print(f"Actual: {y_test[i]:.2f}  Predicted: {y_pred[i][0]:.2f}")
print("Training complete!")

# If you have predictions
print("Sample prediction:", y_pred[:5])

# If you have accuracy
from sklearn.metrics import r2_score
print("R2 Score:", r2_score(y_test, y_pred))