# CNN for Fashion MNIST Classification

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# ---------------------------
# 1. Load Dataset
# ---------------------------
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# ---------------------------
# 2. Preprocessing
# ---------------------------
# Normalize pixel values (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ---------------------------
# 3. Build CNN Model
# ---------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 classes

# ---------------------------
# 4. Compile Model
# ---------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# 5. Train Model
# ---------------------------
model.fit(X_train, y_train,
          epochs=5,
          batch_size=32,
          validation_split=0.1)

# ---------------------------
# 6. Evaluate Model
# ---------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# ---------------------------
# 7. Predict Sample
# ---------------------------
predictions = model.predict(X_test[:5])

print("\nSample Predictions (class index):")
for i in range(5):
    print(f"Predicted: {predictions[i].argmax()}, Actual: {y_test[i]}")