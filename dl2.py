# Binary Classification using Deep Neural Network (IMDB Dataset)

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# 1. Load Dataset
# ---------------------------
vocab_size = 10000  # use top 10k words

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# ---------------------------
# 2. Pad Sequences (make equal length)
# ---------------------------
maxlen = 200

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# ---------------------------
# 3. Build Deep Neural Network
# ---------------------------
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(maxlen,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # binary output

# ---------------------------
# 4. Compile Model
# ---------------------------
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# 5. Train Model
# ---------------------------
model.fit(X_train, y_train,
          epochs=5,
          batch_size=32,
          validation_split=0.2,
          verbose=1)

# ---------------------------
# 6. Evaluate Model
# ---------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# ---------------------------
# 7. Predictions
# ---------------------------
predictions = model.predict(X_test[:5])

print("\nSample Predictions:")
for i in range(5):
    print(f"Review {i+1}: {'Positive' if predictions[i][0] > 0.5 else 'Negative'}")