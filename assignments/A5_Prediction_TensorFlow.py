# ============================================================
# Neural_Network_Regression_Car_Price
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. CREATE TRAINING DATA
# X = Number of Years
# Y = Car Price in Million THB
# ============================================================
x_train = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
display('x data are:', x_train)
y_train = np.array([
    1.445,
    1.30,
    1.14,
    0.90,
    0.85,
    0.72,
    0.66
]).reshape(-1, 1)
display('y data are:', y_train)

# ============================================================
# 3. VISUALIZE CAR PRICE TRAINING DATA
# ============================================================
plt.scatter(
    x_train,
    y_train,
    marker='x',
    s=80,
    c='red'
)

plt.title('Toyota Camry 2.0 year 2020 Prices')
plt.xlabel('Number of years (from 2020)')
plt.ylabel('Price in M THB')
plt.grid(True)
plt.show()

# ============================================================
# 4. IMPORT TENSORFLOW / KERAS
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# ============================================================
# 5. CREATE NEURAL NETWORK REGRESSION MODEL
# 1 Input -> 1 Linear -> 4 Linear -> 1 Linear Output
# ============================================================
model = Sequential(
    [
        Dense(1, activation='linear'),
        Dense(4, activation='linear'),
        Dense(1, activation='linear')
    ]
)

# ============================================================
# 6. COMPILE REGRESSION MODEL
# Mean Squared Error + Adam Optimizer
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mse'
)

# ============================================================
# 7. TRAIN THE MODEL
# ============================================================
model.fit(
    x_train,
    y_train,
    epochs=50
)

# ============================================================
# 8. TEST MODEL WITH YEAR 8
# ============================================================
x_test = np.array([8]).reshape(-1, 1)

y_predicted = model.predict(x_test)

print(f"Predicted value: {y_predicted[0][0]:.2f} M THB")