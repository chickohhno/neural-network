# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. CREATE MULTI-CLASS TRAINING DATA
# 2 Input Features -> 4 Output Classes
# ============================================================
x_train = np.array([[0.5, 0.5], [1,0.5], [0.5, 1], [1.5, 2],
                    [0.5, 3.5], [1, 4],[1.5, 3.5], [0.5, 4],
                    [3, 0.5], [3.5, 0.5],[3.5, 1], [3, 1.5],
                    [3, 4.5], [3.5, 4],[3.5, 4.5], [3, 4] ])
display('x data are:', x_train)
y_train = np.array([1, 1, 1, 1,
                    2, 2, 2, 2,
                    3, 3, 3, 3,
                    4, 4, 4, 4]).reshape(-1,1)
display('y data are:', y_train)

# ============================================================
# 3. CREATE BOOLEAN MASKS FOR THE FOUR CLASSES
# ============================================================
one = y_train == 1
two = y_train == 2
three = y_train == 3
four = y_train == 4

# Flatten the boolean masks for correct indexing
# .flatten(), the one and zero arrays were converted into a 1-dimensional array

# ============================================================
# 4. VISUALIZE THE FOUR CLASSES
# ============================================================
plt.scatter(
    x_train[one.flatten(), 0],
    x_train[one.flatten(), 1],
    marker='x',
    s=80,
    c='red',
    label="y=1",
    lw=3
)

plt.scatter(
    x_train[two.flatten(), 0],
    x_train[two.flatten(), 1],
    marker='o',
    s=80,
    c='blue',
    label="y=2",
    lw=3
)

plt.scatter(
    x_train[three.flatten(), 0],
    x_train[three.flatten(), 1],
    marker='v',
    s=80,
    c='green',
    label="y=3",
    lw=3
)

plt.scatter(
    x_train[four.flatten(), 0],
    x_train[four.flatten(), 1],
    marker='+',
    s=80,
    c='orange',
    label="y=4",
    lw=3
)

plt.title('Multi Class Classification Example')
plt.grid(True)
plt.axis([0, 5, 0, 5])
plt.ylabel('$x_1$')
plt.xlabel('$x_0$')
plt.legend()
plt.show()

# ============================================================
# 5. IMPORT TENSORFLOW / KERAS
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# ============================================================
# 6. CREATE MULTI-CLASS NEURAL NETWORK
# 2 Inputs -> 16 Neurons -> 32 Neurons -> 4 Output Classes
# ============================================================
model = Sequential(
    [
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')
    ]
)

# ============================================================
# 7. COMPILE THE MODEL
# Sparse Categorical Crossentropy + Adam Optimizer
# ============================================================
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)

# ============================================================
# 8. ADJUST CLASS LABELS FROM 1-4 TO 0-3
# TensorFlow SparseCategoricalCrossentropy expects class indices
# ============================================================
y_train_adjusted = y_train - 1

# ============================================================
# 9. TRAIN THE NEURAL NETWORK
# ============================================================
model.fit(
    x_train,
    y_train_adjusted,
    epochs=1000
)

# ============================================================
# 10. TEST CASE 1 - PREDICT CLASS PROBABILITIES
# ============================================================
x_test = np.array([[4, 4]])

y_test = model.predict(x_test)

print('y0 is ', y_test[0, 0])
print('y1 is ', y_test[0, 1])
print('y2 is ', y_test[0, 2])
print('y3 is ', y_test[0, 3])

predicted_class_index = np.argmax(y_test)

predicted_class = predicted_class_index + 1  # Adjust for 1-indexed classes

print(f'Predicted class: y = {predicted_class}')

# ============================================================
# 11. TEST CASE 2 - PREDICT CLASS PROBABILITIES
# ============================================================
x_test = np.array([[1, 1.5]])

y_test = model.predict(x_test)

print('y0 is ', y_test[0, 0])
print('y1 is ', y_test[0, 1])
print('y2 is ', y_test[0, 2])
print('y3 is ', y_test[0, 3])

predicted_class_index = np.argmax(y_test)

predicted_class = predicted_class_index + 1  # Adjust for 1-indexed classes

print(f'Predicted class: y = {predicted_class}')