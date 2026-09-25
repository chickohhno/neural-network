# ============================================================
# Neural_Network_5_Class_Classification_Logits
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. CREATE 5-CLASS TRAINING DATA
# ============================================================
x_train = np.array([[0.5, 0.5], [1,0.5], [0.5, 1], [1.5, 2],
                    [0.5, 3.5], [1, 4],[1.5, 3.5], [0.5, 4],
                    [3, 0.5], [3.5, 0.5],[3.5, 1], [3, 1.5],
                    [2.5, 2], [2.5, 3], [3, 2.5], [2, 2.5],
                    [3, 4.5], [3.5, 4],[3.5, 4.5], [3, 4] ])
display('x data are:', x_train)
y_train = np.array([1, 1, 1, 1,
                    2, 2, 2, 2,
                    3, 3, 3, 3,
                    5, 5, 5, 5,
                    4, 4, 4, 4]).reshape(-1,1)
display('y data are:', y_train)

# ============================================================
# 3. CREATE BOOLEAN MASKS FOR EACH CLASS
# ============================================================
one = y_train == 1
two = y_train == 2
three = y_train == 3
four = y_train == 4
five = y_train == 5

# ============================================================
# 4. VISUALIZE 5-CLASS TRAINING DATA
# ============================================================
# Flatten the boolean masks for correct indexing
# .flatten(), the one and zero arrays were converted into a 1-dimensional array

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

plt.scatter(
    x_train[five.flatten(), 0],
    x_train[five.flatten(), 1],
    marker='*',
    s=80,
    c='pink',
    label="y=5",
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
# 6. CREATE 5-CLASS NEURAL NETWORK
# 2 Inputs -> Hidden Layers -> 5 Linear Outputs
# ============================================================
model = Sequential(
    [
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(5, activation='linear')
    ]
)

# ============================================================
# 7. COMPILE MODEL USING LOGITS
# ============================================================
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01)
)

# ============================================================
# 8. ADJUST CLASS LABELS
# Original: 1, 2, 3, 4, 5
# TensorFlow: 0, 1, 2, 3, 4
# ============================================================
y_train_adjusted = y_train - 1

# ============================================================
# 9. TRAIN MULTICLASS MODEL
# ============================================================
model.fit(
    x_train,
    y_train_adjusted,
    epochs=100
)

# ============================================================
# 10. TEST MODEL WITH [2.5, 2.5]
# ============================================================
x_test = np.array([[2.5, 2.5]])

y_predicted = model.predict(x_test)

print(y_predicted)

predicted_index = np.argmax(y_predicted) + 1

print("Predicted index:", predicted_index)

# ============================================================
# 11. TEST MODEL WITH [1.5, 4.0]
# ============================================================
x_test = np.array([[1.5, 4.0]])

y_predicted = model.predict(x_test)

print(y_predicted)

predicted_class = np.argmax(y_predicted) + 1

print("Predicted class: y =", predicted_class)

# ============================================================
# 12. PREDICT BOTH TEST CASES
# ============================================================
test1 = np.array([[2.5, 2.5]])
test2 = np.array([[1.5, 4.0]])

pred1 = model.predict(test1, verbose=0)
pred2 = model.predict(test2, verbose=0)

class1 = np.argmax(pred1) + 1
class2 = np.argmax(pred2) + 1

print("Test 1 [2.5, 2.5] -> Predicted y =", class1)
print("Test 2 [1.5, 4.0] -> Predicted y =", class2)

# ============================================================
# 13. PREPARE CLASS MASKS FOR FINAL PLOT
# ============================================================
# 3. Plot
one = y_train == 1
two = y_train == 2
three = y_train == 3
four = y_train == 4
five = y_train == 5

# ============================================================
# 14. PLOT TRAINING DATA AND TEST POINTS
# ============================================================
plt.scatter(
    x_train[one.flatten(), 0],
    x_train[one.flatten(), 1],
    marker='x',
    s=80,
    c='red',
    label='y=1',
    lw=3
)

plt.scatter(
    x_train[two.flatten(), 0],
    x_train[two.flatten(), 1],
    marker='o',
    s=80,
    c='blue',
    label='y=2',
    lw=3
)

plt.scatter(
    x_train[three.flatten(), 0],
    x_train[three.flatten(), 1],
    marker='v',
    s=80,
    c='green',
    label='y=3',
    lw=3
)

plt.scatter(
    x_train[four.flatten(), 0],
    x_train[four.flatten(), 1],
    marker='+',
    s=80,
    c='orange',
    label='y=4',
    lw=3
)

plt.scatter(
    x_train[five.flatten(), 0],
    x_train[five.flatten(), 1],
    marker='*',
    s=80,
    c='pink',
    label='y=5',
    lw=3
)

# ============================================================
# 15. PLOT TEST CASE 1
# ============================================================
plt.scatter(
    test1[0, 0],
    test1[0, 1],
    marker='o',
    s=150,
    c='purple',
    label=f'Test 1'
)

# ============================================================
# 16. PLOT TEST CASE 2
# ============================================================
plt.scatter(
    test2[0, 0],
    test2[0, 1],
    marker='o',
    s=150,
    c='black',
    label=f'Test 2'
)

# ============================================================
# 17. DISPLAY FINAL MULTICLASS PLOT
# ============================================================
plt.title('Multi Class Classification Example')
plt.grid(True)
plt.axis([0, 5, 0, 5])
plt.ylabel('$x_1$')
plt.xlabel('$x_0$')
plt.legend()
plt.show()