
# Neural Network Binary Classification
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
# ============================================================
x_train = np.array([
    [0.5, 1.5],
    [1, 1],
    [1.5, 0.5],
    [3, 0.5],
    [2, 2],
    [1, 2.5]
])

display('x data are:', x_train)

y_train = np.array([
    0, 0, 0, 1, 1, 1
]).reshape(-1, 1)

display('y data are:', y_train)


# ============================================================
# 3. CREATE BOOLEAN MASKS
# ============================================================
one = y_train == 1
zero = y_train == 0

print(one.flatten())
print(x_train[one.flatten()])


# ============================================================
# 4. VISUALIZE BINARY CLASSIFICATION DATA
# ============================================================
plt.scatter(
    x_train[one.flatten(), 0],
    x_train[one.flatten(), 1],
    marker='x',
    label='y=1'
)

plt.scatter(
    x_train[zero.flatten(), 0],
    x_train[zero.flatten(), 1],
    marker='o',
    label='y=0'
)

plt.xlabel('$x_1$')
plt.ylabel('$x_0$')
plt.axis([0, 4, 0, 3.5])
plt.title('Boundary Classification Example')
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 5. IMPORT TENSORFLOW / KERAS
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError


# ============================================================
# 6. CREATE BINARY CLASSIFICATION NEURAL NETWORK
# 2 Inputs -> 5 Neurons -> 3 Neurons -> 1 Output
# ============================================================
model = Sequential(
    [
        Dense(5, activation='sigmoid'),
        Dense(3, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ]
)


# ============================================================
# 7. COMPILE MODEL
# ============================================================
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.1)
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================
model.fit(
    x_train,
    y_train,
    epochs=20
)


# ============================================================
# 9. TEST CASE 1
# ============================================================
x_test = np.array([[1.5, 2]])

y_test = model.predict(x_test)

print(y_test)

if y_test > 0.5:
    print('y = 1')
else:
    print('y = 0')


# ============================================================
# 10. TEST CASE 2
# ============================================================
x_test = np.array([[0.5, 0.5]])

y_test = model.predict(x_test)

print('y is:', y_test)

if y_test > 0.5:
    print('y = 1')
else:
    print('y = 0')

# ============================================================
# Neural Network Nonlinear Regression
# ============================================================
# 11. CREATE NONLINEAR TARGET DATA
# y = 1 + x^2
# ============================================================
x = np.arange(0, 20, 1)

y = 1 + x**2

plt.scatter(
    x,
    y,
    marker='x',
    c='r',
    label='Actual Value'
)

plt.grid(True)
plt.title("y = 1 + x**2")


# ============================================================
# 12. CREATE POLYNOMIAL FEATURES
# x, x^2, x^3
# ============================================================
print('Feature aa in')

X_feature = np.c_[
    x,
    x**2,
    x**3
]

y_train = np.array(y)

print('x, x**2, x**3')
print(X_feature)

print('y')
print(y_train)


# ============================================================
# 13. CREATE REGRESSION NEURAL NETWORK
# 3 Inputs -> 4 Neurons -> 8 Neurons -> 1 Output
# ============================================================
model = Sequential(
    [
        Dense(4, activation='linear'),
        Dense(8, activation='linear'),
        Dense(1, activation='linear')
    ]
)


# ============================================================
# 14. COMPILE REGRESSION MODEL
# Mean Squared Error + Adam
# ============================================================
model.compile(
    loss=MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.01)
)


# ============================================================
# 15. TRAIN REGRESSION MODEL
# ============================================================
model.fit(
    X_feature,
    y_train,
    epochs=100
)


# ============================================================
# 16. TEST REGRESSION MODEL - x = 15.5
# ============================================================
x_test = 15.5

x_test_feature = np.array([
    x_test,
    x_test**2,
    x_test**3
])

x_test_feature = x_test_feature.reshape(1, -1)

print(x_test_feature)

y_test = model.predict(x_test_feature)

print(f"Predicted value: {y_test[0][0]:.2f}")


# ============================================================
# 17. TEST REGRESSION MODEL - x = 10.5
# ============================================================
x_test = 10.5

x_test_feature = np.array([
    x_test,
    x_test**2,
    x_test**3
])

x_test_feature = x_test_feature.reshape(1, -1)

print(x_test_feature)

y_test = model.predict(x_test_feature)

print(f"Predicted value: {y_test[0][0]:.2f}")

# ============================================================
# Exam CSV Neural Network Classification
# ============================================================
# 18. UPLOAD EXAM DATA CSV
# ============================================================
from google.colab import files

uploaded = files.upload()

Train_df = pd.read_csv('ExamData.csv')

print("These are the Examination data")
print(Train_df)


# ============================================================
# 19. PREPARE EXAM TRAINING DATA
# ============================================================
x_train = np.array(
    Train_df[['Exam1', 'Exam2']]
)

print('Exam data:')
print(x_train)

y_train = np.array(
    Train_df['Result']
)

print('Result', y_train)


# ============================================================
# 20. CREATE BOOLEAN MASKS
# ============================================================
one = y_train == 1
zero = y_train == 0

print(one.flatten())
print(x_train[one.flatten()])


# ============================================================
# 21. VISUALIZE EXAM CLASSIFICATION DATA
# ============================================================
plt.scatter(
    x_train[one.flatten(), 0],
    x_train[one.flatten(), 1],
    marker='x',
    label='y=1'
)

plt.scatter(
    x_train[zero.flatten(), 0],
    x_train[zero.flatten(), 1],
    marker='o',
    label='y=0'
)

plt.xlabel('$Exam1$')
plt.ylabel('$Exam2$')
plt.axis([0, 100, 0, 100])
plt.title('Boundary Classification Example')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()


# ============================================================
# 22. CREATE EXAM CLASSIFICATION NEURAL NETWORK
# 2 Inputs -> 4 ReLU -> 8 ReLU -> 1 Sigmoid
# ============================================================
model = Sequential(
    [
        Dense(4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)


# ============================================================
# 23. COMPILE MODEL
# ============================================================
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)


# ============================================================
# 24. TRAIN MODEL
# ============================================================
model.fit(
    x_train,
    y_train,
    epochs=300
)


# ============================================================
# 25. TEST CASE 1 - EXAM SCORES [20, 40]
# ============================================================
x_test = np.array([
    [20, 40]
])

y_test = model.predict(x_test)

print(y_test)

if y_test > 0.5:
    print('y = 1')
else:
    print('y = 0')


# ============================================================
# 26. TEST CASE 2 - EXAM SCORES [60, 80]
# ============================================================
x_test = np.array([
    [60, 80]
])

y_test = model.predict(x_test)

print(y_test)

if y_test > 0.5:
    print('y = 1')
else:
    print('y = 0')


# ============================================================
# Full Exam Dataset Classification
# ============================================================
# 27. UPLOAD FULL EXAM DATASET
# ============================================================
from google.colab import files

uploaded = files.upload()

Train_df = pd.read_csv('ExamFullData.csv')

print("These are the Examination data")
print(Train_df)


# ============================================================
# 28. PREPARE FULL EXAM TRAINING DATA
# ============================================================
x_train = np.array(
    Train_df[['Exam1', 'Exam2']]
)

print('Exam data:')
print(x_train)

y_train = np.array(
    Train_df['Result']
)

print('Result', y_train)


# ============================================================
# 29. CREATE BOOLEAN MASKS
# ============================================================
one = y_train == 1
zero = y_train == 0

print(one.flatten())
print(x_train[one.flatten()])


# ============================================================
# 30. VISUALIZE FULL EXAM DATASET
# ============================================================
plt.scatter(
    x_train[one.flatten(), 0],
    x_train[one.flatten(), 1],
    marker='x',
    label='y=1'
)

plt.scatter(
    x_train[zero.flatten(), 0],
    x_train[zero.flatten(), 1],
    marker='o',
    label='y=0'
)

plt.xlabel('$Exam1$')
plt.ylabel('$Exam2$')
plt.axis([0, 100, 0, 100])
plt.title('Boundary Classification Example')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()


# ============================================================
# 31. CREATE NEURAL NETWORK - PART A
# 2 Inputs -> 4 ReLU -> 8 ReLU -> 1 Sigmoid
# ============================================================
model = Sequential(
    [
        Dense(4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)


# ============================================================
# 32. COMPILE MODEL - PART A
# ============================================================
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)


# ============================================================
# 33. TRAIN MODEL - PART A
# ============================================================
model.fit(
    x_train,
    y_train,
    epochs=1000
)


# ============================================================
# 34. GET OUTPUT LAYER WEIGHTS AND BIAS
# ============================================================
w_out, b_out = model.layers[-1].get_weights()

print(
    f"Weights of the output layer: "
    f"{w_out.flatten()}"
)

print(
    f"w of layer1: {w_out[0].flatten()}, "
    f"w of layer2: {w_out[1].flatten()}"
)

print(
    f"Bias of the output layer: {b_out}"
)


# ============================================================
# 35. TEST MODEL - EXAM SCORES [60, 60]
# ============================================================
x_test = np.array([
    [60, 60]
])

y_test = model.predict(x_test)

print(y_test)

if y_test > 0.5:
    print('y = 1')
else:
    print('y = 0')


# ============================================================
# 36. CREATE NEURAL NETWORK - PART B
# 2 Inputs -> 4 ReLU -> 16 ReLU -> 1 Sigmoid
# ============================================================
model = Sequential(
    [
        Dense(4, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)


# ============================================================
# 37. COMPILE MODEL - PART B
# ============================================================
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)


# ============================================================
# 38. TRAIN MODEL - PART B
# ============================================================
model.fit(
    x_train,
    y_train,
    epochs=1000
)


# ============================================================
# 39. GET PART B OUTPUT LAYER WEIGHTS AND BIAS
# ============================================================
w_out, b_out = model.layers[-1].get_weights()

print(
    f"Weights of the output layer: "
    f"{w_out.flatten()}"
)