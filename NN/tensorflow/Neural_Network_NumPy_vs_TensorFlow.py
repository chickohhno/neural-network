# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ============================================================
# 2. CREATE SIGMOID ACTIVATION FUNCTION
# ============================================================
def sigmoid(z):

    g = 1/(1+np.exp(-z))

    return g

# ============================================================
# 3. CREATE CUSTOM DENSE LAYER USING NUMPY
# Z = AT.W + B -> A = sigmoid(Z)
# ============================================================
def my_dense(AT, W, B):
    Z = np.matmul(AT, W) + B
    print('Z value', Z)
    A = sigmoid(Z)
    return(A)

# ============================================================
# 4. NUMPY LAYER 1 - 2 INPUTS TO 3 NEURONS
# ============================================================
w = np.array([ [1,-2, 3],[1, -4, 2] ])
print('w are: \n', w)
b = np.array([ 1, 0, -1 ])
print('b are: \n', b)
X = np.array([[1.5,2]])
layer1 = my_dense(X,w,b)
print(f'a: {layer1}')

# ============================================================
# 5. NUMPY LAYER 2 - 3 INPUTS TO 1 OUTPUT NEURON
# ============================================================
w = np.array([ [-5],[1],[4] ] )
print('w are: \n', w)
b = np.array( [[1]])
print('b are: \n', b)
X = np.array(layer1)
layer2 = my_dense(X,w,b)
print(f'a: {layer2}')

# ============================================================
# 6. TENSORFLOW LAYER 1 - 2 INPUTS TO 3 NEURONS
# ============================================================
w_init = np.array([ [1,-2, 3],[1, -4, 2] ])
b_init = np.array([ [1], [0], [-1] ]).flatten() # convert to 1D array
X = np.array([[1.5,2]]) # Reshape X to be 2-dimensional
layer1 = Dense(units = 3, activation = 'sigmoid')
# Build the layer first to allow setting weights
layer1.build(input_shape=(None, X.shape[1]))
layer1.set_weights([w_init, b_init])
a1 = layer1(X)
print(f'a_out layer1: {a1}')

# ============================================================
# 7. TENSORFLOW LAYER 2 - 3 INPUTS TO 1 OUTPUT NEURON
# ============================================================
w_2 = np.array([ [-5],[1],[4] ] )
b_2 = np.array([1])
X = a1
layer2 = Dense(units = 1, activation = 'sigmoid')
# Build the layer first to allow setting weights
layer2.build(input_shape=(None, X.shape[1]))
layer2.set_weights([w_2, b_2])
a2 = layer2(X)
print(f'a_out layer2: {a2}')

# ============================================================
# 8. DEFINE WEIGHTS, BIASES AND INPUT FOR FULL NETWORK
# ============================================================
W1 = np.array([ [1,-2, 3],[1, -4, 2] ])
b1 = np.array([ [1], [0], [-1] ]).flatten()
W2 = np.array([ [-5],[1],[4] ] )
b2 = np.array([1])
X = np.array([[1.5,2]])

# ============================================================
# 9. CREATE FULL NEURAL NETWORK USING SEQUENTIAL
# Input -> 3-Neuron Hidden Layer -> 1-Neuron Output Layer
# ============================================================
myModel = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(units=3, activation='sigmoid'),
    Dense(units=1, activation='sigmoid') ])

# ============================================================
# 10. SET WEIGHTS AND BIASES FOR EACH TENSORFLOW LAYER
# ============================================================
# Corrected indexing for setting weights
myModel.layers[0].set_weights([W1, b1])
myModel.layers[1].set_weights([W2, b2])

# ============================================================
# 11. RUN FORWARD PROPAGATION THROUGH FULL NETWORK
# ============================================================
layer2_out = myModel(X)
print(f'a_out layer 2: {layer2_out}')

# ============================================================
# 12. MAKE FINAL PREDICTION
# ============================================================
predictions = myModel(X)
print('prediction is: ', predictions)