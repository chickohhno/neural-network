# ============================================================
# Neural_Network_TensorFlow_3_Layer_Forward_Propagation
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ============================================================
# 2. LAYER 1 - 2 INPUTS TO 4 NEURONS
# Linear Activation
# ============================================================
# Layer 1
w_init = np.array([ [1, -2, 3, -3],[-1, 4, 2, 1] ])
b_init = np.array([ [1], [0], [-1], [0] ]).flatten() # convert to 1D array
X = np.array([[2, 3]]) # Reshape X to be 2-dimensional
layer1 = Dense(units = 4, activation = 'linear')
# Build the layer first to allow setting weights
layer1.build(input_shape=(None, X.shape[1]))
layer1.set_weights([w_init, b_init])
a1 = layer1(X)
print(f'a_out layer1: {a1}')

# ============================================================
# 3. LAYER 2 - 4 INPUTS TO 2 NEURONS
# Sigmoid Activation
# ============================================================
# Layer 2
w_2 = np.array([ [3, 6],[-1, 0],[4, -2], [-2, 1] ])
b_2 = np.array([1, 0])
X = a1
layer2 = Dense(units = 2, activation = 'sigmoid')
# Build the layer first to allow setting weights
layer2.build(input_shape=(None, X.shape[1]))
layer2.set_weights([w_2, b_2])
a2 = layer2(X)
print(f'a_out layer2: {a2}')

# ============================================================
# 4. LAYER 3 - 2 INPUTS TO 1 OUTPUT NEURON
# Sigmoid Activation
# ============================================================
# Layer 3
w_3 = np.array([ [2], [-1]])
b_3 = np.array([1])
X = a2
layer3 = Dense(units = 1, activation = 'sigmoid')
# Build the layer first to allow setting weights
layer3.build(input_shape=(None, X.shape[1]))
layer3.set_weights([w_3, b_3])
a3 = layer3(X)
print(f'a_out layer3: {a3}')