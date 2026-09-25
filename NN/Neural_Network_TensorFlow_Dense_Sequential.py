# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ============================================================
# 2. CREATE LAYER 1 USING TENSORFLOW DENSE
# 2 Inputs -> 3 Neurons -> Sigmoid Activation
# ============================================================
w_init = np.array([ [1,-3, 5],[2, 4, -6] ])
b_init = np.array([ [-1], [1], [2] ]).flatten() # convert to 1D array
X = np.array([[200,17]]) # Reshape X to be 2-dimensional
layer1 = Dense(units = 3, activation = 'sigmoid')
# Build the layer first to allow setting weights
layer1.build(input_shape=(None, X.shape[1]))
layer1.set_weights([w_init, b_init])
a1 = layer1(X)
print(f'a_out layer1: {a1}')

# ============================================================
# 3. CREATE LAYER 2 USING TENSORFLOW DENSE
# 3 Inputs -> 1 Output Neuron -> Sigmoid Activation
# ============================================================
w_2 = np.array([ [-7],[8],[9] ] )
b_2 = np.array([3])
X = a1
layer2 = Dense(units = 1, activation = 'sigmoid')
# Build the layer first to allow setting weights
layer2.build(input_shape=(None, X.shape[1]))
layer2.set_weights([w_2, b_2])
a2 = layer2(X)
print(f'a_out layer2: {a2}')

# ============================================================
# 4. DEFINE WEIGHTS, BIASES AND INPUT FOR FULL NETWORK
# ============================================================
W1 = np.array([ [1,-3, 5],[2, 4, -6] ])
b1 = np.array([ [-1], [1], [2] ]).flatten()
W2 = np.array([ [-7],[8],[9] ] )
b2 = np.array([3])
X = np.array([[200,17]])

# ============================================================
# 5. CREATE FULL NEURAL NETWORK USING SEQUENTIAL
# Input -> 3-Neuron Hidden Layer -> 1-Neuron Output Layer
# ============================================================
myModel = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(units=3, activation='sigmoid'),
    Dense(units=1, activation='sigmoid') ])

# ============================================================
# 6. SET WEIGHTS AND BIASES FOR EACH LAYER
# ============================================================
myModel.layers[0].set_weights([W1, b1])
myModel.layers[1].set_weights([W2, b2])

# ============================================================
# 7. RUN FORWARD PROPAGATION THROUGH FULL NETWORK
# ============================================================
layer2_out = myModel(X)
print(f'a_out layer 2: {layer2_out}')

# ============================================================
# 8. MAKE FINAL PREDICTION
# ============================================================
predictions = myModel(X)
print('prediction is: ', predictions)