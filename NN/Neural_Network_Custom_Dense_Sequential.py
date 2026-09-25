# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. CREATE SIGMOID ACTIVATION FUNCTION
# ============================================================
def sigmoid(z):

    g = 1/(1+np.exp(-z))

    return g

# ============================================================
# 3. CREATE CUSTOM DENSE LAYER
# Matrix Multiplication: Z = AT.W + B
# Activation: A = sigmoid(Z)
# ============================================================
def my_dense(AT, W, B):
    Z = np.matmul(AT, W) + B
    print('Z value', Z)
    A = sigmoid(Z)
    return(A)

# ============================================================
# 4. LAYER 1 - 2 INPUTS TO 3 NEURONS
# ============================================================
w = np.array([ [1,-3, 5],[2, 4, -6] ])
print('w are: \n', w)
b = np.array([ -1, 1, 2 ])
print('b are: \n', b)
X = np.array([[200,17]])
layer1 = my_dense(X,w,b)
print(f'a: {layer1}')

# ============================================================
# 5. LAYER 2 - 3 INPUTS TO 1 OUTPUT NEURON
# ============================================================
w = np.array([ [-7],[8],[9] ] )
print('w are: \n', w)
b = np.array( [[3]])
print('b are: \n', b)
X = np.array(layer1)
layer2 = my_dense(X,w,b)
print(f'a: {layer2}')

# ============================================================
# 6. CREATE CUSTOM SEQUENTIAL NEURAL NETWORK
# Connect Layer 1 Output -> Layer 2 Input
# ============================================================
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    print('a1 is: ', a1)
    a2 = my_dense(a1, W2, b2)
    print('a2 is: ', a2)
    return(a2)

# ============================================================
# 7. DEFINE WEIGHTS, BIASES AND INPUT FOR FULL NETWORK
# ============================================================
W1 = np.array([ [1,-3, 5],[2, 4, -6] ])
b1 = np.array([ [-1, 1, 2] ])
W2 = np.array([ [-7],[8],[9] ] )
b2 = np.array([ 3 ])
X = np.array([[200,17]])

# ============================================================
# 8. RUN FORWARD PROPAGATION THROUGH FULL NETWORK
# Input -> Layer 1 -> Layer 2 -> Final Output
# ============================================================
layer2_out = my_sequential(X,W1,b1,W2,b2)
print(f'a_out layer 2: {layer2_out}')