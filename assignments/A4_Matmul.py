# ============================================================
# Neural_Network_3_Layer_MatMul_Forward_Propagation
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
# 3. CREATE VECTORIZED NEURAL NETWORK LAYER
# Z = XW + B -> A = sigmoid(Z)
# ============================================================
def compute_layer_vector(X, W, B):
    Z = np.matmul(X, W) + B
    print('Z value', Z)
    A = sigmoid(Z)
    return(A)

# ============================================================
# 4. LAYER 1 - 2 INPUTS TO 4 NEURONS
# ============================================================
#layer 1
w = np.array([ [1, -2, 3, -3],[-1, 4, 2, 1] ])
b = np.array([ [1, 0, -1, 0] ])
X = np.array([[2, 3]])
layer1 = compute_layer_vector(X,w,b)
print(f'a: {layer1}')

# ============================================================
# 5. LAYER 2 - 4 INPUTS TO 2 NEURONS
# ============================================================
#layer 2
w = np.array([ [3, 6],[-1, 0],[4, -2], [-2, 1] ])
b = np.array( [[1, 0]] )
X = np.array([[0, 8, 11, -3]])
layer2 = compute_layer_vector(X,w,b)
print(f'a: {layer2}')

# ============================================================
# 6. LAYER 3 - 2 INPUTS TO 1 OUTPUT NEURON
# ============================================================
#layer 3
w = np.array([ [2], [-1]])
b = np.array( [[1]] )
X = np.array([[1.00000000e+00, 1.38879439e-11]])
layer3 = compute_layer_vector(X,w,b)
print(f'a: {layer3}')