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
# 3. CREATE NEURAL NETWORK LAYER FUNCTION
# Matrix Multiplication: Z = XW + B
# Activation: A = sigmoid(Z)
# ============================================================
def compute_layer_vector(X, W, B):
    Z = np.matmul(X, W) + B
    print('Z value', Z)
    A = sigmoid(Z)
    return(A)

# ============================================================
# 4. LAYER 1 - 2 INPUTS TO 3 NEURONS
# ============================================================
w = np.array([ [1,-3, 5],[2, 4, -6] ])
b = np.array([ [-1, 1, 2] ])
X = np.array([[200,17]])
layer1 = compute_layer_vector(X,w,b)
print(f'a: {layer1}')

# ============================================================
# 5. LAYER 2 - 3 INPUTS TO 1 OUTPUT NEURON
# ============================================================
w = np.array([ [-7],[8],[9] ] )
b = np.array( [[3]] )
X = np.array([[1,0,1]])
layer2 = compute_layer_vector(X,w,b)
print(f'a: {layer2}')