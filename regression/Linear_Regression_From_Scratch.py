# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. PREPARE TRAINING DATA
# ============================================================
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
data_input_frame = pd.DataFrame(X_train, columns=['size', 'bedrooms', 'floors', 'age'])
print('Train data')
display(data_input_frame)
data_target_frame = pd.DataFrame(y_train, columns=['price'])
print('Target data')
display(data_target_frame)


# ============================================================
# 3. LINEAR REGRESSION MODEL
# f(x) = w.x + b
# ============================================================
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = np.dot(x[i], w) + b
    return f_wb


# ============================================================
# 4. COST FUNCTION
# ============================================================
def compute_cost(X, y, w, b):

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar
    return cost


# ============================================================
# 5. COMPUTE GRADIENT
# ============================================================
def compute_gradient(X, y, w, b):

    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

# ============================================================
# 6. GRADIENT DESCENT
# ============================================================
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history #return final w,b and J history for graphing


# ============================================================
# 7. SET INITIAL PARAMETERS
# ============================================================
initial_w = [0,0,0,0]
initial_b = 0

iterations = 1000
alpha = 5.0e-7

# ============================================================
# 8. TRAIN THE LINEAR REGRESSION MODEL
# ============================================================
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

print (f"b,w found by gradient descent: {b_final:0.2f}, {w_final} ")
print (f"optimal w: {w_final}")
print (f"optimal b: {b_final:0.4f}")


# ============================================================
# 9. TEST THE TRAINED MODEL
# House: 1200 sqft, 3 bedrooms, 1 floor, 40 years old
# ============================================================
x_test = np.array([1200, 3, 1, 40])
houseCost = np.dot(x_test, w_final) + b_final
print(f"Predicted house cost: {houseCost:.2f}")