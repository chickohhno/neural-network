# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. IMPORT SCIKIT-LEARN TOOLS
# ============================================================
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

# ============================================================
# 3. CREATE TRAINING DATA
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
# 4. MODEL OUTPUT / PREDICTION FUNCTION
# ============================================================
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = np.dot(x[i], w) + b
    return f_wb

# ============================================================
# 5. COST FUNCTION
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
# 6. COMPUTE GRADIENT
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
# 7. GRADIENT DESCENT
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
# 8. TRAIN OUR OWN LINEAR REGRESSION MODEL
# ============================================================
initial_w = [0,0,0,0]
initial_b = 0

iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

print (f"b,w found by gradient descent: {b_final:0.2f}, {w_final} ")
print (f"optimal w: {w_final}")
print (f"optimal b: {b_final:0.4f}")

# ============================================================
# 9. TEST OUR OWN MODEL
# ============================================================
x_test = np.array([1200, 3, 1, 40])
houseCost = np.dot(x_test, w_final) + b_final
print(f"Predicted house cost: {houseCost:.2f}")


# ============================================================
# PART 2
# USING SCIKIT-LEARN
# ============================================================

# ============================================================
# 10. NORMALIZE / STANDARDIZE TRAINING DATA
# ============================================================
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# ============================================================
# 11. TRAIN SCIKIT-LEARN MODEL
# ============================================================
sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# ============================================================
# 12. GET SCIKIT-LEARN MODEL PARAMETERS
# ============================================================
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameter: w:{w_norm}, b: {b_norm}")

# ============================================================
# 13. PREDICT USING SCIKIT-LEARN PARAMETERS
# ============================================================
x_test = np.array([1200, 3, 1, 40])
X_test_norm = scaler.transform(x_test.reshape(1, -1))
print(f"Peak to Peak range by column in Normalized X: {np.ptp(X_norm,axis=0)}")
houseCost = np.dot(X_test_norm, w_norm) + b_norm
print(f"Predicted house cose: {houseCost[0]:.2f}")