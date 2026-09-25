# ============================================================
# Polynomial_Regression_From_Scratch_2_Iterations
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. CREATE NONLINEAR DATA
# y = 1 + x^2
# ============================================================
x = np.arange(0, 4, 1)
y = 1 + x**2

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.grid(True); plt.title("y = 1 + x**2")

# ============================================================
# 3. PREPARE TRAINING DATA
# ============================================================
x_train = np.array(x)
print(x_train)

y_train = np.array(y)
print(y_train)

# ============================================================
# 4. CREATE POLYNOMIAL FEATURES
# x -> [x, x^2, x^3]
# ============================================================
print('Feature x in')
X_feature = np.c_[x, x**2, x**3]
y_train = np.array(y)
print('x, x**2, x**3')
print(X_feature)
print('y')
print(y_train)

# ============================================================
# 5. COMPUTE COST FUNCTION
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
# 6. COMPUTE GRADIENTS FOR w AND b
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
# Update w and b using calculated gradients
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
# 8. COMPUTE MODEL OUTPUT
# f(x) = w.x + b
# ============================================================
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = np.dot(x[i], w) + b
    return f_wb

# ============================================================
# 9. SET INITIAL PARAMETERS
# 3 Weights for x, x^2 and x^3
# ============================================================
initial_w = np.array([1, 1, 1])
initial_b = 0

iterations = 2
alpha = 1.0e-3

# ============================================================
# 10. RUN GRADIENT DESCENT FOR 2 ITERATIONS
# ============================================================
w_optimal, b_optimal, J_hist = gradient_descent(X_feature, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

print(f"(w,b) found by gradient descent: ({w_optimal},{b_optimal})")

plt.scatter(x, y, marker='x', c='r', label='Actual Value')
plt.grid(True)
plt.title("with polonmial feature engineering")

# ============================================================
# 11. COMPUTE TRAINED MODEL OUTPUT
# ============================================================
y_target = compute_model_output(X_feature, w_optimal, b_optimal)
print(y_target)

# ============================================================
# 12. VISUALIZE ACTUAL VS ESTIMATED VALUES
# ============================================================
plt.plot(x, y_target, c = 'b', label='Estimation Value')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# ============================================================
# 13. TEST MODEL WITH x = 1.5
# Create [x, x^2, x^3] before prediction
# ============================================================
x_test = 1.5
x_test_feature = np.array([x_test, x_test**2, x_test**3])
y_target = np.dot(x_test_feature, w_optimal) + b_optimal
print(f"Predicted value: {y_target:.2f}")