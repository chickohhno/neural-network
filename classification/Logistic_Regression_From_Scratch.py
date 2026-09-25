# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. CREATE TRAINING DATA / TARGET CLASS LABELS
# ============================================================
x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
display('x data are:', x_train)
y_train = np.array([0, 0, 0, 1, 1, 1]).reshape(-1 ,1)
display('y data are:', y_train)

# ============================================================
# 3. CREATE BOOLEAN MASKS FOR THE TWO CLASSES
# ============================================================
one = y_train == 1
zero = y_train == 0

print(one.flatten())
print(x_train[one.flatten()])

# ============================================================
# 4. VISUALIZE TRAINING DATA
# ============================================================
plt.scatter(x_train[one.flatten(), 0], x_train[one.flatten(), 1], color='r', label='y=1', marker='x')
plt.scatter(x_train[zero.flatten(), 0], x_train[zero.flatten(), 1], color='b', label='y=0', marker='o')
plt.xlabel('$x_1$')
plt.ylabel('$x_0$')
plt.axis([0, 4, 0, 3.5])
plt.title('Boundary Classification Example');
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 5. VISUALIZE DATA WITH INITIAL DECISION BOUNDARY
# Decision Boundary: x0 + x1 = 3
# ============================================================
one = y_train == 1
zero = y_train == 0

print(one.flatten())
print(x_train[one.flatten()])

plt.scatter(x_train[one.flatten(), 0], x_train[one.flatten(), 1], color='r', label='y=1', marker='x')
plt.scatter(x_train[zero.flatten(), 0], x_train[zero.flatten(), 1], color='b', label='y=0', marker='o')

plt.axis([0, 4, 0, 3.5])

plt.xlabel('$x_1$')
plt.ylabel('$x_0$')

x0 = np.arange(0,5)
x1 = 3 - x0

plt.plot(x0, x1, c='b')
plt.axis([0, 4, 0, 3.5])

plt.fill_between(x0,x1,alpha=0.2)

plt.title('Boundary Classification Example');
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 6. SIGMOID FUNCTION
# Converts z into a value between 0 and 1
# ============================================================
def sigmoid(z):

    g = 1/(1+np.exp(-z))

    return g

# ============================================================
# 7. COMPUTE LOGISTIC REGRESSION COST
# Uses binary cross-entropy / log loss
# ============================================================
def compute_cost_logistic(X, y, w, b):

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    cost = cost / m
    return cost

# ============================================================
# 8. COMPUTE GRADIENTS FOR w AND b
# Calculates how much the parameters should change
# ============================================================
def compute_gradient_logistic(X, y, w, b):

    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)
        err_i  = f_wb_i  - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db, dj_dw

# ============================================================
# 9. GRADIENT DESCENT
# Repeatedly updates w and b to reduce the logistic cost
# ============================================================
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    #w = copy.deepcopy(w_in)  #avoid modifying global w within function
    w = w_in
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history         #return final w,b and J history for graphing

# ============================================================
# 10. TRAIN LOGISTIC REGRESSION MODEL
# Start with w = 0 and b = 0
# ============================================================
w_tmp = [0,0]
b_tmp = 0
alpha = 0.1
num_iters = 10000

w_out, b_out, J_hist = gradient_descent(x_train, y_train, w_tmp, b_tmp, alpha, num_iters)
print(f"w optimal {w_out}, and b_optimal {b_out}")

# ============================================================
# 11. MULTIPLE-FEATURE LINEAR MODEL
# z = w.x + b
# ============================================================
def compute_multi_model(x,w,b):
  f_wb = np.dot(x, w) + b
  return f_wb

# ============================================================
# 12. TEST CASE 1 / MAKE CLASSIFICATION
# ============================================================
x_test = np.array([1.5,2])

z_test = compute_multi_model(x_test, w_out, b_out)
y_test = sigmoid(z_test)
print(y_test)

if y_test >= 0.5:
  print('y = 1')
else:
  print('y = 0')

# ============================================================
# 13. TEST CASE 2 / MAKE CLASSIFICATION
# ============================================================
x_test = np.array([2,3])

z_test = compute_multi_model(x_test, w_out, b_out)
y_test = sigmoid(z_test)
print(y_test)

if y_test >= 0.5:
  print('y = 1')
else:
  print('y = 0')