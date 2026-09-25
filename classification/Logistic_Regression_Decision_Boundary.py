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
# 4. VISUALIZE THE TRAINING DATA
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
# 5. VISUALIZE DATA WITH DECISION BOUNDARY
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
# 6. MULTIPLE-FEATURE LINEAR MODEL
# z = w.x + b
# ============================================================
def compute_multi_model(x,w,b):
  f_wb = np.dot(x, w) + b
  return f_wb

# ============================================================
# 7. SIGMOID FUNCTION
# Converts z into a value between 0 and 1
# ============================================================
def sigmoid(z):

  g = 1/(1+np.exp(-z))

  return g

# ============================================================
# 8. TEST CASE / MAKE CLASSIFICATION
# x_test = [x0, x1]
# ============================================================
x_test = [2, 0.5]
w = np.array([1, 1])
b = -3
z_tmp = compute_multi_model(x_test, w, b)
g = sigmoid(z_tmp)
display(g)

if g >= 0.5:
  display("y_classification: True")
else:
  display("y_classification: False")