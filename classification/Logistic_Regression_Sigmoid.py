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
x_data = np.array([0,1,2,3,4,5])
display(x_data)

y_data = np.array([0,0,0,1,1,1])
display(y_data)

# ============================================================
# 3. CREATE BOOLEAN MASKS FOR THE TWO CLASSES
# ============================================================
one = y_data == 1
display(one)

zero = y_data == 0
display(zero)

# ============================================================
# 4. CREATE LINEAR MODEL
# z = wx + b
# ============================================================
def compute_model(x,w,b):

  f_wb = w * x + b

  return f_wb


# ============================================================
# 5. CREATE SIGMOID FUNCTION
# Converts z into a value between 0 and 1
# ============================================================
def sigmoid(z):

  g = 1/(1+np.exp(-z))

  return g

# ============================================================
# 6. CALCULATE MODEL OUTPUT AND APPLY SIGMOID
# Shift by 2.5 to place the classification threshold at x = 2.5
# ============================================================
z_tmp = compute_model(x_data, 1, 0)
g = sigmoid(z_tmp-2.5)
display(g)

# ============================================================
# 7. PLOT BASIC SIGMOID OUTPUT
# ============================================================
plt.plot(z_tmp, g, c="b")
plt.title('Classification with w =1, b = 0')
plt.ylabel('g from sigmoid')
plt.xlabel('z')
plt.grid(True)

# ============================================================
# 8. CREATE SMOOTH VALUES FOR SIGMOID GRAPH
# ============================================================
from matplotlib.lines import lineStyles
g = sigmoid(z_tmp-2.5)
z_smooth_range = np.linspace(x_data.min(),x_data.max(),100)

g_smooth = sigmoid((z_smooth_range-2.5))

# ============================================================
# 9. VISUALIZE SIGMOID, CLASSES AND CLASSIFICATION THRESHOLD
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(z_smooth_range, g_smooth, label='sigmoid(z - 2.5)', c='purple')
plt.scatter(x_data[one], y_data[one], color="r", label="y=1", s=100, marker = 'x')
plt.scatter(x_data[zero], y_data[zero], color="b", label="y=0", s=100, marker = 'o')
plt.title('Sigmoid Function with threshold at (2.5, 0,5)')
plt.ylabel('sigmoid(z - 2.5)')
plt.xlabel('z')
plt.axvline(x=2.5, color='grey', linestyle='--', linewidth=0.7)
plt.axhline(y=0.5, color='grey', linestyle='--', linewidth=0.7)
plt.scatter(2.5, 0.5, color='red', zorder=5, s=100, label='(2.5, 0.5) threshold')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.show()

# ============================================================
# 10. TEST CASE / MAKE CLASSIFICATION
# ============================================================
x_test = 1.5
z_prediction = compute_model(x_test, 1, 0)
g_predict = sigmoid(z_prediction-2.5)
display(g_predict)

if g_predict >= 0.5:
  display("Prediction: 1")
else:
  display("Prediction: 0")