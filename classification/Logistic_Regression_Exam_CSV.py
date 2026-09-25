# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# ============================================================
# 2. UPLOAD AND READ CSV DATA
# ============================================================
from google.colab import files
uploaded = files.upload()
Train_df = pd.read_csv('ExamData.csv')
print("These are the Examination data")
print(Train_df)

# ============================================================
# 3. PREPARE TRAINING DATA / TARGET CLASS LABELS
# ============================================================
x_train = np.array(Train_df[['Exam1', 'Exam2']])
print('Exam data:')
print(x_train)
y_train = np.array(Train_df['Result'])
print('Result', y_train)

# ============================================================
# 4. CREATE BOOLEAN MASKS FOR THE TWO CLASSES
# ============================================================
one = y_train == 1
zero = y_train == 0

print(one.flatten())
print(x_train[one.flatten()])

# ============================================================
# 5. VISUALIZE EXAM DATA BY CLASS
# ============================================================
plt.scatter(x_train[one.flatten(), 0], x_train[one.flatten(), 1], color='r', label='y=1', marker='x')
plt.scatter(x_train[zero.flatten(), 0], x_train[zero.flatten(), 1], color='b', label='y=0', marker='o')
plt.xlabel('$Exam1$')
plt.ylabel('$Exam2$')
plt.axis([0, 100, 0, 100])
plt.title('Boundary Classification Example');
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# ============================================================
# 6. VISUALIZE DATA WITH DECISION BOUNDARY
# Decision Boundary: Exam1 + Exam2 = 80
# ============================================================
one = y_train == 1
zero = y_train == 0

print(one.flatten())
print(x_train[one.flatten()])

plt.scatter(x_train[one.flatten(), 0], x_train[one.flatten(), 1], color='r', label='y=1', marker='x')
plt.scatter(x_train[zero.flatten(), 0], x_train[zero.flatten(), 1], color='b', label='y=0', marker='o')

plt.axis([0, 100, 0, 100])

plt.xlabel('$Exam1$')
plt.ylabel('$Exam2$')

x1 = np.arange(0,100)
x2 = 80 - x1

plt.plot(x1, x2, c='b')
plt.axis([0, 100, 0, 100])

plt.fill_between(x1,x2,alpha=0.2)

plt.title('Boundary Classification Example');
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# ============================================================
# 7. MULTIPLE-FEATURE LINEAR MODEL
# z = w.x + b
# ============================================================
def compute_multi_model(x,w,b):
  f_wb = np.dot(x, w) + b
  return f_wb

# ============================================================
# 8. SIGMOID FUNCTION
# Converts z into a value between 0 and 1
# ============================================================
def sigmoid(z):

  g = 1/(1+np.exp(-z))

  return g

# ============================================================
# 9. TEST CASE / MAKE PASS-FAIL CLASSIFICATION
# ============================================================
x_test = [50, 50]
w = np.array([1, 1])
b = -80
z_tmp = compute_multi_model(x_test, w, b)
y_test = sigmoid(z_tmp)
print(f'Prediction is {y_test}')

# ============================================================
# 10. CLASSIFY USING 0.5 THRESHOLD
# ============================================================
if y_test >= 0.5:
  display("pass and accepted")
else:
  display("fail and not accepted")