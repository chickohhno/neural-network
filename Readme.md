# SC6611 Machine Learning — Final Exam Code Guide

A collection of Python examples for the **SC6611 Final Exam**, organized for quick reference during study and the open-book coding section.

The repository covers:

- Linear Regression
- Multiple Input Regression
- Logistic Regression
- Binary Classification
- Neural Networks
- Forward Propagation
- Matrix Multiplication
- TensorFlow / Keras
- Multiclass Classification
- Neural Network Regression

> The purpose of this repository is to provide simple reference implementations that are easy to understand, modify, and reuse during practice.

---

# 📁 Repository Structure

```text
SC6611/
│
├── assign_sc6611/
│   ├── A2_Regression_MultiInput.py
│   ├── A3_Classification.py
│   ├── A4_Matmul.py
│   ├── A4_TensorFlow.py
│   ├── A5_Multi_Classification_TensorFlow.py
│   └── A5_Prediction_TensorFlow.py
│
├── classification/
│   ├── Logistic_Regression_Decision_Boundary.py
│   ├── Logistic_Regression_Exam_CSV.py
│   ├── Logistic_Regression_From_Scratch.py
│   ├── Logistic_Regression_Sigmoid.py
│   └── Logistic_Regression_Weighted_Boundary.py
│
├── NN/
│   ├── matmul/
│   │   └── MatMul_Forward_Propagation.py
│   │
│   └── tensorflow/
│       ├── Neural_Network_Multiclass_Logits.py
│       ├── Neural_Network_Multiclass_Softmax.py
│       ├── Neural_Network_NumPy_vs_TensorFlow.py
│       ├── Neural_Network_Custom_Dense_Sequential.py
│       ├── Neural_Network_Manual_Layer_Loop.py
│       └── Neural_Network_TensorFlow_Dense_Sequential.py
│
└── regression/
    ├── Linear_Regression_From_Scratch.py
    └── Linear_Regression_Sklearn.py
```

---

# 🚀 Quick Exam Guide

If you are not sure which code to use, start here.

| Question Type | Look For | Main Code |
|---|---|---|
| Linear Regression | Continuous numerical output | `regression/` |
| Multiple Input Regression | Several X features → one numerical Y | `Linear_Regression_From_Scratch.py` |
| Logistic Regression | Binary output 0/1 | `classification/` |
| Decision Boundary | Separate class 0 and class 1 | `Logistic_Regression_Decision_Boundary.py` |
| Logistic Regression Training | Sigmoid + cost + gradient descent | `Logistic_Regression_From_Scratch.py` |
| Neural Network Calculation | Given W, B and X | `NN/matmul/` |
| TensorFlow Neural Network | `Dense()` / `Sequential()` | `NN/tensorflow/` |
| Multiclass Classification | More than 2 classes | `Neural_Network_Multiclass_*` |
| Neural Network Regression | Continuous output | TensorFlow + linear output + MSE |

---

# 1. Linear Regression

Use linear regression when the target `y` is a **continuous numerical value**.

Examples:

- House price
- Car price
- Temperature
- Sales
- Revenue

## Model

For one feature:

```text
f(x) = wx + b
```

For multiple features:

```text
f(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

NumPy:

```python
prediction = np.dot(x, w) + b
```

---

## Cost Function

Linear regression normally uses Mean Squared Error.

In the from-scratch examples:

```text
J(w,b) = 1/(2m) Σ(f(x) - y)²
```

Code pattern:

```python
cost = cost + (f_wb_i - y[i])**2
cost = cost / (2 * m)
```

---

## Gradient Descent

General parameter updates:

```text
w = w - α(dJ/dw)

b = b - α(dJ/db)
```

Code:

```python
w = w - alpha * dj_dw
b = b - alpha * dj_db
```

Where:

```text
alpha = learning rate
```

---

# 2. Feature Scaling

When features have very different ranges, standardization can make gradient-based training easier.

Example:

```text
House Size = 2104
Bedrooms   = 5
Floors     = 1
Age        = 45
```

These features have very different scales.

Using scikit-learn:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_norm = scaler.fit_transform(X_train)
```

For TEST data:

```python
X_test_norm = scaler.transform(X_test)
```

### Important

Training:

```python
fit_transform()
```

Testing:

```python
transform()
```

Do **not** fit the scaler again on test data.

---

# 3. Logistic Regression

Use logistic regression for **binary classification**.

```text
y = 0
or
y = 1
```

The basic calculation is:

```text
z = w·x + b
```

Then apply the sigmoid function.

---

## Sigmoid

```text
              1
sigmoid(z) = ───────
             1 + e⁻ᶻ
```

Python:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Prediction:

```python
z = np.dot(x, w) + b
y_pred = sigmoid(z)
```

Classification:

```python
if y_pred >= 0.5:
    print("y = 1")
else:
    print("y = 0")
```

---

# 4. Logistic Regression Cost

Binary classification uses Binary Cross-Entropy / Log Loss.

```text
J = -1/m Σ[y log(f) + (1-y) log(1-f)]
```

Code pattern:

```python
cost += (
    -y[i] * np.log(f_wb_i)
    - (1 - y[i]) * np.log(1 - f_wb_i)
)
```

---

# 5. Decision Boundary

For logistic regression:

```text
z = w₁x₁ + w₂x₂ + b
```

The decision boundary occurs when:

```text
z = 0
```

Therefore:

```text
w₁x₁ + w₂x₂ + b = 0
```

Solve for `x₂`:

```text
x₂ = -(w₁x₁ + b) / w₂
```

This is useful when a question asks you to draw or calculate the classification boundary.

---

# 6. Neural Network Forward Propagation

The main calculation for each layer is:

```text
Z = XW + B
```

Then:

```text
A = activation(Z)
```

NumPy:

```python
Z = np.matmul(X, W) + B
A = sigmoid(Z)
```

A reusable function:

```python
def compute_layer_vector(X, W, B):
    Z = np.matmul(X, W) + B
    A = sigmoid(Z)
    return A
```

---

# 7. Matrix Dimensions

This is extremely important when working with neural networks.

If:

```text
X = (1 × 2)
W = (2 × 4)
```

then:

```text
XW = (1 × 4)
```

Example network:

```text
Input
(1 × 2)
   ↓
W1 (2 × 4)
   ↓
Layer 1
(1 × 4)
   ↓
W2 (4 × 2)
   ↓
Layer 2
(1 × 2)
   ↓
W3 (2 × 1)
   ↓
Output
(1 × 1)
```

General rule:

```text
(number of inputs × number of neurons)
```

For a Dense layer:

```text
W shape = previous layer units × current layer units
b shape = current layer units
```

---

# 8. TensorFlow Dense Layers

Basic TensorFlow model:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(...),
    Dense(...),
    Dense(...)
])
```

General pattern:

```python
model.compile(
    loss=...,
    optimizer=...
)

model.fit(
    X_train,
    y_train,
    epochs=...
)

prediction = model.predict(X_test)
```

---

# 9. Choosing Output Activation

## Regression

Continuous output:

```python
Dense(1, activation='linear')
```

Typical loss:

```python
loss='mse'
```

---

## Binary Classification

Two possible classes:

```text
0 or 1
```

Output:

```python
Dense(1, activation='sigmoid')
```

Loss:

```python
loss=tf.keras.losses.BinaryCrossentropy()
```

Prediction:

```python
if prediction >= 0.5:
    # class 1
else:
    # class 0
```

---

## Multiclass Classification

Example:

```text
Class 1
Class 2
Class 3
Class 4
Class 5
```

You need one output neuron per class.

For 5 classes:

```python
Dense(5, activation='softmax')
```

or use logits:

```python
Dense(5, activation='linear')
```

with:

```python
SparseCategoricalCrossentropy(
    from_logits=True
)
```

---

# 10. Softmax vs Logits

### Method A — Softmax Output

```python
model = Sequential([
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])
```

Loss:

```python
SparseCategoricalCrossentropy()
```

---

### Method B — Logits Output

```python
model = Sequential([
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(5, activation='linear')
])
```

Loss:

```python
SparseCategoricalCrossentropy(
    from_logits=True
)
```

### Do not mix these patterns accidentally.

---

# 11. Multiclass Prediction

Model output:

```python
y_predicted = model.predict(x_test)
```

Find the highest output:

```python
predicted_class = np.argmax(y_predicted)
```

If your original classes are:

```text
1, 2, 3, 4, 5
```

but TensorFlow training labels were converted to:

```text
0, 1, 2, 3, 4
```

then:

```python
predicted_class = np.argmax(y_predicted) + 1
```

---

# 12. Why Do We Use `y_train - 1`?

TensorFlow sparse categorical labels normally start from `0`.

If the original dataset contains:

```text
1, 2, 3, 4, 5
```

convert it to:

```text
0, 1, 2, 3, 4
```

using:

```python
y_train_adjusted = y_train - 1
```

Then convert predictions back:

```python
predicted_class = np.argmax(prediction) + 1
```

---

# 13. Common Activation Functions

### Linear

```python
activation='linear'
```

```text
a = z
```

Useful for regression outputs.

---

### Sigmoid

```python
activation='sigmoid'
```

```text
        1
a = ───────
     1 + e⁻ᶻ
```

Useful for binary classification.

---

### ReLU

```python
activation='relu'
```

```text
ReLU(z) = max(0, z)
```

Common for hidden layers.

---

### Softmax

```python
activation='softmax'
```

Converts output scores into a probability distribution across multiple classes.

Useful for multiclass classification.

---

# 14. Quick TensorFlow Selection Table

| Problem | Output Layer | Loss |
|---|---|---|
| Regression | `Dense(1, linear)` | MSE |
| Binary Classification | `Dense(1, sigmoid)` | Binary Cross-Entropy |
| Multiclass — Softmax | `Dense(K, softmax)` | Sparse Categorical Cross-Entropy |
| Multiclass — Logits | `Dense(K, linear)` | Sparse Categorical Cross-Entropy with `from_logits=True` |

Where:

```text
K = number of classes
```

---

# 15. NumPy vs TensorFlow

### NumPy

You manually calculate:

```python
Z = np.matmul(X, W) + B
A = sigmoid(Z)
```

Good for understanding:

- Forward propagation
- Matrix multiplication
- Weight dimensions
- Bias
- Activation functions

### TensorFlow

TensorFlow handles the layer calculations:

```python
layer = Dense(
    units=4,
    activation='sigmoid'
)

A = layer(X)
```

Good for:

- Building full neural networks
- Training
- Backpropagation
- Optimization
- Prediction

---

# 16. Useful NumPy Commands

### Convert to NumPy array

```python
x = np.array([...])
```

### Dot product

```python
np.dot(x, w)
```

### Matrix multiplication

```python
np.matmul(X, W)
```

or:

```python
X @ W
```

### Reshape into column

```python
x.reshape(-1, 1)
```

### Reshape into one test sample

```python
x.reshape(1, -1)
```

### Index of largest value

```python
np.argmax(prediction)
```

### Create zeros

```python
np.zeros((n,))
```

---

# 17. Exam Debugging Checklist

If your code does not work, check these first:

1. **Check array dimensions**

```python
print(X.shape)
print(W.shape)
print(y.shape)
```

2. **Check matrix multiplication**

For:

```python
X @ W
```

the inner dimensions must match.

```text
(1 × 2) @ (2 × 4) = (1 × 4) ✓

(1 × 2) @ (4 × 2) = ERROR ✗
```

3. **Check the number of output neurons**

5 classes:

```python
Dense(5, ...)
```

4. **Check the loss function**

Regression:

```python
mse
```

Binary:

```python
BinaryCrossentropy
```

Multiclass:

```python
SparseCategoricalCrossentropy
```

5. **Check logits**

If:

```python
Dense(K, activation='linear')
```

use:

```python
SparseCategoricalCrossentropy(
    from_logits=True
)
```

6. **Check test-data shape**

Instead of:

```python
x_test = np.array([2.5, 2.5])
```

TensorFlow usually expects:

```python
x_test = np.array([[2.5, 2.5]])
```

7. **Check feature scaling**

Do not do this on test data:

```python
scaler.fit_transform(X_test)
```

Use:

```python
scaler.transform(X_test)
```

---

# 18. Exam Quick Decision Tree

```text
What does the question ask?
│
├── Predict a NUMBER?
│      │
│      └── REGRESSION
│           ├── From scratch → regression/
│           └── TensorFlow → Linear output + MSE
│
├── Predict 0 or 1?
│      │
│      └── BINARY CLASSIFICATION
│           ├── Sigmoid
│           ├── Threshold = 0.5
│           └── Binary Cross-Entropy
│
├── Predict one of MANY classes?
│      │
│      └── MULTICLASS CLASSIFICATION
│           ├── K output neurons
│           ├── Softmax OR Logits
│           └── argmax()
│
└── Calculate a neural network manually?
       │
       └── FORWARD PROPAGATION
            ├── Z = XW + B
            ├── Apply activation
            └── Pass A to next layer
```

---

# 19. Key Formulas to Remember

### Linear Regression

```text
f(x) = w·x + b
```

### Linear Regression Cost

```text
J = 1/(2m) Σ(f(x) - y)²
```

### Gradient Descent

```text
w = w - α(dJ/dw)
b = b - α(dJ/db)
```

### Logistic Regression

```text
z = w·x + b
```

### Sigmoid

```text
sigmoid(z) = 1 / (1 + e⁻ᶻ)
```

### Binary Decision

```text
sigmoid(z) >= 0.5 → y = 1
sigmoid(z) <  0.5 → y = 0
```

### Neural Network Layer

```text
Z = XW + B
A = activation(Z)
```

### ReLU

```text
ReLU(z) = max(0, z)
```

### Decision Boundary

```text
w₁x₁ + w₂x₂ + b = 0
```

---

# 20. Before the Exam

Make sure you can quickly identify:

- Regression vs classification
- Binary vs multiclass classification
- `np.dot()` and `np.matmul()`
- Sigmoid calculation
- Cost calculation
- Gradient calculation
- Gradient-descent parameter updates
- Neural-network forward propagation
- Weight and bias dimensions
- TensorFlow `Sequential` and `Dense`
- MSE vs Binary Cross-Entropy vs Sparse Categorical Cross-Entropy
- Softmax vs logits
- `np.argmax()`
- Feature scaling with `StandardScaler`

Most importantly, understand the code before copying it. Small changes in the input dimensions, number of classes, activation function, or loss function can require changes to the model.

---

# ⚠️ Notes

These files are intended as **study and exam-reference material** for SC6611.

Some examples use small datasets and manually selected weights or decision boundaries to demonstrate specific machine-learning concepts. They are educational examples rather than production machine-learning implementations.

Results from neural-network and SGD examples can vary between runs because model parameters may be randomly initialized.

---

# Good Luck 🍀

Know the pattern:

```text
Data
 ↓
Model
 ↓
Prediction
 ↓
Cost / Loss
 ↓
Gradient / Backpropagation
 ↓
Update Parameters
 ↓
Repeat
```

Once you understand that flow, most of the examples in this repository are variations of the same core ideas.