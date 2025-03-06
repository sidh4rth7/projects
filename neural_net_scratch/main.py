import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data using pandas
data = pd.read_csv('data/train.csv')
print(data.head())

# Convert data to numpy array and shuffle it
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Transpose and normalize data
data_test = data[:1000].T
Y_test = data_test[0]
X_test = data_test[1:n] / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255
_, m_train = X_train.shape

# Initialize parameters
def init_params():
    w1 = np.random.randn(10, 784) * np.sqrt(1. / 784)
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * np.sqrt(1. / 10)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

# Activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

# Forward propagation
def forward_propagation(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# Derivative of ReLU
def der_ReLU(Z):
    return Z > 0

# Backward propagation
def backward_propagation(Z1, A1, Z2, A2, w1, w2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = w2.T.dot(dZ2) * der_ReLU(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 -= alpha * dW1
    b1 -= alpha * db1
    w2 -= alpha * dW2
    b2 -= alpha * db2
    return w1, b1, w2, b2

# Get predictions
def get_predictions(A2):
    return np.argmax(A2, axis=0)

# Calculate accuracy
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent training
def gradient_descent(X, Y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print(f'Iteration {i}, Accuracy: {get_accuracy(predictions, Y):.4f}')
    
    return w1, b1, w2, b2

# Train the model
w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)
