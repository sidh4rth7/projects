# Neural Network Training with Numpy and Pandas

## Overview
This project implements a simple feedforward neural network using NumPy for numerical operations and Pandas for loading CSV data. The neural network is trained using gradient descent on a dataset to classify handwritten digits.

## Code Breakdown

### 1. **Data Loading and Preprocessing**
- The dataset is loaded using Pandas (`pd.read_csv`).
- It is then converted into a NumPy array for easier manipulation.
- The dataset is shuffled to ensure randomness during training.
- The data is split into training and testing sets:
  - `X_train` and `Y_train` (Training set)
  - `X_test` and `Y_test` (Test set)
- Features (`X_train`, `X_test`) are normalized by dividing pixel values by 255 to scale them between 0 and 1.

### 2. **Initializing Parameters**
The model consists of two layers:
- **Layer 1:** 784 input neurons (assuming 28x28 images) → 10 neurons (hidden layer)
- **Layer 2:** 10 hidden neurons → 10 output neurons (one for each digit 0-9)

Weights and biases are initialized as:
- `w1`: Random values with small variance (`10x784` matrix)
- `b1`: Zero matrix (`10x1`)
- `w2`: Random values with small variance (`10x10` matrix)
- `b2`: Zero matrix (`10x1`)

### 3. **Activation Functions**
- **ReLU (Rectified Linear Unit):** Used in the hidden layer to introduce non-linearity:
  
  \[ A1 = \max(Z1, 0) \]
  
- **Softmax:** Used in the output layer to convert logits into probabilities:
  
  \[ A2 = \frac{e^{Z2}}{\sum e^{Z2}} \]
  
### 4. **Forward Propagation**
- Computes the intermediate values and activations:
  
  \[ Z1 = w1 \cdot X + b1 \]
  \[ A1 = ReLU(Z1) \]
  \[ Z2 = w2 \cdot A1 + b2 \]
  \[ A2 = Softmax(Z2) \]

### 5. **One-Hot Encoding**
Since the labels are categorical (digits 0-9), they are converted into one-hot vectors. For example, label `3` becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.

### 6. **Backward Propagation**
Calculates the gradients of the loss function with respect to parameters using:

- **Error for the output layer:**
  
  \[ dZ2 = A2 - Y \]

- **Gradients for layer 2:**
  
  \[ dW2 = \frac{1}{m} dZ2 \cdot A1^T \]
  \[ db2 = \frac{1}{m} \sum dZ2 \]

- **Error for the hidden layer:**
  
  \[ dZ1 = w2^T \cdot dZ2 \cdot ReLU'(Z1) \]

- **Gradients for layer 1:**
  
  \[ dW1 = \frac{1}{m} dZ1 \cdot X^T \]
  \[ db1 = \frac{1}{m} \sum dZ1 \]

### 7. **Updating Parameters**
Parameters are updated using gradient descent:

\[ w = w - \alpha \cdot dW \]

\[ b = b - \alpha \cdot db \]

where \( \alpha \) is the learning rate (0.001 in this case).

### 8. **Prediction and Accuracy**
- The predicted class is the index of the highest probability in `A2`.
- Accuracy is computed by comparing predictions with actual labels.

### 9. **Training with Gradient Descent**
The network is trained for 500 iterations. Every 10 iterations, the accuracy is printed to track progress.

## Mathematical Concepts Used
1. **Matrix Multiplication** - Used to compute activations efficiently.
2. **Derivative of ReLU** - Helps in backpropagation.
3. **Softmax Gradient** - Converts logits into class probabilities.
4. **One-hot Encoding** - Converts categorical labels into vector form.
5. **Gradient Descent** - Optimizes weights iteratively to minimize error.

### Performance of this Mode;
For alpha 0.01 & Iteration 490, Accuracy: 0.7578
For alpha 0.01 & Iteration 490, Accuracy: 0.9048

## Usage
Run the script to train the model on the dataset. Ensure the dataset is correctly located in the specified path.
path to data : "/data/train.csv"


## Future Improvements
- Implement batch processing for better efficiency.
- Use a deep learning framework like TensorFlow or PyTorch for scalability.
- Add regularization techniques to prevent overfitting.


