import numpy as np
import data_utils as d
import matplotlib.pyplot as plt
import time

def initialize_params(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(hidden1_size, input_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((hidden1_size, 1))
    W2 = np.random.randn(hidden2_size, hidden1_size) * np.sqrt(2 / hidden1_size)
    b2 = np.zeros((hidden2_size, 1))
    W3 = np.random.randn(output_size, hidden2_size) * np.sqrt(1 / hidden2_size)
    b3 = np.zeros((output_size, 1))
    return W1, b1, W2, b2, W3, b3

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z = np.clip(z, -500, 500)
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_prop(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1,X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = softmax(Z3)
    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[1]
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss

def backward_prop(X, Y, cache, W2, W3):
    m = X.shape[1]
    Z1, A1, Z2, A2, Z3, A3 = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W2, b2, W3, b3

def train(X_train, Y_train, input_size, hidden1_size, hidden2_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2, W3, b3 = initialize_params(input_size, hidden1_size, hidden2_size, output_size)
    batch_size = 64
    num_batches = X_train.shape[1] // batch_size
    losses = []
    accuracies = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train[:, start:end]
            Y_batch = Y_train[:, start:end]
            A3, cache = forward_prop(X_batch, W1, b1, W2, b2, W3, b3)
            loss = cross_entropy_loss(Y_batch, A3)
            epoch_loss += loss
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(X_batch, Y_batch, cache, W2, W3)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)

        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)
        train_preds = predict(X_train, W1, b1, W2, b2, W3, b3)
        train_acc = accuracy(train_preds, np.argmax(Y_train, axis=0))
        accuracies.append(train_acc)
    return W1, b1, W2, b2, W3, b3, losses, accuracies

def predict(X, W1, b1, W2, b2, W3, b3):
    A3, _ = forward_prop(X, W1, b1, W2, b2, W3, b3)
    predictions = np.argmax(A3, axis=0)
    return predictions

def accuracy(predictions, labels):
    return np.mean(predictions == labels) * 100

input_size, hidden1_size, hidden2_size, output_size, epochs, learning_rate = 784,256,64,10,100,0.05
start_time = time.time()
W1, b1, W2, b2, W3, b3, losses, accuracies = train(d.x_train, d.y_train_oh, input_size, hidden1_size, hidden2_size, output_size, epochs, learning_rate)
end_time = time.time()
time = end_time - start_time
preds = predict(d.x_test, W1, b1, W2, b2, W3, b3)
acc = accuracy(preds, np.argmax(d.y_test_oh, axis=0))
print(f"Test Accuracy: {acc:.2f}%")
print("Time: ", time)

plt.plot(losses)
plt.xlabel("Epoch")
plt.xticks(np.arange(0, len(losses), 1))
plt.ylabel("Loss")
plt.title(f"Training Loss Over Time: Epoch: {epochs}, Learning rate: {learning_rate}")
plt.grid(True)
plt.show()

plt.plot(accuracies, label="Accuracy", color='green')
plt.xlabel("Epoch")
plt.xticks(np.arange(0, len(accuracies), 1))
plt.ylabel("Accuracy (%)")
plt.title(f"Training Accuracy Over Epochs: Epoch: {epochs}, Learning rate: {learning_rate}")
plt.grid(True)
plt.show()