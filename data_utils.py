import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0

def one_hot(y):
    one_hot_matrix = np.zeros((y.size, 10))
    one_hot_matrix[np.arange(y.size), y] = 1
    return one_hot_matrix.T

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)
