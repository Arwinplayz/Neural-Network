import numpy as np
from tensorflow.keras.datasets import mnist

# Download and load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise pixel values to [0, 1]
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

# Flatten each 28x28 image into 784 inputs
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

i = 0

print("Label:", y_train[i])
print("Input shape:", x_train[i].shape)
print("Pixel values:", x_train[i])

print("Flattened training images:", x_train.shape)
