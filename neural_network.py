import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0


layers = [784, 16, 16, 10]
eta = 0.01
epochs = 10


with np.load("model.npz") as data:
    weights = [data[f"w{i}"].copy()
               for i in range(len(layers) - 1)]
    biases = [data[f"b{i}"].copy()
              for i in range(len(layers) - 1)]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def forward(inputs):
    activations = [inputs]
    zs = []

    for i in range(len(weights)):
        z = activations[-1] @ weights[i] + biases[i]
        zs.append(z)
        activations.append(sigmoid(z))

    return activations, zs


def train_one(inputs, target):
    global weights, biases

   
    activations, zs = forward(inputs)
    output = activations[-1]

    cost = 0.5 * np.sum((output - target) ** 2)

    
    deltas = [None] * len(weights)

    
    deltas[-1] = (
        (output - target)
        * output
        * (1 - output)
    )

    
    for l in range(len(weights) - 2, -1, -1):
        next_delta = deltas[l + 1]
        next_weights = weights[l + 1]
        activation = activations[l + 1]

        deltas[l] = (
            next_delta @ next_weights.T
        ) * activation * (1 - activation)

    
    for l in range(len(weights)):
        weight_gradient = np.outer(
            activations[l], deltas[l]
        )
        bias_gradient = deltas[l]

        weights[l] -= eta * weight_gradient
        biases[l] -= eta * bias_gradient

    return cost



rng = np.random.default_rng()
targets = np.eye(10, dtype=np.float32)

for epoch in range(epochs):
    order = rng.permutation(len(x_train))
    total_cost = 0.0

    for index in order:
        inputs = x_train[index]
        target = targets[y_train[index]]

        total_cost += train_one(inputs, target)

    average_cost = total_cost / len(x_train)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"average cost = {average_cost:.6f}"
    )

    np.savez(
        "model.npz",
        **{f"w{i}": w for i, w in enumerate(weights)},
        **{f"b{i}": b for i, b in enumerate(biases)}
    )

correct = 0

for index in range(len(x_test)):
    activations, _ = forward(x_test[index])
    prediction = np.argmax(activations[-1])

    if prediction == y_test[index]:
        correct += 1

accuracy = correct / len(x_test) * 100

print(f"Test accuracy: {accuracy:.2f}%")
print("Model saved to model.npz")
