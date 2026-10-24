import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

layers = [x_train.shape[1] * x_train.shape[2], 16, 16, 10]

rng = np.random.default_rng()

weights = []
biases = []

# Xavier uniform weight initialisation
for i in range(len(layers) - 1):
    n_in = layers[i]
    n_out = layers[i + 1]

    bound = np.sqrt(6 / (n_in + n_out))

    weight = rng.uniform(
        low=-bound,
        high=bound,
        size=(n_in, n_out)
    )

    weights.append(weight)

# Zero bias initialisation
for i in range(len(layers) - 1):
    bias = np.zeros(layers[i + 1])
    biases.append(bias)

# Save model parameters
np.savez(
    "model.npz",
    **{f"w{i}": w for i, w in enumerate(weights)},
    **{f"b{i}": b for i, b in enumerate(biases)}
)

print("New weights and biases generated and saved to model.npz")
print(f"Architecture: {layers}")

for i, w in enumerate(weights):
    print(f"weights[{i}]: {w.shape}")
