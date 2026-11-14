
import numpy as np
from tictactoe_mouse import visualize_result

# ACTIVATION FUNCTIONS

def ReLU(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# DERIVATIVES

def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# LAYER CLASS

class Layer:
    def __init__(self, weights, bias, activation, derivative):
        self.weights = np.array(weights, dtype=float)
        self.biases = np.array(bias, dtype=float)
        self.activation = activation
        self.derivative = derivative

    def forward(self, inputs):
        self.inputs = inputs
        self.z = inputs @ self.weights + self.biases

        return self.activation(self.z)

    def backward(self, grad_output, learning_rate, apply_derivative=True):

        # Gradient through activation
        if apply_derivative:
            grad_z = grad_output * self.derivative(self.z)
        else:
            grad_z = grad_output

        # Calculate gradients
        grad_weights = np.outer(self.inputs, grad_z)
        grad_bias = grad_z

        # Calculate gradient for previous layer
        grad_inputs = grad_z @ self.weights.T

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_bias

        return grad_inputs


# NETWORK CLASS

class Network:
    def __init__(self, layers):
        self.layers = layers

    def calculate(self, inputs):
        output = np.array(inputs, dtype=float)

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def train(self, inputs, expected, learning_rate=0.01):

        # Forward propagation
        prediction = self.calculate(inputs)
        expected = np.array(expected, dtype=float)

        # BCE + sigmoid gradient
        grad = (prediction - expected) / prediction.size

        # Backpropagation
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            grad = layer.backward(
                grad,
                learning_rate,
                apply_derivative=(i != len(self.layers) - 1)
            )

        # Calculate BCE loss
        epsilon = 1e-7
        safe_prediction = np.clip(prediction, epsilon, 1 - epsilon)

        loss = -np.mean(
            expected * np.log(safe_prediction)
            + (1 - expected) * np.log(1 - safe_prediction)
        )

        return loss

winning_positions = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6)
]


# -------------------------
# TRAINING DATA
# -------------------------
from synthetic_data import generate_training_data, prepare_training_data

training_data = generate_training_data(50000000, seed=412313213123)
inputs, expected_outputs = prepare_training_data(training_data)


weights1 = np.random.randn(9, 20) * np.sqrt(2 / 9)
biases1 = np.zeros(20)

weights2 = np.random.randn(16, 20) * np.sqrt(2 / 20)
biases2 = np.zeros(20)

weights3 = np.random.randn(20, 9) * np.sqrt(1 / 20)
biases3 = np.zeros(9)


# Create layers

layers = [
    Layer(weights1, biases1, ReLU, relu_derivative),
    Layer(weights2, biases2, ReLU, relu_derivative),
    Layer(weights3, biases3, sigmoid, sigmoid_derivative)
]

# Create network
network = Network(layers)

# Create network
network = Network(layers)


# Convert winning position into 9 output values
formatted_data = []

for board, move in training_data:
    expected = np.zeros(9)
    expected[move] = 1

    formatted_data.append((
        np.array(board, dtype=float),
        expected
    ))


# -------------------------
# TRAIN
# -------------------------

epochs = 1000
learning_rate = 0.02

for epoch in range(epochs):
    total_loss = 0

    for board, expected in formatted_data:
        loss = network.train(board, expected, learning_rate)
        total_loss += loss

    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Loss:", total_loss / len(formatted_data))


# -------------------------
# TEST
# -------------------------

test_board = np.array([0, 1, -1, 0, 1, 0, 0, 0, -1])

output = network.calculate(test_board)

# Ignore occupied positions
output[test_board != 0] = -1

print("Output:", output)
print("Best move:", np.argmax(output))

output = network.calculate(test_board)
visualize_result(test_board, output)