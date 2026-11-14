import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Neural network
# =========================================================

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


class Layer:
    def __init__(self, weights, biases, activation, derivative):
        self.weights = np.array(weights, dtype=float)
        self.biases = np.array(biases, dtype=float)
        self.activation = activation
        self.derivative = derivative

    def forward(self, inputs):
        self.inputs = inputs
        self.z = inputs @ self.weights + self.biases
        return self.activation(self.z)

    def backward(self, grad_output, learning_rate, apply_derivative=True):
        if apply_derivative:
            grad_z = grad_output * self.derivative(self.z)
        else:
            grad_z = grad_output

        grad_weights = np.outer(self.inputs, grad_z)
        grad_biases = grad_z.copy()
        grad_inputs = grad_z @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_inputs


class Network:
    def __init__(self, layers):
        self.layers = layers

    def calculate(self, inputs):
        output = np.array(inputs, dtype=float)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, inputs, expected, learning_rate=0.01):
        prediction = self.calculate(inputs)
        expected = np.array(expected, dtype=float)

        # For sigmoid output with binary cross-entropy.
        grad = prediction - expected

        for i in reversed(range(len(self.layers))):
            grad = self.layers[i].backward(
                grad,
                learning_rate,
                apply_derivative=(i != len(self.layers) - 1)
            )

        epsilon = 1e-7
        safe_prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.mean(
            expected * np.log(safe_prediction)
            + (1 - expected) * np.log(1 - safe_prediction)
        )
        return loss


# =========================================================
# 2. Training examples
# Each item is (board, correct move for X).
# 1 = X, -1 = O, 0 = empty.
# =========================================================

training_data = [
    ([1, 1, 0, 0, -1, 0, 0, -1, 0], 2),
    ([0, 1, 1, -1, 0, 0, -1, 0, 0], 0),
    ([0, 0, -1, 1, 0, -1, 0, 1, 0], 4),
    ([0, 0, -1, 1, 1, 0, -1, 0, 0], 5),
    ([0, 1, 0, -1, 0, -1, 0, 0, 1], 7),
    ([1, 0, 0, 0, 1, -1, -1, 0, 0], 8),
    ([1, 0, -1, 0, 0, 1, -1, 0, 0], 4),
    ([0, -1, 0, 0, 1, 1, 0, -1, 0], 3),
    ([0, 0, 0, 1, 0, -1, 1, -1, 0], 6),
    ([0, 1, -1, 0, 1, 0, 0, -1, 0], 7)
]


def prepare_training_data(data):
    formatted = []
    for board, move in data:
        expected = np.zeros(9)
        expected[move] = 1
        formatted.append((np.array(board, dtype=float), expected))
    return formatted


# =========================================================
# 3. Build and train the network
# =========================================================

np.random.seed(42)

weights1 = np.random.randn(9, 16) * np.sqrt(2 / 9)
biases1 = np.zeros(16)

weights2 = np.random.randn(16, 16) * np.sqrt(2 / 16)
biases2 = np.zeros(16)

weights3 = np.random.randn(16, 9) * np.sqrt(1 / 16)
biases3 = np.zeros(9)

network = Network([
    Layer(weights1, biases1, relu, relu_derivative),
    Layer(weights2, biases2, relu, relu_derivative),
    Layer(weights3, biases3, sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x)))
])

formatted_data = prepare_training_data(training_data)

print("Training neural network...")
for epoch in range(10000):
    total_loss = 0
    for board, expected in formatted_data:
        total_loss += network.train(board, expected, learning_rate=0.01)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {total_loss / len(formatted_data):.5f}")

print("Training finished.")
print("A window will open. Click an empty square to place O.")
print("Close the window to quit.\n")


# =========================================================
# 4. Game logic
# =========================================================

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


def has_won(board, player):
    for a, b, c in winning_positions:
        if board[a] == player and board[b] == player and board[c] == player:
            return True
    return False


# =========================================================
# 5. Mouse-controlled game window
# =========================================================

board = np.zeros(9, dtype=float)
game_over = False
status = "Your turn: click a square to place O."


fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.manager.set_window_title("Tic-Tac-Toe Neural Network")


def draw_board():
    ax.clear()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(status, fontsize=12, pad=18)

    # Draw grid.
    for line in [1, 2]:
        ax.plot([line, line], [0, 3], linewidth=2, color="black")
        ax.plot([0, 3], [line, line], linewidth=2, color="black")

    # Draw X, O, and position numbers.
    for i, value in enumerate(board):
        row, col = divmod(i, 3)
        x = col + 0.5
        y = 2.5 - row

        if value == 1:
            ax.text(x, y, "X", ha="center", va="center",
                    fontsize=42, fontweight="bold", color="tab:blue")
        elif value == -1:
            ax.text(x, y, "O", ha="center", va="center",
                    fontsize=42, fontweight="bold", color="tab:red")
        else:
            ax.text(x, y, str(i), ha="center", va="center",
                    fontsize=18, color="gray", alpha=0.6)

    fig.canvas.draw_idle()


def on_click(event):
    global game_over, status

    if game_over or event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    # Convert mouse coordinates to board index.
    col = int(event.xdata)
    row = 2 - int(event.ydata)
    if not (0 <= row < 3 and 0 <= col < 3):
        return

    human_move = row * 3 + col

    # Ignore clicks on occupied squares.
    if board[human_move] != 0:
        status = "That square is occupied. Choose an empty square."
        draw_board()
        return

    # Human places O.
    board[human_move] = -1

    if has_won(board, -1):
        status = "You win! Close the window to quit."
        game_over = True
        draw_board()
        return

    if np.all(board != 0):
        status = "Draw! Close the window to quit."
        game_over = True
        draw_board()
        return

    # Neural network chooses X's move.
    raw_output = network.calculate(board.copy())
    available_scores = raw_output.copy()
    available_scores[board != 0] = -np.inf
    ai_move = int(np.argmax(available_scores))

    board[ai_move] = 1

    if has_won(board, 1):
        status = f"AI chose square {ai_move}. AI wins! Close the window to quit."
        game_over = True
    elif np.all(board != 0):
        status = "Draw! Close the window to quit."
        game_over = True
    else:
        status = f"AI chose square {ai_move}. Your turn: click an empty square."

    draw_board()


fig.canvas.mpl_connect("button_press_event", on_click)
draw_board()
plt.show()
