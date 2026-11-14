import numpy as np

# Board values:
# 1  = X (the AI)
# -1 = O (the human)
# 0  = empty

X = 1
O = -1
EMPTY = 0

winning_positions = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]

corners = [0, 2, 6, 8]
edges = [1, 3, 5, 7]


def get_empty_cells(board):
    """Return a list of indexes that are still empty."""
    return [index for index, value in enumerate(board) if value == EMPTY]


def check_winner(board):
    """
    Return:
        1 if X has won
        -1 if O has won
        0 if nobody has won yet
    """
    for a, b, c in winning_positions:
        line_sum = board[a] + board[b] + board[c]

        if line_sum == 3:
            return X

        if line_sum == -3:
            return O

    return 0


def is_board_full(board):
    """Return True if there are no empty squares left."""
    return len(get_empty_cells(board)) == 0


def find_winning_move(board, player):
    """
    Find a square that lets `player` win immediately.

    A winning move exists when one line contains:
    - two marks belonging to `player`
    - one empty square

    Returns the empty square index, or None.
    """
    for a, b, c in winning_positions:
        line = [board[a], board[b], board[c]]

        if line.count(player) == 2 and line.count(EMPTY) == 1:
            return [a, b, c][line.index(EMPTY)]

    return None


def teacher_move(board):
    """
    Choose the teacher's recommended move for X.

    Priority:
    1. Win immediately.
    2. Block O from winning on its next move.
    3. Take the center.
    4. Take an available corner.
    5. Take an available edge.
    """
    empty_cells = get_empty_cells(board)

    # 1. Win immediately if possible.
    winning_move = find_winning_move(board, X)
    if winning_move is not None:
        return winning_move

    # 2. Block O if O could win next turn.
    blocking_move = find_winning_move(board, O)
    if blocking_move is not None:
        return blocking_move

    # 3. Take the center.
    if 4 in empty_cells:
        return 4

    # 4. Take an available corner.
    available_corners = [cell for cell in corners if cell in empty_cells]
    if available_corners:
        return available_corners[0]

    # 5. Take an available edge.
    available_edges = [cell for cell in edges if cell in empty_cells]
    if available_edges:
        return available_edges[0]

    # This should not happen for a non-terminal board,
    # but return the first empty cell as a safe fallback.
    return empty_cells[0]


def simulate_one_game(rng):
    """
    Play one random game where O moves first.

    Save every board position immediately after O moves,
    provided that:
    - the game has not ended
    - it is now X's turn

    Returns a list of (board, correct_move) examples.
    """
    board = [EMPTY] * 9
    examples = []

    current_player = O

    while True:
        empty_cells = get_empty_cells(board)

        # No empty cells means the game ended in a draw.
        if not empty_cells:
            break

        # Choose one random legal empty square.
        move = rng.choice(empty_cells)
        move = int(move)

        board[move] = current_player

        # Stop immediately if this move won the game.
        winner = check_winner(board)
        if winner != 0:
            break

        # Stop if the board is now full.
        if is_board_full(board):
            break

        # Switch to the other player.
        current_player = X if current_player == O else O

        # Save the position only when X is about to move.
        if current_player == X:
            correct_move = teacher_move(board)
            examples.append((board.copy(), correct_move))

    return examples


def generate_training_data(amount=10000, seed=None):
    """
    Generate training examples.

    Each example is a tuple:
        (board, correct_move)

    board is a list of nine numbers:
        1 = X
        -1 = O
        0 = empty

    correct_move is an integer from 0 to 8.
    """
    rng = np.random.default_rng(seed)

    training_data = []

    # Keep generating random games until enough examples exist.
    while len(training_data) < amount:
        training_data.extend(simulate_one_game(rng))

    # One game may create several useful X-turn positions,
    # so trim the final list to the requested amount.
    return training_data[:amount]


def prepare_training_data(training_data):
    """
    Convert (board, move) examples into NumPy arrays.

    Returns:
        inputs: NumPy array with shape (number_of_examples, 9)
        expected_outputs: NumPy array with shape (number_of_examples, 9)

    Each expected output is one-hot:
        the correct move is 1
        all other moves are 0
    """
    inputs = []
    expected_outputs = []

    for board, move in training_data:
        inputs.append(board)

        expected = np.zeros(9)
        expected[move] = 1

        expected_outputs.append(expected)

    inputs = np.array(inputs, dtype=float)
    expected_outputs = np.array(expected_outputs, dtype=float)

    return inputs, expected_outputs


# Optional: run this file directly to test it.
if __name__ == "__main__":
    examples = generate_training_data(10, seed=42)

    print("Generated examples:", len(examples))

    for board, move in examples:
        print("Board:", board)
        print("Teacher move:", move)
        print()
