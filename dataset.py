import numpy as np

def make_circles(num_samples=400, noise=0.05):
    """
    Generates concentric circles (an inner disc and an outer ring).
    Target: 0 for inner disc, 1 for outer ring.
    """
    half = num_samples // 2

    # Inner disc (Class 0)
    r_inner = np.random.uniform(0.0, 0.35, half)
    theta_inner = np.random.uniform(0.0, 2 * np.pi, half)
    x_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    y_inner = np.zeros((half, 1))

    # Outer ring (Class 1)
    r_outer = np.random.uniform(0.65, 1.0, half)
    theta_outer = np.random.uniform(0.0, 2 * np.pi, half)
    x_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    y_outer = np.ones((half, 1))

    # Combine and add noise
    X = np.vstack([x_inner, x_outer]) + np.random.randn(num_samples, 2) * noise
    Y = np.vstack([y_inner, y_outer])

    indices = np.random.permutation(num_samples)
    return X[indices].astype(np.float32), Y[indices].astype(np.float32)


def make_moons(num_samples=400, noise=0.08):
    """
    Generates two interlocking half-moon shapes.
    Target: 0 for upper moon, 1 for lower moon.
    """
    half = num_samples // 2

    # Upper moon (Class 0)
    theta_upper = np.linspace(0, np.pi, half)
    x_upper = np.column_stack([np.cos(theta_upper), np.sin(theta_upper)])
    y_upper = np.zeros((half, 1))

    # Lower moon (Class 1)
    theta_lower = np.linspace(0, np.pi, half)
    x_lower = np.column_stack([1.0 - np.cos(theta_lower), 0.5 - np.sin(theta_lower)])
    y_lower = np.ones((half, 1))

    # Combine and add noise
    X = np.vstack([x_upper, x_lower]) + np.random.randn(num_samples, 2) * noise
    Y = np.vstack([y_upper, y_lower])

    indices = np.random.permutation(num_samples)
    return X[indices].astype(np.float32), Y[indices].astype(np.float32)


def make_spirals(num_samples=400, noise=0.1):
    """
    Generates two interlocking spirals (a classic benchmark for neural networks).
    Target: 0 for spiral A, 1 for spiral B.
    """
    half = num_samples // 2

    # Spiral A (Class 0)
    r_a = np.linspace(0.1, 1.0, half)
    theta_a = np.linspace(0, 3 * np.pi, half) + np.random.randn(half) * noise
    x_a = np.column_stack([r_a * np.sin(theta_a), r_a * np.cos(theta_a)])
    y_a = np.zeros((half, 1))

    # Spiral B (Class 1)
    r_b = np.linspace(0.1, 1.0, half)
    theta_b = np.linspace(0, 3 * np.pi, half) + np.pi + np.random.randn(half) * noise
    x_b = np.column_stack([r_b * np.sin(theta_b), r_b * np.cos(theta_b)])
    y_b = np.ones((half, 1))

    # Combine
    X = np.vstack([x_a, x_b])
    Y = np.vstack([y_a, y_b])

    indices = np.random.permutation(num_samples)
    return X[indices].astype(np.float32), Y[indices].astype(np.float32)


if __name__ == "__main__":
    X_circles, Y_circles = make_circles(400)
    X_moons, Y_moons = make_moons(400)
    X_spirals, Y_spirals = make_spirals(400)

    print("Generated datasets successfully:")
    print(f"  Circles: X shape {X_circles.shape}, Y shape {Y_circles.shape}")
    print(f"  Moons:   X shape {X_moons.shape},   Y shape {Y_moons.shape}")
    print(f"  Spirals: X shape {X_spirals.shape}, Y shape {Y_spirals.shape}")
