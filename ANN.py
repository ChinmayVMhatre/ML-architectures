import numpy as np

np.random.seed(42)

# ── Tiny dataset ──────────────────────────────────────────
# XOR problem: 4 samples, 2 features
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)  # (4, 2)
y = np.array([[0],[1],[1],[0]],         dtype=float)  # (4, 1)

# ── Initialize weights ────────────────────────────────────
# Layer 1: 2 inputs → 4 hidden neurons
W1 = np.random.randn(2, 4) * 0.1   # (2, 4)
b1 = np.zeros((1, 4))               # (1, 4)

# Layer 2: 4 hidden → 1 output
W2 = np.random.randn(4, 1) * 0.1   # (4, 1)
b2 = np.zeros((1, 1))               # (1, 1)

# ── Activation ────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)              # derivative of sigmoid

# ── Forward Pass ──────────────────────────────────────────
def forward(X):
    z1 = X  @ W1 + b1              # (4,2)@(2,4) = (4,4)
    a1 = sigmoid(z1)               # (4,4)  hidden activations

    z2 = a1 @ W2 + b2              # (4,4)@(4,1) = (4,1)
    a2 = sigmoid(z2)               # (4,1)  output

    cache = (X, z1, a1, z2, a2)   # save for backward
    return a2, cache

# ── Loss: Mean Squared Error ──────────────────────────────
def loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# ── Backward Pass (manual chain rule) ─────────────────────
def backward(cache, y_true, lr=0.1):
    global W1, b1, W2, b2
    X, z1, a1, z2, a2 = cache
    N = X.shape[0]                 # batch size = 4

    # ── Output layer ──────────────────────────────────────
    # dL/da2
    dL_da2 = 2 * (a2 - y_true) / N         # (4,1)

    # da2/dz2  (sigmoid derivative)
    da2_dz2 = sigmoid_grad(z2)             # (4,1)

    # chain rule: dL/dz2 = dL/da2 * da2/dz2
    dL_dz2 = dL_da2 * da2_dz2             # (4,1)  elementwise

    # dL/dW2 = a1.T @ dL/dz2
    dL_dW2 = a1.T @ dL_dz2                # (4,4).T @ (4,1) = (4,1)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

    # ── Hidden layer ──────────────────────────────────────
    # dL/da1 = dL/dz2 @ W2.T
    dL_da1 = dL_dz2 @ W2.T                # (4,1) @ (1,4) = (4,4)

    # da1/dz1
    da1_dz1 = sigmoid_grad(z1)            # (4,4)

    dL_dz1 = dL_da1 * da1_dz1            # (4,4)  elementwise

    dL_dW1 = X.T @ dL_dz1                # (2,4)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    # ── Gradient descent step ─────────────────────────────
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

# ── Training loop ─────────────────────────────────────────
for epoch in range(5000):
    y_pred, cache = forward(X)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss(y_pred, y):.6f}")
    backward(cache, y)

# ── Final predictions ─────────────────────────────────────
y_pred, _ = forward(X)
print("\nFinal predictions:")
print(np.round(y_pred, 3))
