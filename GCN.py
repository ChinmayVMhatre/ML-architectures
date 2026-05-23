"""
Graph Convolutional Network (GCN) — From Scratch
=================================================
Pure PyTorch. No torch_geometric. No chemistry.

Task:
    Node classification on a school social network.
    Given friendships between 12 students, predict
    which of two clubs (Drama vs Science) each student belongs to.

Why this maps to GCN well:
    - Students in the same club tend to be friends (homophily)
    - GCN smooths features over the neighbourhood
    - After 2 layers, each student "knows" what their friends' friends like
    - That's enough to infer club membership

GCN formula (Kipf & Welling, 2017):
    H^(l+1) = σ( D̂^(-1/2)  Â  D̂^(-1/2)  H^(l)  W^(l) )

    Where:
        Â   = A + I         (adjacency with self-loops added)
        D̂   = degree matrix of Â
        W^(l) = learnable weight matrix for layer l
        σ   = activation function (ReLU for hidden, Softmax for output)

Key difference from MPNN:
    - No learned message function    → W is shared across ALL edges
    - No edge features               → only node features matter
    - No receiver-awareness          → message from j is same regardless of who i is
    - Normalization is FIXED         → 1/sqrt(d_i * d_j), not learned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY — Build the normalised adjacency matrix
# ══════════════════════════════════════════════════════════════════════════════

def normalise_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Computes the symmetrically normalised adjacency matrix used in GCN.

    Steps:
        1. Add self-loops:  Â = A + I
           (so each node aggregates its OWN features too)
        2. Compute degree:  D̂_ii = sum of row i in Â
        3. Normalise:       Â_norm = D̂^(-1/2)  Â  D̂^(-1/2)

    The normalisation prevents high-degree nodes from dominating.
    A node with 10 friends doesn't shout louder than one with 2.

    Args:
        A (Tensor): Raw adjacency matrix, shape (N, N). Binary, undirected.

    Returns:
        Tensor: Normalised adjacency, shape (N, N).

    Example:
        >>> A = torch.tensor([[0,1,1],[1,0,0],[1,0,0]], dtype=torch.float)
        >>> A_norm = normalise_adjacency(A)
        >>> A_norm.shape
        torch.Size([3, 3])
    """
    N   = A.size(0)
    A_hat = A + torch.eye(N)                          # Â = A + I   (N, N)

    # Row sums = degree of each node in Â
    deg   = A_hat.sum(dim=1)                          # (N,)

    # D̂^(-1/2): inverse square root of degree (diagonal matrix as vector)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))            # (N, N)

    # Symmetric normalisation: D̂^(-1/2) Â D̂^(-1/2)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt            # (N, N)


# ══════════════════════════════════════════════════════════════════════════════
# 1. GCN LAYER
# ══════════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """
    A single Graph Convolutional layer.

    Computes:
        H_out = σ( A_norm  H_in  W )

    Where:
        A_norm  = pre-computed normalised adjacency  (N, N)
        H_in    = input node feature matrix          (N, in_dim)
        W       = learnable weight matrix            (in_dim, out_dim)
        σ       = activation (ReLU for hidden layers, identity for last)

    Notice what's MISSING vs MPNN:
        - No message function (W is shared; no per-edge customisation)
        - No edge features    (A_norm carries only topology, not distances)
        - No receiver h_i in the message computation

    Args:
        in_dim  (int): Input feature dimensionality.
        out_dim (int): Output feature dimensionality.

    Example:
        >>> layer  = GCNLayer(in_dim=4, out_dim=8)
        >>> A_norm = normalise_adjacency(torch.eye(5))
        >>> H      = torch.randn(5, 4)
        >>> H_out  = layer(H, A_norm)
        >>> H_out.shape
        torch.Size([5, 8])
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # The ONLY learnable parameter: one weight matrix
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # Initialise with Glorot (Xavier) uniform — standard for GCNs
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H      (Tensor): Node features,           shape (N, in_dim).
            A_norm (Tensor): Normalised adjacency,    shape (N, N).

        Returns:
            Tensor: Updated node features, shape (N, out_dim).

        Computation:
            Step 1 — Linear transform:  H' = H W     (N, out_dim)
            Step 2 — Graph smoothing:   H_out = A_norm H'
                     Each node's new feature = weighted average of its
                     neighbours' transformed features.
                     This is the "message passing" — but fixed, not learned.
        """
        H_transformed = self.W(H)                 # (N, out_dim)  — transform first
        return A_norm @ H_transformed              # (N, out_dim)  — then smooth


# ══════════════════════════════════════════════════════════════════════════════
# 2. FULL GCN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GCN(nn.Module):
    """
    Two-layer GCN for node classification.

    Architecture:
        Input features  (N, in_dim)
            ↓  GCNLayer + ReLU      [learns local neighbourhood patterns]
            ↓  Dropout
        Hidden features (N, hidden_dim)
            ↓  GCNLayer             [propagates patterns one hop further]
        Logits          (N, num_classes)
            ↓  log_softmax
        Log-probs       (N, num_classes)

    After 2 layers, each node has effectively aggregated information
    from its 2-hop neighbourhood — i.e. friends-of-friends.

    Args:
        in_dim      (int):   Input feature size.
        hidden_dim  (int):   Hidden layer size.
        num_classes (int):   Number of output classes.
        dropout     (float): Dropout probability between layers.

    Example:
        >>> model = GCN(in_dim=3, hidden_dim=16, num_classes=2)
        >>> A_norm = normalise_adjacency(torch.zeros(12, 12))
        >>> H      = torch.randn(12, 3)
        >>> out    = model(H, A_norm)
        >>> out.shape
        torch.Size([12, 2])
    """

    def __init__(
        self,
        in_dim:      int,
        hidden_dim:  int  = 16,
        num_classes: int  = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.layer1  = GCNLayer(in_dim,     hidden_dim)
        self.layer2  = GCNLayer(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H      (Tensor): Raw node features,     shape (N, in_dim).
            A_norm (Tensor): Normalised adjacency,  shape (N, N).

        Returns:
            Log-softmax probabilities, shape (N, num_classes).
            Use F.nll_loss(output, labels) as the loss function.
        """
        # ── Layer 1: aggregate 1-hop neighbourhood ─────────────────────
        H = self.layer1(H, A_norm)        # (N, hidden_dim)
        H = F.relu(H)
        H = self.dropout(H)

        # ── Layer 2: aggregate 2-hop neighbourhood ─────────────────────
        H = self.layer2(H, A_norm)        # (N, num_classes)

        return F.log_softmax(H, dim=1)   # log-probs per class


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOY SOCIAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def build_school_graph():
    """
    Builds a toy school social network with 12 students.

    Two clubs exist:
        Club 0 — Drama Club  (students 0–5)
        Club 1 — Science Club (students 6–11)

    Friendship pattern:
        Students mostly befriend others in the same club (homophily),
        with one cross-club bridge friendship (student 4 ↔ student 7).
        This makes the task non-trivial: student 4 and 7 have mixed signals.

    Node features (3 features per student):
        [hours_reading,  hours_acting,  hours_experimenting]
        Drama students  → high acting,   low  experimenting
        Science students → low  acting,  high experimenting
        Small Gaussian noise added to make it realistic.

    Graph layout (friendships):

        Drama Club          Science Club
        0 — 1               6 — 7
        |   |               |   |
        2 — 3           4*— 8 — 9
            |           (bridge)
            4* ——————— 7*
                        |
                       10—11

    Returns:
        A      (Tensor): Adjacency matrix,   shape (12, 12)
        A_norm (Tensor): Normalised adj,     shape (12, 12)
        H      (Tensor): Node features,      shape (12, 3)
        labels (Tensor): Club membership,   shape (12,)  — 0=Drama, 1=Science
    """
    torch.manual_seed(7)

    N = 12

    # ── Adjacency matrix (undirected) ──────────────────────────────────
    edges = [
        # Drama Club (0–5) — internal friendships
        (0, 1), (0, 2), (1, 3), (2, 3), (3, 5), (0, 5),
        # Science Club (6–11) — internal friendships
        (6, 7), (6, 8), (7, 9), (8, 9), (9, 11), (6, 11),
        # Cross-club bridge: student 4 (Drama) ↔ student 10 (Science)
        (4, 10),
        # Students 4 and 10 also connect within their clubs
        (3, 4), (1, 4),
        (10, 8), (10, 11),
    ]

    A = torch.zeros(N, N)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0          # undirected → symmetric

    A_norm = normalise_adjacency(A)

    # ── Node features ─────────────────────────────────────────────────
    # [hours_reading, hours_acting, hours_experimenting]
    drama_base   = torch.tensor([3.0, 8.0, 1.0])   # Drama students
    science_base = torch.tensor([4.0, 1.0, 9.0])   # Science students

    features = []
    for i in range(N):
        base  = drama_base if i < 6 else science_base
        noise = torch.randn(3) * 0.8                # realistic variation
        features.append((base + noise).clamp(min=0))

    H      = torch.stack(features)                 # (12, 3)
    labels = torch.tensor([0]*6 + [1]*6)           # 0=Drama, 1=Science

    return A, A_norm, H, labels


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def accuracy(log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes classification accuracy.

    Args:
        log_probs (Tensor): Model output, shape (N, num_classes).
        labels    (Tensor): True class indices, shape (N,).

    Returns:
        float: Fraction of correctly classified nodes.
    """
    preds = log_probs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_example():
    """
    Full training demo: GCN on the school social network.

    Trains on ALL nodes (transductive setting — standard for GCN).
    Prints training progress, final predictions per student, and
    final gradients at end of last epoch.

    Why transductive?
        GCN needs the full graph structure (A_norm) at train time.
        You can't run inference on unseen nodes without re-computing A_norm.
        This is a known limitation vs. inductive methods like GraphSAGE.
    """
    torch.manual_seed(42)

    # ── Build graph ───────────────────────────────────────────────────
    A, A_norm, H, labels = build_school_graph()

    print("=" * 60)
    print("  School Social Network")
    print("=" * 60)
    print(f"  Students   : 12  (0–5 Drama Club, 6–11 Science Club)")
    print(f"  Friendships: {int(A.sum().item() // 2)}  undirected edges")
    print(f"  Features   : 3   (reading hrs, acting hrs, experiment hrs)")
    print(f"  Task       : Predict club membership  (0=Drama, 1=Science)")
    print(f"  Bridge     : Student 4 (Drama) ↔ Student 10 (Science)")
    print()

    # ── Model ─────────────────────────────────────────────────────────
    model = GCN(in_dim=3, hidden_dim=16, num_classes=2, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params}  (just 2 weight matrices — that's it)")
    print()

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # weight_decay = L2 regularisation — important for small graphs

    # ── Training loop ─────────────────────────────────────────────────
    print("=" * 60)
    print("  Training")
    print("=" * 60)
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Accuracy':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()

        out  = model(H, A_norm)                    # (12, 2) log-probs
        loss = F.nll_loss(out, labels)             # negative log likelihood

        loss.backward()

        # ── Capture gradients at last epoch, before optimizer.step() ──
        if epoch == 299:
            final_grads = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }

        optimizer.step()

        if epoch % 50 == 0 or epoch == 299:
            model.eval()
            with torch.no_grad():
                eval_out = model(H, A_norm)
            acc = accuracy(eval_out, labels)
            print(f"  {epoch:>6}  {loss.item():>10.6f}  {acc:>9.1%}")

    # ── Per-student predictions ────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        final_out = model(H, A_norm)

    club_names  = ["Drama  ", "Science"]
    print()
    print("=" * 60)
    print("  Per-Student Predictions")
    print("=" * 60)
    print(f"  {'Student':<10} {'True Club':<14} {'Predicted':<14} {'P(Drama)':>9} {'P(Science)':>11} {'Correct?':>9}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*9} {'-'*11} {'-'*9}")

    probs = final_out.exp()          # convert log-probs → probs
    for i in range(12):
        true_club = labels[i].item()
        pred_club = probs[i].argmax().item()
        bridge    = "  ← bridge" if i in (4, 10) else ""
        correct   = "✓" if true_club == pred_club else "✗"
        print(
            f"  Student {i:<3} {club_names[true_club]:<14} "
            f"{club_names[pred_club]:<14} "
            f"{probs[i,0].item():>8.3f}  "
            f"{probs[i,1].item():>10.3f}  "
            f"{correct:>8}"
            f"{bridge}"
        )

    # ── Final gradients from last epoch ───────────────────────────────
    print()
    print("=" * 75)
    print("  Gradients — End of Epoch 299  (dL/dW before optimizer.step())")
    print("=" * 75)
    print(f"  {'Parameter':<35} {'Shape':<14} {'Norm':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*35} {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, grad in final_grads.items():
        print(
            f"  {name:<35} {str(list(grad.shape)):<14} "
            f"{grad.norm().item():>10.6f} "
            f"{grad.mean().item():>10.6f} "
            f"{grad.min().item():>10.6f} "
            f"{grad.max().item():>10.6f}"
        )

    # ── GCN vs MPNN: spot the difference in parameter count ───────────
    print()
    print("=" * 60)
    print("  Why GCN Has So Few Parameters")
    print("=" * 60)
    for name, param in model.named_parameters():
        print(f"  {name:<35} {str(list(param.shape)):<14} "
              f"{param.numel()} params")
    print()
    print("  GCN total          :", n_params, "parameters")
    print("  MPNN equivalent    : ~5000+ parameters")
    print()
    print("  GCN's W is shared across ALL edges.")
    print("  No message function. No update function. No edge features.")
    print("  That's both its strength (fast, few params)")
    print("  and its weakness (can't learn edge-specific patterns).")


if __name__ == "__main__":
    run_example()
