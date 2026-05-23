"""
Message Passing Neural Network (MPNN) — From Scratch
=====================================================
Pure PyTorch. No torch_geometric. No magic.

Architecture overview:
    ┌─────────────────────────────────────────────────────┐
    │  Raw node features (atom type, etc.)                │
    │        ↓  Node Encoder (Linear)                     │
    │  Hidden node states  h_i                            │
    │        ↓  × num_layers                              │
    │  ┌─────────────────────────────────────┐            │
    │  │  1. MessageFunction(h_i, h_j, e_ij) │            │
    │  │  2. Aggregate: Σ messages → node    │            │
    │  │  3. UpdateFunction(h_i, agg_msg)    │            │
    │  └─────────────────────────────────────┘            │
    │        ↓  Readout (sum nodes)                       │
    │  Graph-level vector                                 │
    │        ↓  MLP                                       │
    │  Scalar prediction (energy)                         │
    └─────────────────────────────────────────────────────┘

Toy example:
    4-atom graph  (O, H, H, C)
    Edge features = bond distances (Angstrom)
    Task          = predict a scalar "energy" value
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MESSAGE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class MessageFunction(nn.Module):
    """
    Computes the message from neighbor j → node i.

    Concatenates [h_i || h_j || e_ij] and passes through a 2-layer MLP.
    This lets the message depend on:
        - who is sending  (h_j)
        - who is receiving (h_i)   ← makes it asymmetric and expressive
        - the bond between them    (e_ij: distance, angle, etc.)

    Think of it as the "ghostwriter" from the town analogy —
    it reads both houses and the road between them, then writes a letter.

    Args:
        node_dim (int): Size of node feature vectors.
        edge_dim (int): Size of edge feature vectors.
        msg_dim  (int): Size of output message vectors.

    Example:
        >>> msg_fn = MessageFunction(node_dim=16, edge_dim=1, msg_dim=16)
        >>> h_i    = torch.randn(6, 16)   # 6 edges — receiver features
        >>> h_j    = torch.randn(6, 16)   # 6 edges — sender features
        >>> e_ij   = torch.randn(6, 1)    # 6 edges — distances
        >>> msgs   = msg_fn(h_i, h_j, e_ij)
        >>> msgs.shape
        torch.Size([6, 16])
    """

    def __init__(self, node_dim: int, edge_dim: int, msg_dim: int):
        super().__init__()
        in_dim = node_dim * 2 + edge_dim        # concat sender + receiver + edge
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, msg_dim),
            nn.SiLU(),                          # SiLU (Swish) — standard in MACE/NequIP
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(
        self,
        h_i: torch.Tensor,     # (num_edges, node_dim)  receiver features
        h_j: torch.Tensor,     # (num_edges, node_dim)  sender features
        e_ij: torch.Tensor,    # (num_edges, edge_dim)  edge features
    ) -> torch.Tensor:
        """
        Args:
            h_i  : Receiver features for every edge.  Shape (E, node_dim).
            h_j  : Sender features for every edge.    Shape (E, node_dim).
            e_ij : Edge features for every edge.      Shape (E, edge_dim).

        Returns:
            Tensor of messages, shape (E, msg_dim).
        """
        x = torch.cat([h_i, h_j, e_ij], dim=-1)   # (E, 2*node_dim + edge_dim)
        return self.mlp(x)                          # (E, msg_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UPDATE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class UpdateFunction(nn.Module):
    """
    Updates a node's state after it has received all messages.

    Concatenates the node's current state with the *sum* of incoming
    messages, then passes through a 2-layer MLP.

    Args:
        node_dim (int): Input node feature size.
        msg_dim  (int): Aggregated message size.
        out_dim  (int): Output node feature size (usually == node_dim).

    Example:
        >>> upd_fn  = UpdateFunction(node_dim=16, msg_dim=16, out_dim=16)
        >>> h_i     = torch.randn(4, 16)    # 4 nodes, current state
        >>> agg_msg = torch.randn(4, 16)    # 4 nodes, summed messages
        >>> h_new   = upd_fn(h_i, agg_msg)
        >>> h_new.shape
        torch.Size([4, 16])
    """

    def __init__(self, node_dim: int, msg_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + msg_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        h_i: torch.Tensor,      # (N, node_dim)
        agg_msg: torch.Tensor,  # (N, msg_dim)
    ) -> torch.Tensor:
        """
        Args:
            h_i     : Current node features.        Shape (N, node_dim).
            agg_msg : Sum of all incoming messages.  Shape (N, msg_dim).

        Returns:
            Updated node features, shape (N, out_dim).
        """
        x = torch.cat([h_i, agg_msg], dim=-1)   # (N, node_dim + msg_dim)
        return self.mlp(x)                        # (N, out_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ONE MPNN LAYER  (one full round of message passing)
# ══════════════════════════════════════════════════════════════════════════════

class MPNNLayer(nn.Module):
    """
    One complete round of message passing:

        Step 1 — Compose:   compute messages for every directed edge
        Step 2 — Aggregate: sum all incoming messages at each node
        Step 3 — Update:    update each node's hidden state

    After one layer, every node "knows" about its 1-hop neighbourhood.
    After k layers → k-hop neighbourhood.  This is your receptive field.

    Args:
        node_dim (int): Node feature size (same in and out).
        edge_dim (int): Edge feature size.
        msg_dim  (int): Internal message dimensionality.

    Example:
        >>> layer  = MPNNLayer(node_dim=16, edge_dim=1, msg_dim=16)
        >>> h      = torch.randn(4, 16)             # 4 nodes
        >>> e      = torch.randn(6, 1)              # 6 directed edges
        >>> src    = torch.tensor([0, 0, 1, 1, 2, 3])
        >>> dst    = torch.tensor([1, 2, 0, 3, 3, 2])
        >>> h_new  = layer(h, e, src, dst)
        >>> h_new.shape
        torch.Size([4, 16])
    """

    def __init__(self, node_dim: int, edge_dim: int, msg_dim: int):
        super().__init__()
        self.message_fn = MessageFunction(node_dim, edge_dim, msg_dim)
        self.update_fn  = UpdateFunction(node_dim, msg_dim, node_dim)
        self.msg_dim    = msg_dim

    def forward(
        self,
        h: torch.Tensor,           # (N, node_dim)
        edge_feat: torch.Tensor,   # (E, edge_dim)
        src: torch.Tensor,         # (E,)  sender   index per edge
        dst: torch.Tensor,         # (E,)  receiver index per edge
    ) -> torch.Tensor:
        """
        Args:
            h         : Node feature matrix.   Shape (N, node_dim).
            edge_feat : Edge feature matrix.   Shape (E, edge_dim).
            src       : Sender node indices.   Shape (E,).
            dst       : Receiver node indices. Shape (E,).

        Returns:
            Updated node features, shape (N, node_dim).

        Gradient note:
            scatter_add_ is differentiable w.r.t. `messages`, so
            backprop flows cleanly through the aggregation step.
        """
        num_nodes = h.size(0)

        # ── Step 1: compute one message per directed edge ──────────────
        h_i = h[dst]        # receiver features, shape (E, node_dim)
        h_j = h[src]        # sender   features, shape (E, node_dim)

        messages = self.message_fn(h_i, h_j, edge_feat)   # (E, msg_dim)

        # ── Step 2: sum messages arriving at each node ─────────────────
        # scatter_add:  agg[dst[e]] += messages[e]  for each edge e
        # Result is permutation-invariant (order of neighbours doesn't matter)
        agg = torch.zeros(num_nodes, self.msg_dim,
                          dtype=h.dtype, device=h.device)
        idx = dst.unsqueeze(-1).expand_as(messages)        # (E, msg_dim)
        agg.scatter_add_(0, idx, messages)                 # (N, msg_dim)

        # ── Step 3: update node states ─────────────────────────────────
        return self.update_fn(h, agg)                      # (N, node_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL MPNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MPNN(nn.Module):
    """
    Full Message Passing Neural Network for graph-level regression.

    Stacks `num_layers` rounds of MPNNLayer, then applies a sum-readout
    to collapse all node states into a single graph-level vector, which
    is finally mapped to a scalar prediction (e.g. total energy).

    Args:
        node_dim   (int): Raw input node feature size.
        edge_dim   (int): Edge feature size.
        hidden_dim (int): Hidden dimensionality throughout the network.
        num_layers (int): Number of message-passing rounds (= receptive field).
        out_dim    (int): Output size (1 for scalar energy regression).

    Attributes:
        node_encoder : Projects raw node features into hidden_dim space.
        mp_layers    : nn.ModuleList of MPNNLayer rounds.
        readout      : MLP that maps graph-level sum → prediction.

    Example:
        >>> model = MPNN(node_dim=3, edge_dim=1, hidden_dim=16,
        ...              num_layers=3, out_dim=1)
        >>> pred = model(node_feat, edge_feat, src, dst)
        >>> pred.shape
        torch.Size([1, 1])
    """

    def __init__(
        self,
        node_dim:   int,
        edge_dim:   int,
        hidden_dim: int = 16,
        num_layers: int = 3,
        out_dim:    int = 1,
    ):
        super().__init__()

        # Lift raw features into the hidden space before message passing
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        # Stack of message-passing layers
        self.mp_layers = nn.ModuleList([
            MPNNLayer(node_dim=hidden_dim, edge_dim=edge_dim, msg_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        # Readout: sum over nodes → MLP → scalar
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        node_feat: torch.Tensor,   # (N, node_dim)
        edge_feat: torch.Tensor,   # (E, edge_dim)
        src:       torch.Tensor,   # (E,)
        dst:       torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            node_feat : Raw node features.    Shape (N, node_dim).
            edge_feat : Edge features.        Shape (E, edge_dim).
            src       : Sender indices.       Shape (E,).
            dst       : Receiver indices.     Shape (E,).

        Returns:
            Graph-level scalar prediction, shape (1, out_dim).

        Computation graph (for backprop tracing):
            node_feat
              → node_encoder          [Linear]
              → h
              → MPNNLayer × L         [message → aggregate → update]
              → h_final
              → sum over nodes        [differentiable, grad = 1 per node]
              → readout MLP
              → prediction
        """
        # ── Encode raw features ────────────────────────────────────────
        h = self.node_encoder(node_feat)            # (N, hidden_dim)

        # ── Iterative message passing ──────────────────────────────────
        for layer in self.mp_layers:
            h = layer(h, edge_feat, src, dst)       # (N, hidden_dim)

        # ── Readout: collapse all nodes into one graph vector ──────────
        graph_repr = h.sum(dim=0, keepdim=True)     # (1, hidden_dim)

        # ── Predict ────────────────────────────────────────────────────
        return self.readout(graph_repr)             # (1, out_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TOY GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_toy_graph():
    """
    Builds a small 4-atom graph resembling a fragment of a molecule.

    Graph layout:

            H(1)
            |  0.96 Å
        O(0)
            |  0.96 Å
            H(2)
            |  1.43 Å
            C(3)

    Node features — one-hot atom type  [O, H, C]:
        Atom 0 (O) → [1, 0, 0]
        Atom 1 (H) → [0, 1, 0]
        Atom 2 (H) → [0, 1, 0]
        Atom 3 (C) → [0, 0, 1]

    Edge features — bond distance in Ångström (scalar).
    Each undirected bond is stored as 2 directed edges.

    Returns:
        node_feat (Tensor): shape (4, 3)
        edge_feat (Tensor): shape (6, 1)   — distances
        src       (Tensor): shape (6,)
        dst       (Tensor): shape (6,)
        target    (Tensor): shape (1, 1)   — fake DFT energy in eV
    """
    node_feat = torch.tensor([
        [1., 0., 0.],   # Atom 0: Oxygen
        [0., 1., 0.],   # Atom 1: Hydrogen
        [0., 1., 0.],   # Atom 2: Hydrogen
        [0., 0., 1.],   # Atom 3: Carbon
    ])

    # (sender, receiver, distance_Angstrom)
    bonds = [(0, 1, 0.96), (0, 2, 0.96), (0, 3, 1.43)]

    src_list, dst_list, dist_list = [], [], []
    for i, j, d in bonds:
        src_list += [i, j]     # both directions
        dst_list += [j, i]
        dist_list += [d, d]

    src       = torch.tensor(src_list)
    dst       = torch.tensor(dst_list)
    edge_feat = torch.tensor(dist_list).unsqueeze(-1).float()  # (6, 1)

    target = torch.tensor([[-5.23]])    # pretend DFT energy

    return node_feat, edge_feat, src, dst, target


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_example():
    """
    Trains the MPNN on the toy graph for 300 steps and prints progress.

    Loss     : Mean Squared Error between predicted and target energy.
    Optimizer: Adam, lr=1e-3.
    Layers   : 3 rounds of message passing
                 → after 3 rounds each atom "knows about" its 3-hop neighbourhood.

    Gradient flow summary:
        Loss
         └─ dL/dPred
              └─ readout MLP weights
              └─ dL/d(graph_repr)   [sum → grad = 1 per node]
                   └─ dL/dh_final
                        └─ UpdateFunction weights
                        └─ dL/d(agg_msg)   [scatter_add → grad = 1 per edge]
                             └─ MessageFunction weights
                             └─ dL/d(edge_feat)   [← path to positions/forces]
    """
    torch.manual_seed(42)

    node_feat, edge_feat, src, dst, target = build_toy_graph()

    print("=" * 55)
    print("  Toy Graph")
    print("=" * 55)
    print(f"  Nodes      : {node_feat.shape[0]}  (O, H, H, C)")
    print(f"  Edges      : {edge_feat.shape[0]}  directed bonds")
    print(f"  Node feats : {list(node_feat.shape)}  (one-hot atom type)")
    print(f"  Edge feats : {list(edge_feat.shape)}  (bond distance)")
    print(f"  Target     : {target.item():.3f} eV")
    print()

    model = MPNN(
        node_dim   = 3,
        edge_dim   = 1,
        hidden_dim = 32,
        num_layers = 3,
        out_dim    = 1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params}")
    print()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    print("=" * 55)
    print("  Training")
    print("=" * 55)
    print(f"  {'Step':>5}  {'Loss':>12}  {'Prediction':>12}  {'Target':>8}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*8}")

    for step in range(300):
        optimizer.zero_grad()

        pred = model(node_feat, edge_feat, src, dst)
        loss = loss_fn(pred, target)

        loss.backward()    # backprop through all message-passing layers
        optimizer.step()

        if step % 30 == 0 or step == 299:
            print(f"  {step:>5}  {loss.item():>12.6f}  "
                  f"{pred.item():>12.4f}  {target.item():>8.4f}")

    print()
    print("=" * 55)
    print("  Result")
    print("=" * 55)
    with torch.no_grad():
        final = model(node_feat, edge_feat, src, dst)
    print(f"  Predicted  : {final.item():.4f} eV")
    print(f"  Target     : {target.item():.4f} eV")
    print(f"  Abs Error  : {abs(final.item() - target.item()):.4f} eV")
    print()

    # ── Show gradient norms per layer (just for visibility) ───────────
    print("=" * 55)
    print("  Gradient norms after final backward()")
    print("=" * 55)
    # Run one more backward to populate .grad
    pred = model(node_feat, edge_feat, src, dst)
    loss = loss_fn(pred, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name:<45} {param.grad.norm().item():.6f}")


if __name__ == "__main__":
    run_example()
