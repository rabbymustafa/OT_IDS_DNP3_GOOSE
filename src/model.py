"""Stage 1 model: GNN + GRU + attention reconstruction, re-implemented from docs/previous_work.md.

The encoder is a dense, batched equivalent of torch_geometric's
NNConv(7, 32, Linear(24, 32*7), aggr="mean") on the fixed 4 edge graph. Stage 2 evaluates
hundreds of perturbed sequences per alert, so it needs fixed shape tensors instead of PyG
batches. tests/test_model.py checks the equivalence with PyG.

Tensors: node_x [..., 6, 7], edge_x [..., 4, 24]. Sequences add a batch and a time axis in front.

Settings the document does not state (recorded in docs/evaluation_rules.md):
    NNConv keeps the PyG defaults, so a node also gets a linear transform of its own features
    and a bias. Without this the three nodes that have no edges would carry no information.
    The target embedding is not detached from the gradient.
"""
import torch
import torch.nn.functional as F
from torch import nn

EDGE_INDEX = ((1, 1, 4, 5), (4, 5, 1, 1))  # SCADA <-> SEL_451_72, SEL_451_73 (sorted by src, dst)


class DenseNNConv(nn.Module):
    """x'_v = mean_{u->v} x_u W(e_uv) + Theta x_v + b."""

    def __init__(self, in_dim, out_dim, edge_dim, edge_index=EDGE_INDEX, num_nodes=6):
        super().__init__()
        self.in_dim, self.out_dim, self.num_nodes = in_dim, out_dim, num_nodes
        self.edge_net = nn.Linear(edge_dim, in_dim * out_dim)
        self.root = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.uniform_(self.root.weight, -in_dim ** -0.5, in_dim ** -0.5)  # PyG's initialization
        self.bias = nn.Parameter(torch.zeros(out_dim))
        src, dst = torch.as_tensor(edge_index, dtype=torch.long)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        count = torch.bincount(dst, minlength=num_nodes).clamp(min=1).float()
        self.register_buffer("count", count)

    def forward(self, x, e):
        # x: [N, V, in], e: [N, E, edge]
        n = x.shape[0]
        weights = self.edge_net(e).view(n, e.shape[1], self.in_dim, self.out_dim)
        msg = torch.einsum("nei,neio->neo", x[:, self.src, :], weights)
        agg = x.new_zeros(n, self.num_nodes, self.out_dim).index_add(1, self.dst, msg)
        return agg / self.count.view(1, -1, 1) + self.root(x) + self.bias


class ReconstructionModel(nn.Module):
    """Graph encoder -> GRU -> attention over the T windows -> decoder -> Huber reconstruction score."""

    def __init__(self, node_dim=7, edge_dim=24, hidden=32, gru_dim=32):
        super().__init__()
        self.conv = DenseNNConv(node_dim, hidden, edge_dim)
        self.gru = nn.GRU(2 * hidden, gru_dim, batch_first=True)
        self.att = nn.Linear(gru_dim, 1)
        self.dec = nn.Sequential(nn.Linear(gru_dim, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, 2 * hidden))

    def embed_nodes(self, node_x, edge_x):
        """Node embeddings before pooling: [..., V, hidden]."""
        lead = node_x.shape[:-2]
        h = self.conv(node_x.reshape(-1, *node_x.shape[-2:]), edge_x.reshape(-1, *edge_x.shape[-2:]))
        return torch.relu(h).reshape(*lead, *h.shape[-2:])

    def forward_detailed(self, node_x, edge_x):
        h = self.embed_nodes(node_x, edge_x)                       # [B, T, V, hidden]
        z = torch.cat([h.mean(-2), h.max(-2).values], dim=-1)      # [B, T, 2 hidden]
        states, _ = self.gru(z)
        alpha = torch.softmax(self.att(states).squeeze(-1), dim=1)
        context = (alpha.unsqueeze(-1) * states).sum(1)
        recon = self.dec(context)
        target = z.mean(1)
        score = F.smooth_l1_loss(recon, target, beta=1.0, reduction="none").mean(-1)
        return {"score": score, "alpha": alpha, "node_emb": h, "context": context}

    def forward(self, node_x, edge_x):
        return self.forward_detailed(node_x, edge_x)["score"]
