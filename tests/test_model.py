"""The dense encoder must match PyG's NNConv on the real graph."""
import sys
from pathlib import Path

import torch
from torch_geometric.nn import NNConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import EDGE_INDEX, DenseNNConv, ReconstructionModel  # noqa: E402


def test_dense_conv_matches_pyg():
    torch.manual_seed(0)
    dense = DenseNNConv(7, 32, 24)
    pyg = NNConv(7, 32, torch.nn.Linear(24, 32 * 7), aggr="mean")
    with torch.no_grad():
        pyg.nn.weight.copy_(dense.edge_net.weight)
        pyg.nn.bias.copy_(dense.edge_net.bias)
        pyg.lin.weight.copy_(dense.root.weight)
        pyg.bias.copy_(dense.bias.fill_(0.3))
    x, e = torch.randn(6, 7), torch.randn(4, 24)
    edge_index = torch.tensor(EDGE_INDEX)
    assert torch.allclose(dense(x[None], e[None])[0], pyg(x, edge_index, e), atol=1e-5)


def test_shapes_and_isolated_nodes_carry_signal():
    torch.manual_seed(0)
    model = ReconstructionModel()
    node_x, edge_x = torch.randn(2, 3, 6, 7), torch.randn(2, 3, 4, 24)
    out = model.forward_detailed(node_x, edge_x)
    assert out["score"].shape == (2,) and out["alpha"].shape == (2, 3)
    assert out["node_emb"].shape == (2, 3, 6, 32)
    assert torch.allclose(out["alpha"].sum(1), torch.ones(2))
    changed = node_x.clone()
    changed[:, :, 0, :] += 5.0  # RTDS has no edges
    assert not torch.allclose(model.embed_nodes(changed, edge_x)[:, :, 0], out["node_emb"][:, :, 0])
