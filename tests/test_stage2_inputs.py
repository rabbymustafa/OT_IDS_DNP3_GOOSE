"""The Stage 2 inputs must reproduce the repo's training and validation sequences exactly."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import Config  # noqa: E402
from stage2.data import OTData  # noqa: E402
from train import load_split  # noqa: E402

INPUTS = ROOT / "results" / "stage2_inputs"


def _data():
    cfg = Config.load(INPUTS / "config.json")
    return cfg, OTData(cfg, INPUTS / "windows.npz", INPUTS / "runs.csv")


def test_sequences_match_repo():
    _, data = _data()
    with np.load(INPUTS / "windows.npz") as z:
        window_num = z["window_num"]
    for name in ("train", "val"):
        repo, ours = load_split(name), data.seqset(name)
        assert np.array_equal(window_num[ours.windows], repo["windows"])
        assert np.array_equal(ours.label, repo["label"] > 0)
        xn, xe = data.tensors(ours.windows)
        assert np.array_equal(xn.numpy(), repo["node_x"].numpy())
        assert np.array_equal(xe.numpy(), repo["edge_x"].numpy())


def test_groups_partition_features():
    cfg, _ = _data()
    assert set(cfg.groups) == {"GOOSE", "DNP3", "TCP"}
    assert sorted(i for g in cfg.groups.values() for i in g[0]) == list(range(cfg.dn))
    assert sorted(i for g in cfg.groups.values() for i in g[1]) == list(range(cfg.de))
