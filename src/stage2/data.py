"""Windows, annotations and stride-1 sequences.

Data file (``.npz``):
    node_x        float [N, V, d_n]   per-window node features (raw units)
    edge_x        float [N, E, d_e]   per-window edge features (raw units)
    window_start  float [N]           window start time in seconds (optional; default i * window_seconds)
    split         str or int [N]      "train"/"val"/"test" or 0/1/2 (optional; else config.split counts)
    rule_flag     int [N]             1 if any protocol invariant fired in the window (optional)
    edge_index    int [2, E]          checked against the config edge list (optional)

Manifest (``.csv``), one row per attack run:
    run_id, attack_class, target_device, protocol, flow, onset, offset[, path]
    ``onset``/``offset`` use the same time base as ``window_start``. Columns
    ``target_device``, ``protocol`` and ``flow`` accept alternatives separated by
    "|" (for example "DNP3|TCP"). ``path`` is "gnn" or "rule"; if absent it is
    "gnn" for classes listed in ``config.dynamic_classes``.
"""
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

SPLITS = ("train", "val", "test")


def alternatives(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [s.strip() for s in str(value).split("|") if s.strip()]


@dataclass
class Run:
    run_id: str
    attack_class: str
    path: str
    devices: List[str]
    protocols: List[str]
    flows: List[str]
    onset: float
    offset: float


def load_manifest(path, dynamic_classes) -> List[Run]:
    df = pd.read_csv(path)
    required = {"run_id", "attack_class", "target_device", "protocol", "flow", "onset", "offset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    runs = []
    for row in df.to_dict("records"):
        path_ = row.get("path")
        if not isinstance(path_, str) or not path_.strip():
            path_ = "gnn" if str(row["attack_class"]) in dynamic_classes else "rule"
        run = Run(str(row["run_id"]), str(row["attack_class"]), path_.strip().lower(),
                  alternatives(row["target_device"]), alternatives(row["protocol"]),
                  alternatives(row["flow"]), float(row["onset"]), float(row["offset"]))
        if run.offset <= run.onset:
            raise ValueError(f"run {run.run_id}: offset must be after onset")
        runs.append(run)
    return runs


@dataclass
class SeqSet:
    """Stride-1 sequences of ``seq_len`` consecutive windows within one split."""
    windows: np.ndarray   # [n, T] window indices
    label: np.ndarray     # [n] any anomalous window
    anom: np.ndarray      # [n, T] anomalous windows
    run: np.ndarray       # [n] run index of the anomalous windows (majority), -1 if none
    rule: np.ndarray      # [n] any rule flag in the sequence

    def __len__(self) -> int:
        return len(self.label)

    def subset(self, idx) -> "SeqSet":
        return SeqSet(self.windows[idx], self.label[idx], self.anom[idx], self.run[idx], self.rule[idx])


class OTData:
    """Windows in the model's input space (the repo's scaling) plus annotations.

    node_x, edge_x are what Stage 1 consumes. z_node, z_edge are z-scores from normal training
    windows, used only for the nearest normal reference and the raw z-score baseline.
    """

    def __init__(self, cfg, npz_path, manifest_path):
        self.cfg = cfg
        with np.load(npz_path, allow_pickle=False) as z:
            node_x = z["node_x"].astype(np.float64)
            edge_x = z["edge_x"].astype(np.float64)
            n = node_x.shape[0]
            window_start = (z["window_start"].astype(np.float64) if "window_start" in z.files
                            else np.arange(n) * cfg.window_seconds)
            split = z["split"] if "split" in z.files else None
            rule_flag = z["rule_flag"].astype(bool) if "rule_flag" in z.files else None
            edge_index = z["edge_index"] if "edge_index" in z.files else None
            self.synthetic = bool(z["synthetic"]) if "synthetic" in z.files else False

        if node_x.shape[1:] != (cfg.V, cfg.dn):
            raise ValueError(f"node_x has shape {node_x.shape}, expected (N, {cfg.V}, {cfg.dn})")
        if edge_x.shape != (n, cfg.E, cfg.de):
            raise ValueError(f"edge_x has shape {edge_x.shape}, expected ({n}, {cfg.E}, {cfg.de})")
        if edge_index is not None and not np.array_equal(edge_index, cfg.edge_index()):
            raise ValueError("edge_index in the data file does not match the config edge list")
        if np.any(np.diff(window_start) <= 0):
            raise ValueError("window_start must be strictly increasing")

        self.n = n
        self.window_start = window_start
        self.has_rule_flag = rule_flag is not None
        self.rule_flag = rule_flag if rule_flag is not None else np.zeros(n, bool)
        self.split = self._split(split, n)
        self.runs = load_manifest(manifest_path, cfg.dynamic_classes)
        self.window_run = self._assign_runs()
        self.anomalous = self.window_run >= 0

        normal_train = (self.split == "train") & ~self.anomalous
        if normal_train.sum() < 10:
            raise ValueError("fewer than 10 normal training windows")
        self.node_mu, self.node_sd = self._stats(node_x[normal_train])
        self.edge_mu, self.edge_sd = self._stats(edge_x[normal_train])
        self.node_x = node_x.astype(np.float32)
        self.edge_x = edge_x.astype(np.float32)
        self.z_node = ((node_x - self.node_mu) / self.node_sd).astype(np.float32)
        self.z_edge = ((edge_x - self.edge_mu) / self.edge_sd).astype(np.float32)

    @staticmethod
    def _stats(x):
        mu = x.mean(0)
        sd = x.std(0)
        sd[sd < 1e-8] = 1.0
        return mu, sd

    def _split(self, split, n) -> np.ndarray:
        if split is not None:
            split = np.asarray(split)
            names = np.array(SPLITS)[split.astype(int)] if split.dtype.kind in "iu" else split.astype(str)
            unknown = set(np.unique(names)) - set(SPLITS)
            if unknown:
                raise ValueError(f"unknown split labels: {sorted(unknown)}")
            return names
        counts = self.cfg.split or {"train": n * 2 // 3, "val": n // 6, "test": n - n * 2 // 3 - n // 6}
        if sum(counts.values()) != n:
            raise ValueError(f"split counts {counts} do not add up to {n} windows")
        return np.array(["train"] * counts["train"] + ["val"] * counts["val"] + ["test"] * counts["test"])

    def window_overlap(self, run) -> np.ndarray:
        start = self.window_start
        return np.minimum(start + self.cfg.window_seconds, run.offset) - np.maximum(start, run.onset)

    def _assign_runs(self) -> np.ndarray:
        best = np.full(self.n, -1)
        best_overlap = np.zeros(self.n)
        for r, run in enumerate(self.runs):
            overlap = self.window_overlap(run)
            if np.any((overlap > 0) & (best >= 0)):
                warnings.warn(f"run {run.run_id} overlaps another run; windows go to the larger overlap")
            take = overlap > best_overlap
            best[take] = r
            best_overlap[take] = overlap[take]
        for r, run in enumerate(self.runs):
            if not np.any(best == r):
                warnings.warn(f"run {run.run_id} covers no window")
        return best

    def run_first_window(self, r: int) -> Optional[int]:
        hit = np.flatnonzero(self.window_overlap(self.runs[r]) > 0)
        return int(hit[0]) if len(hit) else None

    def seqset(self, split: str) -> SeqSet:
        T = self.cfg.seq_len
        max_gap = 1.5 * self.cfg.window_seconds
        rows = []
        for t in np.flatnonzero(self.split == split):
            w = np.arange(t - T + 1, t + 1)
            if w[0] < 0 or np.any(self.split[w] != split):
                continue
            if T > 1 and np.any(np.diff(self.window_start[w]) > max_gap):
                continue
            rows.append(w)
        windows = np.array(rows, dtype=np.int64).reshape(-1, T)
        anom = self.anomalous[windows]
        runs = np.full(len(windows), -1)
        for i, w in enumerate(windows):
            r = self.window_run[w]
            r = r[r >= 0]
            if len(r):
                runs[i] = np.bincount(r).argmax()
        return SeqSet(windows, anom.any(1), anom, runs, self.rule_flag[windows].any(1))

    def tensors(self, windows, device="cpu"):
        w = np.asarray(windows)
        return (torch.from_numpy(self.node_x[w]).to(device),
                torch.from_numpy(self.edge_x[w]).to(device))

    def flat(self, windows) -> np.ndarray:
        w = np.asarray(windows)
        return np.concatenate([self.z_node[w].reshape(len(w), -1),
                               self.z_edge[w].reshape(len(w), -1)], axis=1)
