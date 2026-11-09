"""Occlusion players, batched masked scoring, and exact Shapley values.

A player is a boolean mask over the entries of one sequence (node features
[T, V, d_n] and edge features [T, E, d_e]). Occluding a player copies its
entries from the counterfactual reference sequence. Players used for Shapley
values must partition all entries, so that the empty coalition reproduces the
reference and the full coalition reproduces the observed sequence.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import OTHER_GROUP


@dataclass
class Players:
    names: List[str]
    node: np.ndarray   # [K, T, V, dn] bool
    edge: np.ndarray   # [K, T, E, de] bool

    def __len__(self) -> int:
        return len(self.names)

    def covers_all(self) -> bool:
        return bool(self.node.any(0).all() and self.edge.any(0).all())

    def disjoint(self) -> bool:
        return bool((self.node.sum(0) <= 1).all() and (self.edge.sum(0) <= 1).all())


def _empty(cfg, k):
    return (np.zeros((k, cfg.seq_len, cfg.V, cfg.dn), bool),
            np.zeros((k, cfg.seq_len, cfg.E, cfg.de), bool))


def window_players(cfg) -> Players:
    node, edge = _empty(cfg, cfg.seq_len)
    for t in range(cfg.seq_len):
        node[t, t] = True
        edge[t, t] = True
    return Players([f"t{t}" for t in range(cfg.seq_len)], node, edge)


def node_players(cfg) -> Players:
    """One player per device: its node features plus the edges it owns (config.edge_owner)."""
    node, edge = _empty(cfg, cfg.V)
    for v in range(cfg.V):
        node[v, :, v, :] = True
    for e in range(cfg.E):
        edge[cfg.edge_owner_index(e), :, e, :] = True
    return Players(list(cfg.devices), node, edge)


def edge_players(cfg) -> Players:
    node, edge = _empty(cfg, cfg.E)
    for e in range(cfg.E):
        edge[e, :, e, :] = True
    return Players([e.name for e in cfg.edges], node, edge)


def group_players(cfg) -> Players:
    names = list(cfg.groups)
    node, edge = _empty(cfg, len(names))
    for k, g in enumerate(names):
        node_cols, edge_cols = cfg.groups[g]
        node[k][..., node_cols] = True
        edge[k][..., edge_cols] = True
    return Players(names, node, edge)


def feature_players(cfg, group: str, device: Optional[int], edge: Optional[int]) -> Tuple[Players, List[Tuple[str, int]]]:
    """Single features of ``group``: node columns at ``device``, edge columns at ``edge``."""
    node_cols, edge_cols = cfg.groups[group]
    keys = []
    if device is not None:
        keys += [("node", c) for c in node_cols]
    if edge is not None:
        keys += [("edge", c) for c in edge_cols]
    node_m, edge_m = _empty(cfg, len(keys))
    for k, (kind, c) in enumerate(keys):
        if kind == "node":
            node_m[k, :, device, c] = True
        else:
            edge_m[k, :, edge, c] = True
    names = [cfg.feature_name(kind, c) for kind, c in keys]
    return Players(names, node_m, edge_m), keys


def coalition_masks(players: Players) -> Tuple[np.ndarray, np.ndarray]:
    """Replacement masks for every coalition (bit k set = player k keeps its observed values)."""
    k_players = len(players)
    if k_players > 12:
        raise ValueError("exact Shapley values are limited to 12 players")
    n = 1 << k_players
    node = np.zeros((n,) + players.node.shape[1:], bool)
    edge = np.zeros((n,) + players.edge.shape[1:], bool)
    for c in range(n):
        for k in range(k_players):
            if not (c >> k) & 1:
                node[c] |= players.node[k]
                edge[c] |= players.edge[k]
    return node, edge


def shapley_from_scores(scores: np.ndarray, k_players: int) -> np.ndarray:
    """Exact Shapley values of the game nu(c) = scores[c] - scores[0]."""
    fact = [math.factorial(i) for i in range(k_players + 1)]
    sizes = [bin(c).count("1") for c in range(1 << k_players)]
    phi = np.zeros(k_players)
    for k in range(k_players):
        bit = 1 << k
        for c in range(1 << k_players):
            if c & bit:
                continue
            weight = fact[sizes[c]] * fact[k_players - sizes[c] - 1] / fact[k_players]
            phi[k] += weight * (scores[c | bit] - scores[c])
    return phi


class MaskedScorer:
    """Scores many occluded copies of one sequence in batches."""

    def __init__(self, model, device="cpu", batch_size=512):
        self.model, self.device, self.batch_size = model, device, batch_size

    @torch.no_grad()
    def __call__(self, xn, xe, rn, re, node_masks, edge_masks) -> np.ndarray:
        # xn, rn: [T, V, dn]; xe, re: [T, E, de]; masks True = take the reference value
        out = []
        for b in range(0, len(node_masks), self.batch_size):
            mn = torch.from_numpy(node_masks[b:b + self.batch_size]).to(self.device)
            me = torch.from_numpy(edge_masks[b:b + self.batch_size]).to(self.device)
            node = torch.where(mn, rn.unsqueeze(0), xn.unsqueeze(0))
            edge = torch.where(me, re.unsqueeze(0), xe.unsqueeze(0))
            out.append(self.model.forward_detailed(node, edge)["score"].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def real_groups(names: List[str]) -> List[int]:
    return [i for i, g in enumerate(names) if g != OTHER_GROUP]
