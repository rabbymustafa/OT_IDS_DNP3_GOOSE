"""Proposed Stage 2 diagnosis (Section V of the paper).

For one alerted sequence X it returns the localization signature
L(x) = <t*, onset vote, v*, g*, e*, m*, kappa>:
    time      window occlusion Delta_i = S(X) - S(X[i <- Xref]); t* = argmax Delta_i      (Eq. 8)
    device    whitened deviation c_v of node embeddings at t* (Ledoit-Wolf)              (Eq. 9)
    protocol  exact group Shapley phi_g with nu(B) = S(X[Pi\\B <- Xref]) - S(Xref)        (Eqs. 10-11)
    flow      edge occlusion among edges incident to v*, or the SCD stream of v*
    mechanism single-feature occlusion inside g*, mapped through config.mechanism_tags
plus the faithfulness checks reported in the paper (node Shapley, deletion).
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

from .players import (MaskedScorer, coalition_masks, edge_players, feature_players, group_players,
                      node_players, real_groups, shapley_from_scores, window_players)


@dataclass
class Localization:
    t_star: int
    device_rank: List[str]
    protocol: Optional[str]
    flow: Optional[str]
    window_importance: np.ndarray            # [T], non-negative, used for onset voting
    extras: Dict = field(default_factory=dict)


class NearestNormalReference:
    """Counterfactual reference: the nearest normal training sequence in standardized space."""

    def __init__(self, data, train_normal_windows):
        self.data = data
        self.windows = np.asarray(train_normal_windows)
        self.nn = NearestNeighbors(n_neighbors=1).fit(data.flat(self.windows))

    def query(self, windows) -> np.ndarray:
        _, idx = self.nn.kneighbors(self.data.flat(np.asarray(windows)))
        return self.windows[idx[:, 0]]


class Stage2Diagnoser:
    name = "proposed"

    def __init__(self, model, data, cfg, device="cpu"):
        self.model, self.data, self.cfg, self.device = model, data, cfg, device
        d = cfg.diagnosis
        self.scorer = MaskedScorer(model, device, d["batch_size"])
        self.topk = d["topk_features"]
        self.win_players = window_players(cfg)
        self.edge_players = edge_players(cfg)
        self.node_players = node_players(cfg)
        self.group_players = group_players(cfg)
        for p in (self.node_players, self.group_players):
            assert p.covers_all() and p.disjoint(), "Shapley players must partition all entries"
        self.group_coal = coalition_masks(self.group_players)
        self.node_coal = coalition_masks(self.node_players) if d["node_shapley"] else None
        self.calibrators = None

    # ------------------------------------------------------------------ fit
    @torch.no_grad()
    def node_embeddings(self, windows_1d) -> np.ndarray:
        self.model.eval()
        out = []
        for b in range(0, len(windows_1d), 1024):
            xn, xe = self.data.tensors(windows_1d[b:b + 1024], self.device)
            out.append(self.model.embed_nodes(xn, xe).cpu().numpy())
        return np.concatenate(out)

    def fit(self, train_normal_windows, val_normal_windows):
        """Reference index, per-node normal embedding statistics, and per-node ECDFs."""
        self.reference = NearestNormalReference(self.data, train_normal_windows)
        emb = self.node_embeddings(np.unique(train_normal_windows))
        self.mu = np.zeros((self.cfg.V, emb.shape[-1]))
        self.precision = np.zeros((self.cfg.V, emb.shape[-1], emb.shape[-1]))
        for v in range(self.cfg.V):
            lw = LedoitWolf().fit(emb[:, v, :])
            self.mu[v], self.precision[v] = lw.location_, lw.precision_
        val_windows = np.unique(val_normal_windows) if len(val_normal_windows) else np.unique(train_normal_windows)
        dev = np.stack([self.deviation(h) for h in self.node_embeddings(val_windows)])
        self.ecdf = [np.sort(dev[:, v]) for v in range(self.cfg.V)]
        return self

    def deviation(self, h, whiten=True) -> np.ndarray:
        """c_v for every node from embeddings h [V, d_g]; whiten=False gives the L2 deviation."""
        d = h - self.mu
        if not whiten:
            return np.linalg.norm(d, axis=1)
        return np.sqrt(np.maximum(np.einsum("vi,vij,vj->v", d, self.precision, d), 0.0))

    def kappa_v(self, v: int, c: float) -> float:
        s = self.ecdf[v]
        return float(np.searchsorted(s, c, side="right") / len(s))

    # ------------------------------------------------------------- diagnose
    def localize(self, windows) -> Localization:
        start = time.perf_counter()
        cfg = self.cfg
        windows = np.asarray(windows)
        xn, xe = self.data.tensors(windows[None], self.device)
        ref_windows = self.reference.query(windows[None])[0]
        rn, re = self.data.tensors(ref_windows[None], self.device)
        with torch.no_grad():
            out = self.model.forward_detailed(xn, xe)
        score = float(out["score"][0])
        alpha = out["alpha"][0].cpu().numpy()
        emb = out["node_emb"][0].cpu().numpy()                     # [T, V, dg]
        xn, xe, rn, re = xn[0], xe[0], rn[0], re[0]

        # Round 1: windows, edges, single nodes, group coalitions, node coalitions.
        blocks = [("win", self.win_players.node, self.win_players.edge),
                  ("edge", self.edge_players.node, self.edge_players.edge),
                  ("node1", self.node_players.node, self.node_players.edge),
                  ("group", *self.group_coal)]
        if self.node_coal is not None:
            blocks.append(("nodecoal", *self.node_coal))
        s = self.scorer(xn, xe, rn, re, np.concatenate([b[1] for b in blocks]),
                        np.concatenate([b[2] for b in blocks]))
        parts, pos = {}, 0
        for name, node_m, _ in blocks:
            parts[name] = s[pos:pos + len(node_m)]
            pos += len(node_m)

        # Time (Eq. 8), compared with the attention estimate.
        delta_t = score - parts["win"]
        t_star = int(np.argmax(delta_t))
        t_alpha = int(np.argmax(alpha))

        # Device: exact node Shapley over all windows. The whitened and L2 deviations at t* are
        # kept for the ablation (see docs/evaluation_rules.md, Changes after results).
        c = self.deviation(emb[t_star])
        c_l2 = self.deviation(emb[t_star], whiten=False)
        phi_v = shapley_from_scores(parts["nodecoal"], cfg.V)
        order = np.argsort(-phi_v)
        v_star = int(order[0])
        device = cfg.devices[v_star]

        # Protocol (Eqs. 10-11): exact group Shapley; efficiency gives S(X) - S(Xref).
        groups = self.group_players.names
        phi_g = shapley_from_scores(parts["group"], len(groups))
        s_ref, s_full = float(parts["group"][0]), float(parts["group"][-1])
        real = real_groups(groups)
        protocol_shapley = groups[real[int(np.argmax(phi_g[real]))]] if real else groups[int(np.argmax(phi_g))]
        # Protocol: the group with the largest departure from normal in the sequence (max |z|).
        # Group Shapley stays as an ablation (docs/evaluation_rules.md, Changes after results).
        zmax = {}
        for g, (nc, ec) in cfg.groups.items():
            zn = np.abs(self.data.z_node[windows][..., nc]).max() if nc else 0.0
            ze = np.abs(self.data.z_edge[windows][..., ec]).max() if ec else 0.0
            zmax[g] = float(max(zn, ze))
        protocol = max((groups[i] for i in real), key=zmax.get) if real else protocol_shapley

        # Flow: edge occlusion among edges incident to v*, or the node's own stream.
        edge_imp = score - parts["edge"]
        e_star = None
        if cfg.group_kind(protocol) == "node":
            flow = cfg.node_stream(device, protocol)
        else:
            cand = cfg.incident_edges(device) or list(range(cfg.E))
            e_star = int(cand[int(np.argmax(edge_imp[cand]))])
            flow = cfg.edges[e_star].name

        # Mechanism: single-feature occlusion inside the protocol group.
        feat_players, keys = feature_players(cfg, protocol,
                                             v_star if cfg.group_kind(protocol) != "edge" else None,
                                             e_star)
        mechanism, top_features = None, []
        if len(feat_players):
            feat_imp = score - self.scorer(xn, xe, rn, re, feat_players.node, feat_players.edge)
            top = np.argsort(-feat_imp)[:self.topk]
            top_features = [(feat_players.names[i], float(feat_imp[i])) for i in top]
            tag_score = {}
            for i in top:
                tag = cfg.feature_tag(*keys[i])
                if tag is not None and feat_imp[i] > 0:
                    tag_score[tag] = tag_score.get(tag, 0.0) + float(feat_imp[i])
            mechanism = max(tag_score, key=tag_score.get) if tag_score else None

        extras = {
            "score": score, "ref_score": s_ref, "alpha": alpha.tolist(),
            "t_alpha": t_alpha, "agree_time": t_alpha == t_star,
            "delta_t": delta_t.tolist(),
            "c": dict(zip(cfg.devices, c.tolist())),
            "device_rank_l2": [cfg.devices[i] for i in np.argsort(-c_l2)],
            "device_rank_whitened": [cfg.devices[i] for i in np.argsort(-c)],
            "kappa_v": self.kappa_v(v_star, float(c[v_star])),
            "phi_group": dict(zip(groups, phi_g.tolist())),
            "protocol_shapley": protocol_shapley, "zmax_group": zmax,
            "efficiency_error": abs((s_full - s_ref) - float(phi_g.sum())),
            "full_coalition_error": abs(s_full - score),
            "edge_importance": dict(zip(self.edge_players.names, edge_imp.tolist())),
            "deletion_drop": (score - float(parts["node1"][v_star])) / score if score > 0 else float("nan"),
            "mechanism": mechanism, "top_features": top_features,
        }
        extras["phi_node"] = dict(zip(cfg.devices, phi_v.tolist()))
        extras["latency_s"] = time.perf_counter() - start
        return Localization(t_star, [cfg.devices[i] for i in order], protocol, flow,
                            np.clip(delta_t, 0.0, None), extras)

    # ---------------------------------------------------------- confidence
    def calibrate(self, locs: List[Localization], correct: np.ndarray):
        """Isotonic map from (kappa_v*, time agreement) to P(signature correct) on validation alerts."""
        kappa = np.array([l.extras["kappa_v"] for l in locs])
        agree = np.array([l.extras["agree_time"] for l in locs], bool)
        correct = np.asarray(correct, float)
        self.calibrators = {}
        for flag in (True, False):
            m = agree == flag
            if m.sum() >= 5 and len(np.unique(correct[m])) > 1:
                self.calibrators[flag] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(kappa[m], correct[m])
        if len(kappa) >= 5 and len(np.unique(correct)) > 1:
            self.calibrators["pooled"] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(kappa, correct)
        return self

    def confidence(self, loc: Localization) -> float:
        kappa = loc.extras["kappa_v"]
        if not self.calibrators:
            return kappa
        iso = self.calibrators.get(bool(loc.extras["agree_time"]), self.calibrators.get("pooled"))
        return float(iso.predict([kappa])[0]) if iso is not None else kappa
