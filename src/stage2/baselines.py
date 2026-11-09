"""Baseline localizers for Table V.

Each exposes ``name`` and ``localize(windows[T]) -> Localization`` so that it is
scored exactly like the proposed diagnosis:
    raw_z                 model-free: mean |z| of standardized features per window / device / group / edge
    attention_l2          argmax attention + unwhitened L2 node deviation (device and time only)
    integrated_gradients  IG of the score from the reference sequence (Sundararajan et al., 2017)
    gnnexplainer          GNNExplainer-style soft masks (Ying et al., 2019) adapted to the
                          reconstruction score: keep S(X) with sparse, near-binary masks
"""
from typing import Dict, List

import numpy as np
import torch

from .diagnosis import Localization
from .players import edge_players, group_players, node_players, real_groups, window_players


class _Aggregator:
    """Sums an entry-level importance over windows, device players, groups and edges."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.win = window_players(cfg)
        self.node = node_players(cfg)
        self.group = group_players(cfg)
        self.edge = edge_players(cfg)

    @staticmethod
    def _reduce(players, imp_node, imp_edge, mean, largest=False):
        out = []
        for k in range(len(players)):
            if largest:
                vals = np.concatenate([imp_node[players.node[k]], imp_edge[players.edge[k]]])
                out.append(vals.max() if len(vals) else 0.0)
                continue
            total = imp_node[players.node[k]].sum() + imp_edge[players.edge[k]].sum()
            count = players.node[k].sum() + players.edge[k].sum()
            out.append(total / max(count, 1) if mean else total)
        return np.array(out)

    def localization(self, imp_node, imp_edge, mean=False, largest=False) -> Localization:
        cfg = self.cfg
        win = self._reduce(self.win, imp_node, imp_edge, mean, largest)
        dev = self._reduce(self.node, imp_node, imp_edge, mean, largest)
        grp = self._reduce(self.group, imp_node, imp_edge, mean, largest)
        order = np.argsort(-dev)
        device = cfg.devices[int(order[0])]
        real = real_groups(self.group.names)
        protocol = self.group.names[real[int(np.argmax(grp[real]))]]
        if cfg.group_kind(protocol) == "node":
            flow = cfg.node_stream(device, protocol)
        else:
            node_cols, edge_cols = cfg.groups[protocol]
            edge_imp = (imp_edge[:, :, edge_cols].max(axis=(0, 2)) if largest
                        else imp_edge[:, :, edge_cols].sum(axis=(0, 2)))
            if mean and not largest:
                edge_imp = edge_imp / max(len(edge_cols) * cfg.seq_len, 1)
            cand = cfg.incident_edges(device) or list(range(cfg.E))
            flow = cfg.edges[int(cand[int(np.argmax(edge_imp[cand]))])].name
        return Localization(int(np.argmax(win)), [cfg.devices[i] for i in order], protocol, flow,
                            np.clip(win, 0.0, None),
                            {"group_importance": dict(zip(self.group.names, grp.tolist()))})


class RawZScore:
    name = "raw_z"

    def __init__(self, data, cfg):
        self.data, self.cfg = data, cfg
        self.agg = _Aggregator(cfg)

    def localize(self, windows) -> Localization:
        w = np.asarray(windows)
        return self.agg.localization(np.abs(self.data.z_node[w]), np.abs(self.data.z_edge[w]), mean=True)


class RawZMax:
    """Model-free: the largest |z| per window, device, group and edge."""
    name = "raw_z_max"

    def __init__(self, data, cfg):
        self.data, self.cfg = data, cfg
        self.agg = _Aggregator(cfg)

    def localize(self, windows) -> Localization:
        w = np.asarray(windows)
        return self.agg.localization(np.abs(self.data.z_node[w]), np.abs(self.data.z_edge[w]), largest=True)


class AttentionL2:
    """The estimator of the accepted digest: argmax attention and the L2 node deviation."""
    name = "attention_l2"

    def __init__(self, diagnoser):
        self.diag = diagnoser

    @torch.no_grad()
    def localize(self, windows) -> Localization:
        d = self.diag
        xn, xe = d.data.tensors(np.asarray(windows)[None], d.device)
        out = d.model.forward_detailed(xn, xe)
        alpha = out["alpha"][0].cpu().numpy()
        t = int(np.argmax(alpha))
        c = d.deviation(out["node_emb"][0, t].cpu().numpy(), whiten=False)
        order = np.argsort(-c)
        return Localization(t, [d.cfg.devices[i] for i in order], None, None, alpha.copy(), {})


class IntegratedGradients:
    name = "integrated_gradients"

    def __init__(self, diagnoser, steps=32):
        self.diag, self.steps = diagnoser, steps
        self.agg = _Aggregator(diagnoser.cfg)

    def localize(self, windows) -> Localization:
        d = self.diag
        w = np.asarray(windows)
        xn, xe = d.data.tensors(w[None], d.device)
        rn, re = d.data.tensors(d.reference.query(w[None]), d.device)
        a = ((torch.arange(self.steps, dtype=xn.dtype) + 0.5) / self.steps).to(d.device)
        path_n = (rn + a.view(-1, 1, 1, 1) * (xn - rn)).requires_grad_(True)
        path_e = (re + a.view(-1, 1, 1, 1) * (xe - re)).requires_grad_(True)
        total = d.model.forward_detailed(path_n, path_e)["score"].sum()
        gn, ge = torch.autograd.grad(total, [path_n, path_e])
        attr_n = ((xn - rn)[0] * gn.mean(0)).detach().cpu().numpy()
        attr_e = ((xe - re)[0] * ge.mean(0)).detach().cpu().numpy()
        loc = self.agg.localization(attr_n, attr_e, mean=False)
        loc.extras["completeness_sum"] = float(attr_n.sum() + attr_e.sum())
        return loc


class MaskExplainer:
    """Soft masks over windows, device players, edges and protocol groups.

    Entry mask = m_window[t] * m_device[owner] * (m_edge[e] for edge entries) * m_group[g];
    X_m = Xref + M * (X - Xref). The masks keep the anomaly score while a size and an
    entropy penalty push them toward a small, near-binary explanation.
    """
    name = "gnnexplainer"

    def __init__(self, diagnoser, steps=150, lr=0.1, size_weight=0.05, entropy_weight=0.05, init=1.0):
        self.init = init
        self.diag, self.steps, self.lr = diagnoser, steps, lr
        self.size_w, self.ent_w = size_weight, entropy_weight
        cfg = diagnoser.cfg
        self.groups = list(cfg.groups)
        node_group = np.zeros(cfg.dn, np.int64)
        edge_group = np.zeros(cfg.de, np.int64)
        for k, g in enumerate(self.groups):
            node_group[cfg.groups[g][0]] = k
            edge_group[cfg.groups[g][1]] = k
        self.node_group = torch.from_numpy(node_group)
        self.edge_group = torch.from_numpy(edge_group)
        self.owner = torch.tensor([cfg.edge_owner_index(e) for e in range(cfg.E)])

    def localize(self, windows) -> Localization:
        d, cfg = self.diag, self.diag.cfg
        w = np.asarray(windows)
        xn, xe = d.data.tensors(w[None], d.device)
        rn, re = d.data.tensors(d.reference.query(w[None]), d.device)
        with torch.no_grad():
            s0 = float(d.model.forward_detailed(xn, xe)["score"][0])
        logits = [torch.full((n,), float(self.init), device=d.device, requires_grad=True)
                  for n in (cfg.seq_len, cfg.V, cfg.E, len(self.groups))]
        opt = torch.optim.Adam(logits, lr=self.lr)
        ng, eg, owner = self.node_group.to(d.device), self.edge_group.to(d.device), self.owner.to(d.device)
        for _ in range(self.steps):
            mw, mv, me, mg = [torch.sigmoid(l) for l in logits]
            mask_n = mw[:, None, None] * mv[None, :, None] * mg[ng][None, None, :]
            mask_e = mw[:, None, None] * mv[owner][None, :, None] * me[None, :, None] * mg[eg][None, None, :]
            s = d.model.forward_detailed(rn + mask_n * (xn - rn), re + mask_e * (xe - re))["score"][0]
            masks = torch.cat([mw, mv, me, mg])
            entropy = -(masks * torch.log(masks + 1e-8) + (1 - masks) * torch.log(1 - masks + 1e-8))
            loss = ((s0 - s) / (s0 + 1e-12)) ** 2 + self.size_w * masks.mean() + self.ent_w * entropy.mean()
            grads = torch.autograd.grad(loss, logits)
            opt.zero_grad()
            for l, g in zip(logits, grads):
                l.grad = g
            opt.step()
        mw, mv, me, mg = [torch.sigmoid(l).detach().cpu().numpy() for l in logits]
        order = np.argsort(-mv)
        device = cfg.devices[int(order[0])]
        real = real_groups(self.groups)
        protocol = self.groups[real[int(np.argmax(mg[real]))]]
        if cfg.group_kind(protocol) == "node":
            flow = cfg.node_stream(device, protocol)
        else:
            cand = cfg.incident_edges(device) or list(range(cfg.E))
            flow = cfg.edges[int(cand[int(np.argmax(me[cand]))])].name
        return Localization(int(np.argmax(mw)), [cfg.devices[i] for i in order], protocol, flow, mw,
                            {"masks": {"window": mw.tolist(), "device": mv.tolist(),
                                       "edge": me.tolist(), "group": dict(zip(self.groups, mg.tolist()))}})


def make_baselines(data, cfg, diagnoser, quick=False) -> List:
    d = cfg.diagnosis
    return [RawZScore(data, cfg),
            RawZMax(data, cfg),
            AttentionL2(diagnoser),
            IntegratedGradients(diagnoser, steps=8 if quick else d["ig_steps"]),
            MaskExplainer(diagnoser, steps=40 if quick else d["mask_steps"], lr=d["mask_lr"],
                          size_weight=d["mask_size_weight"], entropy_weight=d["mask_entropy_weight"],
                          init=d.get("mask_init", 1.0))]
