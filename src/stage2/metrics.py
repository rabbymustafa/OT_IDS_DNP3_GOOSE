"""Localization, onset and calibration metrics, aggregation over seeds, and LaTeX output."""
from dataclasses import dataclass
from math import comb
from typing import Dict, List, Optional

import numpy as np

METRICS = ["window_acc", "onset_err_s", "device_top1", "device_top2", "protocol_acc", "flow_acc"]

TABLE_ROWS = [
    ("chance", "Chance"),
    ("raw_z", r"Raw $z$-score (mean)"),
    ("raw_z_max", r"Raw $z$-score (max)"),
    ("attention_l2", r"Attention + L2"),
    ("integrated_gradients", r"Integr.\ gradients"),
    ("gnnexplainer", r"GNNExplainer"),
    ("proposed", r"\textbf{Proposed}"),
]


@dataclass
class GroundTruth:
    run: int
    attack_class: str
    anom: np.ndarray        # [T]
    devices: List[str]
    protocols: List[str]
    flows: List[str]


def ground_truths(seqs, idx, runs) -> List[GroundTruth]:
    out = []
    for i in idx:
        run = runs[seqs.run[i]]
        out.append(GroundTruth(int(seqs.run[i]), run.attack_class, seqs.anom[i].copy(),
                               run.devices, run.protocols, run.flows))
    return out


def _mean(values) -> float:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


def localization_metrics(locs, gts, cfg) -> Dict[str, float]:
    """Window accuracy is computed on partially anomalous sequences, where the choice is informative."""
    T = cfg.seq_len
    partial = [i for i, g in enumerate(gts) if 0 < g.anom.sum() < T]
    return {
        "n_alerts": len(gts),
        "n_partial": len(partial),
        "window_acc": _mean([float(gts[i].anom[locs[i].t_star]) for i in partial]),
        "device_top1": _mean([float(l.device_rank[0] in g.devices) if g.devices else None
                              for l, g in zip(locs, gts)]),
        "device_top2": _mean([float(any(d in g.devices for d in l.device_rank[:2])) if g.devices else None
                              for l, g in zip(locs, gts)]),
        "protocol_acc": _mean([float(l.protocol in g.protocols) if (l.protocol and g.protocols) else None
                               for l, g in zip(locs, gts)]),
        "flow_acc": _mean([float(l.flow in g.flows) if (l.flow and g.flows) else None
                           for l, g in zip(locs, gts)]),
    }


def chance_metrics(gts, cfg) -> Dict[str, float]:
    """Expected accuracy of uniform guessing over windows, devices, protocol groups and flows."""
    T, V = cfg.seq_len, cfg.V
    protocols = cfg.protocol_names
    flows = cfg.flow_names()
    partial = [g for g in gts if 0 < g.anom.sum() < T]

    def top2(a):
        return 1 - comb(V - a, 2) / comb(V, 2) if V >= 2 else 1.0

    return {
        "n_alerts": len(gts), "n_partial": len(partial),
        "window_acc": _mean([g.anom.sum() / T for g in partial]),
        "onset_err_s": float("nan"),
        "device_top1": _mean([len(set(g.devices) & set(cfg.devices)) / V if g.devices else None for g in gts]),
        "device_top2": _mean([top2(len(set(g.devices) & set(cfg.devices))) if g.devices else None for g in gts]),
        "protocol_acc": _mean([len(set(g.protocols) & set(protocols)) / len(protocols) if g.protocols else None
                               for g in gts]),
        "flow_acc": _mean([len(set(g.flows) & set(flows)) / len(flows) if g.flows else None for g in gts]),
    }


# ------------------------------------------------------------------- onset
def onset_votes(alert_windows, locs, alert_runs) -> Dict[int, Dict[int, float]]:
    """u(w) = sum over alerted sequences containing w of the window's share of importance."""
    votes: Dict[int, Dict[int, float]] = {}
    for windows, loc, run in zip(alert_windows, locs, alert_runs):
        imp = np.clip(np.asarray(loc.window_importance, float), 0, None)
        share = imp / imp.sum() if imp.sum() > 0 else np.full(len(imp), 1.0 / len(imp))
        bucket = votes.setdefault(int(run), {})
        for w, s in zip(windows, share):
            bucket[int(w)] = bucket.get(int(w), 0.0) + float(s)
    return votes


def onset_errors(data, votes, eta) -> Dict[int, float]:
    errors = {}
    for run, bucket in votes.items():
        first = data.run_first_window(run)
        if first is None or not bucket:
            continue
        hits = sorted(w for w, u in bucket.items() if u >= eta)
        w_hat = hits[0] if hits else max(bucket, key=bucket.get)
        errors[run] = abs(data.window_start[w_hat] - data.window_start[first])
    return errors


def tune_eta(data, votes, grid) -> float:
    if not votes:
        return float(grid[len(grid) // 2])
    scores = [np.mean(list(onset_errors(data, votes, eta).values()) or [np.inf]) for eta in grid]
    return float(grid[int(np.argmin(scores))])


# ------------------------------------------------------------- calibration
def calibration_error(confidence, correct, bins=10) -> Dict[str, float]:
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, float)
    if len(confidence) == 0:
        return {"ece": float("nan"), "brier": float("nan")}
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if m.any():
            ece += m.mean() * abs(confidence[m].mean() - correct[m].mean())
    return {"ece": float(ece), "brier": float(np.mean((confidence - correct) ** 2))}


# ---------------------------------------------------------------- outputs
def aggregate(per_seed: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """method -> metric -> {mean, std, n} over seeds (NaNs ignored)."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in per_seed[0]:
        out[method] = {}
        for metric in per_seed[0][method]:
            vals = np.array([s[method].get(metric, np.nan) for s in per_seed], float)
            vals = vals[~np.isnan(vals)]
            out[method][metric] = {"mean": float(vals.mean()) if len(vals) else float("nan"),
                                   "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                                   "n": int(len(vals))}
    return out


def _fmt(stat: Optional[Dict[str, float]], digits: int, with_std: bool) -> str:
    if stat is None or np.isnan(stat["mean"]):
        return "--"
    if with_std and stat["n"] > 1:
        return f"{stat['mean']:.{digits}f}$\\pm${stat['std']:.{digits}f}"
    return f"{stat['mean']:.{digits}f}"


def latex_rows(agg, with_std=False) -> str:
    """Rows for Table V of the paper, in its column order."""
    lines = []
    for key, label in TABLE_ROWS:
        if key not in agg:
            continue
        cells = [_fmt(agg[key].get(m), 1 if m == "onset_err_s" else 2, with_std) for m in METRICS]
        lines.append(" & ".join([label] + cells) + r"\\")
    return "\n".join(lines) + "\n"
