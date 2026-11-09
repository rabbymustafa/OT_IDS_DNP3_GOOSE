"""Stage 1: train the reconstruction model and choose the threshold on validation.

    python src/train.py                  # train every seed, report validation only
    python src/train.py --evaluate-test  # score the test split with the saved models and thresholds

Test sequences are not touched unless --evaluate-test is given (docs/evaluation_rules.md).
Outputs: models/stage1_seed<k>.pt, results/stage1/*.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import ROOT, SEQUENCE_DIR
from model import ReconstructionModel

SEEDS = (42, 1, 2, 3, 4)  # 42 matches the original notebook
EPOCHS = 20
LR = 1e-3
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "stage1"


def load_split(name):
    seqs = torch.load(SEQUENCE_DIR / f"{name}_sequences.pt", weights_only=False)
    return {
        "node_x": torch.stack([torch.stack([g.x for g in s["graphs"]]) for s in seqs]),
        "edge_x": torch.stack([torch.stack([g.edge_attr for g in s["graphs"]]) for s in seqs]),
        "label": np.array([s["sequence_label"] for s in seqs]),
        "graph_labels": np.array([s["graph_labels"] for s in seqs]),
        "windows": np.array([s["windows"] for s in seqs]),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def scores(model, split, batch_size=256):
    model.eval()
    out = [model(split["node_x"][i:i + batch_size], split["edge_x"][i:i + batch_size])
           for i in range(0, len(split["label"]), batch_size)]
    return torch.cat(out).numpy()


def train_model(train, val, seed):
    """One sequence per optimizer step, normal sequences only."""
    set_seed(seed)
    model = ReconstructionModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    normal = np.flatnonzero(train["label"] == 0)
    val_normal = {k: (v[val["label"] == 0]) for k, v in val.items()}
    history = []
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for i in np.random.permutation(normal):
            loss = model(train["node_x"][i:i + 1], train["edge_x"][i:i + 1]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        history.append({"epoch": epoch + 1, "train_loss": total / len(normal),
                        "val_normal_loss": float(scores(model, val_normal).mean())})
    return model, history


def prf(pred, positive):
    tp = int(np.sum(pred & positive))
    fp = int(np.sum(pred & ~positive))
    fn = int(np.sum(~pred & positive))
    tn = int(np.sum(~pred & ~positive))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_threshold(s, label):
    """Best validation F1 over quantile candidates; ties go to precision, then recall."""
    keep = label != 1  # the rule branch owns label 1 sequences
    s, positive = s[keep], label[keep] == 2
    candidates = np.unique(np.concatenate([np.quantile(s[~positive], np.linspace(0.90, 1.00, 101)),
                                           np.quantile(s, np.linspace(0.80, 1.00, 101))]))
    best_key, best_tau = None, None
    for tau in candidates:
        m = prf(s > tau, positive)
        key = (m["f1"], m["precision"], m["recall"])
        if best_key is None or key > best_key:
            best_key, best_tau = key, float(tau)
    return best_tau


def attack_events(split):
    """Contiguous attacked windows with the same window label: [(label, set of windows)]."""
    label_of = {}
    for windows, labels in zip(split["windows"], split["graph_labels"]):
        label_of.update(zip(windows.tolist(), labels.tolist()))
    events = []
    for w in sorted(label_of):
        if label_of[w] == 0:
            continue
        if events and events[-1][0] == label_of[w] and w - 1 in events[-1][1]:
            events[-1][1].add(w)
        else:
            events.append((label_of[w], {w}))
    return events


def evaluate(split, s, tau):
    """Learned branch (label 1 excluded), and the hybrid with label 1 standing in for rule alerts."""
    label, flagged = split["label"], s > tau
    keep = label != 1
    events = attack_events(split)
    dynamic = [w for lab, w in events if lab == 2]
    seq_windows = [set(x) for x in split["windows"][flagged].tolist()]
    covered = sum(any(w & f for f in seq_windows) for w in dynamic)
    hybrid = flagged | (label == 1)
    n_rule = len(events) - len(dynamic)
    return {
        "learned": prf(flagged[keep], label[keep] == 2),
        "hybrid_rule_assumed": prf(hybrid, label != 0),
        "dynamic_events_covered": covered, "dynamic_events": len(dynamic),
        "all_events_covered_rule_assumed": covered + n_rule, "all_events": len(events),
    }


def latency_ms(model, split, n=300):
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(min(n, len(split["label"]))):
            start = time.perf_counter()
            model(split["node_x"][i:i + 1], split["edge_x"][i:i + 1])
            times.append(time.perf_counter() - start)
    return float(np.median(times) * 1e3)


def run_training():
    train, val = load_split("train"), load_split("val")
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        start = time.perf_counter()
        model, history = train_model(train, val, seed)
        s = scores(model, val)
        tau = choose_threshold(s, val["label"])
        torch.save(model.state_dict(), MODEL_DIR / f"stage1_seed{seed}.pt")
        res = {"seed": seed, "tau": tau, "history": history, "val": evaluate(val, s, tau),
               "train_seconds": time.perf_counter() - start, "n_parameters": sum(p.numel() for p in model.parameters())}
        (RESULT_DIR / f"seed{seed}_val.json").write_text(json.dumps(res, indent=2))
        m = res["val"]["learned"]
        print(f"seed {seed}: val P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"events {res['val']['dynamic_events_covered']}/{res['val']['dynamic_events']} "
              f"loss {history[0]['val_normal_loss']:.2e}->{history[-1]['val_normal_loss']:.2e} "
              f"({res['train_seconds']:.0f} s)", flush=True)


def run_test():
    test = load_split("test")
    for seed in SEEDS:
        tau = json.loads((RESULT_DIR / f"seed{seed}_val.json").read_text())["tau"]
        model = ReconstructionModel()
        model.load_state_dict(torch.load(MODEL_DIR / f"stage1_seed{seed}.pt"))
        res = {"seed": seed, "tau": tau, "test": evaluate(test, scores(model, test), tau),
               "latency_ms_per_sequence": latency_ms(model, test)}
        (RESULT_DIR / f"seed{seed}_test.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluate-test", action="store_true")
    run_test() if ap.parse_args().evaluate_test else run_training()
