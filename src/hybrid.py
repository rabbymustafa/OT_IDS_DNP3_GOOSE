"""Hybrid detector on the test split with measured rule alerts (no labels stand in for rules).

    python src/hybrid.py

Reuses the frozen Stage 1 checkpoints and their validation thresholds, and the frozen rules in
src/rules.py. Nothing is fitted here. The learned-branch numbers are recomputed and checked
against results/stage1/seed<k>_test.json so the fusion uses exactly the reported models.
"""
import json

import numpy as np
import pandas as pd
import torch

from config import ROOT
from model import ReconstructionModel
from rules import sequence_flags, window_violations
from train import MODEL_DIR, RESULT_DIR, SEEDS, attack_events, evaluate, load_split, prf, scores

OUT = ROOT / "results" / "rules"


def covered(events, flagged_windows):
    return sum(bool(w & flagged_windows) for _, w in events)


def main():
    test = load_split("test")
    label = test["label"]
    rule = sequence_flags(test, window_violations())
    events = attack_events(test)
    rule_windows = set(np.unique(test["windows"][rule]))
    rows = []
    for seed in SEEDS:
        tau = json.loads((RESULT_DIR / f"seed{seed}_val.json").read_text())["tau"]
        model = ReconstructionModel()
        model.load_state_dict(torch.load(MODEL_DIR / f"stage1_seed{seed}.pt"))
        learned = scores(model, test) > tau
        reported = json.loads((RESULT_DIR / f"seed{seed}_test.json").read_text())["test"]["learned"]
        assert abs(prf(learned[label != 1], label[label != 1] == 2)["f1"] - reported["f1"]) < 1e-9
        hybrid = learned | rule
        lw, hw = set(np.unique(test["windows"][learned])), set(np.unique(test["windows"][hybrid]))
        rows.append({
            "seed": seed,
            "rule_only": prf(rule, label != 0),
            "learned_only_all_attacks": prf(learned, label != 0),
            "hybrid_measured": prf(hybrid, label != 0),
            "attack_sequences_detected": {"rule": int((rule & (label != 0)).sum()), "learned": int((learned & (label != 0)).sum()),
                                          "hybrid": int((hybrid & (label != 0)).sum()), "total": int((label != 0).sum())},
            "events_covered": {"rule": covered(events, rule_windows), "learned": covered(events, lw),
                               "hybrid": covered(events, hw), "total": len(events)},
            "false_alarms": {"rule": int((rule & (label == 0)).sum()), "learned": int((learned & (label == 0)).sum()),
                             "hybrid": int((hybrid & (label == 0)).sum()), "normal": int((label == 0).sum())},
        })
        print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for key in ["rule_only", "learned_only_all_attacks", "hybrid_measured"]:
        summary[key] = {m: [float(np.mean([r[key][m] for r in rows])), float(np.std([r[key][m] for r in rows]))]
                        for m in ["precision", "recall", "f1"]}
    for key in ["attack_sequences_detected", "events_covered", "false_alarms"]:
        summary[key] = {k: [float(np.mean([r[key][k] for r in rows])), int(min(r[key][k] for r in rows)), int(max(r[key][k] for r in rows))]
                        for k in rows[0][key]}
    print(json.dumps(summary, indent=1))
    (OUT / "hybrid_test.json").write_text(json.dumps({"per_seed": rows, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
