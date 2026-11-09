"""Stage 2 diagnosis on the saved Stage 1 models, with the thresholds fixed on validation.

    python src/stage2_run.py                 # validation and training events only
    python src/stage2_run.py --final-test    # also the test events; run once, after the settings are frozen

Validation events tune the onset threshold and the confidence calibration (docs/evaluation_rules.md).
Training events are not used for tuning. The model never trained on attack windows.
Outputs: results/stage2/{dev,final}_results.json, {dev,final}_signatures.csv
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import ROOT
from model import ReconstructionModel
from stage2.baselines import make_baselines
from stage2.config import Config
from stage2.data import OTData
from stage2.diagnosis import Stage2Diagnoser
from stage2.metrics import (aggregate, calibration_error, chance_metrics, ground_truths, latex_rows,
                            localization_metrics, onset_errors, onset_votes, tune_eta)

SEEDS = (42, 1, 2, 3, 4)
INPUTS = ROOT / "results" / "stage2_inputs"
OUT = ROOT / "results" / "stage2"
STAGE1 = ROOT / "results" / "stage1"
CLASSES = ("D2", "D3", "G2", "G4")
FAMILY = {"D2": 0, "D3": 0, "G2": 1, "G4": 1}  # rows of the Stage 3 table


def load_stage3():
    spec = importlib.util.spec_from_file_location("stage3", Path(__file__).parent / "stage3_resilience_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def score(model, data, windows, batch=256):
    model.eval()
    out = []
    for b in range(0, len(windows), batch):
        out.append(model(*data.tensors(windows[b:b + batch])).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def alerts_of(seqs, s, tau, runs):
    """True-positive learned-path alerts: flagged, no rule flag, belongs to a dynamic event."""
    keep = (s > tau) & ~seqs.rule & (seqs.run >= 0)
    return np.array([i for i in np.flatnonzero(keep) if runs[seqs.run[i]].path == "gnn"], dtype=int)


def expected_rrs(gts, locs):
    stage3 = load_stage3()
    rows = stage3.evaluate(stage3.WEIGHT_SETS["AHP baseline"])
    out = {}
    for fam, row in enumerate(rows):
        sel = [i for i, g in enumerate(gts) if FAMILY.get(g.attack_class) == fam and g.flows]
        if not sel:
            continue
        p = float(np.mean([locs[i].flow in gts[i].flows for i in sel]))
        out[row["family"].replace("\n", " ")] = {
            "n_alerts": len(sel), "flow_accuracy": p,
            "expected_rrs": p * row["Flow-level (proposed)"] + (1 - p) * row["wrong"],
            "best_alternative_rrs": row["best_alt"], "break_even_p": row["p_star"]}
    return out


def per_class(locs, gts, cfg):
    out = {}
    for c in CLASSES:
        sel = [i for i, g in enumerate(gts) if g.attack_class == c]
        if sel:
            out[c] = localization_metrics([locs[i] for i in sel], [gts[i] for i in sel], cfg)
    return out


def checks(prop, gts, diag):
    def frac(values):
        values = [float(v) for v in values if v is not None]
        return float(np.mean(values)) if values else float("nan")

    with_key = [i for i, g in enumerate(gts) if g.devices]
    correct = [float(prop[i].device_rank[0] in gts[i].devices) for i in with_key]
    l2 = [float(prop[i].extras["device_rank_l2"][0] in gts[i].devices) for i in with_key]
    wh = [float(prop[i].extras["device_rank_whitened"][0] in gts[i].devices) for i in with_key]
    return {
        "device_top1_whitened": frac(wh), "device_top1_l2": frac(l2),
        "protocol_acc_shapley": frac([l.extras["protocol_shapley"] in g.protocols
                                      for l, g in zip(prop, gts) if g.protocols]),
        "deletion_drop_mean": frac([l.extras["deletion_drop"] for l in prop]),
        "agree_whitened": frac([l.device_rank[0] == l.extras["device_rank_whitened"][0] for l in prop]),
        "agree_attention_occlusion": frac([l.extras["agree_time"] for l in prop]),
        "efficiency_error_max": max([l.extras["efficiency_error"] for l in prop], default=float("nan")),
        "full_coalition_error_max": max([l.extras["full_coalition_error"] for l in prop], default=float("nan")),
        "latency_ms": frac([l.extras["latency_s"] * 1e3 for l in prop]),
        **{f"calibration_{k}": v for k, v in
           calibration_error([diag.confidence(prop[i]) for i in with_key], correct).items()},
    }


def signature_rows(seed, part, method, seqs, idx, locs, gts, runs, diag=None):
    return [{
        "seed": seed, "partition": part, "method": method, "run_id": runs[g.run].run_id,
        "attack_class": g.attack_class, "windows": " ".join(map(str, seqs.windows[i])),
        "anomalous_windows": " ".join(str(int(a)) for a in g.anom), "t_star": loc.t_star,
        "device": loc.device_rank[0], "device_2": loc.device_rank[1], "true_devices": "|".join(g.devices),
        "protocol": loc.protocol, "true_protocols": "|".join(g.protocols), "flow": loc.flow,
        "true_flows": "|".join(g.flows), "mechanism": loc.extras.get("mechanism"),
        "kappa_v": loc.extras.get("kappa_v"), "confidence": diag.confidence(loc) if diag else None,
        "score": loc.extras.get("score"),
    } for i, loc, g in zip(idx, locs, gts)]


def run_seed(cfg, data, seed, parts):
    model = ReconstructionModel()
    model.load_state_dict(torch.load(ROOT / "models" / f"stage1_seed{seed}.pt"))
    tau = json.loads((STAGE1 / f"seed{seed}_val.json").read_text())["tau"]
    seqs = {p: data.seqset(p) for p in {"train", "val", *parts}}
    train, val = seqs["train"], seqs["val"]
    diag = Stage2Diagnoser(model, data, cfg).fit(train.windows[~train.label], val.windows[~val.label])
    methods = [diag] + make_baselines(data, cfg, diag)
    alerts = {p: alerts_of(seqs[p], score(model, data, seqs[p].windows), tau, data.runs) for p in seqs}
    gts = {p: ground_truths(seqs[p], alerts[p], data.runs) for p in seqs}

    result = {"seed": seed, "tau": tau, "n_alerts": {p: int(len(a)) for p, a in alerts.items()}, "partitions": {}}
    rows = []
    for p in parts:
        result["partitions"][p] = {"localization": {"chance": chance_metrics(gts[p], cfg)}, "per_class": {}}
    prop = {}
    for method in methods:
        start = time.perf_counter()
        locs = {p: [method.localize(seqs[p].windows[i]) for i in alerts[p]] for p in seqs}
        ms = (time.perf_counter() - start) / max(sum(len(a) for a in alerts.values()), 1) * 1e3
        eta = tune_eta(data, onset_votes(val.windows[alerts["val"]], locs["val"], val.run[alerts["val"]]),
                       cfg.diagnosis["onset_eta_grid"])
        for p in parts:
            s = seqs[p]
            errors = onset_errors(data, onset_votes(s.windows[alerts[p]], locs[p], s.run[alerts[p]]), eta)
            res = localization_metrics(locs[p], gts[p], cfg)
            res.update(onset_err_s=float(np.mean(list(errors.values()))) if errors else float("nan"),
                       onset_eta=eta, ms_per_alert=ms)
            result["partitions"][p]["localization"][method.name] = res
            result["partitions"][p]["per_class"][method.name] = per_class(locs[p], gts[p], cfg)
            if method is not diag:
                rows += signature_rows(seed, p, method.name, s, alerts[p], locs[p], gts[p], data.runs)
        if method is diag:
            prop = locs

    with_key = [(l, g) for l, g in zip(prop["val"], gts["val"]) if g.devices]
    if with_key:
        diag.calibrate([l for l, _ in with_key], [float(l.device_rank[0] in g.devices) for l, g in with_key])
    for p in parts:
        result["partitions"][p]["checks"] = checks(prop[p], gts[p], diag)
        result["partitions"][p]["expected_rrs"] = expected_rrs(gts[p], prop[p])
        rows += signature_rows(seed, p, diag.name, seqs[p], alerts[p], prop[p], gts[p], data.runs, diag)
    return result, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    parts = ["val", "train"] + (["test"] if args.final_test else [])
    prefix = "final" if args.final_test else "dev"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(INPUTS / "config.json")
    data = OTData(cfg, INPUTS / "windows.npz", INPUTS / "runs.csv")

    per_seed, rows = [], []
    for seed in SEEDS:
        res, r = run_seed(cfg, data, seed, parts)
        per_seed.append(res)
        rows += r
        for p in parts:
            m = res["partitions"][p]["localization"]["proposed"]
            print(f"seed {seed} {p}: alerts {m['n_alerts']} device {m['device_top1']:.2f} protocol {m['protocol_acc']:.2f} "
                  f"flow {m['flow_acc']:.2f} window {m['window_acc']:.2f} onset {m['onset_err_s']:.1f}s", flush=True)

    summary = {p: aggregate([r["partitions"][p]["localization"] for r in per_seed]) for p in parts}
    (OUT / f"{prefix}_results.json").write_text(json.dumps({"summary": summary, "per_seed": per_seed}, indent=2, default=float))
    pd.DataFrame(rows).to_csv(OUT / f"{prefix}_signatures.csv", index=False)
    for p in parts:
        (OUT / f"{prefix}_table_{p}.tex").write_text(latex_rows(summary[p], with_std=True), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
