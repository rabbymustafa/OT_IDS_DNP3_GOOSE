"""Rule-based detector: window-level protocol invariants for DNP3 and IEC 61850 GOOSE.

    python src/rules.py                  # training and validation only
    python src/rules.py --evaluate-test  # the test split, once, after the rules are frozen

Rules come from protocol semantics, not from labels. They run on the unscaled window tables
(data/processed/edge_features.csv and node_features.csv). A sequence is flagged when any of
its windows violates a rule, and each violation keeps the edge or node where it occurred, which
is the rule-path diagnosis.

    D-REQ   an outstation never sends a request function code to the master
    D-RESP  the master never sends a response function code to an outstation
    D-IIN   outstation internal indications (IIN) do not change during normal polling
    G-CONF  a GOOSE publisher's confRev does not change during operation
    G-SEQ   a GOOSE stNum change comes with an sqNum reset, and an sqNum reset comes with an
            stNum change, within the same window (the new state is sent with sqNum 0)
"""
import argparse
import json

import numpy as np
import pandas as pd

from config import EDGE_TABLE, NODE_TABLE, ROOT
from train import attack_events, load_split, prf

OUT = ROOT / "results" / "rules"
MASTER = 1
DEVICES = ["RTDS", "SCADA", "SEL_401_60", "SEL_421_71", "SEL_451_72", "SEL_451_73"]
REQUEST_FC = ["dnp3_F1_Read", "dnp3_F2_Write", "dnp3_F4_Operate", "dnp3_F5_DirectOperate",
              "dnp3_F6_DirectOperate_NoResp", "dnp3_F13_ColdRestart", "dnp3_F14_WarmRestart"]
RESPONSE_FC = ["dnp3_F129_Response", "dnp3_F130_UnsolicitedResponse"]
LINK_DEVICE = {"72": "SEL_451_72", "73": "SEL_451_73"}


def window_violations():
    """One row per (window, rule, place) where a rule is violated."""
    e = pd.read_csv(EDGE_TABLE)
    n = pd.read_csv(NODE_TABLE).sort_values(["node_id", "Window_Num"])
    rows = []

    def add(df, rule, kind):
        for r in df.itertuples():
            place = (f"{DEVICES[r.src_id]}->{DEVICES[r.dst_id]}" if kind == "edge" else DEVICES[r.node_id])
            device = (DEVICES[r.src_id] if r.dst_id == MASTER else DEVICES[r.dst_id]) if kind == "edge" else place
            rows.append({"Window_Num": int(r.Window_Num), "rule": rule, "kind": kind, "place": place, "device": device})

    up, down = e[e.dst_id == MASTER], e[e.src_id == MASTER]
    add(up[(up[REQUEST_FC] > 0).any(axis=1)], "D-REQ", "edge")
    add(down[(down[RESPONSE_FC] > 0).any(axis=1)], "D-RESP", "edge")
    add(up[up.dnp3_IIN_Val_Change > 0], "D-IIN", "edge")
    add(n[n.node_ConfRev_Change > 0], "G-CONF", "node")
    g = n.copy()
    st, rs = g.node_stNum_Change > 0, g.node_sqNum_Reset > 0
    add(g[st ^ rs], "G-SEQ", "node")
    return pd.DataFrame(rows, columns=["Window_Num", "rule", "kind", "place", "device"])


def sequence_flags(split, viol):
    flagged = set(viol.Window_Num)
    return np.array([any(w in flagged for w in ws) for ws in split["windows"].tolist()])


def evaluate(name, viol, meta):
    split = load_split(name)
    label = split["label"]
    r = sequence_flags(split, viol)
    windows = set(np.unique(split["windows"]))
    v = viol[viol.Window_Num.isin(windows)].merge(meta[["Window_Num", "attack_tag", "label"]], on="Window_Num")
    v["tag"] = v.attack_tag.fillna("normal")
    events = attack_events(split)
    det = [w for lab, w in events if lab == 1]
    flagged_windows = set(v.Window_Num)
    res = {
        "sequences": {"normal": int((label == 0).sum()), "deterministic": int((label == 1).sum()), "dynamic": int((label == 2).sum())},
        "rule_only": prf(r, label != 0),
        "deterministic_detected": int((r & (label == 1)).sum()),
        "dynamic_flagged": int((r & (label == 2)).sum()),
        "normal_flagged": int((r & (label == 0)).sum()),
        "deterministic_events_covered": sum(bool(w & flagged_windows) for w in det), "deterministic_events": len(det),
        "violations_by_rule_and_tag": v.groupby(["rule", "tag"]).size().unstack(fill_value=0).to_dict(),
    }
    # Rule-path diagnosis for DNP3 events, against the link named in the capture notes.
    diag = []
    for _, w in events:
        w = sorted(w)
        m = meta[meta.Window_Num.isin(w)]
        if m.label.iloc[0] != 1 or m.attack_tag.iloc[0] != "D1":
            continue
        links = str(m.links_in_notes.iloc[0]).split(";")
        truth = {LINK_DEVICE[l.split("-")[0]] for l in links if l.split("-")[0] in LINK_DEVICE}
        hits = v[v.Window_Num.isin(w)]
        if len(hits):
            pred = hits.device.value_counts().index[0]
            diag.append({"event": int(w[0]), "true": sorted(truth), "pred": pred, "correct": pred in truth,
                         "rules": sorted(set(hits.rule))})
    res["d1_rule_path_diagnosis"] = diag
    return res, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data" / "processed" / "window_metadata.csv")
    viol = window_violations()
    splits = ["test"] if args.evaluate_test else ["train", "val"]
    out = {}
    for s in splits:
        out[s], _ = evaluate(s, viol, meta)
        m = out[s]
        print(f"{s}: deterministic {m['deterministic_detected']}/{m['sequences']['deterministic']} sequences, "
              f"events {m['deterministic_events_covered']}/{m['deterministic_events']}, "
              f"normal flagged {m['normal_flagged']}/{m['sequences']['normal']}, dynamic flagged {m['dynamic_flagged']}, "
              f"D1 device {sum(d['correct'] for d in m['d1_rule_path_diagnosis'])}/{len(m['d1_rule_path_diagnosis'])}", flush=True)
        print("   ", json.dumps(m["violations_by_rule_and_tag"]), flush=True)
    (OUT / ("test.json" if args.evaluate_test else "dev.json")).write_text(json.dumps(out, indent=2, default=int))


if __name__ == "__main__":
    main()
