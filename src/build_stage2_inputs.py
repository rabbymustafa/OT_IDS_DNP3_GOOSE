"""Build the Stage 2 inputs from the repo's processed data, without changing anything in data/.

    python src/build_stage2_inputs.py

Outputs in results/stage2_inputs/:
    windows.npz   one row per 5 s window, in the repo's scaled feature space
    runs.csv      one row per attack event, with the answer key from docs/evaluation_rules.md
    config.json   devices, edges, feature order, protocol groups

Only feature tensors and window labels are read here. No model is scored.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (EDGE_FEATURE_COLS, NODE_FEATURE_COLS, ROOT, SEQUENCE_DIR, WINDOW_METADATA)

OUT = ROOT / "results" / "stage2_inputs"
DEVICES = ["RTDS", "SCADA", "SEL_401_60", "SEL_421_71", "SEL_451_72", "SEL_451_73"]
EDGES = [("SCADA", "SEL_451_72"), ("SCADA", "SEL_451_73"), ("SEL_451_72", "SCADA"), ("SEL_451_73", "SCADA")]
WINDOW_SECONDS = 5

# Answer key (docs/evaluation_rules.md). Empty fields are left out of the scoring.
# G2 and G4 devices stay empty until they come from the testbed records.
KEY = {
    "D2": ("SEL_451_72", "DNP3|TCP", "SCADA->SEL_451_72|SEL_451_72->SCADA"),
    "D3": ("SEL_451_73", "DNP3|TCP", "SCADA->SEL_451_73|SEL_451_73->SCADA"),
    "G2": ("", "GOOSE", ""),
    "G4": ("", "GOOSE", ""),
    "D1": ("", "", ""),
    "G1": ("", "", ""),
}
STREAM_DEVICE_FILE = OUT / "goose_stream_devices.json"  # written by hand once the mapping is known


def build_windows():
    rows = {}
    edge_index = None
    for split in ("train", "val", "test"):
        for s in torch.load(SEQUENCE_DIR / f"{split}_sequences.pt", weights_only=False):
            for w, g, lab in zip(s["windows"], s["graphs"], s["graph_labels"]):
                node, edge = g.x.numpy(), g.edge_attr.numpy()
                if edge_index is None:
                    edge_index = g.edge_index.numpy()
                if w in rows:
                    assert np.array_equal(rows[w][0], node) and np.array_equal(rows[w][1], edge), w
                rows[w] = (node, edge, split, int(lab))
    order = sorted(rows)
    assert order == list(range(order[0], order[0] + len(order))), "windows are not contiguous"
    return order, rows, edge_index


def build_runs(meta):
    a = meta[meta.label > 0].copy()
    a["event"] = (a.Window_Num.diff() != 1).cumsum()
    first = int(meta.Window_Num.min())
    stream_device = json.loads(STREAM_DEVICE_FILE.read_text()) if STREAM_DEVICE_FILE.exists() else {}
    out = []
    for _, g in a.groupby("event"):
        assert g.attack_tag.nunique() == 1 and g.label.nunique() == 1
        tag, start, end = g.attack_tag.iloc[0], int(g.Window_Num.min()), int(g.Window_Num.max())
        device, protocol, flow = KEY[tag]
        stream = g.goose_stream_ids_in_notes.iloc[0]
        if tag in ("G2", "G4") and isinstance(stream, str) and stream in stream_device:
            device = stream_device[stream]
            flow = f"{device}/GOOSE"
        out.append({"run_id": f"{tag}_{start}", "attack_class": tag, "target_device": device,
                    "protocol": protocol, "flow": flow, "onset": (start - first) * WINDOW_SECONDS,
                    "offset": (end + 1 - first) * WINDOW_SECONDS,
                    "path": "gnn" if g.label.iloc[0] == 2 else "rule", "split": g.split.iloc[0]})
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(WINDOW_METADATA)
    order, rows, edge_index = build_windows()
    assert order == meta.Window_Num.tolist()
    label = np.array([rows[w][3] for w in order])
    assert np.array_equal(label, meta.label.values), "sequence labels differ from window metadata"
    idx = {d: i for i, d in enumerate(DEVICES)}
    expected = np.array([[idx[a] for a, _ in EDGES], [idx[b] for _, b in EDGES]])
    assert np.array_equal(edge_index, expected), edge_index
    np.savez_compressed(
        OUT / "windows.npz",
        node_x=np.stack([rows[w][0] for w in order]), edge_x=np.stack([rows[w][1] for w in order]),
        window_start=(np.array(order) - order[0]).astype(float) * WINDOW_SECONDS,
        split=np.array([rows[w][2] for w in order]), rule_flag=(label == 1).astype(int),
        window_num=np.array(order), edge_index=edge_index)
    build_runs(meta).to_csv(OUT / "runs.csv", index=False)
    cfg = {
        "devices": DEVICES,
        "edges": [{"name": f"{a}->{b}", "src": a, "dst": b} for a, b in EDGES],
        "edge_owner": "dst",
        # Both directions of a link belong to its outstation: SCADA is the master on every link.
        "edge_owners": ["SEL_451_72", "SEL_451_73", "SEL_451_72", "SEL_451_73"],
        "node_features": NODE_FEATURE_COLS,
        "edge_features": EDGE_FEATURE_COLS,
        "window_seconds": WINDOW_SECONDS,
        "seq_len": 3,
        "protocol_groups": {"GOOSE": {"node": ["node_*"]}, "DNP3": {"edge": ["dnp3_*"]}, "TCP": {"edge": ["tcp_*"]}},
        "mechanism_tags": {
            "command": ["dnp3_F0_*", "dnp3_F1_*", "dnp3_F2_*", "dnp3_F4_*", "dnp3_F5_*", "dnp3_F6_*",
                        "dnp3_F13_*", "dnp3_F14_*", "dnp3_F129_*", "dnp3_F130_*"],
            "source_change": ["*_Port_Changed"],
            "timing_drop": ["tcp_Flag_Retransmission", "tcp_Flag_Reset", "dnp3_Num_Packets", "tcp_Num_Packets"],
            "rate": ["node_Num_Packets", "node_Avg_Length", "node_Num_Events", "dnp3_Avg_Length", "tcp_Avg_Length"],
            "state": ["node_stNum_Change", "node_sqNum_Reset", "node_ConfRev_Change", "node_Boolean_Data_Change",
                      "dnp3_IIN_Val_Change", "dnp3_Analog_Input_Change", "dnp3_Binary_Input_Change"],
            "session": ["tcp_Flag_Syn", "tcp_Flag_Fin", "tcp_Flag_Ack"],
        },
        "goose_publishers": [d for d in DEVICES if d != "SCADA"],
        "diagnosis": {"mask_size_weight": 0.5, "mask_init": 1.0},  # best of the validation grid
    }
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{len(order)} windows, {int((label > 0).sum())} attacked; wrote {Path(OUT)}")


if __name__ == "__main__":
    main()
