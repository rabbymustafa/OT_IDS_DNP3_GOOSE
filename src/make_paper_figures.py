"""Figures and remaining Stage 1 numbers for the paper, from the frozen models. Nothing is tuned here.

    python src/make_paper_figures.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ROOT
from model import ReconstructionModel
from train import load_split

SEEDS = (42, 1, 2, 3, 4)
FIGS = ROOT / "paper" / "figs"
INK, MUTED, GRID = "#1f1e1b", "#6b6a64", "#e6e5e0"
NORMAL, RULE, DYNAMIC = "#8a8984", "#eb6834", "#2a78d6"
CLASSES = {"D2": "D2", "D3": "D3", "G2": "G2", "G4": "G4"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42,
})


def load(seed):
    m = ReconstructionModel()
    m.load_state_dict(torch.load(ROOT / "models" / f"stage1_seed{seed}.pt"))
    m.eval()
    tau = json.loads((ROOT / "results" / "stage1" / f"seed{seed}_test.json").read_text())["tau"]
    return m, tau


@torch.no_grad()
def forward(m, split):
    out = m.forward_detailed(split["node_x"], split["edge_x"])
    return out["score"].numpy(), out["alpha"].numpy()


test = load_split("test")
label = test["label"]
summary = {"sequences": {"normal": int((label == 0).sum()), "deterministic": int((label == 1).sum()), "dynamic": int((label == 2).sum())}}
cov, alpha_all = [], []
for seed in SEEDS:
    m, tau = load(seed)
    s, a = forward(m, test)
    flagged = s > tau
    cov.append({"learned_all_attacks": int(flagged[label > 0].sum()),
                "learned_on_deterministic": int(flagged[label == 1].sum()),
                "hybrid": int((flagged | (label == 1))[label > 0].sum())})
    alpha_all.append(a[flagged & (label == 2)])
    if seed == 42:
        s42, a42, tau42, f42 = s, a, tau, flagged
summary["coverage"] = cov
alpha = np.concatenate(alpha_all)
summary["attention_on_true_alerts"] = {"min": float(alpha.min()), "max": float(alpha.max()), "mean_of_max": float(alpha.max(1).mean())}
print(json.dumps(summary, indent=1))
(ROOT / "results" / "stage1" / "paper_numbers.json").write_text(json.dumps(summary, indent=2))

# (a) scores, seed 42: magnitude over sequence index, three identities, one threshold
fig, ax = plt.subplots(figsize=(3.45, 2.0))
x = np.arange(len(s42))
for lab, color, name, size, mark in ((0, NORMAL, "Normal", 5, "o"), (1, RULE, "Deterministic (rule path)", 12, "s"), (2, DYNAMIC, "Dynamic (learned path)", 12, "o")):
    k = label == lab
    ax.scatter(x[k], s42[k], s=size, c=color, marker=mark, linewidths=0.3, edgecolors="white", label=name, zorder=2 + lab)
ax.axhline(tau42, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(len(x) - 1, tau42 * 1.25, r"$\tau$", ha="right", va="bottom", color=INK)
ax.set_yscale("log")
ax.set_xlabel("Test sequence (time order)")
ax.set_ylabel(r"Score $S(X)$")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.46, 1.0), handletextpad=0.2, columnspacing=0.8)
fig.savefig(FIGS / "results_scores.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIGS / "results_scores.png", dpi=200, bbox_inches="tight", pad_inches=0.02)

# (b) attention, seed 42: true learned-path alerts, sequential single hue
k = np.flatnonzero(f42 & (label == 2))
fig, ax = plt.subplots(figsize=(3.45, 0.95))
im = ax.imshow(a42[k].T, aspect="auto", cmap="Blues", vmin=0.2, vmax=0.5, interpolation="nearest")
ax.set_yticks([0, 1, 2], [r"$i=1$", r"$i=2$", r"$i=3$"])
ax.set_xlabel("True learned-path alert (time order)")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
bar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
bar.set_label(r"$\alpha_i$")
bar.outline.set_visible(False)
fig.savefig(FIGS / "results_attention.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIGS / "results_attention.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote figures")
