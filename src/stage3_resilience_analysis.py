"""Stage 3 analytical response-resilience study.

Evaluates the Response Resilience Score (RRS), a response-time adaptation of the
Unified Resiliency Score (URS) of Mustafa et al. (IEEE TIA 2026), for four response
policies on the bay-level communication model of the HIL substation testbed
(3 MU-IED bays on the process bus, 3 IED-SCADA DNP3 sessions on the station bus).

Everything here is model-based: the numbers follow from the testbed topology, the
URS weights reported in the TIA paper, and the action definitions in Section VI of
the manuscript. No measured Stage 2 data is used.

Run:  python stage3_resilience_analysis.py
Outputs: ../figs/stage3_resilience.pdf and a printed summary (also ../figs/stage3_results.csv)
"""
import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")

# ---------------------------------------------------------------- model inputs
BAYS = (1, 2, 3)
PROTOCOLS = ("GOOSE", "SV", "DNP3")
# Protocol criticality: GOOSE:DNP3 = 6:4 as in the TIA paper; SV takes the GOOSE
# weight because both carry protection-critical process-bus traffic.
W_PROTO = {"GOOSE": 6 / 16, "SV": 6 / 16, "DNP3": 4 / 16}
BUS = {"GOOSE": "process", "SV": "process", "DNP3": "station"}
# Paths are (protocol, bay). Each bay has one GOOSE exchange, one SV stream and one
# DNP3 session, so |P_p| = 3 for every protocol.
PATHS = [(p, k) for p in PROTOCOLS for k in BAYS]
N_PATHS = {p: sum(1 for q, _ in PATHS if q == p) for p in PROTOCOLS}

# URS weights (AHP baseline and the two sensitivity sets of the TIA paper).
WEIGHT_SETS = {
    "AHP baseline": (0.29, 0.42, 0.29),
    "Set A (protection-first)": (0.35, 0.45, 0.20),
    "Set B (containment)": (0.25, 0.40, 0.35),
}
# BMTCR of the full-traditional configuration: (3 SV + 6 GOOSE + 6 ARP) necessary
# links over (15 + 30 + 12) propagated links, from the TIA paper's path counts.
BMTCR = 15 / 57

# Devices own the paths that terminate on them.
DEVICE_PATHS = {
    **{f"IED{k}": {("GOOSE", k), ("SV", k), ("DNP3", k)} for k in BAYS},
    **{f"MU{k}": {("GOOSE", k), ("SV", k)} for k in BAYS},
}

# Attack families on the representative bay 2 (results are bay-symmetric).
FAMILIES = {
    "DNP3 session\n(D1, D2, D3)": dict(
        compromised={("DNP3", 2)}, devices=("IED2",)),
    "GOOSE exchange\n(G2, G3, G4)": dict(
        compromised={("GOOSE", 2)}, devices=("MU2",)),
    "SV time sync\n(S1)": dict(
        compromised={("SV", 1), ("SV", 2), ("SV", 3)}, devices=("MU1", "MU2", "MU3")),
}

POLICIES = ("No response", "Bus isolation", "Device isolation", "Flow-level (proposed)")


# ------------------------------------------------------------------ the metric
def rrs(compromised_active, unavailable, w):
    """RRS = w1*SAI + w2*(1 - ACR) + w3*BMTCR.

    compromised_active: paths still under adversarial control after the response.
    unavailable:        legitimate paths lost (blocked or untrusted).
    """
    w1, w2, w3 = w
    sai = sum(W_PROTO[p] * sum(1 for q, k in PATHS if q == p and (q, k) not in unavailable)
              / N_PATHS[p] for p in PROTOCOLS)
    acr = sum(W_PROTO[p] * sum(1 for q, _ in compromised_active if q == p) / N_PATHS[p]
              for p in PROTOCOLS)
    return w1 * sai + w2 * (1 - acr) + w3 * BMTCR


def outcome(policy, fam, correct=True):
    """Return (active compromised paths, unavailable paths) after a response."""
    comp = fam["compromised"]
    if policy == "No response":
        return comp, set(comp)
    if policy == "Bus isolation":
        buses = {BUS[p] for p, _ in comp}
        blocked = {(p, k) for p, k in PATHS if BUS[p] in buses}
        return set(), blocked
    if policy == "Device isolation":
        blocked = set().union(*(DEVICE_PATHS[d] for d in fam["devices"]))
        active = comp - blocked
        return active, blocked | active
    if policy == "Flow-level (proposed)":
        if correct:
            return set(), set()
        # Wrong localization: one legitimate path of the same protocol is blocked
        # and the attack persists.
        proto = next(iter(comp))[0]
        spare = sorted(pk for pk in PATHS if pk[0] == proto and pk not in comp)
        return comp, set(comp) | set(spare[:1])
    raise ValueError(policy)


def evaluate(w):
    rows = []
    for name, fam in FAMILIES.items():
        r = {pol: rrs(*outcome(pol, fam), w) for pol in POLICIES}
        r_wrong = rrs(*outcome("Flow-level (proposed)", fam, correct=False), w)
        best_alt = max(r[p] for p in POLICIES[:3])
        p_star = (best_alt - r_wrong) / (r["Flow-level (proposed)"] - r_wrong)
        rows.append(dict(family=name, **r, wrong=r_wrong, best_alt=best_alt,
                         p_star=max(0.0, p_star)))
    return rows


def time_average(level_attack, level_after, t_resp, horizon):
    """Mean RRS over [0, horizon] when the response lands at t_resp."""
    return (level_attack * t_resp + level_after * (horizon - t_resp)) / horizon


# ---------------------------------------------------------------------- report
def main():
    os.makedirs(FIGS, exist_ok=True)
    w = WEIGHT_SETS["AHP baseline"]
    normal = rrs(set(), set(), w)
    rows = evaluate(w)

    print(f"Normal-operation RRS (AHP weights): {normal:.3f}")
    for r in rows:
        fam = r["family"].replace("\n", " ")
        vals = ", ".join(f"{p}={r[p]:.3f}" for p in POLICIES)
        print(f"{fam}: {vals}; wrong-flow={r['wrong']:.3f}; break-even p*={r['p_star']:.2f}")

    # Ranking stability across the TIA paper's weight sets.
    for wname, ws in WEIGHT_SETS.items():
        for r in evaluate(ws):
            order = sorted(POLICIES, key=lambda p: -r[p])
            print(f"[{wname}] {r['family'].replace(chr(10), ' ')}: "
                  + " > ".join(order) + f" | p*={r['p_star']:.2f}")

    # Time-averaged RRS for a 60 s episode. Timing assumptions (stated in the paper):
    # detection at the end of the first anomalous 5-s window, 0.2 s diagnosis,
    # 0.05 s flow-rule installation.
    horizon, t_det, t_diag, t_act = 60.0, 5.0, 0.2, 0.05
    t_resp = {"No response": horizon, "Bus isolation": t_det + t_act,
              "Device isolation": t_det + t_diag + t_act,
              "Flow-level (proposed)": t_det + t_diag + t_act}
    with open(os.path.join(FIGS, "stage3_results.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["family"] + [f"RRS {p}" for p in POLICIES]
                    + ["RRS wrong flow", "break-even p*"]
                    + [f"mean RRS 60s {p}" for p in POLICIES])
        for r in rows:
            lvl0 = r["No response"]
            means = [time_average(lvl0, r[p], min(t_resp[p], horizon), horizon) for p in POLICIES]
            wr.writerow([r["family"].replace("\n", " ")]
                        + [f"{r[p]:.3f}" for p in POLICIES]
                        + [f"{r['wrong']:.3f}", f"{r['p_star']:.2f}"]
                        + [f"{m:.3f}" for m in means])
            print("60 s mean RRS", r["family"].replace("\n", " "),
                  ", ".join(f"{p}={m:.3f}" for p, m in zip(POLICIES, means)))

    plot(rows, normal, t_resp, horizon)


def plot(rows, normal, t_resp, horizon):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5,
        "axes.labelsize": 7.5, "legend.fontsize": 6.8, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42,
    })
    # Baseline in neutral ink; policies take categorical slots 1-3 in fixed order,
    # with line style / hatch as the secondary cue for print and CVD readers.
    style = {
        "No response": dict(color="#8a8984", ls=(0, (1.2, 1.2)), hatch=""),
        "Bus isolation": dict(color="#eb6834", ls=(0, (4, 1.5)), hatch="////"),
        "Device isolation": dict(color="#1baf7a", ls=(0, (6, 1.5, 1.2, 1.5)), hatch="...."),
        "Flow-level (proposed)": dict(color="#2a78d6", ls="-", hatch=""),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.7),
                                   gridspec_kw=dict(height_ratios=[1, 1.05], hspace=0.62))

    # (a) resilience curve for the DNP3-session family.
    r = rows[0]
    for pol in POLICIES:
        t_r = min(t_resp[pol], horizon)
        ts = [-10, 0, 0, t_r, t_r, horizon]
        ys = [normal, normal, r["No response"], r["No response"], r[pol], r[pol]]
        ax1.plot(ts, ys, lw=1.6 if pol.startswith("Flow") else 1.2,
                 color=style[pol]["color"], ls=style[pol]["ls"], label=pol,
                 drawstyle="default")
    ax1.axvline(0, color="#c3c2b7", lw=0.6, zorder=0)
    ax1.axvline(5.0, color="#c3c2b7", lw=0.6, zorder=0)
    ax1.text(-0.8, 0.757, "onset", fontsize=6.5, color="#52514e", ha="right")
    ax1.text(5.8, 0.757, "detect + respond", fontsize=6.5, color="#52514e", ha="left")
    ax1.set_xlim(-10, horizon)
    ax1.set_ylim(0.68, 0.80)
    ax1.set_xlabel("Time from attack onset (s)")
    ax1.set_ylabel("RRS")
    ax1.set_title("(a) RRS over a DNP3-session attack episode", loc="left")
    ax1.grid(axis="y", color="#e6e5e0", lw=0.5)
    fig.legend(*ax1.get_legend_handles_labels(), ncol=2, frameon=False,
               loc="lower center", bbox_to_anchor=(0.52, 0.945),
               handlelength=2.8, columnspacing=1.2)

    # (b) post-response RRS by attack family and policy.
    n = len(POLICIES)
    width = 0.19
    for i, pol in enumerate(POLICIES):
        xs = [j + (i - (n - 1) / 2) * (width + 0.015) for j in range(len(rows))]
        ys = [row[pol] for row in rows]
        ax2.bar(xs, ys, width=width, color=style[pol]["color"], hatch=style[pol]["hatch"],
                edgecolor="white", linewidth=0.6, label=pol)
    ax2.axhline(normal, color="#52514e", lw=0.7, ls=(0, (2, 2)))
    ax2.text(len(rows) - 0.52, normal + 0.006, "normal", fontsize=6.3, color="#52514e",
             ha="right")
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels([row["family"] for row in rows])
    ax2.set_ylim(0.4, 0.82)
    ax2.set_ylabel("Post-response RRS")
    ax2.set_title("(b) Post-response RRS by attack family (bay 2)", loc="left")
    ax2.grid(axis="y", color="#e6e5e0", lw=0.5)
    ax2.set_axisbelow(True)

    out = os.path.join(FIGS, "stage3_resilience.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
