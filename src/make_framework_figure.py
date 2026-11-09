"""Framework figure: standalone script, needs only matplotlib.

    python make_framework_figure.py

Writes framework.pdf and framework.png in the same folder as this script.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
W, H = 7.16, 3.0
LW = 0.6
GREY = "0.88"

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


class Box:
    def __init__(self, cx, cy, w, h, text, diamond=False):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        if diamond:
            ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
                                 closed=True, fc="white", ec="black", lw=LW, zorder=2))
        else:
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec="black", lw=LW, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", zorder=3, linespacing=1.15)

    left = property(lambda s: (s.cx - s.w / 2, s.cy))
    right = property(lambda s: (s.cx + s.w / 2, s.cy))
    top = property(lambda s: (s.cx, s.cy + s.h / 2))
    bottom = property(lambda s: (s.cx, s.cy - s.h / 2))


def arrow(*pts, dashed=False):
    """Polyline through pts with an arrowhead at the last point (segments must be axis-aligned)."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert abs(x0 - x1) < 1e-9 or abs(y0 - y1) < 1e-9, "arrow segments must be horizontal or vertical"
    ls = (0, (3, 2)) if dashed else "-"
    xs, ys = zip(*pts[:-1])
    if len(pts) > 2:
        ax.plot(xs, ys, color="black", lw=LW, ls=ls, solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.18", lw=LW, color="black",
                                shrinkA=0, shrinkB=0, ls=ls))


def panel(x0, x1, title):
    y0, y1 = 0.36, 2.94
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="black", lw=LW, zorder=0))
    ax.add_patch(Rectangle((x0, y1 - 0.2), x1 - x0, 0.2, fc=GREY, ec="black", lw=LW, zorder=0))
    ax.text((x0 + x1) / 2, y1 - 0.1, title, ha="center", va="center", fontweight="bold")
    return (x0 + x1) / 2


def rows(n, top=2.52, bottom=0.62):
    step = (top - bottom) / (n - 1)
    return [top - i * step for i in range(n)]


BH = 0.27  # box height
# ---------------------------------------------------------------- panels
PA = (0.22, 1.40)
PB = (1.58, 2.86)
PC = (3.04, 4.98)
PD = (5.16, 7.08)
ca = panel(*PA, "Substation OT network")
cb = panel(*PB, "Stage 1: Detection")
cc = panel(*PC, "Stage 2: Root-cause diagnosis")
cd = panel(*PD, "Stage 3: Response")

# ---------------------------------------------------------------- substation OT network
y = rows(4)
a1 = Box(ca - 0.29, y[0], 0.54, BH, "Station bus\nDNP3/TCP")
a2 = Box(ca + 0.29, y[0], 0.54, BH, "Process bus\nGOOSE")
a3 = Box(ca, y[1], 1.06, BH, "Port mirroring and\nprotocol parsing")
a4 = Box(ca, y[2], 1.06, BH, "5-s window features\n7 per node, 24 per edge")
a5 = Box(ca, y[3], 1.06, BH, "Graph sequence\n$X=(\\mathcal{G}^{(t-2)},\\ldots,\\mathcal{G}^{(t)})$")
mid = (y[0] + y[1]) / 2
ax.plot([a1.cx, a1.cx, ca], [a1.bottom[1], mid, mid], color="black", lw=LW, zorder=1)
ax.plot([a2.cx, a2.cx, ca], [a2.bottom[1], mid, mid], color="black", lw=LW, zorder=1)
arrow((ca, mid), a3.top)
arrow(a3.bottom, a4.top)
arrow(a4.bottom, a5.top)

# ---------------------------------------------------------------- stage 1
y = rows(5)
bw = 1.14
b1 = Box(cb, y[0], bw, BH, "Edge-conditioned GNN\nnode embeddings $\\mathbf{h}_v^{(t)}$")
b2 = Box(cb, y[1], bw, BH, "Mean and max pooling\ngraph embedding $\\mathbf{z}^{(t)}$")
b3 = Box(cb, y[2], bw, BH, "GRU and temporal\nattention $\\alpha_i$, context $\\mathbf{c}$")
b4 = Box(cb, y[3], bw, BH, "Decoder and Huber\nanomaly score $S(X)$")
b5 = Box(cb, y[4], 0.84, 0.34, "$S(X)>\\tau$", diamond=True)
for p, q in [(b1, b2), (b2, b3), (b3, b4), (b4, b5)]:
    arrow(p.bottom, q.top)

# ---------------------------------------------------------------- stage 2
y = rows(6)
cw = 1.76
c1 = Box(cc, y[0], cw, BH, "Nearest-normal reference $\\bar{X}$\nfrom normal training sequences")
c2 = Box(cc, y[1], cw, BH, "Time: window occlusion\nanomalous window $t^*$, onset $\\hat{w}_0$")
c3 = Box(cc, y[2], cw, BH, "Device: exact node Shapley\nvalues $\\phi_v$, device $v^*$")
c4 = Box(cc, y[3], cw, BH, "Protocol: largest deviation $|z|$\nprotocol $g^*$, attribution $\\phi_g$")
c5 = Box(cc, y[4], cw, BH, "Flow and mechanism: occlusion\nflow $e^*$, mechanism $m^*$")
c6 = Box(cc, y[5], cw, BH, "Confidence $\\kappa$ and signature\n$L(x)=\\langle t^*,\\hat{w}_0,v^*,g^*,e^*,m^*,\\kappa\\rangle$")
for p, q in [(c1, c2), (c2, c3), (c3, c4), (c4, c5), (c5, c6)]:
    arrow(p.bottom, q.top)

# ---------------------------------------------------------------- stage 3
y = rows(5)
dw = 1.70
d1 = Box(cd, y[0], dw, BH, "Technique inference $\\hat{c}$\nMITRE ATT&CK for ICS")
d2 = Box(cd, y[1], dw, BH, "Candidate SDN actions\n$\\mathcal{A}_{\\hat{c}}\\cup\\mathcal{A}_0$ instantiated from $L(x)$")
d3 = Box(cd, y[2], dw, BH, "Select $a^*$ maximizing expected\nRRS subject to C1 to C3")
d4 = Box(cd, y[3], 0.92, 0.34, "$\\kappa\\geq\\theta_{\\mathrm{auto}}$", diamond=True)
d5 = Box(cd - 0.45, y[4], 0.80, BH, "SDN controller\nautomatic action")
d6 = Box(cd + 0.45, y[4], 0.80, BH, "Operator HMI\nadvisory action")
for p, q in [(d1, d2), (d2, d3), (d3, d4)]:
    arrow(p.bottom, q.top)
arrow(d4.left, (d5.cx, d4.cy), d5.top)
arrow(d4.right, (d6.cx, d4.cy), d6.top)
ax.text(d4.left[0] - 0.04, d4.cy + 0.035, "yes", ha="right", va="bottom")
ax.text(d4.right[0] + 0.04, d4.cy + 0.035, "no", ha="left", va="bottom")

# ---------------------------------------------------------------- links between panels
gab = (PA[1] + PB[0]) / 2
arrow(a5.right, (gab, a5.cy), (gab, b1.cy), b1.left)
gbc = (PB[1] + PC[0]) / 2
arrow(b5.right, (gbc, b5.cy), (gbc, c1.cy), c1.left)
ax.text((b5.right[0] + gbc) / 2, b5.cy + 0.025, "alert", ha="center", va="bottom")
gcd = (PC[1] + PD[0]) / 2
arrow(c6.right, (gcd, c6.cy), (gcd, d1.cy), d1.left)

# ---------------------------------------------------------------- feedback to the network
yb, xl = 0.17, 0.10
arrow(d5.bottom, (d5.cx, yb), (xl, yb), (xl, a1.cy), a1.left, dashed=True)
ax.text((xl + d5.cx) / 2, yb + 0.025, "OpenFlow rule with hard timeout (drop, meter, reroute, or quarantine), "
        "followed by renewed Stage 1 monitoring", ha="center", va="bottom")

out = Path(__file__).resolve().parent
fig.savefig(out / "framework.pdf")
fig.savefig(out / "framework.png", dpi=300)
print("wrote", out / "framework.pdf", "and", out / "framework.png")