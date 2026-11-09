"""Framework figure as an editable draw.io file (plain black and white, orthogonal arrows).

    python make_framework_drawio.py

Writes framework.drawio next to this script. Every arrow is a draw.io connector attached to
its boxes with explicit horizontal or vertical waypoints, and the script refuses to write the
file if any text would overlap a box, a panel border, or another label.
"""
from pathlib import Path
from xml.sax.saxutils import quoteattr
import re

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

FONT, FS = "Times New Roman", 10
W, H = 752, 320
PANEL_TOP, PANEL_BOTTOM, TITLE_H = 6, 286, 20
ROW_TOP, ROW_BOTTOM, BH = 50, 260, 28
GREY = "#E0E0E0"

_t2p, _prop = TextToPath(), FontProperties(family=FONT, size=FS)


def text_size(html):
    """Conservative width and height of an HTML label in px (sub/sup measured at full size)."""
    lines = [re.sub(r"<[^>]+>", "", l).replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
             for l in html.split("<br>")]
    w = max(_t2p.get_text_width_height_descent(l, _prop, ismath=False)[0] for l in lines)
    return w, len(lines) * FS * 1.2


cells, boxes, texts, lines_h, lines_v = [], {}, [], [], []
_id = [1]


def nid():
    _id[0] += 1
    return f"c{_id[0]}"


def vertex(x, y, w, h, value, style):
    i = nid()
    cells.append(f'<mxCell id="{i}" value={quoteattr(value)} style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')
    return i


BASE = f"html=1;whiteSpace=wrap;fontFamily={FONT};fontSize={FS};fontColor=#000000;strokeColor=#000000;strokeWidth=1;"


class Box:
    def __init__(self, name, cx, cy, w, h, label, diamond=False):
        self.name, self.cx, self.cy, self.w, self.h, self.diamond = name, cx, cy, w, h, diamond
        shape = "rhombus;" if diamond else "rounded=0;"
        self.id = vertex(cx - w / 2, cy - h / 2, w, h, label, shape + BASE + "fillColor=#FFFFFF;spacing=2;")
        tw, th = text_size(label)
        room = w * (1 - th / h) if diamond else w - 6
        assert tw <= room and th <= h - 2, f"text does not fit in {name}: {tw:.0f}>{room:.0f} or {th:.0f}>{h - 2}"
        boxes[name] = self

    @property
    def bbox(self):
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)


def label(cx, cy, s, align="center"):
    tw, th = text_size(s)
    w, h = tw + 4, th
    x = {"center": cx - w / 2, "left": cx, "right": cx - w}[align]
    vertex(x, cy - h / 2, w, h, s, "text;" + BASE + f"align={align};verticalAlign=middle;strokeColor=none;fillColor=none;spacing=0;")
    texts.append((x, cy - h / 2, x + w, cy + h / 2, s))


def edge(src, sxy, dst, dxy, pts=(), dashed=False):
    """Connector from src port sxy=(fx,fy) to dst port dxy through axis-aligned waypoints."""
    p0 = (src.cx + (sxy[0] - 0.5) * src.w, src.cy + (sxy[1] - 0.5) * src.h)
    p1 = (dst.cx + (dxy[0] - 0.5) * dst.w, dst.cy + (dxy[1] - 0.5) * dst.h)
    path = [p0, *pts, p1]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        assert abs(x0 - x1) < 1e-6 or abs(y0 - y1) < 1e-6, f"slanted segment {src.name}->{dst.name}"
        (lines_h if abs(y0 - y1) < 1e-6 else lines_v).append(((x0, y0), (x1, y1)))
    style = (f"edgeStyle=none;rounded=0;html=1;endArrow=block;endFill=1;endSize=5;strokeColor=#000000;strokeWidth=1;"
             f"exitX={sxy[0]};exitY={sxy[1]};exitDx=0;exitDy=0;entryX={dxy[0]};entryY={dxy[1]};entryDx=0;entryDy=0;"
             + ("dashed=1;dashPattern=4 3;" if dashed else ""))
    pts_xml = "".join(f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in pts)
    cells.append(f'<mxCell id="{nid()}" style="{style}" edge="1" parent="1" source="{src.id}" target="{dst.id}">'
                 f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts_xml}</Array></mxGeometry></mxCell>')


panels = []


def panel(x0, x1, title):
    vertex(x0, PANEL_TOP, x1 - x0, PANEL_BOTTOM - PANEL_TOP, "", "rounded=0;html=1;fillColor=none;strokeColor=#000000;strokeWidth=1;")
    vertex(x0, PANEL_TOP, x1 - x0, TITLE_H, f"<b>{title}</b>", "rounded=0;" + BASE + f"fillColor={GREY};")
    panels.append((x0, x1, title))
    return (x0 + x1) / 2


def rows(n):
    step = (ROW_BOTTOM - ROW_TOP) / (n - 1)
    return [ROW_TOP + i * step for i in range(n)]


# ---------------------------------------------------------------- panels
GAP, X0 = 22, 24
PA = (X0, X0 + 126)
PB = (PA[1] + GAP, PA[1] + GAP + 136)
PC = (PB[1] + GAP, PB[1] + GAP + 196)
PD = (PC[1] + GAP, PC[1] + GAP + 196)
ca = panel(*PA, "Substation OT network")
cb = panel(*PB, "Stage 1: Detection")
cc = panel(*PC, "Stage 2: Root-cause diagnosis")
cd = panel(*PD, "Stage 3: Response")

# ---------------------------------------------------------------- network
y = rows(4)
a1 = Box("a1", ca - 30, y[0], 56, BH, "Station bus<br>DNP3/TCP")
a2 = Box("a2", ca + 30, y[0], 56, BH, "Process bus<br>GOOSE")
a3 = Box("a3", ca, y[1], 112, BH, "Port mirroring and<br>protocol parsing")
a4 = Box("a4", ca, y[2], 112, BH, "5-s window features<br>7 per node, 24 per edge")
a5 = Box("a5", ca, y[3], 112, BH, "Graph sequence<br><i>X</i> = (<i>G</i><sup>(<i>t</i>−2)</sup>, …, <i>G</i><sup>(<i>t</i>)</sup>)")
mid = (y[0] + BH / 2 + y[1] - BH / 2) / 2
edge(a1, (0.5, 1), a3, (0.5, 0), [(a1.cx, mid), (ca, mid)])
edge(a2, (0.5, 1), a3, (0.5, 0), [(a2.cx, mid), (ca, mid)])
edge(a3, (0.5, 1), a4, (0.5, 0))
edge(a4, (0.5, 1), a5, (0.5, 0))

# ---------------------------------------------------------------- stage 1
y = rows(5)
bw = 118
b1 = Box("b1", cb, y[0], bw, BH, "Edge-conditioned GNN<br>node embeddings <b>h</b><sub><i>v</i></sub><sup>(<i>t</i>)</sup>")
b2 = Box("b2", cb, y[1], bw, BH, "Mean and max pooling<br>graph embedding <b>z</b><sup>(<i>t</i>)</sup>")
b3 = Box("b3", cb, y[2], bw, BH, "GRU and temporal<br>attention <i>α</i><sub><i>i</i></sub>, context <b>c</b>")
b4 = Box("b4", cb, y[3], bw, BH, "Decoder and Huber<br>anomaly score <i>S</i>(<i>X</i>)")
b5 = Box("b5", cb, y[4], 74, 34, "<i>S</i>(<i>X</i>) &gt; <i>τ</i>", diamond=True)
for p, q in [(b1, b2), (b2, b3), (b3, b4), (b4, b5)]:
    edge(p, (0.5, 1), q, (0.5, 0))

# ---------------------------------------------------------------- stage 2
y = rows(6)
cw = 180
c1 = Box("c1", cc, y[0], cw, BH, "Nearest-normal reference <i>X̄</i><br>from normal training sequences")
c2 = Box("c2", cc, y[1], cw, BH, "Time: window occlusion<br>anomalous window <i>t</i>*, onset <i>ŵ</i><sub>0</sub>")
c3 = Box("c3", cc, y[2], cw, BH, "Device: exact node Shapley<br>values <i>φ</i><sub><i>v</i></sub>, device <i>v</i>*")
c4 = Box("c4", cc, y[3], cw, BH, "Protocol: largest deviation |<i>z</i>|<br>protocol <i>g</i>*, attribution <i>φ</i><sub><i>g</i></sub>")
c5 = Box("c5", cc, y[4], cw, BH, "Flow and mechanism: occlusion<br>flow <i>e</i>*, mechanism <i>m</i>*")
c6 = Box("c6", cc, y[5], cw, BH, "Confidence <i>κ</i> and signature<br><i>L</i>(<i>x</i>) = ⟨<i>t</i>*, <i>ŵ</i><sub>0</sub>, <i>v</i>*, <i>g</i>*, <i>e</i>*, <i>m</i>*, <i>κ</i>⟩")
for p, q in [(c1, c2), (c2, c3), (c3, c4), (c4, c5), (c5, c6)]:
    edge(p, (0.5, 1), q, (0.5, 0))

# ---------------------------------------------------------------- stage 3
y = rows(5)
dw = 180
d1 = Box("d1", cd, y[0], dw, BH, "Technique inference <i>ĉ</i><br>MITRE ATT&amp;CK for ICS")
d2 = Box("d2", cd, y[1], dw, BH, "Candidate SDN actions <i>A</i><sub><i>ĉ</i></sub> ∪ <i>A</i><sub>0</sub><br>instantiated from <i>L</i>(<i>x</i>)")
d3 = Box("d3", cd, y[2], dw, BH, "Select <i>a</i>* maximizing expected<br>RRS subject to C1 to C3")
d4 = Box("d4", cd, y[3], 72, 34, "<i>κ</i> ≥ <i>θ</i><sub>auto</sub>", diamond=True)
d5 = Box("d5", cd - 48, y[4], 84, BH, "SDN controller<br>automatic action")
d6 = Box("d6", cd + 48, y[4], 84, BH, "Operator HMI<br>advisory action")
for p, q in [(d1, d2), (d2, d3), (d3, d4)]:
    edge(p, (0.5, 1), q, (0.5, 0))
edge(d4, (0, 0.5), d5, (0.5, 0), [(d5.cx, d4.cy)])
edge(d4, (1, 0.5), d6, (0.5, 0), [(d6.cx, d4.cy)])
ytxt = (d4.cy + d5.cy - BH / 2) / 2 + 2
label(d5.cx - 4, ytxt, "yes", align="right")
label(d6.cx + 4, ytxt, "no", align="left")

# ---------------------------------------------------------------- links between panels
gab = (PA[1] + PB[0]) / 2
edge(a5, (1, 0.5), b1, (0, 0.5), [(gab, a5.cy), (gab, b1.cy)])
gbc = (PB[1] + PC[0]) / 2
edge(b5, (1, 0.5), c1, (0, 0.5), [(gbc, b5.cy), (gbc, c1.cy)])
label((b5.cx + b5.w / 2 + PB[1]) / 2, b5.cy - 8, "alert")
gcd = (PC[1] + PD[0]) / 2
edge(c6, (1, 0.5), d1, (0, 0.5), [(gcd, c6.cy), (gcd, d1.cy)])

# ---------------------------------------------------------------- feedback to the network
yb, xl = 310, 12
edge(d5, (0.5, 1), a1, (0, 0.5), [(d5.cx, yb), (xl, yb), (xl, a1.cy)], dashed=True)
label((xl + d5.cx) / 2, (PANEL_BOTTOM + yb) / 2,
      "OpenFlow rule with hard timeout (drop, meter, reroute, or quarantine), followed by renewed Stage 1 monitoring")

# ---------------------------------------------------------------- overlap checks
def inter(a, b, m=1.0):
    return a[0] < b[2] - m and b[0] < a[2] - m and a[1] < b[3] - m and b[1] < a[3] - m


bl = list(boxes.values())
for i, p in enumerate(bl):
    for q in bl[i + 1:]:
        assert not inter(p.bbox, q.bbox, -4), f"boxes {p.name} and {q.name} overlap or are closer than 4 px"
    x0, x1, _ = next(pp for pp in panels if pp[0] < p.cx < pp[1])
    assert x0 + 3 <= p.bbox[0] and p.bbox[2] <= x1 - 3 and p.bbox[1] >= PANEL_TOP + TITLE_H + 3 and p.bbox[3] <= PANEL_BOTTOM - 3, p.name
for t in texts:
    for p in bl:
        assert not inter(t[:4], p.bbox), f"label {t[4][:20]!r} overlaps box {p.name}"
    for x0, x1, _ in panels:  # panel borders
        for seg in [((x0, PANEL_TOP), (x0, PANEL_BOTTOM)), ((x1, PANEL_TOP), (x1, PANEL_BOTTOM))]:
            assert not (t[0] < seg[0][0] < t[2] and t[1] < PANEL_BOTTOM), f"label {t[4][:20]!r} crosses a panel border"
        assert not (t[1] < PANEL_BOTTOM < t[3] and t[0] < x1 and x0 < t[2]), f"label {t[4][:20]!r} crosses a panel bottom"
    for (x0, y0), (x1, y1) in lines_h:
        assert not (min(x0, x1) < t[2] and t[0] < max(x0, x1) and t[1] < y0 < t[3]), f"label {t[4][:20]!r} crosses an arrow"
    for (x0, y0), (x1, y1) in lines_v:
        assert not (min(y0, y1) < t[3] and t[1] < max(y0, y1) and t[0] < x0 < t[2]), f"label {t[4][:20]!r} crosses an arrow"
for i, a in enumerate(texts):
    for b in texts[i + 1:]:
        assert not inter(a[:4], b[:4]), "labels overlap"
for (x0, y0), (x1, y1) in lines_v:  # vertical arrow segments must not pass through a box they do not touch
    for p in bl:
        b = p.bbox
        assert not (b[0] + 1 < x0 < b[2] - 1 and min(y0, y1) < b[3] - 1 and b[1] + 1 < max(y0, y1)), f"arrow through {p.name}"
for (x0, y0), (x1, y1) in lines_h:
    for p in bl:
        b = p.bbox
        assert not (b[1] + 1 < y0 < b[3] - 1 and min(x0, x1) < b[2] - 1 and b[0] + 1 < max(x0, x1)), f"arrow through {p.name}"

xml = (f'<mxfile host="drawio"><diagram name="Framework" id="framework">'
       f'<mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="2" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       f'page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0"><root>'
       f'<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
out = Path(__file__).resolve().parent / "framework.drawio"
out.write_text(xml, encoding="utf-8")
print("all overlap checks passed, wrote", out)
