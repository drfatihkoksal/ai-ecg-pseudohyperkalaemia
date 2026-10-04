"""Figure 1: cohort flow (counts from outputs/paper/flow_counts.csv, written by 31_paper_extras.py)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); PO = f"{ROOT}/outputs/paper"
f = pd.read_csv(f"{PO}/flow_counts.csv", index_col=0)["n"]; f.index = f.index.str.strip()
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#8a8984"
fig, ax = plt.subplots(figsize=(7.0, 6.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
def box(x, y, w, h, txt, bold=False):
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=False, ec=EDGE, lw=0.8))
    ax.text(x, y, txt, ha="center", va="center", fontsize=7.4, color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)
def arrow(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
n = lambda k: f"{int(f[k]):,}"
X, W = 33, 52                                   # main column
box(X, 94, W, 8, f"Potassium results flagged as hemolyzed\nby the laboratory (MIMIC-IV, n = {n('Potassium results flagged hemolyzed')})")
arrow(X, 90, X, 85.5)
box(X, 82, W, 6.5, f"Hemolyzed potassium ≥5.5 mmol/L (n = {n('... with potassium >=5.5 mmol/L')})")
arrow(X, 78.7, X, 74.5)
box(X, 70, W, 8, f"Index events: 12-lead ECG within 2 h\nof the draw (n = {n('... with a 12-lead ECG within 2 h (index events)')})")
indet = sum(int(f[k]) for k in ("indet_no_repeat", "indet_normal_after_6h", "indet_repeat_5.0-5.4", "indet_treated_then_normal"))
box(81, 62, 36, 22, "Indeterminate, excluded (n = {:,})\n\nNo non-hemolyzed repeat\nwithin 12 h: {}\nNormal repeat only after 6 h: {}\nRepeat 5.0–5.4 mmol/L: {}\nTreated, then normal: {}".format(
    indet, n("indet_no_repeat"), n("indet_normal_after_6h"), n("indet_repeat_5.0-5.4"), n("indet_treated_then_normal")))
arrow(X, 66, X, 58.5); arrow(X, 62, 62.8, 62)
box(X, 54, W, 8, f"Classified by the first non-hemolyzed\nrepeat potassium (n = {int(f['pseudo']) + int(f['true']):,})")
arrow(X - 12, 50, 17, 43.5); arrow(X + 12, 50, 50, 43.5)
box(17, 36, 30, 13, f"Pseudohyperkalemia\nrepeat <5.0 mmol/L within 6 h,\nno K-lowering therapy\n(n = {n('pseudo')})")
box(50, 36, 30, 13, f"True hyperkalemia\nrepeat ≥5.5 mmol/L\nwithin 12 h\n(n = {n('true')})")
arrow(17, 29.5, X - 3, 23); arrow(50, 29.5, X + 3, 23)
box(X, 19, W, 8, f"Primary analysis: ECG passed quality control\n(n = {n('Analyzed (ECG passed quality control)')}; 529 pseudo, 99 true)", bold=True)
box(50, 5, 96, 8, f"Secondary reference group: non-hemolyzed potassium ≥5.5 and <6.5 mmol/L confirmed ≥5.5 mmol/L\non a repeat within 12 h (n = {n('Non-hemolyzed confirmed true hyperkalemia, index K <6.5 (reference)')})")
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(f"{PO}/figure1_flow.{e}", dpi=600)
print("wrote figure1_flow")
