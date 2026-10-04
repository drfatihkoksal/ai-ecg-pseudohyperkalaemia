"""Stage 16 (exploratory, post hoc): does the Q1b head-to-head depend on how close the
potassium draws are to the ECGs?

Labs are matched to each ECG within +-12 h (stage 3), so the label is a noisy proxy for
the potassium at the moment of the ECG. If the siamese-over-static gain is a real
physiological signal, it should GROW as both draws get closer to their ECGs
(dose-response) and shrink toward 0 where they are far apart.

Uses the pooled 5-fold out-of-fold scores (14_cv_hyperk.py); no retraining -- the
networks were trained on the full +-12 h set, only the evaluation rows are stratified.
gmax = max(|K draw 1 - ECG1|, |K draw 2 - ECG2|) in hours. NOT pre-registered: report
as exploratory.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _evalutils import DATA, OUT, auc

NB, SEED = 2000, 20260713

o = pd.read_parquet(f"{OUT}/cv/oof_scores.parquet")
o = o[o.hyperk_new.notna()]
p = pd.read_parquet(f"{DATA}/pairs_v2.parquet", columns=["study1", "study2", "t1", "t2", "k_ct", "k_ct2"])
hrs = lambda a, b: ((a - b).dt.total_seconds() / 3600).abs()
p["gmax"] = np.maximum(hrs(p.k_ct, p.t1), hrs(p.k_ct2, p.t2))
o = o.merge(p[["study1", "study2", "gmax"]], on=["study1", "study2"])
o = o.sort_values("subject_id", kind="stable").reset_index(drop=True)


def boot(d, a, b):
    sid, y, A, B = d.subject_id.values, d.hyperk_new.values.astype(int), d[a].values, d[b].values
    st = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]); L0 = np.diff(np.r_[st, len(sid)])
    rng, out = np.random.default_rng(SEED), []
    for _ in range(NB):
        c = rng.integers(0, len(st), len(st)); L = L0[c]
        ix = np.repeat(st[c] - np.r_[0, np.cumsum(L)[:-1]], L) + np.arange(L.sum())
        out.append(auc(y[ix], B[ix]) - auc(y[ix], A[ix]))
    return np.mean(out), *np.percentile(out, [2.5, 97.5])


SUBSETS = [("cumulative <=2h", o.gmax <= 2), ("cumulative <=4h", o.gmax <= 4),
           ("cumulative <=6h", o.gmax <= 6), ("all (<=12h)", o.gmax <= 12),
           ("band 0-4h", o.gmax <= 4), ("band 4-8h", (o.gmax > 4) & (o.gmax <= 8)),
           ("band 8-12h", o.gmax > 8)]
rows = []
for lab, m in SUBSETS:
    d = o[m.values].reset_index(drop=True); y = d.hyperk_new.values
    dd, lo, hi = boot(d, "static_ecg2_hk", "siamese_hk")
    rows.append(dict(subset=lab, pairs=len(d), events=int(y.sum()), event_rate=round(y.mean(), 4),
                     auc_floor=round(auc(y, d.floor_hk), 4), auc_static=round(auc(y, d.static_ecg2_hk), 4),
                     auc_siamese=round(auc(y, d.siamese_hk), 4),
                     siamese_minus_static=round(dd, 4), ci=f"[{lo:+.4f}, {hi:+.4f}]"))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("=== new hyperK, pooled OOF, stratified by lab-ECG distance (EXPLORATORY) ===")
print(R.to_string(index=False))
R.to_csv(f"{OUT}/lab_window.csv", index=False)
print(f"\nwrote {OUT}/lab_window.csv")
