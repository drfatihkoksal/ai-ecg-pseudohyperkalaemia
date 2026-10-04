"""Stage 17: does the learned difference beat the published personal-revision baseline?

Lou et al. 2022 (EHJ Digital Health, doi:10.1093/ehjdh/ztac072) revise a single-ECG
potassium model with the patient's previous annotated ECG:
        K2_hat = f(ECG2) + (k1 - f(ECG1))
i.e. a static absolute-K model plus a personal offset. That is the obvious reviewer
question for this paper: is the siamese difference model better than a good static K
model used in a pairwise way?

Everything here is OUT-OF-FOLD and needs no retraining:
  f(.)    = `scratch`, the standard single-ECG absolute-K model of 16_single_ecg_k.py
            --fold k; its folds are the pair-level CV folds patient-for-patient, so for a
            pair in test fold k both f(ECG1) and f(ECG2) are out-of-fold.
  pred_dK = siamese, 09_train.py --fold k (outputs/cv/preds_siamese_f<k>.parquet).

Models (all on top of floor_K = k1, cr1, age, sex, dt, dHR):
  static_abs   + f(ECG2)                      absolute-K model, no pairing
  lou          + [f(ECG2) - f(ECG1)]          Lou revision; with k1 in the floor this is
                                              the linear form of f2 + (k1 - f1)
  lou_flex     + f(ECG1) + f(ECG2)            free weights -- the strongest static-pair rival
  siamese      + pred_dK
  siamese+lou  + pred_dK + f(ECG1) + f(ECG2)  are they complementary?
Combiner: logistic / ridge cross-fit on the pooled OOF set, 5-fold grouped by patient
(identical procedure for every model). Combiner-free scores are reported alongside:
k1 + pred_dK vs f2 + (k1 - f1).

PRIMARY (fixed before running): new-hyperK AUC, siamese - lou, cross-fit.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _evalutils import DATA, OUT, auc, r2, crossfit

NF, NB, SEED = 5, 2000, 20260713
CV = f"{OUT}/cv"
FLOOR_K = ["cr", "k", "age", "sex", "dt_h", "dhr"]
KEY = ["study1", "study2"]

p = pd.read_parquet(f"{DATA}/pairs_v2.parquet")
p["sex"] = (p.gender == "F").astype(int)
hrs = lambda a, b: ((a - b).dt.total_seconds() / 3600).abs()
p["gmax"] = np.maximum(hrs(p.k_ct, p.t1), hrs(p.k_ct2, p.t2))

s = pd.concat([pd.read_parquet(f"{CV}/preds_single_ecg_k_f{k}.parquet") for k in range(NF)])
f = s.set_index("study").scratch
sia = pd.concat([pd.read_parquet(f"{CV}/preds_siamese_f{k}.parquet").query("split == 'test'")
                 for k in range(NF)])[KEY + ["pred_dk"]]

d = p.merge(sia, on=KEY, how="inner")
d["f1"], d["f2"] = f.reindex(d.study1).values, f.reindex(d.study2).values
assert len(d) == len(p), f"{len(p) - len(d)} pairs lack a siamese OOF prediction"
assert d[["f1", "f2"]].notna().all().all(), "an ECG lacks an OOF single-ECG prediction"
d["lou"] = d.f2 - d.f1
d = d.sort_values("subject_id", kind="stable").reset_index(drop=True)
X0 = d[FLOOR_K].astype(float).fillna(d[FLOOR_K].median())
print(f"pairs: {len(d):,}  patients: {d.subject_id.nunique():,}  "
      f"new-hyperK eligible: {int(d.hyperk_new.notna().sum()):,} "
      f"(events {int(d.hyperk_new.sum()):,})", flush=True)

MODELS = {"floor": [], "static_abs": ["f2"], "lou": ["lou"], "lou_flex": ["f1", "f2"],
          "siamese": ["pred_dk"], "siamese+lou": ["pred_dk", "f1", "f2"]}


def scores(mask, y, metric):
    sid = d.subject_id.values[mask]
    out = {}
    for m, extra in MODELS.items():
        X = np.column_stack([X0.values[mask]] + [d[c].values[mask] for c in extra])
        out[m] = crossfit(X, y, sid, metric)
    return out


def boot(sid, y, a, b, metric):
    st = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]); L0 = np.diff(np.r_[st, len(sid)])
    rng, out = np.random.default_rng(SEED), []
    for _ in range(NB):
        c = rng.integers(0, len(st), len(st)); L = L0[c]
        ix = np.repeat(st[c] - np.r_[0, np.cumsum(L)[:-1]], L) + np.arange(L.sum())
        yy = y[ix]
        if metric == "auc":
            out.append(auc(yy, b[ix]) - auc(yy, a[ix]))
        else:
            m = yy.mean(); out.append(r2(yy, b[ix], m) - r2(yy, a[ix], m))
    out = np.array(out)
    return np.mean(out), np.percentile(out, 2.5), np.percentile(out, 97.5), float((out > 0).mean())


ROWS, CONTR = [], []
CONTRASTS = [("siamese", "lou"), ("siamese", "lou_flex"), ("siamese", "static_abs"),
             ("lou", "static_abs"), ("siamese+lou", "lou_flex"), ("siamese+lou", "siamese")]
hk = d.hyperk_new.notna().values
SUBSETS = {"all": np.ones(len(d), bool), "anchor": d.is_anchor.values.astype(bool),
           "labs<=4h": (d.gmax <= 4).values}

for tgt, metric, base_mask, ycol in (("new hyperK", "auc", hk, "hyperk_new"),
                                     ("dK", "r2", np.ones(len(d), bool), "dk")):
    for sub, sm in SUBSETS.items():
        if tgt == "dK" and sub != "all": continue
        m = base_mask & sm
        y = d[ycol].values[m].astype(float)
        sc = scores(m, y, metric)
        # combiner-free (new hyperK only): rank by the predicted k2 directly
        if metric == "auc":
            sc["free: k1+pred_dK"] = (d.k + d.pred_dk).values[m]
            sc["free: f2+(k1-f1)"] = (d.f2 + d.k - d.f1).values[m]
            sc["free: f2"] = d.f2.values[m]
        sid = d.subject_id.values[m]
        sfn = (lambda p_: auc(y, p_)) if metric == "auc" else (lambda p_: r2(y, p_))
        for name, v in sc.items():
            ROWS.append(dict(target=tgt, subset=sub, model=name, n=int(m.sum()),
                             events=int(y.sum()) if metric == "auc" else None, score=round(sfn(v), 4)))
        pairs = CONTRASTS + ([("free: k1+pred_dK", "free: f2+(k1-f1)")] if metric == "auc" else [])
        for b_, a_ in pairs:
            dd, lo, hi, pg = boot(sid, y, sc[a_], sc[b_], metric)
            CONTR.append(dict(target=tgt, subset=sub, contrast=f"{b_} - {a_}", delta=round(dd, 4),
                              ci=f"[{lo:+.4f}, {hi:+.4f}]", p_gt0=round(pg, 4)))
        print(f"... {tgt} / {sub} done", flush=True)

R, C = pd.DataFrame(ROWS), pd.DataFrame(CONTR)
pd.set_option("display.width", 200)
pr = C[(C.target == "new hyperK") & (C.subset == "all") & (C.contrast == "siamese - lou")].iloc[0]
print(f"\n=== PRIMARY: new-hyperK AUC, siamese - lou (cross-fit, pooled OOF) ===\n"
      f"    delta = {pr.delta:+.4f} {pr.ci}   P(delta>0) = {pr.p_gt0}\n")
print("=== scores ===")
print(R.pivot_table(index=["target", "subset"], columns="model", values="score", sort=False)
      .round(4).to_string(), "\n")
print("=== contrasts (patient-cluster bootstrap) ===")
print(C.to_string(index=False))
R.to_csv(f"{OUT}/lou_baseline_scores.csv", index=False)
C.to_csv(f"{OUT}/lou_baseline_contrasts.csv", index=False)
print(f"\nwrote {OUT}/lou_baseline_scores.csv, lou_baseline_contrasts.csv")
