"""Stage 15: pooled 5-fold verdict on "does dK pre-training give a better single-ECG
potassium model?" (concept.md sec.0.3c / sec.11-2b).

On the single pre-registered split, the siamese encoder fine-tuned on absolute K beat
the from-scratch standard model on hyperkalaemia AUC by +0.013 with a CI lower bound of
0.000 -- one split, one seed. Here every patient is a test patient once
(16_single_ecg_k.py --fold k, k = 0..4; source encoders models/cv/<arm>_f<k>.pt never saw
fold k), and the out-of-fold predictions are pooled.

PRIMARY (fixed before running): AUC_hyperK(ft_siamese) - AUC_hyperK(scratch), pooled.
Secondary: R^2, hypoK AUC, within-patient r, between-patient r; same contrast for the
frozen probe. CIs: patient-cluster bootstrap (2,000). Within-patient r is computed on
patient-demeaned values, which are invariant to resampling whole patients, and
between-patient r on patient means, so both bootstrap exactly by patient.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _evalutils import OUT, auc, r2

NF, NB, SEED = 5, 2000, 20260713
CV = f"{OUT}/cv"
MODELS = ["scratch", "ft_siamese", "probe_siamese", "probe_emb_diff", "probe_scratch",
          "probe_static_ecg2", "probe_random"]

files = [f"{CV}/preds_single_ecg_k_f{k}.parquet" for k in range(NF)]
missing = [f for f in files if not os.path.exists(f)]
assert not missing, f"missing folds: {missing}"
d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
assert d.study.is_unique, "an ECG is out-of-fold in more than one fold"
d = d.sort_values("subject_id", kind="stable").reset_index(drop=True)
y, sid = d.k.values, d.subject_id.values
print(f"pooled out-of-fold: {len(d):,} ECGs, {d.subject_id.nunique():,} patients, "
      f"hyperK (>=5.5) {int((y >= 5.5).sum()):,}, hypoK (<3.5) {int((y < 3.5).sum()):,}\n")

# fixed per-row / per-patient quantities
g = d.groupby("subject_id")
dm = {m: d[m].values - g[m].transform("mean").values for m in MODELS + ["k"]}
pm = g[MODELS + ["k"]].mean()                     # one row per patient, same order as starts
starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
lens = np.diff(np.r_[starts, len(sid)])


def stats(m, ix=None, pix=None):
    ix = slice(None) if ix is None else ix
    pix = slice(None) if pix is None else pix
    yy, pp = y[ix], d[m].values[ix]
    return dict(R2=r2(yy, pp), AUC_hyperK=auc(yy >= 5.5, pp), AUC_hypoK=auc(yy < 3.5, -pp),
                r_within=np.corrcoef(dm[m][ix], dm["k"][ix])[0, 1],
                r_between=np.corrcoef(pm[m].values[pix], pm["k"].values[pix])[0, 1])


point = pd.DataFrame({m: stats(m) for m in MODELS}).T
rng = np.random.default_rng(SEED)
B = {m: [] for m in MODELS}
for _ in range(NB):
    c = rng.integers(0, len(starts), len(starts)); L = lens[c]
    ix = np.repeat(starts[c] - np.r_[0, np.cumsum(L)[:-1]], L) + np.arange(L.sum())
    for m in MODELS:
        B[m].append(stats(m, ix, c))
B = {m: pd.DataFrame(v) for m, v in B.items()}

C = []
for m in MODELS[1:]:
    for met in point.columns:
        diff = B[m][met].values - B["scratch"][met].values
        C.append(dict(model=m, metric=met, scratch=round(point.loc["scratch", met], 4),
                      model_value=round(point.loc[m, met], 4),
                      delta=round(point.loc[m, met] - point.loc["scratch", met], 4),
                      ci=f"[{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]",
                      p_gt0=round(float((diff > 0).mean()), 4)))
C = pd.DataFrame(C)

PF = []
for k in range(NF):
    f = d[d.fold == k]
    r = dict(fold=k, n=len(f), hyperK=int((f.k >= 5.5).sum()))
    for m in ("scratch", "ft_siamese", "probe_siamese"):
        r[f"AUC_{m}"] = round(auc(f.k >= 5.5, f[m]), 4)
        r[f"R2_{m}"] = round(r2(f.k.values, f[m].values), 4)
    r["ft_minus_scratch_AUC"] = round(r["AUC_ft_siamese"] - r["AUC_scratch"], 4)
    PF.append(r)
PF = pd.DataFrame(PF)

pd.set_option("display.width", 220)
p = C[(C.model == "ft_siamese") & (C.metric == "AUC_hyperK")].iloc[0]
print("=== PRIMARY: hyperK AUC, ft_siamese - scratch, pooled out-of-fold ===")
print(f"    {p.scratch:.4f} -> {p.model_value:.4f}   delta = {p.delta:+.4f} {p.ci}   P(delta>0) = {p.p_gt0}\n")
print("=== all models, pooled out-of-fold ===")
print(point.round(4).to_string(), "\n")
print("=== contrasts vs scratch (patient-cluster bootstrap) ===")
print(C.to_string(index=False), "\n")
print("=== per fold ===")
print(PF.to_string(index=False))
print(f"    ft - scratch AUC > 0 in {(PF.ft_minus_scratch_AUC > 0).sum()}/{NF} folds, "
      f"mean {PF.ft_minus_scratch_AUC.mean():+.4f}")
point.to_csv(f"{OUT}/cv_single_k.csv"); C.to_csv(f"{OUT}/cv_single_k_contrasts.csv", index=False)
PF.to_csv(f"{OUT}/cv_single_k_perfold.csv", index=False)
print(f"\nwrote {OUT}/cv_single_k.csv, cv_single_k_contrasts.csv, cv_single_k_perfold.csv")
