"""Stage 12: 5-fold patient-grouped CV for the Q1b new-hyperkalaemia endpoint.

Why: on the single pre-registered test split (233 events) siamese beat static_ecg2 by
+0.020 AUC with a CI crossing 0 -- too few events to decide the Q1c head-to-head on
the clinically relevant endpoint (concept.md sec.11-1). CV makes every patient a test
patient once: ~1,400 events.

Per fold k (09_train.py --fold k): fold k = TEST, fold k+1 = VAL, rest = TRAIN.
  * network trained on TRAIN, checkpoint chosen on VAL R^2(dK) -- same as v2.
  * combiner (floor_K, floor_K + pred_dK) fit on that fold's VAL, scored on its TEST.
Out-of-fold TEST scores are pooled over the 5 folds; the pooled AUC delta is PRIMARY,
CI = patient-cluster bootstrap (2,000) of the difference. Reported alongside:
  * per-fold deltas and their mean (no pooling across fold-specific combiners),
  * a combiner-free score: predicted k2 = k1 + pred_dK, ranked directly,
  * dK R^2 (secondary), subsets anchor and dt<=48.

PRIMARY CONTRAST (fixed before running): AUC(floor_K + siamese) - AUC(floor_K + static_ecg2)
on new hyperkalaemia (k1 < 5.0 -> k2 >= 5.5), pooled out-of-fold.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
from _evalutils import ROOT, DATA, OUT, ARMS, auc, r2, fit_eval

spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

NF, NB = 5, 2000
FLOOR_K = ["cr", "k", "age", "sex", "dt_h", "dhr"]
KEY = ["study1", "study2"]
DIFF_ARMS = ["siamese", "emb_diff", "beat_sub"]
CV = f"{OUT}/cv"

df = pd.read_parquet(f"{DATA}/pairs_v2.parquet")
df["sex"] = (df.gender == "F").astype(int)
df["fold"] = T.cv_folds(df.subject_id, NF)

arms = [a for a in ARMS if all(os.path.exists(f"{CV}/preds_{a}_f{k}.parquet") for k in range(NF))]
missing = sorted(set(ARMS) - set(arms))
if missing: print(f"[warn] incomplete CV, skipping arms: {missing}")
assert "siamese" in arms and "static_ecg2" in arms, "primary contrast needs siamese + static_ecg2"

# ------------------------------------------------------------------ out-of-fold scores
OOF = []
for k in range(NF):
    va = df[df.fold == (k + 1) % NF].reset_index(drop=True)
    te = df[df.fold == k].reset_index(drop=True)
    o = te[KEY + ["subject_id", "fold", "is_anchor", "dt_h", "k", "dk", "hyperk_new"]].copy()
    hv, ht = va.hyperk_new.notna().values, te.hyperk_new.notna().values

    def feats(d, pdk=None):
        M = d[FLOOR_K].astype(float)
        if pdk is not None: M = M.assign(pred_dk=pdk)
        return M.fillna(va[FLOOR_K].median()).values

    o["floor_hk"] = np.nan
    o.loc[ht, "floor_hk"] = fit_eval(feats(va)[hv], va.hyperk_new[hv], feats(te)[ht], "auc")
    o["floor_dk"] = fit_eval(feats(va), va.dk, feats(te), "r2")
    for arm in arms:
        p = pd.read_parquet(f"{CV}/preds_{arm}_f{k}.parquet")
        pv = va[KEY].merge(p, on=KEY, how="left").pred_dk.values
        pt = te[KEY].merge(p, on=KEY, how="left").pred_dk.values
        assert not (np.isnan(pv).any() or np.isnan(pt).any()), f"{arm} f{k}: missing predictions"
        o[f"{arm}_hk"] = np.nan
        o.loc[ht, f"{arm}_hk"] = fit_eval(feats(va, pv)[hv], va.hyperk_new[hv], feats(te, pt)[ht], "auc")
        o[f"{arm}_dk"] = fit_eval(feats(va, pv), va.dk, feats(te, pt), "r2")
        o[f"{arm}_k2hat"] = te.k.values + pt                     # combiner-free
    OOF.append(o)
oof = pd.concat(OOF, ignore_index=True)
oof.to_parquet(f"{CV}/oof_scores.parquet", index=False)

hk = oof[oof.hyperk_new.notna()].reset_index(drop=True)
y = hk.hyperk_new.values.astype(int)
SUB = {"all": np.ones(len(hk), bool), "anchor": hk.is_anchor.values.astype(bool),
       "dt<=48": (hk.dt_h <= 48).values}
print(f"pooled out-of-fold: {len(hk):,} eligible pairs, {y.sum():,} new-hyperK events, "
      f"{hk.subject_id.nunique():,} patients; arms = {arms}\n")


def row(name, a_col, b_col, sub, frame=hk, yy=None, metric="auc"):
    m = SUB[sub] if frame is hk else np.ones(len(frame), bool)
    yv = (y if yy is None else yy)[m]
    a, b = frame[a_col].values[m], frame[b_col].values[m]
    d, lo, hi, pg = boot_delta(frame.subject_id.values[m], yv, a, b, metric, n_boot=NB)
    s = (lambda p: auc(yv, p)) if metric == "auc" else (lambda p: r2(yv, p))
    return dict(contrast=name, subset=sub, n=int(m.sum()),
                events=int(yv.sum()) if metric == "auc" else None,
                base=round(s(a), 4), model=round(s(b), 4), delta=round(d, 4),
                ci=f"[{lo:+.4f}, {hi:+.4f}]", p_gt0=round(pg, 4))


def boot_delta(sid, y, a, b, metric, n_boot=NB, seed=T.SEED):
    """Patient-cluster bootstrap of metric(b) - metric(a); same estimator as
    _evalutils.boot_delta, vectorised index construction for ~80k-row pooled sets."""
    sid, y, a, b = map(np.asarray, (sid, y, a, b))
    order = np.argsort(sid, kind="stable")
    sid, y, a, b = sid[order], y[order], a[order], b[order]
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    lens = np.diff(np.r_[starts, len(sid)])
    rng, K, out = np.random.default_rng(seed), len(starts), []
    for _ in range(n_boot):
        c = rng.integers(0, K, K)
        L = lens[c]
        ix = np.repeat(starts[c] - np.r_[0, np.cumsum(L)[:-1]], L) + np.arange(L.sum())
        yy = y[ix]
        if metric == "auc":
            out.append(auc(yy, b[ix]) - auc(yy, a[ix]))
        else:
            m = yy.mean(); out.append(r2(yy, b[ix], m) - r2(yy, a[ix], m))
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), float((out > 0).mean())


R = []
for sub in SUB:
    for arm in arms:
        R.append(row(f"floor_K -> +{arm}", "floor_hk", f"{arm}_hk", sub))
    for arm in [a for a in DIFF_ARMS if a in arms]:
        R.append(row(f"H2H {arm} - static_ecg2", "static_ecg2_hk", f"{arm}_hk", sub))
    for arm in [a for a in DIFF_ARMS if a in arms]:
        R.append(row(f"H2H combiner-free k1+pred_dK: {arm} - static", "static_ecg2_k2hat",
                     f"{arm}_k2hat", sub))
R = pd.DataFrame(R)

# per-fold head-to-head (no pooling across fold-specific combiners)
PF = []
for k in range(NF):
    f = hk[hk.fold == k]
    yf = f.hyperk_new.values.astype(int)
    r = dict(fold=k, n=len(f), events=int(yf.sum()), floor=round(auc(yf, f.floor_hk), 4))
    for arm in arms:
        r[arm] = round(auc(yf, f[f"{arm}_hk"]), 4)
    r["siamese_minus_static"] = round(r["siamese"] - r["static_ecg2"], 4)
    PF.append(r)
PF = pd.DataFrame(PF)

# secondary: dK R^2, pooled
D = []
for arm in arms:
    D.append(row(f"dK floor_K -> +{arm}", "floor_dk", f"{arm}_dk", "all",
                 frame=oof, yy=oof.dk.values, metric="r2"))
for arm in [a for a in DIFF_ARMS if a in arms]:
    D.append(row(f"dK H2H {arm} - static_ecg2", "static_ecg2_dk", f"{arm}_dk", "all",
                 frame=oof, yy=oof.dk.values, metric="r2"))
D = pd.DataFrame(D)

pd.set_option("display.width", 220)
p = R[(R.contrast == "H2H siamese - static_ecg2") & (R.subset == "all")].iloc[0]
print("=== PRIMARY: AUC(floor_K + siamese) - AUC(floor_K + static_ecg2), new hyperK, pooled OOF ===")
print(f"    {p.base:.4f} -> {p.model:.4f}   delta = {p.delta:+.4f} {p.ci}   P(delta>0) = {p.p_gt0}\n")
for sub in SUB:
    print(f"=== new hyperkalaemia, subset: {sub} ===")
    print(R[R.subset == sub].drop(columns="subset").to_string(index=False), "\n")
print("=== per-fold AUCs (combiner fit on that fold's val) ===")
print(PF.to_string(index=False))
print(f"    mean siamese - static over folds = {PF.siamese_minus_static.mean():+.4f} "
      f"(folds > 0: {(PF.siamese_minus_static > 0).sum()}/{NF})\n")
print("=== secondary: dK R^2, pooled OOF ===")
print(D.drop(columns=["subset", "events"]).to_string(index=False))

R.to_csv(f"{OUT}/cv_hyperk.csv", index=False)
PF.to_csv(f"{OUT}/cv_hyperk_perfold.csv", index=False)
D.to_csv(f"{OUT}/cv_dk.csv", index=False)
print(f"\nwrote {OUT}/cv_hyperk.csv, cv_hyperk_perfold.csv, cv_dk.csv, {CV}/oof_scores.parquet")
