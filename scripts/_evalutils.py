"""Shared evaluation utilities for stages 9/9b/10.

Why this file exists: stages 9, 9b and 10 all estimated "does the d-ECG add
anything over the clinical floor" by fitting the combiner on TRAIN using the
network's own IN-SAMPLE train predictions. The network is heavily overfit
(emb_diff: r(dCr) = 0.41 on train vs 0.13 on test), so the combiner saw an ECG
feature that looked ~3x stronger than it is, over-weighted it, and the combined
model then LOST to the floor on test. Every negative delta in the first run of
outputs/incremental.csv and outputs/incremental_dcr.csv is that artefact, not a
null result.

The combiner must therefore be fit where the network's outputs are out-of-sample.
Three estimators are computed everywhere, and they must agree:

  valfit   PRIMARY. Floor and floor+ECG both fit on VAL, evaluated on TEST.
           Val never entered the weight updates. (It did pick the checkpoint, so
           a sliver of optimism remains -- which is what `crossfit` bounds.)
  crossfit SENSITIVITY. Both models cross-fit on TEST itself, patient-grouped
           5-fold, scored out-of-fold. No val involvement at all.
  trainfit THE OLD, BIASED NUMBER. Kept in the output files so the correction is
           auditable rather than silently overwritten. Do not report it.

Everything is nested-model-fair: the floor and the floor+ECG model are always fit
on the same rows with the same procedure, so the delta is the only moving part.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
WF = f"{DATA}/waveforms"
FLOOR = ["cr", "age", "sex", "dt_h", "dhr"]
ARMS = ["beat_sub", "emb_diff", "siamese", "static_ecg2"]
N_BOOT = 500
SEED = 20260713


def auc(y, s):
    y = np.asarray(y).astype(int)
    npos, nneg = y.sum(), len(y) - y.sum()
    if npos == 0 or nneg == 0: return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def r2(y, p, mean=None):
    m = y.mean() if mean is None else mean
    return 1 - ((y - p) ** 2).sum() / ((y - m) ** 2).sum()


def clusters(sid):
    """Row-index arrays, one per patient -- the resampling unit (pairs are nested
    in patients, ICC=0.37, so the bootstrap must draw patients, not pairs)."""
    return [np.asarray(v) for v in pd.Series(np.arange(len(sid))).groupby(np.asarray(sid)).indices.values()]


def boot_delta(sid, y, a, b, metric, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap of metric(b) - metric(a). Index-based, not DataFrame
    concatenation -- same estimator as before, ~100x faster."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    g, rng = clusters(sid), np.random.default_rng(seed)
    k, out = len(g), []
    for _ in range(n_boot):
        ix = np.concatenate([g[i] for i in rng.integers(0, k, k)])
        yy = y[ix]
        if metric == "auc":
            if len(np.unique(yy)) < 2: continue
            out.append(auc(yy, b[ix]) - auc(yy, a[ix]))
        else:
            m = yy.mean()
            out.append(r2(yy, b[ix], m) - r2(yy, a[ix], m))
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), float((out > 0).mean())


def _fit(X, y, metric):
    if metric == "auc":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X, y)
    return make_pipeline(StandardScaler(), Ridge()).fit(X, y)


def _score(m, X, metric):
    return m.predict_proba(X)[:, 1] if metric == "auc" else m.predict(X)


def fit_eval(Xfit, yfit, Xte, metric):
    """Fit on one split, score another."""
    return _score(_fit(Xfit, yfit, metric), Xte, metric)


def crossfit(X, y, groups, metric, n_splits=5):
    """Out-of-fold predictions on the evaluation set itself, patient-grouped."""
    X, y = np.asarray(X), np.asarray(y)
    oof = np.zeros(len(y), float)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        oof[va] = _score(_fit(X[tr], y[tr], metric), X[va], metric)
    return oof


def three_estimators(metric, y_tr, y_va, y_te, Xf, Xe, sid_te, mask=None):
    """The full comparison for one (arm, target).

    Xf / Xe are dicts split -> feature matrix for the floor and the floor+ECG
    model. Returns one dict per estimator, each with floor / floor+ecg / delta.
    """
    if mask is None:
        mask = np.ones(len(sid_te), bool)
    yte = np.asarray(y_te)[mask]
    sid = np.asarray(sid_te)[mask]
    res = {}

    for tag, src, ysrc in (("valfit", "val", y_va), ("trainfit", "train", y_tr)):
        p0 = fit_eval(Xf[src], ysrc, Xf["test"], metric)[mask]
        p1 = fit_eval(Xe[src], ysrc, Xe["test"], metric)[mask]
        d, lo, hi, pg = boot_delta(sid, yte, p0, p1, metric)
        s = (lambda p: auc(yte, p)) if metric == "auc" else (lambda p: r2(yte, p))
        res[tag] = dict(floor=s(p0), floor_plus_ecg=s(p1), delta=d, lo=lo, hi=hi, p_gt0=pg)

    p0 = crossfit(np.asarray(Xf["test"])[mask], yte, sid, metric)
    p1 = crossfit(np.asarray(Xe["test"])[mask], yte, sid, metric)
    d, lo, hi, pg = boot_delta(sid, yte, p0, p1, metric)
    s = (lambda p: auc(yte, p)) if metric == "auc" else (lambda p: r2(yte, p))
    res["crossfit"] = dict(floor=s(p0), floor_plus_ecg=s(p1), delta=d, lo=lo, hi=hi, p_gt0=pg)
    return res


def fmt(res, tag):
    r = res[tag]
    return dict(**{f"{tag}_floor": round(r["floor"], 4),
                   f"{tag}_floor_plus_ecg": round(r["floor_plus_ecg"], 4),
                   f"{tag}_delta": round(r["delta"], 4),
                   f"{tag}_ci": f"[{r['lo']:+.4f}, {r['hi']:+.4f}]",
                   f"{tag}_p_gt0": round(r["p_gt0"], 3)})
