"""Stage 11 (v2): the re-posed questions, judged with the same machinery as stage 9.

concept.md sec.0 explains why the v1 endpoint (prevalent-AKI state at t2) could not
test the delta thesis. v2 asks, in pre-registered order:

  Q1  PRIMARY. With the last potassium KNOWN, does the ECG change tell how much
      potassium changed -- and does it do so better than a single ECG?
        Q1a  dK regression,       floor_K + pred_dK                     (R^2)
        Q1b  new hyperkalaemia,   k1 < 5.0 -> k2 >= 5.5, floor_K + pred_dK (AUC)
        Q1c  head-to-head: floor_K + difference arm  vs  floor_K + static_ecg2,
             same rows, paired patient bootstrap. THIS is the sec.2 thesis test.
      Bar: floor_K includes k1. v1 reported ECG->dK without k1 in the floor, and k1
      alone explains R^2 ~0.32 of dK (regression to the mean), so the v1 +0.096
      was inflated.

  Q2  SECONDARY, MECHANISTIC (expected negative). Among pairs at risk at t1, does
      d-ECG add to INCIDENT AKI on (t1, t2]?
        Q2a  floor_K + ECG outputs
        Q2b  decisive (sec.5): floor_K + TRUE dK  ->  + ECG outputs
        Q2c  ischaemia sensitivity: floor_K + troponin  ->  + ECG outputs

  Q3  SECONDARY, EARLY WARNING (sec.4). Among pairs at risk at the t2 draw, does
      d-ECG add to AKI onset within 48 h after t2, over everything known at t2?

  Q4  FALSIFICATION for the Q1 signal: B1/B2 donor swaps, B3 antisymmetry on
      predicted dK, and null pairs (same patient, 0.5-6 h apart).

Estimators (see _evalutils.py): `valfit` PRIMARY (combiner fit on val, scored on
test); `crossfit` sensitivity (patient-grouped 5-fold on test). Every CI is a
patient-cluster bootstrap of the DIFFERENCE. Subsets: all test pairs, anchor (T5).
Pre-registered sensitivity: dt_h <= 48.
"""
import os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
from _evalutils import ROOT, DATA, OUT, ARMS, auc, r2, boot_delta, fit_eval, crossfit

spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

FLOOR_K = ["cr", "k", "age", "sex", "dt_h", "dhr"]
FLOOR_T2 = FLOOR_K + ["cr2", "k2"]                 # everything known at the t2 draw
TROP = ["ltrop1", "ltrop2", "trop1_na", "trop2_na"]
ECG_K = ["pred_dk"]
ECG_ALL = ["logit", "pred_dcr", "pred_dk"]
DIFF_ARMS = ["beat_sub", "emb_diff", "siamese"]
KEY = ["study1", "study2"]

# ------------------------------------------------------------------ data
df = pd.read_parquet(f"{DATA}/pairs_v2.parquet")
df["sex"] = (df.gender == "F").astype(int)
for i in (1, 2):
    t = df[f"trop{i}"]
    df[f"trop{i}_na"] = t.isna().astype(int)
    df[f"ltrop{i}"] = np.log(t.fillna(0) + 0.01)
df = df[df.split.isin(["val", "test"])].reset_index(drop=True)

P = {}
for arm in ARMS:
    f = f"{OUT}/preds_{arm}.parquet"
    if not os.path.exists(f):
        print(f"[skip] {arm}: no predictions"); continue
    p = pd.read_parquet(f).drop(columns=["subject_id", "split"])
    m = df[KEY].merge(p, on=KEY, how="left")
    miss = m.pred_dk.isna().sum()
    assert miss == 0, f"{arm}: {miss} val/test pairs have no prediction -> stale file, retrain"
    P[arm] = m
arms = list(P)
va_m, te_m = (df.split == "val").values, (df.split == "test").values
te = df[te_m].reset_index(drop=True)
SUBSETS = {"all": np.ones(len(te), bool), "anchor": te.is_anchor.values.astype(bool),
           "dt<=48": (te.dt_h <= 48).values}


def X(cols, arm=None):
    """Feature matrix over val+test rows; ECG columns come from the arm's predictions."""
    parts = [df[[c for c in cols if c in df]].astype(float)]
    ecg = [c for c in cols if c not in df]
    if ecg: parts.append(P[arm][ecg].astype(float))
    M = pd.concat(parts, axis=1)
    return M.fillna(M[va_m].median()).values


def compare(metric, y, rows, cols0, cols1, arm0=None, arm1=None):
    """Nested comparison model0 -> model1. rows: boolean mask over val+test (defines the
    endpoint's eligible population). Yields one dict per (subset, estimator)."""
    y = np.asarray(y, float)
    X0, X1 = X(cols0, arm0), X(cols1, arm1)
    fv, ft = va_m & rows, te_m & rows
    p0v = fit_eval(X0[fv], y[fv], X0[te_m], metric)          # scored on ALL test rows,
    p1v = fit_eval(X1[fv], y[fv], X1[te_m], metric)          # subset applied below
    elig = rows[te_m]
    yt, sid = y[te_m], te.subject_id.values
    s = (lambda yy, p: auc(yy, p)) if metric == "auc" else (lambda yy, p: r2(yy, p))
    out = []
    for sub, sm in SUBSETS.items():
        m = sm & elig
        if m.sum() < 50 or (metric == "auc" and yt[m].sum() < 10):
            continue
        ests = {"valfit": (p0v[m], p1v[m])}
        ests["crossfit"] = (crossfit(X0[te_m][m], yt[m], sid[m], metric),
                            crossfit(X1[te_m][m], yt[m], sid[m], metric))
        for est, (a, b) in ests.items():
            d, lo, hi, pg = boot_delta(sid[m], yt[m], a, b, metric)
            out.append(dict(subset=sub, estimator=est, n=int(m.sum()),
                            events=int(yt[m].sum()) if metric == "auc" else None,
                            base=round(s(yt[m], a), 4), model=round(s(yt[m], b), 4),
                            delta=round(d, 4), ci=f"[{lo:+.4f}, {hi:+.4f}]", p_gt0=round(pg, 3)))
    return out


rows_all = np.ones(len(df), bool)
hk_rows = df.hyperk_new.notna().values
inc_rows = df.aki_incident.notna().values
early_rows = df.aki_early.notna().values

R, H = [], []
def add(q, arm, comparison, res):
    for r in res: R.append(dict(question=q, arm=arm, comparison=comparison, **r))

for arm in arms:
    print(f"... {arm}", flush=True)
    add("Q1a dK", arm, "floor_K -> +ECG", compare("r2", df.dk, rows_all, FLOOR_K, FLOOR_K + ECG_K, arm1=arm))
    add("Q1b new hyperK", arm, "floor_K -> +ECG",
        compare("auc", df.hyperk_new.fillna(0), hk_rows, FLOOR_K, FLOOR_K + ECG_K, arm1=arm))
    add("Q2a incident AKI", arm, "floor_K -> +ECG",
        compare("auc", df.aki_incident.fillna(0), inc_rows, FLOOR_K, FLOOR_K + ECG_ALL, arm1=arm))
    add("Q2b incident AKI", arm, "floor_K+true dK -> +ECG",
        compare("auc", df.aki_incident.fillna(0), inc_rows, FLOOR_K + ["dk"],
                FLOOR_K + ["dk"] + ECG_ALL, arm0=arm, arm1=arm))
    add("Q2c incident AKI", arm, "floor_K+trop -> +ECG",
        compare("auc", df.aki_incident.fillna(0), inc_rows, FLOOR_K + TROP,
                FLOOR_K + TROP + ECG_ALL, arm0=arm, arm1=arm))
    add("Q3 early AKI 48h", arm, "floor_t2 -> +ECG",
        compare("auc", df.aki_early.fillna(0), early_rows, FLOOR_T2, FLOOR_T2 + ECG_ALL, arm1=arm))

# Q1c: difference arm vs static arm, both on top of floor_K, same rows
if "static_ecg2" in P:
    for arm in [a for a in DIFF_ARMS if a in P]:
        for q, metric, y, rows, ecg in (
                ("Q1a dK", "r2", df.dk, rows_all, ECG_K),
                ("Q1b new hyperK", "auc", df.hyperk_new.fillna(0), hk_rows, ECG_K),
                # is the incident-AKI signal difference-specific, or does ECG2 alone carry it?
                ("Q2a incident AKI", "auc", df.aki_incident.fillna(0), inc_rows, ECG_ALL)):
            for r in compare(metric, y, rows, FLOOR_K + ecg, FLOOR_K + ecg,
                             arm0="static_ecg2", arm1=arm):
                H.append(dict(question=q, diff_arm=arm, **r))

# ------------------------------------------------------------------ Q4 falsification
abl = pd.read_parquet(f"{DATA}/ablation_index.parquet")
abl = abl[abl.split == "test"].merge(te[KEY + ["dk"]], on=KEY).reset_index(drop=True)
nul = pd.read_parquet(f"{DATA}/null_pairs_final.parquet")
F = []
for arm in arms:
    ckf = f"{DATA}/models/{arm}.pt"
    ck = torch.load(ckf, map_location="cuda", weights_only=False)
    model = T.Net(arm).to("cuda"); model.load_state_dict(ck["state"]); model.eval()
    sd, mu, use_beats = ck["sd"], ck["mu"], arm == "beat_sub"

    def pdk(d, i1="wf_idx1", i2="wf_idx2"):
        d = d.copy()
        for t in T.TARGETS: d[f"{t}_z"] = 0.0
        d["aki_incident"] = np.nan
        dl = DataLoader(T.PairDS(d, arm, ck["scale"], use_beats, idx1=i1, idx2=i2),
                        batch_size=256, shuffle=False, num_workers=8)
        reg, _ = T.predict(model, dl)
        return reg[:, 1] * sd["dk"] + mu["dk"]

    p0 = pdk(abl)
    rr = lambda p: np.corrcoef(p, abl.dk)[0, 1]
    pn = pdk(nul)
    F.append(dict(arm=arm, r_dk_intact=round(rr(p0), 3),
                  r_dk_B1_ecg1_donor=round(rr(pdk(abl, "abl1_wf_idx1", "abl1_wf_idx2")), 3),
                  r_dk_B2_ecg2_donor=round(rr(pdk(abl, "abl2_wf_idx1", "abl2_wf_idx2")), 3),
                  antisym_B3=round(np.corrcoef(pdk(abl, "swap_wf_idx1", "swap_wf_idx2"),
                                               -(p0 - mu["dk"]))[0, 1], 3),
                  null_pred_dk_sd=round(pn.std(), 3), null_true_dk_sd=round(nul.dk.std(), 3),
                  real_pred_dk_sd=round(p0.std(), 3), null_r_dk=round(np.corrcoef(pn, nul.dk)[0, 1], 3)))

# ------------------------------------------------------------------ report
R, H, F = pd.DataFrame(R), pd.DataFrame(H), pd.DataFrame(F)
pd.set_option("display.width", 220)
cols = ["arm", "comparison", "subset", "n", "events", "base", "model", "delta", "ci", "p_gt0"]
for q in R.question.unique():
    print(f"\n=== {q}  (PRIMARY estimator: valfit) ===")
    print(R[(R.question == q) & (R.estimator == "valfit")][cols].to_string(index=False))
print("\n=== HEAD-TO-HEAD (Q1c, + Q2a): floor_K+difference arm minus floor_K+static_ecg2 (valfit) ===")
print("    delta > 0 with CI excluding 0  =>  the intra-patient difference beats a single ECG")
if len(H):
    print(H[H.estimator == "valfit"][["question", "diff_arm", "subset", "n", "events", "base",
                                      "model", "delta", "ci", "p_gt0"]].to_string(index=False))
print("\n=== crossfit sensitivity (must agree in sign with valfit) ===")
print(R[R.estimator == "crossfit"][["question", "arm", "comparison", "subset", "delta", "ci"]]
      .to_string(index=False))
print("\n=== Q4 FALSIFICATION on predicted dK (test) ===")
print("    B1: ECG1 from another patient -> r should fall for a difference model")
print("    B3: swapping ECG1/ECG2 should flip the predicted dK (antisym near +1)")
print("    null pairs: predicted dK spread should be well below the real-pair spread")
print(F.to_string(index=False))

R.to_csv(f"{OUT}/v2_results.csv", index=False)
H.to_csv(f"{OUT}/v2_head_to_head.csv", index=False)
F.to_csv(f"{OUT}/v2_falsification.csv", index=False)
print(f"\nwrote {OUT}/v2_results.csv, v2_head_to_head.csv, v2_falsification.csv")
