"""Stage 9: judge the four arms against the bar that was fixed BEFORE training.

Nothing here reports a bare AUC. Two things are asked of every arm:

  1. INCREMENTAL VALUE. The clinical floor (known creatinine + demographics + dt +
     dHR, no ECG at all) already reaches AUC 0.725. The only question that matters
     is whether the d-ECG adds anything ON TOP -- with the CI on the DIFFERENCE,
     cluster-bootstrapped by patient (pairs are nested in patients, ICC=0.37).

     WHERE THE COMBINER IS FIT IS NOT A DETAIL. The first version of this script
     fit floor+ECG on TRAIN, feeding it the network's own in-sample train
     predictions. The network is heavily overfit (emb_diff: r(dCr)=0.41 train vs
     0.13 test), so the combiner over-weighted the ECG feature and the combined
     model then lost to the floor on test. That produced a negative delta for
     every arm -- an artefact that reads exactly like a null result. See
     scripts/_evalutils.py. The primary estimator is now VAL-fit; the test
     cross-fit is reported alongside it and must agree; the old train-fit number
     is kept in the CSV, labelled, so the correction is auditable.

  2. IS IT ACTUALLY A DIFFERENCE MODEL? The pre-registered ablations:
       B1  ECG1 <- another patient's ECG1 (target untouched). If the arm does not
           drop, it never used ECG1: a static model in a difference costume, and
           concept.md sec.2/sec.9-1 are void.
       B2  ECG2 <- another patient's ECG2. Must drop hard, else B1 proves nothing.
       B3  swap ECG1<->ECG2. A true difference model is antisymmetric: predicted
           dCr must flip sign.
     Plus the NULL PAIRS (same patient, ~3 h apart, nothing changed): the arm must
     not hallucinate change there.

  The static_ecg2 arm is the control that decides whether the paper's thesis holds.
"""
import os
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

import importlib.util
from _evalutils import (ROOT, DATA, OUT, FLOOR, ARMS, auc, boot_delta,
                        three_estimators, fmt, fit_eval)

spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)


def load_arm(arm):
    ck = torch.load(f"{DATA}/models/{arm}.pt", map_location="cuda", weights_only=False)
    m = T.Net(arm).to("cuda"); m.load_state_dict(ck["state"]); m.eval()
    return m, ck


def predict_pairs(model, df, arm, scale, use_beats, i1="wf_idx1", i2="wf_idx2"):
    ds = T.PairDS(df, arm, scale, use_beats, idx1=i1, idx2=i2)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8)
    return T.predict(model, dl)


# ------------------------------------------------------------------
df = pd.read_parquet(f"{DATA}/pairs_final.parquet")
df = df[df.aki.notna()].copy()
df["aki"] = df.aki.astype(int)
df["sex"] = (df.gender == "F").astype(int)
for t in T.TARGETS:
    df[f"{t}_z"] = 0.0        # PairDS needs the columns; values unused at predict time
SPLITS = {s: df[df.split == s] for s in ("train", "val", "test")}
tr, va, te = SPLITS["train"], SPLITS["val"], SPLITS["test"]

abl = pd.read_parquet(f"{DATA}/ablation_index.parquet")
nul = pd.read_parquet(f"{DATA}/null_pairs_final.parquet")
for t in T.TARGETS:
    if t not in nul: nul[t] = 0.0
    nul[f"{t}_z"] = 0.0
nul["aki"] = 0.0

Xf = {s: d[FLOOR].fillna(0).values for s, d in SPLITS.items()}

print("CLINICAL FLOOR (no ECG), fit on val, scored on test:")
te_floor = fit_eval(Xf["val"], va.aki, Xf["test"], "auc")
print(f"  test   AUC = {auc(te.aki, te_floor):.3f}")
print(f"  anchor AUC = {auc(te.aki.values[te.is_anchor.values], te_floor[te.is_anchor.values]):.3f}\n")

rows, abl_rows = [], []
for arm in ARMS:
    if not os.path.exists(f"{DATA}/models/{arm}.pt"):
        print(f"[skip] {arm}: not trained yet"); continue
    model, ck = load_arm(arm)
    scale, use_beats = ck["scale"], (arm == "beat_sub")
    mu, sd = ck["mu"], ck["sd"]

    # ---------- predictions on every split (val and test are out-of-sample;
    #            train is needed only to reproduce the old biased number)
    P = {}
    for s, d in SPLITS.items():
        reg, logit = predict_pairs(model, d, arm, scale, use_beats)
        P[s] = dict(logit=logit, dcr=reg[:, 0] * sd["dcr"] + mu["dcr"])

    ecg_auc = auc(te.aki, P["test"]["logit"])
    r_dcr = np.corrcoef(P["test"]["dcr"], te.dcr)[0, 1]

    # ---------- incremental over the floor
    Xe = {s: np.column_stack([Xf[s], P[s]["logit"], P[s]["dcr"]]) for s in SPLITS}
    for sub, mask in (("all", np.ones(len(te), bool)), ("anchor", te.is_anchor.values)):
        res = three_estimators("auc", tr.aki.values, va.aki.values, te.aki.values,
                               Xf, Xe, te.subject_id.values, mask=mask)
        rows.append(dict(arm=arm, subset=sub,
                         ecg_only_auc=round(auc(te.aki.values[mask],
                                                P["test"]["logit"][mask]), 3),
                         r_dcr=round(r_dcr, 3),
                         **fmt(res, "valfit"), **fmt(res, "crossfit"),
                         **fmt(res, "trainfit")))
        print(f"  [{arm}/{sub}] valfit d={res['valfit']['delta']:+.4f}  "
              f"crossfit d={res['crossfit']['delta']:+.4f}  "
              f"(old trainfit d={res['trainfit']['delta']:+.4f})", flush=True)

    # ---------- ablations, on the test rows only (no combiner -> unaffected by the bug)
    at = abl[abl.split == "test"].merge(
        te[["subject_id", "study1", "study2", "aki", "dcr"]], on=["subject_id", "study1", "study2"],
        suffixes=("", "_y"))
    at = at.reset_index(drop=True)
    for t in T.TARGETS:
        if t not in at: at[t] = 0.0
        at[f"{t}_z"] = 0.0

    def run(i1, i2):
        r, l = predict_pairs(model, at, arm, scale, use_beats, i1=i1, i2=i2)
        return r[:, 0] * sd["dcr"] + mu["dcr"], l

    p_orig_dcr, l_orig = run("wf_idx1", "wf_idx2")
    p_b1, l_b1 = run("abl1_wf_idx1", "abl1_wf_idx2")     # ECG1 <- donor
    p_b2, l_b2 = run("abl2_wf_idx1", "abl2_wf_idx2")     # ECG2 <- donor
    p_b3, l_b3 = run("swap_wf_idx1", "swap_wf_idx2")     # swapped

    a0 = auc(at.aki, l_orig)
    abl_rows.append(dict(
        arm=arm,
        auc_intact=round(a0, 3),
        auc_B1_ecg1_swapped=round(auc(at.aki, l_b1), 3),
        drop_B1=round(a0 - auc(at.aki, l_b1), 3),
        auc_B2_ecg2_swapped=round(auc(at.aki, l_b2), 3),
        drop_B2=round(a0 - auc(at.aki, l_b2), 3),
        antisym_B3=round(np.corrcoef(p_b3, -p_orig_dcr)[0, 1], 3),
    ))

    # ---------- null pairs: does it hallucinate change where there is none?
    nr, nl = predict_pairs(model, nul, arm, scale, use_beats)
    npred = nr[:, 0] * sd["dcr"] + mu["dcr"]
    print(f"[{arm}] null-pair predicted dCr: mean={npred.mean():+.3f} sd={npred.std():.3f}"
          f"   (true sd there = {nul.dcr.std():.3f}; real pairs sd = {te.dcr.std():.3f})")

r = pd.DataFrame(rows)
show = ["arm", "subset", "ecg_only_auc", "valfit_floor", "valfit_floor_plus_ecg",
        "valfit_delta", "valfit_ci", "valfit_p_gt0"]
print("\n=== INCREMENTAL VALUE OVER THE CLINICAL FLOOR (PRIMARY: combiner fit on val) ===")
print("    delta = AUC(floor + dECG) - AUC(floor) on test. This is the paper's endpoint.")
for sub in ("all", "anchor"):
    print(f"\n  -- test subset: {sub}")
    print(r[r.subset == sub][show].to_string(index=False))

print("\n=== SENSITIVITY: same delta, combiner cross-fit on test (no val involvement) ===")
print(r[["arm", "subset", "crossfit_delta", "crossfit_ci", "crossfit_p_gt0"]].to_string(index=False))

print("\n=== SUPERSEDED: the old train-fit estimator (in-sample stacking, biased low) ===")
print("    Reported here only so the correction is auditable. Do not cite these.")
print(r[["arm", "subset", "trainfit_delta", "trainfit_ci", "trainfit_p_gt0"]].to_string(index=False))

print("\n=== FALSIFICATION: is it really a difference model? (test set) ===")
print("    drop_B1 ~ 0  =>  ECG1 unused  =>  static model in disguise, sec.2 void")
print("    antisym_B3 near +1 => prediction flips sign when the ECGs are swapped")
print(pd.DataFrame(abl_rows).to_string(index=False))

r.to_csv(f"{OUT}/incremental.csv", index=False)
pd.DataFrame(abl_rows).to_csv(f"{OUT}/ablations.csv", index=False)
print(f"\nwrote {OUT}/incremental.csv, {OUT}/ablations.csv")
