"""Stage 9b: the dCr regression arm, evaluated properly.

The floor analysis (stage 7) said the classification arm was never where the ECG
had room: the floor already explains AUC 0.725 of AKI but only R^2=0.06 of dCr.
The regression arm is the one that can carry a real renal signal, and it is the
quantity sec.5's PID needs.

So this asks, for dCr:
  1. incremental R^2 over the clinical floor (floor vs floor + dECG), on test
  2. SIGNAL-TO-NOISE: the spread the model emits on REAL pairs vs on NULL pairs
     (same patient ~3 h apart, nothing changed). If those spreads are similar, the
     model's output is mostly per-acquisition nuisance -- the exact failure sec.5
     item 3 predicted differencing would amplify.
  3. the same, for dK -- because sec.5's whole question is whether the renal signal
     survives once potassium is accounted for.
  4. the head-to-head that sec.9 contribution 1 actually rests on: does a
     DIFFERENCE arm beat the STATIC comparator on the delta target?

The first version of this script fit the floor+ECG ridge on TRAIN with the
network's in-sample train predictions. On a network this overfit (emb_diff:
r(dCr)=0.41 train vs 0.13 test) that pushes the combined model below the floor and
manufactures a negative delta. Primary estimator is now VAL-fit, with the test
cross-fit as sensitivity and the old train-fit kept, labelled, for audit. See
scripts/_evalutils.py.
"""
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

import importlib.util
from _evalutils import (ROOT, DATA, OUT, FLOOR, ARMS, r2, boot_delta,
                        three_estimators, fmt, crossfit, clusters)

spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

df = pd.read_parquet(f"{DATA}/pairs_final.parquet")
df = df[df.aki.notna()].copy()
df["aki"] = df.aki.astype(int)
df["sex"] = (df.gender == "F").astype(int)
for t in T.TARGETS: df[f"{t}_z"] = 0.0
SPLITS = {s: df[df.split == s] for s in ("train", "val", "test")}
tr, va, te = SPLITS["train"], SPLITS["val"], SPLITS["test"]
Xf = {s: d[FLOOR].fillna(0).values for s, d in SPLITS.items()}

nul = pd.read_parquet(f"{DATA}/null_pairs_final.parquet")
for t in T.TARGETS:
    if t not in nul: nul[t] = 0.0
    nul[f"{t}_z"] = 0.0
nul["aki"] = 0.0


def predict(model, d, arm, scale, use_beats):
    dl = DataLoader(T.PairDS(d, arm, scale, use_beats), batch_size=256,
                    shuffle=False, num_workers=8)
    return T.predict(model, dl)


rows = []
head2head = {}                      # arm -> cross-fit dCr prediction on test
for arm in ARMS:
    ck = torch.load(f"{DATA}/models/{arm}.pt", map_location="cuda", weights_only=False)
    m = T.Net(arm).to("cuda"); m.load_state_dict(ck["state"]); m.eval()
    scale, use_beats, mu, sd = ck["scale"], (arm == "beat_sub"), ck["mu"], ck["sd"]

    R = {s: predict(m, d, arm, scale, use_beats)[0] for s, d in SPLITS.items()}
    rnu, _ = predict(m, nul, arm, scale, use_beats)

    for j, tgt in enumerate(["dcr", "dk"]):
        P = {s: R[s][:, j] * sd[tgt] + mu[tgt] for s in SPLITS}
        p_nu = rnu[:, j] * sd[tgt] + mu[tgt]
        Xe = {s: np.column_stack([Xf[s], P[s]]) for s in SPLITS}

        for sub, mask in (("all", np.ones(len(te), bool)), ("anchor", te.is_anchor.values)):
            res = three_estimators("r2", tr[tgt].values, va[tgt].values, te[tgt].values,
                                   Xf, Xe, te.subject_id.values, mask=mask)
            snr = P["test"][mask].std() / p_nu.std() if p_nu.std() > 0 else np.nan
            rows.append(dict(arm=arm, target=tgt, subset=sub,
                             r_pred_true=round(np.corrcoef(P["test"][mask],
                                                           te[tgt].values[mask])[0, 1], 3),
                             **fmt(res, "valfit"), **fmt(res, "crossfit"),
                             **fmt(res, "trainfit"),
                             sd_pred_real=round(float(P["test"][mask].std()), 3),
                             sd_pred_null=round(float(p_nu.std()), 3),
                             snr=round(float(snr), 2)))
            print(f"  [{arm}/{tgt}/{sub}] valfit dR2={res['valfit']['delta']:+.4f}  "
                  f"crossfit dR2={res['crossfit']['delta']:+.4f}  "
                  f"(old trainfit {res['trainfit']['delta']:+.4f})", flush=True)

        if tgt == "dcr":
            head2head[arm] = crossfit(np.column_stack([Xf["test"], P["test"]]),
                                      te.dcr.values, te.subject_id.values, "r2")

r = pd.DataFrame(rows)
show = ["arm", "subset", "r_pred_true", "valfit_floor", "valfit_floor_plus_ecg",
        "valfit_delta", "valfit_ci", "valfit_p_gt0", "sd_pred_real", "sd_pred_null", "snr"]
print("\n=== dCr / dK REGRESSION: incremental R^2 over the floor (PRIMARY: fit on val) ===")
for tgt in ("dcr", "dk"):
    for sub in ("all", "anchor"):
        print(f"\n  -- target={tgt}  test subset={sub}")
        print(r[(r.target == tgt) & (r.subset == sub)][show].to_string(index=False))

print("\n=== SENSITIVITY: combiner cross-fit on test ===")
print(r[["arm", "target", "subset", "crossfit_delta", "crossfit_ci",
         "crossfit_p_gt0"]].to_string(index=False))
print("\n=== SUPERSEDED: old train-fit estimator (in-sample stacking, biased low) ===")
print(r[["arm", "target", "subset", "trainfit_delta", "trainfit_ci",
         "trainfit_p_gt0"]].to_string(index=False))

# ---- sec.9 contribution 1: difference arm vs the static comparator, same target,
#      same combiner procedure. The only comparison that earns the thesis.
print("\n=== HEAD-TO-HEAD on dCr (cross-fit on test, cluster-bootstrapped by patient) ===")
floor_oof = crossfit(Xf["test"], te.dcr.values, te.subject_id.values, "r2")
head2head["floor"] = floor_oof
y = te.dcr.values
for name in ("floor", "static_ecg2", "beat_sub", "emb_diff", "siamese"):
    if name in head2head:
        print(f"  {name:<12s} R2 = {r2(y, head2head[name]):.4f}")
print()
for a, b in (("siamese", "static_ecg2"), ("emb_diff", "static_ecg2"),
             ("siamese", "floor"), ("static_ecg2", "floor")):
    if a in head2head and b in head2head:
        d, lo, hi, pg = boot_delta(te.subject_id.values, y, head2head[b], head2head[a], "r2")
        print(f"  {a} - {b:<14s} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p(>0)={pg:.3f}")

print("""
  snr = sd(prediction on REAL pairs) / sd(prediction on NULL pairs).
  Null pairs are the same patient ~3 h apart, where essentially nothing changed.
  snr ~ 1 means the model emits as much "change" when nothing changed as when
  something did. Read it with care: both spreads also contain a patient-level
  component the model reads off ECG2, so snr ~ 1 bounds the nuisance share from
  above rather than proving the output IS nuisance -- the incremental R^2 above is
  the test that decides that.
""")
r.to_csv(f"{OUT}/incremental_dcr.csv", index=False)
print(f"wrote {OUT}/incremental_dcr.csv")
