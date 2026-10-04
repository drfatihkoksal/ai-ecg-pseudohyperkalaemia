"""Stage 10: is the potassium channel a useful mediator, or a dead end?

Stage 9/9b establish that d-ECG reads dK strongly (dR^2 ~ +0.10) and dCr weakly but
significantly (dR^2 ~ +0.014). For the K-mediated story, a second link must hold:

        d-ECG  --(shown, strong)-->  dK  --(?)-->  AKI / dCr

If dK itself adds nothing to the clinical floor, then the ECG's ability to read
potassium is clinically inert for this endpoint -- a perfect dK reading would still
predict no AKI -- and the "potassium-mediated renal signal" framing collapses into
"the ECG reads potassium, which is a known result, and that is all".

So we test each link with the SAME machinery (fit on VAL, evaluate on TEST,
cluster-bootstrap by patient):

  L1  floor + TRUE dK              -> does potassium change matter at all?
  L2  floor + ECG-PREDICTED dK     -> does the ECG's estimate of it matter?
  L3  floor + TRUE dK + ECG        -> once true dK is known, does the ECG add
                                      ANYTHING further? If yes, there is a non-K
                                      channel in the ECG. If no, K is the whole story
                                      and sec.5's decisive question is answered:
                                      no unique dCr information with dK held fixed.

L3 is the direct, model-based version of the question sec.5 poses for PID. It is not
a substitute for the PID decomposition, but if L3 is flat, PID will not rescue it.

FIT SPLIT: every feature set is fit on VAL, where the network's outputs are
out-of-sample, and scored on TEST. The earlier version fit on TRAIN using the
network's in-sample train predictions, which systematically penalises any set
containing an ECG feature -- i.e. it biased L2 and L3 (the ECG-bearing sets) down
relative to L1 (which has none), which is exactly the contrast this script exists
to measure. The test cross-fit is printed as a sensitivity check.

Also reported with baseline potassium (k1) in the floor, since a clinician knows the
current potassium too -- the honest bar is "known Cr AND known K".
"""
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

import importlib.util
from _evalutils import (ROOT, DATA, OUT, FLOOR, auc, r2, boot_delta,
                        fit_eval, crossfit)

spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

ARM = "emb_diff"                      # the strongest dK reader

df = pd.read_parquet(f"{DATA}/pairs_final.parquet")
df = df[df.aki.notna()].copy()
df["aki"] = df.aki.astype(int)
df["sex"] = (df.gender == "F").astype(int)
for t in T.TARGETS: df[f"{t}_z"] = 0.0

# ---- ECG predictions from the strongest dK-reading arm
ck = torch.load(f"{DATA}/models/{ARM}.pt", map_location="cuda", weights_only=False)
m = T.Net(ARM).to("cuda"); m.load_state_dict(ck["state"]); m.eval()
mu, sd = ck["mu"], ck["sd"]

SPLITS = {}
for s in ("val", "test"):
    d = df[df.split == s].copy()
    dl = DataLoader(T.PairDS(d, ARM, ck["scale"], False), batch_size=256,
                    shuffle=False, num_workers=8)
    reg, logit = T.predict(m, dl)
    d["ecg_dcr"] = reg[:, 0] * sd["dcr"] + mu["dcr"]
    d["ecg_dk"] = reg[:, 1] * sd["dk"] + mu["dk"]
    d["ecg_logit"] = logit
    SPLITS[s] = d
va, te = SPLITS["val"], SPLITS["test"]

# feature sets: each is the floor plus something
SETS = {
    "floor (no ECG, no K)":                    FLOOR,
    "floor + known K at t1":                   FLOOR + ["k"],
    "L1  floor + TRUE dK":                     FLOOR + ["k", "dk"],
    "L2  floor + ECG-PREDICTED dK":            FLOOR + ["k", "ecg_dk"],
    "     floor + full ECG output":            FLOOR + ["k", "ecg_dk", "ecg_dcr", "ecg_logit"],
    "L3  floor + TRUE dK + full ECG output":   FLOOR + ["k", "dk", "ecg_dk", "ecg_dcr", "ecg_logit"],
}

print(f"=== Does the potassium channel actually lead anywhere? (arm={ARM}) ===")
print("    fit on VAL, scored on TEST; cross-fit-on-test shown as sensitivity\n")
for target, metric in (("aki", "auc"), ("dcr", "r2")):
    print(f"--- target: {target.upper()}  ({'AUC' if metric == 'auc' else 'R^2'}) ---")
    y = te[target].values
    yfit = va[target].values
    scores, cf_scores, base_pred, rows = {}, {}, None, []
    for name, cols in SETS.items():
        Xv, Xt = va[cols].fillna(0).values, te[cols].fillna(0).values
        p = fit_eval(Xv, yfit, Xt, metric)
        pc = crossfit(Xt, y, te.subject_id.values, metric)
        sc = auc(y, p) if metric == "auc" else r2(y, p)
        sc_c = auc(y, pc) if metric == "auc" else r2(y, pc)
        scores[name], cf_scores[name] = p, pc
        if base_pred is None:
            base_pred, base_cf = p, pc
            rows.append(dict(model=name, score=round(sc, 4), delta_vs_floor="—", ci="—",
                             p_gt0="—", crossfit_score=round(sc_c, 4), crossfit_delta="—"))
        else:
            d, lo, hi, pg = boot_delta(te.subject_id.values, y, base_pred, p, metric)
            dc, _, _, _ = boot_delta(te.subject_id.values, y, base_cf, pc, metric)
            rows.append(dict(model=name, score=round(sc, 4), delta_vs_floor=f"{d:+.4f}",
                             ci=f"[{lo:+.4f}, {hi:+.4f}]", p_gt0=round(pg, 3),
                             crossfit_score=round(sc_c, 4), crossfit_delta=f"{dc:+.4f}"))
    print(pd.DataFrame(rows).to_string(index=False))

    # the decisive contrast: L3 vs L1 -- does the ECG add anything ON TOP of true dK?
    L1, L3 = "L1  floor + TRUE dK", "L3  floor + TRUE dK + full ECG output"
    d, lo, hi, pg = boot_delta(te.subject_id.values, y, scores[L1], scores[L3], metric)
    dc, loc, hic, pgc = boot_delta(te.subject_id.values, y, cf_scores[L1], cf_scores[L3], metric)
    print(f"\n  >> DECISIVE (sec.5): ECG on top of floor+TRUE dK  =  {d:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  p(>0)={pg:.3f}")
    print(f"     sensitivity (cross-fit on test)               =  {dc:+.4f} "
          f"[{loc:+.4f}, {hic:+.4f}]  p(>0)={pgc:.3f}")
    print("     (if ~0: with dK held fixed, the ECG carries no further renal information)\n")
