"""v3 stage 5: does the AI-ECG potassium estimate help decide whether a haemolysed high
potassium is real? (concept.md sec.0.5; analyses fixed before this script was run)

Inputs: labels (20), Khodorkovsky features (23), cross-fitted AI-ECG K (22: every index /
prior ECG scored by a model that never saw its patient).

Outcome: TRUE hyperkalaemia (1) vs PSEUDO (0).
Sets: CLEAN    haemolysed indices only (pseudo vs true)          -- the clinical decision
      PRIMARY  pseudo vs true + true_nonhemolysed (index K < 6.5) -- powered ECG question
Floor (no AI): index K, prior K (48 h, non-haemolysed; + missing flag), eGFR (+ missing
      flag), Khodorkovsky rule, in-admission flag (hadm_id present; ED/outpatient proxy --
      threat #1 in sec.0.5).
AI:   ECG-K = cross-fitted AI-ECG potassium of the index ECG.

Analyses
  A1  AUC: ECG-K alone; floor; floor + ECG-K (patient-grouped 5-fold cross-fit combiner);
      delta with patient-cluster bootstrap (2,000).
  A2  CLEAN, decision: at <= 2 % of true hyperkalaemia missed (the Khodorkovsky rule's
      observed miss rate), the share of pseudo cases spared a repeat. The threshold is
      chosen on the training folds and applied to the held-out fold (no in-sample
      optimism). Compared with the rule (48/530 = 9.1 %) and with the floor model.
  A3  prior-ECG change (secondary): among indices with a prior ECG, add
      dECG-K = ECG-K(index) - ECG-K(prior ECG > 12 h earlier).
  Sensitivity: broad-therapy labels (label_sens), ECG within 1 h, index K 5.5-5.9,
      first index per patient.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
sys.path.insert(0, f"{ROOT}/scripts")
from _evalutils import auc
NB, SEED, MISS_MAX = 2000, 20260713, 0.02
ap = argparse.ArgumentParser()
ap.add_argument("--ecgk", default=f"{OUT}/v3/ecgk_oof.parquet",
                help="AI-ECG K source: MIMIC cross-fitted (default) or outputs/heedb/mimic_ecgk_external.parquet")
ap.add_argument("--tag", default="", help="suffix for output files, e.g. _external")
A = ap.parse_args()
print(f"AI-ECG K source: {A.ecgk}")

d = pd.read_parquet(f"{DATA}/pseudohyperk_index.parquet").merge(
    pd.read_parquet(f"{DATA}/pseudohyperk_features.parquet").drop(columns=["cr"]), on="specimen_id")
o = pd.read_parquet(A.ecgk)[["study_id", "pred_k"]].drop_duplicates("study_id").set_index("study_id").pred_k
d["ecg_k"] = o.reindex(d.ecg_study).values
d["prior_ecg_k"] = o.reindex(d.prior_ecg_study).values
d["d_ecg_k"] = d.ecg_k - d.prior_ecg_k
d["k_prior_na"] = d.k_prior.isna().astype(int)
d["egfr_na"] = d.egfr.isna().astype(int)
d["in_adm"] = d.hadm_id.notna().astype(int)
d["khod"] = d.khod_rule.astype(int)
FLOOR = ["k", "k_prior", "k_prior_na", "egfr", "egfr_na", "khod", "in_adm"]

d["y_main"] = np.where(d.label.isin(["true", "true_nonhemolysed"]), 1, np.where(d.label == "pseudo", 0, np.nan))
d["y_sens"] = np.where(d.label_sens == "true", 1, np.where(d.label_sens == "pseudo", 0, np.nan))
miss_ecg = d.ecg_k.isna() & d.label.isin(["pseudo", "true", "true_nonhemolysed"])
print(f"index events in analysis labels without an AI-ECG score (ECG failed QC): {int(miss_ecg.sum())}")


def X(s, cols):
    M = s[cols].astype(float)
    return M.fillna(M.median()).values


def crossfit_proba(Xm, y, g, k=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(k).split(Xm, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xm[tr], y[tr])
        p[te] = m.predict_proba(Xm[te])[:, 1]
    return p


def spared_at_miss(Xm, y, g, k=5):
    """Threshold chosen on training folds so that <= MISS_MAX of TRUE fall below it;
    applied to the held-out fold. Returns (pseudo spared, true missed) counts."""
    spared, missed = 0, 0
    for tr, te in GroupKFold(k).split(Xm, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xm[tr], y[tr])
        ptr, pte = m.predict_proba(Xm[tr])[:, 1], m.predict_proba(Xm[te])[:, 1]
        thr = np.quantile(ptr[y[tr] == 1], MISS_MAX)          # below thr -> "pseudo, no repeat"
        spared += int(((pte < thr) & (y[te] == 0)).sum()); missed += int(((pte < thr) & (y[te] == 1)).sum())
    return spared, missed


def boot(sid, y, a, b):
    order = np.argsort(sid, kind="stable"); sid, y, a, b = sid[order], y[order], a[order], b[order]
    st = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]); L0 = np.diff(np.r_[st, len(sid)])
    rng, out = np.random.default_rng(SEED), []
    for _ in range(NB):
        c = rng.integers(0, len(st), len(st)); L = L0[c]
        ix = np.repeat(st[c] - np.r_[0, np.cumsum(L)[:-1]], L) + np.arange(L.sum())
        out.append(auc(y[ix], b[ix]) - auc(y[ix], a[ix]))
    out = np.array(out)
    return np.mean(out), np.percentile(out, 2.5), np.percentile(out, 97.5)


def analyse(name, s, ycol="y_main"):
    s = s[s[ycol].notna() & s.ecg_k.notna()].reset_index(drop=True)
    y, g = s[ycol].values.astype(int), s.subject_id.values
    if y.sum() < 15 or (1 - y).sum() < 15:
        return dict(analysis=name, n=len(s), true=int(y.sum()), pseudo=int((1 - y).sum()), note="too few")
    pf = crossfit_proba(X(s, FLOOR), y, g)
    pa = crossfit_proba(X(s, FLOOR + ["ecg_k"]), y, g)
    dd, lo, hi = boot(g, y, pf, pa)
    return dict(analysis=name, n=len(s), true=int(y.sum()), pseudo=int((1 - y).sum()),
                auc_ecgk_alone=round(auc(y, s.ecg_k.values), 3), auc_index_k=round(auc(y, s.k.values), 3),
                auc_floor=round(auc(y, pf), 3), auc_floor_ecgk=round(auc(y, pa), 3),
                delta=round(dd, 3), ci=f"[{lo:+.3f}, {hi:+.3f}]")


CLEAN = d.label.isin(["pseudo", "true"])
PRIM = CLEAN | ((d.label == "true_nonhemolysed") & d.k_matched)
first = d.sort_values("charttime").groupby("subject_id").head(1).index
R = [analyse("CLEAN", d[CLEAN]), analyse("PRIMARY", d[PRIM]),
     analyse("CLEAN, broad-therapy labels", d[d.hemo & d.label_sens.notna()], "y_sens"),
     analyse("PRIMARY, ECG within 1 h", d[PRIM & (d.ecg_gap_min.abs() <= 60)]),
     analyse("PRIMARY, index K 5.5-5.9", d[PRIM & (d.k < 6.0)]),
     analyse("PRIMARY, first index per patient", d[PRIM & d.index.isin(first)])]
R = pd.DataFrame(R)

# A2: decision at the rule's miss rate, CLEAN
s = d[CLEAN & d.ecg_k.notna()].reset_index(drop=True)
y, g = s.y_main.values.astype(int), s.subject_id.values
rule_sp, rule_ms = int(((s.khod == 1) & (y == 0)).sum()), int(((s.khod == 1) & (y == 1)).sum())
fl_sp, fl_ms = spared_at_miss(X(s, FLOOR), y, g)
ai_sp, ai_ms = spared_at_miss(X(s, FLOOR + ["ecg_k"]), y, g)
n0, n1 = int((y == 0).sum()), int(y.sum())
A2 = pd.DataFrame([
    dict(strategy="Khodorkovsky rule (machine normal ECG + eGFR>=60)", pseudo_spared=f"{rule_sp}/{n0} ({rule_sp/n0:.1%})", true_missed=f"{rule_ms}/{n1} ({rule_ms/n1:.1%})"),
    dict(strategy="floor model, threshold at <=2% missed (cross-fitted)", pseudo_spared=f"{fl_sp}/{n0} ({fl_sp/n0:.1%})", true_missed=f"{fl_ms}/{n1} ({fl_ms/n1:.1%})"),
    dict(strategy="floor + AI-ECG K, threshold at <=2% missed (cross-fitted)", pseudo_spared=f"{ai_sp}/{n0} ({ai_sp/n0:.1%})", true_missed=f"{ai_ms}/{n1} ({ai_ms/n1:.1%})")])

# A3: prior-ECG change, among indices with a scored prior ECG
A3 = []
for name, m in (("CLEAN", CLEAN), ("PRIMARY", PRIM)):
    s = d[m & d.ecg_k.notna() & d.d_ecg_k.notna() & d.y_main.notna()].reset_index(drop=True)
    y, g = s.y_main.values.astype(int), s.subject_id.values
    if y.sum() < 15: A3.append(dict(set=name, n=len(s), true=int(y.sum()), note="too few")); continue
    p1 = crossfit_proba(X(s, FLOOR + ["ecg_k"]), y, g)
    p2 = crossfit_proba(X(s, FLOOR + ["ecg_k", "d_ecg_k"]), y, g)
    dd, lo, hi = boot(g, y, p1, p2)
    A3.append(dict(set=name, n=len(s), true=int(y.sum()), auc_dECGk_alone=round(auc(y, s.d_ecg_k.values), 3),
                   auc_floor_ecgk=round(auc(y, p1), 3), auc_plus_dECGk=round(auc(y, p2), 3),
                   delta=round(dd, 3), ci=f"[{lo:+.3f}, {hi:+.3f}]"))
A3 = pd.DataFrame(A3)

pd.set_option("display.width", 250)
print("\n=== A1  AUC for TRUE vs PSEUDO hyperkalaemia (combiner cross-fitted by patient) ===")
print(R.to_string(index=False))
print("\n=== A2  CLEAN: pseudo cases spared a repeat at <= 2 % true hyperkalaemia missed ===")
print(A2.to_string(index=False))
print("\n=== A3  prior-ECG change (secondary) ===")
print(A3.to_string(index=False))
os.makedirs(f"{OUT}/v3", exist_ok=True)
R.to_csv(f"{OUT}/v3/eval_auc{A.tag}.csv", index=False); A2.to_csv(f"{OUT}/v3/eval_decision{A.tag}.csv", index=False)
A3.to_csv(f"{OUT}/v3/eval_prior_ecg{A.tag}.csv", index=False)
print(f"\nwrote {OUT}/v3/eval_auc{A.tag}.csv, eval_decision{A.tag}.csv, eval_prior_ecg{A.tag}.csv")
