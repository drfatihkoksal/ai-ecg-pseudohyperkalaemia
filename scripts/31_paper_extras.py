"""Paper extras for the J Emerg Med submission (concept.md sec.0.5-0.8). CLEAN set only
(haemolysed index K >= 5.5: 529 pseudo vs 99 true), AI-ECG K from the FROZEN external HEEDB
model (outputs/heedb/mimic_ecgk_external.parquet). Nothing here changes the model.

  A  decision analysis with CIs: patient-cluster bootstrap (1,000) that REFITS the
     cross-fitted model and its <= 2 %-miss threshold in every replicate, so threshold
     uncertainty is inside the interval. Rule, floor, floor + AI-ECG; paired differences.
  B  decision curve analysis: "act" = treat the result as possibly real (redraw / treat);
     net benefit over threshold probabilities 1-30 %, cross-fitted probabilities.
  C  ROC curves (Figure 2) and DCA (Figure 3), print-ready PNG (600 dpi) + PDF.
  D  Table 1: characteristics of pseudo vs true (haemolysed) and the non-haemolysed true
     reference (index K < 6.5).
  E  flow-diagram counts (Figure 1).
Outputs: outputs/paper/.
"""
import os, sys
import numpy as np, pandas as pd, duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ECGDIR = os.environ.get("MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg/1.0")
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
PO = f"{OUT}/paper"; os.makedirs(PO, exist_ok=True)
sys.path.insert(0, f"{ROOT}/scripts")
from _evalutils import auc
SEED, NB, MISS = 20260713, 1000, 0.02
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C_AI, C_FLOOR, C_ALONE, C_RULE = "#2a78d6", "#1baf7a", "#eda100", "#eb6834"   # validated (light)

# ------------------------------------------------------------------ data
d = pd.read_parquet(f"{DATA}/pseudohyperk_index.parquet").merge(
    pd.read_parquet(f"{DATA}/pseudohyperk_features.parquet").drop(columns=["cr"]), on="specimen_id")
ext = pd.read_parquet(f"{OUT}/heedb/mimic_ecgk_external.parquet").drop_duplicates("study_id").set_index("study_id").pred_k
d["ecg_k"] = ext.reindex(d.ecg_study).values
d["k_prior_na"] = d.k_prior.isna().astype(int); d["egfr_na"] = d.egfr.isna().astype(int)
d["in_adm"] = d.hadm_id.notna().astype(int); d["khod"] = d.khod_rule.astype(int)
FLOOR = ["k", "k_prior", "k_prior_na", "egfr", "egfr_na", "khod", "in_adm"]
c = d[d.label.isin(["pseudo", "true"]) & d.ecg_k.notna()].reset_index(drop=True)
c["y"] = (c.label == "true").astype(int)
y, g = c.y.values, c.subject_id.values
print(f"CLEAN: {len(c)} events ({y.sum()} true, {(1-y).sum()} pseudo), {c.subject_id.nunique()} patients")


def Xm(s, cols):
    M = s[cols].astype(float); return M.fillna(M.median()).values


def model(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def crossfit(X, y, g):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def decide(X, y, g):
    """Out-of-fold 'no redraw' decisions at a threshold set on the training folds (<= MISS)."""
    spare = np.zeros(len(y), bool)
    for tr, te in GroupKFold(5).split(X, y, g):
        m = model().fit(X[tr], y[tr])
        p_tr = m.predict_proba(X[tr])[:, 1]
        thr = np.quantile(p_tr[y[tr] == 1], MISS)        # below thr -> "spurious, no redraw"
        spare[te] = m.predict_proba(X[te])[:, 1] < thr
    return spare


def rates(spare, y):
    return spare[y == 0].mean(), spare[y == 1].mean()


XF, XA = Xm(c, FLOOR), Xm(c, FLOOR + ["ecg_k"])
pf, pa = crossfit(XF, y, g), crossfit(XA, y, g)
sp_f, sp_a, sp_r = decide(XF, y, g), decide(XA, y, g), c.khod.values.astype(bool)

# ------------------------------------------------------------------ A  bootstrap with refit
rng = np.random.default_rng(SEED)
pats = c.subject_id.unique(); rows_of = c.groupby("subject_id").indices
B = []
for b in range(NB):
    pick = rng.choice(pats, len(pats), replace=True)
    ix = np.concatenate([rows_of[p] for p in pick])
    gb = np.concatenate([np.full(len(rows_of[p]), j) for j, p in enumerate(pick)])   # resampled clusters stay distinct
    yb = y[ix]
    if yb.sum() < 15: continue
    sf, sa = decide(XF[ix], yb, gb), decide(XA[ix], yb, gb)
    r_f, r_a, r_r = rates(sf, yb), rates(sa, yb), rates(sp_r[ix], yb)
    B.append(dict(spare_rule=r_r[0], miss_rule=r_r[1], spare_floor=r_f[0], miss_floor=r_f[1],
                  spare_ai=r_a[0], miss_ai=r_a[1], d_spare_ai_floor=r_a[0] - r_f[0], d_spare_ai_rule=r_a[0] - r_r[0],
                  d_miss_ai_floor=r_a[1] - r_f[1]))
    if (b + 1) % 200 == 0: print(f"  bootstrap {b+1}/{NB}", flush=True)
B = pd.DataFrame(B)
pt = {"rule": rates(sp_r, y), "floor": rates(sp_f, y), "ai": rates(sp_a, y)}
ci = lambda col: f"[{100*B[col].quantile(.025):.1f}, {100*B[col].quantile(.975):.1f}]"
n0, n1 = int((y == 0).sum()), int(y.sum())
A = pd.DataFrame([
    dict(strategy="Khodorkovsky rule (machine-read normal ECG + eGFR >= 60)",
         pseudo_spared=f"{int(sp_r[y==0].sum())}/{n0} ({100*pt['rule'][0]:.1f}%) {ci('spare_rule')}",
         true_missed=f"{int(sp_r[y==1].sum())}/{n1} ({100*pt['rule'][1]:.1f}%) {ci('miss_rule')}"),
    dict(strategy="Clinical model (no ECG), threshold for <=2% missed",
         pseudo_spared=f"{int(sp_f[y==0].sum())}/{n0} ({100*pt['floor'][0]:.1f}%) {ci('spare_floor')}",
         true_missed=f"{int(sp_f[y==1].sum())}/{n1} ({100*pt['floor'][1]:.1f}%) {ci('miss_floor')}"),
    dict(strategy="Clinical model + AI-ECG potassium, threshold for <=2% missed",
         pseudo_spared=f"{int(sp_a[y==0].sum())}/{n0} ({100*pt['ai'][0]:.1f}%) {ci('spare_ai')}",
         true_missed=f"{int(sp_a[y==1].sum())}/{n1} ({100*pt['ai'][1]:.1f}%) {ci('miss_ai')}")])
Ad = pd.DataFrame([
    dict(contrast="pseudo spared: AI model - clinical model", delta=f"{100*(pt['ai'][0]-pt['floor'][0]):+.1f} pp",
         ci=ci("d_spare_ai_floor"), p_gt0=round((B.d_spare_ai_floor > 0).mean(), 3)),
    dict(contrast="pseudo spared: AI model - rule", delta=f"{100*(pt['ai'][0]-pt['rule'][0]):+.1f} pp",
         ci=ci("d_spare_ai_rule"), p_gt0=round((B.d_spare_ai_rule > 0).mean(), 3)),
    dict(contrast="true missed: AI model - clinical model", delta=f"{100*(pt['ai'][1]-pt['floor'][1]):+.1f} pp",
         ci=ci("d_miss_ai_floor"), p_gt0=round((B.d_miss_ai_floor > 0).mean(), 3))])
A.to_csv(f"{PO}/decision_ci.csv", index=False); Ad.to_csv(f"{PO}/decision_ci_contrasts.csv", index=False)
B.to_csv(f"{PO}/decision_bootstrap_replicates.csv", index=False)

# ------------------------------------------------------------------ B  decision curve
prev = y.mean()
ths = np.round(np.arange(0.01, 0.301, 0.01), 2)
def nb(act, t): return act[y == 1].sum() / len(y) - act[y == 0].sum() / len(y) * t / (1 - t)
D = pd.DataFrame([dict(threshold=t,
                       redraw_all=prev - (1 - prev) * t / (1 - t), redraw_none=0.0,
                       rule=nb(~sp_r, t), clinical=nb(pf >= t, t), clinical_ai=nb(pa >= t, t),
                       redraws_avoided_per100_ai=100 * ((pa < t).sum() / len(y)),
                       redraws_avoided_per100_clinical=100 * ((pf < t).sum() / len(y))) for t in ths])
D.to_csv(f"{PO}/decision_curve.csv", index=False)

# ------------------------------------------------------------------ C  figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.6, 3.6))
ax.plot([0, 1], [0, 1], color="#c3c2bd", lw=0.7)   # chance line: light, solid -- distinct from the dotted index-K curve
curves = [("Clinical model + AI-ECG potassium", pa, C_AI, "-"), ("Clinical model", pf, C_FLOOR, "--"),
          ("AI-ECG potassium alone", c.ecg_k.values, C_ALONE, "-."), ("Index potassium alone", c.k.values, INK2, ":")]
for lab, s, col, ls in curves:
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=col, lw=1.6, ls=ls, label=f"{lab} (AUC {auc(y, s):.2f})")
ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity for true hyperkalemia")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.5)
ax.legend(loc="lower right", fontsize=6.5, frameon=False)
fig.tight_layout()
for ext_ in ("png", "pdf"): fig.savefig(f"{PO}/figure2_roc.{ext_}", dpi=600)
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(D.threshold * 100, D.redraw_all, color=INK2, lw=1.0, ls=(0, (4, 3)), label="Redraw all")
ax.plot(D.threshold * 100, D.redraw_none, color=INK, lw=0.8, label="Redraw none")
ax.plot(D.threshold * 100, D.rule, color=C_RULE, lw=1.6, ls="-.", label="Khodorkovsky rule")
ax.plot(D.threshold * 100, D.clinical, color=C_FLOOR, lw=1.6, ls="--", label="Clinical model")
ax.plot(D.threshold * 100, D.clinical_ai, color=C_AI, lw=1.8, label="Clinical model + AI-ECG potassium")
ax.set_xlabel("Threshold probability of true hyperkalemia (%)"); ax.set_ylabel("Net benefit")
ax.set_xlim(1, 30); ax.set_ylim(-0.02, max(prev, D.clinical_ai.max()) + 0.02); ax.grid(color=GRID, lw=0.5)
ax.legend(fontsize=6.5, frameon=False, loc="upper right")
fig.tight_layout()
for ext_ in ("png", "pdf"): fig.savefig(f"{PO}/figure3_dca.{ext_}", dpi=600)
plt.close(fig)

# ------------------------------------------------------------------ D  Table 1
con = duckdb.connect(); con.execute("PRAGMA threads=16")
t1 = d[d.label.isin(["pseudo", "true"]) | ((d.label == "true_nonhemolysed") & d.k_matched)].copy()
con.register("t1", t1[["specimen_id", "subject_id", "charttime", "ecg_study"]])
rep = " || ' | ' || ".join(f"coalesce(lower(trim(report_{i})), '')" for i in range(18))
extra = con.sql(f"""
WITH mm AS (SELECT CAST(study_id AS BIGINT) study_id, {rep} txt FROM read_csv_auto('{ECGDIR}/machine_measurements.csv', all_varchar=true)),
esrd AS (SELECT DISTINCT subject_id FROM read_csv_auto('{MIMIC}/hosp/diagnoses_icd.csv.gz', types={{'icd_code':'VARCHAR'}})
         WHERE icd_code IN ('N186','Z992','5856','V4511')),
icu AS (SELECT subject_id, CAST(intime AS TIMESTAMP) a, CAST(outtime AS TIMESTAMP) b FROM read_csv_auto('{MIMIC}/icu/icustays.csv.gz')),
adm AS (SELECT subject_id, CAST(admittime AS TIMESTAMP) a, CAST(dischtime AS TIMESTAMP) b, CAST(edregtime AS TIMESTAMP) ed
        FROM read_csv_auto('{MIMIC}/hosp/admissions.csv.gz'))
SELECT t1.specimen_id, t1.subject_id IN (SELECT subject_id FROM esrd) esrd,
  (mm.txt LIKE '%pacemaker%' OR mm.txt LIKE '%paced%') paced,
  (mm.txt LIKE '%atrial fibrillation%' OR mm.txt LIKE '%atrial flutter%') af,
  (mm.txt LIKE '%bundle branch block%' OR mm.txt LIKE '%iv conduction%' OR mm.txt LIKE '%intraventricular%') bbb,
  CASE WHEN EXISTS (SELECT 1 FROM icu WHERE icu.subject_id=t1.subject_id AND t1.charttime BETWEEN icu.a AND icu.b) THEN 'ICU'
       WHEN EXISTS (SELECT 1 FROM adm WHERE adm.subject_id=t1.subject_id AND t1.charttime BETWEEN adm.a AND adm.b) THEN 'Ward'
       WHEN EXISTS (SELECT 1 FROM adm WHERE adm.subject_id=t1.subject_id AND adm.ed IS NOT NULL AND t1.charttime BETWEEN adm.ed AND adm.a) THEN 'ED, then admitted'
       ELSE 'ED/outpatient, not admitted' END setting
FROM t1 LEFT JOIN mm ON mm.study_id = t1.ecg_study
""").fetchdf()
t1 = t1.merge(extra, on="specimen_id")
t1["grp"] = t1.label.map({"pseudo": "Pseudohyperkalemia (hemolyzed)", "true": "True hyperkalemia (hemolyzed)",
                           "true_nonhemolysed": "True hyperkalemia (non-hemolyzed, K <6.5)"})
t1["female"] = t1.gender.eq("F")


def med(s, dec=1): s = s.dropna(); return f"{s.median():.{dec}f} ({s.quantile(.25):.{dec}f}-{s.quantile(.75):.{dec}f})"
def pct(s): s = s.fillna(False).astype(bool); return f"{int(s.sum())} ({100*s.mean():.1f})"


rows = {}
for grp, s in t1.groupby("grp"):
    rows[grp] = {
        "Events, n (patients)": f"{len(s)} ({s.subject_id.nunique()})",
        "Age, y, median (IQR)": med(s.age, 0), "Female, n (%)": pct(s.female),
        "Index potassium, mmol/L, median (IQR)": med(s.k),
        "Repeat potassium, mmol/L, median (IQR)": med(s.k_rep),
        "Hours to repeat, median (IQR)": med(s.rep_after_h),
        "Prior potassium within 48 h available, n (%)": pct(s.k_prior.notna()),
        "Prior potassium, mmol/L, median (IQR)": med(s.k_prior),
        "eGFR, mL/min/1.73 m2, median (IQR)": med(s.egfr, 0), "eGFR >=60, n (%)": pct(s.egfr >= 60),
        "ESRD or dialysis, n (%)": pct(s.esrd),
        **{f"Setting: {k}, n (%)": pct(s.setting.eq(k)) for k in ("ICU", "Ward", "ED, then admitted", "ED/outpatient, not admitted")},
        "ECG minutes from draw, median (IQR)": med(s.ecg_gap_min, 0), "ECG recorded before draw, n (%)": pct(s.ecg_gap_min < 0),
        "Paced rhythm, n (%)": pct(s.paced), "Atrial fibrillation/flutter, n (%)": pct(s.af),
        "Bundle-branch block/IV conduction delay, n (%)": pct(s.bbb),
        "Machine-read 'normal ECG', n (%)": pct(s.ecg_normal),
        "No machine-read hyperkalemic feature, n (%)": pct(s.ecg_no_hyperk),
        "Khodorkovsky rule positive, n (%)": pct(s.khod_rule),
        "AI-ECG potassium, mmol/L, median (IQR)": med(pd.Series(ext.reindex(s.ecg_study).values), 2),
        "K-lowering therapy before repeat (broad), n (%)": pct(s.ther_broad)}
T1 = pd.DataFrame(rows)[["Pseudohyperkalemia (hemolyzed)", "True hyperkalemia (hemolyzed)", "True hyperkalemia (non-hemolyzed, K <6.5)"]]
T1.to_csv(f"{PO}/table1.csv")

# ------------------------------------------------------------------ E  flow counts
lab = pd.read_parquet(f"{DATA}/pseudohyperk_index.parquet")
fc = con.sql(f"""SELECT COUNT(*) FILTER (WHERE c LIKE '%hemoly%' AND c NOT LIKE '%not hemoly%') hemolysed,
  COUNT(*) FILTER (WHERE c LIKE '%hemoly%' AND c NOT LIKE '%not hemoly%' AND v >= 5.5) hemolysed_ge55
  FROM (SELECT lower(coalesce(comments,'')) c, valuenum v FROM read_parquet('{DATA}/labs_k_comments.parquet') WHERE valuenum BETWEEN 1 AND 12)""").fetchone()
F = pd.Series({"Potassium results flagged hemolyzed": fc[0], "... with potassium >=5.5 mmol/L": fc[1],
               "... with a 12-lead ECG within 2 h (index events)": int(lab.hemo.sum()),
               **{f"   {k}": int(v) for k, v in lab[lab.hemo].label.value_counts().items()},
               "Analyzed (ECG passed quality control)": len(c),
               "Non-hemolyzed confirmed true hyperkalemia, index K <6.5 (reference)": int(((lab.label == "true_nonhemolysed") & lab.k_matched).sum())})
F.to_csv(f"{PO}/flow_counts.csv", header=["n"])

pd.set_option("display.width", 220, "display.max_colwidth", 80)
print("\n=== A  decision analysis (CLEAN; 95% CI = patient bootstrap with model + threshold refit) ===")
print(A.to_string(index=False)); print(Ad.to_string(index=False))
print(f"   ({len(B)} valid bootstrap replicates)")
print("\n=== B  decision curve (selected thresholds) ===")
print(D[D.threshold.isin([0.02, 0.05, 0.10, 0.15, 0.20])].round(4).to_string(index=False))
print("\n=== D  Table 1 ===\n" + T1.to_string())
print("\n=== E  flow counts ===\n" + F.to_string())
print(f"\nwrote {PO}/: decision_ci*.csv, decision_curve.csv, figure2_roc.(png|pdf), figure3_dca.(png|pdf), table1.csv, flow_counts.csv")
