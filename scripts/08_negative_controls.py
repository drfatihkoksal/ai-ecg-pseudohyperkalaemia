"""Stage 7: the falsification suite. Built BEFORE any model, so the bar is fixed
in advance and cannot drift to fit whatever the model happens to score.

Three controls, each killing a different failure mode:

(A) NULL PAIRS (same patient, 0.5-6 h apart). dCr is ~0 by construction. A model
    reading a genuine renal signal must predict ~no change here. If it emits large
    |dCr|, it is reading per-acquisition nuisance -- exactly what sec.5 item 3
    warns differencing amplifies. These pairs are NOT in the main manifest (which
    requires dt>=12h), so they are built here and their ECGs preprocessed.

(B) MISMATCHED PAIRS (two DIFFERENT patients). This is the control that tests the
    paper's actual thesis. sec.2 claims the intra-patient difference cancels the
    identity channel and leaves an acute-change residual. If a model does just as
    well on cross-patient pairs -- where no such cancellation can occur -- then it
    never used the difference structure; it is a static "read absolute state off
    ECG2" model wearing a difference costume, and sec.9 contribution 1 is void.
    Built here as a reproducible sampling procedure, dt-matched to the real pairs.

(C) THE CLINICAL FLOOR (no waveform at all). Predict AKI / dCr from what a
    clinician already has for free: dt, dHR, age, sex, and -- crucially -- the
    KNOWN creatinine at t1. Serum creatinine is not a hidden variable in the ward;
    the only interesting question is whether d-ECG adds anything OVER it. This
    runs today, needs no model, and sets the number the whole paper must beat.
    Reviewers will ask for it; better we ask first.
"""
import os, duckdb, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ECGDIR = os.environ.get("MIMIC_ECG_ROOT",
                        "/path/to/mimic-iv-ecg/1.0")
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
CREATININE, POTASSIUM = 50912, 50971


def hosp(name):
    for ext in (".csv", ".csv.gz"):
        p = f"{MIMIC}/hosp/{name}{ext}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{MIMIC}/hosp/{name}.csv[.gz]")


LAB_WIN_H = 12
NULL_MIN_H, NULL_MAX_H = 0.5, 6.0
SEED = 20260713
rng = np.random.default_rng(SEED)

con = duckdb.connect(); con.execute("PRAGMA threads=16")

# ============================================================ (A) NULL PAIRS
con.execute(f"""CREATE TEMP TABLE ecg AS SELECT subject_id, study_id,
  CAST(ecg_time AS TIMESTAMP) ecg_time, path FROM read_csv_auto('{ECGDIR}/record_list.csv')""")
con.execute(f"""CREATE TEMP TABLE adm AS SELECT subject_id, hadm_id,
  CAST(admittime AS TIMESTAMP) admittime, CAST(dischtime AS TIMESTAMP) dischtime
  FROM read_csv_auto('{hosp("admissions")}')""")
con.execute(f"CREATE TEMP TABLE labs AS SELECT * FROM read_parquet('{DATA}/labs_cr_k.parquet')")

con.execute(f"""
CREATE TEMP TABLE el AS
WITH ea AS (
  SELECT e.subject_id, e.study_id, e.ecg_time, e.path, a.hadm_id
  FROM ecg e JOIN adm a ON e.subject_id=a.subject_id
         AND e.ecg_time BETWEEN a.admittime AND a.dischtime
),
cr AS (SELECT ea.study_id,
         arg_min(l.valuenum, abs(date_diff('minute', ea.ecg_time, l.charttime))) cr
       FROM ea JOIN labs l ON l.subject_id=ea.subject_id AND l.itemid={CREATININE}
         AND l.charttime BETWEEN ea.ecg_time - INTERVAL {LAB_WIN_H} HOUR
                             AND ea.ecg_time + INTERVAL {LAB_WIN_H} HOUR
       GROUP BY 1),
k AS (SELECT ea.study_id,
        arg_min(l.valuenum, abs(date_diff('minute', ea.ecg_time, l.charttime))) k
      FROM ea JOIN labs l ON l.subject_id=ea.subject_id AND l.itemid={POTASSIUM}
         AND l.charttime BETWEEN ea.ecg_time - INTERVAL {LAB_WIN_H} HOUR
                             AND ea.ecg_time + INTERVAL {LAB_WIN_H} HOUR
      GROUP BY 1)
SELECT ea.*, cr.cr, k.k FROM ea JOIN cr USING(study_id) JOIN k USING(study_id)
""")

con.execute(f"""
CREATE TEMP TABLE null_pairs AS
WITH seq AS (
  SELECT *, LEAD(ecg_time) OVER w t2, LEAD(study_id) OVER w study2, LEAD(path) OVER w path2,
            LEAD(cr) OVER w cr2, LEAD(k) OVER w k2
  FROM el WINDOW w AS (PARTITION BY hadm_id ORDER BY ecg_time)
)
SELECT subject_id, hadm_id, study_id AS study1, study2, path AS path1, path2,
       ecg_time AS t1, t2,
       date_diff('minute', ecg_time, t2)/60.0 AS dt_h,
       cr, cr2, cr2-cr AS dcr, k, k2, k2-k AS dk
FROM seq
WHERE t2 IS NOT NULL
  AND date_diff('minute', ecg_time, t2)/60.0 BETWEEN {NULL_MIN_H} AND {NULL_MAX_H}
""")

r = con.sql("""SELECT COUNT(*) n, COUNT(DISTINCT subject_id) s,
  ROUND(MEDIAN(dt_h),2) med_dt, ROUND(AVG(dcr),4) mean_dcr, ROUND(STDDEV(dcr),4) sd_dcr,
  ROUND(STDDEV(dk),4) sd_dk,
  ROUND(100.0*AVG(CASE WHEN abs(dcr) < 0.1 THEN 1 ELSE 0 END),1) pct_dcr_lt_0p1
  FROM null_pairs""").fetchdf().iloc[0]
print("=== (A) NULL PAIRS: same patient, 0.5-6 h apart ===")
print(f"  pairs={r.n:,.0f}  patients={r.s:,.0f}  median dt={r.med_dt} h")
print(f"  dCr: mean={r.mean_dcr:+.4f}  sd={r.sd_dcr:.4f}   ({r.pct_dcr_lt_0p1:.1f}% have |dCr|<0.1)")
print(f"  dK : sd={r.sd_dk:.4f}")
print("  -> this sd is the NOISE FLOOR: the spread a model may legitimately emit")
print("     when nothing has actually changed. Compare against sd(dCr)=0.66 in the")
print("     real pairs; the real signal must exceed this floor to mean anything.")

# how many of these ECGs are new (not already preprocessed)?
n_new = con.sql(f"""
WITH s AS (SELECT study1 AS sid, path1 AS p FROM null_pairs
           UNION SELECT study2, path2 FROM null_pairs)
SELECT COUNT(*) FROM s
WHERE sid NOT IN (SELECT study_id FROM read_parquet('{DATA}/waveforms/ecg_meta.parquet'))
""").fetchone()[0]
print(f"  ECGs needing extra preprocessing: {n_new:,}")

con.execute(f"COPY (SELECT * FROM null_pairs) TO '{DATA}/null_pairs.parquet' (FORMAT parquet)")

# ============================================================ (B) INPUT ABLATIONS
# NOT a separate cross-patient dataset. Building one would compare apples to pears:
# cross-patient dCr has sd ~2.4 vs ~0.66 within-patient, because different patients
# sit at different chronic creatinine levels. A static model would score WELL on
# such a set -- chronic CKD morphology is genuinely visible in the ECG -- so a good
# score there proves nothing, and the target distributions are not comparable.
#
# The clean design keeps the TARGET FIXED (real dCr, real pairs, real test set) and
# corrupts the INPUT. Any drop is then attributable to the input, not the target.
#
#   B1  ECG1 <- a random other patient's ECG1.  If performance does NOT drop, the
#       model ignores ECG1: it is a static "read state off ECG2" model and sec.2's
#       identity-cancellation claim is void. THIS IS THE DECISIVE ONE.
#   B2  ECG2 <- a random other patient's ECG2.  Expected to drop hard (ECG2 carries
#       the current state); serves as the positive control for B1's sensitivity.
#   B3  swap ECG1 <-> ECG2.  A true difference model is antisymmetric: predicted
#       dCr must flip sign. corr(pred_swapped, -pred_orig) should approach 1.
#
# We emit the index mappings so the perturbations are reproducible and pre-registered
# rather than invented after seeing the model's score.
f = con.sql(f"SELECT * FROM read_parquet('{DATA}/pairs_final.parquet')").fetchdf()
abl = []
for split, g in f.groupby("split"):
    g = g.reset_index(drop=True)
    n = len(g)
    # derangement-ish: draw a donor row from a DIFFERENT patient
    donor = rng.permutation(n)
    bad = g.subject_id.values[donor] == g.subject_id.values
    for _ in range(10):                      # re-draw the few self-collisions
        if not bad.any(): break
        donor[bad] = rng.permutation(n)[: bad.sum()]
        bad = g.subject_id.values[donor] == g.subject_id.values
    abl.append(pd.DataFrame(dict(
        split=split, subject_id=g.subject_id, study1=g.study1, study2=g.study2,
        wf_idx1=g.wf_idx1, wf_idx2=g.wf_idx2,
        # B1: ECG1 replaced by a donor patient's ECG1 (target untouched)
        abl1_wf_idx1=g.wf_idx1.values[donor], abl1_wf_idx2=g.wf_idx2,
        # B2: ECG2 replaced by a donor patient's ECG2
        abl2_wf_idx1=g.wf_idx1, abl2_wf_idx2=g.wf_idx2.values[donor],
        # B3: temporal swap
        swap_wf_idx1=g.wf_idx2, swap_wf_idx2=g.wf_idx1,
        donor_subject=g.subject_id.values[donor],
        dcr=g.dcr, aki=g.aki,
    )))
abl = pd.concat(abl, ignore_index=True)
abl.to_parquet(f"{DATA}/ablation_index.parquet", index=False)
self_collisions = int((abl.donor_subject == abl.subject_id).sum())
print(f"\n=== (B) INPUT ABLATIONS: target fixed, input corrupted ===")
print(f"  rows={len(abl):,}   donor-is-same-patient collisions={self_collisions}")
print("  B1 ECG1<-other patient : if AUC does NOT drop -> model ignores ECG1 -> static model, sec.2 void")
print("  B2 ECG2<-other patient : expected to drop hard  -> confirms B1 is sensitive enough to detect a drop")
print("  B3 swap ECG1<->ECG2    : predicted dCr must flip sign (antisymmetry)")

# ============================================================ (C) CLINICAL FLOOR
print("\n=== (C) CLINICAL FLOOR: what you get with NO waveform at all ===")
print("    (the number the d-ECG model has to beat to be worth anything)\n")

f = f[f.aki.notna()].copy()
f["sex"] = (f.gender == "F").astype(int)
tr = f[f.split == "train"]
te = f[f.split == "test"]

def auc(y, s):
    y = np.asarray(y).astype(int)
    npos, nneg = y.sum(), (1 - y).sum()
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def boot_auc(df, y, s, n_boot=300):
    d = pd.DataFrame(dict(sid=df.subject_id.values, y=np.asarray(y).astype(int), s=s))
    groups = [g for _, g in d.groupby("sid")]
    k = len(groups)
    out = []
    for _ in range(n_boot):
        b = pd.concat([groups[i] for i in rng.integers(0, k, k)])
        if b.y.nunique() < 2: continue
        out.append(auc(b.y, b.s))
    out = np.array(out)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

FEATSETS = {
    "dt only":                       ["dt_h"],
    "dt + dHR":                      ["dt_h", "dhr"],
    "demographics only":             ["age", "sex"],
    "known creatinine (cr1) only":   ["cr"],
    "cr1 + demographics":            ["cr", "age", "sex"],
    "cr1 + demo + dt + dHR  <- BAR": ["cr", "age", "sex", "dt_h", "dhr"],
}

rows_all = []
for name, cols in FEATSETS.items():
    Xtr, Xte = tr[cols].fillna(0).values, te[cols].fillna(0).values
    # --- AKI classification
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xtr, tr.aki.astype(int))
    p = m.predict_proba(Xte)[:, 1]
    a = auc(te.aki.astype(int), p)
    lo, hi = boot_auc(te, te.aki.astype(int), p)
    # --- dCr regression
    rg = make_pipeline(StandardScaler(), Ridge())
    rg.fit(Xtr, tr.dcr)
    pr = rg.predict(Xte)
    r2 = 1 - ((te.dcr - pr) ** 2).sum() / ((te.dcr - tr.dcr.mean()) ** 2).sum()
    rows_all.append(dict(features=name, aki_auc=round(a, 3),
                         ci=f"[{lo:.3f}, {hi:.3f}]", dcr_r2=round(r2, 4)))
print(pd.DataFrame(rows_all).to_string(index=False))

# same, on the anchor test set
print("\n  ... restricted to the ANCHOR (T5) test set:")
tea = te[te.is_anchor]
rows = []
for name, cols in FEATSETS.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(tr[cols].fillna(0).values, tr.aki.astype(int))
    p = m.predict_proba(tea[cols].fillna(0).values)[:, 1]
    a = auc(tea.aki.astype(int), p)
    lo, hi = boot_auc(tea, tea.aki.astype(int), p)
    rows.append(dict(features=name, aki_auc=round(a, 3), ci=f"[{lo:.3f}, {hi:.3f}]"))
print(pd.DataFrame(rows).to_string(index=False))

pd.DataFrame(rows).to_csv(f"{OUT}/clinical_floor.csv", index=False)

bar = [r for r in rows_all if r["features"].startswith("cr1 + demo")][0]["aki_auc"]
cr_only = [r for r in rows_all if r["features"].startswith("known creatinine")][0]["aki_auc"]
print(f"""
=== WHAT THIS MEANS FOR THE PAPER ===
  The floor is AUC {bar} with NO ECG (known creatinine + demographics + dt + dHR),
  and {cr_only} from the known creatinine ALONE. That is structural, not a fluke: the
  KDIGO label is a function of the creatinine trajectory, and yesterday's creatinine
  already encodes much of it.

  Consequence: a d-ECG model scoring AUC 0.75 is NOT a 0.75 result. It is a +0.025
  result over a model that never looked at an ECG. The paper's primary endpoint must
  therefore be INCREMENTAL -- d-ECG ON TOP OF the clinical floor -- and the headline
  comparison is (floor + dECG) vs (floor). Reporting a bare AUC would be, at best,
  uninformative and, at worst, misleading.

  The dCr REGRESSION arm is where the ECG has real room: the floor explains only
  R^2=0.06 of dCr. If d-ECG carries an independent renal signal, that is where it
  will show, and it is also exactly the quantity sec.5's PID decomposition needs.
""")
print(f"wrote {DATA}/null_pairs.parquet, {DATA}/ablation_index.parquet, {OUT}/clinical_floor.csv")
