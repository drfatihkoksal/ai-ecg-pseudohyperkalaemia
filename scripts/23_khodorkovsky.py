"""v3 stage 4: the published clinical comparator -- Khodorkovsky et al. 2014 (J Emerg Med,
doi:10.1016/j.jemermed.2014.04.019): in ED patients with a haemolysed K >= 5.5, eGFR >= 60
plus a NORMAL ECG had NPV 100 % (45 patients) for true hyperkalaemia, so the repeat was
proposed to be unnecessary. Disputed by Cervellin et al. 2016.

Their "normal ECG" was a physician reading; here it is approximated from the MIMIC-IV-ECG
machine report (machine_measurements.csv), fixed before any outcome was examined:
  ecg_normal   (PRIMARY) a report line "Normal ECG" or "Normal ECG except for rate".
               ECGs with no global statement count as NOT normal (conservative: the rule
               then says "repeat").
  ecg_no_hyperk (SENSITIVITY) no hyperkalaemic feature: QRS < 120 ms (machine onset/end),
               no "peaked"/"tall T"/"hyperkal" statement, no A-V block, no junctional
               rhythm, no pacemaker rhythm, heart rate >= 50 (RR <= 1200 ms).
  egfr         CKD-EPI 2021 (race-free) from the nearest creatinine within +-12 h of the
               index draw; missing -> rule negative.
  khod_rule    eGFR >= 60 AND ecg_normal          -> "pseudohyperkalaemia, no repeat"
  khod_rule_b  eGFR >= 60 AND ecg_no_hyperk

Reports, per analysis set, what the rule claims: among rule-positive results, how many are
TRUE hyperkalaemia (the misses; Khodorkovsky: 0/42) -> NPV for true hyperK with exact
Clopper-Pearson CI; and how many pseudo cases the rule would spare a repeat.
Writes data/pseudohyperk_features.parquet (specimen_id + the columns above).
"""
import os
import numpy as np, pandas as pd, duckdb
from scipy.stats import beta

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ECGDIR = os.environ.get("MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg/1.0")
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
MISSING = 29999  # MIMIC-IV-ECG sentinel for an unmeasured fiducial point

con = duckdb.connect(); con.execute("PRAGMA threads=16")
rep = " || ' | ' || ".join(f"coalesce(lower(trim(report_{i})), '')" for i in range(18))
con.execute(f"""
CREATE TEMP TABLE mm AS
SELECT CAST(study_id AS BIGINT) study_id, {rep} AS txt,
  TRY_CAST(rr_interval AS DOUBLE) rr, TRY_CAST(qrs_onset AS DOUBLE) qon, TRY_CAST(qrs_end AS DOUBLE) qend
FROM read_csv_auto('{ECGDIR}/machine_measurements.csv', all_varchar=true)
""")
con.execute(f"CREATE TEMP TABLE ix AS SELECT * FROM read_parquet('{DATA}/pseudohyperk_index.parquet')")
con.execute(f"""CREATE TEMP TABLE pat AS SELECT subject_id, gender, anchor_age, anchor_year
  FROM read_csv_auto('{MIMIC}/hosp/patients.csv.gz')""")
con.execute(f"""
CREATE TEMP TABLE feat AS
WITH j AS (
  SELECT ix.specimen_id, ix.cr, p.gender,
    p.anchor_age + (EXTRACT(year FROM ix.charttime) - p.anchor_year) AS age,
    mm.txt, mm.rr,
    CASE WHEN mm.qon < {MISSING} AND mm.qend < {MISSING} THEN mm.qend - mm.qon END AS qrs_ms
  FROM ix JOIN pat p USING(subject_id) LEFT JOIN mm ON mm.study_id = ix.ecg_study
)
SELECT specimen_id, age, gender, cr, qrs_ms, rr, txt,
  (txt LIKE '%normal ecg%' AND txt NOT LIKE '%abnormal ecg%') AS ecg_normal,
  (txt LIKE '%normal ecg%' OR txt LIKE '%borderline ecg%' OR txt LIKE '%abnormal ecg%') AS has_global_statement,
  (coalesce(qrs_ms, 999) < 120
   AND txt NOT LIKE '%peaked%' AND txt NOT LIKE '%tall t%' AND txt NOT LIKE '%hyperkal%'
   AND txt NOT LIKE '%a-v block%' AND txt NOT LIKE '%av block%' AND txt NOT LIKE '%junctional%'
   AND txt NOT LIKE '%pacemaker%' AND txt NOT LIKE '%paced%'
   AND coalesce(rr, 9999) <= 1200) AS ecg_no_hyperk,
  CASE WHEN cr IS NOT NULL THEN
    142.0 * pow(LEAST(cr / (CASE WHEN gender='F' THEN 0.7 ELSE 0.9 END), 1.0),
                (CASE WHEN gender='F' THEN -0.241 ELSE -0.302 END))
          * pow(GREATEST(cr / (CASE WHEN gender='F' THEN 0.7 ELSE 0.9 END), 1.0), -1.200)
          * pow(0.9938, age) * (CASE WHEN gender='F' THEN 1.012 ELSE 1.0 END) END AS egfr
FROM j
""")
f = con.sql("SELECT * FROM feat").fetchdf()
f["ecg_normal"] = f.ecg_normal.fillna(False).astype(bool)
f["ecg_no_hyperk"] = f.ecg_no_hyperk.fillna(False).astype(bool)
f["egfr_ge60"] = (f.egfr >= 60).fillna(False)
f["khod_rule"] = f.egfr_ge60 & f.ecg_normal
f["khod_rule_b"] = f.egfr_ge60 & f.ecg_no_hyperk
f.drop(columns=["txt"]).to_parquet(f"{DATA}/pseudohyperk_features.parquet", index=False)

d = pd.read_parquet(f"{DATA}/pseudohyperk_index.parquet").merge(f.drop(columns=["cr"]), on="specimen_id")


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


SETS = {
    "CLEAN   (haemolysed: pseudo vs true)": d.label.isin(["pseudo", "true"]),
    "PRIMARY (pseudo vs true + true_nonhemolysed K<6.5)":
        (d.label.isin(["pseudo", "true"])) | ((d.label == "true_nonhemolysed") & d.k_matched),
}
print("=== coverage (all index events) ===")
print(f"  machine report found: {d.txt.notna().mean():.1%}   global statement: {d.has_global_statement.mean():.1%}   "
      f"QRS measured: {d.qrs_ms.notna().mean():.1%}   eGFR computable: {d.egfr.notna().mean():.1%}")
rows = []
for sname, m in SETS.items():
    s = d[m]
    is_true = s.label.isin(["true", "true_nonhemolysed"])
    for rule in ("khod_rule", "khod_rule_b"):
        pos = s[rule].values
        n_pos, k_true_pos = int(pos.sum()), int((pos & is_true.values).sum())
        npv = 1 - k_true_pos / n_pos if n_pos else np.nan
        lo, hi = cp(n_pos - k_true_pos, n_pos) if n_pos else (np.nan, np.nan)
        rows.append(dict(set=sname, rule=rule, n=len(s), n_true=int(is_true.sum()), n_pseudo=int((~is_true).sum()),
                         rule_positive=n_pos, true_among_rule_pos=k_true_pos,
                         NPV_for_true_hyperK=f"{npv:.3f} [{lo:.3f}, {hi:.3f}]",
                         pseudo_spared_repeat=f"{int((pos & ~is_true.values).sum())}/{int((~is_true).sum())} "
                                              f"({(pos & ~is_true.values).sum() / max((~is_true).sum(), 1):.1%})",
                         true_missed=f"{k_true_pos}/{int(is_true.sum())} ({k_true_pos / max(is_true.sum(), 1):.1%})"))
    print(f"\n=== {sname}: component rates ===")
    print(s.assign(grp=np.where(is_true, "true", "pseudo")).groupby("grp")[
        ["egfr_ge60", "ecg_normal", "ecg_no_hyperk", "has_global_statement", "khod_rule", "khod_rule_b"]].mean()
        .round(3).to_string())
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n=== Khodorkovsky rule: 'eGFR >= 60 and normal ECG -> pseudohyperkalaemia, no repeat' ===")
print("    (original report: 0 true hyperkalaemia among 42 rule-positive, NPV 100 % [93.1, 100])")
print(R.to_string(index=False))
R.to_csv(f"{OUT}/khodorkovsky_rule.csv", index=False)
print(f"\nwrote {DATA}/pseudohyperk_features.parquet, {OUT}/khodorkovsky_rule.csv")
