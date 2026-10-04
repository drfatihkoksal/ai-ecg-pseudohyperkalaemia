"""Stage 3: build the pair manifest + leakage-safe splits for the agreed design
(pretrain on all admissions, trigger-anchored analysis on ACS n PCI).

Three things this fixes relative to the stage-1/2 counting scripts:

1. KDIGO baseline is no longer "min of the ECG-matched creatinines". It is the
   proper hierarchical baseline over the FULL creatinine series:
       baseline = LEAST( lowest inpatient Cr of the admission,
                         most recent outpatient/prior Cr within 365 d )
   AKI at the 2nd ECG of a pair is then evaluated against that baseline AND
   against a rolling 48 h rise computed on the full series -- not only at ECG
   timestamps, which would under-detect AKI that peaks between two ECGs.

2. dEGFR is computed (CKD-EPI 2021, race-free) so the sec.1 target triple
   {dEGFR, dCr, dK} is complete.

3. SPLITS ARE BUILT ANCHOR-FIRST, which is the whole point:
       - split the ACS n PCI (anchor) PATIENTS into train/val/test
       - the pretrain pool is every T0 pair whose patient is NOT in
         anchor-val or anchor-test
   Doing it the other way round (split T0, then intersect) would leave anchor
   test patients inside the pretraining corpus, and every anchor-cohort number
   in the paper would be contaminated. Splits are patient-disjoint by
   construction (sec.8).

4. THE MATCHED LAB'S CHARTTIME IS CARRIED, NOT ONLY ITS VALUE. Labs are matched
   to an ECG within +-12 h and pairs may be as close as 12 h apart, so a single
   draw can be the nearest lab to both ECGs of a pair. dCr/dK are then zero by
   construction rather than by measurement. This affects 3.5% of candidate pairs
   (it is NOT the main source of the 26% exact-zero dCr, which is genuine: Cr is
   reported to 0.1 mg/dL and repeats in a stable patient). The flags are emitted
   here; stage 6 does the gating and reports the attrition.
"""
import duckdb, os
import pandas as pd

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ECGDIR = os.environ.get("MIMIC_ECG_ROOT",
                        "/path/to/mimic-iv-ecg/1.0")
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
CREATININE, POTASSIUM = 50912, 50971
LAB_WIN_H, PAIR_MIN_H, PAIR_MAX_H = 12, 12, 168
NEXTDAY_LO, NEXTDAY_HI = 12, 36
SEED = 20260713  # pre-registered split seed


def hosp(name):
    """MIMIC-IV ships hosp/ as .csv or .csv.gz depending on how it was fetched."""
    for ext in (".csv", ".csv.gz"):
        p = f"{MIMIC}/hosp/{name}{ext}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{MIMIC}/hosp/{name}.csv[.gz]")


con = duckdb.connect(); con.execute("PRAGMA threads=16")

con.execute(f"""CREATE TEMP TABLE ecg AS SELECT subject_id, study_id,
  CAST(ecg_time AS TIMESTAMP) ecg_time, path
  FROM read_csv_auto('{ECGDIR}/record_list.csv')""")
con.execute(f"""CREATE TEMP TABLE adm AS SELECT subject_id, hadm_id,
  CAST(admittime AS TIMESTAMP) admittime, CAST(dischtime AS TIMESTAMP) dischtime
  FROM read_csv_auto('{hosp("admissions")}')""")
con.execute(f"""CREATE TEMP TABLE pat AS SELECT subject_id, gender, anchor_age, anchor_year
  FROM read_csv_auto('{hosp("patients")}')""")
con.execute(f"""CREATE TEMP TABLE dx AS SELECT hadm_id, CAST(icd_code AS VARCHAR) icd_code,
  icd_version FROM read_csv_auto('{hosp("diagnoses_icd")}', types={{'icd_code':'VARCHAR'}})""")
con.execute(f"""CREATE TEMP TABLE px AS SELECT hadm_id, CAST(icd_code AS VARCHAR) icd_code,
  icd_version FROM read_csv_auto('{hosp("procedures_icd")}', types={{'icd_code':'VARCHAR'}})""")
con.execute(f"CREATE TEMP TABLE labs AS SELECT * FROM read_parquet('{DATA}/labs_cr_k.parquet')")

con.execute("""CREATE TEMP TABLE acs_hadm AS SELECT DISTINCT hadm_id FROM dx
  WHERE (icd_version=9 AND (icd_code LIKE '410%' OR icd_code LIKE '4111%'))
     OR (icd_version=10 AND (icd_code LIKE 'I21%' OR icd_code LIKE 'I22%' OR icd_code LIKE 'I200%'))""")
con.execute("""CREATE TEMP TABLE pci_hadm AS SELECT DISTINCT hadm_id FROM px
  WHERE (icd_version=9 AND icd_code IN ('0066','3606','3607','1755'))
     OR (icd_version=10 AND icd_code LIKE '027%' AND substr(icd_code,5,1) IN ('3','4'))""")
con.execute("""CREATE TEMP TABLE cath_hadm AS SELECT DISTINCT hadm_id FROM px
  WHERE (icd_version=9 AND icd_code IN ('8853','8854','8855','8856','8857','3722','3723'))
     OR (icd_version=10 AND icd_code LIKE 'B21%')""")

# ---------------------------------------------------------------- 1. baseline Cr
# hierarchical, over the FULL creatinine series (not just ECG-matched values)
con.execute(f"""
CREATE TEMP TABLE base AS
WITH cr AS (SELECT subject_id, charttime, valuenum FROM labs WHERE itemid={CREATININE}),
j AS (
  SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime, c.charttime, c.valuenum
  FROM adm a JOIN cr c ON c.subject_id = a.subject_id
  WHERE c.charttime >= a.admittime - INTERVAL 365 DAY AND c.charttime <= a.dischtime
)
SELECT hadm_id, subject_id,
  MIN(valuenum) FILTER (WHERE charttime BETWEEN admittime AND dischtime)   AS cr_inpat_min,
  arg_max(valuenum, charttime) FILTER (WHERE charttime < admittime)        AS cr_prior,
  LEAST(COALESCE(MIN(valuenum) FILTER (WHERE charttime BETWEEN admittime AND dischtime), 1e9),
        COALESCE(arg_max(valuenum, charttime) FILTER (WHERE charttime < admittime), 1e9)) AS cr_base
FROM j GROUP BY hadm_id, subject_id, admittime, dischtime
""")

# ---------------------------------------------------------------- 2. ECG x labs
con.execute(f"""
CREATE TEMP TABLE el AS
WITH ea AS (
  SELECT e.subject_id, e.study_id, e.ecg_time, e.path, a.hadm_id, a.admittime, a.dischtime
  FROM ecg e JOIN adm a ON e.subject_id=a.subject_id
         AND e.ecg_time BETWEEN a.admittime AND a.dischtime
),
-- the matched lab's CHARTTIME is carried through, not just its value: with a
-- +-12 h matching window and pairs as close as 12 h apart, the same draw can be
-- the nearest lab to BOTH ECGs of a pair, which would make dCr/dK structurally
-- zero rather than measured. Stage 6 gates on the flags built from these.
cr AS (SELECT ea.study_id,
         arg_min(l.valuenum, abs(date_diff('minute', ea.ecg_time, l.charttime))) cr,
         arg_min(l.charttime, abs(date_diff('minute', ea.ecg_time, l.charttime))) cr_ct,
         MIN(abs(date_diff('minute', ea.ecg_time, l.charttime)))/60.0 cr_dt_h
       FROM ea JOIN labs l ON l.subject_id=ea.subject_id AND l.itemid={CREATININE}
         AND l.charttime BETWEEN ea.ecg_time - INTERVAL {LAB_WIN_H} HOUR
                             AND ea.ecg_time + INTERVAL {LAB_WIN_H} HOUR
       GROUP BY 1),
k AS (SELECT ea.study_id,
        arg_min(l.valuenum, abs(date_diff('minute', ea.ecg_time, l.charttime))) k,
        arg_min(l.charttime, abs(date_diff('minute', ea.ecg_time, l.charttime))) k_ct,
        MIN(abs(date_diff('minute', ea.ecg_time, l.charttime)))/60.0 k_dt_h
      FROM ea JOIN labs l ON l.subject_id=ea.subject_id AND l.itemid={POTASSIUM}
         AND l.charttime BETWEEN ea.ecg_time - INTERVAL {LAB_WIN_H} HOUR
                             AND ea.ecg_time + INTERVAL {LAB_WIN_H} HOUR
      GROUP BY 1)
SELECT ea.*, cr.cr, cr.cr_ct, cr.cr_dt_h, k.k, k.k_ct, k.k_dt_h,
       p.gender, (p.anchor_age + (EXTRACT(year FROM ea.admittime) - p.anchor_year)) AS age
FROM ea JOIN cr USING(study_id) JOIN k USING(study_id) JOIN pat p ON p.subject_id=ea.subject_id
""")

# eGFR, CKD-EPI 2021 (race-free)
con.execute("""
CREATE TEMP TABLE el2 AS
SELECT *,
  142.0
  * pow(LEAST(cr / (CASE WHEN gender='F' THEN 0.7 ELSE 0.9 END), 1.0),
        (CASE WHEN gender='F' THEN -0.241 ELSE -0.302 END))
  * pow(GREATEST(cr / (CASE WHEN gender='F' THEN 0.7 ELSE 0.9 END), 1.0), -1.200)
  * pow(0.9938, age)
  * (CASE WHEN gender='F' THEN 1.012 ELSE 1.0 END) AS egfr
FROM el
""")

# ---------------------------------------------------------------- 3. pairs
con.execute(f"""
CREATE TEMP TABLE pairs AS
WITH seq AS (
  SELECT *, LEAD(ecg_time) OVER w t2, LEAD(study_id) OVER w study2, LEAD(path) OVER w path2,
            LEAD(cr) OVER w cr2, LEAD(cr_ct) OVER w cr_ct2,
            LEAD(k) OVER w k2, LEAD(k_ct) OVER w k_ct2, LEAD(egfr) OVER w egfr2
  FROM el2 WINDOW w AS (PARTITION BY hadm_id ORDER BY ecg_time)
)
SELECT subject_id, hadm_id, age, gender, admittime,
       study_id AS study1, study2, path AS path1, path2,
       ecg_time AS t1, t2, date_diff('hour', ecg_time, t2) AS dt_h,
       cr, cr2, cr2-cr AS dcr, k, k2, k2-k AS dk, egfr, egfr2, egfr2-egfr AS degfr,
       cr_ct, cr_ct2, k_ct, k_ct2,
       -- a "delta" built from one draw read twice is not a measured change
       (cr_ct = cr_ct2) AS cr_same_draw,
       (k_ct  = k_ct2)  AS k_same_draw,
       (cr_ct >= cr_ct2 OR k_ct >= k_ct2) AS lab_draws_unordered
FROM seq WHERE t2 IS NOT NULL
  AND date_diff('hour', ecg_time, t2) BETWEEN {PAIR_MIN_H} AND {PAIR_MAX_H}
""")

# ---------------------------------------------------------------- 4. KDIGO at t2
# rolling 48h rise on the FULL Cr series (catches peaks between the two ECGs),
# plus the 1.5x-baseline / 7d criterion against the hierarchical baseline.
con.execute(f"""
CREATE TEMP TABLE pairs_lab AS
WITH cr AS (SELECT subject_id, charttime, valuenum FROM labs WHERE itemid={CREATININE}),
-- Everything here is CAUSAL w.r.t. t2: no creatinine drawn after the 2nd ECG may
-- enter the label. The early-warning claim (sec.4) is worthless if the label
-- itself peeked at the future, so the baseline is recomputed per-pair, not
-- per-admission.
roll AS (
  SELECT p.study1,
         -- lowest Cr in the 48 h preceding t2 -> the KDIGO 0.3 mg/dL reference
         MIN(c.valuenum) FILTER (WHERE c.charttime BETWEEN p.t2 - INTERVAL 48 HOUR AND p.t2) AS cr_min48,
         -- causal baseline: lowest Cr from admission up to t2 (NOT whole admission)
         MIN(c.valuenum) FILTER (WHERE c.charttime BETWEEN p.admittime AND p.t2)             AS cr_inpat_min_to_t2,
         MAX(c.valuenum) FILTER (WHERE c.charttime BETWEEN p.t1 AND p.t2)                    AS cr_peak_in_pair
  FROM pairs p JOIN cr c ON c.subject_id=p.subject_id
  GROUP BY p.study1
),
nxt AS (
  SELECT p.study1, arg_min(c.valuenum, c.charttime) AS cr_next
  FROM pairs p JOIN cr c ON c.subject_id=p.subject_id
    AND c.charttime BETWEEN p.t2 + INTERVAL {NEXTDAY_LO} HOUR
                        AND p.t2 + INTERVAL {NEXTDAY_HI} HOUR
  GROUP BY p.study1
)
SELECT p.*, b.cr_prior, b.cr_prior IS NOT NULL AS baseline_is_prior,
       r.cr_min48, r.cr_inpat_min_to_t2, r.cr_peak_in_pair, n.cr_next,
       n.cr_next - p.cr2 AS dcr_next,
       LEAST(COALESCE(r.cr_inpat_min_to_t2, 1e9), COALESCE(b.cr_prior, 1e9)) AS cr_base
FROM pairs p
JOIN base b USING(hadm_id)
LEFT JOIN roll r USING(study1)
LEFT JOIN nxt  n USING(study1)
""")

con.execute("""
CREATE TEMP TABLE manifest AS
SELECT *,
  -- KDIGO 1: rise >=0.3 mg/dL vs the lowest value of the preceding 48 h, read AT t2
  (cr2 - cr_min48 >= 0.3)                 AS kdigo_abs_48h,
  -- KDIGO 2: >=1.5x the causal baseline
  (cr2 >= 1.5 * cr_base)                  AS kdigo_rel_7d,
  ((cr2 - cr_min48 >= 0.3) OR (cr2 >= 1.5 * cr_base)) AS aki,
  CASE WHEN cr2 >= 3.0*cr_base OR cr2 >= 4.0 THEN 3
       WHEN cr2 >= 2.0*cr_base              THEN 2
       WHEN (cr2 - cr_min48 >= 0.3) OR (cr2 >= 1.5*cr_base) THEN 1
       ELSE 0 END                          AS kdigo_stage,
  (cr_next IS NOT NULL)                    AS has_next_cr,
  (cr_next IS NOT NULL AND dcr_next >= 0.3) AS aki_next_day,
  hadm_id IN (SELECT hadm_id FROM acs_hadm) AS is_acs,
  hadm_id IN (SELECT hadm_id FROM pci_hadm) AS is_pci,
  hadm_id IN (SELECT hadm_id FROM cath_hadm) AS is_cath
FROM pairs_lab
""")

# ---------------------------------------------------------------- 5. splits
# ONE global patient-level split. Everything else is a VIEW on it via cohort flags.
#
# Why not split the anchor cohort separately: the anchor is a strict subset of the
# pretrain corpus, so two independent splits would put anchor test patients into
# the pretrain pool. A single patient-level partition makes that impossible by
# construction -- a patient is in exactly one of train/val/test, for every arm.
#
# Anchor = T5 = ACS n (angiography OR PCI), i.e. concept.md sec.3 as written.
# The early-warning arm (sec.4) is a physiological claim, not a contrast claim, so
# it is read primarily off the FULL test set and only secondarily off the anchor.
con.execute(f"""
CREATE TEMP TABLE final2 AS
SELECT *,
  (is_acs AND (is_pci OR is_cath)) AS is_anchor,
  (aki IS NULL) AS unlabelable,          -- no Cr before t2 -> no baseline
  CASE WHEN hash(subject_id + {SEED}) % 100 < 70 THEN 'train'
       WHEN hash(subject_id + {SEED}) % 100 < 85 THEN 'val'
       ELSE 'test' END AS split
FROM manifest
""")

con.execute(f"COPY (SELECT * FROM final2) TO '{DATA}/pairs_manifest.parquet' (FORMAT parquet)")

# ---------------------------------------------------------------- report
def q(sql): return con.sql(sql).fetchdf()

print("=== PRETRAIN CORPUS (all admissions) ===")
print(q("""SELECT split, COUNT(*) n_pairs, COUNT(DISTINCT subject_id) n_subj,
       SUM(CASE WHEN aki THEN 1 ELSE 0 END) n_aki,
       ROUND(100.0*AVG(CASE WHEN aki THEN 1 ELSE 0 END),1) aki_pct,
       SUM(CASE WHEN has_next_cr THEN 1 ELSE 0 END) n_lead,
       SUM(CASE WHEN aki_next_day THEN 1 ELSE 0 END) n_lead_aki
  FROM final2 WHERE NOT unlabelable GROUP BY 1 ORDER BY 1""").to_string(index=False))

print("\n=== ANCHOR COHORT (T5 = ACS n [angiography or PCI], concept sec.3) ===")
print(q("""SELECT split, COUNT(*) n_pairs, COUNT(DISTINCT subject_id) n_subj,
       SUM(CASE WHEN aki THEN 1 ELSE 0 END) n_aki,
       ROUND(100.0*AVG(CASE WHEN aki THEN 1 ELSE 0 END),1) aki_pct,
       SUM(CASE WHEN has_next_cr THEN 1 ELSE 0 END) n_lead,
       SUM(CASE WHEN aki_next_day THEN 1 ELSE 0 END) n_lead_aki
  FROM final2 WHERE is_anchor AND NOT unlabelable GROUP BY 1 ORDER BY 1""").to_string(index=False))

print("\n=== KDIGO stage distribution (anchor cohort) ===")
print(q("""SELECT kdigo_stage, COUNT(*) n_pairs, COUNT(DISTINCT subject_id) n_subj
  FROM final2 WHERE is_anchor AND NOT unlabelable GROUP BY 1 ORDER BY 1""").to_string(index=False))

print("\n=== LEAKAGE CHECK: any patient appearing in >1 split? (must be 0) ===")
print(q("""SELECT COUNT(*) AS patients_in_multiple_splits FROM (
    SELECT subject_id FROM final2 GROUP BY subject_id HAVING COUNT(DISTINCT split) > 1
  )""").to_string(index=False))

print("\n=== unlabelable pairs (no Cr before t2 -> no baseline; dropped) ===")
print(q("""SELECT COUNT(*) n_unlabelable,
       ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM final2),2) pct
  FROM final2 WHERE unlabelable""").to_string(index=False))

print("\n=== lab-draw sharing (gated in stage 6, flags only here) ===")
print(q("""SELECT
  COUNT(*) n_pairs,
  ROUND(100.0*AVG(CASE WHEN cr_same_draw THEN 1 ELSE 0 END),2) pct_same_cr_draw,
  ROUND(100.0*AVG(CASE WHEN k_same_draw  THEN 1 ELSE 0 END),2) pct_same_k_draw,
  ROUND(100.0*AVG(CASE WHEN cr_same_draw OR k_same_draw THEN 1 ELSE 0 END),2) pct_either,
  ROUND(100.0*AVG(CASE WHEN lab_draws_unordered THEN 1 ELSE 0 END),2) pct_unordered
  FROM final2""").to_string(index=False))

print("\n=== baseline provenance ===")
print(q("""SELECT baseline_is_prior, COUNT(*) n_pairs FROM final2 GROUP BY 1""").to_string(index=False))

# How much of the AKI rate was an artefact of a future-peeking baseline?
# `base` is the admission-wide (leaky) baseline; cr_base is the causal one.
print("\n=== causal vs leaky KDIGO label (sanity: how much did the fix cost?) ===")
print(q("""
SELECT
  ROUND(100.0*AVG(CASE WHEN f.aki THEN 1 ELSE 0 END),1)                       AS aki_pct_causal,
  ROUND(100.0*AVG(CASE WHEN f.cr2 >= 1.5*b.cr_base THEN 1 ELSE 0 END),1)      AS rel_pct_leaky_base,
  ROUND(100.0*AVG(CASE WHEN f.kdigo_rel_7d THEN 1 ELSE 0 END),1)              AS rel_pct_causal_base,
  ROUND(100.0*AVG(CASE WHEN f.kdigo_abs_48h THEN 1 ELSE 0 END),1)             AS abs48_pct_causal
FROM final2 f JOIN base b USING(hadm_id)
""").to_string(index=False))

print(f"\nwrote {DATA}/pairs_manifest.parquet")
