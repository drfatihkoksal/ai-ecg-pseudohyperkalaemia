"""Stage 1 (concept.md sec.7/sec.11): the gating cohort count.

Question this script exists to answer, before ANY modelling:
  how many intra-patient ECG PAIRS survive the intersection
    ACS/PCI admission  x  >=2 ECGs in-admission  x  Cr+K matched to each ECG
  and how many of those support the sec.4 lead hypothesis (next-day dCr)?

We do NOT report a single number. The design's fatal risk is that the strict
cohort is too small, so we walk an attrition ladder over nested cohort tiers and
show exactly which constraint destroys N. Every count is reported at the level
that matters for a paired design: PAIRS and PATIENTS, not ECGs.
"""
import duckdb, os, json
import pandas as pd

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ECGDIR = os.environ.get("MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg/1.0")
_ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "outputs")

CREATININE, POTASSIUM = 50912, 50971

# --- pairing / window parameters (pre-registered knobs, sec.6) -----------------
LAB_WIN_H   = 12     # an ECG is "lab-matched" if a Cr (and K) exists within +-12h
PAIR_MIN_H  = 12     # dt lower bound for a usable delta pair
PAIR_MAX_H  = 168    # dt upper bound = KDIGO 7d
NEXTDAY_LO  = 12     # lead hypothesis: a further Cr 12-36h after the 2nd ECG
NEXTDAY_HI  = 36

con = duckdb.connect(); con.execute("PRAGMA threads=16")

# ---------------------------------------------------------------- source tables
con.execute(f"""
CREATE TEMP TABLE ecg AS
SELECT subject_id, study_id, CAST(ecg_time AS TIMESTAMP) AS ecg_time
FROM read_csv_auto('{ECGDIR}/record_list.csv')
""")
con.execute(f"""
CREATE TEMP TABLE adm AS
SELECT subject_id, hadm_id,
       CAST(admittime AS TIMESTAMP) AS admittime,
       CAST(dischtime AS TIMESTAMP) AS dischtime
FROM read_csv_auto('{MIMIC}/hosp/admissions.csv')
""")
con.execute(f"""
CREATE TEMP TABLE dx AS
SELECT subject_id, hadm_id, CAST(icd_code AS VARCHAR) icd_code, icd_version
FROM read_csv_auto('{MIMIC}/hosp/diagnoses_icd.csv', types={{'icd_code':'VARCHAR'}})
""")
con.execute(f"""
CREATE TEMP TABLE px AS
SELECT subject_id, hadm_id, CAST(icd_code AS VARCHAR) icd_code, icd_version,
       CAST(chartdate AS DATE) chartdate
FROM read_csv_auto('{MIMIC}/hosp/procedures_icd.csv', types={{'icd_code':'VARCHAR'}})
""")
con.execute(f"CREATE TEMP TABLE labs AS SELECT * FROM read_parquet('{DATA}/labs_cr_k.parquet')")

# ---------------------------------------------------------------- cohort flags
# ACS diagnosis: ICD-9 410.x (AMI), 411.1 (unstable angina)
#                ICD-10 I21.x/I22.x (AMI/reinfarct), I20.0 (unstable angina)
con.execute("""
CREATE TEMP TABLE acs_hadm AS
SELECT DISTINCT hadm_id FROM dx
WHERE (icd_version=9  AND (icd_code LIKE '410%' OR icd_code LIKE '4111%'))
   OR (icd_version=10 AND (icd_code LIKE 'I21%' OR icd_code LIKE 'I22%'
                           OR icd_code LIKE 'I200%'))
""")
# PCI: ICD-9 00.66/36.06/36.07/17.55 ; ICD-10-PCS 027* with percutaneous approach
con.execute("""
CREATE TEMP TABLE pci_hadm AS
SELECT hadm_id, MIN(chartdate) AS pci_date FROM px
WHERE (icd_version=9  AND icd_code IN ('0066','3606','3607','1755'))
   OR (icd_version=10 AND icd_code LIKE '027%' AND substr(icd_code,5,1) IN ('3','4'))
GROUP BY hadm_id
""")
# Coronary angiography / cath (contrast exposure without necessarily PCI)
con.execute("""
CREATE TEMP TABLE cath_hadm AS
SELECT hadm_id, MIN(chartdate) AS cath_date FROM px
WHERE (icd_version=9  AND icd_code IN ('8853','8854','8855','8856','8857','3722','3723'))
   OR (icd_version=10 AND icd_code LIKE 'B21%')
GROUP BY hadm_id
""")

# ---------------------------------------------------------------- ECG x admission
# record_list has NO hadm_id -> link by time containment in the admission.
con.execute("""
CREATE TEMP TABLE ecg_adm AS
SELECT e.subject_id, e.study_id, e.ecg_time, a.hadm_id, a.admittime, a.dischtime
FROM ecg e JOIN adm a
  ON e.subject_id = a.subject_id
 AND e.ecg_time >= a.admittime
 AND e.ecg_time <= a.dischtime
""")

# ---------------------------------------------------------------- ECG x labs
# nearest Cr and nearest K within +-LAB_WIN_H of the ECG
con.execute(f"""
CREATE TEMP TABLE ecg_lab AS
WITH cr AS (
  SELECT e.study_id,
         arg_min(l.valuenum, abs(date_diff('minute', e.ecg_time, l.charttime))) AS cr,
         MIN(abs(date_diff('minute', e.ecg_time, l.charttime)))/60.0 AS cr_dt_h
  FROM ecg_adm e JOIN labs l
    ON l.subject_id=e.subject_id AND l.itemid={CREATININE}
   AND l.charttime BETWEEN e.ecg_time - INTERVAL {LAB_WIN_H} HOUR
                       AND e.ecg_time + INTERVAL {LAB_WIN_H} HOUR
  GROUP BY e.study_id
),
k AS (
  SELECT e.study_id,
         arg_min(l.valuenum, abs(date_diff('minute', e.ecg_time, l.charttime))) AS k,
         MIN(abs(date_diff('minute', e.ecg_time, l.charttime)))/60.0 AS k_dt_h
  FROM ecg_adm e JOIN labs l
    ON l.subject_id=e.subject_id AND l.itemid={POTASSIUM}
   AND l.charttime BETWEEN e.ecg_time - INTERVAL {LAB_WIN_H} HOUR
                       AND e.ecg_time + INTERVAL {LAB_WIN_H} HOUR
  GROUP BY e.study_id
)
SELECT e.*, cr.cr, cr.cr_dt_h, k.k, k.k_dt_h
FROM ecg_adm e LEFT JOIN cr USING(study_id) LEFT JOIN k USING(study_id)
""")

# ---------------------------------------------------------------- tiers
TIERS = {
  "T0_all_admissions":      "1=1",
  "T1_cath_or_pci":         "h.hadm_id IN (SELECT hadm_id FROM cath_hadm UNION SELECT hadm_id FROM pci_hadm)",
  "T2_acs_any":             "h.hadm_id IN (SELECT hadm_id FROM acs_hadm)",
  "T3_pci_any":             "h.hadm_id IN (SELECT hadm_id FROM pci_hadm)",
  "T4_acs_AND_pci":         "h.hadm_id IN (SELECT hadm_id FROM acs_hadm) AND h.hadm_id IN (SELECT hadm_id FROM pci_hadm)",
  "T5_acs_AND_cath_or_pci": ("h.hadm_id IN (SELECT hadm_id FROM acs_hadm) AND h.hadm_id IN "
                             "(SELECT hadm_id FROM cath_hadm UNION SELECT hadm_id FROM pci_hadm)"),
}

rows = []
for tier, pred in TIERS.items():
    # -- ECG-level within tier
    q = f"""
    WITH t AS (
      SELECT e.* FROM ecg_lab e JOIN adm h USING(hadm_id) WHERE {pred}
    ),
    lab_ok AS (SELECT * FROM t WHERE cr IS NOT NULL AND k IS NOT NULL),
    -- admissions with >=2 in-admission ECGs (no lab requirement)
    a2 AS (SELECT hadm_id FROM t GROUP BY hadm_id HAVING COUNT(*)>=2),
    -- admissions with >=2 LAB-MATCHED ECGs  <- the design's real unit
    a2lab AS (SELECT hadm_id FROM lab_ok GROUP BY hadm_id HAVING COUNT(*)>=2),
    -- all usable consecutive delta pairs (consecutive lab-matched ECGs in admission)
    seq AS (
      SELECT *, LEAD(ecg_time) OVER w AS t2,
                LEAD(study_id) OVER w AS study2,
                LEAD(cr) OVER w AS cr2,
                LEAD(k)  OVER w AS k2
      FROM lab_ok WINDOW w AS (PARTITION BY hadm_id ORDER BY ecg_time)
    ),
    pairs AS (
      SELECT subject_id, hadm_id, ecg_time AS t1, t2,
             date_diff('hour', ecg_time, t2) AS dt_h,
             cr2 - cr AS dcr, k2 - k AS dk
      FROM seq
      WHERE t2 IS NOT NULL
        AND date_diff('hour', ecg_time, t2) BETWEEN {PAIR_MIN_H} AND {PAIR_MAX_H}
    )
    SELECT
      (SELECT COUNT(*) FROM t)                                   AS n_ecg,
      (SELECT COUNT(DISTINCT hadm_id) FROM t)                    AS n_hadm,
      (SELECT COUNT(DISTINCT subject_id) FROM t)                 AS n_subj,
      (SELECT COUNT(*) FROM lab_ok)                              AS n_ecg_lab,
      (SELECT COUNT(*) FROM a2)                                  AS n_hadm_2ecg,
      (SELECT COUNT(*) FROM a2lab)                               AS n_hadm_2ecg_lab,
      (SELECT COUNT(*) FROM pairs)                               AS n_pairs,
      (SELECT COUNT(DISTINCT subject_id) FROM pairs)             AS n_pair_subj,
      (SELECT MEDIAN(dt_h) FROM pairs)                           AS med_dt_h
    """
    r = con.sql(q).fetchdf().iloc[0].to_dict()
    r = {"tier": tier, **r}
    rows.append(r)
    print(f"{tier:24s} ECG={r['n_ecg']:>7,.0f} hadm={r['n_hadm']:>6,.0f} "
          f"ECG_lab={r['n_ecg_lab']:>7,.0f} hadm>=2ECG_lab={r['n_hadm_2ecg_lab']:>6,.0f} "
          f"PAIRS={r['n_pairs']:>7,.0f} pair_subj={r['n_pair_subj']:>6,.0f}", flush=True)

pd.DataFrame(rows).to_csv(f"{OUT}/cohort_count_tiers.csv", index=False)
print(f"\nwrote {OUT}/cohort_count_tiers.csv")
