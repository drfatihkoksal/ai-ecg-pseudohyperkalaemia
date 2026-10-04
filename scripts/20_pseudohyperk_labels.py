"""v3 stage 1: labels for "can the ECG tell haemolytic pseudohyperkalaemia from true
hyperkalaemia?" (concept.md sec.0.5). Labels only -- no model, no ECG waveforms.

Index event: a serum potassium >= 5.5 (labevents 50971) with a 12-lead ECG within +-2 h.
Haemolysis: the lab's own result comment ("Hemolysis falsely elevates ...", "Slightly
Hemolyzed specimen ..."), excluding "Specimen not Hemolyzed".

Reference standard = the first NON-haemolysed repeat potassium (different specimen)
within 12 h of the index draw. Windows are asymmetric on purpose: serum K does not rise
spontaneously, so a repeat still >= 5.5 up to 12 h later confirms true hyperkalaemia,
whereas a normal value long after the index could reflect a genuine fall -- so "pseudo"
needs the normal repeat within 6 h.
  pseudo  index haemolysed; repeat < 5.0 within 6 h; NO potassium-lowering therapy (strict
          set) between index draw - 1 h and the repeat draw.
  true    index haemolysed; repeat >= 5.5 within 12 h (therapy allowed: still >= 5.5 after
          treatment is true hyperkalaemia a fortiori).
  true_nonhemolysed  index NOT haemolysed and confirmed by a repeat >= 5.5 within 12 h.
          The ECG of true hyperkalaemia does not depend on whether the index tube happened
          to haemolyse, so this is a valid -- and far larger -- positive reference for the
          ECG question. Primary analyses restrict it to the pseudo spectrum (index K < 6.5).
  indeterminate  everything else: repeat 5.0-5.4, therapy then repeat < 5.0 (cannot tell
          treated-true from spurious), normal repeat only after 6 h, or no repeat.

Count history (kept for audit): with a symmetric 6 h window and haemolysed indices only,
true = 61 vs pseudo = 526 -- too few positives; hence the 12 h confirmation window and the
non-haemolysed confirmed reference (2026-10-01, before any ECG was scored).

Potassium-lowering therapy (fixed before any outcome was looked at):
  STRICT (primary): any administered insulin; dextrose 50 %; K binders (sodium polystyrene
          sulfonate, sodium zirconium cyclosilicate, patiromer); calcium gluconate / chloride
          (excluding the ionised-calcium sliding scale, oncology replacement and CRRT
          calcium); haemodialysis / CRRT / peritoneal dialysis running in the window.
  BROAD (sensitivity): STRICT + albuterol-family nebulisers / inhalers, loop diuretics
          (furosemide, bumetanide, torsemide), sodium bicarbonate.
Sources: hosp/emar (administered events) + icu/inputevents + icu/procedureevents.

Also carried, for the clinical floor a model must beat: the index K value, haemolysis
grade, the most recent NON-haemolysed K in the prior 48 h, nearest creatinine (+-12 h).
And, for later stages: the nearest ECG's study_id, whether its waveform is already
preprocessed, and the most recent ECG > 12 h before the index (prior-ECG analyses).
"""
import os, duckdb
import numpy as np, pandas as pd

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ECGDIR = os.environ.get("MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg/1.0")
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
K_HI, K_NORMAL, ECG_WIN_H, REPEAT_H, CONFIRM_H, PRIOR_K_H, K_MATCH = 5.5, 5.0, 2, 6, 12, 48, 6.5

con = duckdb.connect(); con.execute("PRAGMA threads=16")

# ------------------------------------------------------------------ potassium + haemolysis
if not os.path.exists(f"{DATA}/labs_k_comments.parquet"):
    con.execute(f"""COPY (SELECT subject_id, hadm_id, specimen_id, itemid,
        CAST(charttime AS TIMESTAMP) charttime, valuenum, comments, flag
      FROM read_csv_auto('{MIMIC}/hosp/labevents.csv.gz',
        types={{'valuenum':'DOUBLE','hadm_id':'BIGINT','itemid':'BIGINT','comments':'VARCHAR','flag':'VARCHAR'}})
      WHERE itemid=50971 AND valuenum IS NOT NULL) TO '{DATA}/labs_k_comments.parquet' (FORMAT parquet)""")
con.execute(f"""
CREATE TEMP TABLE k AS
SELECT subject_id, hadm_id, specimen_id, charttime, valuenum AS k,
  lower(coalesce(comments, '')) AS c
FROM read_parquet('{DATA}/labs_k_comments.parquet') WHERE valuenum BETWEEN 1 AND 12
""")
con.execute("""
CREATE TEMP TABLE k2 AS
SELECT *,
  (c LIKE '%hemoly%' AND c NOT LIKE '%not hemoly%')                 AS hemo,
  CASE WHEN NOT (c LIKE '%hemoly%' AND c NOT LIKE '%not hemoly%') THEN 'none'
       WHEN c LIKE '%gross%'                     THEN 'gross'
       WHEN c LIKE '%moderat%'                   THEN 'moderate'
       WHEN c LIKE '%slight%'                    THEN 'slight'
       ELSE 'unspecified' END                                       AS hemo_grade
FROM k
""")

# ------------------------------------------------------------------ therapy events
emar_raw = f"{DATA}/emar_klower_raw.parquet"
if not os.path.exists(emar_raw):
    RX = ("(insulin|polystyrene|kayexalate|patiromer|veltassa|zirconium|lokelma|calcium gluconate|"
          "calcium chloride|albuterol|dextrose 50|d50|sodium bicarbonate|furosemide|bumetanide|torsemide)")
    con.execute(f"""COPY (SELECT subject_id, hadm_id, CAST(charttime AS TIMESTAMP) charttime, medication, event_txt
      FROM read_csv_auto('{MIMIC}/hosp/emar.csv.gz', all_varchar=true)
      WHERE regexp_matches(lower(medication), '{RX}')) TO '{emar_raw}' (FORMAT parquet)""")
GIVEN = ("'Administered','Confirmed','Started','Restarted','Delayed Administered',"
         "'Administered in Other Location','Partial Administered','in Other Location'")
con.execute(f"""
CREATE TEMP TABLE ther AS
WITH e AS (
  SELECT CAST(subject_id AS BIGINT) subject_id, charttime t0, charttime t1, lower(medication) m
  FROM read_parquet('{emar_raw}') WHERE event_txt IN ({GIVEN}) AND charttime IS NOT NULL
)
SELECT subject_id, t0, t1,
  CASE WHEN m LIKE '%insulin%' OR m LIKE '%dextrose 50%' OR m LIKE '%polystyrene%'
         OR m LIKE '%zirconium%' OR m LIKE '%lokelma%' OR m LIKE '%patiromer%' OR m LIKE '%veltassa%'
         OR ((m LIKE '%calcium gluconate%' OR m LIKE '%calcium chloride%')
             AND m NOT LIKE '%sliding scale%' AND m NOT LIKE '%oncology%' AND m NOT LIKE '%crrt%')
       THEN 'strict'
       WHEN m LIKE '%albuterol%' OR m LIKE '%furosemide%' OR m LIKE '%bumetanide%' OR m LIKE '%torsemide%'
         OR (m LIKE '%sodium bicarbonate%' AND m NOT LIKE '%omeprazole%')
       THEN 'broad' END AS tier
FROM e
""")
con.execute(f"""
INSERT INTO ther
SELECT subject_id, CAST(starttime AS TIMESTAMP), CAST(endtime AS TIMESTAMP),
  CASE WHEN itemid IN (223257,223258,223259,223260,223261,223262,229299,229619,  -- insulin
                       220952,                                                    -- D50
                       221456,228317,229640,229618)                               -- calcium
       THEN 'strict' ELSE 'broad' END
FROM read_csv_auto('{MIMIC}/icu/inputevents.csv.gz')
WHERE itemid IN (223257,223258,223259,223260,223261,223262,229299,229619,220952,
                 221456,228317,229640,229618,
                 220995,221211,227533,221794,228340,229639)                       -- bicarb, loops
""")
con.execute(f"""
INSERT INTO ther
SELECT subject_id, CAST(starttime AS TIMESTAMP), CAST(endtime AS TIMESTAMP), 'strict'
FROM read_csv_auto('{MIMIC}/icu/procedureevents.csv.gz')
WHERE itemid IN (225441, 225802, 225803, 225805, 225809, 225955)                  -- HD / CRRT / PD
""")
con.execute("DELETE FROM ther WHERE tier IS NULL")
print(con.sql("SELECT tier, COUNT(*) n FROM ther GROUP BY 1").fetchdf().to_string(index=False))

# ------------------------------------------------------------------ ECGs
con.execute(f"""CREATE TEMP TABLE ecg AS SELECT subject_id, study_id, CAST(ecg_time AS TIMESTAMP) t
  FROM read_csv_auto('{ECGDIR}/record_list.csv')""")
meta = f"{DATA}/waveforms/ecg_meta.parquet"
con.execute(f"CREATE TEMP TABLE meta AS SELECT study_id, ok, n_bad_leads FROM read_parquet('{meta}')")

# ------------------------------------------------------------------ index events
con.execute(f"""
CREATE TEMP TABLE idx AS
WITH hi AS (SELECT * FROM k2 WHERE k >= {K_HI}),
ne AS (
  SELECT hi.specimen_id,
    arg_min(e.study_id, abs(date_diff('second', e.t, hi.charttime))) ecg_study,
    arg_min(date_diff('minute', hi.charttime, e.t), abs(date_diff('second', e.t, hi.charttime))) ecg_gap_min
  FROM hi JOIN ecg e ON e.subject_id = hi.subject_id
   AND e.t BETWEEN hi.charttime - INTERVAL {ECG_WIN_H} HOUR AND hi.charttime + INTERVAL {ECG_WIN_H} HOUR
  GROUP BY 1
),
rp AS (   -- first NON-haemolysed repeat within the confirmation window
  SELECT hi.specimen_id,
    arg_min(r.k, r.charttime) k_rep, FALSE rep_hemo, min(r.charttime) t_rep,
    date_diff('minute', hi.charttime, min(r.charttime)) / 60.0 rep_after_h
  FROM hi JOIN k2 r ON r.subject_id = hi.subject_id AND r.specimen_id <> hi.specimen_id AND NOT r.hemo
   AND r.charttime > hi.charttime AND r.charttime <= hi.charttime + INTERVAL {CONFIRM_H} HOUR
  GROUP BY 1, hi.charttime
),
pk AS (
  SELECT hi.specimen_id, arg_max(p.k, p.charttime) k_prior,
         date_diff('minute', max(p.charttime), hi.charttime) / 60.0 k_prior_age_h
  FROM hi JOIN k2 p ON p.subject_id = hi.subject_id AND NOT p.hemo
   AND p.charttime >= hi.charttime - INTERVAL {PRIOR_K_H} HOUR AND p.charttime < hi.charttime
  GROUP BY 1, hi.charttime
),
cr AS (
  SELECT hi.specimen_id, arg_min(l.valuenum, abs(date_diff('second', l.charttime, hi.charttime))) cr
  FROM hi JOIN read_parquet('{DATA}/labs_cr_k.parquet') l ON l.subject_id = hi.subject_id AND l.itemid = 50912
   AND l.charttime BETWEEN hi.charttime - INTERVAL 12 HOUR AND hi.charttime + INTERVAL 12 HOUR
  GROUP BY 1
),
pe AS (
  SELECT hi.specimen_id, arg_max(e.study_id, e.t) prior_ecg_study
  FROM hi JOIN ecg e ON e.subject_id = hi.subject_id AND e.t < hi.charttime - INTERVAL 12 HOUR
  GROUP BY 1
)
SELECT hi.subject_id, hi.hadm_id, hi.specimen_id, hi.charttime, hi.k, hi.hemo, hi.hemo_grade,
  ne.ecg_study, ne.ecg_gap_min, rp.k_rep, rp.rep_hemo, rp.t_rep, rp.rep_after_h,
  pk.k_prior, pk.k_prior_age_h, cr.cr, pe.prior_ecg_study
FROM hi JOIN ne USING(specimen_id)
LEFT JOIN rp USING(specimen_id) LEFT JOIN pk USING(specimen_id)
LEFT JOIN cr USING(specimen_id) LEFT JOIN pe USING(specimen_id)
""")

# therapy between (index draw - 1 h) and the repeat draw (or index + 6 h if no repeat)
con.execute(f"""
CREATE TEMP TABLE th AS
SELECT i.specimen_id,
  bool_or(t.tier = 'strict') AS ther_strict,
  bool_or(t.tier IN ('strict', 'broad')) AS ther_broad
FROM idx i JOIN ther t ON t.subject_id = i.subject_id
 AND t.t0 <= coalesce(i.t_rep, i.charttime + INTERVAL {CONFIRM_H} HOUR)
 AND coalesce(t.t1, t.t0) >= i.charttime - INTERVAL 1 HOUR
GROUP BY 1
""")

con.execute(f"""
CREATE TEMP TABLE lab AS
SELECT i.*, coalesce(th.ther_strict, FALSE) ther_strict, coalesce(th.ther_broad, FALSE) ther_broad,
  m.ok AS ecg_ok, m.n_bad_leads AS ecg_bad_leads, (m.study_id IS NOT NULL) AS ecg_preprocessed,
  CASE
    WHEN NOT i.hemo AND i.k_rep >= {K_HI} THEN 'true_nonhemolysed'
    WHEN NOT i.hemo THEN 'nonhemolysed_unconfirmed'
    WHEN i.k_rep IS NULL THEN 'indet_no_repeat'
    WHEN i.k_rep >= {K_HI} THEN 'true'
    WHEN i.k_rep < {K_NORMAL} AND i.rep_after_h > {REPEAT_H} THEN 'indet_normal_after_6h'
    WHEN i.k_rep < {K_NORMAL} AND NOT coalesce(th.ther_strict, FALSE) THEN 'pseudo'
    WHEN i.k_rep < {K_NORMAL} THEN 'indet_treated_then_normal'
    ELSE 'indet_repeat_5.0-5.4' END AS label,
  (i.k < {K_MATCH}) AS k_matched,
  -- sensitivity: broad therapy exclusion for pseudo; true threshold relaxed to >= 5.0
  CASE
    WHEN NOT i.hemo OR i.k_rep IS NULL THEN NULL
    WHEN i.k_rep >= {K_NORMAL} THEN 'true'
    WHEN i.rep_after_h <= {REPEAT_H} AND NOT coalesce(th.ther_broad, FALSE) THEN 'pseudo' END AS label_sens
FROM idx i LEFT JOIN th USING(specimen_id) LEFT JOIN meta m ON m.study_id = i.ecg_study
""")
con.execute(f"COPY lab TO '{DATA}/pseudohyperk_index.parquet' (FORMAT parquet)")

# ------------------------------------------------------------------ report
q = lambda s: con.sql(s).fetchdf().to_string(index=False)
print("\n=== index events: K >= 5.5 with a 12-lead ECG within +-2 h ===")
print(q("""SELECT label, COUNT(*) n, COUNT(DISTINCT subject_id) patients,
  SUM((abs(ecg_gap_min) <= 60)::INT) ecg_within_1h, ROUND(MEDIAN(k),2) med_k,
  ROUND(MEDIAN(k_rep),2) med_k_repeat, ROUND(100.0*AVG(ecg_preprocessed::INT),1) pct_ecg_preprocessed
  FROM lab GROUP BY 1 ORDER BY 2 DESC"""))
print("\n=== analysis sets ===")
print("  PRIMARY  : pseudo  vs  true + true_nonhemolysed (index K < 6.5, spectrum-matched)")
print("  CLEAN    : pseudo  vs  true (haemolysed indices only; small)")
print(q("""SELECT label, k_matched, COUNT(*) n, COUNT(DISTINCT subject_id) patients,
  ROUND(MEDIAN(k),2) med_k, ROUND(100.0*AVG((k_prior IS NOT NULL)::INT),1) pct_prior_k,
  ROUND(100.0*AVG((prior_ecg_study IS NOT NULL)::INT),1) pct_prior_ecg,
  ROUND(100.0*AVG(ecg_preprocessed::INT),1) pct_ecg_ready
  FROM lab WHERE label IN ('pseudo','true','true_nonhemolysed') GROUP BY 1, 2 ORDER BY 1, 2"""))
print(q("""SELECT label, hemo_grade, COUNT(*) n FROM lab WHERE label IN ('pseudo','true')
  GROUP BY 1, 2 ORDER BY 1, 2"""))
print("\n=== ECGs that still need waveform preprocessing (primary set) ===")
print(q("""SELECT COUNT(DISTINCT ecg_study) ecgs_needed FROM lab
  WHERE (label IN ('pseudo','true') OR (label = 'true_nonhemolysed' AND k_matched)) AND NOT ecg_preprocessed"""))
print("\n=== sensitivity label (broad therapy exclusion; true = repeat >= 5.0) ===")
print(q("SELECT label_sens, COUNT(*) n FROM lab WHERE label_sens IS NOT NULL GROUP BY 1"))
print("\n=== first index per patient (primary set) ===")
print(q("""SELECT label, COUNT(*) n FROM (SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY charttime) rn
  FROM lab WHERE label IN ('pseudo','true') OR (label = 'true_nonhemolysed' AND k_matched)) WHERE rn = 1 GROUP BY 1"""))
print(f"\nwrote {DATA}/pseudohyperk_index.parquet")
