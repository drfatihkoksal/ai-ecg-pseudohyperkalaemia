"""Stage 6b: v2 labels -- the endpoints the paper should have been asking about.

WHY v1 FAILED (diagnosed 2026-10-01, see concept.md sec.0). The v1 `aki` flag is a
STATE read at t2 ("is the patient in AKI now?"), not an EVENT on the t1->t2 interval:
  * 46% of AKI-positive pairs already had Cr >= 1.5x baseline at t1,
  * 29% of AKI-positive pairs had dCr <= 0 (creatinine flat or FALLING),
  * P(aki | previous pair aki) = 0.67 vs 0.14 otherwise.
A state label is predictable from the known creatinine (floor AUC 0.70) and from a
single ECG (static_ecg2 = difference arms), so it could never test the delta thesis.
It also used a 365-day outpatient baseline (KDIGO's 1.5x criterion is a 7-day one) and
a 48 h rise window unrelated to the pair interval.

v2 labels, all computed on the FULL creatinine series, causally:

  per in-admission creatinine c at time tau:
      flag(c) = c - min(Cr in [tau-48h, tau)) >= 0.3
             OR c >= 1.5 * min(Cr in [tau-7d, tau))
  onset(hadm) = first flagged tau within the admission
  poa(hadm)   = first in-admission Cr >= 1.5x the most recent pre-admission Cr
                (<=365 d) -- AKI/CKD progression present on admission; such
                admissions are never "at risk"

  The windows are anchored on the MATCHED CREATININE DRAWS (cr_ct, cr_ct2), not on
  the ECG times. With +-12 h lab matching, an ECG-anchored window put the onset at or
  before the t1 draw in 14% of positives (cr1 WAS the onset value, dCr <= 0 by
  construction). Draw-anchored, the event is measured on the same interval as dCr/dK.

  aki_incident  at risk at the t1 draw (no onset <= cr_ct, not poa); 1 if onset in
                (cr_ct, cr_ct2]; 0 otherwise (cr_ct2 itself is an observation).
                Transient peaks between the draws still count -- they are KDIGO AKI.
  aki_early     at risk at the t2 draw; 1 if onset in (cr_ct2, t2+48h]; 0 if not and
                >=1 Cr was drawn in that window; else NULL. The sec.4 early-warning test.
  hyperk_new    defined only where k1 < 5.0: 1 if k2 >= 5.5
  trop1, trop2  nearest troponin T within +-12 h of each ECG (ischemia covariate)
"""
import os, duckdb

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
CREATININE, TROP_T = 50912, 51003
EARLY_H, TROP_WIN_H = 48, 12
HYPERK_FROM, HYPERK_TO = 5.0, 5.5

adm_csv = next(f"{MIMIC}/hosp/admissions{e}" for e in (".csv", ".csv.gz")
               if os.path.exists(f"{MIMIC}/hosp/admissions{e}"))
con = duckdb.connect(); con.execute("PRAGMA threads=16")

con.execute(f"CREATE TEMP TABLE p AS SELECT * FROM read_parquet('{DATA}/pairs_final.parquet')")
con.execute(f"""CREATE TEMP TABLE h AS
  SELECT DISTINCT p.hadm_id, p.subject_id, p.admittime, CAST(a.dischtime AS TIMESTAMP) dischtime
  FROM p JOIN read_csv_auto('{adm_csv}') a USING(hadm_id)""")
con.execute(f"""CREATE TEMP TABLE cr AS
  SELECT l.subject_id, l.charttime, l.valuenum
  FROM read_parquet('{DATA}/labs_cr_k.parquet') l
  WHERE l.itemid={CREATININE} AND l.subject_id IN (SELECT subject_id FROM h)""")

# ---- KDIGO flag on every creatinine (references strictly BEFORE the draw)
con.execute("""
CREATE TEMP TABLE crf AS
SELECT subject_id, charttime, valuenum,
  MIN(valuenum) OVER (PARTITION BY subject_id ORDER BY charttime
        RANGE BETWEEN INTERVAL 48 HOUR PRECEDING AND INTERVAL 1 SECOND PRECEDING) AS min48,
  MIN(valuenum) OVER (PARTITION BY subject_id ORDER BY charttime
        RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 SECOND PRECEDING)   AS min7d
FROM cr
""")
con.execute("""
CREATE TEMP TABLE hadm_lab AS
WITH ina AS (
  SELECT h.hadm_id, c.charttime, c.valuenum,
         (c.valuenum - c.min48 >= 0.3) OR (c.valuenum >= 1.5 * c.min7d) AS flag
  FROM h JOIN crf c ON c.subject_id = h.subject_id
   AND c.charttime BETWEEN h.admittime AND h.dischtime
),
prior AS (
  SELECT h.hadm_id, arg_max(c.valuenum, c.charttime) AS cr_prior365
  FROM h JOIN cr c ON c.subject_id = h.subject_id
   AND c.charttime >= h.admittime - INTERVAL 365 DAY AND c.charttime < h.admittime
  GROUP BY 1
),
agg AS (
  SELECT hadm_id,
         MIN(charttime) FILTER (WHERE flag)  AS onset,
         arg_min(valuenum, charttime)        AS cr_first_inadm
  FROM ina GROUP BY 1
)
SELECT h.hadm_id, a.onset, a.cr_first_inadm, pr.cr_prior365,
       COALESCE(a.cr_first_inadm >= 1.5 * pr.cr_prior365, FALSE) AS poa
FROM h LEFT JOIN agg a USING(hadm_id) LEFT JOIN prior pr USING(hadm_id)
""")

# ---- observability counts: was creatinine actually drawn in each window?
con.execute(f"""
CREATE TEMP TABLE obs AS
SELECT p.study1, p.study2,
  COUNT(*) FILTER (WHERE c.charttime >  p.t1 AND c.charttime <= p.t2)                         AS n_cr_between,
  COUNT(*) FILTER (WHERE c.charttime >  p.cr_ct2 AND c.charttime <= p.t2 + INTERVAL {EARLY_H} HOUR
                     AND c.charttime <= h.dischtime)                                             AS n_cr_next
FROM p JOIN h USING(hadm_id) LEFT JOIN cr c ON c.subject_id = p.subject_id
  AND c.charttime > p.t1 AND c.charttime <= p.t2 + INTERVAL {EARLY_H} HOUR
GROUP BY 1, 2
""")

trop = f"{DATA}/labs_trop.parquet"
if os.path.exists(trop):
    con.execute(f"""
    CREATE TEMP TABLE tr AS
    WITH t AS (SELECT subject_id, charttime, valuenum FROM read_parquet('{trop}') WHERE itemid={TROP_T})
    SELECT p.study1, p.study2,
      arg_min(t.valuenum, abs(date_diff('minute', p.t1, t.charttime)))
        FILTER (WHERE t.charttime BETWEEN p.t1 - INTERVAL {TROP_WIN_H} HOUR AND p.t1 + INTERVAL {TROP_WIN_H} HOUR) AS trop1,
      arg_min(t.valuenum, abs(date_diff('minute', p.t2, t.charttime)))
        FILTER (WHERE t.charttime BETWEEN p.t2 - INTERVAL {TROP_WIN_H} HOUR AND p.t2 + INTERVAL {TROP_WIN_H} HOUR) AS trop2
    FROM p JOIN t ON t.subject_id = p.subject_id
      AND t.charttime BETWEEN p.t1 - INTERVAL {TROP_WIN_H} HOUR AND p.t2 + INTERVAL {TROP_WIN_H} HOUR
    GROUP BY 1, 2
    """)
else:
    print(f"[warn] {trop} missing -- run 00b_extract_troponin.py; trop columns will be NULL")
    con.execute("CREATE TEMP TABLE tr AS SELECT study1, study2, NULL::DOUBLE trop1, NULL::DOUBLE trop2 FROM p")

con.execute(f"""
CREATE TEMP TABLE v2 AS
SELECT p.*, l.onset AS aki_onset, l.poa, l.cr_prior365, o.n_cr_between, o.n_cr_next,
  tr.trop1, tr.trop2,
  (NOT l.poa AND (l.onset IS NULL OR l.onset > p.cr_ct))  AS at_risk_t1,
  (NOT l.poa AND (l.onset IS NULL OR l.onset > p.cr_ct2)) AS at_risk_t2,
  CASE WHEN l.poa OR l.onset <= p.cr_ct THEN NULL
       WHEN l.onset <= p.cr_ct2         THEN 1
       ELSE 0 END AS aki_incident,
  CASE WHEN l.poa OR l.onset <= p.cr_ct2                       THEN NULL
       WHEN l.onset <= p.t2 + INTERVAL {EARLY_H} HOUR          THEN 1
       WHEN o.n_cr_next = 0                                     THEN NULL
       ELSE 0 END AS aki_early,
  CASE WHEN p.k < {HYPERK_FROM} THEN (p.k2 >= {HYPERK_TO})::INT END AS hyperk_new
FROM p JOIN hadm_lab l USING(hadm_id)
LEFT JOIN obs o USING(study1, study2)
LEFT JOIN tr USING(study1, study2)
""")

n_in, n_out = con.sql("SELECT (SELECT COUNT(*) FROM p), (SELECT COUNT(*) FROM v2)").fetchone()
assert n_in == n_out, f"row count changed {n_in} -> {n_out}"
con.execute(f"COPY (SELECT * FROM v2) TO '{DATA}/pairs_v2.parquet' (FORMAT parquet)")

print("=== v2 endpoints: labelled rows / events, by split ===")
print(con.sql("""
SELECT split, is_anchor, COUNT(*) n_pairs,
  COUNT(aki_incident) inc_n, SUM(aki_incident) inc_ev, COUNT(DISTINCT subject_id) FILTER (WHERE aki_incident=1) inc_pat,
  COUNT(aki_early) early_n, SUM(aki_early) early_ev,
  COUNT(hyperk_new) hk_n, SUM(hyperk_new) hk_ev,
  ROUND(100.0*AVG((trop2 IS NOT NULL)::INT),1) trop2_pct
FROM v2 GROUP BY ALL ORDER BY 2, 1""").fetchdf().to_string(index=False))

print("\n=== sanity: v1 state label vs v2 incident label ===")
print(con.sql("""
SELECT ROUND(100.0*AVG(poa::INT),1) poa_pct,
  ROUND(100.0*AVG(at_risk_t1::INT),1) at_risk_t1_pct,
  ROUND(AVG(dcr) FILTER (WHERE aki_incident=1),3) mean_dcr_incident_pos,
  ROUND(AVG(dcr) FILTER (WHERE aki=1),3)          mean_dcr_v1_pos,
  ROUND(100.0*AVG((dcr<=0)::INT) FILTER (WHERE aki_incident=1),1) pct_dcr_le0_incident_pos,
  ROUND(100.0*AVG((dcr<=0)::INT) FILTER (WHERE aki=1),1)          pct_dcr_le0_v1_pos
FROM v2""").fetchdf().to_string(index=False))
print(f"\nwrote {DATA}/pairs_v2.parquet")
