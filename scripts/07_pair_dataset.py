"""Stage 6: the analysis-ready pair table.

QC has to propagate: a difference is only as trustworthy as the WORSE of its two
ECGs, so a pair dies if either member failed preprocessing. This is stricter than
it sounds -- sec.5 item 3 warns that differencing AMPLIFIES per-acquisition
nuisance, so a marginal ECG that would be tolerable in a static model is not
tolerable here.

Emits dHR explicitly. sec.5/sec.6 want heart rate entered as a nuisance channel
(regressed out of the difference features, and as an explicit PID nuisance
source), which is only possible if it is carried alongside every pair.
"""
import os, duckdb, pandas as pd

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
WF = f"{DATA}/waveforms"

con = duckdb.connect(); con.execute("PRAGMA threads=16")

con.execute(f"""
CREATE TEMP TABLE pairs AS
SELECT p.*,
       m1.idx AS wf_idx1, m2.idx AS wf_idx2,
       m1.ok AS ok1, m2.ok AS ok2,
       m1.reason AS reason1, m2.reason AS reason2,
       m1.hr AS hr1, m2.hr AS hr2, m2.hr - m1.hr AS dhr,
       m1.n_bad_leads AS bad1, m2.n_bad_leads AS bad2
FROM read_parquet('{DATA}/pairs_manifest.parquet') p
LEFT JOIN read_parquet('{WF}/ecg_meta.parquet') m1 ON m1.study_id = p.study1
LEFT JOIN read_parquet('{WF}/ecg_meta.parquet') m2 ON m2.study_id = p.study2
""")

# The lab gate (stage 3 emits the flags): a pair whose two creatinines are the
# SAME draw read twice has dCr = 0 by construction, not by measurement, and the
# same for dK. Those rows would train the model to predict "no change" from a
# genuine ECG difference -- target noise, not signal. Ordering is required too
# (draw 1 strictly before draw 2), which in practice costs nothing extra.
LAB_GATE = "NOT cr_same_draw AND NOT k_same_draw AND NOT lab_draws_unordered"

print("=== pair-level attrition ===")
steps = [
    ("all candidate pairs",            "1=1"),
    ("both ECGs preprocessed OK",      "ok1 AND ok2"),
    ("+ labelable (Cr before t2)",     "ok1 AND ok2 AND NOT unlabelable"),
    ("+ <=1 bad lead in either ECG",   "ok1 AND ok2 AND NOT unlabelable AND bad1<=1 AND bad2<=1"),
    ("+ distinct, ordered lab draws",  f"ok1 AND ok2 AND NOT unlabelable AND bad1<=1 AND bad2<=1 AND {LAB_GATE}"),
]
prev = None
for name, pred in steps:
    r = con.sql(f"""SELECT COUNT(*) n, COUNT(DISTINCT subject_id) s
                    FROM pairs WHERE {pred}""").fetchone()
    drop = "" if prev is None else f"  (-{prev - r[0]:,})"
    print(f"  {name:<32s} pairs={r[0]:>7,}  patients={r[1]:>6,}{drop}")
    prev = r[0]

FINAL = f"ok1 AND ok2 AND NOT unlabelable AND bad1<=1 AND bad2<=1 AND {LAB_GATE}"
con.execute(f"CREATE TEMP TABLE final AS SELECT * FROM pairs WHERE {FINAL}")

print("\n=== final analysis set ===")
print(con.sql("""
SELECT split,
  COUNT(*) n_pairs, COUNT(DISTINCT subject_id) n_subj,
  SUM(CASE WHEN aki THEN 1 ELSE 0 END) n_aki,
  ROUND(100.0*AVG(CASE WHEN aki THEN 1 ELSE 0 END),1) aki_pct,
  SUM(CASE WHEN has_next_cr THEN 1 ELSE 0 END) n_lead,
  SUM(CASE WHEN aki_next_day THEN 1 ELSE 0 END) n_lead_aki
FROM final GROUP BY 1 ORDER BY 1
""").fetchdf().to_string(index=False))

print("\n=== anchor cohort (T5) within the final set ===")
print(con.sql("""
SELECT split,
  COUNT(*) n_pairs, COUNT(DISTINCT subject_id) n_subj,
  SUM(CASE WHEN aki THEN 1 ELSE 0 END) n_aki,
  ROUND(100.0*AVG(CASE WHEN aki THEN 1 ELSE 0 END),1) aki_pct,
  SUM(CASE WHEN has_next_cr THEN 1 ELSE 0 END) n_lead,
  SUM(CASE WHEN aki_next_day THEN 1 ELSE 0 END) n_lead_aki
FROM final WHERE is_anchor GROUP BY 1 ORDER BY 1
""").fetchdf().to_string(index=False))

print("\n=== nuisance + target distributions (sec.5) ===")
print(con.sql("""
SELECT
  ROUND(MEDIAN(dt_h),1) med_dt_h,
  ROUND(AVG(dhr),2) mean_dhr, ROUND(STDDEV(dhr),1) sd_dhr,
  ROUND(STDDEV(dcr),3) sd_dcr, ROUND(STDDEV(dk),3) sd_dk, ROUND(STDDEV(degfr),1) sd_degfr,
  ROUND(CORR(dcr,dk),3)   corr_dcr_dk,
  ROUND(CORR(dcr,dhr),3)  corr_dcr_dhr,
  ROUND(CORR(dk,dhr),3)   corr_dk_dhr,
  ROUND(CORR(dcr,degfr),3) corr_dcr_degfr
FROM final
""").fetchdf().to_string(index=False))

con.execute(f"COPY (SELECT * FROM final) TO '{DATA}/pairs_final.parquet' (FORMAT parquet)")
print(f"\nwrote {DATA}/pairs_final.parquet")

# --- the null-pair control set gets the SAME waveform indices and the SAME QC gate.
# A control held to a laxer standard than the analysis set proves nothing.
import os
if os.path.exists(f"{DATA}/null_pairs.parquet"):
    con.execute(f"""
    CREATE TEMP TABLE nullf AS
    SELECT p.*, m1.idx AS wf_idx1, m2.idx AS wf_idx2,
           m1.hr AS hr1, m2.hr AS hr2, m2.hr - m1.hr AS dhr
    FROM read_parquet('{DATA}/null_pairs.parquet') p
    JOIN read_parquet('{WF}/ecg_meta.parquet') m1 ON m1.study_id = p.study1
    JOIN read_parquet('{WF}/ecg_meta.parquet') m2 ON m2.study_id = p.study2
    WHERE m1.ok AND m2.ok AND m1.n_bad_leads <= 1 AND m2.n_bad_leads <= 1
    """)
    con.execute(f"COPY (SELECT * FROM nullf) TO '{DATA}/null_pairs_final.parquet' (FORMAT parquet)")
    print("\n=== null-pair control set, after the same QC gate ===")
    print(con.sql("""
    SELECT COUNT(*) n_pairs, COUNT(DISTINCT subject_id) n_subj,
      ROUND(MEDIAN(dt_h),2) med_dt_h,
      ROUND(STDDEV(dcr),3) sd_dcr, ROUND(STDDEV(dk),3) sd_dk, ROUND(STDDEV(dhr),1) sd_dhr
    FROM nullf""").fetchdf().to_string(index=False))
    print(f"wrote {DATA}/null_pairs_final.parquet")
