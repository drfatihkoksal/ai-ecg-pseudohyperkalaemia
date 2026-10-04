"""Stage 0b: troponin, the ischemia covariate concept.md sec.5 asked for and the v1
pipeline never extracted.

In ACS the ECG difference is dominated by ST evolution. Any "renal" or "potassium"
signal claimed from d-ECG must survive an ischemia covariate, so troponin is carried
alongside every pair (stage 7b) and entered in the v2 sensitivity floors.

  51003 = Troponin T            (the assay MIMIC-IV used for most of its span)
  51002, 52642 = Troponin I     (kept, flagged; not pooled with T on the raw scale)
"""
import duckdb, time, os

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/data/labs_trop.parquet"
TROP_T, TROP_I = 51003, (51002, 52642)

src = next(f"{MIMIC}/hosp/labevents{e}" for e in (".csv", ".csv.gz")
           if os.path.exists(f"{MIMIC}/hosp/labevents{e}"))
con = duckdb.connect(); con.execute("PRAGMA threads=16")
t0 = time.time()
print(f"scanning {src} ...", flush=True)
con.execute(f"""
COPY (
  SELECT subject_id, hadm_id, itemid, CAST(charttime AS TIMESTAMP) AS charttime, valuenum
  FROM read_csv_auto('{src}',
         types={{'valuenum':'DOUBLE','hadm_id':'BIGINT','itemid':'BIGINT'}})
  WHERE itemid IN ({TROP_T}, {TROP_I[0]}, {TROP_I[1]})
    AND valuenum IS NOT NULL AND valuenum >= 0 AND valuenum < 1000
) TO '{OUT}' (FORMAT parquet)
""")
print(f"done in {time.time()-t0:.0f}s -> {OUT}", flush=True)
print(con.sql(f"""SELECT itemid, COUNT(*) n, COUNT(DISTINCT subject_id) subj,
  MEDIAN(valuenum) med FROM read_parquet('{OUT}') GROUP BY 1 ORDER BY 1""").fetchdf())
