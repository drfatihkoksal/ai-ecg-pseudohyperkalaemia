"""Stage 0: one full scan of labevents.csv (18 GB) -> cached parquet of the
only two analytes this project needs.

  50912 = Creatinine, Blood, Chemistry   (KDIGO target)
  50971 = Potassium,  Blood, Chemistry   (the main confounder, concept.md sec.5)

Everything downstream reads the parquet, never the CSV again.
"""
import duckdb, time, os

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
DATA = os.path.join(os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
OUT = os.path.join(DATA, "labs_cr_k.parquet")

CREATININE, POTASSIUM = 50912, 50971

con = duckdb.connect()
con.execute("PRAGMA threads=16")

t0 = time.time()
print("scanning labevents.csv ...", flush=True)
con.execute(f"""
COPY (
  SELECT subject_id, hadm_id, itemid,
         CAST(charttime AS TIMESTAMP) AS charttime,
         valuenum
  FROM read_csv_auto('{MIMIC}/hosp/labevents.csv.gz',
         types={{'valuenum':'DOUBLE','hadm_id':'BIGINT','itemid':'BIGINT'}})
  WHERE itemid IN ({CREATININE}, {POTASSIUM})
    AND valuenum IS NOT NULL
    -- physiologic plausibility guards
    AND ( (itemid={CREATININE} AND valuenum BETWEEN 0.1 AND 25)
       OR (itemid={POTASSIUM}  AND valuenum BETWEEN 1.0 AND 10) )
) TO '{OUT}' (FORMAT parquet)
""")
print(f"done in {time.time()-t0:.0f}s -> {OUT}", flush=True)

r = con.sql(f"""
SELECT itemid,
       COUNT(*) AS n_rows,
       COUNT(DISTINCT subject_id) AS n_subj,
       MIN(valuenum) AS lo, MEDIAN(valuenum) AS med, MAX(valuenum) AS hi
FROM read_parquet('{OUT}') GROUP BY itemid ORDER BY itemid
""").fetchall()
for itemid, n_rows, n_subj, lo, med, hi in r:
    name = "creatinine" if itemid == CREATININE else "potassium"
    print(f"{name:11s} itemid={itemid}  rows={n_rows:>10,}  subjects={n_subj:>8,}  "
          f"range=[{lo:.1f}, {hi:.1f}] median={med:.2f}", flush=True)
