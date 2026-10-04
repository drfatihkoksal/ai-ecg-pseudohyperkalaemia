"""v3/HEEDB stage 1: ECG-potassium pairs from HEEDB site I0001 (BDSP), the external
training / validation corpus (concept.md sec.0.6).

Verified on one of the 11 lab files (2026-10-01):
  * the lab SpecimenTakenTimeDTS is shifted consistently with the ECG metadata
    ECGAcquisitionTime: the signed lab-ECG gap histogram peaks sharply at 0 (2 h bins:
    ~30k baseline vs 72k / 96k in the two bins around 0). PrioritizeDTS is shifted
    differently (flat histogram) and must NOT be used.
  * haemolysed potassium is mostly NOT reported as a number ("RESULT NOT REPORTED,
    HEMOLYSIS", "Totally hemolyzed specimen"); only numeric results are kept, so
    haemolysed values largely self-exclude. There is no comment column, so the
    pseudohyperkalaemia question cannot be posed at this site.

Potassium: LOINC 2823-3 (serum/plasma), 6298-4 (blood), 32713-0 (arterial blood) or a
component name starting with "potassium"; numeric result in 1-12 mmol/L.
Pair: each ECG with the nearest such K within +-2 h (same rule as 21_preprocess_v3.py).

Writes data/heedb/heedb_ecg_k.parquet: pid, file (absolute WFDB path, no extension),
ecg_time, k, gap_h, k_source (loinc), plus sex and age at acquisition.
"""
import os, duckdb

HEEDB = os.environ.get("HEEDB_ROOT", "/path/to/heedb")
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTD = f"{ROOT}/data/heedb"
LABS = f"{HEEDB}/EHR/I0001/Labs/*.parquet"
META = f"{HEEDB}/ECG/I0001/metadata/metadata.csv"
WIN_H = 2

os.makedirs(OUTD, exist_ok=True)
con = duckdb.connect(); con.execute("PRAGMA threads=24"); con.execute("PRAGMA memory_limit='48GB'")
con.execute(f"SET temp_directory='{HEEDB}/duckdb_tmp'")

con.execute(f"""
CREATE TEMP TABLE k AS
SELECT BDSPPatientID AS pid, try_cast(ResultTXT AS DOUBLE) AS k,
       try_cast(SpecimenTakenTimeDTS AS TIMESTAMP) AS t, LoincTXT AS loinc
FROM '{LABS}'
WHERE (LoincTXT IN ('2823-3', '6298-4', '32713-0') OR lower(ComponentNM) LIKE 'potassium%')
  AND try_cast(ResultTXT AS DOUBLE) BETWEEN 1 AND 12
  AND try_cast(SpecimenTakenTimeDTS AS TIMESTAMP) IS NOT NULL
  AND strftime(try_cast(SpecimenTakenTimeDTS AS TIMESTAMP), '%H:%M:%S') <> '00:00:00'   -- date-only rows
""")
print(con.sql("SELECT COUNT(*) k_results, COUNT(DISTINCT pid) patients FROM k").fetchdf().to_string(index=False), flush=True)

cols = ["BDSPPatientID", "FileName", "SexDSC", "ECGAcquisitionTime", "AgeAtAcquisition"]
con.execute(f"""
CREATE TEMP TABLE e AS
SELECT CAST(BDSPPatientID AS BIGINT) pid, FileName, SexDSC sex,
       try_cast(ECGAcquisitionTime AS TIMESTAMP) t, try_cast(AgeAtAcquisition AS DOUBLE) / 365.25 age
FROM read_csv('{META}', header=true, all_varchar=true)
WHERE try_cast(ECGAcquisitionTime AS TIMESTAMP) IS NOT NULL
  AND CAST(BDSPPatientID AS BIGINT) IN (SELECT DISTINCT pid FROM k)
""")
con.execute(f"""
CREATE TEMP TABLE pairs AS
SELECT e.pid, '{HEEDB}/ECG/I0001/WFDB' || regexp_replace(e.FileName, '\\.hea$', '') AS file,
       e.t AS ecg_time, e.sex, e.age,
       arg_min(k.k, abs(date_diff('second', k.t, e.t))) AS k,
       arg_min(k.loinc, abs(date_diff('second', k.t, e.t))) AS k_source,
       min(abs(date_diff('second', k.t, e.t))) / 3600.0 AS gap_h
FROM e JOIN k ON k.pid = e.pid AND k.t BETWEEN e.t - INTERVAL {WIN_H} HOUR AND e.t + INTERVAL {WIN_H} HOUR
GROUP BY 1, 2, 3, 4, 5
""")
con.execute(f"COPY pairs TO '{OUTD}/heedb_ecg_k.parquet' (FORMAT parquet)")
print(con.sql("""SELECT COUNT(*) pairs, COUNT(DISTINCT pid) patients,
  SUM((k >= 5.5)::INT) hyperK, SUM((k >= 6.0)::INT) k_ge_6, SUM((k < 3.5)::INT) hypoK,
  SUM((gap_h <= 1)::INT) within_1h, ROUND(MEDIAN(k), 2) med_k, MIN(ecg_time) t0, MAX(ecg_time) t1
  FROM pairs""").fetchdf().T.to_string(), flush=True)
print(f"wrote {OUTD}/heedb_ecg_k.parquet")
