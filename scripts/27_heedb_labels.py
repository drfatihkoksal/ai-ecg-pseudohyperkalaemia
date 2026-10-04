"""v3/HEEDB stage 3: clean the HEEDB potassium training labels of probable undetected
haemolysis (concept.md sec.0.6).

HEEDB has no result-comment column: grossly haemolysed K is suppressed (non-numeric, already
excluded), but mildly haemolysed K is probably reported as a number with a comment we do not
get (suppressed share 0.36 % vs 2.5 % flagged in MIMIC). Two flags, fixed before training:

  hemo_grade_pos   a haemolysis-grade row (LOINC 20395-0 / "hemolysis grad*" / "specimen
                   hemolysis") at the same patient + specimen time reporting >= 1+ / trace
                   haemolysis. Rare (~1e-4 of K draws).
  spike_unconfirmed  K >= 5.5 whose first later K within 6 h is < 5.0 with NO potassium-
                   lowering therapy given between draw - 1 h and that repeat -- the MIMIC
                   pseudohyperkalaemia definition. Therapy (Medications MAR, administered
                   actions only): insulin, dextrose 50 %, K binders (polystyrene, patiromer,
                   zirconium cyclosilicate), calcium gluconate / chloride. Dialysis is not in the
                   MAR, so patients with an ESRD / dialysis ICD code (N18.6, Z99.2, 585.6,
                   V45.11) are never flagged (a post-dialysis normal K is not spurious).

Medication times were verified aligned with lab times (K-lowering drugs peak 0-2 h after a
K >= 6.5 draw). Which cleaning level to train on is chosen on HEEDB validation only.

Writes data/heedb/heedb_ecg_k_labels.parquet = heedb_ecg_k_idx.parquet + k_time,
hemo_grade_pos, spike_unconfirmed, esrd.
"""
import os, duckdb

HEEDB = os.environ.get("HEEDB_ROOT", "/path/to/heedb")
ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTD = f"{ROOT}/data/heedb"
LABS, MEDS = f"{HEEDB}/EHR/I0001/Labs/*.parquet", f"{HEEDB}/EHR/I0001/Medications/*.parquet"
ICD10, ICD9 = f"{HEEDB}/ECG/I0001/ICD_codes/icd10_codes.csv", f"{HEEDB}/ECG/I0001/ICD_codes/icd9_codes.csv"
GIVEN = "'Given','New Bag','Rate Change','Rate Verify','Restarted','Bolus','Push','Given During Downtime'"
RX = ("insulin|dextrose 50|d50w|polystyrene|kayexalate|patiromer|veltassa|zirconium|lokelma|"
      "calcium gluconate|calcium chloride")

con = duckdb.connect(); con.execute("PRAGMA threads=24"); con.execute("PRAGMA memory_limit='48GB'")
con.execute(f"SET temp_directory='{HEEDB}/duckdb_tmp'")
con.execute(f"CREATE TEMP TABLE p AS SELECT * FROM read_parquet('{OUTD}/heedb_ecg_k_idx.parquet')")

con.execute(f"""
CREATE TEMP TABLE k AS
SELECT BDSPPatientID pid, try_cast(ResultTXT AS DOUBLE) k, try_cast(SpecimenTakenTimeDTS AS TIMESTAMP) t
FROM '{LABS}'
WHERE (LoincTXT IN ('2823-3', '6298-4', '32713-0') OR lower(ComponentNM) LIKE 'potassium%')
  AND try_cast(ResultTXT AS DOUBLE) BETWEEN 1 AND 12
  AND strftime(try_cast(SpecimenTakenTimeDTS AS TIMESTAMP), '%H:%M:%S') <> '00:00:00'
  AND BDSPPatientID IN (SELECT DISTINCT pid FROM p)
""")
# the K matched to each pair (same nearest-within-2h rule as 25_heedb_pairs.py), now with its time
con.execute("""
CREATE TEMP TABLE pk AS
SELECT p.file, arg_min(k.t, abs(date_diff('second', k.t, p.ecg_time))) k_time
FROM p JOIN k ON k.pid = p.pid AND k.t BETWEEN p.ecg_time - INTERVAL 2 HOUR AND p.ecg_time + INTERVAL 2 HOUR
GROUP BY 1
""")
con.execute(f"""
CREATE TEMP TABLE hg AS
SELECT DISTINCT BDSPPatientID pid, try_cast(SpecimenTakenTimeDTS AS TIMESTAMP) t
FROM '{LABS}'
WHERE (LoincTXT = '20395-0' OR lower(ComponentNM) LIKE 'hemolysis grad%' OR lower(ComponentNM) LIKE '%specimen hemolysis%')
  AND (lower(ResultTXT) LIKE '%+%' OR lower(ResultTXT) LIKE '%trace%' OR lower(ResultTXT) LIKE '%gross%'
       OR lower(ResultTXT) LIKE '%moderate%' OR lower(ResultTXT) LIKE '%slight%')
  AND lower(ResultTXT) NOT LIKE '%no hemolysis%'
""")
con.execute(f"""
CREATE TEMP TABLE rx AS
SELECT BDSPPatientID pid, try_cast(MedicationTakenDTS AS TIMESTAMP) t
FROM '{MEDS}'
WHERE MARActionDSC IN ({GIVEN}) AND regexp_matches(lower(MedicationDSC), '{RX}')
  AND BDSPPatientID IN (SELECT DISTINCT pid FROM p)
""")
con.execute(f"""
CREATE TEMP TABLE esrd AS
SELECT DISTINCT CAST(BDSPPatientID AS BIGINT) pid FROM read_csv('{ICD10}', header=true, all_varchar=true)
 WHERE DIAGNOSIS_ICD10_CD IN ('N18.6', 'Z99.2')
UNION
SELECT DISTINCT CAST(BDSPPatientID AS BIGINT) FROM read_csv('{ICD9}', header=true, all_varchar=true)
 WHERE DIAGNOSIS_ICD9_CD IN ('585.6', 'V45.11')
""")
con.execute("""
CREATE TEMP TABLE hi AS
SELECT DISTINCT p.pid, pk.k_time FROM p JOIN pk USING(file) WHERE p.k >= 5.5
""")
con.execute("""
CREATE TEMP TABLE sp AS
WITH rep AS (
  SELECT hi.pid, hi.k_time, arg_min(k.k, k.t) k_rep, min(k.t) t_rep
  FROM hi JOIN k ON k.pid = hi.pid AND k.t > hi.k_time AND k.t <= hi.k_time + INTERVAL 6 HOUR
  GROUP BY 1, 2
)
SELECT rep.pid, rep.k_time,
  (rep.k_rep < 5.0
   AND NOT EXISTS (SELECT 1 FROM rx WHERE rx.pid = rep.pid
                   AND rx.t BETWEEN rep.k_time - INTERVAL 1 HOUR AND rep.t_rep)) AS spike_unconfirmed
FROM rep
""")
con.execute(f"""
COPY (
  SELECT p.*, pk.k_time,
    EXISTS (SELECT 1 FROM hg WHERE hg.pid = p.pid AND hg.t = pk.k_time) AS hemo_grade_pos,
    (p.pid IN (SELECT pid FROM esrd)) AS esrd,
    coalesce(sp.spike_unconfirmed, FALSE) AND p.pid NOT IN (SELECT pid FROM esrd) AS spike_unconfirmed
  FROM p JOIN pk USING(file)
  LEFT JOIN sp ON sp.pid = p.pid AND sp.k_time = pk.k_time
) TO '{OUTD}/heedb_ecg_k_labels.parquet' (FORMAT parquet)
""")
print(con.sql(f"""SELECT COUNT(*) pairs, SUM((k >= 5.5)::INT) hyperK,
  SUM(hemo_grade_pos::INT) hemo_grade_pos, SUM(esrd::INT) esrd_pairs,
  SUM(spike_unconfirmed::INT) spike_unconfirmed,
  ROUND(100.0 * SUM(spike_unconfirmed::INT) / SUM((k >= 5.5)::INT), 1) pct_of_hyperK_flagged
  FROM read_parquet('{OUTD}/heedb_ecg_k_labels.parquet')""").fetchdf().T.to_string())
print(f"wrote {OUTD}/heedb_ecg_k_labels.parquet")
