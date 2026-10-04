"""Stage 6c: give the null pairs the SAME patient-level split as the analysis pairs.

The v2 null-pair check (13_evaluate_v2.py, Q4) found the difference arms emit 75-78%
of their real-pair dK spread on null pairs (same patient, 0.5-6 h apart): most of the
output variance is per-acquisition nuisance (concept.md sec.5 item 3).

The nuisance-reduction step (09_train.py --null_reg) TRAINS on null pairs, so the null
check can no longer use all of them. Null pairs inherit the stage-3 split rule
(hash(subject_id + SEED)), which reproduces pairs_v2's split for 31,737/31,737
patients: train-split null pairs regularise, test-split null pairs evaluate, and no
patient crosses.
"""
import os, duckdb

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = f"{ROOT}/data"
SEED = 20260713  # stage-3 pre-registered split seed

con = duckdb.connect()
con.execute(f"""
CREATE TEMP TABLE n AS
SELECT *, CASE WHEN hash(subject_id + {SEED}) % 100 < 70 THEN 'train'
               WHEN hash(subject_id + {SEED}) % 100 < 85 THEN 'val'
               ELSE 'test' END AS split
FROM read_parquet('{DATA}/null_pairs_final.parquet')
""")
bad = con.sql(f"""SELECT COUNT(*) FROM n JOIN (SELECT DISTINCT subject_id, split s2
                 FROM read_parquet('{DATA}/pairs_v2.parquet')) p USING(subject_id)
                 WHERE n.split <> p.s2""").fetchone()[0]
assert bad == 0, f"{bad} null pairs disagree with the analysis split"
con.execute(f"COPY n TO '{DATA}/null_pairs_v2.parquet' (FORMAT parquet)")
print(con.sql("""SELECT split, COUNT(*) n, COUNT(DISTINCT subject_id) subj,
  ROUND(100.0*AVG((dk=0)::INT),1) pct_dk0 FROM n GROUP BY 1 ORDER BY 1""").fetchdf().to_string(index=False))
print(f"wrote {DATA}/null_pairs_v2.parquet")
