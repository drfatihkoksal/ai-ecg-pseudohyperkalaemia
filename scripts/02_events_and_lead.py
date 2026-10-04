"""Stage 2: the numbers that actually decide the paper.

Stage 1 showed the paired cohort survives. But pair count alone does not gate the
design. Two further numbers do:

  (a) KDIGO AKI EVENT count with ECG coverage  -> powers the classification arm
      and the whole "AKI" framing. A regression on dCr can be well-powered while
      the AKI arm is not.
  (b) Lead-hypothesis pairs (concept.md sec.4): dECG(t) with a creatinine
      measured 12-36 h AFTER the second ECG -> powers the early-warning claim,
      which concept.md calls "the strongest clinical argument in the paper".

Also reports the dK/dCr coupling, because sec.5 says the paper lives or dies on
whether the renal signal is separable from the potassium signal.
"""
import duckdb, os
import pandas as pd

MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ECGDIR = os.environ.get("MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg/1.0")
_ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "outputs")
CREATININE, POTASSIUM = 50912, 50971
LAB_WIN_H, PAIR_MIN_H, PAIR_MAX_H = 12, 12, 168
NEXTDAY_LO, NEXTDAY_HI = 12, 36

con = duckdb.connect(); con.execute("PRAGMA threads=16")
con.execute(f"""CREATE TEMP TABLE ecg AS SELECT subject_id, study_id,
  CAST(ecg_time AS TIMESTAMP) ecg_time FROM read_csv_auto('{ECGDIR}/record_list.csv')""")
con.execute(f"""CREATE TEMP TABLE adm AS SELECT subject_id, hadm_id,
  CAST(admittime AS TIMESTAMP) admittime, CAST(dischtime AS TIMESTAMP) dischtime
  FROM read_csv_auto('{MIMIC}/hosp/admissions.csv')""")
con.execute(f"""CREATE TEMP TABLE dx AS SELECT hadm_id, CAST(icd_code AS VARCHAR) icd_code,
  icd_version FROM read_csv_auto('{MIMIC}/hosp/diagnoses_icd.csv', types={{'icd_code':'VARCHAR'}})""")
con.execute(f"""CREATE TEMP TABLE px AS SELECT hadm_id, CAST(icd_code AS VARCHAR) icd_code,
  icd_version FROM read_csv_auto('{MIMIC}/hosp/procedures_icd.csv', types={{'icd_code':'VARCHAR'}})""")
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

# analysis cohorts: strict (ACS n PCI) and the operational fallback (ACS n any-contrast)
COHORTS = {
  "T4_acs_AND_pci": "hadm_id IN (SELECT hadm_id FROM acs_hadm) AND hadm_id IN (SELECT hadm_id FROM pci_hadm)",
  "T5_acs_AND_cath_or_pci": ("hadm_id IN (SELECT hadm_id FROM acs_hadm) AND hadm_id IN "
                             "(SELECT hadm_id FROM cath_hadm UNION SELECT hadm_id FROM pci_hadm)"),
  "T0_all_admissions": "1=1",
}

summary = []
for tier, pred in COHORTS.items():
    con.execute(f"DROP TABLE IF EXISTS coh; CREATE TEMP TABLE coh AS SELECT hadm_id FROM adm WHERE {pred}")

    # ---- ECGs in cohort admissions, with nearest Cr / K within +-LAB_WIN_H
    con.execute(f"""
    DROP TABLE IF EXISTS el; CREATE TEMP TABLE el AS
    WITH ea AS (
      SELECT e.subject_id, e.study_id, e.ecg_time, a.hadm_id, a.admittime
      FROM ecg e JOIN adm a ON e.subject_id=a.subject_id
             AND e.ecg_time BETWEEN a.admittime AND a.dischtime
      JOIN coh c USING(hadm_id)
    ),
    cr AS (SELECT ea.study_id,
             arg_min(l.valuenum, abs(date_diff('minute', ea.ecg_time, l.charttime))) cr
           FROM ea JOIN labs l ON l.subject_id=ea.subject_id AND l.itemid={CREATININE}
             AND l.charttime BETWEEN ea.ecg_time - INTERVAL {LAB_WIN_H} HOUR
                                 AND ea.ecg_time + INTERVAL {LAB_WIN_H} HOUR
           GROUP BY 1),
    k AS (SELECT ea.study_id,
             arg_min(l.valuenum, abs(date_diff('minute', ea.ecg_time, l.charttime))) k
          FROM ea JOIN labs l ON l.subject_id=ea.subject_id AND l.itemid={POTASSIUM}
             AND l.charttime BETWEEN ea.ecg_time - INTERVAL {LAB_WIN_H} HOUR
                                 AND ea.ecg_time + INTERVAL {LAB_WIN_H} HOUR
          GROUP BY 1)
    SELECT ea.*, cr.cr, k.k FROM ea JOIN cr USING(study_id) JOIN k USING(study_id)
    """)

    # ---- consecutive lab-matched ECG pairs
    con.execute(f"""
    DROP TABLE IF EXISTS pairs; CREATE TEMP TABLE pairs AS
    WITH seq AS (
      SELECT *, LEAD(ecg_time) OVER w t2, LEAD(study_id) OVER w study2,
                LEAD(cr) OVER w cr2, LEAD(k) OVER w k2
      FROM el WINDOW w AS (PARTITION BY hadm_id ORDER BY ecg_time)
    )
    SELECT subject_id, hadm_id, study_id AS study1, study2, ecg_time AS t1, t2,
           date_diff('hour', ecg_time, t2) dt_h,
           cr, cr2, cr2-cr AS dcr, k, k2, k2-k AS dk
    FROM seq WHERE t2 IS NOT NULL
      AND date_diff('hour', ecg_time, t2) BETWEEN {PAIR_MIN_H} AND {PAIR_MAX_H}
    """)

    # ---- KDIGO on the pair itself:
    #   stage>=1 if dCr >= 0.3 within 48h, OR cr2 >= 1.5 * baseline (admission-min Cr, 7d)
    con.execute("""
    DROP TABLE IF EXISTS pairs_k; CREATE TEMP TABLE pairs_k AS
    WITH base AS (  -- KDIGO baseline: lowest creatinine of the admission
      SELECT hadm_id, MIN(cr) AS cr_base FROM el GROUP BY hadm_id
    )
    SELECT p.*, b.cr_base,
      (p.dcr >= 0.3 AND p.dt_h <= 48) AS kdigo_abs,
      (p.cr2 >= 1.5 * b.cr_base)      AS kdigo_rel,
      ((p.dcr >= 0.3 AND p.dt_h <= 48) OR (p.cr2 >= 1.5 * b.cr_base)) AS aki_pair
    FROM pairs p JOIN base b USING(hadm_id)
    """)

    # ---- lead hypothesis: a creatinine 12-36h AFTER the 2nd ECG of the pair
    con.execute(f"""
    DROP TABLE IF EXISTS lead; CREATE TEMP TABLE lead AS
    SELECT p.*,
           n.cr_next, n.cr_next - p.cr2 AS dcr_next
    FROM pairs_k p
    LEFT JOIN (
      SELECT p.study1,
             arg_min(l.valuenum, l.charttime) AS cr_next
      FROM pairs_k p JOIN labs l ON l.subject_id=p.subject_id AND l.itemid={CREATININE}
        AND l.charttime BETWEEN p.t2 + INTERVAL {NEXTDAY_LO} HOUR
                            AND p.t2 + INTERVAL {NEXTDAY_HI} HOUR
      GROUP BY p.study1
    ) n USING(study1)
    """)

    r = con.sql("""
    SELECT
      (SELECT COUNT(*) FROM pairs_k)                               n_pairs,
      (SELECT COUNT(DISTINCT subject_id) FROM pairs_k)             n_subj,
      (SELECT SUM(CASE WHEN aki_pair THEN 1 ELSE 0 END) FROM pairs_k) n_aki_pairs,
      (SELECT COUNT(DISTINCT hadm_id) FROM pairs_k WHERE aki_pair) n_aki_hadm,
      (SELECT COUNT(DISTINCT subject_id) FROM pairs_k WHERE aki_pair) n_aki_subj,
      (SELECT COUNT(*) FROM lead WHERE cr_next IS NOT NULL)         n_lead_pairs,
      (SELECT COUNT(DISTINCT subject_id) FROM lead WHERE cr_next IS NOT NULL) n_lead_subj,
      (SELECT SUM(CASE WHEN cr_next IS NOT NULL AND dcr_next>=0.3 THEN 1 ELSE 0 END) FROM lead) n_lead_aki,
      (SELECT CORR(dcr, dk) FROM pairs_k)                          corr_dcr_dk,
      (SELECT STDDEV(dcr) FROM pairs_k)                            sd_dcr,
      (SELECT STDDEV(dk) FROM pairs_k)                             sd_dk,
      (SELECT MEDIAN(dt_h) FROM pairs_k)                           med_dt_h
    """).fetchdf().iloc[0].to_dict()
    r = {"tier": tier, **r}
    r["aki_pair_rate"] = r["n_aki_pairs"] / max(r["n_pairs"], 1)
    summary.append(r)

    print(f"\n=== {tier} ===")
    print(f"  usable delta pairs        : {r['n_pairs']:>8,.0f}   ({r['n_subj']:,.0f} patients)")
    print(f"  KDIGO-positive pairs      : {r['n_aki_pairs']:>8,.0f}   "
          f"({r['aki_pair_rate']*100:.1f}% of pairs; {r['n_aki_subj']:,.0f} patients)")
    print(f"  lead-hypothesis pairs     : {r['n_lead_pairs']:>8,.0f}   "
          f"(next-day Cr present; {r['n_lead_subj']:,.0f} patients)")
    print(f"    of which next-day AKI   : {r['n_lead_aki']:>8,.0f}")
    print(f"  median dt between ECGs    : {r['med_dt_h']:>8.1f} h")
    print(f"  corr(dCr, dK)             : {r['corr_dcr_dk']:>8.3f}   "
          f"(sd dCr={r['sd_dcr']:.2f}, sd dK={r['sd_dk']:.2f})")

pd.DataFrame(summary).to_csv(f"{OUT}/events_and_lead.csv", index=False)
con.execute(f"COPY (SELECT * FROM lead) TO '{DATA}/pairs_T0.parquet' (FORMAT parquet)")
print(f"\nwrote {OUT}/events_and_lead.csv")
