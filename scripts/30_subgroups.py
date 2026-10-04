"""v3/HEEDB stage 6: why does the frozen HEEDB model lose hyperkalaemia accuracy on MIMIC-IV
(hyperK AUC 0.872 -> 0.767; K >= 6.0 0.892 -> 0.712) while hypokalaemia transfers (0.846 ->
0.852)? A MIMIC-trained model hits the same ceiling, so the cause is in the MIMIC cohort/labels.

Exploratory, post hoc (after the one external scoring; the frozen model is not changed).
Same subgroups at both sites, so case mix (more of a hard subgroup) can be told apart from
within-subgroup degradation:
  esrd       ESRD / dialysis ICD code ever (MIMIC N186, Z992, 5856, V4511; HEEDB N18.6, Z99.2,
             585.6, V45.11)
  paced      paced rhythm (MIMIC machine text "pacemaker"/"paced"; HEEDB 12SL 183-186, 289-298, 326)
  af         atrial fibrillation / flutter (MIMIC text; HEEDB 12SL 161, 162, 273, 288)
  bbb        bundle-branch block / IV conduction delay (MIMIC text; HEEDB 12SL 440, 442, 460, 482, 487)
  plain      none of paced / af / bbb
  setting    MIMIC only: ECG during an ICU stay / in hospital outside ICU / outside any admission
Label noise: spike_unconfirmed (K >= 5.5, repeat < 5.0 within 6 h, no strict K-lowering therapy,
ESRD never flagged) -- the HEEDB cleaning rule, now applied to MIMIC with its own therapy tables.
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, duckdb, torch
from torch.utils.data import DataLoader

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEEDB = os.environ.get("HEEDB_ROOT", "/path/to/heedb")
MIMIC = os.environ.get("MIMIC_ROOT", "/path/to/mimiciv/3.1")
ECGDIR = os.environ.get("MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg/1.0")
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
sys.path.insert(0, f"{ROOT}/scripts")
from _evalutils import auc, r2

con = duckdb.connect(); con.execute("PRAGMA threads=16"); con.execute("PRAGMA memory_limit='24GB'")

# ------------------------------------------------------------------ MIMIC
d = pd.read_parquet(f"{DATA}/v3_ecg_k.parquet").merge(
    pd.read_parquet(f"{DATA}/waveforms_v3/meta.parquet")[["idx", "ok", "n_bad_leads"]], on="idx")
d = d[d.in_pool & d.k.notna() & d.ok & (d.n_bad_leads <= 1)].merge(
    pd.read_parquet(f"{OUT}/heedb/mimic_ecgk_external.parquet"), on="study_id")
con.register("pool", d[["study_id", "subject_id", "k"]])
rep = " || ' | ' || ".join(f"coalesce(lower(trim(report_{i})), '')" for i in range(18))
f = con.sql(f"""
WITH e AS (SELECT CAST(study_id AS BIGINT) study_id, CAST(ecg_time AS TIMESTAMP) t
           FROM read_csv_auto('{ECGDIR}/record_list.csv')),
mm AS (SELECT CAST(study_id AS BIGINT) study_id, {rep} txt,
         TRY_CAST(qrs_onset AS DOUBLE) qon, TRY_CAST(qrs_end AS DOUBLE) qend
       FROM read_csv_auto('{ECGDIR}/machine_measurements.csv', all_varchar=true)),
esrd AS (SELECT DISTINCT subject_id FROM read_csv_auto('{MIMIC}/hosp/diagnoses_icd.csv.gz', types={{'icd_code':'VARCHAR'}})
         WHERE icd_code IN ('N186', 'Z992', '5856', 'V4511')),
icu AS (SELECT subject_id, CAST(intime AS TIMESTAMP) a, CAST(outtime AS TIMESTAMP) b FROM read_csv_auto('{MIMIC}/icu/icustays.csv.gz')),
adm AS (SELECT subject_id, CAST(admittime AS TIMESTAMP) a, CAST(dischtime AS TIMESTAMP) b FROM read_csv_auto('{MIMIC}/hosp/admissions.csv.gz'))
SELECT p.study_id, e.t,
  p.subject_id IN (SELECT subject_id FROM esrd) esrd,
  (mm.txt LIKE '%pacemaker%' OR mm.txt LIKE '%paced%') paced,
  (mm.txt LIKE '%atrial fibrillation%' OR mm.txt LIKE '%atrial flutter%') af,
  (mm.txt LIKE '%bundle branch block%' OR mm.txt LIKE '%iv conduction%' OR mm.txt LIKE '%intraventricular%') bbb,
  CASE WHEN EXISTS (SELECT 1 FROM icu WHERE icu.subject_id = p.subject_id AND e.t BETWEEN icu.a AND icu.b) THEN 'ICU'
       WHEN EXISTS (SELECT 1 FROM adm WHERE adm.subject_id = p.subject_id AND e.t BETWEEN adm.a AND adm.b) THEN 'ward'
       ELSE 'outside admission' END setting
FROM pool p JOIN e USING(study_id) LEFT JOIN mm USING(study_id)
""").fetchdf()
d = d.merge(f, on="study_id")
for c in ("paced", "af", "bbb"): d[c] = d[c].fillna(False).astype(bool)
d["plain"] = ~(d.paced | d.af | d.bbb)

# MIMIC label noise: unconfirmed spikes (same rule as HEEDB; therapy from MIMIC tables)
con.register("dd", d[["study_id", "subject_id", "k", "t", "esrd"]])
GIVEN = ("'Administered','Confirmed','Started','Restarted','Delayed Administered',"
         "'Administered in Other Location','Partial Administered','in Other Location'")
sp = con.sql(f"""
WITH k AS (SELECT subject_id, charttime t, valuenum k FROM read_parquet('{DATA}/labs_k_comments.parquet')
           WHERE valuenum BETWEEN 1 AND 12
             AND NOT (lower(coalesce(comments,'')) LIKE '%hemoly%' AND lower(coalesce(comments,'')) NOT LIKE '%not hemoly%')),
hi AS (SELECT dd.study_id, dd.subject_id, arg_min(k.t, abs(date_diff('second', k.t, dd.t))) kt
       FROM dd JOIN k ON k.subject_id = dd.subject_id AND k.t BETWEEN dd.t - INTERVAL 2 HOUR AND dd.t + INTERVAL 2 HOUR
       WHERE dd.k >= 5.5 AND NOT dd.esrd GROUP BY 1, 2),
rep AS (SELECT hi.study_id, hi.subject_id, hi.kt, arg_min(k.k, k.t) k_rep, min(k.t) t_rep
        FROM hi JOIN k ON k.subject_id = hi.subject_id AND k.t > hi.kt AND k.t <= hi.kt + INTERVAL 6 HOUR GROUP BY 1, 2, 3),
rx AS (SELECT CAST(subject_id AS BIGINT) subject_id, charttime t0, charttime t1 FROM read_parquet('{DATA}/emar_klower_raw.parquet')
       WHERE event_txt IN ({GIVEN}) AND regexp_matches(lower(medication),
         'insulin|dextrose 50|polystyrene|zirconium|lokelma|patiromer|veltassa|calcium gluconate|calcium chloride')
         AND lower(medication) NOT LIKE '%sliding scale%'
       UNION ALL SELECT subject_id, CAST(starttime AS TIMESTAMP), CAST(endtime AS TIMESTAMP) FROM read_csv_auto('{MIMIC}/icu/inputevents.csv.gz')
       WHERE itemid IN (223257,223258,223259,223260,223261,223262,229299,229619,220952,221456,228317,229640,229618)
       UNION ALL SELECT subject_id, CAST(starttime AS TIMESTAMP), CAST(endtime AS TIMESTAMP) FROM read_csv_auto('{MIMIC}/icu/procedureevents.csv.gz')
       WHERE itemid IN (225441, 225802, 225803, 225805, 225809, 225955))
SELECT rep.study_id, (rep.k_rep < 5.0 AND NOT EXISTS (SELECT 1 FROM rx WHERE rx.subject_id = rep.subject_id
        AND rx.t0 <= rep.t_rep AND coalesce(rx.t1, rx.t0) >= rep.kt - INTERVAL 1 HOUR)) spike_unconfirmed
FROM rep
""").fetchdf()
d = d.merge(sp, on="study_id", how="left"); d["spike_unconfirmed"] = d.spike_unconfirmed.astype("boolean").fillna(False).astype(bool)

# ------------------------------------------------------------------ HEEDB internal test
h = pd.read_parquet(f"{DATA}/heedb/heedb_ecg_k_labels.parquet").merge(
    pd.read_parquet(f"{HEEDB}/derived/meta.parquet")[["idx", "ok", "n_bad_leads"]], on="idx")
h = h[h.ok & (h.n_bad_leads <= 1)].reset_index(drop=True)
pats = np.sort(h.pid.unique()); u = np.random.default_rng(20260713).random(len(pats))
h["split"] = pd.Series(np.where(u < 0.8, "train", np.where(u < 0.9, "val", "test")), index=pats).reindex(h.pid).values
h = h[h.split == "test"].reset_index(drop=True)
spec = importlib.util.spec_from_file_location("tr", f"{ROOT}/scripts/28_train_k_heedb.py")
TR = importlib.util.module_from_spec(spec); spec.loader.exec_module(TR)
import json
fz = json.load(open(f"{DATA}/models/heedb/FROZEN.json"))
ck = torch.load(fz["model"], map_location="cpu", weights_only=False)
m = TR.KNet().to("cuda"); m.load_state_dict(ck["state"])
h["pred_k"] = TR.predict(m, DataLoader(TR.DS(h, ck["scale"], ck["mu"], ck["sd"]), batch_size=512, num_workers=16)) * ck["sd"] + ck["mu"]
h["key"] = h.file.str.replace(f"{HEEDB}/ECG/I0001/WFDB/", "", regex=False)
con.register("hk", h[["key"]])
codes = con.sql(f"""
WITH c AS (SELECT regexp_replace(regexp_replace(regexp_replace(FileName, '[\\r\\n]', '', 'g'), '^\\./', ''), '\\.hea\\s*$', '') AS fkey, codes
           FROM read_csv('{HEEDB}/ECG/I0001/12SL_diagnoses/diagnoses_v24.csv', header=true, all_varchar=true))
SELECT fkey AS "key", codes FROM c WHERE fkey IN (SELECT "key" FROM hk)
""").fetchdf()
codes["cs"] = codes.codes.fillna("").str.split(r",\s*").apply(lambda L: set(int(x) for x in L if x.strip().isdigit()))
h = h.merge(codes[["key", "cs"]], on="key", how="left")
h["cs"] = h.cs.apply(lambda s: s if isinstance(s, set) else set())
has = lambda S: h.cs.apply(lambda s: bool(s & S))
h["paced"] = has({183, 184, 185, 186, 289, 290, 291, 292, 293, 295, 296, 297, 298, 326})
h["af"] = has({161, 162, 273, 288}); h["bbb"] = has({440, 442, 460, 482, 487})
h["plain"] = ~(h.paced | h.af | h.bbb)
h["spike_unconfirmed"] = h.spike_unconfirmed.astype(bool)
print(f"HEEDB test pairs {len(h):,}; 12SL codes matched for {h.cs.apply(len).gt(0).mean():.1%}")


def row(site, name, s):
    y, p = s.k.values, s.pred_k.values
    nh = int((y >= 5.5).sum())
    return dict(site=site, subgroup=name, n=len(s), hyperK=nh, hyperK_prev=round(nh / max(len(s), 1), 4),
                AUC_hyperK=round(auc(y >= 5.5, p), 3) if nh >= 20 else np.nan,
                AUC_K6=round(auc(y >= 6.0, p), 3) if (y >= 6.0).sum() >= 20 else np.nan,
                AUC_hypoK=round(auc(y < 3.5, -p), 3), R2=round(r2(y, p), 3))


R = []
for site, s in (("MIMIC (external)", d), ("HEEDB (internal test)", h)):
    R.append(row(site, "ALL", s))
    for c in ("esrd", "paced", "af", "bbb", "plain"):
        R.append(row(site, c, s[s[c]])); R.append(row(site, f"not {c}", s[~s[c]]))
    R.append(row(site, "plain & not esrd", s[s.plain & ~s.esrd]))
    R.append(row(site, "ALL minus unconfirmed spikes", s[~s.spike_unconfirmed]))
for st in ("ICU", "ward", "outside admission"):
    R.append(row("MIMIC (external)", f"setting: {st}", d[d.setting == st]))
R = pd.DataFrame(R)

# case mix among hyperK cases
mix = []
for site, s in (("MIMIC", d), ("HEEDB", h)):
    hk = s[s.k >= 5.5]
    mix.append(dict(site=site, hyperK=len(hk), **{f"% {c}": round(100 * hk[c].mean(), 1) for c in ("esrd", "paced", "af", "bbb", "plain")},
                    **{"% unconfirmed spike": round(100 * hk.spike_unconfirmed.mean(), 1)}))
M = pd.DataFrame(mix)
pd.set_option("display.width", 250)
print("\n=== case mix among hyperkalaemic ECGs ===")
print(M.to_string(index=False))
print("\n=== accuracy by subgroup (frozen HEEDB model) ===")
print(R.to_string(index=False))
os.makedirs(f"{OUT}/heedb", exist_ok=True)
R.to_csv(f"{OUT}/heedb/subgroups.csv", index=False); M.to_csv(f"{OUT}/heedb/subgroups_casemix.csv", index=False)
print(f"\nwrote {OUT}/heedb/subgroups.csv, subgroups_casemix.csv")
