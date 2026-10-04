"""v3 stage 2: waveform store for the pseudohyperkalaemia study (concept.md sec.0.5).

Which ECGs:
  POOL   every MIMIC-IV-ECG record with a NON-haemolysed serum K within +-2 h -- the
         training pool of the absolute-K ECG model (label = nearest such K). The
         literature standard is lab within 1-4 h; v1/v2 used +-12 h.
  INDEX  the nearest ECG of every v3 index event (pseudohyperk_index.parquet, all labels)
  PRIOR  the most recent ECG > 12 h before each index (prior-ECG analyses)

Why a separate store: the v1/v2 store (waveforms/strips.npy, 20 GB float32) is left
untouched for reproducibility, and the disk has ~90 GB free -- so strips are kept as
float16 (mV; ~1e-3 relative precision, far below ECG noise) and median beats are not
written (no beat-subtraction arm in v3). Preprocessing itself is IDENTICAL to stage 5:
the same preprocess_one() is imported from 06_preprocess_waveforms.py.

Writes: waveforms_v3/strips16.npy, waveforms_v3/meta.parquet (idx, study_id, ok, hr,
n_bad_leads, reason), and data/v3_ecg_k.parquet (study_id, subject_id, k, gap_h, role).
Resumable: an existing meta.parquet with the same study list is reused.
"""
import os, sys, time, argparse, importlib.util
import numpy as np, pandas as pd, duckdb
from concurrent.futures import ProcessPoolExecutor

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = f"{ROOT}/data"
WF3 = f"{DATA}/waveforms_v3"
ECGDIR = os.environ.get("MIMIC_ECG_ROOT", "/path/to/mimic-iv-ecg/1.0")
spec = importlib.util.spec_from_file_location("pp", f"{ROOT}/scripts/06_preprocess_waveforms.py")
PP = importlib.util.module_from_spec(spec); sys.modules["pp"] = PP; spec.loader.exec_module(PP)  # picklable for workers
LAB_WIN_H = 2


def build_list():
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    con.execute(f"""CREATE TEMP TABLE ecg AS SELECT subject_id, study_id, CAST(ecg_time AS TIMESTAMP) t, path
      FROM read_csv_auto('{ECGDIR}/record_list.csv')""")
    con.execute(f"""CREATE TEMP TABLE k AS SELECT subject_id, charttime, valuenum k
      FROM read_parquet('{DATA}/labs_k_comments.parquet')
      WHERE valuenum BETWEEN 1 AND 12
        AND NOT (lower(coalesce(comments,'')) LIKE '%hemoly%' AND lower(coalesce(comments,'')) NOT LIKE '%not hemoly%')""")
    con.execute(f"""CREATE TEMP TABLE pool AS
      SELECT e.study_id, e.subject_id, e.path,
        arg_min(k.k, abs(date_diff('second', k.charttime, e.t))) k,
        min(abs(date_diff('second', k.charttime, e.t))) / 3600.0 gap_h
      FROM ecg e JOIN k ON k.subject_id = e.subject_id
       AND k.charttime BETWEEN e.t - INTERVAL {LAB_WIN_H} HOUR AND e.t + INTERVAL {LAB_WIN_H} HOUR
      GROUP BY 1, 2, 3""")
    con.execute(f"""CREATE TEMP TABLE extra AS
      SELECT ecg_study AS study_id, 'index' AS src FROM read_parquet('{DATA}/pseudohyperk_index.parquet')
      UNION SELECT prior_ecg_study, 'prior' AS src FROM read_parquet('{DATA}/pseudohyperk_index.parquet')
       WHERE prior_ecg_study IS NOT NULL""")
    lst = con.sql("""
      WITH s AS (SELECT study_id FROM pool UNION SELECT study_id FROM extra)
      SELECT s.study_id, e.subject_id, e.path, p.k, p.gap_h,
        (p.study_id IS NOT NULL) in_pool,
        s.study_id IN (SELECT study_id FROM extra WHERE src='index') is_index,
        s.study_id IN (SELECT study_id FROM extra WHERE src='prior') is_prior
      FROM s JOIN ecg e USING(study_id) LEFT JOIN pool p USING(study_id)
      ORDER BY s.study_id""").fetchdf()
    return lst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    os.makedirs(WF3, exist_ok=True)
    lst = build_list()
    lst["idx"] = np.arange(len(lst))
    n = len(lst)
    print(f"ECGs: {n:,}  (pool {int(lst.in_pool.sum()):,}, index {int(lst.is_index.sum()):,}, "
          f"prior {int(lst.is_prior.sum()):,}; patients {lst.subject_id.nunique():,})", flush=True)
    lst[["idx", "study_id", "subject_id", "k", "gap_h", "in_pool", "is_index", "is_prior"]] \
        .to_parquet(f"{DATA}/v3_ecg_k.parquet", index=False)

    mpath, spath = f"{WF3}/meta.parquet", f"{WF3}/strips16.npy"
    if os.path.exists(mpath) and os.path.exists(spath):
        old = pd.read_parquet(mpath)
        if len(old) == n and (old.study_id.values == lst.study_id.values).all():
            print("store already complete -- nothing to do"); return
    strips = np.lib.format.open_memmap(spath, mode="w+", dtype=np.float16, shape=(n, 12, PP.N_OUT))

    tasks = [(int(r.idx), int(r.study_id), r.path) for r in lst.itertuples()]
    meta, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for m, x, _ in ex.map(PP.preprocess_one, tasks, chunksize=64):
            if m["ok"]:
                strips[m["idx"]] = np.clip(x, -60000, 60000).astype(np.float16)
            meta.append({k: m[k] for k in ("idx", "study_id", "ok", "reason", "hr", "n_bad_leads", "n_beats")})
            done += 1
            if done % 10000 == 0:
                ok = sum(d["ok"] for d in meta); el = time.time() - t0
                print(f"  {done:>7,}/{n:,}  ok={100*ok/done:.1f}%  {el/60:.0f} min elapsed, "
                      f"~{el/done*(n-done)/60:.0f} min left", flush=True)
    strips.flush()
    md = pd.DataFrame(meta).sort_values("idx")
    md.to_parquet(mpath, index=False)
    print(f"\nusable {md.ok.sum():,}/{n:,} ({100*md.ok.mean():.1f}%)")
    print(md[~md.ok].reason.value_counts().to_string())
    print(f"wrote {spath} ({strips.nbytes/1e9:.1f} GB), {mpath}, {DATA}/v3_ecg_k.parquet")


if __name__ == "__main__":
    main()
