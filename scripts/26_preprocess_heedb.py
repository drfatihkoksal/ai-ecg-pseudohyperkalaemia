"""v3/HEEDB stage 2: waveform store for the HEEDB ECG-potassium pairs (25_heedb_pairs.py).

Identical preprocessing to MIMIC (06_preprocess_waveforms.preprocess_one, which now takes
the record's own sampling rate: HEEDB mixes 250 Hz and 500 Hz; MIMIC output verified
bit-identical after that change). Strips are stored float16 like the MIMIC v3 store.
Store lives on /home (~120 GB for ~2 M ECGs; the project disk is near full).

Writes <HEEDB>/derived/strips16.npy, <HEEDB>/derived/meta.parquet (idx, file, ok, reason,
hr, n_bad_leads), and adds idx to data/heedb/heedb_ecg_k.parquet -> heedb_ecg_k_idx.parquet.
"""
import os, sys, time, argparse, importlib.util
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEEDB = os.environ.get("HEEDB_ROOT", "/path/to/heedb")
OUTD, STORE = f"{ROOT}/data/heedb", f"{HEEDB}/derived"
spec = importlib.util.spec_from_file_location("pp", f"{ROOT}/scripts/06_preprocess_waveforms.py")
PP = importlib.util.module_from_spec(spec); sys.modules["pp"] = PP; spec.loader.exec_module(PP)


def main():
    """Chunked and resumable: CHUNK ECGs at a time, the memmap is flushed and a per-chunk
    meta file written after each chunk; finished chunks are skipped on restart. (The first,
    unchunked run was reaped by the host under memory pressure while a 48 GB DuckDB job ran
    alongside it.)"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=200_000)
    a = ap.parse_args()
    os.makedirs(STORE, exist_ok=True)
    p = pd.read_parquet(f"{OUTD}/heedb_ecg_k.parquet")
    files = np.sort(p.file.unique())
    idx = pd.Series(np.arange(len(files)), index=files)
    p["idx"] = idx.reindex(p.file).values
    if not os.path.exists(f"{OUTD}/heedb_ecg_k_idx.parquet"):
        p.to_parquet(f"{OUTD}/heedb_ecg_k_idx.parquet", index=False)
    n = len(files)
    print(f"HEEDB ECGs to preprocess: {n:,} (pairs {len(p):,}, patients {p.pid.nunique():,})", flush=True)

    spath = f"{STORE}/strips16.npy"
    shape = (n, 12, PP.N_OUT)
    if os.path.exists(spath) and np.load(spath, mmap_mode="r").shape == shape:
        strips = np.load(spath, mmap_mode="r+")
    else:
        strips = np.lib.format.open_memmap(spath, mode="w+", dtype=np.float16, shape=shape)

    starts = list(range(0, n, a.chunk))
    t0 = time.time()
    for ci, s0 in enumerate(starts):
        mp = f"{STORE}/meta_part_{ci:03d}.parquet"
        if os.path.exists(mp):
            print(f"  [skip] chunk {ci+1}/{len(starts)}", flush=True); continue
        tasks = [(i, 0, files[i]) for i in range(s0, min(s0 + a.chunk, n))]
        meta = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for m, x, _ in ex.map(PP.preprocess_one, tasks, chunksize=64):
                if m["ok"]:
                    strips[m["idx"]] = x.astype(np.float16)
                meta.append({k: m[k] for k in ("idx", "ok", "reason", "hr", "n_bad_leads", "n_beats")})
        strips.flush()
        md = pd.DataFrame(meta); md["file"] = files[md.idx.values]
        md.to_parquet(mp, index=False)
        el = time.time() - t0
        print(f"  chunk {ci+1}/{len(starts)} done: ok={100*md.ok.mean():.1f}%  {el/60:.1f} min elapsed", flush=True)

    md = pd.concat([pd.read_parquet(f"{STORE}/meta_part_{ci:03d}.parquet") for ci in range(len(starts))]).sort_values("idx")
    assert len(md) == n and (md.idx.values == np.arange(n)).all(), "meta incomplete"
    md.to_parquet(f"{STORE}/meta.parquet", index=False)
    print(f"\nusable {md.ok.sum():,}/{n:,} ({100*md.ok.mean():.1f}%)")
    print(md[~md.ok].reason.value_counts().to_string())
    print(f"wrote {spath} ({strips.nbytes/1e9:.0f} GB), {STORE}/meta.parquet, {OUTD}/heedb_ecg_k_idx.parquet")


if __name__ == "__main__":
    main()
