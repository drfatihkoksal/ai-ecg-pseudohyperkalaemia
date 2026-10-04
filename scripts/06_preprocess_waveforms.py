"""Stage 5: waveform preprocessing for the difference model.

Two decisions here are load-bearing and are NOT the reflex defaults:

1. NO PER-RECORD NORMALISATION. The reflex move in ECG deep learning is to
   z-score each record. That would be actively destructive here: the ECG signature
   of dK is T-wave AMPLITUDE, and per-record z-scoring rescales every record to
   unit variance, deleting the between-record amplitude change that the whole
   difference model is trying to read. Signals stay in physical mV, on a fixed
   global scale.

2. MEDIAN-BEAT REPRESENTATION, not raw-strip subtraction. Two 10-s strips taken a
   day apart are not phase-aligned, so their pointwise difference is dominated by
   where the beats happen to fall -- i.e. by heart rate and phase, which sec.5
   lists as nuisance. Subtracting R-peak-aligned median beats is the only way
   "raw waveform subtraction" (sec.6) means anything. We emit BOTH:
     - strips (12 x 2500 @250Hz) for the Siamese / learned-embedding arms
     - median beats (12 x 200, R at 300 ms) for the raw-subtraction arm

Also emits per-record QC and heart rate, because differencing AMPLIFIES
per-acquisition noise (sec.5 item 3) -- a pair is only as good as its worse ECG,
so QC has to propagate to pair level downstream.
"""
import os, sys, argparse
import numpy as np, pandas as pd, duckdb, wfdb
from scipy import signal as sg
from concurrent.futures import ProcessPoolExecutor

ECGDIR = os.environ.get("MIMIC_ECG_ROOT",
                        "/path/to/mimic-iv-ecg/1.0")
DATA = os.path.join(os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
WF = os.path.join(DATA, "waveforms")

FS_IN, FS_OUT = 500, 250
N_OUT = 2500                 # 10 s @ 250 Hz
LEADS = ["I","II","III","aVR","aVF","aVL","V1","V2","V3","V4","V5","V6"]
BEAT_PRE, BEAT_POST = 75, 125   # samples @250Hz: -300 ms .. +500 ms (covers P through T)
BEAT_LEN = BEAT_PRE + BEAT_POST

# filters (designed once, at module import, so workers inherit them)
SOS_BP = sg.butter(3, [0.5, 40.0], btype="bandpass", fs=FS_OUT, output="sos")
B_NOTCH, A_NOTCH = sg.iirnotch(60.0, Q=30.0, fs=FS_OUT)


def preprocess_one(args):
    idx, study_id, path = args
    out = dict(idx=idx, study_id=study_id, ok=False, reason="",
               hr=np.nan, n_beats=0, n_bad_leads=0, frac_nan=0.0, max_abs_mv=np.nan)
    try:
        rec = wfdb.rdrecord(os.path.join(ECGDIR, path))
    except Exception as e:
        out["reason"] = f"read_fail:{type(e).__name__}"
        return out, None, None

    x = rec.p_signal  # (n, 12) in mV; MIMIC-IV-ECG is 500 Hz, HEEDB is 250 Hz
    fs_in = int(round(rec.fs)) if rec.fs else FS_IN
    if x is None or x.shape[0] < fs_in * 8:
        out["reason"] = "too_short"; return out, None, None

    # reorder to canonical lead order
    try:
        order = [rec.sig_name.index(l) for l in LEADS]
    except ValueError:
        out["reason"] = "missing_lead"; return out, None, None
    x = x[:, order].astype(np.float64).T          # (12, 5000)

    # --- NaNs: MIMIC-IV-ECG has records with NaN runs. Interpolate short gaps,
    # reject the record if a lead is mostly missing.
    frac_nan = float(np.isnan(x).mean())
    out["frac_nan"] = frac_nan
    if frac_nan > 0.5:
        out["reason"] = "mostly_nan"; return out, None, None
    for i in range(12):
        v = x[i]
        m = np.isnan(v)
        if m.all():
            x[i] = 0.0
        elif m.any():
            v[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), v[~m])

    # --- resample to 250 Hz (no-op for 250 Hz sources), then filter
    if fs_in != FS_OUT:
        g = np.gcd(FS_OUT, fs_in)
        x = sg.resample_poly(x, FS_OUT // g, fs_in // g, axis=1)  # (12, 2500)
    if x.shape[1] < N_OUT:
        x = np.pad(x, ((0, 0), (0, N_OUT - x.shape[1])), mode="edge")
    x = x[:, :N_OUT]
    x = sg.sosfiltfilt(SOS_BP, x, axis=1)                   # baseline wander + HF noise
    x = sg.filtfilt(B_NOTCH, A_NOTCH, x, axis=1)            # 60 Hz powerline

    # --- QC. Amplitudes stay in mV; we only FLAG, never rescale.
    ptp = np.ptp(x, axis=1)
    bad = (ptp < 0.05) | (ptp > 15.0)        # dead/flat lead, or saturated/artefact
    out["n_bad_leads"] = int(bad.sum())
    out["max_abs_mv"] = float(np.abs(x).max())
    if bad.sum() > 3:
        out["reason"] = "too_many_bad_leads"; return out, None, None

    # --- R peaks on lead II (fallback: the lead with the largest QRS energy)
    ref = 1 if not bad[1] else int(np.argmax(ptp))
    v = x[ref]
    # QRS enhancement: differentiate, square, moving-window integrate
    d = np.diff(v, prepend=v[0])
    e = sg.convolve(d ** 2, np.ones(int(0.08 * FS_OUT)) / (0.08 * FS_OUT), mode="same")
    thr = np.percentile(e, 98) * 0.35
    peaks, _ = sg.find_peaks(e, height=thr, distance=int(0.25 * FS_OUT))  # >=250 ms apart
    if len(peaks) < 3:
        out["reason"] = "no_rpeaks"; return out, None, None

    # snap each detection to the local |v| maximum (true R apex)
    r = []
    for p in peaks:
        a, b = max(0, p - 20), min(N_OUT, p + 20)
        r.append(a + int(np.argmax(np.abs(v[a:b]))))
    r = np.array(sorted(set(r)))

    rr = np.diff(r) / FS_OUT
    rr = rr[(rr > 0.25) & (rr < 2.5)]
    if len(rr) == 0:
        out["reason"] = "no_valid_rr"; return out, None, None
    out["hr"] = float(60.0 / np.median(rr))

    # --- median beat: R-aligned, per lead
    keep = [p for p in r if p - BEAT_PRE >= 0 and p + BEAT_POST < N_OUT]
    if len(keep) < 2:
        out["reason"] = "too_few_full_beats"; return out, None, None
    stack = np.stack([x[:, p - BEAT_PRE: p + BEAT_POST] for p in keep], axis=0)  # (nb,12,200)
    beat = np.median(stack, axis=0)                                              # (12,200)
    # isoelectric reference = PR segment just before the beat window's QRS
    beat = beat - np.median(beat[:, :25], axis=1, keepdims=True)
    out["n_beats"] = len(keep)
    out["ok"] = True
    return out, x.astype(np.float32), beat.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()

    # Union of the analysis pairs AND the null-pair control set (stage 7), so the
    # negative control is served from the same cache and the same preprocessing.
    con = duckdb.connect()
    null_src = (f"""
      UNION SELECT study1, path1 FROM read_parquet('{DATA}/null_pairs.parquet')
      UNION SELECT study2, path2 FROM read_parquet('{DATA}/null_pairs.parquet')
    """ if os.path.exists(f"{DATA}/null_pairs.parquet") else "")
    studies = con.sql(f"""
    WITH s AS (
      SELECT study1 AS study_id, path1 AS path FROM read_parquet('{DATA}/pairs_manifest.parquet')
      UNION
      SELECT study2, path2 FROM read_parquet('{DATA}/pairs_manifest.parquet')
      {null_src}
    ) SELECT study_id, path FROM s ORDER BY study_id
    """).fetchdf()
    if a.limit:
        studies = studies.head(a.limit)
    n = len(studies)
    print(f"preprocessing {n:,} unique ECGs -> {WF}", flush=True)

    os.makedirs(WF, exist_ok=True)
    strips = np.lib.format.open_memmap(f"{WF}/strips.npy", mode="w+",
                                       dtype=np.float32, shape=(n, 12, N_OUT))
    beats = np.lib.format.open_memmap(f"{WF}/beats.npy", mode="w+",
                                      dtype=np.float32, shape=(n, 12, BEAT_LEN))

    tasks = [(i, int(r.study_id), r.path) for i, r in enumerate(studies.itertuples())]
    meta = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for m, x, b in ex.map(preprocess_one, tasks, chunksize=64):
            if m["ok"]:
                strips[m["idx"]] = x
                beats[m["idx"]] = b
            meta.append(m)
            done += 1
            if done % 5000 == 0:
                ok = sum(d["ok"] for d in meta)
                print(f"  {done:>7,}/{n:,}  ok={ok:,} ({100*ok/done:.1f}%)", flush=True)

    strips.flush(); beats.flush()
    md = pd.DataFrame(meta).sort_values("idx")
    md.to_parquet(f"{WF}/ecg_meta.parquet", index=False)

    print(f"\n=== QC summary ({len(md):,} ECGs) ===")
    print(f"  usable            : {md.ok.sum():,} ({100*md.ok.mean():.1f}%)")
    print("  rejection reasons :")
    rr = md[~md.ok].reason.value_counts()
    for k, v in rr.items():
        print(f"    {k:<24s} {v:>7,}")
    print(f"\n  heart rate  median={md.hr.median():.0f} bpm  IQR="
          f"[{md.hr.quantile(.25):.0f}, {md.hr.quantile(.75):.0f}]")
    print(f"  beats/record median={md.n_beats.median():.0f}")
    print(f"  records with >=1 bad lead: {(md.n_bad_leads>0).sum():,} "
          f"({100*(md.n_bad_leads>0).mean():.1f}%)")
    print(f"\nwrote {WF}/strips.npy  ({strips.nbytes/1e9:.1f} GB)")
    print(f"wrote {WF}/beats.npy   ({beats.nbytes/1e9:.2f} GB)")
    print(f"wrote {WF}/ecg_meta.parquet")


if __name__ == "__main__":
    main()
