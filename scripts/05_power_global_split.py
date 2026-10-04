"""Stage 4b: cost the proposed fix before committing to it.

Stage 4 said: the AKI-classification arm is fine on a single held-out test set,
but the early-warning arm (sec.4) is not -- 74 case-patients, +-0.047 on AUC.

Two things follow, and this script checks both:

(A) concept.md sec.3 says "ACS patients undergoing coronary angiography / PCI".
    That is ACS n (cath OR PCI) = our T5 -- NOT the ACS n PCI (T4) cohort the
    splits were anchored on. T4 is narrower than the concept actually asks for.

(B) the early-warning claim is a PHYSIOLOGICAL claim (does dECG lead dCr?), not a
    contrast claim. It does not need the trigger cohort at all -- it can be read
    off the full all-admission test set, where events are ~10x more plentiful.
    The trigger cohort is only needed for the CA-AKI natural-experiment framing.

So the proposed structure is ONE patient-level global split (train/val/test) with
cohort flags on top, rather than an anchor-only split:
    - pretrain            = global train
    - anchor analysis     = T5 n global test   (trigger-anchored, honest CI)
    - early-warning arm   = global test        (precise, general claim)
                            + T5 n global test (trigger-anchored, secondary)
This script reports what each arm would then have.
"""
import os
import duckdb, numpy as np, pandas as pd

_ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "outputs")
SEED = 20260713
rng = np.random.default_rng(SEED)

con = duckdb.connect()
m = con.sql(f"""
SELECT *, CASE WHEN hash(subject_id + {SEED} + 99) % 100 < 70 THEN 'train'
               WHEN hash(subject_id + {SEED} + 99) % 100 < 85 THEN 'val'
               ELSE 'test' END AS global_split
FROM read_parquet('{DATA}/pairs_manifest.parquet')
""").fetchdf()
m["is_t5"] = m.is_acs & (m.is_pci | m.is_cath)

def _probit(p):
    from math import erf
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / np.sqrt(2))) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def auc(y, s):
    y = np.asarray(y); npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def cluster_boot(df, ycol, target, n_boot=300):
    # `aki` is NA where the pair has no Cr before t2 to form a baseline from --
    # those pairs are unlabelable and must be dropped, not silently coerced.
    df = df[df[ycol].notna()]
    y = df[ycol].values.astype(float)
    s = rng.normal(0, 1, len(y)) + np.sqrt(2) * _probit(target) * y
    d = df.assign(_y=y, _s=s)
    groups = [g for _, g in d.groupby("subject_id")]
    k = len(groups)
    out = []
    for _ in range(n_boot):
        b = pd.concat([groups[i] for i in rng.integers(0, k, k)])
        a = auc(b._y.values, b._s.values)
        if not np.isnan(a): out.append(a)
    out = np.array(out)
    return (np.percentile(out, 97.5) - np.percentile(out, 2.5)) / 2

test = m[m.global_split == "test"]
arms = {
  "AKI class.  | global test (all adm)":      (test, "aki"),
  "AKI class.  | T5 n global test (anchor)":  (test[test.is_t5], "aki"),
  "Early-warn  | global test (all adm)":      (test[test.has_next_cr], "aki_next_day"),
  "Early-warn  | T5 n global test (anchor)":  (test[test.is_t5 & test.has_next_cr], "aki_next_day"),
}

n_unlabelable = int(m.aki.isna().sum())
print(f"NOTE: {n_unlabelable:,} of {len(m):,} pairs ({100*n_unlabelable/len(m):.1f}%) are UNLABELABLE "
      f"(no creatinine before t2 to build a baseline from) and are dropped from every arm.\n")

print("=== proposed GLOBAL split (70/15/15 by patient), evaluated at true AUC 0.75 ===\n")
rows = []
for name, (df, ycol) in arms.items():
    d = df[df[ycol].notna()]
    hw = cluster_boot(d, ycol, 0.75)
    rows.append(dict(arm=name, pairs=len(d), patients=d.subject_id.nunique(),
                     events=int(d[ycol].sum()),
                     case_patients=int(d.groupby("subject_id")[ycol].max().sum()),
                     auc_half_width=round(hw, 3)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== pretrain pool under the global split ===")
tr = m[m.global_split == "train"]
va = m[m.global_split == "val"]
print(f"  pretrain train : {len(tr):>7,} pairs / {tr.subject_id.nunique():>6,} patients  "
      f"(AKI {100*tr.aki.mean():.1f}%)")
print(f"  pretrain val   : {len(va):>7,} pairs / {va.subject_id.nunique():>6,} patients")
print(f"  T5 train pairs available for fine-tuning: {len(tr[tr.is_t5]):,} "
      f"({tr[tr.is_t5].subject_id.nunique():,} patients)")

pd.DataFrame(rows).to_csv(f"{OUT}/power_global_split.csv", index=False)
print(f"\nwrote {OUT}/power_global_split.csv")
