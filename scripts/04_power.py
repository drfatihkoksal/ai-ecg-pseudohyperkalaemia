"""Stage 4: is the anchor test set big enough to say anything?

The pair counts look reassuring, but pairs are NESTED IN PATIENTS (anchor test:
1,281 pairs from only 506 patients, ~2.5 pairs/patient) and the AKI label is
strongly correlated within a patient -- a patient who is in AKI at ECG-pair k is
usually still in AKI at pair k+1. So the effective sample size is much closer to
the patient count than to the pair count, and any CI computed at pair level is
anticonservative.

This script does not assume a design effect; it MEASURES the intra-patient
correlation of the label in the real manifest, then cluster-bootstraps (resampling
PATIENTS, not pairs) to get honest CI widths for a range of plausible true AUCs.
It then compares the candidate evaluation designs so the choice is made on
numbers rather than taste.
"""
import os
import duckdb, numpy as np, pandas as pd

_ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "outputs")
rng = np.random.default_rng(20260713)

con = duckdb.connect()
m = con.sql(f"""
SELECT subject_id, hadm_id, aki, has_next_cr, aki_next_day, anchor_split,
       is_acs, is_pci, is_cath, dcr, dk
FROM read_parquet('{DATA}/pairs_manifest.parquet')
""").fetchdf()

anchor = m[m.is_acs & m.is_pci].copy()

# ---------------------------------------------------------------- 1. what we actually have
print("=== event counts: PAIRS vs PATIENTS (the number that matters) ===")
rows = []
for split in ["train", "val", "test"]:
    s = anchor[anchor.anchor_split == split]
    pos_pat = s.groupby("subject_id").aki.max()           # patient is a case if ANY pair is AKI
    lead = s[s.has_next_cr]
    lead_pos_pat = lead.groupby("subject_id").aki_next_day.max()
    rows.append(dict(
        split=split,
        pairs=len(s), patients=s.subject_id.nunique(),
        aki_pairs=int(s.aki.sum()), aki_patients=int(pos_pat.sum()),
        lead_pairs=len(lead), lead_patients=lead.subject_id.nunique(),
        lead_aki_pairs=int(lead.aki_next_day.sum()),
        lead_aki_patients=int(lead_pos_pat.sum()),
    ))
tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))

# ---------------------------------------------------------------- 2. measure the clustering
def icc_binary(df, col):
    """ANOVA-style ICC of a binary label across patient clusters."""
    g = df.groupby("subject_id")[col]
    ni = g.size().values
    yi = g.mean().values
    ybar = df[col].mean()
    k = len(ni)
    if k < 2: return np.nan
    msb = np.sum(ni * (yi - ybar) ** 2) / (k - 1)
    within = df[col].values - df.subject_id.map(g.mean()).values
    msw = np.sum(within ** 2) / max(len(df) - k, 1)
    n0 = (np.sum(ni) - np.sum(ni ** 2) / np.sum(ni)) / (k - 1)
    return max((msb - msw) / (msb + (n0 - 1) * msw), 0.0)

test = anchor[anchor.anchor_split == "test"]
icc = icc_binary(test, "aki")
mbar = len(test) / test.subject_id.nunique()
deff = 1 + (mbar - 1) * icc
print(f"\n=== clustering in the anchor TEST set ===")
print(f"  pairs/patient (mean)      : {mbar:.2f}")
print(f"  intra-patient ICC of AKI  : {icc:.3f}")
print(f"  design effect             : {deff:.2f}")
print(f"  effective n (pairs / DE)  : {len(test)/deff:.0f}  (vs {len(test)} nominal pairs)")

# ---------------------------------------------------------------- 3. cluster bootstrap AUC CI
def simulate_scores(labels, auc_target, rng):
    """Binormal scores whose expected AUC == auc_target."""
    d = np.sqrt(2) * _probit(auc_target)
    s = rng.normal(0, 1, len(labels))
    return s + d * labels

def _probit(p):
    from math import erf
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1 + erf(mid / np.sqrt(2)))
        if cdf < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def cluster_boot_ci(df, ycol, auc_target, n_boot=400, rng=rng):
    """Resample PATIENTS with replacement; recompute AUC each time."""
    y = df[ycol].values.astype(float)
    s = simulate_scores(y, auc_target, rng)
    df = df.assign(_y=y, _s=s)
    groups = [g for _, g in df.groupby("subject_id")]
    k = len(groups)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        b = pd.concat([groups[i] for i in idx])
        a = auc(b._y.values, b._s.values)
        if not np.isnan(a): out.append(a)
    out = np.array(out)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

print("\n=== 95% CI half-width for AUC, cluster-bootstrapped by patient ===")
print("(simulated scores at each true AUC; real label + real clustering)\n")

designs = {
    "anchor test only":        anchor[anchor.anchor_split == "test"],
    "anchor val + test":       anchor[anchor.anchor_split.isin(["val", "test"])],
    "anchor ALL (5-fold CV)":  anchor,
    "T5 test (ACS n cath/PCI)": None,   # filled below
}
t5 = m[m.is_acs & (m.is_pci | m.is_cath)].copy()
# apply the same hash split rule to T5 patients for a like-for-like test share
t5_test_subj = con.sql(f"""
SELECT DISTINCT subject_id FROM read_parquet('{DATA}/pairs_manifest.parquet')
WHERE is_acs AND (is_pci OR is_cath) AND hash(subject_id + 20260713) % 100 >= 80
""").fetchdf().subject_id
designs["T5 test (ACS n cath/PCI)"] = t5[t5.subject_id.isin(t5_test_subj)]

res = []
for name, df in designs.items():
    for target in [0.70, 0.75, 0.80]:
        lo, hi = cluster_boot_ci(df, "aki", target)
        res.append(dict(design=name, pairs=len(df), patients=df.subject_id.nunique(),
                        aki_pairs=int(df.aki.sum()), true_auc=target,
                        ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                        half_width=round((hi - lo) / 2, 3)))
r = pd.DataFrame(res)
print(r.to_string(index=False))

# ---------------------------------------------------------------- 4. the lead-hypothesis arm
print("\n=== lead hypothesis (sec.4): next-day AKI, the scarcest arm ===")
lead_designs = {
    "anchor test only":  anchor[(anchor.anchor_split == "test") & anchor.has_next_cr],
    "anchor val + test": anchor[anchor.anchor_split.isin(["val", "test"]) & anchor.has_next_cr],
    "anchor ALL (CV)":   anchor[anchor.has_next_cr],
    "T5 ALL (CV)":       t5[t5.has_next_cr],
}
res = []
for name, df in lead_designs.items():
    lo, hi = cluster_boot_ci(df, "aki_next_day", 0.70)
    res.append(dict(design=name, pairs=len(df), patients=df.subject_id.nunique(),
                    events=int(df.aki_next_day.sum()),
                    ci_lo=round(lo, 3), ci_hi=round(hi, 3), half_width=round((hi - lo) / 2, 3)))
print(pd.DataFrame(res).to_string(index=False))

# ---------------------------------------------------------------- 5. regression arm
print("\n=== regression arm (dCr): precision on Pearson r, patient-clustered ===")
for name, df in [("anchor test", anchor[anchor.anchor_split == "test"]),
                 ("anchor ALL (CV)", anchor)]:
    k = df.subject_id.nunique()
    # CI half-width for r at r=0.3, accounting for the design effect
    icc_d = icc_binary(df.assign(b=(df.dcr > df.dcr.median()).astype(int)), "b")
    mb = len(df) / k
    de = 1 + (mb - 1) * icc_d
    n_eff = len(df) / de
    z_se = 1 / np.sqrt(max(n_eff - 3, 1))
    r0 = 0.30
    z0 = np.arctanh(r0)
    lo, hi = np.tanh(z0 - 1.96 * z_se), np.tanh(z0 + 1.96 * z_se)
    print(f"  {name:16s} pairs={len(df):>6,} patients={k:>5,} n_eff={n_eff:>7.0f}  "
          f"r=0.30 -> 95% CI [{lo:.3f}, {hi:.3f}]  (+-{(hi-lo)/2:.3f})")

r.to_csv(f"{OUT}/power_analysis.csv", index=False)
print(f"\nwrote {OUT}/power_analysis.csv")
