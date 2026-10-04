"""Stage 13: did the nuisance-reduction variants reduce nuisance -- without losing signal?

Variants (run_nuisance.sh, pre-registered split, lambda=1, crop=2000 fixed in advance):
  _reg      null-pair regulariser: predicted change on TRAIN-split null pairs -> 0
  _tta      random 8 s crops in training, 5-window averaging at test
  _reg_tta  both
Baselines: the v2 models (siamese, emb_diff, static_ecg2).

NUISANCE metric (primary for this step): on TEST-split null pairs (same patient,
0.5-6 h apart; patients disjoint from every training null pair), the spread of the
predicted dK relative to its spread on TEST real pairs:
        null/real = SD(pred dK | null) / SD(pred dK | real)
A global shrink of the output does not change it. CI: patient-cluster bootstrap,
null and real sets resampled independently (they are different patients' pairs).

SIGNAL metrics (must not degrade): dK R^2 over floor_K and new-hyperK AUC over floor_K
(combiner fit on VAL, scored on TEST, as in 13_evaluate_v2.py); paired delta of each
variant vs its own baseline; head-to-head vs static_ecg2; swap antisymmetry.
"""
import os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
from _evalutils import ROOT, DATA, OUT, auc, r2, boot_delta, fit_eval, clusters

spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

FLOOR_K = ["cr", "k", "age", "sex", "dt_h", "dhr"]
KEY = ["study1", "study2"]
VARIANTS = {  # name -> baseline it is compared against
    "siamese": None, "siamese_reg": "siamese", "siamese_tta": "siamese", "siamese_reg_tta": "siamese",
    "emb_diff": None, "emb_diff_reg": "emb_diff", "emb_diff_tta": "emb_diff", "emb_diff_reg_tta": "emb_diff",
    "static_ecg2": None, "static_ecg2_tta": "static_ecg2",
}
NB = 1000

df = pd.read_parquet(f"{DATA}/pairs_v2.parquet")
df["sex"] = (df.gender == "F").astype(int)
va, te = (df[df.split == s].reset_index(drop=True) for s in ("val", "test"))
nul = pd.read_parquet(f"{DATA}/null_pairs_v2.parquet")
nul = nul[nul.split == "test"].reset_index(drop=True)
abl = pd.read_parquet(f"{DATA}/ablation_index.parquet")
abl = abl[abl.split == "test"].reset_index(drop=True)
print(f"test: {len(te):,} real pairs, {len(nul):,} null pairs ({nul.subject_id.nunique():,} patients)")

hv, ht = va.hyperk_new.notna().values, te.hyperk_new.notna().values
Fv = va[FLOOR_K].astype(float).fillna(va[FLOOR_K].median())
Ft = te[FLOOR_K].astype(float).fillna(va[FLOOR_K].median())
floor_dk = fit_eval(Fv.values, va.dk, Ft.values, "r2")
floor_hk = fit_eval(Fv.values[hv], va.hyperk_new[hv], Ft.values[ht], "auc")


def run_model(name, d, i1="wf_idx1", i2="wf_idx2"):
    ck = torch.load(f"{DATA}/models/{name}.pt", map_location="cuda", weights_only=False)
    m = T.Net(ck["arm"], crop=ck.get("crop", 0)).to("cuda"); m.load_state_dict(ck["state"]); m.eval()
    d = d.copy()
    for t in T.TARGETS: d[f"{t}_z"] = 0.0
    d["aki_incident"] = np.nan
    dl = DataLoader(T.PairDS(d, ck["arm"], ck["scale"], ck["arm"] == "beat_sub", idx1=i1, idx2=i2),
                    batch_size=256, shuffle=False, num_workers=8)
    reg, _ = T.predict(m, dl)
    return reg[:, 1] * ck["sd"]["dk"] + ck["mu"]["dk"], ck


def ratio_ci(pn, sn, pr, sr, seed=T.SEED):
    gn, gr, rng, out = clusters(sn), clusters(sr), np.random.default_rng(seed), []
    for _ in range(NB):
        a = np.concatenate([gn[i] for i in rng.integers(0, len(gn), len(gn))])
        b = np.concatenate([gr[i] for i in rng.integers(0, len(gr), len(gr))])
        out.append(pn[a].std() / pr[b].std())
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


S, P = [], {}
for name, base in VARIANTS.items():
    if not os.path.exists(f"{DATA}/models/{name}.pt"):
        print(f"[skip] {name}: not trained"); continue
    print(f"... {name}", flush=True)
    p = pd.read_parquet(f"{OUT}/preds_{name}.parquet")
    pv = va[KEY].merge(p, on=KEY, how="left").pred_dk.values
    pt = te[KEY].merge(p, on=KEY, how="left").pred_dk.values
    assert not (np.isnan(pv).any() or np.isnan(pt).any()), f"{name}: stale predictions"
    pn, ck = run_model(name, nul)
    ratio = pn.std() / pt.std()
    lo, hi = ratio_ci(pn, nul.subject_id.values, pt, te.subject_id.values)

    s_dk = fit_eval(np.column_stack([Fv, pv]), va.dk, np.column_stack([Ft, pt]), "r2")
    s_hk = fit_eval(np.column_stack([Fv, pv])[hv], va.hyperk_new[hv],
                    np.column_stack([Ft, pt])[ht], "auc")
    P[name] = dict(dk=s_dk, hk=s_hk)

    p0, _ = run_model(name, abl)
    psw, _ = run_model(name, abl, "swap_wf_idx1", "swap_wf_idx2")
    mu = ck["mu"]["dk"]
    S.append(dict(variant=name, best_ep=ck.get("best_ep"),
                  null_real_ratio=round(ratio, 3), ratio_ci=f"[{lo:.3f}, {hi:.3f}]",
                  null_pred_sd=round(pn.std(), 3), real_pred_sd=round(pt.std(), 3),
                  r_dk_real=round(np.corrcoef(pt, te.dk)[0, 1], 3),
                  dk_R2_gain=round(r2(te.dk.values, s_dk) - r2(te.dk.values, floor_dk), 4),
                  hk_AUC_gain=round(auc(te.hyperk_new.values[ht], s_hk) -
                                    auc(te.hyperk_new.values[ht], floor_hk), 4),
                  antisym_B3=round(np.corrcoef(psw - mu, -(p0 - mu))[0, 1], 3)))
S = pd.DataFrame(S)

# paired signal contrasts: variant vs its own baseline, and vs static_ecg2
C = []
y_dk, y_hk = te.dk.values, te.hyperk_new.values[ht]
sid, sid_hk = te.subject_id.values, te.subject_id.values[ht]
for name, base in VARIANTS.items():
    if name not in P: continue
    refs = ([("vs own baseline", base)] if base in P else []) + \
           ([("vs static_ecg2", "static_ecg2")] if name.split("_")[0] != "static" and "static_ecg2" in P else [])
    for lab, ref in refs:
        for tgt, y, s, metric in (("dK R2", y_dk, sid, "r2"), ("new hyperK AUC", y_hk, sid_hk, "auc")):
            k = "dk" if tgt.startswith("dK") else "hk"
            d, lo, hi, pg = boot_delta(s, y, P[ref][k], P[name][k], metric, n_boot=NB)
            C.append(dict(variant=name, contrast=f"{lab} ({ref})", target=tgt,
                          delta=round(d, 4), ci=f"[{lo:+.4f}, {hi:+.4f}]", p_gt0=round(pg, 3)))
C = pd.DataFrame(C)

pd.set_option("display.width", 220)
print("\n=== NUISANCE + SIGNAL SUMMARY (test; null pairs patient-disjoint from all training) ===")
print("    null_real_ratio: lower = less output on pairs where nothing changed")
print(S.to_string(index=False))
print("\n=== PAIRED SIGNAL CONTRASTS (combiner on floor_K, fit on val; patient bootstrap) ===")
print(C.to_string(index=False))
S.to_csv(f"{OUT}/nuisance_summary.csv", index=False)
C.to_csv(f"{OUT}/nuisance_contrasts.csv", index=False)
print(f"\nwrote {OUT}/nuisance_summary.csv, nuisance_contrasts.csv")
