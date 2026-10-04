"""v3/HEEDB stage 5: EXTERNAL validation of the frozen HEEDB ECG-potassium model on MIMIC-IV
(concept.md sec.0.6 design: HEEDB = development, MIMIC-IV = external validation only).

Run ONCE, after data/models/heedb/FROZEN.json exists. Nothing here trains, tunes or picks a
threshold on MIMIC; recalibration is reported as a sensitivity analysis only.

Scores every usable ECG of the MIMIC v3 store (pool + v3 index + prior ECGs) and writes
outputs/heedb/mimic_ecgk_external.parquet (study_id, pred_k) -- the input that
24_eval_v3.py --ecgk takes to re-run the pseudohyperkalaemia analysis externally.

Reports on the MIMIC pool (non-haemolysed K within +-2 h, and within 1 h):
  external   frozen HEEDB model as is
  recalib.   + linear recalibration (fitted and scored on MIMIC; sensitivity only -- AUCs unchanged)
  internal   the MIMIC cross-fitted model (22_train_k_v3.py) on the same ECGs, for reference
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
sys.path.insert(0, f"{ROOT}/scripts")
from _evalutils import auc, r2
spec = importlib.util.spec_from_file_location("tr", f"{ROOT}/scripts/28_train_k_heedb.py")
TR = importlib.util.module_from_spec(spec); spec.loader.exec_module(TR)

FZ = f"{DATA}/models/heedb/FROZEN.json"
assert os.path.exists(FZ), "model not frozen -- refuse to touch MIMIC"
fz = json.load(open(FZ))
print(f"frozen model: {fz['model']} (variant {fz['variant']}, frozen {fz['frozen_at']})")
ck = torch.load(fz["model"], map_location="cpu", weights_only=False)
m = TR.KNet().to("cuda"); m.load_state_dict(ck["state"]); m.eval()

d = pd.read_parquet(f"{DATA}/v3_ecg_k.parquet").merge(
    pd.read_parquet(f"{DATA}/waveforms_v3/meta.parquet")[["idx", "ok", "n_bad_leads"]], on="idx")
d = d[d.ok & (d.n_bad_leads <= 1)].reset_index(drop=True)
TR.STORE = f"{DATA}/waveforms_v3"            # the MIMIC float16 store; same format as HEEDB
d["k_fill"] = d.k.fillna(ck["mu"])
dl = DataLoader(TR.DS(d.assign(k=d.k_fill), ck["scale"], ck["mu"], ck["sd"]), batch_size=512, num_workers=16)
d["pred_k"] = TR.predict(m, dl) * ck["sd"] + ck["mu"]
os.makedirs(f"{OUT}/heedb", exist_ok=True)
d[["study_id", "pred_k"]].to_parquet(f"{OUT}/heedb/mimic_ecgk_external.parquet", index=False)

internal = pd.read_parquet(f"{OUT}/v3/ecgk_oof.parquet")[["study_id", "pred_k"]] \
    .drop_duplicates("study_id").rename(columns={"pred_k": "pred_k_internal"})
p = d[d.in_pool & d.k.notna()].merge(internal, on="study_id", how="left")


def met(y, s):
    return dict(n=len(y), R2=round(r2(y, s), 4), MAE=round(float(np.abs(y - s).mean()), 4),
                AUC_hyperK=round(auc(y >= 5.5, s), 4), AUC_K6=round(auc(y >= 6.0, s), 4),
                AUC_hypoK=round(auc(y < 3.5, -s), 4))


rows = []
for lab, s in (("lab within 2 h", p), ("lab within 1 h", p[p.gap_h <= 1])):
    y, e = s.k.values, s.pred_k.values
    b1, b0 = np.polyfit(e, y, 1)                     # calibration slope / intercept
    rows.append(dict(set=lab, model="EXTERNAL (HEEDB, frozen)", calib_slope=round(b1, 3), calib_intercept=round(b0, 3),
                     mean_pred=round(e.mean(), 3), mean_true=round(y.mean(), 3), **met(y, e)))
    rows.append(dict(set=lab, model="  external + MIMIC recalibration (sensitivity)", **met(y, b0 + b1 * e)))
    si = s[s.pred_k_internal.notna()]
    rows.append(dict(set=lab, model="  internal (MIMIC cross-fitted, reference)", **met(si.k.values, si.pred_k_internal.values)))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n=== EXTERNAL VALIDATION on MIMIC-IV (non-haemolysed K paired with the ECG) ===")
print(R.to_string(index=False))
R.to_csv(f"{OUT}/heedb/external_mimic_accuracy.csv", index=False)
print(f"\nwrote {OUT}/heedb/mimic_ecgk_external.parquet, {OUT}/heedb/external_mimic_accuracy.csv")
