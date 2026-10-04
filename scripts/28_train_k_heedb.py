"""v3/HEEDB stage 4: develop the ECG potassium model on HEEDB I0001 ONLY (concept.md sec.0.6,
design locked before training: HEEDB = development, MIMIC-IV = external validation).

Split: patient-level, fixed by SEED, 80 % train / 10 % validation / 10 % internal test.
Nothing from MIMIC is read here.

Label variants (choice made on HEEDB validation only):
  A  all numeric K paired within +-2 h
  B  A minus hemo_grade_pos and spike_unconfirmed (probable undetected haemolysis, 27_*)
Both are scored on the CLEAN validation subset (B's exclusions), so the comparison is not
tilted by the noisy labels themselves. Selection rule, fixed in advance: higher validation
hyperK (>= 5.5) AUC; ties (< 0.005) broken by validation R^2.

Model: the 1-D ResNet encoder used throughout (09_train.Encoder) + MLP head, MSE on
standardised K, AdamW + OneCycle, checkpoint on validation R^2; the global amplitude scale
comes from HEEDB training ECGs. QC: preprocessing ok and <= 1 bad lead.

Writes data/models/heedb/ecgk_{A,B}.pt, outputs/heedb/val_selection.csv,
outputs/heedb/internal_test.csv, and data/models/heedb/FROZEN.json naming the selected
variant (the file 29_* requires before it may touch MIMIC).
"""
import os, sys, time, json, argparse, importlib.util
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEEDB = os.environ.get("HEEDB_ROOT", "/path/to/heedb")
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
STORE, MD, OH = f"{HEEDB}/derived", f"{DATA}/models/heedb", f"{OUT}/heedb"
sys.path.insert(0, f"{ROOT}/scripts")
from _evalutils import auc, r2
spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
DEV, SEED = "cuda", T.SEED


class DS(Dataset):
    def __init__(self, d, scale, mu, sd):
        self.i = d.idx.values.astype(int)
        self.y = ((d.k.values - mu) / sd).astype(np.float32)
        self.scale, self.X = scale, None
    def __len__(self): return len(self.i)
    def __getitem__(self, j):
        if self.X is None: self.X = np.load(f"{STORE}/strips16.npy", mmap_mode="r")
        return np.asarray(self.X[self.i[j]], dtype=np.float32) / self.scale, self.y[j]


class KNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = T.Encoder(256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(self, x): return self.head(self.enc(x)).squeeze(-1)


@torch.no_grad()
def predict(m, dl):
    m.eval(); out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    return dict(n=len(y), R2=round(r2(y, p), 4), MAE=round(float(np.abs(y - p).mean()), 4),
                AUC_hyperK=round(auc(y >= 5.5, p), 4), AUC_K6=round(auc(y >= 6.0, p), 4),
                AUC_hypoK=round(auc(y < 3.5, -p), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    a = ap.parse_args()
    os.makedirs(MD, exist_ok=True); os.makedirs(OH, exist_ok=True)

    d = pd.read_parquet(f"{DATA}/heedb/heedb_ecg_k_labels.parquet").merge(
        pd.read_parquet(f"{STORE}/meta.parquet")[["idx", "ok", "n_bad_leads"]], on="idx")
    d = d[d.ok & (d.n_bad_leads <= 1)].reset_index(drop=True)
    pats = np.sort(d.pid.unique())
    u = np.random.default_rng(SEED).random(len(pats))
    split = pd.Series(np.where(u < 0.8, "train", np.where(u < 0.9, "val", "test")), index=pats)
    d["split"] = split.reindex(d.pid).values
    d["noisy"] = d.hemo_grade_pos | d.spike_unconfirmed
    print(f"usable pairs {len(d):,}, patients {len(pats):,}; flagged noisy {int(d.noisy.sum()):,} "
          f"({int((d.noisy & (d.k >= 5.5)).sum()):,} of {int((d.k >= 5.5).sum()):,} hyperK)", flush=True)
    print(d.groupby("split").agg(pairs=("idx", "size"), patients=("pid", "nunique"),
                                 hyperK=("k", lambda s: int((s >= 5.5).sum()))).to_string(), flush=True)

    va_clean = d[(d.split == "val") & ~d.noisy]
    te_clean = d[(d.split == "test") & ~d.noisy]
    X = np.load(f"{STORE}/strips16.npy", mmap_mode="r")
    sel = []
    for var in ("A", "B"):
        tr = d[(d.split == "train") & ((var == "A") | ~d.noisy)]
        va = d[(d.split == "val") & ((var == "A") | ~d.noisy)]
        torch.manual_seed(SEED); np.random.seed(SEED)
        mu, sd = tr.k.mean(), tr.k.std()
        scale = float(np.std(np.asarray(X[np.sort(tr.idx.sample(5000, random_state=SEED).values)], dtype=np.float32)))
        mk = lambda dd, sh: DataLoader(DS(dd, scale, mu, sd), batch_size=a.bs, shuffle=sh, num_workers=16,
                                       pin_memory=True, persistent_workers=True, drop_last=sh)
        dtr, dva = mk(tr, True), mk(va, False)
        m = KNet().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(dtr))
        print(f"[{var}] train {len(tr):,} val {len(va):,}  K mu={mu:.2f} sd={sd:.2f} scale={scale:.4f}", flush=True)
        best, best_state, best_ep = -np.inf, None, 0
        for ep in range(a.epochs):
            t0 = time.time(); m.train()
            for x, y in dtr:
                x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.mse_loss(m(x).float(), y)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
            pv = predict(m, dva) * sd + mu
            rv = r2(va.k.values, pv)
            print(f"  [{var}] ep{ep+1:02d} val_R2={rv:.4f} val_AUC_hyperK={auc(va.k.values >= 5.5, pv):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if rv > best:
                best, best_ep = rv, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state)
        torch.save({"state": best_state, "scale": scale, "mu": mu, "sd": sd, "best_ep": best_ep,
                    "variant": var, "site": "HEEDB-I0001"}, f"{MD}/ecgk_{var}.pt")
        pc = predict(m, mk(va_clean, False)) * sd + mu
        sel.append(dict(variant=var, best_ep=best_ep, **metrics(va_clean.k.values, pc)))
        print(f"[{var}] CLEAN-val {sel[-1]}", flush=True)

    S = pd.DataFrame(sel); S.to_csv(f"{OH}/val_selection.csv", index=False)
    a_, b_ = S.set_index("variant").loc["A"], S.set_index("variant").loc["B"]
    if abs(a_.AUC_hyperK - b_.AUC_hyperK) >= 0.005:
        chosen = "A" if a_.AUC_hyperK > b_.AUC_hyperK else "B"
    else:
        chosen = "A" if a_.R2 >= b_.R2 else "B"
    print(f"\n=== validation selection (CLEAN val) ===\n{S.to_string(index=False)}\n-> chosen variant: {chosen}")

    # internal test, once, with the chosen model
    ck = torch.load(f"{MD}/ecgk_{chosen}.pt", map_location="cpu", weights_only=False)
    m = KNet().to(DEV); m.load_state_dict(ck["state"])
    dl = DataLoader(DS(te_clean, ck["scale"], ck["mu"], ck["sd"]), batch_size=a.bs, num_workers=16)
    pt = predict(m, dl) * ck["sd"] + ck["mu"]
    R = pd.DataFrame([dict(set="HEEDB internal test (clean labels)", **metrics(te_clean.k.values, pt)),
                      dict(set="  ... lab within 1 h", **metrics(te_clean.k.values[te_clean.gap_h.values <= 1],
                                                                pt[te_clean.gap_h.values <= 1]))])
    R.to_csv(f"{OH}/internal_test.csv", index=False)
    print(f"\n=== HEEDB internal test ===\n{R.to_string(index=False)}")
    json.dump({"variant": chosen, "model": f"{MD}/ecgk_{chosen}.pt", "frozen_at": time.strftime("%Y-%m-%d %H:%M"),
               "note": "selected on HEEDB validation only; MIMIC not read"}, open(f"{MD}/FROZEN.json", "w"), indent=1)
    print(f"froze {MD}/FROZEN.json")


if __name__ == "__main__":
    main()
