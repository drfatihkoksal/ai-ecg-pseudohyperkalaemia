"""v3 stage 3: absolute-potassium ECG model, patient-level 5-fold cross-fitting.

Why cross-fitting and not "exclude the analysis patients": every patient with a
hyperkalaemic ECG is in the v3 index cohort, so excluding them leaves a training set
with ZERO hyperkalaemia. Instead, all ECGs (pool + index + prior) are split into 5
patient-level folds; the model for fold f is trained on the other folds' POOL ECGs and
predicts every ECG of fold f. Each index / prior ECG therefore gets a prediction from a
model that never saw its patient.

Labels: nearest NON-haemolysed serum K within +-2 h (21_preprocess_v3.py). Haemolysed
values never enter training (they were ~3 % of v1/v2 K labels).
Model: the v1/v2 1-D ResNet encoder (09_train.Encoder) + MLP head, MSE on standardised K,
AdamW + OneCycle, checkpoint on val R^2 (val = 10 % of the training patients).
QC: ECG preprocessing ok and <= 1 bad lead.

Writes outputs/v3/ecgk_oof.parquet (study_id, subject_id, fold, k, gap_h, role flags,
pred_k) and prints out-of-fold accuracy on pool ECGs, to be set against the literature
(hyperK AUC 0.85-0.93 with lab within 1-4 h).
"""
import os, sys, time, argparse, importlib.util
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
WF3, O3 = f"{DATA}/waveforms_v3", f"{OUT}/v3"
sys.path.insert(0, f"{ROOT}/scripts")
from _evalutils import auc, r2
spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
DEV, SEED, NF = "cuda", T.SEED, 5


class DS(Dataset):
    def __init__(self, d, scale, mu, sd):
        self.i = d.idx.values.astype(int)
        self.y = ((d.k.fillna(mu).values - mu) / sd).astype(np.float32)
        self.scale, self.X = scale, None
    def __len__(self): return len(self.i)
    def __getitem__(self, j):
        if self.X is None: self.X = np.load(f"{WF3}/strips16.npy", mmap_mode="r")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    os.makedirs(O3, exist_ok=True); os.makedirs(f"{DATA}/models/v3", exist_ok=True)

    d = pd.read_parquet(f"{DATA}/v3_ecg_k.parquet").merge(
        pd.read_parquet(f"{WF3}/meta.parquet")[["idx", "ok", "n_bad_leads"]], on="idx")
    d = d[d.ok & (d.n_bad_leads <= 1)].reset_index(drop=True)
    d["fold"] = T.cv_folds(d.subject_id, NF)
    print(f"usable ECGs {len(d):,} (pool {int(d.in_pool.sum()):,}; patients {d.subject_id.nunique():,})", flush=True)

    for f in range(NF):
        pf = f"{O3}/ecgk_oof_f{f}.parquet"
        if os.path.exists(pf):
            print(f"[skip] fold {f}"); continue
        torch.manual_seed(SEED + f); np.random.seed(SEED + f)
        tr_all = d[(d.fold != f) & d.in_pool]
        pats = np.sort(tr_all.subject_id.unique())
        va_p = set(np.random.default_rng(SEED + f).choice(pats, size=len(pats) // 10, replace=False))
        va, tr = tr_all[tr_all.subject_id.isin(va_p)], tr_all[~tr_all.subject_id.isin(va_p)]
        te = d[d.fold == f]
        mu, sd = tr.k.mean(), tr.k.std()
        X = np.load(f"{WF3}/strips16.npy", mmap_mode="r")
        scale = float(np.std(np.asarray(X[np.sort(tr.idx.values[:3000])], dtype=np.float32)))
        mk = lambda dd, sh: DataLoader(DS(dd, scale, mu, sd), batch_size=a.bs, shuffle=sh, num_workers=12,
                                       pin_memory=True, persistent_workers=True, drop_last=sh)
        dtr, dva, dte = mk(tr, True), mk(va, False), mk(te, False)
        m = KNet().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(dtr))
        print(f"[fold {f}] train {len(tr):,} val {len(va):,} test {len(te):,}  K mu={mu:.2f} sd={sd:.2f} "
              f"scale={scale:.4f}", flush=True)
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
            print(f"  [fold {f}] ep{ep+1:02d} val_R2(K)={rv:.4f} val_AUC(K>=5.5)={auc(va.k.values >= 5.5, pv):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if rv > best:
                best, best_ep = rv, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state)
        torch.save({"state": best_state, "scale": scale, "mu": mu, "sd": sd, "best_ep": best_ep},
                   f"{DATA}/models/v3/ecgk_f{f}.pt")
        out = te[["idx", "study_id", "subject_id", "fold", "k", "gap_h", "in_pool", "is_index", "is_prior"]].copy()
        out["pred_k"] = predict(m, dte) * sd + mu
        out.to_parquet(pf, index=False)
        print(f"[fold {f}] best ep {best_ep} val R2 {best:.4f} -> wrote {pf}", flush=True)

    o = pd.concat([pd.read_parquet(f"{O3}/ecgk_oof_f{f}.parquet") for f in range(NF)], ignore_index=True)
    o.to_parquet(f"{O3}/ecgk_oof.parquet", index=False)
    p = o[o.in_pool & o.k.notna()]
    print("\n=== out-of-fold accuracy, pool ECGs (non-haemolysed K) ===")
    for lab, s in (("lab within 2 h", p), ("lab within 1 h", p[p.gap_h <= 1])):
        y, pr = s.k.values, s.pred_k.values
        print(f"  {lab:<15s} n={len(s):>7,}  R2={r2(y, pr):.3f}  MAE={np.abs(y-pr).mean():.3f}  "
              f"AUC hyperK(>=5.5)={auc(y >= 5.5, pr):.3f}  AUC K>=6.0={auc(y >= 6.0, pr):.3f}  "
              f"AUC hypoK(<3.5)={auc(y < 3.5, -pr):.3f}")
    print(f"wrote {O3}/ecgk_oof.parquet")


if __name__ == "__main__":
    main()
