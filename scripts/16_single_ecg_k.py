"""Stage 14: does a dK-trained encoder read ABSOLUTE potassium from a single ECG better
than a model trained to do exactly that?

Single-ECG dataset: every ECG of the v2 pair table with its matched potassium
(139,435 ECGs, 31,737 patients), same patient-level split. All models below see the
same ECGs of the same TRAIN patients; only the objective differs.

  scratch      standard model: same 1-D ResNet encoder + head, trained end-to-end on
               absolute K (MSE), checkpoint on val R^2. The reference.
  ft_siamese   same training, encoder INITIALISED from the dK-trained siamese.
  probe_*      FROZEN encoder -> 256-d embedding -> ridge (alpha chosen on val):
                 probe_siamese, probe_emb_diff   dK-trained difference encoders
                 probe_static_ecg2               trained on dK from ECG2 alone
                 probe_scratch                   the standard model's own encoder
                                                 (probe-vs-probe parity)
                 probe_random                    untrained encoder (floor)

Metrics on TEST: R^2, MAE, hyperK (K >= 5.5) AUC, hypoK (K < 3.5) AUC, and the split
that matters for an identity-cancelling objective:
  within-patient r   corr of patient-demeaned prediction vs patient-demeaned K
                     (patients with >= 2 test ECGs) -- tracking change
  between-patient r  corr of patient-mean prediction vs patient-mean K -- level
Contrasts vs `scratch`: patient-cluster bootstrap of the difference.
"""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
from _evalutils import ROOT, DATA, OUT, auc, r2

spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/09_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
WF, DEV, SEED, EPOCHS, NB = f"{DATA}/waveforms", "cuda", T.SEED, 30, 1000
torch.manual_seed(SEED); np.random.seed(SEED)
ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, default=None,
                help="5-fold CV (same patient folds as 09_train.py --fold): fold k = TEST, k+1 = VAL. "
                     "Source encoders come from models/cv/<arm>_f<k>.pt, which never saw fold k.")
ap.add_argument("--nfolds", type=int, default=5)
A = ap.parse_args()
SRC = (lambda arm: f"{DATA}/models/{arm}.pt") if A.fold is None else \
      (lambda arm: f"{DATA}/models/cv/{arm}_f{A.fold}.pt")
SUF = "" if A.fold is None else f"_f{A.fold}"

# ------------------------------------------------------------------ single-ECG table
p = pd.read_parquet(f"{DATA}/pairs_v2.parquet")
e = pd.concat([p[["study1", "wf_idx1", "k", "subject_id", "split"]]
                 .set_axis(["study", "wf", "k", "subject_id", "split"], axis=1),
               p[["study2", "wf_idx2", "k2", "subject_id", "split"]]
                 .set_axis(["study", "wf", "k", "subject_id", "split"], axis=1)])
e = e.drop_duplicates("study").reset_index(drop=True)
if A.fold is not None:
    # cv_folds permutes the sorted unique patients, and e has exactly the patients of
    # pairs_v2, so this reproduces the pair-level CV folds patient for patient.
    fold = T.cv_folds(e.subject_id, A.nfolds)
    e["split"] = np.where(fold == A.fold, "test",
                 np.where(fold == (A.fold + 1) % A.nfolds, "val", "train"))
    e["fold"] = fold
S = {s: e[e.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
mu_k, sd_k = S["train"].k.mean(), S["train"].k.std()
scale = torch.load(SRC("siamese"), map_location="cpu", weights_only=False)["scale"]
print(f"single ECGs: " + ", ".join(f"{s}={len(d):,}" for s, d in S.items()) +
      f"   K mean={mu_k:.2f} sd={sd_k:.2f}   amplitude scale={scale:.4f}", flush=True)


class SingleDS(Dataset):
    def __init__(self, d):
        self.i, self.y = d.wf.values.astype(int), ((d.k.values - mu_k) / sd_k).astype(np.float32)
        self.X = None
    def __len__(self): return len(self.i)
    def __getitem__(self, j):
        if self.X is None: self.X = np.load(f"{WF}/strips.npy", mmap_mode="r")
        return np.asarray(self.X[self.i[j]], dtype=np.float32) / scale, self.y[j]


def loader(d, sh): return DataLoader(SingleDS(d), batch_size=256, shuffle=sh, num_workers=10,
                                     pin_memory=True, persistent_workers=True, drop_last=sh)
DL = {s: loader(d, s == "train") for s, d in S.items()}
DLe = {s: loader(d, False) for s, d in S.items()}      # unshuffled, for embeddings


class KNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = T.Encoder(256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(self, x): return self.head(self.enc(x)).squeeze(-1)


@torch.no_grad()
def run(fn, dl):
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(fn(x.to(DEV)).float().cpu().numpy())
    return np.concatenate(out)


def train_k(name, init_from=None):
    m = KNet().to(DEV)
    if init_from:
        st = torch.load(SRC(init_from), map_location="cpu", weights_only=False)["state"]
        m.enc.load_state_dict({k[4:]: v for k, v in st.items() if k.startswith("enc.")})
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, epochs=EPOCHS, steps_per_epoch=len(DL["train"]))
    yv, best, best_state = S["val"].k.values, -np.inf, None
    for ep in range(EPOCHS):
        t0 = time.time(); m.train()
        for x, y in DL["train"]:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(m(x).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
        m.eval()
        rv = r2(yv, run(m, DLe["val"]) * sd_k + mu_k)
        print(f"  [{name}] ep{ep+1:02d} val_R2(K)={rv:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if rv > best:
            best, best_state = rv, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    os.makedirs(f"{DATA}/models/single_k", exist_ok=True)
    torch.save({"state": best_state, "scale": scale, "mu_k": mu_k, "sd_k": sd_k},
               f"{DATA}/models/single_k/{name}{SUF}.pt")
    return m


def probe(enc):
    enc.eval()
    Z = {s: run(enc, DLe[s]) for s in S}
    sc = StandardScaler().fit(Z["train"])
    Z = {s: sc.transform(z) for s, z in Z.items()}
    best = max((r2(S["val"].k.values, Ridge(alpha=al).fit(Z["train"], S["train"].k).predict(Z["val"])), al)
               for al in (0.1, 1, 10, 100, 1000, 10000))
    return Ridge(alpha=best[1]).fit(Z["train"], S["train"].k).predict(Z["test"]), best[1]


# ------------------------------------------------------------------ models
PRED = {}
m_s = train_k("scratch")
PRED["scratch"] = run(m_s, DLe["test"]) * sd_k + mu_k
m_f = train_k("ft_siamese", init_from="siamese")
PRED["ft_siamese"] = run(m_f, DLe["test"]) * sd_k + mu_k

ALPHA = {}
for src in ("siamese", "emb_diff", "static_ecg2"):
    st = torch.load(SRC(src), map_location="cpu", weights_only=False)["state"]
    enc = T.Encoder(256).to(DEV); enc.load_state_dict({k[4:]: v for k, v in st.items() if k.startswith("enc.")})
    PRED[f"probe_{src}"], ALPHA[f"probe_{src}"] = probe(enc)
PRED["probe_scratch"], ALPHA["probe_scratch"] = probe(m_s.enc)
torch.manual_seed(SEED)
PRED["probe_random"], ALPHA["probe_random"] = probe(T.Encoder(256).to(DEV))

# ------------------------------------------------------------------ metrics
te = S["test"]
y, sid = te.k.values, te.subject_id.values
multi = te.groupby("subject_id").k.transform("size").values >= 2


def metrics(pr):
    d = te.assign(p=pr)
    dm = d[multi]
    w = np.corrcoef(dm.p - dm.groupby("subject_id").p.transform("mean"),
                    dm.k - dm.groupby("subject_id").k.transform("mean"))[0, 1]
    g = d.groupby("subject_id")[["p", "k"]].mean()
    return dict(R2=r2(y, pr), MAE=np.abs(y - pr).mean(), AUC_hyperK=auc(y >= 5.5, pr),
                AUC_hypoK=auc(y < 3.5, -pr), r_within=w, r_between=np.corrcoef(g.p, g.k)[0, 1])


# vectorised patient-cluster bootstrap of metric(b) - metric(a)
order = np.argsort(sid, kind="stable")
starts = np.flatnonzero(np.r_[True, sid[order][1:] != sid[order][:-1]])
lens = np.diff(np.r_[starts, len(sid)])
def boot(a, b, fn):
    rng, out = np.random.default_rng(SEED), []
    ya, aa, bb = y[order], a[order], b[order]
    for _ in range(NB):
        c = rng.integers(0, len(starts), len(starts)); L = lens[c]
        ix = np.repeat(starts[c] - np.r_[0, np.cumsum(L)[:-1]], L) + np.arange(L.sum())
        out.append(fn(ya[ix], bb[ix]) - fn(ya[ix], aa[ix]))
    out = np.array(out)
    return np.mean(out), np.percentile(out, 2.5), np.percentile(out, 97.5)


M = pd.DataFrame({k: metrics(v) for k, v in PRED.items()}).T.round(4)
M["ridge_alpha"] = pd.Series(ALPHA)
C = []
for k in PRED:
    if k == "scratch": continue
    for lab, fn in (("R2", lambda yy, pp: r2(yy, pp)), ("AUC_hyperK", lambda yy, pp: auc(yy >= 5.5, pp))):
        d, lo, hi = boot(PRED["scratch"], PRED[k], fn)
        C.append(dict(model=k, metric=lab, delta_vs_scratch=round(d, 4), ci=f"[{lo:+.4f}, {hi:+.4f}]"))
C = pd.DataFrame(C)

pd.set_option("display.width", 200)
print(f"\n=== ABSOLUTE K FROM A SINGLE ECG (test: {len(te):,} ECGs, {te.subject_id.nunique():,} patients; "
      f"within-patient r on {multi.sum():,} ECGs of patients with >=2) ===")
print(M.to_string())
print("\n=== CONTRASTS vs the standard model (scratch), patient-cluster bootstrap ===")
print(C.to_string(index=False))
od = OUT if A.fold is None else f"{OUT}/cv"
M.to_csv(f"{od}/single_ecg_k{SUF}.csv"); C.to_csv(f"{od}/single_ecg_k_contrasts{SUF}.csv", index=False)
pd.DataFrame({"study": te.study, "subject_id": sid, "k": y, **PRED}) \
  .assign(fold=-1 if A.fold is None else A.fold) \
  .to_parquet(f"{od}/preds_single_ecg_k{SUF}.parquet", index=False)
print(f"\nwrote {od}/single_ecg_k{SUF}.csv, single_ecg_k_contrasts{SUF}.csv, preds_single_ecg_k{SUF}.parquet")
