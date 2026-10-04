"""Stage 8: the sec.6 comparison of difference representations.

FOUR arms, same encoder budget, same schedule:

  beat_sub    raw median-beat subtraction: input = beat2 - beat1        (sec.6 opt 1)
  emb_diff    shared encoder, z = f(x2) - f(x1)                          (sec.6 opt 2)
  siamese     shared encoder, head on [f(x1), f(x2), f(x2)-f(x1)]        (sec.6 opt 3)
  static_ecg2 ECG2 ONLY -- no difference at all                          <-- the honest comparator

The fourth arm is not in concept.md and it is the one that can sink the paper.
sec.1 claims the difference framing beats a static model; the only way to earn that
claim is to train the static model on the same data with the same budget and beat it.
If static_ecg2 matches the difference arms, the thesis is decorative.

Targets are multi-task: dCr / dK / dEGFR (standardised regression) + INCIDENT AKI
(masked BCE on `aki_incident` from stage 6b; NULL rows -- not at risk at t1 -- carry no
classification loss). Reporting is INCREMENTAL over the clinical floor, see
13_evaluate_v2.py.

v2 changes (2026-10-01, concept.md sec.0):
  * the v1 head learned the STATE label `aki` (prevalent AKI at t2), which a single ECG
    reads as well as a pair -- it pulled every encoder toward static features.
  * the checkpoint was chosen on that head's val AUC. It is now chosen on val R^2 of
    dK, the pre-registered primary target (--select), and that choice is identical
    for every arm, so the static-vs-difference contrast is not tilted by selection.
  * predictions for VAL and TEST are written keyed on (study1, study2), so a stale
    prediction file can no longer be silently joined to a newer pair table.

Amplitude handling: signals are divided by ONE GLOBAL SCALAR (not per record).
Per-record normalisation would delete the between-record amplitude change that is
the dK signature and the whole point of the difference.
"""
import os, argparse, time, json
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("MIMICABY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, OUT = f"{ROOT}/data", f"{ROOT}/outputs"
WF = f"{DATA}/waveforms"
DEV = "cuda"
TARGETS = ["dcr", "dk", "degfr"]
SEED = 20260713


# ------------------------------------------------------------------ data
class PairDS(Dataset):
    def __init__(self, df, arm, scale, use_beats, idx1="wf_idx1", idx2="wf_idx2"):
        self.df, self.arm, self.scale, self.use_beats = df.reset_index(drop=True), arm, scale, use_beats
        self.i1, self.i2 = df[idx1].values.astype(int), df[idx2].values.astype(int)
        self.y = df[[f"{t}_z" for t in TARGETS]].values.astype(np.float32)
        self.a = df["aki_incident"].fillna(-1).values.astype(np.float32)  # -1 = masked
        self.X = None  # opened lazily, per worker

    def _open(self):
        f = "beats.npy" if self.use_beats else "strips.npy"
        self.X = np.load(f"{WF}/{f}", mmap_mode="r")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        if self.X is None: self._open()
        x1 = np.asarray(self.X[self.i1[i]], dtype=np.float32) / self.scale
        x2 = np.asarray(self.X[self.i2[i]], dtype=np.float32) / self.scale
        return x1, x2, self.y[i], self.a[i]


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False);     self.b2 = nn.BatchNorm1d(cout)
        self.sc = (nn.Sequential() if (stride == 1 and cin == cout) else
                   nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.do(h)
        h = self.b2(self.c2(h))
        return F.relu(h + self.sc(x))


class Encoder(nn.Module):
    """1-D ResNet over 12-lead input. Shared across the two streams."""
    def __init__(self, emb=256, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, width, 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(width), nn.ReLU())
        chans = [width, width * 2, width * 4, width * 8]
        blocks, cin = [], width
        for c in chans:
            blocks += [Block(cin, c, 2), Block(c, c, 1)]
            cin = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(cin, emb)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = h.mean(-1)                    # global average pool over time
        return self.head(h)


class Net(nn.Module):
    """crop > 0 (strip arms only): train on random `crop`-sample windows, drawn
    independently for ECG1 and ECG2; at eval, average the outputs over `n_tta` evenly
    spaced windows. The encoder global-average-pools over time, so window length is
    free. Aim: average out beat-to-beat / acquisition nuisance (concept.md sec.5-3)."""
    def __init__(self, arm, emb=256, crop=0, n_tta=5):
        super().__init__()
        self.arm, self.crop, self.n_tta = arm, crop, n_tta
        self.enc = Encoder(emb)
        fdim = {"beat_sub": emb, "emb_diff": emb, "static_ecg2": emb, "siamese": emb * 3}[arm]
        self.head = nn.Sequential(nn.Linear(fdim, 128), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(128, len(TARGETS) + 1))

    def features(self, x1, x2):
        if self.arm == "beat_sub":                      # subtraction happens in SIGNAL space
            return self.enc(x2 - x1)
        if self.arm == "static_ecg2":                   # ECG1 never touched
            return self.enc(x2)
        z1, z2 = self.enc(x1), self.enc(x2)             # shared weights
        if self.arm == "emb_diff":                      # subtraction in EMBEDDING space
            return z2 - z1
        return torch.cat([z1, z2, z2 - z1], dim=1)      # siamese

    def _out(self, x1, x2):
        o = self.head(self.features(x1, x2))
        return o[:, :len(TARGETS)], o[:, len(TARGETS)]

    def forward(self, x1, x2):
        L, c = x1.shape[-1], self.crop
        if not c or L <= c:
            return self._out(x1, x2)
        if self.training:
            o1, o2 = (int(v) for v in torch.randint(0, L - c + 1, (2,)))
            return self._out(x1[..., o1:o1 + c], x2[..., o2:o2 + c])
        regs, logits = zip(*(self._out(x1[..., o:o + c], x2[..., o:o + c])
                             for o in np.linspace(0, L - c, self.n_tta).astype(int)))
        return torch.stack(regs).mean(0), torch.stack(logits).mean(0)


# ------------------------------------------------------------------ train / eval
def pair_loss(reg, logit, y, a):
    """MSE on the three deltas + BCE on incident AKI where it is defined (a >= 0)."""
    m = a >= 0
    bce = (F.binary_cross_entropy_with_logits(logit[m], a[m]) if m.any()
           else logit.sum() * 0.0)
    return F.mse_loss(reg, y) + bce


def run_epoch(model, dl, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for x1, x2, y, a in dl:
        x1, x2, y, a = (t.to(DEV, non_blocking=True) for t in (x1, x2, y, a))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reg, logit = model(x1, x2)
            loss = pair_loss(reg, logit, y, a)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        tot += loss.item() * len(a); n += len(a)
    return tot / n


@torch.no_grad()
def predict(model, dl):
    model.eval()
    R, L = [], []
    for x1, x2, y, a in dl:
        x1, x2 = x1.to(DEV), x2.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reg, logit = model(x1, x2)
        R.append(reg.float().cpu().numpy()); L.append(logit.float().cpu().numpy())
    return np.concatenate(R), np.concatenate(L)


def auc(y, s):
    y = np.asarray(y).astype(int)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def cv_folds(subject_id, nfolds=5):
    """Patient-level fold id, fixed by SEED: every pair of a patient lands in one fold."""
    subj = np.sort(pd.unique(subject_id))
    perm = np.random.default_rng(SEED).permutation(len(subj))
    fold_of = pd.Series(perm % nfolds, index=subj)
    return fold_of.loc[np.asarray(subject_id)].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["beat_sub", "emb_diff", "siamese", "static_ecg2"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--select", default="dk", choices=["dk", "aki_incident"],
                    help="checkpoint criterion on VAL: R^2 of dK (pre-registered) or incident-AKI AUC")
    ap.add_argument("--fold", type=int, default=None,
                    help="5-fold patient-grouped CV: fold k is TEST, fold k+1 is VAL, the rest TRAIN "
                         "(overrides the pre-registered split; outputs go to models/cv, outputs/cv)")
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--null_reg", type=float, default=0.0,
                    help="lambda: penalise predicted (standardised) change on TRAIN-split null "
                         "pairs (same patient, 0.5-6 h apart) toward 0. Difference arms only.")
    ap.add_argument("--crop", type=int, default=0,
                    help="random-crop length in samples for training + 5-window TTA (strips only)")
    ap.add_argument("--suffix", default="", help="appended to model/prediction names")
    a = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)

    use_beats = (a.arm == "beat_sub")
    df = pd.read_parquet(f"{DATA}/pairs_v2.parquet")
    assert not (a.null_reg and a.arm == "static_ecg2"), "null_reg needs a pair model"
    assert not (a.crop and a.arm == "beat_sub"), "crop applies to strip arms only"
    tag, mdir, odir = a.arm, f"{DATA}/models", OUT
    if a.fold is not None:
        fold = cv_folds(df.subject_id, a.nfolds)
        df["split"] = np.where(fold == a.fold, "test",
                      np.where(fold == (a.fold + 1) % a.nfolds, "val", "train"))
        tag, mdir, odir = f"{a.arm}_f{a.fold}", f"{DATA}/models/cv", f"{OUT}/cv"
        print(f"[{tag}] CV fold {a.fold}/{a.nfolds}: "
              f"{df.split.value_counts().to_dict()}", flush=True)
    tag += a.suffix

    tr = df[df.split == "train"]
    mu = {t: tr[t].mean() for t in TARGETS}
    sd = {t: tr[t].std() for t in TARGETS}
    for t in TARGETS:
        df[f"{t}_z"] = (df[t] - mu[t]) / sd[t]

    # ONE global amplitude scalar, from the training ECGs. Not per record.
    X = np.load(f"{WF}/{'beats' if use_beats else 'strips'}.npy", mmap_mode="r")
    samp = np.asarray(X[tr.wf_idx1.values[:3000].astype(int)], dtype=np.float32)
    scale = float(np.std(samp))
    print(f"[{a.arm}] global amplitude scale = {scale:.4f} mV  "
          f"(input {'beats 12x200' if use_beats else 'strips 12x2500'})", flush=True)

    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    mk = lambda d, sh: DataLoader(PairDS(d, a.arm, scale, use_beats), batch_size=a.bs,
                                  shuffle=sh, num_workers=10, pin_memory=True,
                                  persistent_workers=True, drop_last=sh)
    dtr, dva, dte = mk(tr, True), mk(va, False), mk(te, False)

    # null pairs, split by the same patient rule (stage 6c). TRAIN ones regularise,
    # VAL ones are only monitored -- the TEST ones are left for 15_nuisance_eval.py.
    nul = pd.read_parquet(f"{DATA}/null_pairs_v2.parquet")
    if a.fold is not None:      # keep null pairs patient-disjoint from the CV test fold
        nul["split"] = df.drop_duplicates("subject_id").set_index("subject_id").split \
                         .reindex(nul.subject_id).fillna("train").values
    for t in TARGETS: nul[f"{t}_z"] = 0.0
    nul["aki_incident"] = np.nan
    nl = lambda d, sh, bs: DataLoader(PairDS(d, a.arm, scale, use_beats), batch_size=bs,
                                      shuffle=sh, num_workers=4, pin_memory=True,
                                      persistent_workers=True, drop_last=sh)
    dnv = nl(nul[nul.split == "val"], False, 256)
    if a.null_reg:
        dnt = nl(nul[nul.split == "train"], True, a.bs // 2)
        def null_batches():
            while True:
                yield from dnt
        nit = null_batches()
        print(f"[{tag}] null_reg lambda={a.null_reg} on {int((nul.split == 'train').sum()):,} "
              f"train-split null pairs", flush=True)

    model = Net(a.arm, crop=a.crop).to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(dtr))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{a.arm}] params={n_par/1e6:.2f}M  train={len(tr):,} val={len(va):,} test={len(te):,}",
          flush=True)

    best, best_state, best_ep = -np.inf, None, 0
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x1, x2, y, aa in dtr:
            x1, x2, y, aa = (t.to(DEV, non_blocking=True) for t in (x1, x2, y, aa))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reg, logit = model(x1, x2)
                loss = pair_loss(reg, logit, y, aa)
                if a.null_reg:
                    n1, n2, _, _ = next(nit)
                    regn, _ = model(n1.to(DEV, non_blocking=True), n2.to(DEV, non_blocking=True))
                    loss = loss + a.null_reg * regn.float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item() * len(aa); n += len(aa)
        rv, lv = predict(model, dva)
        m = va.aki_incident.notna().values
        va_auc = auc(va.aki_incident.values[m], lv[m])
        va_r = np.corrcoef(rv[:, 0], va["dcr_z"].values)[0, 1]
        yk = va["dk_z"].values
        va_r2k = 1 - ((yk - rv[:, 1]) ** 2).sum() / ((yk - yk.mean()) ** 2).sum()
        rn, _ = predict(model, dnv)
        ratio = rn[:, 1].std() / rv[:, 1].std()     # null / real spread of predicted dK
        print(f"  ep{ep+1:02d} loss={tot/n:.4f} val_auc(inc)={va_auc:.3f} val_r(dCr)={va_r:.3f} "
              f"val_R2(dK)={va_r2k:.3f} null/real={ratio:.2f} ({time.time()-t0:.0f}s)", flush=True)
        crit = va_r2k if a.select == "dk" else va_auc
        if crit > best:
            best, best_ep = crit, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"[{a.arm}] selected epoch {best_ep} on val {a.select} = {best:.4f}", flush=True)
    os.makedirs(mdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    torch.save({"state": best_state, "arm": a.arm, "scale": scale, "mu": mu, "sd": sd,
                "select": a.select, "best_ep": best_ep, "pairs": "pairs_v2",
                "crop": a.crop, "null_reg": a.null_reg},
               f"{mdir}/{tag}.pt")

    outs = []
    for name, d, dl in (("val", va, dva), ("test", te, dte)):
        r_, l_ = predict(model, dl)
        o = d[["study1", "study2", "subject_id", "split"]].copy()
        o["logit"] = l_
        for i, t in enumerate(TARGETS):
            o[f"pred_{t}"] = r_[:, i] * sd[t] + mu[t]
        outs.append(o)
    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(f"{odir}/preds_{tag}.parquet", index=False)

    ot = out[out.split == "test"].merge(te[["study1", "study2", "dk", "dcr", "aki_incident"]],
                                         on=["study1", "study2"])
    m = ot.aki_incident.notna()
    print(f"\n[{a.arm}] TEST  auc(inc)={auc(ot.aki_incident[m].values, ot.logit[m].values):.3f}  "
          f"r(dCr)={np.corrcoef(ot.pred_dcr, ot.dcr)[0,1]:.3f}  r(dK)={np.corrcoef(ot.pred_dk, ot.dk)[0,1]:.3f}")
    print(f"wrote {odir}/preds_{tag}.parquet")


if __name__ == "__main__":
    main()
