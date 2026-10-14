"""EVALUATED, NOT USED IN THE FINAL MODEL (fails the 2025 gate, see prc.validate.tabm_blend and README).
TabM-style neural partner for departures whose own ground track was not seen (uncovered rows).

Trees and a neural network make different errors; a small share of a network's prediction lowers the squared error of the blend even when
the network alone is worse. The network is a BatchEnsemble MLP ("TabM-mini"): k = 16 members share their weights and differ by rank-1
scaling vectors. Inputs: scaled numeric features with a missing-flag per feature, and learned embeddings of the categorical features
(ADS-B columns are not used: the partner is meant for rows without ADS-B). Target: y - b(row) clipped to [-1800, 3600] s for training stability only.
Trained on all matched 2025 rows; the six out-of-fold folds are the same as in prc.models.pipeline.

Usage:  python -m prc.validate.tabm --mode oof                  (six folds, writes oof_tabm.parquet)
        python -m prc.validate.tabm --mode final --seeds 0,1,2  (fits on all 2025, writes pred_tabm_s<seed>.parquet)
        python -m prc.validate.tabm --mode oof --smoke 20000    (code-path check on a small sample)
"""
import argparse, math, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from prc.models import pipeline as P

EPOCHS, K, HIDDEN, LAYERS, EMB, DROPOUT = 8, 16, 512, 3, 8, 0.1
LR, BATCH, WD, CLIP_LO, CLIP_HI = 2e-3, 4096, 1e-5, -1800.0, 3600.0
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

class TabM(nn.Module):
    def __init__(self, n_num, cat_cards, k=K, hidden=HIDDEN, layers=LAYERS, emb=EMB, drop=DROPOUT):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 1, emb) for c in cat_cards]); d_in = n_num + emb * len(cat_cards)
        dims = [d_in] + [hidden] * layers
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        ones = lambda n: nn.Parameter(torch.randn(k, n) * 0.1 + 1.0)
        self.r0 = ones(d_in); self.r = nn.ParameterList([ones(dims[i]) for i in range(layers)]); self.s = nn.ParameterList([ones(dims[i + 1]) for i in range(layers)])
        self.drop = nn.Dropout(drop); self.head = nn.Parameter(torch.randn(k, hidden) * 0.02); self.bias = nn.Parameter(torch.zeros(k))
    def forward(self, xn, xc):
        x = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)], 1).unsqueeze(1) * self.r0
        for i, lin in enumerate(self.lins):
            x = self.drop(torch.relu((x * self.r[i]) @ lin.weight.T * self.s[i] + lin.bias))
        return (x * self.head).sum(-1) + self.bias                     # (batch, k): one prediction per member

class Prep:
    """Median / 1-99 percentile scaling of the numeric columns and top-2000 category codes, fitted on the training rows only."""
    def __init__(self, df, nums, cats):
        self.nums, self.cats, self.stats = nums, cats, {}
        for c in nums:
            v = df[c].to_numpy(dtype="float32"); med = np.nanmedian(v); lo, hi = np.nanpercentile(v, [1, 99])
            self.stats[c] = (med, lo, hi, np.nanstd(np.clip(v, lo, hi)) + 1e-6)
        self.maps = {c: {k: j + 1 for j, k in enumerate(df[c].astype(str).value_counts().index[:2000])} for c in cats}
    def __call__(self, df):
        xn = np.empty((len(df), 2 * len(self.nums)), dtype=np.float32)
        for i, c in enumerate(self.nums):
            med, lo, hi, sd = self.stats[c]; v = df[c].to_numpy(dtype="float32"); miss = np.isnan(v)
            xn[:, 2 * i] = (np.clip(np.where(miss, med, v), lo, hi) - med) / sd; xn[:, 2 * i + 1] = miss
        xc = np.stack([df[c].astype(str).map(self.maps[c]).fillna(0).astype(np.int64).to_numpy() for c in self.cats], 1)
        return np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0), xc

def fit_predict(tr, targets, nums, cats, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    prep = Prep(tr, nums, cats); xn, xc = prep(tr)
    y = np.clip(tr.y.to_numpy() - P.base_zf(tr), CLIP_LO, CLIP_HI).astype(np.float32); ys = float(y.std())
    model = TabM(xn.shape[1], [max(m.values()) if m else 1 for m in prep.maps.values()]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); n = len(y)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * math.ceil(n / BATCH))
    xn_t, xc_t, y_t = torch.from_numpy(xn).to(dev), torch.from_numpy(xc).to(dev), torch.from_numpy(y / ys).to(dev); t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]; opt.zero_grad(); loss = ((model(xn_t[idx], xc_t[idx]) - y_t[idx].unsqueeze(1)) ** 2).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item() * len(idx)
        log(f"  epoch {ep + 1}/{EPOCHS} train mse {tot / n:.4f} ({time.time() - t0:.0f}s)")
    model.eval(); outs = []
    with torch.no_grad():
        for te in targets:
            xn2, xc2 = prep(te); pred = [model(torch.from_numpy(xn2[i:i + 16384]).to(dev), torch.from_numpy(xc2[i:i + 16384]).to(dev)).mean(1).cpu().numpy()
                                         for i in range(0, len(te), 16384)]
            outs.append(np.concatenate(pred) * ys + P.base_zf(te))
    return outs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="oof", choices=["oof", "final"]); ap.add_argument("--seeds", default="0")
    ap.add_argument("--smoke", type=int, default=0); a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    torch.set_num_threads(4); dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); log(f"device {dev}")
    f, fb, _ = P.load(); f = f.reset_index(drop=True)
    cats = [c for c in P.CATS if c in fb]; nums = [c for c in fb if c not in cats]
    lab = f[~f.is_rank]; nums = [c for c in nums if lab[c].notna().mean() > 0.001]       # all-empty columns would give NaN scaling
    matched = (lab.aobt3_missing == 0) & (lab.y >= 0)
    if a.mode == "oof":
        out = []
        for fm in P.FOLDS:
            tr = lab[matched & ~lab.month.isin(fm)]; te = lab[lab.month.isin(fm)]
            if a.smoke: tr = tr.sample(a.smoke, random_state=0); te = te.sample(min(a.smoke, len(te)), random_state=0)
            log(f"fold {fm}: TabM on {len(tr):,} rows, predicting {len(te):,}"); p = fit_predict(tr, [te], nums, cats, seeds[0], dev)[0]
            out.append(pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "pred": p})); e = te.y.to_numpy() - p
            log(f"fold {fm}: RMSE all rows {P.rmse(e):.2f}")
            pd.concat(out).to_parquet(f"{P.OUT}/oof_tabm.parquet", index=False)         # written after every fold: a crash keeps finished folds
    else:
        tr = lab[matched]; te = f[f.is_rank]
        for sd in seeds:
            fn = f"{P.OUT}/pred_tabm_s{sd}.parquet"
            if os.path.exists(fn): continue
            if a.smoke: tr = tr.sample(a.smoke, random_state=0)
            p = fit_predict(tr, [te], nums, cats, sd, dev)[0]; pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "pred": p}).to_parquet(fn, index=False)
            log(f"TabM final seed {sd}: {len(te):,} target rows, mean {p.mean():.1f}")

if __name__ == "__main__":
    main()
