"""Two further ideas, tested on our 2025 out-of-fold predictions (prc.models.pipeline, six 2-month folds).
Gate: keep only if it improves Jan, Feb, Jun and Jul 2025 (the four final months) and the gain is not carried by fewer than 100 rows.

 1. congestion floor for departures without an NM off-block clock: on congested days their taxi-out tracks the live taxi level of
    matched neighbours; floor = max(pred, k * level) when level > 1200 or 1500 s (k 0.8 or 0.95).
 2. same-runway ARRIVAL taxi context: counts / mean taxi-in of arrivals that reached the stand in the previous 15/60/180 min on the
    departure's runway, and arrivals still taxiing at the departure's off-block estimate. Tested as a small LightGBM corrector of the
    pipeline residual (same folds), weights 0.25/0.5/1, against a control corrector without the new columns.
Result: both fail the gate (see README, "What did not work").
Usage: python -m prc.validate.extra_ideas [--build-features]"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from prc import config as C
from prc.models import pipeline as P

OUT = P.OUT; MONTHS = (1, 2, 6, 7)
def rmse(e): return float(np.sqrt(np.mean(np.square(e))))
def report(name, y, p0, p1, month, sel_rows):
    out = []; ok = True
    for m in MONTHS:
        s = month == m; d = rmse(y[s] - p1[s]) - rmse(y[s] - p0[s]); out.append(f"m{m} {d:+.3f}"); ok &= d < 0
    g = (y - p0) ** 2 - (y - p1) ** 2; tot = g.sum(); top = np.sort(g)[::-1][:100].sum() / tot if tot > 0 else np.nan
    allm = np.isin(month, MONTHS); d4 = rmse(y[allm] - p1[allm]) - rmse(y[allm] - p0[allm])
    print(f"{name:52s} rows {int(sel_rows.sum()):6,} | {' '.join(out)} | 4 months {d4:+.3f} | top-100 share {top:.0%} | {'PASS' if ok and top < 1 else 'fail'}", flush=True)

def arrival_rwy_features():
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "RUNWAY_mvt"]
    mv = C.read_all(cols)
    ep = lambda s: (s.dt.tz_convert(None) - pd.Timestamp(0)).dt.total_seconds().to_numpy()
    arr = mv[mv.PHASE_mvt == "ARR"].copy(); dep = mv[mv.PHASE_mvt == "DEP"].copy()
    arr["ldg"] = ep(arr.MVT_TIME_UTC_mvt); arr["inb"] = ep(arr.BLOCK_TIME_UTC_mvt); arr["tin"] = arr.inb - arr.ldg
    arr = arr[(arr.tin >= 0) & (arr.tin <= 7200)]
    dep["tko"] = ep(dep.MVT_TIME_UTC_mvt); a3 = ep(dep.AOBT_3_flt); dep["obt"] = np.where(np.isfinite(a3) & (dep.tko - a3 < 7200), a3, dep.tko - 900)
    dep = dep.reset_index(drop=True); res = {}
    def fill(idx, A, t, key):
        o = np.argsort(A.inb.values); inb = A.inb.values[o]; cs = np.r_[0, np.cumsum(A.tin.values[o])]; ldg_s = np.sort(A.ldg.values)
        for W in (15, 60, 180):
            lo = np.searchsorted(inb, t - W * 60); hi = np.searchsorted(inb, t); n = hi - lo
            res.setdefault(f"ra_{key}_n_{W}", np.full(len(dep), np.nan))[idx] = n
            res.setdefault(f"ra_{key}_tin_{W}", np.full(len(dep), np.nan))[idx] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        res.setdefault(f"ra_{key}_taxiing", np.full(len(dep), np.nan))[idx] = np.searchsorted(ldg_s, t) - np.searchsorted(inb, t)
    for apt, g in dep.groupby("ADEP_mvt"):
        A = arr[arr.ADES_mvt == apt]; fill(g.index.to_numpy(), A, g.obt.to_numpy(), "apt")
        for r, gr in g.groupby("RUNWAY_mvt"):
            Ar = A[A.RUNWAY_mvt == r]
            if len(Ar): fill(gr.index.to_numpy(), Ar, gr.obt.to_numpy(), "rwy")
    F = pd.concat([dep[["MVT_ID_mvt"]], pd.DataFrame(res)], axis=1)
    F.to_parquet(f"{OUT}/rwy_arr_feats.parquet", index=False); print("saved", F.shape, list(F.columns)[1:]); return F

def main():
    f = pd.read_parquet(P.FEAT); lab = f[~f.is_rank].reset_index(drop=True)
    p, c = P._assemble(lab, "oof"); y = lab.y.to_numpy(); month = lab.month.to_numpy(); ln = c["lane"]
    print(f"OOF pipeline: " + " ".join(f"m{m} {rmse((y - p)[month == m]):.2f}" for m in MONTHS))
    # ---- idea 1: congestion floor ----
    for nm, lanes in (("no NM clock, all", ("unm_nr", "unm_rome")), ("no NM clock, non-Rome", ("unm_nr",))):
        for lev_col in ("nb_naive_med_m30", "nb_naive_mean_m30"):
            for thr, k in ((1500, 0.95), (1500, 0.8), (1200, 0.95)):
                lev = lab[lev_col].to_numpy(); sel = np.isin(ln, lanes) & np.isfinite(lev) & (lev > thr) & (p < k * lev)
                p1 = np.where(sel, k * lev, p); report(f"floor {nm} | {lev_col} > {thr} | x{k}", y, p, p1, month, sel)
    # ---- idea 2: same-runway arrival context ----
    import os
    A = pd.read_parquet(f"{OUT}/rwy_arr_feats.parquet") if os.path.exists(f"{OUT}/rwy_arr_feats.parquet") and "--build-features" not in sys.argv else arrival_rwy_features()
    X = lab[["MVT_ID_mvt", "apt", "hour", "naive_raw", "arr_taxiin_mean_m30", "arr_excess_mean_m30"]].merge(A, on="MVT_ID_mvt", how="left")
    X["p"] = p; X["lane_c"] = pd.Categorical(ln); X["apt"] = X.apt.astype("category"); cols = [c_ for c_ in X.columns if c_ != "MVT_ID_mvt"]
    e = np.clip(y - p, -3600, 3600); ok = P.good(lab); corr = np.zeros(len(lab))
    prm = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=50, verbose=-1, num_threads=P.THREADS)
    for fm in P.FOLDS:
        te = np.isin(month, fm); tr = ~te & ok
        corr[te] = lgb.train(prm, lgb.Dataset(X.loc[tr, cols], e[tr]), 300).predict(X.loc[te, cols])
    base_cols = [c_ for c_ in cols if not c_.startswith("ra_")]; corr0 = np.zeros(len(lab))   # control: same corrector WITHOUT the new columns
    for fm in P.FOLDS:
        te = np.isin(month, fm); tr = ~te & ok
        corr0[te] = lgb.train(prm, lgb.Dataset(X.loc[tr, base_cols], e[tr]), 300).predict(X.loc[te, base_cols])
    mt = np.isin(ln, ("covered", "uncovered"))
    for w in (0.25, 0.5, 1.0):
        report(f"control corrector (no new cols) w{w}", y, p, np.where(mt, p + w * corr0, p), month, mt)
        report(f"runway-arrival corrector w{w}", y, p, np.where(mt, p + w * corr, p), month, mt)
        report(f"  increment of the new cols over control w{w}", y, np.where(mt, p + w * corr0, p), np.where(mt, p + w * corr, p), month, mt)

if __name__ == "__main__":
    main()
