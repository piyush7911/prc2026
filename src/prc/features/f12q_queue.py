"""Step 12q: runway-queue neighbour features (raw observables only, no labels, no model outputs).
Errors cluster on the same runway within about 15 min, and the queue shows in how late the NEIGHBOURS take off relative to their planned
clocks (LOBT, schedule, EOBT_1); this also works where the NM off-block clock is a default (e.g. LTFM). For every departure: leave-one-out
mean over departures on the same runway (and at the same airport) within +-W min of take-off, W in {15, 30}, of clipped (tko-LOBT),
(tko-sched), (tko-EOBT_1); the same minus the airport-day mean (relative); neighbour counts. Computed per airport and day on training and
target rows alike. Usage: python -m prc.features.f12q_queue IN.parquet OUT.parquet"""
import sys
import numpy as np, pandas as pd
C = ["MVT_ID_mvt", "is_rank", "year", "doy", "minute_of_day", "apt", "rwy", "tko_minus_lobt", "tko_minus_sched", "tko_minus_eobt1"]
g = pd.read_parquet(sys.argv[1], columns=C); g["apt"] = g.apt.astype(str); g["rwy"] = g.rwy.astype(str)
g["t"] = g.doy.astype(int) * 1440 + g.minute_of_day.astype(int); g["day"] = g.year.astype(int) * 1000 + g.doy.astype(int)
V = {"lobt": g.tko_minus_lobt.clip(-3600, 7200), "sched": g.tko_minus_sched.clip(-3600, 7200), "eobt": g.tko_minus_eobt1.clip(-3600, 7200)}
out = {}
def window_mean(t, x, W):
    ok = ~np.isnan(x); xv = np.where(ok, x, 0.0); cs = np.cumsum(np.r_[0, xv]); cn = np.cumsum(np.r_[0, ok.astype(float)])
    lo = np.searchsorted(t, t - W); hi = np.searchsorted(t, t + W, side="right"); s = cs[hi] - cs[lo] - xv; n = cn[hi] - cn[lo] - ok
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n
for scope, keys in (("rwy", ["apt", "yearmark", "rwy"]), ("apt", ["apt", "yearmark"])):
    g["yearmark"] = g.year.astype(int)
    for key, idx in g.groupby(keys).groups.items():
        d = g.loc[idx].sort_values("t"); t = d.t.values
        for W in (15, 30):
            for nm, s in V.items():
                m, n = window_mean(t, s.loc[d.index].values.astype(float), W)
                out.setdefault(f"q_{scope}{W}_{nm}", pd.Series(np.nan, index=g.index)).loc[d.index] = m
                if nm == "lobt": out.setdefault(f"q_{scope}{W}_n", pd.Series(np.nan, index=g.index)).loc[d.index] = n
Q = pd.DataFrame(out)
dm = {nm: s.groupby([g.apt, g.day]).transform("mean") for nm, s in V.items()}
for c in list(Q.columns):
    nm = c.split("_")[-1]
    if nm in dm: Q[c + "_rel"] = Q[c] - dm[nm]
f = pd.read_parquet(sys.argv[1]); f = pd.concat([f, Q.astype("float32")], axis=1)
print("new columns:", len(Q.columns), "| null share (median):", round(float(Q.isna().mean().median()), 3))
for c in ["q_rwy15_lobt", "q_rwy15_lobt_rel", "q_rwy15_n"]:
    print(f"  {c}: 2025 Jan/Jul median {f.loc[(~f.is_rank) & f.month.isin([1,7]), c].median():.1f}  2026 median {f.loc[f.is_rank, c].median():.1f}")
f.to_parquet(sys.argv[2]); print("saved", sys.argv[2], f.shape)
