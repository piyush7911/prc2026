"""Step 10: push-delay state, runway headways and timestamp precision.
  - push-delay state: mean / frac>20min / count of OTHER departures' (AOBT_3 - EOBT_1) with AOBT_3 in the
    previous 30 min at the same airport (AOBT_3 <= my takeoff, so ranking-safe); own flight excluded.
  - headways: seconds since the previous DEP takeoff on my runway / at my airport, seconds to the next DEP
    takeoff on my runway, seconds since the previous landing on my runway (all takeoff/landing times are known in ranking), capped 3600.
  - timestamp precision: second-of-minute of MVT_TIME and SCHED_TIME (reporting-format signal).
Usage: python -m prc.features.f10_pushdelay_headway <in.parquet> <out.parquet>
"""
import glob, time
import numpy as np, pandas as pd
from prc.features._util import log
import sys

DATA = "data"; t0 = time.time()
files = sorted(glob.glob(f"{DATA}/training_*.parquet"))
from prc import config as C
mv = C.read_all()
for c in mv.columns:
    if str(mv[c].dtype).startswith("datetime64[us, UTC]"): mv[c] = mv[c].dt.tz_convert(None)
ep = lambda s: (s - pd.Timestamp("2025-01-01")).dt.total_seconds().to_numpy()
dep = mv[mv.PHASE_mvt == "DEP"].copy(); arr = mv[mv.PHASE_mvt == "ARR"]
log(f"loaded {len(dep)} departures, {len(arr)} arrivals")
n = len(dep); pos = pd.Series(np.arange(n), index=dep.index)
out = {k: np.full(n, np.nan, dtype=np.float32) for k in ["push_delay_mean_m30", "push_delay_frac20_m30", "push_delay_cnt_m30", "push_delay_mean_m60",
                                                        "rwy_headway_prev", "rwy_headway_next", "apt_headway_prev", "rwy_arr_since_last"]}
t_all = ep(dep.MVT_TIME_UTC_mvt); a_all = ep(dep.AOBT_3_flt); e_all = ep(dep.EOBT_1_flt)
implied = t_all - a_all; ok_push = np.isfinite(a_all) & np.isfinite(e_all) & (implied >= 120) & (implied <= 7200)
lat_all = np.clip(a_all - e_all, -3600, 4 * 3600)
for apt, g in dep.groupby("ADEP_mvt"):
    p = pos[g.index].to_numpy(); t = t_all[p]
    # push-delay state from others: index AOBTs, prefix sums of lateness and of (lateness > 1200)
    ok = ok_push[p]; a = a_all[p][ok]; lat = lat_all[p][ok]; o = np.argsort(a); a = a[o]; lat = lat[o]
    cs = np.concatenate([[0.0], np.cumsum(lat)]); c20 = np.concatenate([[0.0], np.cumsum(lat > 1200)])
    for w, key in [(1800, "m30"), (3600, "m60")]:
        L = np.searchsorted(a, t - w, "left"); R = np.searchsorted(a, t, "right")
        s = cs[R] - cs[L]; k = c20[R] - c20[L]; cnt = (R - L).astype(float)
        # exclude self when own AOBT is in the window
        self_in = ok & (a_all[p] >= t - w) & (a_all[p] <= t)
        s = s - np.where(self_in, lat_all[p], 0.0); k = k - np.where(self_in, (lat_all[p] > 1200), 0.0); cnt = cnt - self_in
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"push_delay_mean_{key}"][p] = np.where(cnt > 0, s / cnt, np.nan)
            if key == "m30": out["push_delay_frac20_m30"][p] = np.where(cnt > 0, k / cnt, np.nan); out["push_delay_cnt_m30"][p] = cnt
    # airport headway
    ts = np.sort(t); i = np.searchsorted(ts, t, "left"); prev = np.where(i > 0, t - ts[np.maximum(i - 1, 0)], 3600.0); out["apt_headway_prev"][p] = np.minimum(prev, 3600)
    A = arr[arr.ADES_mvt == apt]
    for rwy, gr in g.groupby("RUNWAY_mvt"):
        pr = pos[gr.index].to_numpy(); tr_ = t_all[pr]; ts = np.sort(tr_)
        i = np.searchsorted(ts, tr_, "left"); j = np.searchsorted(ts, tr_, "right")
        out["rwy_headway_prev"][pr] = np.minimum(np.where(i > 0, tr_ - ts[np.maximum(i - 1, 0)], 3600.0), 3600)
        out["rwy_headway_next"][pr] = np.minimum(np.where(j < len(ts), ts[np.minimum(j, len(ts) - 1)] - tr_, 3600.0), 3600)
        la = np.sort(ep(A[A.RUNWAY_mvt == rwy].MVT_TIME_UTC_mvt)); k = np.searchsorted(la, tr_, "left")
        out["rwy_arr_since_last"][pr] = np.minimum(np.where(k > 0, tr_ - la[np.maximum(k - 1, 0)], 3600.0), 3600) if len(la) else 3600.0
new = pd.DataFrame(out, index=dep.index)
new["mvt_second"] = dep.MVT_TIME_UTC_mvt.dt.second.astype(np.float32).to_numpy(); new["sched_second"] = dep.SCHED_TIME_UTC_mvt.dt.second.astype(np.float32).to_numpy()
new["sched_min_mod5"] = (dep.SCHED_TIME_UTC_mvt.dt.minute % 5).astype(np.float32).to_numpy()
new["MVT_ID_mvt"] = dep.MVT_ID_mvt.to_numpy()
log(f"new features built in {time.time() - t0:.0f}s; describe:\n{new.drop(columns='MVT_ID_mvt').describe().T[['count', 'mean', '50%', 'max']].round(1).to_string()}")
f8 = pd.read_parquet(sys.argv[1]); assert not any(c in f8.columns for c in new.columns if c != "MVT_ID_mvt")
f10 = f8.merge(new, on="MVT_ID_mvt", how="left"); assert len(f10) == len(f8)
f10.to_parquet(sys.argv[2]); log(f"saved {sys.argv[2]} {f10.shape}")
