"""Airport data-defect state, computed from ARRIVAL rows, whose labels are visible in 2025 AND 2026.
For each departure: among arrivals at the same airport in the previous 6 / 24 h,
  - defect rate: in-block time equals the scheduled arrival time (the same defaulting that fills departure blocks)
  - degraded rate: no NM match or no aircraft type
  - the same two among arrivals of the SAME airline code
Plus the same-airline degraded share among recent departures (a row property, not a label).
Usage: python -m prc.features.f12_defects <in.parquet> <out.parquet>
"""
import glob, time
import numpy as np, pandas as pd
from prc.features._util import log
import sys

t0 = time.time()
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt"]
from prc import config as C
mv = C.read_all(cols)
for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]: mv[c] = mv[c].dt.tz_convert(None)
mv["al"] = mv.FLIGHT_mvt.astype(str).str.extract(r"^([A-Z0-9]{2,3}?)\d")[0].fillna("NA")
ep = lambda s: (s - pd.Timestamp("2025-01-01")).dt.total_seconds().to_numpy()
dep = mv[mv.PHASE_mvt == "DEP"].copy(); arr = mv[mv.PHASE_mvt == "ARR"].copy()
# arrival defect: in-block == scheduled arrival (to the minute) -> the airport wrote the schedule instead of an actual time
arr["defect"] = ((arr.BLOCK_TIME_UTC_mvt - arr.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60).astype(np.float32)
arr["degr"] = (arr.FLIGHT_ID_mvt.isna() | arr.AIRCRAFT_TYPE_mvt.isna()).astype(np.float32)
dep["degr"] = (dep.FLIGHT_ID_mvt.isna() | dep.AIRCRAFT_TYPE_mvt.isna()).astype(np.float32)
n = len(dep); pos = pd.Series(np.arange(n), index=dep.index)
out = {k: np.full(n, np.nan, dtype=np.float32) for k in
       ["arr_defect_6h", "arr_defect_24h", "arr_degr_24h", "arr_defect_al_24h", "arr_degr_al_24h", "dep_degr_al_24h", "dep_degr_6h", "arr_n_24h"]}


def roll(ref, ev_t, ev_v, w):
    """mean of ev_v over events with time in [ref-w, ref); ev sorted by time"""
    o = np.argsort(ev_t); t = ev_t[o]; v = np.nan_to_num(ev_v[o]).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    L = np.searchsorted(t, ref - w, "left"); R = np.searchsorted(ref * 0 + t, ref, "left") if False else np.searchsorted(t, ref, "left")
    cnt = (R - L).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[R] - cs[L]) / np.where(cnt > 0, cnt, 1), np.nan), cnt


for apt, g in dep.groupby("ADEP_mvt"):
    p = pos[g.index].to_numpy(); ref = ep(g.MVT_TIME_UTC_mvt)
    A = arr[arr.ADES_mvt == apt]; at = ep(A.MVT_TIME_UTC_mvt)
    D = dep[dep.ADEP_mvt == apt]; dt_ = ep(D.MVT_TIME_UTC_mvt)
    out["arr_defect_6h"][p] = roll(ref, at, A.defect.to_numpy(), 6 * 3600)[0]
    m24, c24 = roll(ref, at, A.defect.to_numpy(), 24 * 3600); out["arr_defect_24h"][p] = m24; out["arr_n_24h"][p] = c24
    out["arr_degr_24h"][p] = roll(ref, at, A.degr.to_numpy(), 24 * 3600)[0]
    out["dep_degr_6h"][p] = roll(ref, dt_, D.degr.to_numpy(), 6 * 3600)[0]
    for al, ga in g.groupby("al"):
        pa = pos[ga.index].to_numpy(); ra = ep(ga.MVT_TIME_UTC_mvt)
        Aa = A[A.al == al]; Da = D[D.al == al]
        if len(Aa):
            out["arr_defect_al_24h"][pa] = roll(ra, ep(Aa.MVT_TIME_UTC_mvt), Aa.defect.to_numpy(), 24 * 3600)[0]
            out["arr_degr_al_24h"][pa] = roll(ra, ep(Aa.MVT_TIME_UTC_mvt), Aa.degr.to_numpy(), 24 * 3600)[0]
        if len(Da):
            out["dep_degr_al_24h"][pa] = roll(ra, ep(Da.MVT_TIME_UTC_mvt), Da.degr.to_numpy(), 24 * 3600)[0]
new = pd.DataFrame(out, index=dep.index); new["MVT_ID_mvt"] = dep.MVT_ID_mvt.to_numpy()
log(f"defect features in {time.time() - t0:.0f}s\n" + new.drop(columns="MVT_ID_mvt").describe().T[["count", "mean", "50%", "max"]].round(4).to_string())
f = pd.read_parquet(sys.argv[1])
f = f.merge(new, on="MVT_ID_mvt", how="left"); f.to_parquet(sys.argv[2])
log(f"saved {sys.argv[2]} {f.shape}")
