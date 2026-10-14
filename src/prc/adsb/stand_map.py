"""Learn STAND LOCATIONS from adsb.lol ground tracks.

A short stationary episode is ambiguous (runway hold, taxiway pause, de-icing pad, or a stand). A LONG one is
almost always a stand. So: find long stationary episodes, grid their positions, and treat dense cells as stands.
Position then replaces dwell length as the discriminator, so that short dwells at a known stand can also mark the off-block time.
The map feeds only the `ads_sv*` columns, which the final model does not read; it is kept for reuse.

Output: configs/stand_map.parquet  (airport, clat, clon, n_ep, n_long, med_dur)
Usage: python -m prc.adsb.stand_map [--days 40] [--minlong 300]
"""
import argparse, glob, os, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=40); ap.add_argument("--minlong", type=float, default=300.0)
a = ap.parse_args()
LAT_CELL, LON_CELL = 0.00030, 0.00045          # ~33 m x ~33 m at European latitudes
files = sorted(glob.glob("data/adsb_lol/*.parquet"))
sel = files[:: max(1, len(files)//a.days)][:a.days]
print(f"scanning {len(sel)} of {len(files)} days for stationary episodes", flush=True)
acc = {}
t0 = time.time()
for i, p in enumerate(sel):
    d = pd.read_parquet(p, columns=["icao","airport","t","lat","lon","on_ground","gs"])
    d = d[(d.on_ground == 1)]
    d["gs"] = d.gs.fillna(0.0); d = d[d.gs <= 2.0]
    if d.empty: continue
    d = d.sort_values(["airport","icao","t"])
    newrun = (d.icao != d.icao.shift()) | (d.airport != d.airport.shift()) | (d.t.diff().dt.total_seconds() > 180)
    d["run"] = newrun.cumsum()
    g = d.groupby("run").agg(airport=("airport","first"), t0=("t","min"), t1=("t","max"),
                             lat=("lat","median"), lon=("lon","median"), n=("t","size"))
    g["dur"] = (g.t1 - g.t0).dt.total_seconds()
    g = g[g.dur >= a.minlong]
    if g.empty: continue
    g["clat"] = np.round(g.lat / LAT_CELL).astype(np.int64)
    g["clon"] = np.round(g.lon / LON_CELL).astype(np.int64)
    for (apt, cla, clo), z in g.groupby(["airport","clat","clon"]):
        k = (apt, cla, clo); c = acc.get(k)
        if c is None: acc[k] = [len(z), int((z.dur >= 900).sum()), list(z.dur)]
        else: c[0] += len(z); c[1] += int((z.dur >= 900).sum()); c[2].extend(z.dur.tolist())
    if (i+1) % 10 == 0: print(f"  {i+1}/{len(sel)} days, {len(acc)} cells ({time.time()-t0:.0f}s)", flush=True)
rows = [dict(airport=k[0], clat=k[1], clon=k[2], n_ep=v[0], n_long=v[1], med_dur=float(np.median(v[2]))) for k, v in acc.items()]
S = pd.DataFrame(rows)
S.to_parquet("configs/stand_map.parquet", index=False)
print(f"\nstand-map cells: {len(S)}  ({time.time()-t0:.0f}s)")
print(S.groupby("airport").agg(cells=("n_ep","size"), episodes=("n_ep","sum"), long_eps=("n_long","sum"),
                               cells_with_long=("n_long", lambda z: int((z>0).sum()))).to_string())
print("\ncell quality distribution (n_ep per cell):")
print(S.n_ep.describe().round(1).to_string())
