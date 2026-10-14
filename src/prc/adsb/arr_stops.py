"""Parking position of ARRIVING aircraft (2025): an arrival taxis in and stops at its stand, so its slow ground samples around the recorded
in-block time locate that stand without any departure label, and without the post-pushback hold that a departure's stops are confused with.
Join as for departures (flight number, callsign, IATA->ICAO translated flight number) at the arrival airport; samples on the ground at
<= 3 kn from 60 s before to 600 s after the in-block time; parking position = their median (at least 2 samples).
Usage (run from the folder that holds data/): python -m prc.adsb.arr_stops <out.parquet>"""
import json, os, pathlib, sys, time
import numpy as np, pandas as pd
from prc import config as C
from prc.features._util import log

I2I = json.load(open(C.CONFIGS / "iata2icao.json"))
out_path = sys.argv[1]
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt"]
arr = pd.concat([pd.read_parquet(p, columns=cols) for p in C.train_files()], ignore_index=True)
arr = arr[(arr.PHASE_mvt == "ARR") & arr.BLOCK_TIME_UTC_mvt.notna()].copy()
for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]: arr[c] = arr[c].dt.tz_convert(None)
arr["day"] = arr.BLOCK_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
arr["cs1"] = arr.FLIGHT_mvt.astype("string").str.strip().str.upper(); arr["cs2"] = arr.CALLSIGN_flt.astype("string").str.strip().str.upper()
m3 = arr.cs1.str.extract(r"^([A-Z0-9]{2})(\d{1,4})([A-Z]?)$"); arr["cs3"] = (m3[0].map(I2I) + m3[1].str.lstrip("0") + m3[2]).where(arr.cs2.isna())

def load(day):
    p = pathlib.Path(f"data/adsb_lol/{day}.parquet")
    if not p.exists(): return None
    a = pd.read_parquet(p, columns=["callsign", "airport", "t", "on_ground", "gs", "lat", "lon"])
    a = a[a.callsign.notna() & (a.on_ground == 1)].copy(); a["cs"] = a.callsign.astype("string").str.strip().str.upper()
    return a[~a.cs.isin(["NONE", "NAN", ""])]

t0 = time.time(); out = []
days = sorted(arr.day.unique())
if os.environ.get("ADSB_DAYS"): days = [x for x in days if x in set(open(os.environ["ADSB_DAYS"]).read().split())]   # a fixed day subset
for day in days:
    a = load(day)
    if a is None: continue
    nxt = load((pd.Timestamp(day) + pd.Timedelta("1D")).strftime("%Y-%m-%d"))   # arrivals just before midnight park after it
    if nxt is not None: a = pd.concat([a, nxt[nxt.t < pd.Timestamp(day) + pd.Timedelta("1D") + pd.Timedelta("15min")]], ignore_index=True)
    d = arr[arr.day == day]
    long = pd.concat([d[["MVT_ID_mvt", "ADES_mvt", "BLOCK_TIME_UTC_mvt", c]].rename(columns={c: "cs"}) for c in ["cs1", "cs2", "cs3"]])
    long = long.dropna(subset=["cs"]).drop_duplicates(["MVT_ID_mvt", "cs"]); long = long[~long.cs.isin(["NAN", "NONE", "", "<NA>"])]
    j = long.merge(a, left_on=["cs", "ADES_mvt"], right_on=["cs", "airport"], how="inner")
    rel = (j.t - j.BLOCK_TIME_UTC_mvt).dt.total_seconds()
    j = j[(rel >= -60) & (rel <= 600) & (j.gs <= 3)]
    if len(j):
        g = j.groupby("MVT_ID_mvt"); f = pd.DataFrame({"arr_lat": g.lat.median(), "arr_lon": g.lon.median(), "arr_n": g.size()})
        out.append(f[f.arr_n >= 2].reset_index())
    if len(out) and len(out) % 60 == 0: log(f"{day}: {sum(len(x) for x in out):,} arrivals parked ({time.time() - t0:.0f}s)")
F = pd.concat(out, ignore_index=True).merge(arr[["MVT_ID_mvt", "ADES_mvt", "STAND_mvt"]], on="MVT_ID_mvt", how="left")
F.to_parquet(out_path, index=False)
log(f"saved {out_path}: {len(F):,} parked arrivals of {len(arr):,} ({len(F) / len(arr):.1%}), by airport {F.ADES_mvt.value_counts().to_dict()} ({time.time() - t0:.0f}s)")
