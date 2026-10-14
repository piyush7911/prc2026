"""Assigned-stand positions learned from 2025 only. A 2025 departure whose ADS-B stop ends within 60 s of its recorded off-block time
(label) parked at its assigned stand; its parking position (median of the stationary samples of that stop, prc.adsb.features) is one
observation of the stand. Per airport and stand: the median position of at least 3 observations whose median distance to it is at most 30 m.
Optional third input (prc.adsb.arr_stops): parking positions of 2025 ARRIVALS, which stop at their stand after taxi-in (no label involved).
Stands not located by the departure rule are added from them when at least 5 arrivals parked there, at least 60 % of them within 30 m of
the median and their median distance is at most 20 m. They are used only at the airports in ARR_STAND_AIRPORTS (default EDDF, EDDM, EHAM, LEBL,
LSZH): there the stand times they give align with the 2025 recorded off-block as well as those of departure-located stands (53-83 % within
60 s); at EGLL, LEMD and LIRF the departing aircraft is rarely received at its stand before pushback and they align with 1-10 %.
Usage: python -m prc.adsb.stand_pos <adsb_feats.parquet with ads_park_*> <out.parquet> [<arrival stops.parquet>]"""
import os, sys
import numpy as np, pandas as pd
from prc import config as C
from prc.features._util import log
src, out = sys.argv[1], sys.argv[2]
A = pd.read_parquet(src, columns=["MVT_ID_mvt", "ads_dwell_end", "ads_park_lat", "ads_park_lon", "ads_park_n"])
T = C.read_train(["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "TAXITIME_SEC_mvt"]); T = T[T.PHASE_mvt == "DEP"]
x = T.merge(A, on="MVT_ID_mvt"); x = x[x.STAND_mvt.notna() & x.ads_park_lat.notna() & (x.ads_park_n >= 3)]
x = x[(x.ads_dwell_end + x.TAXITIME_SEC_mvt).abs() <= 60]          # the stop ended at the recorded off-block: it was the stand
g = x.groupby(["ADEP_mvt", "STAND_mvt"]).agg(lat=("ads_park_lat", "median"), lon=("ads_park_lon", "median"), n=("ads_park_lat", "size")).reset_index()
x = x.merge(g, on=["ADEP_mvt", "STAND_mvt"])
x["d"] = np.sqrt(((x.ads_park_lat - x.lat) * 111320) ** 2 + ((x.ads_park_lon - x.lon) * 111320 * np.cos(np.radians(x.lat))) ** 2)
g = g.merge(x.groupby(["ADEP_mvt", "STAND_mvt"]).d.median().rename("spread_m").reset_index(), on=["ADEP_mvt", "STAND_mvt"])
keep = g[(g.n >= 3) & (g.spread_m <= 30)].rename(columns={"ADEP_mvt": "airport", "STAND_mvt": "stand"}); keep["source"] = "departure"
if len(sys.argv) > 3:
    z = pd.read_parquet(sys.argv[3]); z = z[z.STAND_mvt.notna()].copy(); z["STAND_mvt"] = z.STAND_mvt.astype(str)
    h = z.groupby(["ADES_mvt", "STAND_mvt"]).agg(lat=("arr_lat", "median"), lon=("arr_lon", "median"), n=("arr_lat", "size")).reset_index()
    z = z.merge(h, on=["ADES_mvt", "STAND_mvt"])
    z["d"] = np.sqrt(((z.arr_lat - z.lat) * 111320) ** 2 + ((z.arr_lon - z.lon) * 111320 * np.cos(np.radians(z.lat))) ** 2)
    h = h.merge(z.groupby(["ADES_mvt", "STAND_mvt"]).agg(share30=("d", lambda v: (v <= 30).mean()), spread_m=("d", "median")).reset_index(), on=["ADES_mvt", "STAND_mvt"])
    h = h[(h.n >= 5) & (h.share30 >= 0.6) & (h.spread_m <= 20)].rename(columns={"ADES_mvt": "airport", "STAND_mvt": "stand"}).drop(columns="share30")
    AIRPORTS = os.environ.get("ARR_STAND_AIRPORTS", "EDDF,EDDM,EHAM,LEBL,LSZH").split(",")
    k = keep.assign(stand=keep.stand.astype(str)).merge(h, on=["airport", "stand"], suffixes=("", "_a"))
    dd = np.sqrt(((k.lat - k.lat_a) * 111320) ** 2 + ((k.lon - k.lon_a) * 111320 * np.cos(np.radians(k.lat))) ** 2)
    log(f"arrival positions: {len(h)} stands; where the departure rule also located the stand ({len(k)}), median distance {dd.median():.1f} m, {(dd <= 15).mean():.0%} within 15 m")
    h = h[~h.set_index(["airport", "stand"]).index.isin(keep.assign(stand=keep.stand.astype(str)).set_index(["airport", "stand"]).index) & h.airport.isin(AIRPORTS)]; h["source"] = "arrival"
    keep = pd.concat([keep.assign(stand=keep.stand.astype(str)), h], ignore_index=True)
keep.to_parquet(out, index=False)
log(f"{out}: {len(keep)} stands ({(keep.source == 'departure').sum()} from {len(x):,} reliable 2025 departures of {len(g)} stands seen, {(keep.source == 'arrival').sum()} from arrivals) | by airport: {keep.airport.value_counts().to_dict()} | "
    f"median spread {keep.spread_m.median():.1f} m")
