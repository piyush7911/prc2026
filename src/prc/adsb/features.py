"""Per-departure ADS-B features from data/adsb_lol/<day>.parquet (adsb.lol globe_history, ODbL 1.0).
For each departure (2025 training + 2026 ranking) whose day file exists: samples of the same callsign (FLIGHT_mvt or CALLSIGN_flt)
inside the departure airport's box within [takeoff - 3 h, takeoff]. Features (seconds relative to TAKEOFF, negative = before):
  ads_cov      coverage code: -1 day not extracted, 0 no samples, 1 low-altitude only, 2 ground moving only, 3 seen stationary (gs<=2)
  ads_first_seen, ads_first_ground, ads_first_mov (gs>5), ads_dwell_end (last stationary sample before first moving), ads_first_fast (gs>15)
  ads_n_ground, ads_n_still, ads_max_gs, ads_apt_day_cov (share of that airport-day's departures with code>=2: feeder availability)
  ads_dwell_minus_aobt, ads_mov_minus_aobt (relative to AOBT_3 when present)
  ads_park_lat/lon/n  position of the stop that ends at the dwell end (used by prc.adsb.stand_pos to learn the stand positions)
  ads_as_*     with STAND_POS: samples within 40 m of the ASSIGNED stand (position learned on 2025): first/last time there, the time the
               aircraft left it (last sample there before it is next seen > 80 m away), sample count, closest distance
Joins: callsign (type 1), aircraft address when the callsign brings no ground sample (ICAO_LINK, type 3), and the unique unclaimed
aircraft lifting off within +-60 s of the take-off (LIFTOFF_LINK, type 4).
Output: $ADSB_OUT (default outputs/adsb_feats.parquet) keyed by MVT_ID_mvt. Usage: python -m prc.adsb.features
"""
import glob, pathlib, time
from prc import config as C
import pandas as pd
import numpy as np, pandas as pd
from prc.features._util import log

import json, os; I2I = json.load(open(C.CONFIGS / "iata2icao.json"))
LAT_CELL, LON_CELL = 0.00030, 0.00045
try:
    _sm = pd.read_parquet(C.CONFIGS / "stand_map.parquet")
    _sm = _sm[(_sm.n_ep >= 2) | (_sm.n_long >= 1)]          # a cell is a STAND if aircraft park there repeatedly or for a long time
    STAND = set(zip(_sm.airport.astype(str), _sm.clat.astype(int), _sm.clon.astype(int)))
except Exception:
    STAND = set()
TKO_MATCH = os.environ.get("TKO_MATCH", "0") == "1"   # takeoff-event matcher: off by default (corr(y, ADS-B taxi) 0.02)
ICAO_LINK = os.environ.get("ICAO_LINK", "0") == "1"   # join through the aircraft (ICAO24) when the callsign join finds no ground samples
LIFTOFF_LINK = os.environ.get("LIFTOFF_LINK", "0") == "1"   # departures still without ground samples: the unique unclaimed aircraft lifting off within +-60 s
STAND_POS = os.environ.get("STAND_POS", "")             # parquet (airport, stand, lat, lon) of assigned-stand positions learned on 2025 (prc.adsb.stand_pos)
SP = {}
if STAND_POS and os.path.exists(STAND_POS):
    _sp = pd.read_parquet(STAND_POS); SP = {(str(r.airport), str(r.stand)): (float(r.lat), float(r.lon)) for r in _sp.itertuples()}
OUT_PATH = os.environ.get("ADSB_OUT", "outputs/adsb_feats.parquet")
log(f"stand map: {len(STAND)} cells")
t0 = time.time(); days = sorted(pathlib.Path(p).stem for p in glob.glob("data/adsb_lol/*.parquet") if ".slow" not in p)
if os.environ.get("ADSB_DAYS"): _keep = set(open(os.environ["ADSB_DAYS"]).read().split()); days = [x for x in days if x in _keep]   # reproduce a fixed day set
ICAO_TRIM = os.environ.get("ICAO_TRIM", "0") == "1"   # ICAO-linked tracks: start at the LAST long gate stop (>=600 s stationary) before takeoff, dropping the inbound taxi
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "STAND_mvt"]
dep = pd.concat([pd.read_parquet(p, columns=cols) for p in C.train_files()] + [C.read_target(cols)], ignore_index=True)
dep = dep[dep.PHASE_mvt == "DEP"].copy()
for c in ["MVT_TIME_UTC_mvt", "AOBT_3_flt"]: dep[c] = dep[c].dt.tz_convert(None)
dep["day"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
out = []
for day in days:
    a_all = pd.read_parquet(f"data/adsb_lol/{day}.parquet")
    a = a_all[a_all.callsign.notna()].copy(); a["cs"] = a.callsign.astype("string").str.strip().str.upper(); a = a[a.cs.ne("NONE") & a.cs.ne("NAN") & a.cs.ne("")]
    prev_day = (pd.Timestamp(day) - pd.Timedelta("1D")).strftime("%Y-%m-%d")
    if pathlib.Path(f"data/adsb_lol/{prev_day}.parquet").exists():   # departures shortly after midnight need the previous day's late samples
        b = pd.read_parquet(f"data/adsb_lol/{prev_day}.parquet"); b = b[b.callsign.notna()]; b["cs"] = b.callsign.astype("string").str.strip().str.upper(); b = b[b.t >= pd.Timestamp(day) - pd.Timedelta("3h")]; a = pd.concat([a, b], ignore_index=True)
    d = dep[dep.day == day].copy()
    if d.empty: continue
    d["cs1"] = d.FLIGHT_mvt.astype("string").str.strip().str.upper(); d["cs2"] = d.CALLSIGN_flt.astype("string").str.strip().str.upper()
    # rows without a flight-plan record carry only the airport's IATA flight number (KL1587); ADS-B broadcasts the ICAO callsign
    # (KLM1587): translate the airline prefix with the map learned from 2025 rows that have both fields (configs/iata2icao.json)
    m3 = d.cs1.str.extract(r"^([A-Z0-9]{2})(\d{1,4})([A-Z]?)$"); d["cs3"] = (m3[0].map(I2I) + m3[1].str.lstrip("0") + m3[2]).where(d.cs2.isna())
    key = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
    long = pd.concat([d[key + [c]].rename(columns={c: "cs"}) for c in ["cs1", "cs2", "cs3"]]).dropna(subset=["cs"]).drop_duplicates(["MVT_ID_mvt", "cs"])
    long = long[~long.cs.isin(["NAN", "NONE", "", "<NA>"])]   # NA keys would merge with NA keys (pandas) -> every callsign-less sample
    j = long.merge(a[["cs", "airport", "t", "on_ground", "alt", "gs", "lat", "lon", "icao"]], left_on=["cs", "ADEP_mvt"], right_on=["cs", "airport"], how="inner")
    j = j[(j.t >= j.MVT_TIME_UTC_mvt - pd.Timedelta("3h")) & (j.t <= j.MVT_TIME_UTC_mvt)]
    j["join_type"] = 1
    if ICAO_LINK:
        # Ground transponders often broadcast a blank callsign, or the stale INBOUND flight number until just before takeoff,
        # so the callsign join misses the taxi track although the aircraft is seen. For departures whose direct join has NO
        # ground sample: find the aircraft (icao) that carries the departure's callsign in [tko-3h, tko+20min] at this airport;
        # if exactly one, take ALL its samples in [tko-3h, tko] (any callsign) and keep the final segment after the last
        # >600 s gap, which excludes the same aircraft's inbound leg (the existing guard of the takeoff-event matcher).
        # Rows already covered by the direct join are left untouched, so their features are byte-identical.
        has_ground = set(j.loc[j.on_ground == 1, "MVT_ID_mvt"])
        need3 = long[~long.MVT_ID_mvt.isin(has_ground)]
        if len(need3):
            lk = need3.merge(a[["cs", "airport", "t", "icao"]], left_on=["cs", "ADEP_mvt"], right_on=["cs", "airport"], how="inner")
            lk = lk[(lk.t >= lk.MVT_TIME_UTC_mvt - pd.Timedelta("3h")) & (lk.t <= lk.MVT_TIME_UTC_mvt + pd.Timedelta("20min"))]
            nic = lk.groupby("MVT_ID_mvt").icao.nunique()
            lk = lk[lk.MVT_ID_mvt.isin(nic[nic == 1].index)].drop_duplicates("MVT_ID_mvt")[key + ["icao"]]
            if len(lk):
                aa = a_all[a_all.icao.isin(set(lk.icao))][["icao", "airport", "t", "on_ground", "alt", "gs", "lat", "lon"]]
                s3 = lk.merge(aa, left_on=["icao", "ADEP_mvt"], right_on=["icao", "airport"], how="inner")
                s3 = s3[(s3.t >= s3.MVT_TIME_UTC_mvt - pd.Timedelta("3h")) & (s3.t <= s3.MVT_TIME_UTC_mvt)].sort_values(["MVT_ID_mvt", "t"])
                if len(s3):
                    gap = s3.groupby("MVT_ID_mvt").t.diff().dt.total_seconds() > 600
                    seg = gap.groupby(s3.MVT_ID_mvt).cumsum()
                    s3 = s3[seg == seg.groupby(s3.MVT_ID_mvt).transform("max")]
                    if ICAO_TRIM and len(s3):
                        # parked aircraft keep broadcasting, so no >600 s gap separates inbound taxi / gate stop / outbound taxi:
                        # keep the track from the START of the last stationary run (on ground, gs<=2) lasting >=600 s that ends >=60 s before takeoff
                        s3 = s3.sort_values(["MVT_ID_mvt", "t"]).reset_index(drop=True)
                        st = (s3.on_ground == 1) & (s3.gs.fillna(0) <= 2)
                        rid = (st != st.groupby(s3.MVT_ID_mvt).shift()).groupby(s3.MVT_ID_mvt).cumsum()
                        R_ = s3[st].assign(rid=rid[st]).groupby(["MVT_ID_mvt", "rid"]).agg(t0=("t", "min"), t1=("t", "max"), tko=("MVT_TIME_UTC_mvt", "first")).reset_index()
                        R_ = R_[((R_.t1 - R_.t0).dt.total_seconds() >= 600) & ((R_.tko - R_.t1).dt.total_seconds() >= 60)]
                        cut = R_.groupby("MVT_ID_mvt").t0.max()
                        if len(cut):
                            c_ = s3.MVT_ID_mvt.map(cut); keep_ = c_.isna().to_numpy().copy(); hv = ~keep_
                            keep_[hv] = (s3.t[hv] >= pd.to_datetime(c_[hv])).to_numpy(); s3 = s3[keep_]
                    s3 = s3[s3.groupby("MVT_ID_mvt").on_ground.transform("max") == 1]     # only link if it actually brings ground samples
                    s3["cs"] = "ICAO:" + s3.icao.astype(str); s3["join_type"] = 3
                    j = pd.concat([j[~j.MVT_ID_mvt.isin(set(s3.MVT_ID_mvt))], s3[j.columns]], ignore_index=True)
    if LIFTOFF_LINK:
        # Departures whose callsign and address joins bring no ground sample: the aircraft that lifts off at this airport within +-60 s of the
        # recorded take-off (last ground sample followed by an airborne sample above 300 ft or 120 kn within 300 s), not claimed by another
        # departure within 15 min, unique (or unique after matching the aircraft type), with at least 20 ground samples in [tko-3h, tko].
        # Track = its final segment after the last > 600 s gap (excludes the inbound leg).
        has_ground = set(j.loc[j.on_ground == 1, "MVT_ID_mvt"]); need4 = d[~d.MVT_ID_mvt.isin(has_ground)]
        if len(need4):
            asrt = a_all.sort_values(["icao", "t"]); gi = asrt.groupby("icao")
            nxt_t, nxt_g, nxt_alt, nxt_gs = gi.t.shift(-1), gi.on_ground.shift(-1), gi.alt.shift(-1), gi.gs.shift(-1)
            ev = asrt[(asrt.on_ground == 1) & (nxt_g == 0) & ((nxt_alt > 300) | (nxt_gs > 120)) & ((nxt_t - asrt.t).dt.total_seconds() <= 300)][["icao", "airport", "t", "type"]]
            used = j[["icao", "MVT_TIME_UTC_mvt"]].drop_duplicates().rename(columns={"MVT_TIME_UTC_mvt": "t_used"})
            ev = ev.merge(used, on="icao", how="left"); ev["taken"] = (ev.t_used - ev.t).abs() <= pd.Timedelta("15min")
            ev = ev[~ev.groupby(["icao", "t"]).taken.transform("any")].drop_duplicates(["icao", "t"])
            cand = need4[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "AIRCRAFT_TYPE_mvt"]].merge(ev, left_on="ADEP_mvt", right_on="airport", how="inner")
            cand = cand[(cand.t - cand.MVT_TIME_UTC_mvt).abs() <= pd.Timedelta("60s")]
            nc = cand.groupby("MVT_ID_mvt").icao.nunique()
            typ = cand[cand.type.astype(str) == cand.AIRCRAFT_TYPE_mvt.astype(str)]; nt = typ.groupby("MVT_ID_mvt").icao.nunique()
            pick = pd.concat([cand[cand.MVT_ID_mvt.isin(nc[nc == 1].index)], typ[typ.MVT_ID_mvt.isin(nt[nt == 1].index) & ~typ.MVT_ID_mvt.isin(nc[nc == 1].index)]])
            pick = pick.drop_duplicates("MVT_ID_mvt")
            pick = pick[~pick.duplicated("icao", keep=False)]          # one aircraft cannot serve two departures
            if len(pick):
                aa = a_all[a_all.icao.isin(set(pick.icao))][["icao", "airport", "t", "on_ground", "alt", "gs", "lat", "lon"]]
                s4 = pick[key + ["icao"]].merge(aa, left_on=["icao", "ADEP_mvt"], right_on=["icao", "airport"], how="inner")
                s4 = s4[(s4.t >= s4.MVT_TIME_UTC_mvt - pd.Timedelta("3h")) & (s4.t <= s4.MVT_TIME_UTC_mvt)].sort_values(["MVT_ID_mvt", "t"])
                if len(s4):
                    gap = s4.groupby("MVT_ID_mvt").t.diff().dt.total_seconds() > 600; seg = gap.groupby(s4.MVT_ID_mvt).cumsum()
                    s4 = s4[seg == seg.groupby(s4.MVT_ID_mvt).transform("max")]
                    s4 = s4[s4.groupby("MVT_ID_mvt").on_ground.transform("sum") >= 20]
                    s4["cs"] = "LIFTOFF:" + s4.icao.astype(str); s4["join_type"] = 4
                    j = pd.concat([j[~j.MVT_ID_mvt.isin(set(s4.MVT_ID_mvt))], s4[j.columns]], ignore_index=True)
    # --- second pass: unmatched rows (no flight plan) whose callsign never joined -> takeoff-EVENT match on the aircraft (icao):
    # the aircraft whose last ground sample is within [tko-150 s, tko+60 s] and which is airborne within 300 s after; unique, or
    # unique after matching the aircraft type. Track = that icao's final segment (after the last > 600 s gap = the gate stop),
    # which excludes the same aircraft's inbound leg. Validated on 2025 block times: corr(y, ADS-B taxi) 0.63, median |err| 186 s.
    need = d[d.cs2.isna() & ~d.MVT_ID_mvt.isin(j.MVT_ID_mvt.unique()) & d.AIRCRAFT_TYPE_mvt.notna()] if "AIRCRAFT_TYPE_mvt" in d.columns else d.iloc[0:0]
    if len(need) and TKO_MATCH:
        asrt = a.sort_values(["icao", "t"]); nxt_t = asrt.groupby("icao").t.shift(-1); nxt_g = asrt.groupby("icao").on_ground.shift(-1); nxt_alt = asrt.groupby("icao").alt.shift(-1); nxt_gs = asrt.groupby("icao").gs.shift(-1)
        ev = asrt[(asrt.on_ground == 1) & (nxt_g == 0) & ((nxt_alt > 300) | (nxt_gs > 120)) & ((nxt_t - asrt.t).dt.total_seconds() <= 300)][["icao", "airport", "t", "type"]]
        # guard: an aircraft whose callsign already joined to some departure taking off within 15 min belongs to THAT departure
        used = j[["icao", "MVT_TIME_UTC_mvt"]].drop_duplicates(); ev = ev.merge(used.rename(columns={"MVT_TIME_UTC_mvt": "t_used"}), on="icao", how="left")
        ev["taken"] = (ev.t_used - ev.t).abs() <= pd.Timedelta("15min"); ev = ev[~ev.groupby(["icao", "t"]).taken.transform("any")].drop_duplicates(["icao", "t"])
        extra = []
        for _, r in need.iterrows():
            e = ev[(ev.airport == r.ADEP_mvt) & (ev.t >= r.MVT_TIME_UTC_mvt - pd.Timedelta("90s")) & (ev.t <= r.MVT_TIME_UTC_mvt + pd.Timedelta("45s"))]
            if len(e) > 1: e2 = e[e.type.astype(str) == str(r.AIRCRAFT_TYPE_mvt)]; e = e2 if len(e2) == 1 else e
            if len(e) != 1: continue
            s_ = a[(a.icao == e.icao.iloc[0]) & (a.airport == r.ADEP_mvt) & (a.t >= r.MVT_TIME_UTC_mvt - pd.Timedelta("3h")) & (a.t <= r.MVT_TIME_UTC_mvt)].sort_values("t")
            if s_.empty: continue
            seg = (s_.t.diff().dt.total_seconds() > 600).cumsum(); s_ = s_[seg == seg.iloc[-1]].copy()
            s_["MVT_ID_mvt"] = r.MVT_ID_mvt; s_["ADEP_mvt"] = r.ADEP_mvt; s_["MVT_TIME_UTC_mvt"] = r.MVT_TIME_UTC_mvt; s_["AOBT_3_flt"] = r.AOBT_3_flt; s_["cs"] = "ICAO:" + str(e.icao.iloc[0]); s_["join_type"] = 2; extra.append(s_[j.columns])
        if extra: j = pd.concat([j] + extra, ignore_index=True)
    if j.empty:
        f = d[["MVT_ID_mvt"]].copy(); f["ads_cov"] = 0; out.append(f); continue
    j["rel"] = (j.t - j.MVT_TIME_UTC_mvt).dt.total_seconds(); j["gs"] = j.gs.fillna(0); j["ground"] = j.on_ground == 1
    j["still"] = j.ground & (j.gs <= 2); j["mov"] = j.ground & (j.gs > 5); j["fast"] = j.ground & (j.gs > 15); j["slow"] = j.ground & (j.gs >= 1)
    j = j.sort_values(["MVT_ID_mvt", "t"]); first = j.groupby("MVT_ID_mvt")[["lat", "lon"]].transform("first")
    j["disp_m"] = np.sqrt(((j.lat - first.lat) * 111320) ** 2 + ((j.lon - first.lon) * 111320 * np.cos(np.radians(j.lat))) ** 2); j["moved"] = j.ground & (j.disp_m > 15)
    g = j.groupby("MVT_ID_mvt")
    f = pd.DataFrame({"ads_first_seen": g.rel.min(), "ads_n_ground": g.ground.sum(), "ads_n_still": g.still.sum(), "ads_max_gs": g.gs.max(),
                      "ads_first_ground": j[j.ground].groupby("MVT_ID_mvt").rel.min(), "ads_first_mov": j[j.mov].groupby("MVT_ID_mvt").rel.min(), "ads_first_fast": j[j.fast].groupby("MVT_ID_mvt").rel.min(),
                      "ads_first_slow": j[j.slow].groupby("MVT_ID_mvt").rel.min(), "ads_first_disp": j[j.moved].groupby("MVT_ID_mvt").rel.min(),
                      "ads_stop_after_mov": j[j.still].groupby("MVT_ID_mvt").rel.max(), "ads_n_stop_samples_after": None})
    fm = f["ads_first_mov"]; st = j[j.still].merge(fm.rename("fm"), left_on="MVT_ID_mvt", right_index=True, how="left"); st = st[st.rel > st.fm]
    f["ads_n_stop_samples_after"] = st.groupby("MVT_ID_mvt").size(); f["ads_stop_after_mov"] = st.groupby("MVT_ID_mvt").rel.max()
    f["ads_last_stop_end"] = f["ads_stop_after_mov"]
    # dwell end: last stationary sample before the first moving sample
    jm = j.merge(f[["ads_first_mov"]], left_on="MVT_ID_mvt", right_index=True, how="left"); de = jm[jm.still & (jm.rel < jm.ads_first_mov.fillna(0))].groupby("MVT_ID_mvt").rel.max(); f["ads_dwell_end"] = de
    # multipath guard: a movement must be SUSTAINED into the next sample (<=300 s later) to count.
    j2 = j.sort_values(["MVT_ID_mvt", "rel"]); g2 = j2.groupby("MVT_ID_mvt")
    nxt_rel = g2.rel.shift(-1); gap_ok = (nxt_rel - j2.rel) <= 300
    j2["mov_s"] = j2.mov & g2.mov.shift(-1).fillna(False) & gap_ok
    j2["disp_s"] = j2.moved & g2.moved.shift(-1).fillna(False) & gap_ok
    f["ads_first_mov_sust"] = j2[j2.mov_s].groupby("MVT_ID_mvt").rel.min()
    f["ads_first_disp_sust"] = j2[j2.disp_s].groupby("MVT_ID_mvt").rel.min()
    # jitter amplitude while still parked (before the first sustained movement): multipath scale at this stand
    jm = j2.merge(f[["ads_first_mov_sust"]], left_on="MVT_ID_mvt", right_index=True, how="left")
    f["ads_jitter_m"] = jm[jm.rel < jm.ads_first_mov_sust.fillna(-1e9)].groupby("MVT_ID_mvt").disp_m.max()
    # taxi-path descriptors between the first sustained movement and takeoff (the real ground run)
    jm2 = j2.merge(f[["ads_first_mov_sust"]], left_on="MVT_ID_mvt", right_index=True, how="left")
    tx = jm2[jm2.ground & (jm2.rel >= jm2.ads_first_mov_sust.fillna(1e9))].copy()
    if len(tx):
        gtx = tx.groupby("MVT_ID_mvt"); dlat = gtx.lat.diff(); dlon = gtx.lon.diff()
        tx["step_m"] = np.sqrt((dlat * 111320) ** 2 + (dlon * 111320 * np.cos(np.radians(tx.lat))) ** 2).fillna(0.0)
        tx["dt"] = gtx.rel.diff().fillna(0.0); gtx = tx.groupby("MVT_ID_mvt")
        f["ads_path_m"] = gtx.step_m.sum()
        f["ads_taxi_stop_s"] = tx[tx.gs <= 2].groupby("MVT_ID_mvt").dt.sum()
        f["ads_taxi_gs_mean"] = tx[tx.gs > 2].groupby("MVT_ID_mvt").gs.mean()
        f["ads_taxi_gs_max"] = gtx.gs.max()
        f["ads_taxi_n_stops"] = tx.assign(st=(tx.gs <= 2) & (gtx.gs.shift(1) > 2)).groupby("MVT_ID_mvt").st.sum()
    # pushback = end of the LAST long stationary episode (>= 900 s) before takeoff: robust to earlier same-callsign movements
    # (taxi-in of the same rotation, reused GA callsigns) and to short runway holds.
    j3 = j2.copy(); j3["still_i"] = j3.still.astype(int); j3["grp"] = (j3.still_i != j3.groupby("MVT_ID_mvt").still_i.shift()).cumsum()
    ep = j3[j3.still].groupby(["MVT_ID_mvt", "grp"]).rel.agg(["min", "max"]).reset_index(); ep["dur"] = ep["max"] - ep["min"]
    if STAND:
        pos = j3[j3.still].groupby(["MVT_ID_mvt", "grp"]).agg(la=("lat", "median"), lo=("lon", "median"), ns=("rel", "size")).reset_index()
        epp = ep.merge(pos, on=["MVT_ID_mvt", "grp"], how="left")
        epp = epp.merge(d[["MVT_ID_mvt", "ADEP_mvt"]], on="MVT_ID_mvt", how="left")
        cl = np.round(epp.la.to_numpy() / LAT_CELL); co = np.round(epp.lo.to_numpy() / LON_CELL)
        ok = ~(np.isnan(cl) | np.isnan(co))
        keys = [(a_, int(x), int(y)) if o else None for a_, x, y, o in zip(epp.ADEP_mvt.astype(str), np.nan_to_num(cl), np.nan_to_num(co), ok)]
        epp["at_stand"] = [k is not None and k in STAND for k in keys]
        for T in (0, 60, 180):
            sv = epp[epp.at_stand & (epp.dur >= T)].sort_values(["MVT_ID_mvt", "max"]).groupby("MVT_ID_mvt").last()
            f[f"ads_sv{T}"] = sv["max"]; f[f"ads_sv{T}_dur"] = sv["dur"]; f[f"ads_sv{T}_n"] = sv["ns"]
        f["ads_n_stand_ep"] = epp[epp.at_stand].groupby("MVT_ID_mvt").size()
    ep_long = ep[ep.dur >= 900].sort_values(["MVT_ID_mvt", "max"]).groupby("MVT_ID_mvt").last()
    f["ads_stand_leave"] = ep_long["max"]; f["ads_stand_dwell_s"] = ep_long["dur"]
    # relaxed dwell thresholds: transponders often power up shortly before pushback, so a 900 s stationary
    # episode is rare. Record the end of the last stationary episode of at least T seconds, for several T,
    # plus the dwell length and sample count so a model can judge reliability.
    for T in (30, 60, 120, 240, 480):
        e2 = ep[ep.dur >= T].sort_values(["MVT_ID_mvt", "max"]).groupby("MVT_ID_mvt").last()
        f[f"ads_sl{T}"] = e2["max"]; f[f"ads_sl{T}_dur"] = e2["dur"]
    cnt = j3[j3.still].groupby(["MVT_ID_mvt", "grp"]).size().rename("nn").reset_index()
    ep2 = ep.merge(cnt, on=["MVT_ID_mvt", "grp"], how="left")
    e30 = ep2[ep2.dur >= 30].sort_values(["MVT_ID_mvt", "max"]).groupby("MVT_ID_mvt").last()
    f["ads_sl30_n"] = e30["nn"]
    ep_any = ep.sort_values(["MVT_ID_mvt", "max"]).groupby("MVT_ID_mvt").last(); f["ads_last_still_end"] = ep_any["max"]; f["ads_last_still_dur"] = ep_any["dur"]
    # first movement AFTER the long dwell (the pushback proper) and taxi duration measured from it
    jm3 = j3.merge(f[["ads_stand_leave"]], left_on="MVT_ID_mvt", right_index=True, how="left")
    f["ads_mov_after_dwell"] = jm3[jm3.mov & (jm3.rel >= jm3.ads_stand_leave.fillna(1e9))].groupby("MVT_ID_mvt").rel.min()
    # parking position: median position of the stationary samples of the stop that ends at the dwell end (learning the stand positions)
    jp = j.merge(f[["ads_dwell_end"]], left_on="MVT_ID_mvt", right_index=True, how="left")
    pk = jp[jp.still & (jp.rel <= jp.ads_dwell_end) & (jp.rel >= jp.ads_dwell_end - 600)].groupby("MVT_ID_mvt")
    f["ads_park_lat"] = pk.lat.median(); f["ads_park_lon"] = pk.lon.median(); f["ads_park_n"] = pk.size()
    if SP:
        # assigned stand: samples within 40 m of the departure's assigned stand position; the last of them before the aircraft is next
        # seen more than 80 m away is the moment it left the stand
        st_ = d[["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt"]].copy(); st_["k"] = list(zip(st_.ADEP_mvt.astype(str), st_.STAND_mvt.astype(str)))
        st_["s_lat"] = st_.k.map(lambda k: SP.get(k, (np.nan, np.nan))[0]); st_["s_lon"] = st_.k.map(lambda k: SP.get(k, (np.nan, np.nan))[1])
        js = j[j.ground].merge(st_[["MVT_ID_mvt", "s_lat", "s_lon"]].dropna(), on="MVT_ID_mvt", how="inner")
        if len(js):
            js["d_st"] = np.sqrt(((js.lat - js.s_lat) * 111320) ** 2 + ((js.lon - js.s_lon) * 111320 * np.cos(np.radians(js.lat))) ** 2)
            at = js[js.d_st <= 40]; ga = at.groupby("MVT_ID_mvt")
            f["ads_as_n"] = ga.size(); f["ads_as_first"] = ga.rel.min(); f["ads_as_last"] = ga.rel.max(); f["ads_as_dmin"] = js.groupby("MVT_ID_mvt").d_st.min()
            away = js[js.d_st > 80].merge(f[["ads_as_first"]], left_on="MVT_ID_mvt", right_index=True, how="inner")
            away = away[away.rel > away.ads_as_first]; f["ads_as_away"] = away.groupby("MVT_ID_mvt").rel.min()
            lv = at.merge(f[["ads_as_away"]], left_on="MVT_ID_mvt", right_index=True, how="left")
            f["ads_as_leave"] = lv[lv.rel < lv.ads_as_away.fillna(1e9)].groupby("MVT_ID_mvt").rel.max()
    # track validity: does the joined track actually reach this takeoff? (unmatched rows: reused/mistranslated callsigns)
    f["ads_last_rel"] = j.groupby("MVT_ID_mvt").rel.max(); f["ads_last_ground_rel"] = j[j.ground].groupby("MVT_ID_mvt").rel.max()
    f["ads_span"] = f.ads_last_rel - f.ads_first_seen; f["ads_n_icao"] = j.groupby("MVT_ID_mvt").icao.nunique(); f["ads_n_cs"] = j.groupby("MVT_ID_mvt").cs.nunique()
    f["ads_join_type"] = j.groupby("MVT_ID_mvt").join_type.max()
    f["ads_cov"] = np.where(f.ads_n_still > 0, 3, np.where(f.ads_n_ground > 0, 2, 1)); f = f.reset_index()
    f = d[["MVT_ID_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt"]].merge(f, on="MVT_ID_mvt", how="left"); f["ads_cov"] = f.ads_cov.fillna(0).astype(int)
    aob = (f.AOBT_3_flt - f.MVT_TIME_UTC_mvt).dt.total_seconds()   # AOBT relative to takeoff (negative)
    f["ads_dwell_minus_aobt"] = f.ads_dwell_end - aob; f["ads_mov_minus_aobt"] = f.ads_first_mov - aob
    f["ads_slow_minus_aobt"] = f.ads_first_slow - aob; f["ads_disp_minus_aobt"] = f.ads_first_disp - aob
    f["ads_movsust_minus_aobt"] = f.ads_first_mov_sust - aob; f["ads_dispsust_minus_aobt"] = f.ads_first_disp_sust - aob
    f["ads_standleave_minus_aobt"] = f.ads_stand_leave - aob
    if "ads_as_leave" in f: f["ads_asleave_minus_aobt"] = f.ads_as_leave - aob; f["ads_aslast_minus_aobt"] = f.ads_as_last - aob
    for T in (0, 60, 180):
        if f"ads_sv{T}" in f: f[f"ads_sv{T}_minus_aobt"] = f[f"ads_sv{T}"] - aob; f["ads_movafter_minus_aobt"] = f.ads_mov_after_dwell - aob
    out.append(f.drop(columns=["AOBT_3_flt", "MVT_TIME_UTC_mvt"]))
    log(f"{day}: deps {len(d)}, cov>=2 {(f.ads_cov >= 2).mean():.2f}, cov 3 {(f.ads_cov == 3).mean():.2f} ({time.time() - t0:.0f}s)")
F = pd.concat(out, ignore_index=True)
F = F.merge(dep[["MVT_ID_mvt", "ADEP_mvt", "day"]], on="MVT_ID_mvt", how="left")
F["ads_apt_day_cov"] = F.groupby(["ADEP_mvt", "day"]).ads_cov.transform(lambda s: (s >= 2).mean()).astype(np.float32); F = F.drop(columns=["ADEP_mvt", "day"])
F.to_parquet(OUT_PATH, index=False); log(f"saved {OUT_PATH} rows {len(F)} days {len(days)} ({time.time() - t0:.0f}s)")
