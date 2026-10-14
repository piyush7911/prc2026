"""Collect the numbers behind the paper's data figures into docs/paper/figure_data.json (small, committed), so that
docs/paper/make_figures.py can redraw every figure without the large inputs.
Needs the pipeline outputs (outputs/final), the organisers' 2025 files and the adsb.lol day of the example departure.
Usage (repository root): PYTHONPATH=src python docs/paper/figure_data.py"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from prc.models import pipeline as P
from prc.models import daily as DL

OUT = "docs/paper/figure_data.json"
D = os.environ.get("PRC_DATA", "data")   # same setting as src/prc/config.py
EXAMPLE = dict(mvt_id=190517528, month_file=f"{D}/training_2025-06-01_2025-07-01.parquet", adsb_day=f"{D}/adsb_lol/2025-06-09.parquet", airport="LSZH")
data = {}

# 1. concentration of the final 2025 out-of-fold squared error (five corrupt records excluded), and the lane sizes of the 2026 file
f, _, _ = P.load(); lab = f[~f.is_rank].reset_index(drop=True); ok = P.good(lab)
q, c = P._assemble(lab, "oof"); e2 = np.sort(((lab.y.to_numpy() - q) ** 2)[ok])[::-1]; cs = np.cumsum(e2) / e2.sum()
ks = np.unique(np.round(np.logspace(0, np.log10(len(e2)), 120)).astype(int))
data["concentration"] = {"n_rows": int(len(e2)), "k": ks.tolist(), "share": [float(cs[k - 1]) for k in ks],
                         "top": {str(k): float(cs[k - 1]) for k in (100, 1000, 10000)}}

# 2. identity between EUROCONTROL's daily total and the labels (2025, per airport-day)
t = DL.frame(lab, q, c["lane"]); t["c_true"] = np.maximum(0.0, t.d - t.y)
g = t.groupby(["apt", "date"]).c_true.sum().reset_index().merge(DL.load_K(), on=["apt", "date"])
data["identity"] = {"apt": g.apt.tolist(), "label_sum_min": (g.c_true / 60).round(1).tolist(), "published_min": (g.K / 60).round(1).tolist(),
                    "corr": float(np.corrcoef(g.K, g.c_true)[0, 1]), "median_ratio": float(np.nanmedian(g.K / g.c_true.replace(0, np.nan)))}
J = json.load(open(f"{P.OUT}/daily.json")); dd = J["deltas"][f"lam{J['lam']}_phi{J['phi']}"]
data["daily_gain"] = {m: dd[m] for m in ("1", "2", "6", "7", "four")}

# 3. the example departure: clocks relative to take-off and its adsb.lol ground speed (anonymised: no flight number, callsign or aircraft)
r = pd.read_parquet(EXAMPLE["month_file"]); r = r[r.MVT_ID_mvt == EXAMPLE["mvt_id"]].iloc[0]
tko = r.MVT_TIME_UTC_mvt
rel = lambda ts: float((ts - tko).total_seconds())
data["example"] = {"airport": EXAMPLE["airport"], "month": "June 2025", "takeoff_utc": tko.strftime("%H:%M:%S"),
                   "clocks_s": {"schedule": rel(r.SCHED_TIME_UTC_mvt), "EOBT1": rel(r.EOBT_1_flt), "LOBT": rel(r.LOBT_flt), "IOBT": rel(r.IOBT_flt),
                                "AOBT3": rel(r.AOBT_3_flt), "recorded_offblock": rel(r.BLOCK_TIME_UTC_mvt)}}
fe = f.set_index("MVT_ID_mvt").loc[EXAMPLE["mvt_id"], ["ads_first_seen", "ads_first_ground", "ads_dwell_end", "ads_as_leave", "ads_first_mov", "ads_first_fast"]]
data["example"]["adsb_events_s"] = {k: (None if pd.isna(v) else float(v)) for k, v in fe.items()}
a = pd.read_parquet(EXAMPLE["adsb_day"]); cs_ = str(r.CALLSIGN_flt).strip()
s = a[(a.airport == EXAMPLE["airport"]) & (a.callsign.str.strip() == cs_)].copy()
s["rel"] = (s.t - tko.tz_localize(None)).dt.total_seconds(); s = s[(s.rel > -1900) & (s.rel < 30)].sort_values("rel")
data["example"]["track"] = {"t_s": s.rel.round(1).tolist(), "gs_kn": s.gs.round(1).tolist(), "on_ground": s.on_ground.astype(int).tolist()}
data["example"]["y_s"] = float(r.TAXITIME_SEC_mvt)

# 4. 2026 lane sizes and the 2025 per-month lane RMSE (validation report)
v = json.load(open(f"{P.OUT}/validation_2025.json"))
data["lanes_rmse"] = {m: {l: v[m][l] for l in ("covered", "uncovered", "unm_nr", "unm_rome")} for m in ("1", "2", "6", "7")}
rk = f[f.is_rank]; data["lanes_2026"] = pd.Series(P.lanes(rk)).value_counts().to_dict()

os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(data, open(OUT, "w"))
print(f"wrote {OUT}: concentration top100/1k/10k {[round(data['concentration']['top'][k], 3) for k in ('100', '1000', '10000')]}, "
      f"identity corr {data['identity']['corr']:.4f} median ratio {data['identity']['median_ratio']:.4f} on {len(g)} airport-days")
