"""Numbers for the paper's appendix: baseline (NM clock anchor alone), feature groups with counts, and LightGBM gain importance of the base
learner (lr 0.05) and of the ADS-B stacker, both fitted on all 2025 rows with the pipeline's settings. Writes docs/paper/feature_importance.json.
Usage (repository root): PYTHONPATH=src python docs/paper/feature_importance.py [--groups-only]"""
import json, re, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "src")
from prc.models import pipeline as P

GROUPS = [  # (name, regex) in order of precedence
    ("ADS-B ground-track timing", r"^ads_"),
    ("recording-defect state", r"^(arr_defect|arr_degr|dep_degr|degr_|defect|mvt_second|sched_second|sched_min_mod)"),
    ("NM clocks and clock gaps", r"^(naive|aobt3|tko_minus_|eobt1_|lobt_|iobt_|sched_minus|arvt3_|.*_minus_(sched|eobt1|lobt|iobt|aobt3))"),
    ("target encodings", r"^(te_|ter_)"),
    ("weather and de-icing", r"^(wx_|atmap|freeze|frost|cold|snow|deice|remote_deice|is_snow|visib|wind|temp|crosswind|headwind|dp_spread)"),
    ("neighbour NM clocks", r"^nb_"),
    ("runway queue and traffic", r"^(dep_|rwy_|hourly_|day_|sched_dep|q_|taxiing_|push_|n_dep|n_arr|min_since|headway|hw_|op_dep|same_ades|frac_long_taxi)"),
    ("arrivals, stands and references", r"^(arr_|prev_arr|stand_|unimp_|overnight|parked)"),
    ("calendar", r"^(month|hour|minute_of_day|dow|is_holiday|near_holiday|is_weekend|week|season)"),
    ("flight and route", r"^(planned_dur|actual_dur|m3_taxi|domestic|diverted)"),
    ("aircraft, operator, stand and runway identity", r"^(apt|rwy|stand|actype|wtc|operator|airline|segment|ftype|frule|ades|flt_|actype_)"),
]
def group(c):
    for name, rx in GROUPS:
        if re.match(rx, c): return name
    return "other"

f, fb, fa = P.load(); lab = f[~f.is_rank].reset_index(drop=True); out = {}
gb = pd.Series({c: group(c) for c in fb}); ga = pd.Series({c: group(c) for c in fa})
out["groups_base"] = gb.value_counts().to_dict(); out["groups_stacker"] = ga.value_counts().to_dict()
out["n_base"] = len(fb); out["n_stacker"] = len(fa) + 1
print("base features", len(fb), out["groups_base"]); print("stacker features", len(fa) + 1)
print("other:", [c for c in fa if group(c) == "other"])
if "--groups-only" in sys.argv: sys.exit()

# baseline: the anchor b alone (best plausible NM clock, schedule for LIRF rows without NM initial time)
y = lab.y.to_numpy(); mo = lab.month.to_numpy(); ok = P.good(lab); b = P.base_zf(lab)
q, c = P._assemble(lab, "oof"); normal = ok & (np.abs(y - c["base"]) < 3600) & (y < 20000)
r = lambda e: float(np.sqrt(np.mean(e ** 2)))
out["baseline"] = {str(m): r((y - b)[ok & (mo == m)]) for m in (1, 2, 6, 7)}
s4 = ok & np.isin(mo, (1, 2, 6, 7)); out["baseline"]["four"] = r((y - b)[s4]); out["baseline"]["normal_four"] = r((y - b)[s4 & normal])
print("baseline", {k: round(v, 2) for k, v in out["baseline"].items()})

def importance(model, cols):
    g = pd.Series(model.feature_importance("gain"), index=cols); g = g / g.sum()
    return {"top": [(k, float(v)) for k, v in g.sort_values(ascending=False).head(15).items()],
            "by_group": {k: float(v) for k, v in g.groupby(pd.Series({c: group(c) for c in cols})).sum().sort_values(ascending=False).items()}}
cats = [k for k in P.CATS if k in fb]
tr = lab[lab.y >= 0]; t = tr.y.to_numpy() - P.base_zf(tr)
m = lgb.train(dict(P.HP, learning_rate=0.05, seed=0), lgb.Dataset(tr[fb], t, categorical_feature=cats), 760)
out["importance_base"] = importance(m, fb); print("base done", out["importance_base"]["by_group"])
lab["base"] = c["base"]; lab["lane"] = c["lane"]; cov = lab[(lab.lane == "covered") & P.good(lab)]
FE = fa + ["base"]; cats_a = [k for k in P.CATS if k in fa]
m2 = lgb.train(dict(P.STK, seed=0), lgb.Dataset(cov[FE], (cov.y - cov.base).to_numpy(), categorical_feature=cats_a), P.STK_ROUNDS)
out["importance_stacker"] = importance(m2, FE); print("stacker done", out["importance_stacker"]["by_group"])
json.dump(out, open("docs/paper/feature_importance.json", "w"), indent=1)
