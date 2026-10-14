"""Daily reconciliation with EUROCONTROL's published pre-departure delay (all_pre_departure_delays_YYYY.csv, ansperformance.eu).

For each airport and take-off date the file gives DLY_ALL_PRE_2 in minutes. On 2025 labels, K = 60 * DLY_ALL_PRE_2 equals the day's sum of
max(0, off-block - schedule) over departures (correlation 0.9996, median ratio 1.0001). With d = takeoff - schedule known for every row and
y = takeoff - off-block, a flight contributes c = max(0, d - y). The day total therefore constrains the predictions jointly:

  * ambiguous Rome rows (LIRF, d > 1 h): the off-block was recorded either at the schedule (y = d, c = 0) or after a real hold at the gate
    (y ~ tau, c = d - tau). Their prior probability of the first mode follows from the incoming prediction p = pi d + (1 - pi) tau.
    The posterior over the 2^m mode combinations of the day (exact; m capped at 12) weighs each combination by a Student-t likelihood
    (df 3, variance widened by phi) of the day's remaining discrepancy, so a row moves only when the day total is decisive.
  * every other row whose prediction implies a delay (p < d): prior y ~ N(p, v), v the 2025 out-of-fold error variance of its class; the day's
    remaining discrepancy is shared by a Kalman update, row gain v / (sum v + s2), shrunk by lambda. The expected gap that the convexity of max(0, .) creates between
    the true and the predicted day sum (fitted per airport on 2025) is removed first.

Fitted on 2025 only (per-airport offset and noise of the published total against our rows, class variances, tau, bias); lambda and phi are
chosen by the gate (prc.models.pipeline daily); validated by refitting
without each held-out fold. The 2026 take-off dither (+-12 s per row) is added to the noise.
"""
import json, os
import numpy as np, pandas as pd

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
EC_DIR = os.environ.get("PRC_EC_DIR", "data/eurocontrol_pru")
EC_FILES = {2025: "all_pre_departure_delays_2025.csv", 2026: "all_pre_departure_delays_2026_latest.csv"}
DBANDS = [-1e9, 900, 1800, 3600, 7200, 1e12]
M_MAX = 12

def load_K():
    """Published daily totals K = 60 * DLY_ALL_PRE_2 (seconds) per airport and date, 2025 and 2026 files."""
    out = []
    for yr, fn in EC_FILES.items():
        e = pd.read_csv(os.path.join(EC_DIR, fn)); e = e[e.APT_ICAO.isin(APTS)].dropna(subset=["DLY_ALL_PRE_2"])
        out.append(pd.DataFrame({"apt": e.APT_ICAO.values, "date": e.FLT_DATE.values, "K": 60.0 * e.DLY_ALL_PRE_2.values, "n_ec": e.FLT_DEP_IFR_2.values}))
    return pd.concat(out, ignore_index=True)

def frame(d, p, lanes):
    """Row table: airport, take-off date, d, prediction, lane, class keys."""
    date = (pd.to_datetime(d.year.astype(int).astype(str) + "-01-01") + pd.to_timedelta(d.doy.astype(int) - 1, unit="D")).dt.strftime("%Y-%m-%d")
    t = pd.DataFrame({"apt": d.apt.astype(str).values, "date": date.values, "d": d.tko_minus_sched.to_numpy(float), "p": np.asarray(p, float),
                      "lane": np.asarray(lanes), "month": d.month.to_numpy()})
    if "y" in d: t["y"] = d.y.to_numpy(float)
    t["amb"] = (t.apt == "LIRF") & (t.d > 3600) & (t.lane == "unm_rome")       # Rome, no NM clock, more than 1 h late
    t["dband"] = pd.cut(t.d, DBANDS, labels=False)
    t["cls"] = t.lane + "|" + t.dband.astype(str)
    return t

def fit(t, K):
    """Parameters from labelled rows t (2025)."""
    t = t.copy(); t["c_true"] = np.maximum(0, t.d - t.y); t["r"] = np.clip(t.y - t.p, -3600, 3600)
    g = t.groupby(["apt", "date"]).c_true.sum().reset_index().merge(K, on=["apt", "date"]); g["a"] = g.K - g.c_true
    mu = g.groupby("apt").a.median().to_dict()
    sd = g.groupby("apt").a.apply(lambda s: 1.4826 * (s - s.median()).abs().median() + 30.0).to_dict()
    dl = t[(t.p < t.d) & ~t.amb]
    var = dl.groupby("cls").r.apply(lambda s: float(np.mean(s ** 2))).to_dict()
    # expected gap per delayed ordinary row between the true and the predicted day sum (convexity of max(0, d - y))
    o = t[~t.amb].assign(cp=lambda x: np.maximum(0, x.d - x.p), dl=lambda x: (x.p < x.d).astype(int))
    go = o.groupby(["apt", "date"]).agg(ct=("c_true", "sum"), cp=("cp", "sum"), n=("dl", "sum")).reset_index()
    bias = go.assign(b=(go.ct - go.cp) / go.n.clip(lower=1)).groupby("apt").b.median().to_dict()
    amb = t[t.amb & (t.y < t.d - 900)]
    tau = float(np.median(amb.y)) if len(amb) > 20 else 1000.0
    return {"mu": mu, "sd": sd, "var": var, "var_default": float(np.mean(dl.r ** 2)), "tau": tau, "bias": bias}

def apply(t, K, prm, lam=1.0, dither=0.0, rome_posterior=False, use_bias=True, phi=1.0, df=3.0):
    """Return adjusted predictions for the rows of t (index-aligned)."""
    out = t.p.to_numpy(float).copy()
    tt = t.reset_index(drop=True)
    tt["v"] = tt.cls.map(prm["var"]).fillna(prm["var_default"]).astype(float)
    Kd = K.set_index(["apt", "date"])
    tau = prm["tau"]
    for (apt, date), g in tt.groupby(["apt", "date"]):
        if (apt, date) not in Kd.index: continue
        k = float(Kd.loc[(apt, date), "K"]) - prm["mu"].get(apt, 0.0)
        idx = g.index.to_numpy(); amb = g.amb.to_numpy().copy(); d = g.d.to_numpy(); p = g.p.to_numpy()
        ia = np.flatnonzero(amb)
        if len(ia) > M_MAX: amb[ia[np.argsort(-d[ia])[M_MAX:]]] = False            # overflow rows are handled as ordinary rows
        o = ~amb; dly = o & (p < d)
        c_ord = float(np.sum(np.maximum(0, d[o] - p[o])))
        s2_ord = float(np.sum(g.v.to_numpy()[dly])) + prm["sd"].get(apt, 1000.0) ** 2 + dither * len(g)
        R = k - c_ord - (prm.get("bias", {}).get(apt, 0.0) * int(dly.sum()) if use_bias else 0.0)
        e_amb = 0.0
        ia = np.flatnonzero(amb)
        if len(ia) and not rome_posterior:                                      # keep these rows; count their expected contribution under the prior
            pi = np.clip((p[ia] - tau) / np.maximum(d[ia] - tau, 1.0), 0.0, 1.0)
            e_amb = float(np.sum((1 - pi) * np.maximum(d[ia] - tau, 0.0)))
        elif len(ia):
            da = d[ia]; held = np.maximum(da - tau, 0.0)
            pi = np.clip((p[ia] - tau) / np.maximum(da - tau, 1.0), 0.02, 0.98)        # prior P(recorded at schedule)
            combos = np.array(np.meshgrid(*[[0, 1]] * len(ia))).reshape(len(ia), -1).T  # 1 = recorded at schedule
            contrib = ((1 - combos) * held).sum(1)
            # Student-t likelihood (df) with the variance widened by phi: a row moves only when the day total is decisive
            logp = (combos * np.log(pi) + (1 - combos) * np.log(1 - pi)).sum(1) - 0.5 * (df + 1) * np.log1p((R - contrib) ** 2 / (df * phi * s2_ord))
            w = np.exp(logp - logp.max()); w /= w.sum()
            pF = (w[:, None] * combos).sum(0)
            out[idx[ia]] = pF * da + (1 - pF) * tau
            e_amb = float((w * contrib).sum())
        Ro = R - e_amb
        if dly.any():
            v = g.v.to_numpy()[dly]; gain = v / (v.sum() + prm["sd"].get(apt, 1000.0) ** 2 + dither * len(g))
            out[idx[dly]] = p[dly] - lam * gain * Ro
    return np.clip(out, 0, None)

def save(prm, lam, path):
    json.dump({"lam": lam, **prm}, open(path, "w"), indent=1)
