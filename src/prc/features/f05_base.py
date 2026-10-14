"""Build one feature table for 2025 training departures + 2026 ranking departures.

Leakage rule: every feature must be computable from the ranking file, where departures
have BLOCK_TIME/TAXITIME blanked. Departure pushback therefore always comes from AOBT_3_flt.
Arrivals keep all their times in the ranking file, so they are used in full.
"""
import glob, sys, time
import numpy as np
import pandas as pd

from prc import config as C
DATA = C.DATA
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/features_base.parquet"

WX_FLAGS = {"FG": "fog", "BR": "mist", "SN": "snow", "RA": "rain", "TS": "thunder",
            "FZ": "freezing", "DZ": "drizzle", "SH": "shower", "GR": "hail", "HZ": "haze"}
SKY_CODE = {"CLR": 0, "SKC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 5}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_movements():
    tr = C.read_train()
    tr["is_rank"] = False
    rk = C.read_target()
    rk["is_rank"] = True
    mv = pd.concat([tr, rk], ignore_index=True)
    for c in mv.columns:
        if str(mv[c].dtype).startswith("datetime64[us, UTC]"):
            mv[c] = mv[c].dt.tz_convert(None)
    # AOBT_3 is unreliable when the implied taxi is <120 s or >3600 s (2025 truth is then a normal taxi in >90% of cases).
    # Mask it so every AOBT_3-derived feature (and the residual base) falls back to stand/runway statistics.
    implied = (mv.MVT_TIME_UTC_mvt - mv.AOBT_3_flt).dt.total_seconds()
    bad = (mv.PHASE_mvt == "DEP") & implied.notna() & ((implied < 120) | (implied > 3600))
    mv["aobt3_masked"] = bad.astype(np.int8)
    mv.loc[bad, "AOBT_3_flt"] = pd.NaT
    log(f"masked unreliable AOBT_3 on {int(bad.sum())} departures (train {int((bad & ~mv.is_rank).sum())}, rank {int((bad & mv.is_rank).sum())})")
    return mv


def secs(a, b):
    return (a - b).dt.total_seconds()


def window_count(ref_times, event_times, lo, hi):
    """For each ref time t, count events in [t+lo, t+hi). Both arrays in float seconds."""
    ev = np.sort(event_times)
    return np.searchsorted(ev, ref_times + hi, side="left") - np.searchsorted(ev, ref_times + lo, side="left")


def congestion_features(mv, dep):
    """Airport-level and runway-level traffic counts around each departure."""
    log("congestion features")
    ep = lambda s: (s - pd.Timestamp("2025-01-01")).dt.total_seconds().to_numpy()
    out = {}
    dep_t = ep(dep.MVT_TIME_UTC_mvt)  # takeoff (known in ranking)
    dep_a = ep(dep.AOBT_3_flt)        # NM pushback (known in ranking, may be NaN)
    idx = dep.index.to_numpy()

    cols = {k: np.full(len(dep), np.nan) for k in [
        "dep_tko_m15", "dep_tko_m30", "dep_tko_p15", "dep_tko_m5", "dep_tko_p5",
        "rwy_tko_m15", "rwy_tko_p15", "rwy_tko_m5", "rwy_tko_p5", "rwy_tko_m30",
        "dep_push_m15", "dep_push_m30", "rwy_push_m15", "rwy_push_m30",
        "arr_ldg_m15", "arr_ldg_m30", "arr_ldg_p15", "rwy_ldg_m15", "rwy_ldg_p15",
        "arr_inblk_m30", "sched_dep_pm30", "sched_dep_pm60",
        "q_ahead_rwy", "q_ahead_apt", "taxiing_at_push", "taxiing_at_push_rwy",
        "push_before_me_tko_after_me_rwy", "hourly_dep_tko", "hourly_arr_ldg",
    ]}
    pos_of = pd.Series(np.arange(len(dep)), index=dep.index)

    for apt, g in dep.groupby("ADEP_mvt"):
        p = pos_of[g.index].to_numpy()
        t = dep_t[p]; a = dep_a[p]
        all_dep = mv[(mv.PHASE_mvt == "DEP") & (mv.ADEP_mvt == apt)]
        all_arr = mv[(mv.PHASE_mvt == "ARR") & (mv.ADES_mvt == apt)]
        D_t = ep(all_dep.MVT_TIME_UTC_mvt); D_a = ep(all_dep.AOBT_3_flt); D_a_ok = D_a[~np.isnan(D_a)]
        D_t_witha = np.sort(D_t[~np.isnan(D_a)])  # takeoffs of the same population that has a pushback time
        D_s = ep(all_dep.SCHED_TIME_UTC_mvt)
        A_l = ep(all_arr.MVT_TIME_UTC_mvt); A_b = ep(all_arr.BLOCK_TIME_UTC_mvt); A_b = A_b[~np.isnan(A_b)]

        cols["dep_tko_m15"][p] = window_count(t, D_t, -900, 0) - 1  # exclude self
        cols["dep_tko_m30"][p] = window_count(t, D_t, -1800, 0) - 1
        cols["dep_tko_m5"][p] = window_count(t, D_t, -300, 0) - 1
        cols["dep_tko_p15"][p] = window_count(t, D_t, 0, 900)
        cols["dep_tko_p5"][p] = window_count(t, D_t, 0, 300)
        cols["hourly_dep_tko"][p] = window_count(t, D_t, -1800, 1800) - 1
        cols["dep_push_m15"][p] = window_count(t, D_a_ok, -900, 0)
        cols["dep_push_m30"][p] = window_count(t, D_a_ok, -1800, 0)
        cols["arr_ldg_m15"][p] = window_count(t, A_l, -900, 0)
        cols["arr_ldg_m30"][p] = window_count(t, A_l, -1800, 0)
        cols["arr_ldg_p15"][p] = window_count(t, A_l, 0, 900)
        cols["hourly_arr_ldg"][p] = window_count(t, A_l, -1800, 1800)
        cols["arr_inblk_m30"][p] = window_count(t, A_b, -1800, 0)
        cols["sched_dep_pm30"][p] = window_count(t, D_s, -1800, 1800) - 1
        cols["sched_dep_pm60"][p] = window_count(t, D_s, -3600, 3600) - 1

        # queue ahead: other departures that took off between my pushback and my takeoff
        ok = ~np.isnan(a)
        q = np.full(len(p), np.nan)
        q[ok] = np.searchsorted(np.sort(D_t), t[ok], "left") - np.searchsorted(np.sort(D_t), a[ok], "left") - 1
        cols["q_ahead_apt"][p] = q
        # taxiing at my pushback: pushed before me and not yet airborne
        tap = np.full(len(p), np.nan)
        tap[ok] = np.searchsorted(np.sort(D_a_ok), a[ok], "left") - np.searchsorted(D_t_witha, a[ok], "left")
        cols["taxiing_at_push"][p] = tap

        # runway-level
        for rwy, gr in g.groupby("RUNWAY_mvt"):
            pr = pos_of[gr.index].to_numpy(); tr_ = dep_t[pr]; ar = dep_a[pr]
            Dr = all_dep[all_dep.RUNWAY_mvt == rwy]
            Dr_t = np.sort(ep(Dr.MVT_TIME_UTC_mvt)); Dr_a_raw = ep(Dr.AOBT_3_flt); Dr_a = np.sort(Dr_a_raw[~np.isnan(Dr_a_raw)])
            Dr_t_witha = np.sort(ep(Dr.MVT_TIME_UTC_mvt)[~np.isnan(Dr_a_raw)])
            Ar_l = ep(all_arr[all_arr.RUNWAY_mvt == rwy].MVT_TIME_UTC_mvt)
            cols["rwy_tko_m15"][pr] = window_count(tr_, Dr_t, -900, 0) - 1
            cols["rwy_tko_m30"][pr] = window_count(tr_, Dr_t, -1800, 0) - 1
            cols["rwy_tko_m5"][pr] = window_count(tr_, Dr_t, -300, 0) - 1
            cols["rwy_tko_p15"][pr] = window_count(tr_, Dr_t, 0, 900)
            cols["rwy_tko_p5"][pr] = window_count(tr_, Dr_t, 0, 300)
            cols["rwy_push_m15"][pr] = window_count(tr_, Dr_a, -900, 0)
            cols["rwy_push_m30"][pr] = window_count(tr_, Dr_a, -1800, 0)
            cols["rwy_ldg_m15"][pr] = window_count(tr_, Ar_l, -900, 0)
            cols["rwy_ldg_p15"][pr] = window_count(tr_, Ar_l, 0, 900)
            okr = ~np.isnan(ar)
            qr = np.full(len(pr), np.nan)
            qr[okr] = np.searchsorted(Dr_t, tr_[okr], "left") - np.searchsorted(Dr_t, ar[okr], "left") - 1
            cols["q_ahead_rwy"][pr] = qr
            tpr = np.full(len(pr), np.nan)
            tpr[okr] = np.searchsorted(Dr_a, ar[okr], "left") - np.searchsorted(Dr_t_witha, ar[okr], "left")
            cols["taxiing_at_push_rwy"][pr] = tpr
            # pushed before me but took off after me (I overtook / they were held): proxy for ordering noise
            pb = np.full(len(pr), np.nan)
            pb[okr] = np.searchsorted(Dr_a, ar[okr], "left") - (np.searchsorted(Dr_t_witha, tr_[okr], "left") - 1)
            cols["push_before_me_tko_after_me_rwy"][pr] = pb
    for k, v in cols.items():
        out[k] = v
    return pd.DataFrame(out, index=dep.index)


def rolling_state_features(mv, dep):
    """Live congestion state from neighbours, using only ranking-available fields:
    - recent departures' naive taxi (MVT - AOBT_3), excluding self
    - recent arrivals' actual taxi-in (fully known in ranking)
    - pushback-relative traffic counts
    """
    log("rolling state features")
    ep = lambda s: (s - pd.Timestamp("2025-01-01")).dt.total_seconds().to_numpy()
    names = ["nb_naive_mean_m30", "nb_naive_mean_m60", "nb_naive_med_m30", "nb_naive_max_m30", "nb_naive_cnt_m30",
             "nb_rwy_naive_mean_m30", "nb_rwy_naive_med_m30",
             "nb_naive_mean_p30", "nb_rwy_naive_mean_p30",
             "arr_taxiin_mean_m30", "arr_taxiin_mean_m60", "arr_taxiin_med_m30", "arr_taxiin_cnt_m30",
             "push_dep_pm10", "push_dep_m20", "push_arr_ldg_pm10", "push_rwy_dep_pm10", "push_rwy_arr_pm10",
             "day_dep_cnt", "day_arr_cnt", "day_naive_mean", "day_arr_taxiin_mean"]
    cols = {k: np.full(len(dep), np.nan) for k in names}
    pos_of = pd.Series(np.arange(len(dep)), index=dep.index)
    dep_t = ep(dep.MVT_TIME_UTC_mvt); dep_a = ep(dep.AOBT_3_flt)

    def win_stats(ref, ev_t, ev_v, lo, hi, self_v=None):
        """mean/median/max/count of ev_v for events with time in [ref+lo, ref+hi), excluding one self value."""
        o = np.argsort(ev_t); et = ev_t[o]; ev = ev_v[o]
        cs = np.concatenate([[0.0], np.cumsum(ev)])
        L = np.searchsorted(et, ref + lo, "left"); R = np.searchsorted(et, ref + hi, "left")
        cnt = (R - L).astype(float); sm = cs[R] - cs[L]
        if self_v is not None:
            cnt = cnt - 1; sm = sm - self_v
        mean = np.where(cnt > 0, sm / np.where(cnt > 0, cnt, 1), np.nan)
        # median/max via loop over unique windows is slow; approximate median with mean of middle? use exact for m30 only
        return mean, cnt, et, ev, L, R

    def med_max(et, ev, L, R, self_v):
        med = np.full(len(L), np.nan); mx = np.full(len(L), np.nan)
        for i in range(len(L)):
            if R[i] - L[i] > 1:
                w = ev[L[i]:R[i]]
                if self_v is not None:
                    # remove one occurrence of self value
                    j = np.where(w == self_v[i])[0]
                    if len(j): w = np.delete(w, j[0])
                if len(w): med[i] = np.median(w); mx[i] = w.max()
        return med, mx

    for apt, g in dep.groupby("ADEP_mvt"):
        p = pos_of[g.index].to_numpy(); t = dep_t[p]; a = dep_a[p]
        all_dep = mv[(mv.PHASE_mvt == "DEP") & (mv.ADEP_mvt == apt)]
        all_arr = mv[(mv.PHASE_mvt == "ARR") & (mv.ADES_mvt == apt)]
        D_t = ep(all_dep.MVT_TIME_UTC_mvt); D_n = D_t - ep(all_dep.AOBT_3_flt)
        okd = ~np.isnan(D_n); D_t_ok = D_t[okd]; D_n_ok = D_n[okd]
        my_n = t - a  # own naive (nan if missing)
        self_v = np.where(np.isnan(my_n), 0.0, my_n)  # if own naive missing, self not in event set -> subtract 0 but count-1 wrong
        # handle: for rows with missing naive, self is not in the event set; fix count by adding back 1
        m, c, et, ev, L, R = win_stats(t, D_t_ok, D_n_ok, -1800, 0 + 1, self_v)
        fix = np.isnan(my_n); c[fix] += 1; m[fix] = np.where(c[fix] > 0, (m[fix] * (c[fix] - 1) + 0) / np.where(c[fix] > 0, c[fix], 1), np.nan)
        # recompute mean properly for fix rows
        cs = np.concatenate([[0.0], np.cumsum(ev)]); sm = cs[R] - cs[L]
        m[fix] = np.where(c[fix] > 0, sm[fix] / np.where(c[fix] > 0, c[fix], 1), np.nan)
        cols["nb_naive_mean_m30"][p] = m; cols["nb_naive_cnt_m30"][p] = c
        med, mx = med_max(et, ev, L, R, np.where(fix, np.nan, my_n))
        cols["nb_naive_med_m30"][p] = med; cols["nb_naive_max_m30"][p] = mx
        m60, c60, *_ = win_stats(t, D_t_ok, D_n_ok, -3600, 1, self_v); c60[fix] += 1
        cols["nb_naive_mean_m60"][p] = m60
        mp, cp, et2, ev2, L2, R2 = win_stats(t, D_t_ok, D_n_ok, 0, 1800, None)
        # exclude self from p30 (self at ref+0 is included in [0,1800)) -> subtract when own naive present
        cs2 = np.concatenate([[0.0], np.cumsum(ev2)]); sm2 = cs2[R2] - cs2[L2]
        cp2 = cp - (~fix); sm2 = sm2 - np.where(fix, 0, my_n)
        cols["nb_naive_mean_p30"][p] = np.where(cp2 > 0, sm2 / np.where(cp2 > 0, cp2, 1), np.nan)

        A_l = ep(all_arr.MVT_TIME_UTC_mvt); A_b = ep(all_arr.BLOCK_TIME_UTC_mvt); A_tx = A_b - A_l
        oka = ~np.isnan(A_tx); A_b_ok = A_b[oka]; A_tx_ok = A_tx[oka]  # index arrivals by in-block time (fully complete before ref)
        ma, ca, eta, eva, La, Ra = win_stats(t, A_b_ok, A_tx_ok, -1800, 0, None)
        cols["arr_taxiin_mean_m30"][p] = ma; cols["arr_taxiin_cnt_m30"][p] = ca
        meda, _ = med_max(eta, eva, La, Ra, None); cols["arr_taxiin_med_m30"][p] = meda
        ma60, *_ = win_stats(t, A_b_ok, A_tx_ok, -3600, 0, None); cols["arr_taxiin_mean_m60"][p] = ma60

        # pushback-relative counts (only where own AOBT3 present)
        ok = ~np.isnan(a); aa = a[ok]
        D_a_ok = np.sort(ep(all_dep.AOBT_3_flt)); D_a_ok = D_a_ok[~np.isnan(D_a_ok)]
        cols["push_dep_pm10"][p[ok]] = window_count(aa, D_a_ok, -600, 600) - 1
        cols["push_dep_m20"][p[ok]] = window_count(aa, D_a_ok, -1200, 0) - 1
        cols["push_arr_ldg_pm10"][p[ok]] = window_count(aa, A_l, -600, 600)

        # day-level
        day = (t // 86400).astype(int)
        dd = pd.Series(day); cnt_day = dd.map(pd.Series(D_t // 86400).value_counts()).to_numpy()
        cols["day_dep_cnt"][p] = cnt_day
        cols["day_arr_cnt"][p] = dd.map(pd.Series(A_l // 86400).value_counts()).to_numpy()
        dn = pd.DataFrame({"d": (D_t_ok // 86400).astype(int), "v": D_n_ok}).groupby("d").v.mean()
        cols["day_naive_mean"][p] = dd.map(dn).to_numpy()
        da = pd.DataFrame({"d": (A_l[oka] // 86400).astype(int), "v": A_tx_ok}).groupby("d").v.mean()
        cols["day_arr_taxiin_mean"][p] = dd.map(da).to_numpy()

        for rwy, gr in g.groupby("RUNWAY_mvt"):
            pr = pos_of[gr.index].to_numpy(); tr_ = dep_t[pr]; ar = dep_a[pr]; myn = tr_ - ar; fixr = np.isnan(myn)
            Dr = all_dep[all_dep.RUNWAY_mvt == rwy]; Dr_t = ep(Dr.MVT_TIME_UTC_mvt); Dr_n = Dr_t - ep(Dr.AOBT_3_flt)
            okr = ~np.isnan(Dr_n); Dr_t_ok = Dr_t[okr]; Dr_n_ok = Dr_n[okr]
            if len(Dr_t_ok) == 0: continue
            sv = np.where(fixr, 0.0, myn)
            mr, cr, etr, evr, Lr, Rr = win_stats(tr_, Dr_t_ok, Dr_n_ok, -1800, 1, sv); cr[fixr] += 1
            csr = np.concatenate([[0.0], np.cumsum(evr)]); smr = csr[Rr] - csr[Lr]
            mr[fixr] = np.where(cr[fixr] > 0, smr[fixr] / np.where(cr[fixr] > 0, cr[fixr], 1), np.nan)
            cols["nb_rwy_naive_mean_m30"][pr] = mr
            medr, _ = med_max(etr, evr, Lr, Rr, np.where(fixr, np.nan, myn)); cols["nb_rwy_naive_med_m30"][pr] = medr
            mpr, cpr, etr2, evr2, Lr2, Rr2 = win_stats(tr_, Dr_t_ok, Dr_n_ok, 0, 1800, None)
            csr2 = np.concatenate([[0.0], np.cumsum(evr2)]); smr2 = csr2[Rr2] - csr2[Lr2] - np.where(fixr, 0, myn); cpr2 = cpr - (~fixr)
            cols["nb_rwy_naive_mean_p30"][pr] = np.where(cpr2 > 0, smr2 / np.where(cpr2 > 0, cpr2, 1), np.nan)
            okk = ~np.isnan(ar); ark = ar[okk]
            Dr_a = np.sort(ep(Dr.AOBT_3_flt)); Dr_a = Dr_a[~np.isnan(Dr_a)]
            Ar_l = ep(all_arr[all_arr.RUNWAY_mvt == rwy].MVT_TIME_UTC_mvt)
            cols["push_rwy_dep_pm10"][pr[okk]] = window_count(ark, Dr_a, -600, 600) - 1
            cols["push_rwy_arr_pm10"][pr[okk]] = window_count(ark, Ar_l, -600, 600)
    return pd.DataFrame(cols, index=dep.index)


def config_features(mv, dep):
    """Runway configuration and operator-wave features (ranking-safe)."""
    log("config features")
    ep = lambda s: (s - pd.Timestamp("2025-01-01")).dt.total_seconds().to_numpy()
    names = ["n_dep_rwys_pm30", "n_arr_rwys_pm30", "rwy_share_dep_pm30", "rwy_mixed_mode", "rwy_dep_share_of_apt_m30",
             "min_since_rwy_first_use", "op_dep_m30", "op_dep_pm15", "same_ades_dep_pm30",
             "arr_excess_mean_m30", "arr_excess_mean_m60", "arr_excess_max_m30", "nb_excess_mean_m30", "nb_rwy_excess_mean_m30"]
    cols = {k: np.full(len(dep), np.nan) for k in names}
    pos_of = pd.Series(np.arange(len(dep)), index=dep.index)
    dep_t = ep(dep.MVT_TIME_UTC_mvt)
    config_str = np.empty(len(dep), dtype=object)

    def win_mean(ref, ev_t, ev_v, lo, hi, exclude_self_v=None):
        o = np.argsort(ev_t); et = ev_t[o]; ev = ev_v[o]
        cs = np.concatenate([[0.0], np.cumsum(ev)])
        L = np.searchsorted(et, ref + lo, "left"); R = np.searchsorted(et, ref + hi, "left")
        cnt = (R - L).astype(float); sm = cs[R] - cs[L]
        if exclude_self_v is not None:
            has = ~np.isnan(exclude_self_v); cnt = cnt - has; sm = sm - np.where(has, exclude_self_v, 0)
        return np.where(cnt > 0, sm / np.where(cnt > 0, cnt, 1), np.nan), L, R, et, ev

    # unimpeded per (airport, runway) for arrivals: p10 taxi-in over 2025 training arrivals
    arr_tr = mv[(mv.PHASE_mvt == "ARR") & (~mv.is_rank)]
    arr_unimp = arr_tr.groupby(["ADES_mvt", "RUNWAY_mvt"]).TAXITIME_SEC_mvt.quantile(0.10)
    dep_tr = mv[(mv.PHASE_mvt == "DEP") & (~mv.is_rank)]
    dep_unimp = dep_tr.groupby(["ADEP_mvt", "RUNWAY_mvt"]).TAXITIME_SEC_mvt.quantile(0.10)

    for apt, g in dep.groupby("ADEP_mvt"):
        p = pos_of[g.index].to_numpy(); t = dep_t[p]
        all_dep = mv[(mv.PHASE_mvt == "DEP") & (mv.ADEP_mvt == apt)]
        all_arr = mv[(mv.PHASE_mvt == "ARR") & (mv.ADES_mvt == apt)]
        D_t = ep(all_dep.MVT_TIME_UTC_mvt); D_r = all_dep.RUNWAY_mvt.to_numpy()
        A_t = ep(all_arr.MVT_TIME_UTC_mvt); A_r = all_arr.RUNWAY_mvt.to_numpy()
        # arrivals excess taxi-in, indexed by in-block time
        A_b = ep(all_arr.BLOCK_TIME_UTC_mvt); A_tx = A_b - A_t
        A_un = pd.Series(list(zip([apt] * len(all_arr), A_r))).map(lambda k: arr_unimp.get(k, np.nan)).to_numpy()
        A_ex = A_tx - A_un; oka = ~np.isnan(A_ex) & ~np.isnan(A_b)
        m30, La, Ra, eta, eva = win_mean(t, A_b[oka], A_ex[oka], -1800, 0)
        cols["arr_excess_mean_m30"][p] = m30
        cols["arr_excess_mean_m60"][p] = win_mean(t, A_b[oka], A_ex[oka], -3600, 0)[0]
        mx = np.full(len(p), np.nan)
        for i in range(len(p)):
            if Ra[i] > La[i]: mx[i] = eva[La[i]:Ra[i]].max()
        cols["arr_excess_max_m30"][p] = mx
        # neighbours' excess (naive - dep unimpeded per rwy), excluding self
        D_a = ep(all_dep.AOBT_3_flt); D_n = D_t - D_a
        D_un = pd.Series(list(zip([apt] * len(all_dep), D_r))).map(lambda k: dep_unimp.get(k, np.nan)).to_numpy()
        D_ex = D_n - D_un; okd = ~np.isnan(D_ex)
        my_ex = (t - ep(g.AOBT_3_flt)) - pd.Series(list(zip([apt] * len(g), g.RUNWAY_mvt))).map(lambda k: dep_unimp.get(k, np.nan)).to_numpy()
        cols["nb_excess_mean_m30"][p] = win_mean(t, D_t[okd], D_ex[okd], -1800, 1, my_ex)[0]
        # runway configuration in +/-30 min: which runways carry >=10% of movements
        # per departure: count of distinct dep runways & arr runways; share of own runway
        order_d = np.argsort(D_t); Dt_s = D_t[order_d]; Dr_s = D_r[order_d]
        order_a = np.argsort(A_t); At_s = A_t[order_a]; Ar_s = A_r[order_a]
        Ld = np.searchsorted(Dt_s, t - 1800); Rd = np.searchsorted(Dt_s, t + 1800)
        La_ = np.searchsorted(At_s, t - 1800); Ra_ = np.searchsorted(At_s, t + 1800)
        my_r = g.RUNWAY_mvt.to_numpy()
        # runway first-use tracking: time since this runway started being used for departures after a gap > 2h
        rwy_first_use = {}
        for r in np.unique(Dr_s):
            tt = Dt_s[Dr_s == r]; gaps = np.diff(tt) > 7200; starts = np.concatenate([[tt[0]], tt[1:][gaps]])
            rwy_first_use[r] = starts
        for i in range(len(p)):
            dr = Dr_s[Ld[i]:Rd[i]]; ar = Ar_s[La_[i]:Ra_[i]]
            if len(dr):
                u, c = np.unique(dr, return_counts=True); sig = u[c >= max(1, 0.1 * len(dr))]
                cols["n_dep_rwys_pm30"][p[i]] = len(sig)
                cols["rwy_share_dep_pm30"][p[i]] = c[u == my_r[i]].sum() / len(dr) if (u == my_r[i]).any() else 0.0
                dep_cfg = "|".join(sorted(sig))
            else:
                dep_cfg = ""
            if len(ar):
                u2, c2 = np.unique(ar, return_counts=True); sig2 = u2[c2 >= max(1, 0.1 * len(ar))]
                cols["n_arr_rwys_pm30"][p[i]] = len(sig2)
                cols["rwy_mixed_mode"][p[i]] = float((ar == my_r[i]).sum()) / len(ar)
                arr_cfg = "|".join(sorted(sig2))
            else:
                arr_cfg = ""
            config_str[p[i]] = f"{apt}:D[{dep_cfg}]A[{arr_cfg}]"
            st = rwy_first_use.get(my_r[i])
            if st is not None:
                j = np.searchsorted(st, t[i], "right") - 1
                if j >= 0: cols["min_since_rwy_first_use"][p[i]] = (t[i] - st[j]) / 60.0
        # operator waves & same-destination
        D_op = all_dep.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy(); D_des = all_dep.ADES_mvt.astype(str).to_numpy()
        for key_arr, my_key, name, lo, hi in [(D_op, g.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy(), "op_dep_m30", -1800, 0),
                                              (D_op, g.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy(), "op_dep_pm15", -900, 900),
                                              (D_des, g.ADES_mvt.astype(str).to_numpy(), "same_ades_dep_pm30", -1800, 1800)]:
            df = pd.DataFrame({"k": key_arr, "t": D_t}).sort_values(["k", "t"])
            grp = {k: v.to_numpy() for k, v in df.groupby("k").t}
            out = np.full(len(p), np.nan)
            for i in range(len(p)):
                arr = grp.get(my_key[i])
                if arr is not None:
                    out[i] = np.searchsorted(arr, t[i] + hi, "left") - np.searchsorted(arr, t[i] + lo, "left") - (1 if lo <= 0 < hi else 0)
            cols[name][p] = out
    res = pd.DataFrame(cols, index=dep.index)
    res["rwy_config"] = pd.Series(config_str, index=dep.index)
    return res


def holiday_features(dep):
    """Public holidays per airport country (fixed-date + Easter-based), ranking-safe."""
    from datetime import date, timedelta
    def easter(y):
        a=y%19;b=y//100;c=y%100;d=b//4;e=b%4;f=(b+8)//25;g=(b-f+1)//3;h=(19*a+b-d-g+15)%30;i=c//4;k=c%4
        l=(32+2*e+2*i-h-k)%7;m=(a+11*h+22*l)//451;mo=(h+l-7*m+114)//31;da=((h+l-7*m+114)%31)+1
        return date(y,mo,da)
    country = {"EDDF":"DE","EDDM":"DE","EGLL":"GB","EHAM":"NL","LEBL":"ES","LEMD":"ES","LFPG":"FR","LIRF":"IT","LTFM":"TR","LSZH":"CH"}
    fixed = {"DE":[(1,1),(5,1),(10,3),(12,25),(12,26)],"GB":[(1,1),(12,25),(12,26)],"NL":[(1,1),(4,27),(5,5),(12,25),(12,26)],
             "ES":[(1,1),(1,6),(5,1),(8,15),(10,12),(11,1),(12,6),(12,8),(12,25)],"FR":[(1,1),(5,1),(5,8),(7,14),(8,15),(11,1),(11,11),(12,25)],
             "IT":[(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)],"TR":[(1,1),(4,23),(5,1),(5,19),(7,15),(8,30),(10,29)],
             "CH":[(1,1),(8,1),(12,25),(12,26)]}
    hol = set()
    for y in range(int(dep.MVT_TIME_UTC_mvt.dt.year.min()), int(dep.MVT_TIME_UTC_mvt.dt.year.max()) + 1):   # any period
        e = easter(y)
        for cc, days in fixed.items():
            for (m, d) in days: hol.add((cc, date(y, m, d)))
            for off in [-2, 1]: hol.add((cc, e + timedelta(days=off)))  # Good Friday, Easter Monday
            if cc in ("DE","NL","FR","CH"):
                for off in [39, 50]: hol.add((cc, e + timedelta(days=off)))  # Ascension, Whit Monday
    dts = dep.MVT_TIME_UTC_mvt.dt.date.to_numpy(); cc = dep.ADEP_mvt.map(country).to_numpy()
    is_h = np.array([(c, d) in hol for c, d in zip(cc, dts)], dtype=np.int8)
    near = np.array([((c, d + timedelta(days=1)) in hol) or ((c, d - timedelta(days=1)) in hol) for c, d in zip(cc, dts)], dtype=np.int8)
    return pd.DataFrame({"is_holiday": is_h, "near_holiday": near}, index=dep.index)


def weather_features(dep):
    log("weather features")
    frames = []
    for f in glob.glob(f"{C.WEATHER}/*.csv"):
        w = pd.read_csv(f, na_values=["M", ""])
        frames.append(w)
    w = pd.concat(frames, ignore_index=True)
    w["valid"] = pd.to_datetime(w["valid"])
    w = w.sort_values("valid")
    num = ["tmpf", "dwpf", "relh", "sknt", "drct", "gust", "vsby", "p01i", "skyl1", "alti"]
    for c in num:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w["sky"] = w["skyc1"].map(SKY_CODE)
    w["wxcodes"] = w["wxcodes"].fillna("")
    for code, name in WX_FLAGS.items():
        w[f"wx_{name}"] = w["wxcodes"].str.contains(code).astype(np.int8)
    w["wx_any"] = (w["wxcodes"].str.len() > 0).astype(np.int8)
    w["deice_risk"] = ((w.tmpf <= 37) & ((w.wx_any == 1) | (w.relh >= 90))).astype(np.int8)
    keep = ["station", "valid"] + num + ["sky"] + [f"wx_{n}" for n in WX_FLAGS.values()] + ["wx_any", "deice_risk"]
    w = w[keep].rename(columns={"station": "ADEP_mvt"})
    # rename to wx_ prefix for numerics
    w = w.rename(columns={c: f"wx_{c}" for c in num + ["sky"]})

    d = dep[["ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]].copy()
    d["ref"] = d["AOBT_3_flt"].fillna(d["MVT_TIME_UTC_mvt"] - pd.Timedelta(minutes=15))
    d = d.reset_index().sort_values("ref")
    m = pd.merge_asof(d, w.sort_values("valid"), left_on="ref", right_on="valid", by="ADEP_mvt",
                      direction="nearest", tolerance=pd.Timedelta("90min"))
    m = m.set_index("index").sort_index()
    return m.drop(columns=["ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "ref", "valid"])


def runway_heading(rwy):
    num = pd.to_numeric(rwy.str.extract(r"^(\d{2})")[0], errors="coerce")
    return num * 10.0


def lomo_target_stats(dep, keys, name, target="TAXITIME_SEC_mvt", min_count=1):
    """Leave-one-month-out mean/count of target for training rows; full-2025 stats for ranking rows."""
    tr = dep[~dep.is_rank]
    g = tr.groupby(keys + ["month"])[target].agg(["sum", "count"]).reset_index()
    tot = g.groupby(keys)[["sum", "count"]].sum().rename(columns={"sum": "tsum", "count": "tcnt"}).reset_index()
    g = g.merge(tot, on=keys)
    g[f"{name}_mean"] = (g.tsum - g["sum"]) / (g.tcnt - g["count"]).replace(0, np.nan)
    g[f"{name}_cnt"] = g.tcnt - g["count"]
    lomo = g[keys + ["month", f"{name}_mean", f"{name}_cnt"]]
    full = tot.assign(**{f"{name}_mean": tot.tsum / tot.tcnt, f"{name}_cnt": tot.tcnt * 11.0 / 12.0})[keys + [f"{name}_mean", f"{name}_cnt"]]
    a = dep[~dep.is_rank][keys + ["month"]].reset_index().merge(lomo, on=keys + ["month"], how="left").set_index("index")
    b = dep[dep.is_rank][keys].reset_index().merge(full, on=keys, how="left").set_index("index")
    res = pd.concat([a, b]).sort_index()[[f"{name}_mean", f"{name}_cnt"]]
    return res


def build():
    log("loading")
    mv = load_movements()
    dep = mv[mv.PHASE_mvt == "DEP"].copy()
    log(f"departures: {len(dep)}  (rank {dep.is_rank.sum()})")

    f = pd.DataFrame(index=dep.index)
    f["MVT_ID_mvt"] = dep.MVT_ID_mvt
    f["is_rank"] = dep.is_rank
    f["y"] = dep.TAXITIME_SEC_mvt.astype(float)
    t = dep.MVT_TIME_UTC_mvt
    f["month"] = t.dt.month; dep["month"] = f["month"]
    f["hour"] = t.dt.hour; f["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    f["dow"] = t.dt.dayofweek; f["doy"] = t.dt.dayofyear; f["is_weekend"] = (f.dow >= 5).astype(np.int8)
    f["year"] = t.dt.year

    # --- timing features (all available in ranking) ---
    log("timing features")
    f["naive"] = secs(t, dep.AOBT_3_flt)
    f["aobt3_missing"] = dep.AOBT_3_flt.isna().astype(np.int8)
    f["aobt3_masked"] = dep.aobt3_masked.astype(np.int8)
    f["aobt3_sec_nonzero"] = (dep.AOBT_3_flt.dt.second.fillna(0) != 0).astype(np.int8)
    f["tko_minus_eobt1"] = secs(t, dep.EOBT_1_flt)
    f["tko_minus_lobt"] = secs(t, dep.LOBT_flt)
    f["tko_minus_iobt"] = secs(t, dep.IOBT_flt)
    f["tko_minus_sched"] = secs(t, dep.SCHED_TIME_UTC_mvt)
    f["aobt3_minus_eobt1"] = secs(dep.AOBT_3_flt, dep.EOBT_1_flt)
    f["aobt3_minus_sched"] = secs(dep.AOBT_3_flt, dep.SCHED_TIME_UTC_mvt)
    f["aobt3_minus_lobt"] = secs(dep.AOBT_3_flt, dep.LOBT_flt)
    f["eobt1_minus_iobt"] = secs(dep.EOBT_1_flt, dep.IOBT_flt)
    f["lobt_minus_iobt"] = secs(dep.LOBT_flt, dep.IOBT_flt)
    f["lobt_minus_eobt1"] = secs(dep.LOBT_flt, dep.EOBT_1_flt)
    f["sched_minus_eobt1"] = secs(dep.SCHED_TIME_UTC_mvt, dep.EOBT_1_flt)
    f["arvt3_minus_arvt1"] = secs(dep.ARVT_3_flt, dep.ARVT_1_flt)
    f["planned_dur"] = secs(dep.ARVT_1_flt, dep.EOBT_1_flt)
    f["actual_dur_m3"] = secs(dep.ARVT_3_flt, dep.AOBT_3_flt)
    f["arvt3_minus_tko"] = secs(dep.ARVT_3_flt, t)
    # implied taxi from M3 if flight duration were as planned: (ARVT3 - planned airborne) heuristic
    f["m3_taxi_implied"] = f["actual_dur_m3"] - (f["arvt3_minus_tko"])

    # --- categoricals ---
    log("categoricals")
    f["apt"] = dep.ADEP_mvt
    f["rwy"] = dep.RUNWAY_mvt
    f["apt_rwy"] = dep.ADEP_mvt + "_" + dep.RUNWAY_mvt
    f["stand"] = dep.ADEP_mvt + "_" + dep.STAND_mvt.astype(str)
    f["stand_prefix"] = dep.ADEP_mvt + "_" + dep.STAND_mvt.astype(str).str.extract(r"^([A-Za-z]*)")[0].fillna("")
    f["stand_num"] = pd.to_numeric(dep.STAND_mvt.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    f["stand_rwy"] = f["stand"] + "_" + dep.RUNWAY_mvt
    f["actype"] = dep.AIRCRAFT_TYPE_mvt.fillna(dep.AIRCRAFT_TYPE_flt)
    f["wtc"] = dep.WK_TBL_CAT_flt
    f["operator"] = dep.AIRCRAFT_OPERATOR_flt.astype(str).str[:12]
    f["airline_code"] = dep.FLIGHT_mvt.astype(str).str.extract(r"^([A-Z0-9]{2,3}?)\d")[0]
    f["segment"] = dep.MARKET_SEGMENT_flt
    f["ftype"] = dep.FLIGHT_TYPE_flt
    f["frule"] = dep.FLIGHT_RULE_mvt
    f["ades"] = dep.ADES_mvt
    f["ades_country"] = dep.ADES_mvt.astype(str).str[:2]
    f["domestic"] = (dep.ADES_mvt.astype(str).str[:2] == dep.ADEP_mvt.str[:2]).astype(np.int8)
    f["diverted"] = (dep.ADES_flt.notna() & (dep.ADES_flt != dep.ADES_FILED_flt)).astype(np.int8)
    f["flt_unmatched"] = dep.FLIGHT_ID_mvt.isna().astype(np.int8)
    f["actype_mismatch"] = (dep.AIRCRAFT_TYPE_flt.notna() & (dep.AIRCRAFT_TYPE_mvt != dep.AIRCRAFT_TYPE_flt)).astype(np.int8)
    f["rwy_hdg"] = runway_heading(dep.RUNWAY_mvt)

    # --- congestion ---
    cf = congestion_features(mv, dep)
    f = f.join(cf)

    rs = rolling_state_features(mv, dep)
    f = f.join(rs)
    f["naive_minus_nb_mean_m30"] = f["naive"] - f["nb_naive_mean_m30"]
    f["naive_minus_nb_rwy_mean_m30"] = f["naive"] - f["nb_rwy_naive_mean_m30"]

    cf2 = config_features(mv, dep)
    f = f.join(cf2)
    f = f.join(holiday_features(dep))

    # --- weather ---
    wf = weather_features(dep)
    f = f.join(wf)
    rel = np.deg2rad(f["wx_drct"] - f["rwy_hdg"])
    f["crosswind"] = (f["wx_sknt"] * np.abs(np.sin(rel))).astype(float)
    f["headwind"] = (f["wx_sknt"] * np.cos(rel)).astype(float)

    # --- leave-one-month-out target stats ---
    log("target encodings")
    dep["stand_rwy"] = f["stand_rwy"]; dep["stand"] = f["stand"]; dep["apt_rwy"] = f["apt_rwy"]
    dep["operator"] = f["operator"]; dep["stand_prefix"] = f["stand_prefix"]
    dep["resid"] = f["y"] - f["naive"]
    f = f.join(lomo_target_stats(dep, ["stand_rwy"], "te_stand_rwy"))
    f = f.join(lomo_target_stats(dep, ["stand"], "te_stand"))
    f = f.join(lomo_target_stats(dep, ["apt_rwy"], "te_apt_rwy"))
    f = f.join(lomo_target_stats(dep, ["stand_prefix", "RUNWAY_mvt"], "te_sp_rwy"))
    f = f.join(lomo_target_stats(dep, ["ADEP_mvt", "operator"], "te_apt_op"))
    f = f.join(lomo_target_stats(dep, ["stand_rwy"], "ter_stand_rwy", target="resid"))
    f = f.join(lomo_target_stats(dep, ["ADEP_mvt", "operator"], "ter_apt_op", target="resid"))
    f = f.join(lomo_target_stats(dep, ["stand"], "ter_stand", target="resid"))
    # unimpeded proxy: 10th percentile per stand_rwy over full 2025 (mildly leaky for train rows; flagged)
    tr = dep[~dep.is_rank]
    p10 = tr.groupby("stand_rwy").TAXITIME_SEC_mvt.quantile(0.10).rename("unimp_p10_stand_rwy")
    f = f.join(p10, on="stand_rwy")
    p10b = tr.groupby("apt_rwy").TAXITIME_SEC_mvt.quantile(0.10).rename("unimp_p10_apt_rwy")
    f = f.join(p10b, on="apt_rwy")
    f["naive_minus_unimp"] = f["naive"] - f["unimp_p10_stand_rwy"]
    f["nb_mean_minus_unimp"] = f["nb_naive_mean_m30"] - f["unimp_p10_apt_rwy"]
    f["nb_rwy_mean_minus_unimp"] = f["nb_rwy_naive_mean_m30"] - f["unimp_p10_apt_rwy"]

    for c in ["apt", "rwy", "apt_rwy", "stand", "stand_prefix", "stand_rwy", "actype", "wtc", "operator",
              "airline_code", "segment", "ftype", "frule", "ades", "ades_country", "rwy_config"]:
        f[c] = f[c].astype("category")
    log(f"features: {f.shape}")
    f.to_parquet(OUT)
    log(f"saved {OUT}")


if __name__ == "__main__":
    build()

