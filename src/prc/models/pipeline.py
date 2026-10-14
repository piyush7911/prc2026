"""Model layer of the jovial-uniform taxi-out model (PRC Data Challenge 2026). Trained on 2025 labels only; period-agnostic.

  stage 0   base     : residual target y - b(row), b = takeoff - best plausible off-block clock (NM AOBT_3, else LOBT, else EOBT_1, else 1,000 s;
                       Rome rows without an NM initial time use the schedule). LightGBM lr .05 and lr .02 and CatBoost on the non-ADS-B features,
                       base = 0.45 * mean(LightGBM) + 0.55 * CatBoost; on rows with an NM clock four decorrelators (the same LightGBM anchored on
                       other clocks) enter with weight 0.05 each; extra members (MEMBERS, e.g. the mixed-target CatBoost and LightGBM) enter
                       where and as strongly as their gates decide.
  stage 1   stacker  : ADS-B corrector on COVERED rows (NM clock present and the departure's own ground track seen by adsb.lol).
  stage 1b  unmstk   : ADS-B corrector for departures without NM clock outside Rome whose track reaches take-off.
  stage 2   stage2   : collective correction from neighbours' corrections on the same runway and at the airport (matched rows, shrink 0.75).
  stage 2b  ltfm     : Istanbul runway-queue correction, replacing stage 2 at LTFM.
  stage 3   daily    : reconciliation with EUROCONTROL's published daily pre-departure delay (prc.models.daily).
  stage 4   readout  : covered departures with an assigned-stand leave time are moved towards the direct reading of that clock, more strongly the
                       closer the reading and the prediction agree.
  last      bounds   : limits implied by how the data set was built (NM join window, LEMD/LEBL maximum, cap at d for flights > 2 h late).
  gates     dec, member, unmstk, ltfm, u95, daily, readout: each candidate is scored on the 2025 out-of-fold predictions through the complete pipeline;
            kept only if every month in GATE_MONTHS improves and the 100 most-improved rows carry less than the whole gain. Decisions: $PRC_OUT/*.json.

Out-of-fold scheme: six two-month folds {1,7} {2,8} {3,9} {4,10} {5,11} {6,12} of 2025. The OOF predictions are (a) the honest 2025 validation
of the whole pipeline and (b) the training input of the next stage, so every stage sees out-of-sample inputs exactly as on the scored file.

Environment: PRC_FEAT (feature table with ADS-B columns), PRC_OUT (output folder), PRC_TEMPLATE (submission template), PRC_TEAM / PRC_VERSION
(output file name), PRC_GATE_MONTHS, PRC_THREADS, PRC_SMOKE (3 % of rows, short fits), PRC_GPU / PRC_GPU_DEVICES (CatBoost on NVIDIA GPUs).
Usage (scripts/run_final.sh runs all of it in order):
        python -m prc.models.pipeline base --learner NAME --mode oof | final [--seeds 0,1,2]     (NAME: a key of LEARNERS)
        python -m prc.models.pipeline stacker | unmstk | stage2 | ltfm --mode oof | final
        python -m prc.models.pipeline dec | member --learner NAME | u95 | daily | readout       (gates)
        python -m prc.models.pipeline assemble                                                   (2025 validation report + submission file)
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, lightgbm as lgb

FEAT = os.environ.get("PRC_FEAT", "outputs/final/features_ads.parquet")
OUT = os.environ.get("PRC_OUT", "outputs/final")
TEMPLATE = os.environ.get("PRC_TEMPLATE", "data/final_submitting.parquet")
TEAM = os.environ.get("PRC_TEAM", "jovial-uniform"); VERSION = os.environ.get("PRC_VERSION", "final")
THREADS = int(os.environ.get("PRC_THREADS", "8"))
SMOKE = os.environ.get("PRC_SMOKE") == "1"          # pre-flight: 3% of rows, rounds / 20 (code path check only)
FOLDS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
GATE_MONTHS = tuple(int(x) for x in os.environ.get("PRC_GATE_MONTHS", "1,2,6,7").split(","))   # training-year analogues of the scored months; every gate needs each to improve
APTS9 = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH", "LEMD", "LIRF", "LFPG"}       # LTFM: no adsb.lol ground coverage
DROP = {"MVT_ID_mvt", "is_rank", "y", "year", "doy"}
CATS = ["apt", "rwy", "apt_rwy", "stand", "stand_prefix", "stand_rwy", "actype", "wtc", "operator", "airline_code", "segment", "ftype",
        "frule", "ades", "ades_country", "rwy_config", "apt_airline", "apt_ades"]
HP = dict(objective="regression", num_leaves=511, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
          lambda_l2=50.0, cat_smooth=200, cat_l2=10, max_bin=255, max_cat_to_onehot=8, verbose=-1, num_threads=THREADS)
LEARNERS = {"lgb05": dict(lr=0.05, rounds=760), "lgb02": dict(lr=0.02, rounds=1650), "cat": dict(lr=0.08, iters=1500, depth=8),
            "lgb05dy": dict(lr=0.05, rounds=760, drop_above=80000.0),   # lgb05dy: partner for rows without NM clock, trained without day-early labels
            # decorrelators: the same LightGBM with the residual anchored on another clock (B: LOBT, D: none, E: EOBT_1, F: IOBT)
            "decB": dict(lr=0.05, rounds=760, anchor="B"), "decD": dict(lr=0.05, rounds=760, anchor="D"),
            "decE": dict(lr=0.05, rounds=760, anchor="E"), "decF": dict(lr=0.05, rounds=760, anchor="F"),
            # mixed-target member: departures up to 1 h late predict min(y, 2 h) directly; later ones predict y / d with weight min(d, 12 h)^2
            # (their squared error in seconds scales with d^2), d = takeoff - schedule
            "catmix": dict(lr=0.08, iters=1500, depth=8, target="mix"),
            # members gated like catmix: deep CatBoost on the residual target (GPU), and the mixed target with LightGBM heads
            "catd10": dict(lr=0.06, iters=3000, depth=10), "lgbmix": dict(lr=0.05, rounds=760, target="mix"),
            # the same two GPU members trained only on departures WITH an NM clock (specialists for the matched lanes)
            # multiplicative target: y / anchor with weight anchor^2 (the loss stays in seconds); anchor = the zf off-block clock gap, at least 300 s
            "lgbratio": dict(lr=0.05, rounds=760, target="ratio"),
            "catd10m": dict(lr=0.06, iters=3000, depth=10, rows="matched"), "catmixm": dict(lr=0.06, iters=3000, depth=10, target="mix", rows="matched")}
MEMBERS = ("catmix", "catd10", "lgbmix", "catd10m", "catmixm", "lgbratio")   # extra base members, each kept only if its gate (step `member`) passes
GPU = os.environ.get("PRC_GPU") == "1"                 # CatBoost on an NVIDIA GPU (task_type GPU); PRC_CATMIX_DEPTH / PRC_CATMIX_ITERS override the mixed member
if os.environ.get("PRC_CATMIX_DEPTH"): LEARNERS["catmix"]["depth"] = int(os.environ["PRC_CATMIX_DEPTH"])
if os.environ.get("PRC_CATMIX_ITERS"): LEARNERS["catmix"]["iters"] = int(os.environ["PRC_CATMIX_ITERS"])
if os.environ.get("PRC_CATMIX_LR"): LEARNERS["catmix"]["lr"] = float(os.environ["PRC_CATMIX_LR"])
DECS = ("decB", "decD", "decE", "decF")
BASE_LEARNERS = ("lgb05", "lgb02", "cat")
if SMOKE: LEARNERS = {k: {kk: (max(vv // 20, 10) if kk in ("rounds", "iters") else vv) for kk, vv in v.items()} for k, v in LEARNERS.items()}
W_LGB, W_CAT = 0.45, 0.55              # blend weights of the base: 2025 out-of-fold optimum (pooled 12 months 0.55; leave-one-fold-out optima 0.50 to 0.60)
STK = dict(objective="regression", learning_rate=0.03, num_leaves=511, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=50.0, cat_smooth=200, verbose=-1, num_threads=THREADS)
STK_ROUNDS = 40 if SMOKE else 800
S2 = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=50, verbose=-1, num_threads=THREADS)
os.makedirs(OUT, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def rmse(e): e = np.asarray(e, float); return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")

def base_zf(d):
    """Seconds from the best recorded off-block clock to take-off: NM actual off-block, else LOBT, else EOBT_1 (each used only if the implied
    taxi time lies in [-2 h, 48 h]), else 1,000 s; never negative. Rome departures with no NM initial time use take-off minus schedule."""
    b = pd.Series(np.nan, index=d.index)
    for clock in (d.naive_raw, d.tko_minus_lobt, d.tko_minus_eobt1):
        b = b.fillna(clock.where((clock >= -7200) & (clock <= 172800)))
    b = b.fillna(1000.0).clip(lower=0)
    no_nm_rome = (d.apt.astype(str) == "LIRF") & d.tko_minus_iobt.isna()
    return np.where(no_nm_rome, d.tko_minus_sched.clip(0, 172800), b)

def _band(s): return s.where(s.between(-7200, 172800))
ANCHORS = {"B": lambda d: _band(d.tko_minus_lobt).fillna(_band(d.naive_raw)).fillna(1000.0).clip(lower=0).to_numpy(),
           "D": lambda d: np.zeros(len(d)),
           "E": lambda d: _band(d.tko_minus_eobt1).fillna(_band(d.naive_raw)).fillna(1000.0).clip(lower=0).to_numpy(),
           "F": lambda d: _band(d.tko_minus_iobt).fillna(_band(d.naive_raw)).fillna(1000.0).clip(lower=0).to_numpy()}

def load():
    """Feature table with categoricals typed; returns (frame, base feature names without ADS-B columns, all feature names)."""
    f = pd.read_parquet(FEAT)
    if SMOKE: f = f[f.MVT_ID_mvt.mod(33) == 0].reset_index(drop=True)
    for c in CATS:
        if c in f and not isinstance(f[c].dtype, pd.CategoricalDtype): f[c] = f[c].astype(str).astype("category")
    f["apt"] = f["apt"].astype("category")
    feats_all = [c for c in f.columns if c not in DROP and not c.startswith("ads_nb_")]
    feats_base = [c for c in feats_all if not c.startswith("ads_")]
    return f, feats_base, feats_all

def lanes(d):
    """matched = NM clock present; covered = matched & own ground track seen (APTS9); unm_rome / unm_nr = no NM clock."""
    apt = d.apt.astype(str).to_numpy(); mt = d.aobt3_missing.to_numpy() == 0
    cov = mt & (d.ads_cov.fillna(-1).to_numpy() >= 2) & np.isin(apt, list(APTS9))
    return np.select([cov, mt, apt == "LIRF"], ["covered", "uncovered", "unm_rome"], "unm_nr")

def good(d):  # training-row filter of the corrector stages: the 5 day-early records of 2025 (y > 20,000 s with takeoff < 45 min late)
    return ~((d.y.to_numpy() > 20000) & (d.tko_minus_sched.to_numpy() < 2700))

def _fit_predict(learner, tr, te, feats, seed=0):
    """Fit one learner of LEARNERS on tr and predict te, in seconds (residual target added back to its anchor; mix and ratio targets converted)."""
    L = LEARNERS[learner]; anc = ANCHORS[L["anchor"]] if "anchor" in L else base_zf
    if "drop_above" in L: tr = tr[tr.y <= L["drop_above"]]
    if L.get("rows") == "matched": tr = tr[tr.aobt3_missing == 0]
    cats = [c for c in CATS if c in feats]
    if L.get("target") == "mix": return _fit_predict_mix(L, tr, te, feats, cats, seed)
    if L.get("target") == "ratio":
        s_tr = np.maximum(anc(tr), 300.0); s_te = np.maximum(anc(te), 300.0)
        m = lgb.train(dict(HP, learning_rate=L["lr"], seed=seed), lgb.Dataset(tr[feats], tr.y.to_numpy() / s_tr, weight=s_tr ** 2 / np.mean(s_tr ** 2),
                      categorical_feature=cats), L["rounds"])
        return np.clip(m.predict(te[feats]) * s_te, 0, None)
    t = tr.y.to_numpy() - anc(tr)
    if "iters" in L:          # CatBoost learners
        from catboost import CatBoostRegressor, Pool
        def prep(X):
            X = X[feats].copy()
            for c in cats: X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
            return X
        m = CatBoostRegressor(iterations=L["iters"], learning_rate=L["lr"], depth=L["depth"], loss_function="RMSE", thread_count=THREADS,
                              random_seed=seed, verbose=250, one_hot_max_size=8, max_ctr_complexity=2, border_count=254, allow_writing_files=False,
                              **({"task_type": "GPU", "devices": os.environ.get("PRC_GPU_DEVICES", "0")} if GPU else {}))
        m.fit(Pool(prep(tr), t, cat_features=cats)); return m.predict(prep(te)) + anc(te)
    m = lgb.train(dict(HP, learning_rate=L["lr"], seed=seed), lgb.Dataset(tr[feats], t, categorical_feature=cats), L["rounds"])
    return m.predict(te[feats]) + anc(te)

def _fit_predict_mix(L, tr, te, feats, cats, seed=0):
    """Two heads (CatBoost, or LightGBM when the learner has 'rounds') by delay to take-off d = takeoff - schedule: d <= 1 h predicts min(y, 2 h) in seconds; d > 1 h predicts y / d
    (clipped to [0, 1.5]) with weight min(d, 12 h)^2, because a long row's squared error in seconds scales with d^2."""
    from catboost import CatBoostRegressor, Pool
    def prep(X):
        X = X[feats].copy()
        for c in cats: X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
        return X
    dtr = tr.tko_minus_sched.fillna(0).to_numpy(float); dte = te.tko_minus_sched.fillna(0).to_numpy(float); out = np.zeros(len(te))
    for long_ in (False, True):
        a_ = dtr > 3600 if long_ else dtr <= 3600; b_ = dte > 3600 if long_ else dte <= 3600
        if not b_.any(): continue
        y_ = tr.y.to_numpy()[a_]
        t = np.clip(y_ / np.maximum(dtr[a_], 1.0), 0, 1.5) if long_ else np.minimum(y_, 7200.0)
        w = (np.minimum(dtr[a_], 43200.0) / 3600.0) ** 2 if long_ else None
        if "rounds" in L:
            m = lgb.train(dict(HP, learning_rate=L["lr"], seed=seed), lgb.Dataset(tr[feats][a_], t, weight=w, categorical_feature=cats), L["rounds"])
            pr = m.predict(te[feats][b_]); out[b_] = pr * dte[b_] if long_ else pr; continue
        m = CatBoostRegressor(iterations=L["iters"], learning_rate=L["lr"], depth=L["depth"] if not long_ else min(L["depth"], 8), loss_function="RMSE",
                              thread_count=THREADS, random_seed=seed, verbose=250, one_hot_max_size=8, max_ctr_complexity=2, border_count=254,
                              allow_writing_files=False, **({"task_type": "GPU", "devices": os.environ.get("PRC_GPU_DEVICES", "0")} if GPU else {}))
        m.fit(Pool(prep(tr[a_]), t, cat_features=cats, weight=w))
        pr = m.predict(prep(te[b_])); out[b_] = pr * dte[b_] if long_ else pr
    return np.clip(out, 0, None)

def cmd_base(a):
    """Base learner: six-fold out-of-fold predictions on 2025 (oof_base_<learner>) or final fits on all of 2025, one file per seed (pred_base_<learner>[_s<k>])."""
    f, fb, _ = load(); lab = f[(~f.is_rank)]
    if a.mode == "oof":
        out = []
        for fm in FOLDS:
            tr = lab[~lab.month.isin(fm) & (lab.y >= 0)]; te = lab[lab.month.isin(fm)]
            p = _fit_predict(a.learner, tr, te, fb); out.append(pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "pred": p}))
            e = te.y.to_numpy() - p; log(f"base {a.learner} fold {fm}: RMSE {rmse(e):.2f} | " + " ".join(f"m{m} {rmse(e[te.month.values == m]):.2f}" for m in fm))
        pd.concat(out).to_parquet(f"{OUT}/oof_base_{a.learner}.parquet", index=False)
    else:
        tr = lab[lab.y >= 0]; te = f[f.is_rank]
        for sd in a.seeds:     # seed 0 -> pred_base_<learner>.parquet, seed k -> pred_base_<learner>_s<k>.parquet (averaged in _base)
            fn = f"{OUT}/pred_base_{a.learner}" + ("" if sd == 0 else f"_s{sd}") + ".parquet"
            if os.path.exists(fn): continue
            p = _fit_predict(a.learner, tr, te, fb, seed=sd); pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "pred": p}).to_parquet(fn, index=False)
            log(f"base {a.learner} final seed {sd}: {len(te):,} target rows, mean {p.mean():.1f}")

def _base(kind, ids):
    """Blended zf base for the given MVT ids from oof_* (kind='oof') or pred_* (kind='pred') files."""
    import glob
    def avg(k):   # final predictions average every available seed; OOF files are single-seed
        fs = [f"{OUT}/{kind}_base_{k}.parquet"] + (sorted(glob.glob(f"{OUT}/{kind}_base_{k}_s*.parquet")) if kind == "pred" else [])
        return np.mean([pd.read_parquet(x).set_index("MVT_ID_mvt").pred.reindex(ids).to_numpy() for x in fs], axis=0)
    P = {k: avg(k) for k in BASE_LEARNERS}; b = W_LGB * 0.5 * (P["lgb05"] + P["lgb02"]) + W_CAT * P["cat"]
    dj = f"{OUT}/dec_weight.json"
    if os.path.exists(dj) and json.load(open(dj))["gate_passed"]:     # decorrelators only where the NM clock exists (their anchors have no Rome schedule fallback)
        a = json.load(open(dj))["a"]; mt = _matched(ids); b = np.where(mt, (1 - len(DECS) * a) * b + a * sum(avg(k) for k in DECS), b)
    for k in MEMBERS:     # extra members in gate order (scope: all rows, rows at most 1 h late, or rows with an NM clock)
        cj = f"{OUT}/{k}_weight.json"
        if os.path.exists(cj) and json.load(open(cj))["gate_passed"]:
            J = json.load(open(cj)); sc = {"matched": _matched, "short": _short, "uncovered": _uncov}.get(J["scope"], lambda i: np.ones(len(i), bool))(ids)
            b = np.where(sc, (1 - J["a"]) * b + J["a"] * avg(k), b)
    return b

_SH = None
def _short(ids):   # take-off at most 1 h after schedule (the catmix member's short head)
    global _SH
    if _SH is None: _SH = pd.read_parquet(FEAT, columns=["MVT_ID_mvt", "tko_minus_sched"]).set_index("MVT_ID_mvt").tko_minus_sched.le(3600)
    return _SH.reindex(ids).fillna(False).to_numpy()

_UC = None
def _uncov(ids):   # NM clock present, but no usable own ground track (the 'uncovered' lane)
    global _UC
    if _UC is None:
        u = pd.read_parquet(FEAT, columns=["MVT_ID_mvt", "aobt3_missing", "ads_cov", "apt"])
        _UC = pd.Series(lanes(u) == "uncovered", index=u.MVT_ID_mvt.to_numpy())
    return _UC.reindex(ids).fillna(False).to_numpy()

_MT = None
def _matched(ids):
    global _MT
    if _MT is None: _MT = pd.read_parquet(FEAT, columns=["MVT_ID_mvt", "aobt3_missing"]).set_index("MVT_ID_mvt").aobt3_missing.eq(0)
    return _MT.reindex(ids).fillna(False).to_numpy()

def cmd_stacker(a):
    """Stage 1: LightGBM on all features + the base prediction, target y - base, trained on covered 2025 rows (out-of-fold, or 3 seeds for the final)."""
    f, _, fa = load(); FE = fa + ["base"]; cats = [c for c in CATS if c in fa]
    lab = f[~f.is_rank].copy(); lab["base"] = _base("oof", lab.MVT_ID_mvt.to_numpy()); lab["lane"] = lanes(lab)
    trn = lab[(lab.lane == "covered") & good(lab)]
    def fit(tr, seeds): return [lgb.train(dict(STK, seed=s), lgb.Dataset(tr[FE], (tr.y - tr.base).to_numpy(), categorical_feature=cats), STK_ROUNDS) for s in range(seeds)]
    def pred(ms, X): return np.mean([m.predict(X[FE]) for m in ms], axis=0)
    if a.mode == "oof":
        out = []
        for fm in FOLDS:
            te = lab[lab.month.isin(fm) & (lab.lane == "covered")]
            if not len(te): continue
            c = pred(fit(trn[~trn.month.isin(fm)], 1), te); out.append(pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "corr": c}))
            log(f"stacker fold {fm}: covered {len(te):,} rows RMSE {rmse(te.y - te.base):.2f} -> {rmse(te.y - te.base - c):.2f}")
        pd.concat(out).to_parquet(f"{OUT}/oof_corr.parquet", index=False)
    else:
        rk = f[f.is_rank].copy(); rk["base"] = _base("pred", rk.MVT_ID_mvt.to_numpy()); rk["lane"] = lanes(rk); te = rk[rk.lane == "covered"]
        c = pred(fit(trn, 3), te); pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "corr": c}).to_parquet(f"{OUT}/pred_corr.parquet", index=False)
        log(f"stacker final: {len(te):,} covered target rows, corr mean {c.mean():+.1f} rms {rmse(c):.1f}")

# ---------------- stage 1b: ADS-B corrector for departures WITHOUT an NM clock ----------------
# Rows: no NM clock, not Rome, own ground track seen and reaching take-off (last sample at most 600 s before take-off).
# Training: those 2025 rows (weight 1) plus covered matched rows (weight 0.2, flag is_unm = 0) so the timing relations are learned on a
# large sample; target y - base (out-of-fold). The weight is chosen on the 2025 out-of-fold predictions and the layer is kept only if each
# of Jan, Feb, Jun and Jul improves (gate written to unm_weight.json, read by assemble).
U = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=THREADS)
U_ROUNDS = 40 if SMOKE else 800
def _unm_frame(d, kind):
    A = pd.read_parquet(f"{OUT}/adsb_feats.parquet", columns=["MVT_ID_mvt", "ads_last_rel", "ads_last_ground_rel"])
    d = d.merge(A, on="MVT_ID_mvt", how="left"); d["base"] = _base(kind, d.MVT_ID_mvt.to_numpy()); d["lane"] = lanes(d)
    d["is_unm"] = (d.lane == "unm_nr").astype(int)
    d["reach"] = (d.ads_cov.fillna(-1) >= 2) & (d.ads_last_rel >= -600)
    return d
def cmd_unmstk(a):
    """Stage 1b with its gate: ADS-B corrector for departures without NM clock (outside Rome) whose ground track reaches take-off."""
    f, _, fa = load(); FE = fa + ["base", "is_unm", "ads_last_rel", "ads_last_ground_rel"]; cats = [c for c in CATS if c in fa]
    lab = _unm_frame(f[~f.is_rank], "oof"); ok = good(lab)
    use = ok & lab.reach & ((lab.lane == "unm_nr") | (lab.lane == "covered"))
    def fit(tr):
        wt = np.where(tr.lane == "unm_nr", 1.0, 0.2)
        return lgb.train(U, lgb.Dataset(tr[FE], (tr.y - tr.base).to_numpy(), weight=wt, categorical_feature=cats), U_ROUNDS)
    tgt = lab.reach & (lab.lane == "unm_nr")
    if a.mode == "oof":
        out = []
        for fm in FOLDS:
            te = lab[tgt & lab.month.isin(fm)]
            if not len(te): continue
            c = fit(lab[use & ~lab.month.isin(fm)]).predict(te[FE]); out.append(pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "ucorr": c}))
            log(f"unm-stacker fold {fm}: {len(te):,} rows RMSE {rmse(te.y - te.base):.1f} -> {rmse(te.y - te.base - c):.1f}")
        U_ = pd.concat(out); U_.to_parquet(f"{OUT}/oof_ucorr.parquet", index=False)
        # gate on the whole pipeline (2025 out-of-fold)
        full = f[~f.is_rank].copy(); p, c0 = _assemble(full, "oof", use_unm=False, use_bounds=False); y = full.y.to_numpy(); mo = full.month.to_numpy(); okf = good(full)
        uc = U_.set_index("MVT_ID_mvt").ucorr.reindex(full.MVT_ID_mvt).fillna(0.0).to_numpy()
        grid = [0.25, 0.5, 0.75, 1.0]; res = {}
        for wgt in grid:
            q = np.clip(p + wgt * uc, 0, None)
            res[wgt] = {m: rmse((y - q)[okf & (mo == m)]) - rmse((y - p)[okf & (mo == m)]) for m in GATE_MONTHS}
            s4 = okf & np.isin(mo, GATE_MONTHS); res[wgt]["four"] = rmse((y - q)[s4]) - rmse((y - p)[s4]); res[wgt]["all12"] = rmse((y - q)[okf]) - rmse((y - p)[okf])
            g = (y - p) ** 2 - (y - q) ** 2; gg = np.sort(g[s4])[::-1]; res[wgt]["top100"] = float(gg[:100].sum() / g[s4].sum()) if g[s4].sum() > 0 else float("nan")
            log(f"  w={wgt}: " + " ".join(f"{k} {v:+.3f}" for k, v in res[wgt].items()))
        best = min(grid, key=lambda wg: res[wg]["all12"]); r = res[best]
        passed = all(r[m] < 0 for m in GATE_MONTHS) and r["four"] < 0 and r["top100"] < 1.0
        json.dump({"w": best if passed else 0.0, "gate_passed": bool(passed), "chosen_on": "pooled twelve-month optimum", "deltas": {str(k): v for k, v in res.items()}},
                  open(f"{OUT}/unm_weight.json", "w"), indent=1)
        log(f"unm-stacker gate {'PASSED' if passed else 'FAILED'} (w = {best})")
    else:
        rk = _unm_frame(f[f.is_rank], "pred"); te = rk[rk.reach & (rk.lane == "unm_nr")]
        c = fit(lab[use]).predict(te[FE]); pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "ucorr": c}).to_parquet(f"{OUT}/pred_ucorr.parquet", index=False)
        log(f"unm-stacker final: {len(te):,} target rows, corr mean {c.mean():+.1f} rms {rmse(c):.1f}")

# ---------------- stage 2: collective correction from runway and airport neighbours ----------------
S2X = ["cov", "corr", "hour", "apt_c"] + [f"{k}_{W}" for W in (15, 30, 60) for k in ("nb_cov_n", "nb_cov_corr", "nb_p1n")] + \
      [f"{k}_{W}" for W in (5, 15, 30) for k in ("rw_cov_n", "rw_cov_corr")]
def _nb(d, wins, by_runway):
    d = d.sort_values("t"); t = d.t.values; ci = d["cov"].values.astype(float); cc = np.where(d["cov"].values, d["corr"].values, 0.0)
    mi = d.mt.values.astype(float); pn = np.where(d.mt.values, d.p1n.values, 0.0); out = {}
    Cc, Nc, Cp, Nm = (np.cumsum(np.r_[0, x]) for x in (cc, ci, pn, mi))
    for W in wins:
        lo = np.searchsorted(t, t - W); hi = np.searchsorted(t, t + W, side="right"); n = Nc[hi] - Nc[lo] - ci
        k = "rw" if by_runway else "nb"
        out[f"{k}_cov_n_{W}"] = n; out[f"{k}_cov_corr_{W}"] = np.where(n > 0, (Cc[hi] - Cc[lo] - cc) / np.maximum(n, 1), np.nan)
        if not by_runway:
            m = Nm[hi] - Nm[lo] - mi; out[f"nb_p1n_{W}"] = np.where(m > 0, (Cp[hi] - Cp[lo] - pn) / np.maximum(m, 1), np.nan)
    return pd.DataFrame(out, index=d.index)
def _s2frame(v):
    """v: rows of ONE scored period (all departures of the period, so neighbours are complete) with base, corr, lane, naive_raw."""
    v = v.copy(); v["mt"] = (v.aobt3_missing == 0) & v.naive_raw.notna(); v["cov"] = (v.lane == "covered").astype(int)
    v["corr"] = np.where(v["cov"] == 1, v["corr"].fillna(0.0), 0.0); v["p1"] = v.base + v["corr"]; v["p1n"] = v.p1 - v.naive_raw
    v["t"] = (v.year.astype(int) * 400 + v.doy.astype(int)) * 1440 + v.minute_of_day
    v = v.join(pd.concat([_nb(g, (15, 30, 60), False) for _, g in v.groupby(v.apt.astype(str))]))
    v = v.join(pd.concat([_nb(g, (5, 15, 30), True) for _, g in v.groupby(v.apt.astype(str) + "|" + v.rwy.astype(str))]))
    v["apt_c"] = pd.Categorical(v.apt.astype(str), categories=sorted(APTS9 | {"LTFM"})); return v
def cmd_stage2(a):
    """Stage 2: small LightGBM predicting a matched flight's stage-1 error from neighbour aggregates (self excluded)."""
    f, _, _ = load(); cols = ["MVT_ID_mvt", "is_rank", "year", "month", "doy", "minute_of_day", "hour", "apt", "rwy", "y", "aobt3_missing",
                              "naive_raw", "tko_minus_sched", "ads_cov"]
    lab = f.loc[~f.is_rank, cols].copy(); lab["base"] = _base("oof", lab.MVT_ID_mvt.to_numpy()); lab["lane"] = lanes(lab)
    lab["corr"] = pd.read_parquet(f"{OUT}/oof_corr.parquet").set_index("MVT_ID_mvt")["corr"].reindex(lab.MVT_ID_mvt).to_numpy()
    D = _s2frame(lab); D = D[D.mt & good(D)].copy(); D["e"] = np.clip(D.y - D.p1, -3600, 3600)
    if a.mode == "oof":
        out = []
        for fm in FOLDS:
            tr, te = D[~D.month.isin(fm)], D[D.month.isin(fm)]
            s2 = lgb.train(S2, lgb.Dataset(tr[S2X], tr.e), 400).predict(te[S2X]); out.append(pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.values, "s2": s2}))
            log(f"stage2 fold {fm}: matched {len(te):,} RMSE {rmse(te.y - te.p1):.2f} -> {rmse(te.y - te.p1 - 0.75 * s2):.2f} (k .75)")
        pd.concat(out).to_parquet(f"{OUT}/oof_s2.parquet", index=False)
    else:
        m = lgb.train(S2, lgb.Dataset(D[S2X], D.e), 400)
        rk = f.loc[f.is_rank, cols].copy(); rk["base"] = _base("pred", rk.MVT_ID_mvt.to_numpy()); rk["lane"] = lanes(rk)
        rk["corr"] = pd.read_parquet(f"{OUT}/pred_corr.parquet").set_index("MVT_ID_mvt")["corr"].reindex(rk.MVT_ID_mvt).to_numpy()
        R = _s2frame(rk); s2 = np.where(R.mt, m.predict(R[S2X]), 0.0)
        pd.DataFrame({"MVT_ID_mvt": R.MVT_ID_mvt.values, "s2": s2}).to_parquet(f"{OUT}/pred_s2.parquet", index=False)
        log(f"stage2 final: rms {rmse(s2):.1f}")

# ---------------- stage 2b: Istanbul (LTFM) runway-queue correction ----------------
# At LTFM there is no ground ADS-B and the NM off-block clock is mostly a default, so the clock gap carries the runway queue. A small
# LightGBM learns the stage-1 error from how the NEIGHBOURS on the same runway (+-5/15/30/60 min, past/next 30 min) and at the airport
# (+-15/60 min) behave, each relative to the airport-day mean: their stage-1 prediction minus the stand/runway mean taxi, their stage-1
# prediction minus their NM clock gap, and their take-off delay against LOBT and schedule. Kept only if the 2025 gate passes (ltfm_weight.json).
L2 = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000, lambda_l2=10, feature_fraction=0.9,
          bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=THREADS, seed=0)
def _wmean(t, x, lo_off, hi_off, excl):
    ok = ~np.isnan(x); xx = np.where(ok, x, 0.0); cs = np.r_[0.0, np.cumsum(xx)]; cn = np.r_[0, np.cumsum(ok)]
    lo = np.searchsorted(t, t + lo_off, side="left"); hi = np.searchsorted(t, t + hi_off, side="right")
    sm = cs[hi] - cs[lo]; n = (cn[hi] - cn[lo]).astype(float)
    if excl: sm = sm - xx; n = n - ok
    return np.where(n > 0, sm / np.maximum(n, 1), np.nan)
def _ltfm_feats(D):
    """D: LTFM departures of one or more months with columns p1 (stage-1 prediction), mt, te_stand_rwy_mean, naive_raw, tko_minus_lobt,
    tko_minus_sched, rwy, t (seconds), day. Returns day-relative neighbour features (self excluded)."""
    out = []
    for _, g in D.groupby("day"):
        g = g.sort_values("t", kind="stable"); n = len(g); F = {}
        q = {"pte": g.p1 - g.te_stand_rwy_mean, "dh": g.p1 - g.naive_raw, "tkl": g.tko_minus_lobt, "tks": g.tko_minus_sched}
        q = {k: np.where(g.mt.values, np.clip(v.to_numpy(float), -7200, 7200), np.nan) for k, v in q.items()}
        rw = g.rwy.astype(str).values
        for k, x in q.items():
            dm = np.nanmean(x) if np.isfinite(x).any() else np.nan
            for W in (5, 15, 30, 60):
                v = np.full(n, np.nan)
                for r in np.unique(rw):
                    ii = np.flatnonzero(rw == r); v[ii] = _wmean(g.t.values[ii], x[ii], -W * 60, W * 60, True)
                F[f"rel_{k}_{W}"] = v - dm
            F[f"rel_own_{k}"] = x - dm
        out.append(pd.DataFrame(F, index=g.index))
    return pd.concat(out).reindex(D.index).astype("float32")
def _ltfm_frame(d, kind):
    d = d[d.apt.astype(str) == "LTFM"].copy(); d["p1"] = _base(kind, d.MVT_ID_mvt.to_numpy())
    d["mt"] = (d.aobt3_missing == 0) & d.naive_raw.notna(); d["t"] = ((d.year.astype(int) * 400 + d.doy.astype(int)) * 1440 + d.minute_of_day) * 60.0
    d["day"] = d.year.astype(int) * 1000 + d.doy.astype(int); X = _ltfm_feats(d); X["hour"] = d.hour.values
    return d, X
def cmd_ltfm(a):
    """Stage 2b with its gate: Istanbul runway-queue correction (replace or add to stage 2 there, shrink on a grid)."""
    f, _, _ = load(); cols = ["MVT_ID_mvt", "is_rank", "year", "month", "doy", "minute_of_day", "hour", "apt", "rwy", "y", "aobt3_missing", "naive_raw",
                              "tko_minus_sched", "tko_minus_lobt", "te_stand_rwy_mean"]
    lab, X = _ltfm_frame(f.loc[~f.is_rank, cols], "oof"); e = np.clip((lab.y - lab.p1).to_numpy(), -1800, 1800); trm = lab.mt.values & good(lab)
    if a.mode == "oof":
        out = []
        for fm in FOLDS:
            te = lab.month.isin(fm).values; tr = trm & ~te
            s = lgb.train(L2, lgb.Dataset(X[tr], e[tr]), 300).predict(X[te]); out.append(pd.DataFrame({"MVT_ID_mvt": lab.MVT_ID_mvt.values[te], "lt": s}))
            r0 = (lab.y - lab.p1).values; mm = lab.mt.values[te]
            log(f"ltfm fold {fm}: matched LTFM {mm.sum():,} RMSE {rmse(r0[te][mm]):.2f} -> {rmse(r0[te][mm] - s[mm]):.2f}")
        Lt = pd.concat(out); Lt.to_parquet(f"{OUT}/oof_lt.parquet", index=False)
        full = f[~f.is_rank].copy(); p, c0 = _assemble(full, "oof", use_ltfm=False, use_bounds=False); y = full.y.to_numpy(); mo = full.month.to_numpy(); okf = good(full)
        lt = Lt.set_index("MVT_ID_mvt")["lt"].reindex(full.MVT_ID_mvt).fillna(0.0).to_numpy(); isl = (full.apt.astype(str).values == "LTFM") & (c0["lane"] == "uncovered")
        s2part = np.where(isl, K_S2["uncovered"] * c0["s2"], 0.0); res = {}
        for mode in ("replace", "add"):
            for k in (0.25, 0.5, 0.75, 1.0):
                q = np.clip(p - (s2part if mode == "replace" else 0.0) + np.where(isl, k * lt, 0.0), 0, None)
                rr = {m: rmse((y - q)[okf & (mo == m)]) - rmse((y - p)[okf & (mo == m)]) for m in GATE_MONTHS}
                s4 = okf & np.isin(mo, GATE_MONTHS); rr["four"] = rmse((y - q)[s4]) - rmse((y - p)[s4]); rr["all12"] = rmse((y - q)[okf]) - rmse((y - p)[okf])
                g_ = (y - p) ** 2 - (y - q) ** 2; gg = np.sort(g_[s4])[::-1]; rr["top100"] = float(gg[:100].sum() / g_[s4].sum()) if g_[s4].sum() > 0 else float("nan")
                res[(mode, k)] = rr; log(f"  {mode} k={k}: " + " ".join(f"{kk} {v:+.3f}" for kk, v in rr.items()))
        best = min(res, key=lambda z: res[z]["all12"]); r = res[best]
        passed = all(r[m] < 0 for m in GATE_MONTHS) and r["four"] < 0 and r["top100"] < 1.0
        json.dump({"mode": best[0], "k": best[1] if passed else 0.0, "gate_passed": bool(passed), "chosen_on": "pooled twelve-month optimum",
                   "deltas": {f"{m}_{k}": v for (m, k), v in res.items()}}, open(f"{OUT}/ltfm_weight.json", "w"), indent=1)
        log(f"ltfm gate {'PASSED' if passed else 'FAILED'} ({best})")
    else:
        rk, Xr = _ltfm_frame(f.loc[f.is_rank, cols], "pred")
        s = lgb.train(L2, lgb.Dataset(X[trm], e[trm]), 300).predict(Xr)
        pd.DataFrame({"MVT_ID_mvt": rk.MVT_ID_mvt.values, "lt": s}).to_parquet(f"{OUT}/pred_lt.parquet", index=False); log(f"ltfm final: {len(rk):,} rows, rms {rmse(s):.1f}")

def cmd_dec(a):
    """Gate of the decorrelators in the base, on rows WITH an NM clock: base' = (1 - 4a) * base + a * (decB + decD + decE + decF). The pipeline is re-assembled with
    base' (downstream stages unchanged) as a first estimate; a is the pooled twelve-month optimum; all four final months must improve."""
    f, _, _ = load(); full = f[~f.is_rank].copy(); ids = full.MVT_ID_mvt.to_numpy(); y = full.y.to_numpy(); mo = full.month.to_numpy(); okf = good(full)
    dj = f"{OUT}/dec_weight.json"
    if os.path.exists(dj): os.remove(dj)
    p, c0 = _assemble(full, "oof", use_bounds=False); b0 = c0["base"]; mt = np.isin(c0["lane"], ("covered", "uncovered"))
    L = lambda k: pd.read_parquet(f"{OUT}/oof_base_{k}.parquet").set_index("MVT_ID_mvt").pred.reindex(ids).to_numpy()
    sd = sum(L(k) for k in DECS); res = {}
    for w in (0.025, 0.05, 0.075, 0.1, 0.125):
        q = np.clip(np.where(mt, p - b0 + (1 - 4 * w) * b0 + w * sd, p), 0, None)
        rr = {m: rmse((y - q)[okf & (mo == m)]) - rmse((y - p)[okf & (mo == m)]) for m in GATE_MONTHS}
        s4 = okf & np.isin(mo, GATE_MONTHS); rr["four"] = rmse((y - q)[s4]) - rmse((y - p)[s4]); rr["all12"] = rmse((y - q)[okf]) - rmse((y - p)[okf])
        g_ = (y - p) ** 2 - (y - q) ** 2; gg = np.sort(g_[s4])[::-1]; rr["top100"] = float(gg[:100].sum() / g_[s4].sum()) if g_[s4].sum() > 0 else float("nan")
        res[w] = rr; log(f"  dec a={w}: " + " ".join(f"{kk} {v:+.3f}" for kk, v in rr.items()))
    best = min(res, key=lambda z: res[z]["all12"]); r = res[best]
    passed = all(r[m] < 0 for m in GATE_MONTHS) and r["four"] < 0 and r["top100"] < 1.0
    json.dump({"a": best if passed else 0.0, "gate_passed": bool(passed), "deltas": {str(k): v for k, v in res.items()}}, open(dj, "w"), indent=1)
    log(f"dec gate {'PASSED' if passed else 'FAILED'} (a = {best})")

def cmd_member(a):
    """Gate of an extra base member (--learner; catmix, catd10, lgbmix): base' = (1 - a) * base + a * member, on all rows, on rows at most 1 h late (short) or on uncovered rows (NM clock, no ground track).
    Each scope's pooled twelve-month optimum must pass; among passing scopes the better twelve-month total is kept.
    The pipeline is re-assembled with base' (downstream stages unchanged, daily refitted per fold) as a first estimate; a and the scope are the
    pooled twelve-month optimum, checked by choosing them on five folds and scoring the sixth; all four final months must improve."""
    f, _, _ = load(); full = f[~f.is_rank].reset_index(drop=True); ids = full.MVT_ID_mvt.to_numpy(); y = full.y.to_numpy(); mo = full.month.to_numpy(); okf = good(full)
    cj = f"{OUT}/{a.learner}_weight.json"
    if os.path.exists(cj): os.remove(cj)
    p, c0 = _assemble(full, "oof"); res = {}; Q = {}
    try:
      for scope in ("all", "short", "uncovered"):
        for w in (0.1, 0.2, 0.3, 0.4, 0.5):
            json.dump({"a": w, "scope": scope, "gate_passed": True}, open(cj, "w")); q = _assemble(full, "oof")[0]; Q[(scope, w)] = q
            rr = {m: rmse((y - q)[okf & (mo == m)]) - rmse((y - p)[okf & (mo == m)]) for m in GATE_MONTHS}
            s4 = okf & np.isin(mo, GATE_MONTHS); rr["four"] = rmse((y - q)[s4]) - rmse((y - p)[s4]); rr["all12"] = rmse((y - q)[okf]) - rmse((y - p)[okf])
            g_ = (y - p) ** 2 - (y - q) ** 2; gg = np.sort(g_[s4])[::-1]; rr["top100"] = float(gg[:100].sum() / g_[s4].sum()) if g_[s4].sum() > 0 else float("nan")
            res[(scope, w)] = rr; log(f"  {a.learner} {scope} a={w}: " + " ".join(f"{kk} {v:+.3f}" for kk, v in rr.items()))
    finally:
        if os.path.exists(cj): os.remove(cj)   # never leave a trial weight behind
    s4 = okf & np.isin(mo, GATE_MONTHS); ok_scopes = {}
    for scope in ("all", "short", "uncovered"):
        cand = {k: v for k, v in res.items() if k[0] == scope}; best = min(cand, key=lambda z: cand[z]["all12"]); r = cand[best]
        nest = p.copy(); picks = []
        for fm in FOLDS:
            tr = okf & ~np.isin(mo, fm); te = np.isin(mo, fm); k = min(cand, key=lambda z: np.sum((y - Q[z])[tr] ** 2)); picks.append(k[1]); nest[te] = Q[k][te]
        rn = {m: rmse((y - nest)[okf & (mo == m)]) - rmse((y - p)[okf & (mo == m)]) for m in GATE_MONTHS}; rn["four"] = rmse((y - nest)[s4]) - rmse((y - p)[s4])
        passed = all(r[m] < 0 for m in GATE_MONTHS) and r["four"] < 0 and r["top100"] < 1.0 and rn["four"] < 0
        log(f"  scope {scope}: pooled optimum a={best[1]} {'PASS' if passed else 'FAIL'} | nested picks {picks}: " + " ".join(f"{kk} {v:+.3f}" for kk, v in rn.items()))
        if passed: ok_scopes[best] = (r, rn)
    if ok_scopes: best = min(ok_scopes, key=lambda z: res[z]["all12"]); rn = ok_scopes[best][1]
    else: best, rn = ("none", 0.0), {}
    json.dump({"a": best[1], "scope": best[0], "gate_passed": bool(ok_scopes), "chosen_on": "pooled twelve-month optimum per scope",
               "nested": rn, "deltas": {f"{s_}_{w}": v for (s_, w), v in res.items()}}, open(cj, "w"), indent=1)
    log(f"{a.learner} gate {'PASSED' if ok_scopes else 'FAILED'} ({best})")

def cmd_daily(a):
    """Gate of the daily reconciliation (prc.models.daily): parameters refitted without each held-out fold, shrink lambda = pooled optimum."""
    from prc.models import daily as DL
    f, _, _ = load(); full = f[~f.is_rank].reset_index(drop=True); y = full.y.to_numpy(); mo = full.month.to_numpy(); okf = good(full)
    p, c0 = _assemble(full, "oof", use_daily=False, use_readout=False); t = DL.frame(full, p, c0["lane"]); K = DL.load_K(); res = {}; Q = {}
    PRM = {fm: DL.fit(t[~np.isin(mo, fm) & okf], K) for fm in FOLDS}
    for lam, phi in [(l, f_) for l in (0.5, 0.75, 1.0) for f_ in (1.0, 2.0, 4.0, 9.0)]:
        q = p.copy()
        for fm in FOLDS:
            te = np.isin(mo, fm)
            q[te] = DL.apply(t[te], K, PRM[fm], lam=lam, rome_posterior=True, phi=phi)
        lam = (lam, phi)
        q = bounds(full, q); Q[lam] = q
        rr = {m: rmse((y - q)[okf & (mo == m)]) - rmse((y - p)[okf & (mo == m)]) for m in GATE_MONTHS}
        s4 = okf & np.isin(mo, GATE_MONTHS); rr["four"] = rmse((y - q)[s4]) - rmse((y - p)[s4]); rr["all12"] = rmse((y - q)[okf]) - rmse((y - p)[okf])
        g_ = (y - p) ** 2 - (y - q) ** 2; gg = np.sort(g_[s4])[::-1]; rr["top100"] = float(gg[:100].sum() / g_[s4].sum()) if g_[s4].sum() > 0 else float("nan")
        res[lam] = rr; log(f"  daily lam={lam}: " + " ".join(f"{kk} {v:+.3f}" for kk, v in rr.items()))
    best = min(res, key=lambda z: res[z]["all12"]); r = res[best]
    passed = all(r[m] < 0 for m in GATE_MONTHS) and r["four"] < 0 and r["top100"] < 1.0
    lnn = c0["lane"]; best_lam, best_phi = best
    for l in ("covered", "uncovered", "unm_nr", "unm_rome"):
        s = okf & np.isin(mo, GATE_MONTHS) & (lnn == l); log(f"    lane {l:9s}: {rmse((y-p)[s]):.1f} -> {rmse((y-Q[best])[s]):.1f}")
    prm = DL.fit(t[okf], K)
    json.dump({"lam": best_lam if passed else 0.0, "phi": best_phi, "rome_posterior": True, "gate_passed": bool(passed), "params": prm,
               "deltas": {f"lam{k[0]}_phi{k[1]}": v for k, v in res.items()}}, open(f"{OUT}/daily.json", "w"), indent=1)
    log(f"daily gate {'PASSED' if passed else 'FAILED'} (lambda = {best_lam}, phi = {best_phi})")

def cmd_u95(a):
    """Gate of the day-early-free LightGBM partner on rows without NM clock (non-Rome): pred += w * (lgb05dy - lgb05)."""
    f, _, _ = load(); full = f[~f.is_rank].copy(); ids = full.MVT_ID_mvt.to_numpy()
    p, c0 = _assemble(full, "oof", use_u95=False, use_bounds=False); y = full.y.to_numpy(); mo = full.month.to_numpy(); okf = good(full)
    L = lambda k: pd.read_parquet(f"{OUT}/oof_base_{k}.parquet").set_index("MVT_ID_mvt").pred.reindex(ids).to_numpy()
    dlt = np.where(c0["lane"] == "unm_nr", L("lgb05dy") - L("lgb05"), 0.0); res = {}
    for w in (0.25, 0.5, 0.75, 1.0):
        q = np.clip(p + w * dlt, 0, None); rr = {m: rmse((y - q)[okf & (mo == m)]) - rmse((y - p)[okf & (mo == m)]) for m in GATE_MONTHS}
        s4 = okf & np.isin(mo, GATE_MONTHS); rr["four"] = rmse((y - q)[s4]) - rmse((y - p)[s4]); rr["all12"] = rmse((y - q)[okf]) - rmse((y - p)[okf])
        g_ = (y - p) ** 2 - (y - q) ** 2; gg = np.sort(g_[s4])[::-1]; rr["top100"] = float(gg[:100].sum() / g_[s4].sum()) if g_[s4].sum() > 0 else float("nan")
        res[w] = rr; log(f"  u95 w={w}: " + " ".join(f"{kk} {v:+.3f}" for kk, v in rr.items()))
    best = min(res, key=lambda z: res[z]["all12"]); r = res[best]
    passed = all(r[m] < 0 for m in GATE_MONTHS) and r["four"] < 0 and r["top100"] < 1.0
    json.dump({"w": best if passed else 0.0, "gate_passed": bool(passed), "deltas": {str(k): v for k, v in res.items()}}, open(f"{OUT}/u95_weight.json", "w"), indent=1)
    log(f"u95 gate {'PASSED' if passed else 'FAILED'} (w = {best})")

K_S2 = {"covered": 0.75, "uncovered": 0.75}   # stage-2 shrink: pooled twelve-month optimum of the first build (0.5 is 0.01 s better on the final pipeline; kept)
def _assemble(d, kind, use_unm=True, use_ltfm=True, use_u95=True, use_bounds=True, use_daily=True, use_readout=True):
    """Prediction of the whole pipeline for rows d from the oof_* (kind='oof', 2025 validation) or pred_* (kind='pred') stage outputs and the
    gate decisions; the use_* switches remove one component (used by the gates). Returns (prediction, components)."""
    ids = d.MVT_ID_mvt.to_numpy(); base = _base(kind, ids)
    corr = pd.read_parquet(f"{OUT}/{kind}_corr.parquet").set_index("MVT_ID_mvt")["corr"].reindex(ids).fillna(0.0).to_numpy()
    s2 = pd.read_parquet(f"{OUT}/{kind}_s2.parquet").set_index("MVT_ID_mvt").s2.reindex(ids).fillna(0.0).to_numpy()
    ln = lanes(d); corr = np.where(ln == "covered", corr, 0.0); k = np.vectorize(lambda l: K_S2.get(l, 0.0))(ln)
    lj = f"{OUT}/ltfm_weight.json"
    if use_ltfm and os.path.exists(lj) and os.path.exists(f"{OUT}/{kind}_lt.parquet"):
        L_ = json.load(open(lj)); isl = (d.apt.astype(str).to_numpy() == "LTFM") & (ln == "uncovered")
        lt = pd.read_parquet(f"{OUT}/{kind}_lt.parquet").set_index("MVT_ID_mvt")["lt"].reindex(ids).fillna(0.0).to_numpy()
        if L_["gate_passed"]:
            if L_["mode"] == "replace": s2 = np.where(isl, 0.0, s2)
            corr = corr + np.where(isl, L_["k"] * lt, 0.0)
    uj = f"{OUT}/u95_weight.json"
    if use_u95 and os.path.exists(uj) and json.load(open(uj))["gate_passed"]:
        def L(k):
            fs = [f"{OUT}/{kind}_base_{k}.parquet"] + (sorted(glob.glob(f"{OUT}/{kind}_base_{k}_s*.parquet")) if kind == "pred" else [])
            return np.mean([pd.read_parquet(x).set_index("MVT_ID_mvt").pred.reindex(ids).to_numpy() for x in fs], axis=0)
        corr = corr + np.where(ln == "unm_nr", json.load(open(uj))["w"] * (L("lgb05dy") - L("lgb05")), 0.0)
    uc = np.zeros(len(d)); wj = f"{OUT}/unm_weight.json"
    if use_unm and os.path.exists(wj) and os.path.exists(f"{OUT}/{kind}_ucorr.parquet"):
        wu = json.load(open(wj))["w"]
        uc = wu * pd.read_parquet(f"{OUT}/{kind}_ucorr.parquet").set_index("MVT_ID_mvt").ucorr.reindex(ids).fillna(0.0).to_numpy()
        uc = np.where(ln == "unm_nr", uc, 0.0)
    q = np.clip(base + corr + k * s2 + uc, 0, None)
    if use_bounds: q = bounds(d, q, dither=25.0 if kind == "pred" else 0.0)
    dj2 = f"{OUT}/daily.json"
    if use_daily and os.path.exists(dj2) and json.load(open(dj2))["gate_passed"]:
        from prc.models import daily as DL
        J = json.load(open(dj2)); t = DL.frame(d, q, ln); K = DL.load_K()
        kw = dict(lam=J["lam"], rome_posterior=J.get("rome_posterior", False), phi=J.get("phi", 1.0))
        if kind == "pred": q = DL.apply(t, K, J["params"], dither=48.0, **kw)
        else:   # out-of-fold: the daily parameters are refitted without the held-out fold, as in the gate
            mo = d.month.to_numpy(); okf = good(d)
            for fm in FOLDS:
                te = np.isin(mo, fm)
                if te.any(): q[te] = DL.apply(t[te], K, DL.fit(t[~np.isin(mo, fm) & okf], K), **kw)
        if use_bounds: q = bounds(d, q, dither=25.0 if kind == "pred" else 0.0)
    rj = f"{OUT}/readout.json"
    if use_readout and os.path.exists(rj) and json.load(open(rj))["gate_passed"]:
        lv = np.where(ln == "covered", _ro_leave(d), np.nan); apt = d.apt.astype(str).to_numpy()
        if kind == "pred": q = ro_apply(q, lv, apt, json.load(open(rj))["params"])
        else:   # out-of-fold: offsets and weights refitted without the held-out fold, as in the gate
            mo = d.month.to_numpy(); okf = good(d); y = d.y.to_numpy(float); q0 = q.copy()
            for fm in FOLDS:
                te = np.isin(mo, fm)
                if te.any(): q[te] = ro_apply(q0[te], lv[te], apt[te], ro_fit(y, q0, lv, apt, ~np.isin(mo, fm) & okf & np.isfinite(lv)))
        if use_bounds: q = bounds(d, q, dither=25.0 if kind == "pred" else 0.0)
    return q, dict(base=base, corr=corr, s2=s2, ucorr=uc, lane=ln)

# ---------------- stage 4: assigned-stand readout ----------------
# On a covered departure with an assigned-stand leave time L (prc.adsb.features: last sample within 40 m of its stand), r = -L + offset(airport)
# reads the taxi-out directly. It is often the sharper of the two (2025 median absolute error at EDDF, EDDM, LEBL and LSZH 10-23 s against 19-32 s
# for the prediction q), but it fails badly when the stand sample is not the pushback. q is therefore moved towards r with a weight that
# depends only on the agreement |r - q| (bins RO_EDGES; least squares per bin, clipped to [0, 1]; no change at 600 s or more). Offsets (median of
# y + L per airport) and weights are fitted on 2025 (out-of-fold: without the held-out fold); the gate writes readout.json.
RO_EDGES = (15, 30, 60, 120, 300, 600)
def _ro_leave(d):
    A = pd.read_parquet(f"{OUT}/adsb_feats.parquet", columns=["MVT_ID_mvt", "ads_as_leave"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").ads_as_leave
    return A.reindex(d.MVT_ID_mvt.to_numpy()).to_numpy(float)

def _ro_x(q, lv, apt, off):
    r = -lv + pd.Series(apt).map(off).to_numpy(float); x = r - q
    return x, np.digitize(np.abs(np.nan_to_num(x, nan=1e9)), RO_EDGES)      # bin 0..5 inside 600 s, 6 = no change

def ro_fit(y, q, lv, apt, sel):
    """Readout parameters from rows sel: offset per airport and one weight per agreement bin."""
    off = pd.Series(y[sel] + lv[sel]).groupby(apt[sel]).median().to_dict()
    x, b = _ro_x(q, lv, apt, off); w = []
    for k in range(len(RO_EDGES)):
        m = sel & (b == k) & np.isfinite(x); den = float((x[m] ** 2).sum())
        w.append(float(np.clip((x[m] * (y - q)[m]).sum() / den, 0.0, 1.0)) if den > 0 else 0.0)
    return {"off": off, "w": w}

def ro_apply(q, lv, apt, prm):
    x, b = _ro_x(q, lv, apt, prm["off"]); w = np.append(prm["w"], 0.0)[b]
    return np.where(np.isfinite(x), q + w * np.nan_to_num(x), q)

def bounds(d, q, dither=0.0):
    """Last step: limits that follow from how the data set was built, all verified on 2025 labels.
    (a) the organisers joined an NM flight only when |off-block - LOBT| <= 3,606 s (2,062,577 matched 2025 departures, no exception), so a departure
        with LOBT has takeoff - LOBT - 3,606 <= y <= takeoff - LOBT + 3,606 (widened by the +-12 s take-off dither of the scored file; lower end at least 60 s);
    (d) LEMD and LEBL taxi-out never exceeded 3,604 s in 2025;
    (c) a departure more than 2 h late (d = takeoff - schedule) is not predicted above d."""
    tml = d.tko_minus_lobt.to_numpy(float); dl = d.tko_minus_sched.to_numpy(float); apt = d.apt.astype(str).to_numpy()
    has = np.isfinite(tml); w = 3606.0 + dither
    q = np.where(has, np.clip(q, np.maximum(60.0, tml - w), tml + w), q)
    q = np.where(np.isin(apt, ["LEMD", "LEBL"]), np.minimum(q, 3604.0), q)
    q = np.where(np.isfinite(dl) & (dl > 7200), np.minimum(q, dl), q)
    return np.clip(q, 0, None)

def cmd_readout(a):
    """Gate of stage 4 (assigned-stand readout): out-of-fold as in _assemble; kept only if every month in GATE_MONTHS and the four months improve and the
    100 most-improved rows carry less than the whole gain. Also reports the weights fitted without each fold and the size of the change on the scored rows."""
    f, _, _ = load(); full = f[~f.is_rank].reset_index(drop=True); y = full.y.to_numpy(float); mo = full.month.to_numpy(); okf = good(full)
    p, c0 = _assemble(full, "oof", use_readout=False); apt = full.apt.astype(str).to_numpy()
    lv = np.where(c0["lane"] == "covered", _ro_leave(full), np.nan); q = p.copy(); W = {}
    for fm in FOLDS:
        te = np.isin(mo, fm); prm = ro_fit(y, p, lv, apt, ~te & okf & np.isfinite(lv)); W[str(fm)] = [round(v, 3) for v in prm["w"]]
        q[te] = ro_apply(p[te], lv[te], apt[te], prm)
    q = bounds(full, q)
    rr = {m: rmse((y - q)[okf & (mo == m)]) - rmse((y - p)[okf & (mo == m)]) for m in GATE_MONTHS}
    s4 = okf & np.isin(mo, GATE_MONTHS); rr["four"] = rmse((y - q)[s4]) - rmse((y - p)[s4]); rr["all12"] = rmse((y - q)[okf]) - rmse((y - p)[okf])
    g_ = (y - p) ** 2 - (y - q) ** 2; gg = np.sort(g_[s4])[::-1]; rr["top100"] = float(gg[:100].sum() / g_[s4].sum()) if g_[s4].sum() > 0 else float("nan")
    passed = all(rr[m] < 0 for m in GATE_MONTHS) and rr["four"] < 0 and rr["top100"] < 1.0
    for k, v in W.items(): log(f"  weights without fold {k}: {v}")
    prm = ro_fit(y, p, lv, apt, okf & np.isfinite(lv))
    rk = f[f.is_rank].reset_index(drop=True); pr, cr = _assemble(rk, "pred", use_readout=False)
    lvr = np.where(cr["lane"] == "covered", _ro_leave(rk), np.nan); qr = bounds(rk, ro_apply(pr, lvr, rk.apt.astype(str).to_numpy(), prm), dither=25.0)
    sz = {"2025": rmse((q - p)[np.isin(mo, GATE_MONTHS)]), "2026": rmse(qr - pr), "2026_mean": float(np.mean(qr - pr)), "2026_rows_changed": int((qr != pr).sum())}
    log("  readout " + " ".join(f"{k} {v:+.3f}" for k, v in rr.items()) + f" | size of the change 2025 {sz['2025']:.2f} s, scored rows {sz['2026']:.2f} s (mean {sz['2026_mean']:+.2f}), ratio {sz['2026'] / sz['2025']:.2f}")
    json.dump({"gate_passed": bool(passed), "edges": list(RO_EDGES), "params": prm, "fold_weights": W, "deltas": rr, "size": sz}, open(f"{OUT}/readout.json", "w"), indent=1)
    log(f"readout gate {'PASSED' if passed else 'FAILED'}; weights {[round(v, 3) for v in prm['w']]}")

def cmd_assemble(a):
    """2025 validation report by month and lane (validation_2025.json) and the submission file in the template's row order."""
    f, _, _ = load(); rep = {}
    lab = f[~f.is_rank].copy(); p, c = _assemble(lab, "oof"); y = lab.y.to_numpy(); ok = good(lab)
    trim = ok & (np.abs(y - c["base"]) < 3600) & (y < 20000)
    log("2025 out-of-fold validation (whole pipeline), by month:  full | full excl. 5 corrupt | normal flights | lanes")
    for m in range(1, 13):
        s = lab.month.to_numpy() == m; e = (y - p)[s]; ln = c["lane"][s]
        rep[m] = dict(full=rmse(e), xc=rmse(e[ok[s]]), normal=rmse(e[trim[s]]), **{l: rmse(e[ln == l]) for l in ("covered", "uncovered", "unm_nr", "unm_rome")},
                      base_only=rmse((y - c["base"])[s]))
        log(f"  2025-{m:02d}: {rep[m]['full']:7.2f} | {rep[m]['xc']:7.2f} | {rep[m]['normal']:7.2f} | " + " ".join(f"{l} {rep[m][l]:.1f}" for l in ("covered", "uncovered", "unm_nr", "unm_rome")) + f" | base only {rep[m]['base_only']:.2f}")
    for nm, ms in (("Jan+Jul", (1, 7)), ("Feb+Jun", (2, 6)), ("Jan+Feb+Jun+Jul", (1, 2, 6, 7)), ("all 12", tuple(range(1, 13)))):
        s = lab.month.isin(ms).to_numpy(); rep[nm] = dict(full=rmse((y - p)[s]), xc=rmse((y - p)[s & ok]), normal=rmse((y - p)[s & trim]))
        log(f"  {nm:16s}: full {rep[nm]['full']:.2f}  excl. corrupt {rep[nm]['xc']:.2f}  normal {rep[nm]['normal']:.2f}")
    json.dump(rep, open(f"{OUT}/validation_2025.json", "w"), indent=1)
    rk = f[f.is_rank].copy(); p, c = _assemble(rk, "pred")
    t = pd.read_parquet(TEMPLATE)[["MVT_ID_mvt"]]
    if SMOKE: t = t[t.MVT_ID_mvt.isin(rk.MVT_ID_mvt)]
    o = t.merge(pd.DataFrame({"MVT_ID_mvt": rk.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": p}), on="MVT_ID_mvt", how="left")
    assert len(o) == len(t) and o.TAXITIME_SEC_mvt.notna().all() and o.MVT_ID_mvt.is_unique and (o.TAXITIME_SEC_mvt >= 0).all()
    fn = f"{OUT}/{TEAM}_{VERSION}.parquet"; o.to_parquet(fn, index=False)
    log(f"wrote {fn}: {len(o):,} rows, mean {o.TAXITIME_SEC_mvt.mean():.1f} | lanes " + str(pd.Series(c['lane']).value_counts().to_dict()))
    for m in sorted(rk.month.unique()):
        s = rk.month.to_numpy() == m; log(f"  target month {m}: rows {s.sum():,} mean pred {p[s].mean():.1f} | corr rms {rmse(c['corr'][s & (c['lane'] == 'covered')]):.1f} | s2 rms {rmse(c['s2'][s]):.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["base", "stacker", "unmstk", "stage2", "ltfm", "u95", "dec", "member", "daily", "readout", "assemble"])
    ap.add_argument("--learner", default="lgb05", choices=list(LEARNERS)); ap.add_argument("--mode", default="oof", choices=["oof", "final"])
    ap.add_argument("--seeds", default="0", help="final fits: comma list of seeds, e.g. 0,1,2,3"); a = ap.parse_args(); a.seeds = [int(x) for x in a.seeds.split(",")]
    {"base": cmd_base, "stacker": cmd_stacker, "unmstk": cmd_unmstk, "stage2": cmd_stage2, "ltfm": cmd_ltfm, "u95": cmd_u95, "dec": cmd_dec, "member": cmd_member, "daily": cmd_daily, "readout": cmd_readout, "assemble": cmd_assemble}[a.step](a)
