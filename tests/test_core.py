"""Fast checks of the pieces whose behaviour the pipeline depends on. Run: python -m pytest tests -q  (or python tests/test_core.py)"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from prc.models import pipeline as P

def frame(**cols):
    n = len(next(iter(cols.values()))); d = pd.DataFrame(cols)
    for c in ("naive_raw", "tko_minus_lobt", "tko_minus_eobt1", "tko_minus_iobt", "tko_minus_sched"):
        if c not in d: d[c] = np.nan
    if "apt" not in d: d["apt"] = "EDDF"
    return d

def test_base_uses_nm_clock_then_lobt_then_eobt_then_default():
    d = frame(naive_raw=[900.0, np.nan, np.nan, np.nan], tko_minus_lobt=[1200.0, 1100.0, np.nan, np.nan], tko_minus_eobt1=[1300.0, 1300.0, 1000.0, np.nan])
    assert P.base_zf(d).tolist() == [900.0, 1100.0, 1000.0, 1000.0]

def test_base_rejects_implausible_clocks_and_never_negative():
    d = frame(naive_raw=[-9000.0, 200000.0, -100.0], tko_minus_lobt=[700.0, 800.0, np.nan], tko_minus_eobt1=[np.nan] * 3)
    assert P.base_zf(d).tolist() == [700.0, 800.0, 0.0]          # -100 s is inside the band and clipped to 0

def test_rome_without_nm_initial_time_uses_schedule_gap():
    d = frame(apt=["LIRF", "LIRF", "EDDF"], naive_raw=[np.nan] * 3, tko_minus_iobt=[np.nan, 500.0, np.nan], tko_minus_sched=[4000.0, 4000.0, 4000.0])
    assert P.base_zf(d).tolist() == [4000.0, 1000.0, 1000.0]

def test_lanes():
    d = pd.DataFrame({"apt": ["EDDF", "EDDF", "LTFM", "LIRF", "EDDM"], "aobt3_missing": [0, 0, 0, 1, 1], "ads_cov": [2, 1, 3, -1, -1]})
    assert P.lanes(d).tolist() == ["covered", "uncovered", "uncovered", "unm_rome", "unm_nr"]      # LTFM never counts as covered

def test_day_early_filter():
    d = pd.DataFrame({"y": [90000.0, 90000.0, 1000.0], "tko_minus_sched": [100.0, 5000.0, 100.0]})
    assert P.good(d).tolist() == [False, True, True]

def test_blend_weights_sum_to_one():
    assert abs(P.W_LGB + P.W_CAT - 1.0) < 1e-12


def test_bounds_window_cap_and_spain():
    d = pd.DataFrame({"tko_minus_lobt": [1000.0, 9000.0, np.nan, np.nan], "tko_minus_sched": [500.0, 9000.0, 9000.0, 100.0], "apt": ["EDDF", "EDDF", "LIRF", "LEMD"]})
    q = P.bounds(d, np.array([6000.0, 2000.0, 12000.0, 5000.0]))
    assert q.tolist() == [4606.0, 5394.0, 9000.0, 3604.0]   # window top, window bottom, cap at d, LEMD max

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k)
