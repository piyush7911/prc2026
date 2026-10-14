"""Run configuration, all from environment variables so the same code serves 2026 ranking, a hidden test period, or a 2025 dry run.
  PRC_DATA            folder with raw challenge files (default: data)
  PRC_TRAIN_GLOB      labelled training files        (default: $PRC_DATA/training_*.parquet)
  PRC_TARGET          file to predict, same schema as ranking.parquet (departure BLOCK/TAXITIME blank) (default: $PRC_DATA/ranking.parquet)
  PRC_WEATHER         METAR folder (default: $PRC_DATA/weather)
  PRC_EXCLUDE_MONTHS  comma list of calendar months (YYYY-MM or M) removed from TRAINING rows (dry run on a held-out 2025 month)"""
import glob, os, pathlib
import pandas as pd
CONFIGS = pathlib.Path(__file__).resolve().parents[2] / "configs"   # small derived files shipped with the repository
DATA = os.environ.get("PRC_DATA", "data")
TRAIN_GLOB = os.environ.get("PRC_TRAIN_GLOB", f"{DATA}/training_*.parquet")
TARGET = os.environ.get("PRC_TARGET", f"{DATA}/ranking.parquet")
WEATHER = os.environ.get("PRC_WEATHER", f"{DATA}/weather")   # METAR csv folder ({ICAO}_{YYYYMM}.csv)
EXCLUDE = [x for x in os.environ.get("PRC_EXCLUDE_MONTHS", "").split(",") if x]

def train_files():
    return sorted(glob.glob(TRAIN_GLOB))

def _excluded(ts):
    if not EXCLUDE: return pd.Series(False, index=ts.index)
    ym = ts.dt.strftime("%Y-%m"); m = ts.dt.month.astype(str)
    return ym.isin(EXCLUDE) | m.isin(EXCLUDE)

def read_train(columns=None):
    """All labelled rows minus the excluded months (exclusion by MVT_TIME_UTC_mvt calendar month)."""
    cols = None if columns is None else list(dict.fromkeys(list(columns) + ["MVT_TIME_UTC_mvt"]))
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in train_files()], ignore_index=True)
    d = d[~_excluded(d.MVT_TIME_UTC_mvt)]
    return d if columns is None or "MVT_TIME_UTC_mvt" in columns else d.drop(columns="MVT_TIME_UTC_mvt")

def read_target(columns=None):
    return pd.read_parquet(TARGET, columns=columns)

def read_all(columns=None):
    """Training rows followed by target rows (the order every feature step assumes)."""
    return pd.concat([read_train(columns), read_target(columns)], ignore_index=True)
