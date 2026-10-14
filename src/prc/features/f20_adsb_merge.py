"""Step 20: join the per-departure ADS-B features (prc.adsb.features) onto the feature table.
Keeps the 35 ADS-B columns validated in 2025 (configs/v22_ads_cols.json: 29 ground-track timing columns plus six assigned-stand columns) plus ads_join_type. Departures on days without an extracted
adsb.lol file get ads_cov = -1 and are handled as 'uncovered' by the model layer.
Usage: python -m prc.features.f20_adsb_merge <features_q.parquet> <adsb_feats.parquet> <out.parquet>"""
import sys, json, pandas as pd
from prc import config as C
from prc.features._util import log
src, ads, out = sys.argv[1], sys.argv[2], sys.argv[3]
f = pd.read_parquet(src); A = pd.read_parquet(ads)
import os
keep = [c for c in json.load(open(C.CONFIGS / os.environ.get("ADS_COLS", "v22_ads_cols.json"))) if c in A.columns] + ["ads_join_type"]
f = f.merge(A[["MVT_ID_mvt"] + keep], on="MVT_ID_mvt", how="left")
f["ads_cov"] = f.ads_cov.fillna(-1).astype("int16"); f["ads_join_type"] = f.ads_join_type.fillna(0).astype("int8")
f.to_parquet(out)
cov = f.assign(c=f.ads_cov >= 2).groupby([f.is_rank, f.year, f.month]).c.mean().round(3)
log(f"{out}: {f.shape} | share of departures with own ground track:\n{cov.to_string()}")
