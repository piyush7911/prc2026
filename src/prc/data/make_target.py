"""Build a hidden-test-style target file from a LABELLED month (dry run): departures get BLOCK_TIME and TAXITIME blanked exactly like
data/ranking.parquet; arrivals keep all fields. The labels are saved separately for scoring.
Usage: python -m prc.data.make_target data/training_2025-07-01_2025-08-01.parquet outputs/dryrun_2025-07"""
import os, sys
import pandas as pd
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
d = pd.read_parquet(src); rk = pd.read_parquet("data/ranking.parquet", columns=None).iloc[:0]
dep = d.PHASE_mvt == "DEP"
d.loc[dep, ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]].to_parquet(f"{out}/labels.parquet", index=False)
for c in ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]:
    d.loc[dep, c] = pd.NA if not str(d[c].dtype).startswith("datetime") else pd.NaT
missing = [c for c in rk.columns if c not in d.columns]; assert not missing, missing
d[rk.columns].to_parquet(f"{out}/target.parquet", index=False)
print(f"target {out}/target.parquet: {len(d):,} rows ({int(dep.sum()):,} departures, labels blanked); labels -> {out}/labels.parquet")
