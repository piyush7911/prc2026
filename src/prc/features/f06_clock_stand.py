"""Step 06: raw NM-clock gaps and previous-arrival-at-stand context.
Usage: python -m prc.features.f06_clock_stand IN.parquet OUT.parquet"""
import sys
import pandas as pd, numpy as np, glob
f=pd.read_parquet(sys.argv[1])
from prc import config as C
cols=['MVT_ID_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','SCHED_TIME_UTC_mvt','EOBT_1_flt','STAND_mvt','ADEP_mvt','ADES_mvt','BLOCK_TIME_UTC_mvt','FLIGHT_ID_mvt','AIRCRAFT_TYPE_mvt','FLIGHT_mvt']
raw=C.read_all(cols)
dep=raw[raw.PHASE_mvt=='DEP'].copy()
dep['naive_raw']=(dep.MVT_TIME_UTC_mvt-dep.AOBT_3_flt).dt.total_seconds()
dep['naive_raw_gt3600']=(dep.naive_raw>3600).astype(np.int8); dep['naive_raw_lt120']=(dep.naive_raw<120).astype(np.int8)
dep['aobt3_minus_sched_raw']=(dep.AOBT_3_flt-dep.SCHED_TIME_UTC_mvt).dt.total_seconds()
# stand context (same as regime.py) so the main model can also use it
arr=raw[(raw.PHASE_mvt=='ARR')&raw.BLOCK_TIME_UTC_mvt.notna()][['ADES_mvt','STAND_mvt','BLOCK_TIME_UTC_mvt','FLIGHT_mvt','FLIGHT_ID_mvt','AIRCRAFT_TYPE_mvt']].rename(columns={'ADES_mvt':'ADEP_mvt','BLOCK_TIME_UTC_mvt':'arr_inblk','FLIGHT_mvt':'arr_flt','FLIGHT_ID_mvt':'arr_fid','AIRCRAFT_TYPE_mvt':'arr_act'}).sort_values('arr_inblk')
dep=dep.sort_values('MVT_TIME_UTC_mvt')
dep=pd.merge_asof(dep, arr, left_on='MVT_TIME_UTC_mvt', right_on='arr_inblk', by=['ADEP_mvt','STAND_mvt'], direction='backward', tolerance=pd.Timedelta('12h'))
dep['prev_arr_gap_h']=(dep.MVT_TIME_UTC_mvt-dep.arr_inblk).dt.total_seconds()/3600
dep['prev_arr_missing']=dep.arr_inblk.isna().astype(np.int8); dep['prev_arr_unmatched']=(dep.arr_fid.isna()&dep.arr_inblk.notna()).astype(np.int8)
dep['prev_arr_same_airline']=(dep.FLIGHT_mvt.astype(str).str[:2]==dep.arr_flt.astype(str).str[:2]).astype(np.int8)
dep['day']=dep.MVT_TIME_UTC_mvt.dt.date; dep['n_dep_stand_day']=dep.groupby(['ADEP_mvt','day','STAND_mvt']).MVT_ID_mvt.transform('size')
new=['naive_raw','naive_raw_gt3600','naive_raw_lt120','aobt3_minus_sched_raw','prev_arr_gap_h','prev_arr_missing','prev_arr_unmatched','prev_arr_same_airline','n_dep_stand_day']
f6=f.merge(dep[['MVT_ID_mvt']+new], on='MVT_ID_mvt', how='left')
assert len(f6)==len(f) and (f6.MVT_ID_mvt.to_numpy()==f.MVT_ID_mvt.to_numpy()).all()
f6['naive_raw_minus_unimp']=f6.naive_raw-f6.unimp_p10_stand_rwy
f6.to_parquet(sys.argv[2]); print('v6 saved', f6.shape, ' naive_raw notna frac', f6.naive_raw.notna().mean().round(4))
