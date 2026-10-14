"""Step 08: ATMAP-style freezing class and de-icing interactions.
Usage: python -m prc.features.f08_winter IN.parquet OUT.parquet"""
import sys
import pandas as pd, numpy as np, time
t0=time.time()
f=pd.read_parquet(sys.argv[1])
# ATMAP-style freezing class from METAR columns already joined (wx_tmpf F, wx_dwpf F, wx_* flags)
tC=(f.wx_tmpf-32)*5/9; dC=(f.wx_dwpf-32)*5/9; spread=tC-dC
moist=(f.wx_snow==1)|(f.wx_freezing==1)|(f.wx_fog==1)|(f.wx_mist==1)|(f.wx_rain==1)|(f.wx_drizzle==1)|(spread<3)
frz=(tC<3)&moist
coef=np.zeros(len(f)); coef[frz.to_numpy()]=1
coef[(frz&((f.wx_snow==1)|(f.wx_freezing==1))).to_numpy()]=3
coef[(tC<=-15).to_numpy()]=4
f['atmap_freeze']=coef.astype(np.int8); f['tempC']=tC; f['dp_spread']=spread
remote={'EHAM':1.0,'LFPG':1.0,'EDDM':1.0,'LSZH':0.9,'LTFM':1.0,'EDDF':0.5,'EGLL':0.1,'LIRF':0.3,'LEMD':0.3,'LEBL':0.3}
f['remote_deice']=f.apt.astype(str).map(remote).astype(float)
f['freeze_x_remote']=f.atmap_freeze*f.remote_deice
f['frost_first_wave']=((tC<=1)&(f.overnight_parked==1)&(f.hour<=8)).astype(np.int8)   # hoarfrost pre-deicing (UTC hour ~ local-1)
f['freeze_x_overnight']=f.atmap_freeze*f.overnight_parked
f['freeze_x_widebody']=f.atmap_freeze*f.wtc.astype(str).isin(['H','J']).astype(int)
f['freeze_x_groundtime']=f.atmap_freeze*f.prev_arr_gap_h.fillna(12).clip(upper=12)
# de-icing queue proxy: departures pushed back (AOBT3) in last 60 min under freezing (airport-level count already: dep_push_m30 x freeze)
f['freeze_x_push_m30']=f.atmap_freeze*f.dep_push_m30
f['freeze_x_fraclong']=f.atmap_freeze*f.frac_long_taxi_m60.fillna(0)
f.to_parquet(sys.argv[2]); print('v8 saved', f.shape, 'in %.0f s' % (time.time()-t0))
