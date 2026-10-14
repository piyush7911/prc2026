"""Step 07: operator/apron/WTC neighbour clocks, day-so-far disruption, overnight-parked, cold.
Usage: python -m prc.features.f07_disruption IN.parquet OUT.parquet"""
import sys
import pandas as pd, numpy as np, glob, time
t0=time.time()
f=pd.read_parquet(sys.argv[1])
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','AIRCRAFT_OPERATOR_flt','WK_TBL_CAT_flt','STAND_mvt','FLIGHT_mvt']
from prc import config as C
raw=C.read_all(cols)
dep=raw[raw.PHASE_mvt=='DEP'].copy()
dep['t']=(dep.MVT_TIME_UTC_mvt-pd.Timestamp('2025-01-01',tz='UTC')).dt.total_seconds()
dep['naive']=(dep.MVT_TIME_UTC_mvt-dep.AOBT_3_flt).dt.total_seconds()
dep.loc[(dep.naive<120)|(dep.naive>7200),'naive']=np.nan   # keep plausible values only (7200 cap, not 3600, to retain real events)
dep['op']=dep.AIRCRAFT_OPERATOR_flt.astype(str).str[:12].where(dep.AIRCRAFT_OPERATOR_flt.notna(), dep.FLIGHT_mvt.astype(str).str[:3])
dep['apron']=dep.ADEP_mvt+'_'+dep.STAND_mvt.astype(str).str.extract(r'^([A-Za-z]*\d?)')[0].fillna('')
dep['day']=dep.MVT_TIME_UTC_mvt.dt.floor('D')
out={k:np.full(len(dep),np.nan) for k in ['nb_op_naive_mean_m60','nb_op_naive_cnt_m60','nb_apron_naive_mean_m60','nb_wtc_naive_mean_m60','frac_long_taxi_m60','day_naive_p90_sofar','day_frac_long_sofar','day_n_sofar']}
pos=pd.Series(np.arange(len(dep)),index=dep.index)
def win_mean_excl(t, ev_t, ev_v, lo, hi, self_v):
    o=np.argsort(ev_t); et=ev_t[o]; ev=ev_v[o]; cs=np.concatenate([[0.0],np.cumsum(ev)])
    L=np.searchsorted(et,t+lo,'left'); R=np.searchsorted(et,t+hi,'left'); cnt=(R-L).astype(float); sm=cs[R]-cs[L]
    has=~np.isnan(self_v); cnt=cnt-has; sm=sm-np.where(has,self_v,0)
    return np.where(cnt>0, sm/np.where(cnt>0,cnt,1), np.nan), cnt
for apt,g in dep.groupby('ADEP_mvt'):
    p=pos[g.index].to_numpy(); t=g.t.to_numpy(); nv=g.naive.to_numpy(); ok=~np.isnan(nv)
    # airport-level: fraction of departures with naive>2400 in last 60 min (excluding self)
    ind=np.where(ok, (nv>2400).astype(float), np.nan)
    m,c=win_mean_excl(t, t[ok], ind[ok], -3600, 1, np.where(ok, ind, np.nan)); out['frac_long_taxi_m60'][p]=m
    for key,name in [('op','nb_op_naive_mean_m60'),('apron','nb_apron_naive_mean_m60'),('WK_TBL_CAT_flt','nb_wtc_naive_mean_m60')]:
        for kv,gg in g.groupby(key):
            pp=pos[gg.index].to_numpy(); tt=gg.t.to_numpy(); vv=gg.naive.to_numpy(); okk=~np.isnan(vv)
            if okk.sum()==0: continue
            m,c=win_mean_excl(tt, tt[okk], vv[okk], -3600, 1, vv); out[name][pp]=m
            if name=='nb_op_naive_mean_m60': out['nb_op_naive_cnt_m60'][pp]=c
    # day-so-far disruption intensity (departures earlier the same day): p90 of naive, fraction long, count
    for dday,gg in g.groupby('day'):
        pp=pos[gg.index].to_numpy(); o=np.argsort(gg.t.to_numpy()); tt=gg.t.to_numpy()[o]; vv=gg.naive.to_numpy()[o]
        p90=np.full(len(o),np.nan); fl=np.full(len(o),np.nan); nn=np.arange(len(o),dtype=float)
        vals=[]
        for i in range(len(o)):
            if i>=10:
                arr=np.array(vals); arr=arr[~np.isnan(arr)]
                if len(arr)>=5: p90[i]=np.percentile(arr,90); fl[i]=(arr>2400).mean()
            vals.append(vv[i])
        out['day_naive_p90_sofar'][pp[o]]=p90; out['day_frac_long_sofar'][pp[o]]=fl; out['day_n_sofar'][pp[o]]=nn
new=pd.DataFrame(out,index=dep.index); new['MVT_ID_mvt']=dep.MVT_ID_mvt.to_numpy()
f7=f.merge(new,on='MVT_ID_mvt',how='left'); assert len(f7)==len(f)
f7['overnight_parked']=((f7.prev_arr_gap_h>6)|(f7.prev_arr_missing==1)).astype(np.int8)
f7['cold']=(f7.wx_tmpf<=37).astype(np.int8); f7['cold_overnight']=f7.cold*f7.overnight_parked
f7.to_parquet(sys.argv[2]); print('v7 saved', f7.shape, 'in %.0f s' % (time.time()-t0))
print(f7[['nb_op_naive_mean_m60','nb_apron_naive_mean_m60','frac_long_taxi_m60','day_naive_p90_sofar','day_frac_long_sofar','overnight_parked']].describe().T[['count','mean','50%','max']].round(2))
