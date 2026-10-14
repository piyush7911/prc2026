"""Fetch METAR observations (Iowa Environmental Mesonet ASOS archive, open data) for every airport x calendar month present in the
training + target files, into $PRC_DATA/weather/{ICAO}_{YYYYMM}.csv (the IEM ASOS csv columns). Existing files are kept.
Usage: PYTHONPATH=src python -m prc.data.fetch_weather [--dry]"""
import os, subprocess, sys, time
import pandas as pd
from prc import config as C
COLS = ["tmpf", "dwpf", "relh", "sknt", "drct", "gust", "vsby", "p01i", "skyc1", "skyl1", "wxcodes", "feel", "alti"]
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}&" + "&".join(f"data={c}" for c in COLS) +
       "&year1={y1}&month1={m1}&day1=1&year2={y2}&month2={m2}&day2=1&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=M"
       "&trace=T&direct=no&report_type=3&report_type=4")
def months_needed():
    d = C.read_all(["ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]); d = d[d.PHASE_mvt == "DEP"]
    return sorted(set(zip(d.ADEP_mvt, d.MVT_TIME_UTC_mvt.dt.year, d.MVT_TIME_UTC_mvt.dt.month)))
def main():
    out = C.WEATHER; os.makedirs(out, exist_ok=True); todo = []
    have = set()   # airport-months already covered by ANY file (a file may hold a whole year, e.g. EDDF_202501.csv = all of 2025)
    for fn in os.listdir(out):
        if fn.endswith(".csv"):
            w = pd.read_csv(f"{out}/{fn}", usecols=["station", "valid"]); t = pd.to_datetime(w.valid)
            have |= set(zip(w.station, t.dt.year, t.dt.month))
    for apt, y, m in months_needed():
        if (apt, y, m) not in have: todo.append((apt, y, m, f"{out}/{apt}_{y}{m:02d}.csv"))
    print(f"{len(todo)} airport-months to fetch"); 
    if "--dry" in sys.argv: print(todo[:20]); return
    for apt, y, m, fn in todo:
        y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
        for k in range(3):
            try:
                txt = subprocess.run(["curl", "-s", "--max-time", "90", URL.format(st=apt, y1=y, m1=m, y2=y2, m2=m2)], capture_output=True, text=True).stdout   # hard timeout: urllib can hang on a stalled stream
                if not txt.startswith("station"): raise ValueError(txt[:200])
                open(fn, "w").write(txt); print(f"  {fn}: {txt.count(chr(10)) - 1} obs"); break
            except Exception as e:
                print(f"  retry {apt} {y}-{m:02d}: {e}"); time.sleep(10 * (k + 1))
        time.sleep(2)   # be polite to IEM
if __name__ == "__main__": main()
