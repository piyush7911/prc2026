#!/bin/bash
# Download and reduce adsb.lol daily archives (github.com/adsblol/globe_history_YYYY, ODbL 1.0) for every day of a month:
# each day's tar (~3 GB) is streamed, traces inside the ten airports' boxes are kept in data/adsb_lol/<day>.parquet, raw parts deleted.
# Usage: bash scripts/fetch_adsb.sh 2026-02 [2026-06 ...]
PY=${PY:-python}; export PYTHONPATH=src
for M in "$@"; do
  for d in $(seq -w 1 31); do day="$M-$d"
    $PY -c "import datetime; datetime.date.fromisoformat('$day')" 2>/dev/null || continue
    [ -f "data/adsb_lol/$day.parquet" ] && continue
    $PY -m prc.adsb.download "$day"                      # resumable parallel byte-range download into data/adsb_lol_raw/
    for attempt in 1 2 3; do $PY -m prc.adsb.extract --day "$day" --workers 2 && break; echo "[$day] attempt $attempt failed"; sleep 30; done
  done
done
