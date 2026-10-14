#!/bin/bash
# Full feature chain from raw challenge data (run from the project root that holds data/). One step at a time.
set -e
PY=${PY:-python}; export PYTHONPATH=src
O=${OUT:-outputs}; mkdir -p $O
nice -n 10 $PY -u -m prc.features.f05_base $O/features_base.parquet
nice -n 10 $PY -u -m prc.features.f06_clock_stand     $O/features_base.parquet $O/f06.parquet
nice -n 10 $PY -u -m prc.features.f07_disruption      $O/f06.parquet $O/f07.parquet
nice -n 10 $PY -u -m prc.features.f08_winter          $O/f07.parquet $O/f08.parquet
nice -n 10 $PY -u -m prc.features.f10_pushdelay_headway $O/f08.parquet $O/f10.parquet
nice -n 10 $PY -u -m prc.features.f12_defects         $O/f10.parquet $O/features.parquet
nice -n 10 $PY -u -m prc.features.f12q_queue          $O/features.parquet $O/features_q.parquet
echo "FEATURE CHAIN DONE"
