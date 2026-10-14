#!/bin/bash
# End-to-end reproduction of the final submission from raw challenge data + open data (run from the project root that holds data/).
# Every heavy step runs one at a time (8 threads). Environment overrides: PRC_TARGET, PRC_TEMPLATE, OUT, PY, PRC_GPU, PRC_CPU_ONLY.
#   data/training_2025-*.parquet, data/final_ranking.parquet, data/final_submitting.parquet   (challenge bucket)
#   data/weather/*.csv     <- prc.data.fetch_weather   (Iowa Environmental Mesonet METAR archive)
#   data/adsb_lol/*.parquet <- scripts/fetch_adsb.sh    (adsb.lol globe_history daily archive, ODbL 1.0)
#   data/eurocontrol_pru/all_pre_departure_delays_{2025,2026_latest}.csv  (ansperformance.eu, daily pre-departure delay)
# The gates (dec, catmix, unmstk, ltfm, u95, daily, readout) re-derive every decision from the 2025 out-of-fold predictions and write it to
# $OUT/*.json; the decisions of the submitted file are copied in docs/decisions/. The catmix member is meant for an NVIDIA GPU
# (PRC_GPU=1; two T4 GPUs: about 25 min per fold); on CPU it runs but takes many hours.
set -e
PY=${PY:-python}; export PYTHONPATH=src
export PRC_TARGET=${PRC_TARGET:-data/final_ranking.parquet} PRC_TEMPLATE=${PRC_TEMPLATE:-data/final_submitting.parquet}
export OUT=${OUT:-outputs/final}; export PRC_OUT=$OUT PRC_FEAT=$OUT/features_ads.parquet PRC_EC_DIR=${PRC_EC_DIR:-data/eurocontrol_pru}; mkdir -p $OUT
M="$PY -u -m prc.models.pipeline"
step() { echo "[$(date +%H:%M)] $*"; }
seeds_of() { case $1 in lgb05|lgb05dy) echo 0,1,2,3;; catmix) echo 0,1,2,3,4,5,6,7;; dec*) echo 0;; *) echo 0,1,2;; esac; }   # final fits average these seeds
downstream() {   # stages that read the base: re-trained (and re-gated where they have a gate) whenever the base changes
  for s in stacker stage2 unmstk ltfm; do step "$s oof"; $M $s --mode oof; step "$s final"; $M $s --mode final; done   # unmstk and ltfm gates assemble the pipeline, so stage 2 comes first
}
step "weather";            $PY -m prc.data.fetch_weather
step "features 05-12q";    OUT=$OUT bash scripts/build_features.sh
export ICAO_LINK=1 LIFTOFF_LINK=1                 # ADS-B joins: callsign, then aircraft address, then the unique lift-off at the take-off time
step "ADS-B features 1";   ADSB_OUT=$OUT/adsb_feats_p1.parquet $PY -u -m prc.adsb.features            # pass 1: parking positions
step "arrival stops";      $PY -m prc.adsb.arr_stops $OUT/arr_stops_2025.parquet                # parking positions of 2025 arrivals
step "stand positions";    $PY -m prc.adsb.stand_pos $OUT/adsb_feats_p1.parquet $OUT/stand_positions.parquet $OUT/arr_stops_2025.parquet
step "ADS-B features 2";   STAND_POS=$OUT/stand_positions.parquet ADSB_OUT=$OUT/adsb_feats.parquet $PY -u -m prc.adsb.features   # + assigned-stand timing
step "merge";              $PY -m prc.features.f20_adsb_merge $OUT/features_q.parquet $OUT/adsb_feats.parquet $OUT/features_ads.parquet
for L in lgb05 lgb02 cat lgb05dy decB decD decE decF lgbmix; do
  step "base $L oof";      $M base --learner $L --mode oof
  step "base $L final";    $M base --learner $L --mode final --seeds $(seeds_of $L)
done
for P in oof final; do     # GPU member (CatBoost, mixed target); PRC_CPU_ONLY=1 skips it (see README, "Running without a GPU")
  [ "${PRC_CPU_ONLY:-0}" = 1 ] && break
  step "base catmix $P";   PRC_GPU=${PRC_GPU:-1} PRC_CATMIX_DEPTH=10 PRC_CATMIX_ITERS=3000 PRC_CATMIX_LR=0.06 \
                           $M base --learner catmix --mode $P $([ $P = final ] && echo --seeds $(seeds_of catmix))
done
rm -f $OUT/dec_weight.json $OUT/catmix_weight.json $OUT/daily.json $OUT/readout.json
downstream                                       # first build on the plain base (stages 1, 1b, 2, 2b with their gates)
step "gate: decorrelators";  $M dec;    downstream
step "gate: daily";          $M daily   # the catmix gate was run with the daily layer in place
[ "${PRC_CPU_ONLY:-0}" = 1 ] || { step "gate: catmix"; $M member --learner catmix; downstream; }
step "gate: lgbmix";         $M member --learner lgbmix; downstream
step "gate: u95";            $M u95
step "gate: daily (again)";  $M daily
step "gate: readout";        $M readout   # stage 4, after the daily layer
step "assemble";             $M assemble
step "DONE"
