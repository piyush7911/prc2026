# Features

The feature table has one row per departure of the ten airports (2,085,047 in the 2025 training months and 670,790 in the scored months) and 282 columns;
arrival records enter through the arrival-side features. Every feature is computed from
fields that exist in the scored file: the recorded off-block time of **departures** and the departure label are never used; take-off and landing times of all movements, the
Network Manager (NM) clocks (`AOBT_3`, `LOBT`, `IOBT`, `EOBT_1`, schedule) and all arrival fields are. Steps run in order with `scripts/build_features.sh`; each step
reads the previous step's parquet and writes the next.

| Step | Module | Adds |
|---|---|---|
| 05 | `f05_base` | calendar, aircraft, stand, runway, airline and destination fields; METAR weather (`wx_*`); traffic counts around the take-off in windows (`dep_*`, `rwy_*`, `arr_*`, `hourly_*`, `day_*`); neighbour NM-clock taxi (`nb_*`); taxi-in of recent arrivals (`arr_taxiin_*`, `arr_excess_*`); leave-one-month-out target encodings (`te_*`, `ter_*`); unimpeded-taxi references (`unimp_*`) |
| 06 | `f06_clock_stand` | raw NM-clock gaps (`naive_raw`, `*_minus_sched`, `aobt3_*` flags) and previous arrival at the same stand (`prev_arr_*`) |
| 07 | `f07_disruption` | neighbour clocks by operator, apron and wake class (`nb_op_*`, `nb_apron_*`, `nb_wtc_*`), day-so-far disruption (`day_naive_p90_sofar`, `day_frac_long_sofar`), overnight-parked and cold-soak flags |
| 08 | `f08_winter` | ATMAP-style freezing class, de-icing interactions (`freeze_x_*`, `atmap_freeze`) |
| 10 | `f10_pushdelay_headway` | push-delay state of other departures (`push_delay_*`), headways between movements on the runway and airport, second-of-minute timestamp precision |
| 12 | `f12_defects` | recording-defect state of the airport and airline from the previous 6 / 24 h of **arrivals**, whose labels are visible in 2025 and 2026 (`arr_defect_*`, `arr_degr_*`) |
| 12q | `f12q_queue` | runway-queue neighbour features: how late the neighbours within ±15/30 min on the same runway take off relative to their planned clocks (`q_*`) |
| 20 | `f20_adsb_merge` | 35 ADS-B timing columns (`configs/v22_ads_cols.json`) + `ads_join_type` from `prc.adsb.features` (`ads_*`); used by the stacker (and stage 1b) only |

Conventions: `_m<N>` / `_p<N>` = window of N minutes before / after the movement; `_cnt` = counts; `naive` = take-off minus NM actual off-block (plausible values only, otherwise missing);
`*_minus_*` = difference of two recorded clocks in seconds. Columns of the table are listed by `python -c "import pyarrow.parquet as pq; print(pq.read_schema('outputs/final/features_ads.parquet').names)"`.

## ADS-B columns (stacker only)

For each departure, the samples of the same aircraft (matched by callsign, flight number or, when the callsign is missing, aircraft address) inside the airport box in the three hours
before take-off, relative to take-off: first sample seen (`ads_first_seen`), first ground sample, first movement above 5 kt (`ads_first_mov`, and the sustained variant `ads_first_mov_sust`),
first displacement above 15 m (`ads_first_disp`), end of the last stationary episode (`ads_dwell_end`), path length and stops of the taxi run, and the same times relative to `AOBT_3`
(`ads_mov_minus_aobt` and similar). `ads_cov` is the coverage code (-1 day not extracted, 0 no samples, 1 low altitude only, 2 ground movement seen, 3 stationary seen).
Assigned-stand timing (second extraction pass, with the stand positions of `prc.adsb.stand_pos`: from 2025 departures whose stop ended at the recorded off-block and, at five airports, from 2025 arrivals, `prc.adsb.arr_stops`): samples within 40 m of the departure's assigned stand
(`ads_as_n`, closest distance `ads_as_dmin`, last such sample `ads_as_last`) and the time the aircraft left the stand (`ads_as_leave`: last sample within 40 m before it is
next seen more than 80 m away), also relative to `AOBT_3` (`ads_asleave_minus_aobt`, `ads_aslast_minus_aobt`). `ads_join_type` records how the track was joined
(callsign or flight number, aircraft address, or the unique unclaimed aircraft lifting off within ±60 s of the take-off).
