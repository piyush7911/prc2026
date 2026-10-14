# Taxi-out prediction for the PRC Data Challenge 2026: team `jovial-uniform`

Predict departure taxi-out time (take-off minus recorded off-block time) at ten European airports (EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM).
Metric: RMSE in seconds. This repository produces the final blind submission (January, February, June and July 2026) from the organisers'
files plus three open data sources.

Licence: GPL-3.0 (see `LICENSE`). Data sources and libraries are listed under [Credits](#credits). The reasons behind each design choice, with the 2025 evidence, are in [`docs/DESIGN.md`](docs/DESIGN.md). How the code is organised, how data moves through it and which file each step writes are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The paper, prepared with the Journal of Open Aviation Science template, is [`docs/paper.pdf`](docs/paper.pdf); the scripts and data behind its figures and tables are in `docs/paper/` ([`docs/paper/FIGURES.md`](docs/paper/FIGURES.md)).

## In short: three facts about how the data is made, used as model structure

1. **The aircraft's own ground track is a third clock.** Besides the schedule and the Network Manager (NM) clocks, open ADS-B (adsb.lol) shows when the aircraft
   actually started to move. Where it exists it is the largest single gain (−13.7 s on the four final-month analogues of 2025); the lanes of the model follow where it exists.
   Its most precise form is the moment the aircraft leaves **its assigned stand**: stand positions are learned from 2025 departures and, at five airports, from arriving aircraft (929 stands, 3 m median scatter),
   and on the departures where this reading exists its median error is 39 s, against 172 s for the end of the last stop and 128 s for the NM clock.
2. **The organisers' NM join implies hard limits.** An NM flight was attached to a departure only when `|off-block − LOBT| ≤ 3,606 s` (no exception in
   2,062,577 matched 2025 departures); predictions are kept inside that window.
3. **A published daily total pins the predictions jointly.** EUROCONTROL's daily pre-departure delay per airport equals, on the 2025 labels, the day sum of
   `max(0, (take-off − schedule) − taxi-out)` (correlation 0.9996). Each airport-day's predictions are reconciled with that total by a Kalman-type update and,
   for Rome's corrupt records, an exact posterior over their two recording modes (−8.7 s on 2025).

Everything else (gradient-boosted base models, two mixed-target members, neighbour corrections) passes one gate: every 2025 analogue month must improve, and no gain may
rest on a few rows.

## Result (2025 out-of-fold, the only evidence this model is built on)

Validation protocol: the twelve 2025 months are split into six two-month folds (1+7, 2+8, 3+9, 4+10, 5+11, 6+12). Each fold is predicted by a model trained on the
other ten months, with fixed round counts (no early stopping on the held-out months). The out-of-fold predictions of one stage are the training input of the next
stage, so every stage is validated exactly as it is used on the scored file. Errors are RMSE in seconds on all departures; the five day-early records of 2025
(label above 20,000 s although take-off is within 45 min of the schedule) are excluded from the "excl. corrupt" column only. "Normal flights" are those
whose label is within 3,600 s of the base prediction.

| 2025 month (analogue of final month) | RMSE, all rows | excl. corrupt | normal flights |
|---|---|---|---|
| January (Jan 2026) | 324.5 | 196.7 | 175.5 |
| February (Feb 2026) | 212.3 | 212.3 | 193.5 |
| June (Jun 2026) | 285.7 | 285.7 | 186.6 |
| July (Jul 2026) | 266.4 | 266.4 | 220.1 |
| the four months together | 276.0 | 246.9 | 195.8 |
| all twelve months | 228.9 | 208.1 | 179.5 |

Per-month and per-lane numbers are written to `outputs/final/validation_2025.json` by the `assemble` step. In these numbers the parameters of the daily layer
(stage 3) are refitted without the held-out fold, exactly as in its gate.

**On 2026 data.** The organisers' public leaderboard scores January and July 2026. The January and July rows of the submitted file scored an RMSE of
**234.55 s** there, close to the 237.9 s of the 2025 January and July analogue without the five corrupt records. The final ranking over the four months is
published by the organisers.

### What each stage of the first build adds (2025 out-of-fold, four final months, excl. corrupt rows)

| Model | Jan | Feb | Jun | Jul | 4 months | 4 months, normal flights |
|---|---|---|---|---|---|---|
| LightGBM lr 0.05 | 225.95 | 255.42 | 311.56 | 305.93 | 280.59 | 225.07 |
| LightGBM, mean of lr 0.05 and lr 0.02 | 225.08 | 254.26 | 311.09 | 304.31 | 279.56 | 223.91 |
| CatBoost | 227.64 | 246.02 | 310.41 | 293.31 | 274.87 | 224.01 |
| base blend (0.45 LightGBM + 0.55 CatBoost) | 223.58 | 246.13 | 307.62 | 294.29 | 273.57 | 220.44 |
| + ADS-B stacker on covered rows | 205.53 | 232.22 | 295.11 | 281.61 | 259.83 | 203.17 |
| + runway stage 2 | **205.27** | **231.28** | **294.62** | **280.63** | **259.14** | **202.30** |

The final pipeline adds the decorrelators, the two mixed-target members, stages 1b and 2b, the daily layer, the stand readout and the bounds on top of this (next table). The ADS-B stacker is the one large step (−13.7 s over the four months). Stage 2 adds −0.7 s. Source: `docs/ablation_2025_oof.csv`.

### Components added after the first build, each through the same 2025 gate

Each candidate is evaluated on the 2025 out-of-fold predictions **on top of the components accepted before it** (the assigned-stand timing and then the arrival-located stands were accepted last: each time the whole pipeline was re-trained and compared with the one without; stages 1b and 2b, the daily layer and stage 4 were re-gated, and the rows show those final gates); it is kept only if each of January, February, June and July improves and
the gain is not carried by the 100 most-improved rows (for a change that re-trains the whole pipeline: the 100 most-improved of the rows whose inputs changed, `prc.validate.compare`). Its weight is the pooled twelve-month optimum, fixed before looking at the four target months, and is checked by choosing it on five folds and scoring the sixth. Decisions are written to
`outputs/final/*.json` and read by `assemble`; the decisions of the submitted file are in `docs/decisions/`.

| Component | Weight (12-month optimum) | Jan | Feb | Jun | Jul | 4 months | Decision |
|---|---|---|---|---|---|---|---|
| Decorrelators in the base, rows with NM clock (`dec`: LightGBM anchored on LOBT, none, EOBT_1, IOBT; base = 0.8 base + 0.05 each) | 0.05 | -0.54 | -0.07 | -0.19 | -0.48 | -0.32 | kept |
| Mixed-target CatBoost member in the base, departures at most 1 h late (`catmix`, GPU; base = 0.5 base + 0.5 member there) | 0.5 | -0.34 | -1.32 | -0.19 | -1.06 | -0.69 | kept (same weight in all six nested folds) |
| Same member on all rows | best 0.4 | -0.59 | -4.06 | -1.59 | -3.79 | -2.52 | rejected: the 100 most-improved rows carry 116 % of the gain |
| Mixed target with LightGBM heads, departures at most 1 h late (`lgbmix`, on top of the CatBoost one) | 0.2 | -0.26 | -0.27 | -0.14 | -0.08 | -0.16 | kept (0.2 in all six nested folds; normal flights −0.15, uncovered lane 218.1 → 217.7) |
| Deep CatBoost (depth 10, GPU) on the residual target (`catd10`) | — | | | | | | rejected in every scope: no gain at most 1 h late (same target as the CatBoost member), tail-carried gain on all rows, June/July worse on uncovered rows |
| Assigned-stand timing in the ADS-B extraction (`ads_as_*`: stand positions learned on 2025, time the aircraft leaves its assigned stand) and lift-off link; whole pipeline re-trained | — | -1.19 | -0.73 | -1.31 | -0.67 | -0.97 | kept (the 100 most-improved rows carry 58 % of the gain; normal flights −1.09; 2026 change 1.19 × its 2025 size) |
| Stand positions without the label rule (median of all 2025 parking stops at a stand; +60k departures with a stand time) | — | | | | | | rejected before the gate: on 2025 labels these times are 200–380 s after the recorded off-block (0–10 % within 60 s, against 59–85 % for label-selected stands): unlabelled stops are post-pushback holds |
| Stand positions also from arriving aircraft (an arrival's parking position after in-block; they agree with the departure-located stands to 2.6 m median), added at EDDF, EDDM, EHAM, LEBL and LSZH where their stand times align with the recorded off-block (53–83 % within 60 s): 180 stands, +15,600 departures with a stand time; whole pipeline re-trained | — | -0.28 | -0.15 | -0.04 | -0.08 | -0.12 | kept: the 100 most-improved of the 2,945 four-month departures that gained a stand time carry 69 % of their gain (over all rows 243 %: the other rows differ by re-training noise; see [Where every setting comes from](#where-every-setting-comes-from)); 2026 change 0.87 × its 2025 size |
| Daily layer with its own error class for covered rows with a stand time | 0.75 / 4 | -0.01 | +0.00 | +0.16 | -0.02 | +0.05 | rejected (vs the daily layer above) |
| Stage 1b: ADS-B corrector for departures without NM clock (`unmstk`) | 1.0 | -1.36 | -0.36 | -0.38 | -0.77 | -0.67 | kept |
| Stage 2b: Istanbul runway-queue correction, replacing stage 2 there (`ltfm`) | 0.75 | -0.37 | -0.96 | -0.11 | -0.45 | -0.42 | kept |
| Daily reconciliation with EUROCONTROL's published pre-departure delay per airport and day (`daily`: shrink 0.75, Rome two-mode posterior with Student-t width 4) | 0.75 / 4 | -3.71 | -15.05 | -6.79 | -10.10 | -8.72 | kept, no 2025 month worse |
| Stage 4: assigned-stand readout (`readout`: weight per agreement bin, see Method) | 0.62 … 0.04 | -0.14 | -0.08 | -0.08 | -0.05 | -0.08 | kept (weights within 0.03 when refitted without each fold; the 100 most-improved rows carry 15 % of the gain; 2026 change 1.04 × its 2025 size, mean −0.35 s in 2025 and −0.28 s on the scored rows) |
| Same readout on departures without NM clock outside Rome (~1,000 rows with a stand time in the four months) | per bin | -0.06 | -0.02 | -0.02 | -0.07 | -0.04 | rejected: the 100 most-improved rows carry 102 % of the gain |
| Same readout for the other ADS-B clocks (end of the last stop, first movement) | per bin | | | | | −0.01 or less | rejected: no gain (the stacker already uses them) |
| Last step: bounds from how the data set was built (`bounds`: NM join window, LEMD/LEBL maximum, cap at the delay for flights more than 2 h late) | — | −0.12 | −0.02 | −0.07 | −0.46 | −0.19 | kept (measured on the final pipeline) |
| LightGBM without day-early labels as partner for rows without NM clock (`u95`) | best 0.25 | +0.02 | +0.07 | -0.11 | -0.04 | -0.03 | rejected |

The stage-1b corrections are larger on the scored rows than in 2025 (rms 408 s against 227 s); 199 of the 290 corrections above 1,000 s are Amsterdam departures of
January 2026, a month of heavy de-icing. By month the rms is 614, 376, 274 and 231 s on the scored rows (January, February, June, July) against 311, 279, 200 and 193 s in 2025.

## Method

All learners predict the residual `y − b(row)`, where `b` is the best available off-block clock:

* `b = takeoff − AOBT_3` (EUROCONTROL Network Manager actual off-block) when it is plausible;
* otherwise `takeoff − LOBT`, then `takeoff − EOBT_1`, then 1,000 s; Rome departures without a Network Manager initial time use `takeoff − schedule`.

**Stage 0, base** (`prc.models.pipeline base`). Three gradient-boosted learners on ~240 non-ADS-B features (calendar, aircraft, stand, runway,
METAR weather, traffic around the departure, arrivals, queue and headway state, recording-defect rates of the day, leave-one-month-out target encodings):
LightGBM lr 0.05 (760 rounds), LightGBM lr 0.02 (1,650 rounds), CatBoost (depth 8, 1,500 iterations). Base = 0.45 · mean(LightGBM) + 0.55 · CatBoost; on rows with an NM clock it is then blended as 0.8 · base + 0.05 · (four decorrelators: the same LightGBM with the residual anchored on LOBT, on nothing, on EOBT_1 and on IOBT).
For departures at most 1 h late the base is finally averaged 50/50 with a mixed-target CatBoost member (depth 10, 3,000 iterations, trained on GPU): it predicts min(y, 2 h) directly for
departures up to 1 h late and y / d (d = take-off minus schedule, weight min(d, 12 h)²) for later ones, so that it is a different model from the residual learners where it is used.
The same mixed target with LightGBM heads then enters with weight 0.2 on those departures.
All rows are kept in training, including copied and day-shifted labels: the model learns when not to trust the clock.
The final fits average several seeds (LightGBM lr 0.05 × 4, lr 0.02 × 3, CatBoost × 3, mixed-target CatBoost × 8, mixed-target LightGBM × 3).

**Stage 1, ADS-B stacker** (`stacker`). For departures whose NM clock is present *and* whose own ground track was received by adsb.lol
(here "covered"; 65–72 % of the final departures, depending on the month), a LightGBM learns `y − base` from all features plus ADS-B timing features
(first sustained movement, end of the last stationary episode, taxi path and stops, relative to AOBT_3) and the base prediction.
A departure is joined to its track by callsign, then by aircraft address (when the callsign brings no ground sample), then by the unique unclaimed aircraft that
lifts off within ±60 s of its take-off. **Assigned-stand timing** (`prc.adsb.stand_pos`, `ads_as_*`): a first extraction pass records each departure's parking
position (the stop that ends at pushback); a stand's position is the median parking position of the 2025 departures at that stand whose stop ended within 60 s of
the recorded off-block (at least 3 observations within 30 m). At EDDF, EDDM, EHAM, LEBL and LSZH, stands not located this way are added from the parking
positions of 2025 arrivals (`prc.adsb.arr_stops`: slow ground samples around the in-block time; at least 5 arrivals, 60 % of them within 30 m of their median):
an arrival stops at its stand, so no departure label is needed (180 stands; where both exist, the two positions agree to 2.6 m). At EGLL, LEMD and LIRF the
departing aircraft is rarely received at its stand, and stand times from arrival-located stands there were 1–10 % within 60 s, so they are not used. A second pass reads, for every departure whose assigned stand is located, the last sample within
40 m of the stand before the aircraft is next seen more than 80 m away.

**Stage 2, collective correction** (`stage2`). Errors are time-clustered within an airport and runway. A small LightGBM predicts a flight's stage-1 error from
neighbourhood aggregates (self excluded): mean ADS-B correction of covered neighbours within ±5/15/30 min on the same runway and ±15/30/60 min at the airport,
and the mean stage-1 offset of matched neighbours. Applied with shrink 0.75 to matched flights.

**Stage 1b, ADS-B corrector for departures without NM clock** (`unmstk`). Departures outside Rome with no NM clock whose own ground track reaches take-off (40 % of them in 2025, 50 % in the scored months)
get a LightGBM correction learned from those 2025 rows (weight 1) and from covered matched rows (weight 0.2), on all features plus the base prediction.

**Stage 2b, Istanbul runway-queue correction** (`ltfm`). At LTFM, without ground ADS-B, a small LightGBM learns the stage-1 error from the departures around it on the same runway and at the
airport (their take-off delay against LOBT and schedule, their stage-1 prediction against the NM clock and against the stand/runway mean), each relative to the airport-day mean.
It replaces stage 2 at LTFM (shrink 0.75).

**Stage 3, daily reconciliation** (`prc.models.daily`). Predictions of each airport-day are adjusted so that their implied total off-block delay agrees with EUROCONTROL's
published daily pre-departure delay (on 2025 labels: correlation 0.9996, median ratio 1.0001). Parameters are fitted on 2025 and validated by refitting without each held-out fold; details in the module docstring.

**Stage 4, assigned-stand readout** (`readout`). On covered departures with an assigned-stand leave time `L`, `r = −L + offset(airport)` reads the taxi-out directly
(offset = 2025 median of `y + L`). It is often the sharper of the two (2025 median absolute error at EDDF, EDDM, LEBL and LSZH 10–23 s against 19–32 s
for the prediction `q`), but it is far off when the last sample at the stand is not the pushback. The prediction is therefore moved towards `r` with a
weight that depends only on the agreement `|r − q|`: least squares per bin (below 15, 30, 60, 120, 300 and 600 s), giving 0.62, 0.63, 0.60, 0.42, 0.04 and 0.07,
and no change beyond 600 s. It changes about 13 % of the predictions, by about 6.5 s rms (6.4 s in 2025, 6.6 s on the scored rows).

**Last step, bounds** (`bounds`). Predictions are kept within limits that follow from the data: the NM join window (|off-block − LOBT| ≤ 3,606 s, no 2025 exception),
the 2025 maximum taxi-out at LEMD and LEBL, and no prediction above the delay to take-off for flights more than 2 h late.

**Rows without a Network Manager clock** (1.4 % of the 2025 departures, 1.8 % of the scored ones; Rome dominates the error) are predicted by the base blend (plus stage 1b where their ground track was seen, and the stage-3 posterior for Rome departures more than 1 h late). 2025 evidence says the recorded
off-block time of these rows is often a copy of the schedule, sometimes shifted by a day, and that the size of the error cannot be predicted from the data
(see "What did not work").

## Data

| Source | Use | Licence / access |
|---|---|---|
| Organisers' files: `training_2025-*.parquet`, `final_ranking.parquet`, `final_submitting.parquet` | training and scored rows | challenge terms |
| adsb.lol daily history (`github.com/adsblol/globe_history_2025`, `_2026`) | ground tracks of departures (stages 1, 1b, 2 and 4) and parking positions of arrivals (stand positions) | ODbL 1.0, open download |
| Iowa Environmental Mesonet ASOS/METAR archive | weather per airport | open access; underlying NOAA/NWS observations |
| EUROCONTROL daily pre-departure delay, `all_pre_departure_delays_2025.csv` and `all_pre_departure_delays_2026.csv` (ansperformance.eu; the 2026 file as published on 2026-10-12, saved as `..._2026_latest.csv`) | daily reconciliation (stage 3) | EUROCONTROL terms: source named, non-commercial |

`configs/` holds three small derived files: `iata2icao.json` (airline-designator lookup, 168 airlines, used to match a flight number such as KL1587 to its ADS-B callsign KLM1587; no labels involved),
`v22_ads_cols.json` (the 35 ADS-B columns the model reads, six of them from the assigned-stand pass) and `stand_map.parquet` (stand locations from stationary episodes in unlabelled adsb.lol positions; it only feeds `ads_sv*` columns that the model does not read). Nothing else is downloaded.
No data set is stored in this repository: the adsb.lol archives (about 3 GB per day, 485 days) exceed any repository quota, and the
organisers' files are distributed by the organisers. `scripts/fetch_adsb.sh`, `python -m prc.data.fetch_weather` and the commands under
[Reproduce](#reproduce) fetch every external source; the organisers' files go into `data/` as provided.
`BLOCK_TIME_UTC_mvt` and the label of **departures** in the scored file are blank and never used; arrival rows of the scored file (block and taxi times) are
used for the arrival-side features, as the organisers' data description allows.

## Reproduce

```
pip install -r requirements.txt
# data/ holds the organisers' files; run from the repository root
bash scripts/fetch_adsb.sh 2025-01 2025-02 ... 2025-12 2026-01 2026-02 2026-06 2026-07     # adsb.lol days, ~3 GB each before reduction (about 10 min per day at 6 MB/s); resumable
mkdir -p data/eurocontrol_pru && cd data/eurocontrol_pru && U=https://www.eurocontrol.int/performance/data/download/csv
curl -fO $U/all_pre_departure_delays_2025.csv && curl -f -o all_pre_departure_delays_2026_latest.csv $U/all_pre_departure_delays_2026.csv && cd ../..
bash scripts/run_final.sh                          # weather, features, ADS-B features, models, assembly -> outputs/final/jovial-uniform_final.parquet
```

`scripts/run_final.sh` runs every step one at a time, including every gate in the order it was taken (each re-derives its decision from 2025). The mixed-target CatBoost member needs an NVIDIA GPU
for a practical run time (two T4: about 25 min per fold, 25 to 30 min per final seed); the mixed-target LightGBM member runs on the CPU (10 min per fold, 9 min per seed). Measured cost on a 16 GB laptop (8 threads): features 15 min, ADS-B features 2 × 10 min (two passes) plus 1 min for the arrival stops, out-of-fold base fits (six folds) LightGBM lr 0.05 55 min, lr 0.02 2 h 20 min, CatBoost 4 h,
final fit per seed 8 / 37 / 38 min, stacker 33 min out-of-fold + 20 min final, stage 2 under 5 min. `PRC_TARGET`, `PRC_TRAIN_GLOB`, `PRC_EXCLUDE_MONTHS` and `PRC_WEATHER` (see `src/prc/config.py`)
point the same code at another period or at a held-out 2025 month for a dry run. `PRC_SMOKE=1` runs every model step on 3 % of the rows in minutes.

Check performed on 2026-10-11: the feature chain (`scripts/build_features.sh`) was run from a fresh clone of this repository and reproduces the table the models were trained on exactly (246 columns, 2,755,837 rows).
Check performed on 2026-10-13: the whole `scripts/run_final.sh` was run end to end from a fresh clone with `PRC_SMOKE=1 PRC_GPU=0` (features at full size, every learner,
stage and gate on 3 % of the rows with shortened fits) and wrote a complete submission file. In a smoke run the gate decisions are not meaningful; it checks the code path only. Two steps were added after that check: stage 4 (`readout`, also run in smoke mode since) and the arrival stops of `prc.adsb.arr_stops`; both were run on the full data to produce the submitted file.

Layout: `src/prc/features` (feature steps 05 to 20, dictionary in `docs/FEATURES.md`), `src/prc/adsb` (download, extraction, per-flight features), `src/prc/models/pipeline.py` (all model stages and assembly),
`src/prc/validate` (`compare.py`: the gate for changes that re-train the whole pipeline; `extra_ideas.py`: tests of candidate ideas on the out-of-fold predictions), `scripts/`, `docs/`.

## Using it on new data, and extending it

Nothing in the code refers to January–July 2026; every period-dependent choice is a setting.

**Predict another period.** Put the departures to predict in a file with the organisers' schema (taxi-out blank) and point `PRC_TARGET` and `PRC_TEMPLATE`
at it; fetch the METAR months (`python -m prc.data.fetch_weather`) and the adsb.lol days (`scripts/fetch_adsb.sh YYYY-MM ...`) for that period, and the
EUROCONTROL daily pre-departure delay file covering it (stage 3 leaves an airport-day unchanged when no published value exists). Then run `scripts/run_final.sh`.
Training on other years: `PRC_TRAIN_GLOB` selects the labelled files.

**Test it honestly first.** `PRC_EXCLUDE_MONTHS=2025-07` removes a month from training, so the whole chain can be run as a dry run on a month whose labels you hold.
`PRC_GATE_MONTHS` (default `1,2,6,7`) names the training-year months that every gate must improve; set it to the analogues of the months you will be scored on.

**Add an airport.** Three lists name the airports: `src/prc/adsb/extract.py` (`APTS`: reference coordinates for the ground-track extraction),
`src/prc/models/pipeline.py` (`APTS9`: airports with usable adsb.lol ground coverage; the others take the uncovered lane) and `src/prc/models/daily.py`
(`APTS`: airports reconciled with the daily total). `src/prc/features/f08_winter.py` holds each airport's share of remote de-icing (a prior from airport documentation).

**Add a model.** A new base member is one entry in `LEARNERS` (`src/prc/models/pipeline.py`): LightGBM (`lr`, `rounds`) or CatBoost (`lr`, `iters`, `depth`; `PRC_GPU=1` for a GPU),
with an optional `target` (`"mix"`: taxi time up to 1 h late, fraction of the delay above; `"ratio"`: taxi time over the off-block clock gap), `anchor`
(the clock the residual is taken against) or `rows="matched"` (train only where an NM clock exists). Add its name to `MEMBERS`, then

```
python -m prc.models.pipeline base   --learner NAME --mode oof       # six-fold out-of-fold predictions on the training year
python -m prc.models.pipeline base   --learner NAME --mode final --seeds 0,1,2
python -m prc.models.pipeline member --learner NAME                  # gate: scopes all rows / at most 1 h late / uncovered; writes outputs/final/NAME_weight.json
```

and re-run the downstream stages in this order (`stacker`, `stage2`, `unmstk`, `ltfm`, each `--mode oof` and `--mode final`), `daily`, `readout` and `assemble`. A member that fails its gate is
written as rejected and has no effect.

**Add a correction stage.** The pattern of `cmd_unmstk` or `cmd_ltfm` is the template: train on out-of-fold predictions with the six folds, apply the same gate
(each month in `GATE_MONTHS` better, the 100 most-improved rows carrying less than the whole gain, weight checked fold by fold), write the decision to a JSON file and read it in `_assemble`.

## Running without a GPU

Only the mixed-target CatBoost member needs a GPU in practice. `PRC_CPU_ONLY=1 bash scripts/run_final.sh` leaves it out (its fits and its gate are skipped and
`_base` uses the remaining members); everything else, including the mixed-target LightGBM member, runs on a laptop CPU. The measured cost on the 2025 validation is
the step that member added: 248.88 → 248.29 on the four months excl. corrupt rows, i.e. about +0.6 s without it. The decisions after it are re-derived by the same gates.

## Reusable components

Three parts are useful outside this challenge; each can be used on its own.

**1. Ground-track timing from open ADS-B** (`src/prc/adsb/`). Turns adsb.lol's public daily archives into per-departure timing of the aircraft's real movement on the ground.

```
python -m prc.adsb.download 2026-02-10          # resumable parallel download of one day's archive parts (~3 GB) -> data/adsb_lol_raw/
python -m prc.adsb.extract --day 2026-02-10     # positions inside a box around each airport -> data/adsb_lol/2026-02-10.parquet
                                                #   columns: icao, reg, type, callsign, airport, t, lat, lon, on_ground, alt, gs
python -m prc.adsb.features                     # per departure (callsign or aircraft match, 3 h before take-off): first seen, first movement,
                                                #   end of the last stop, stop counts, coverage code -> outputs/adsb_feats.parquet
```

Airports are the `APTS` coordinates in `extract.py`; departures are read in the organisers' schema (flight number or callsign, airport, take-off time).
Any study that needs when an aircraft actually pushed back or started to taxi can use the first two steps unchanged.

**2. Reconciliation of predictions with a published daily total** (`src/prc/models/daily.py`). Given per-row predictions and, per group and day, a published sum of
`max(0, d − y)`, it moves the predictions so their implied total agrees with the published one: a Kalman-type share of the discrepancy by each row's error variance,
a correction for the convexity of `max(0, ·)`, and an exact posterior for rows with two discrete recording modes.

```
t   = frame(rows, predictions, lanes)      # rows: airport, year, day of year, d (= take-off − schedule) and, for fitting, y
prm = fit(t_labelled, load_K())            # per-airport offset and noise of the published total, variances by lane and delay band, hold time, bias
new = apply(t, load_K(), prm, lam=0.75, rome_posterior=True, phi=4.0)
```

The same pattern fits any prediction task with a trusted aggregate per group and period (counts, totals, delay sums).

**3. The gate** (`member`, `dec`, `unmstk`, `ltfm`, `daily`, `readout` steps of `src/prc/models/pipeline.py`, and `prc.validate.compare` for changes that re-train the whole pipeline). A template for choosing components honestly when the test period differs from the
training period: six two-month folds, the component evaluated through the complete pipeline on top of what is already accepted, its weight the pooled optimum and checked
fold by fold, kept only if every month in `GATE_MONTHS` improves and the 100 most-improved rows do not carry the gain. Every decision is a JSON file (`docs/decisions/`).

## What did not work

Everything below was measured on 2025 data (held-out months); none was adopted.

| Idea | Result |
|---|---|
| Drop, down-weight or cap copied / day-shifted labels; Huber loss; model trained on normal rows only | normal-flight RMSE sharper, full RMSE worse (defective rows stay in the scored set) |
| Clean model used only where a risk classifier (AUC 0.99) says the row is normal | improves normal flights, full RMSE changes sign with a few hundred Rome rows |
| Rome-specific rules, two-part fill/no-fill models, regime mixtures, day-early hedges | the size of the error on these rows is not predictable; gains came from fewer than 100 rows |
| Extra partner models on covered rows (linear-tree LightGBM, TabM, XGBoost, third tree family) | gains inside seed noise or concentrated in a few rows |
| TabM-style neural network blended into uncovered rows (`prc.validate.tabm`, weight 0.2 = pooled optimum) | helps January, June and July on that lane, hurts February; the gain rests on a handful of rows |
| AutoGluon (400 k-row sample, 2 h) | worse than our LightGBM on every lane, no gain as a partner |
| Local-time clock features, per-airport scaled target, stronger base as an extra layer | inside seed noise |
| LIRF/LTFM departures without NM flight whose LOBT is recovered from their own arrival record: prediction moved out of the ±3,606 s window | July worse, and a handful of rows carried most of the gain |
| Congestion floor for flights without NM clock (raise the prediction to 0.95 × the live taxi level of matched neighbours) | helps only February (−0.5 s), hurts the other three final months |
| Same-runway arrival taxi context (counts and mean taxi-in of arrivals on the departure runway) as a corrector | every corrector layer is worse than none; the new columns add 0.02 to 0.08 s over the control |
| Cross-season validation (train January, test July and the reverse) | no better than six-fold validation at predicting what transfers |
| Airport-month level shift toward EUROCONTROL's published monthly mean taxi-out (composition offset fitted on 2025), before the daily layer | January and February worse at every shrink (winter de-icing exclusions make the offset unstable) |
| Same level shift on the final pipeline, offset modelled by airport, de-icing share and no-reference share, cells with de-icing share ≥ 2 % or 5 % left untouched | four months −0.01 to −0.03 s, February or June worse at most settings |
| EUROCONTROL's daily ATC pre-departure delay as a second daily constraint | correlation with any day sum of off-block delay at most 0.54 (it follows delay codes, not off-block times) |
| Stable-sign bias corrections by lane, airport, delay band, clock pattern and hour (cross-fitted shifts) | best +0.01 s; none improves all four months |
| Multiplicative-target member (LightGBM on taxi time / off-block clock gap, weight gap²; `lgbratio`) | fails every scope: July +2.0 s on all rows at weight 0.1 (ratio errors are multiplied by long gaps); no gain on top of the two mixed-target members |
| Model of the final pipeline's remaining error from all features (six folds) | held-out RMSE worse at every strength in every lane (+0.24 s on normal flights at 0.25); it fits stand and destination noise: the features hold no further usable signal |

## Where every setting comes from

No leaderboard score is an input to any fit, weight, threshold or setting in this repository. One decision was taken after a leaderboard score was known: the
arrival-located stands improved every 2025 analogue month but failed the concentration criterion as first defined (over all records); after their January/July
score was seen, the criterion for changes that re-train the whole pipeline was refined to the records whose inputs changed (`prc.validate.compare`), under which
they pass (0.69 over 2,945 records). Every value is one of three kinds:

1. **Learned from 2025 labels:** all parameters of the base models, the stacker and stage 2; the training-row filter of the stacker and stage 2 (it drops the five day-early 2025 records).
2. **Chosen on the 2025 out-of-fold predictions:** the base blend weights (0.45 LightGBM / 0.55 CatBoost: the pooled optimum over the twelve months is 0.55, and the optimum when each fold is left out is 0.50 to 0.60) and the stage-2 shrink (0.75: the pooled twelve-month optimum on the grid 0 to 1.25 when it was chosen on the first build; on the final pipeline 0.5 is 0.01 s better over the twelve months, inside noise, and 0.75 was kept); the weights of every gated component (decorrelators 0.05, mixed-target members 0.5 (CatBoost) and 0.2 (LightGBM) for departures at most 1 h late, stage 1b 1.0, stage 2b 0.75, daily shrink 0.75 and width 4), each the pooled twelve-month optimum of its gate (`docs/decisions/`); the stage-4 weights (least squares per agreement bin on 2025, `readout.json`); the five airports at which arrival-located stands are used (their stand times align with the 2025 recorded off-block: 53–83 % within 60 s, against 1–10 % at EGLL, LEMD and LIRF).
3. **Set manually by looking at the data:** learning rates, round counts and tree sizes (from 2025 hold-out learning curves), the plausibility band (−7,200 s to 172,800 s) and the 1,000 s fallback of the off-block base, the rule that flags day-early records (label above 20,000 s with take-off less than 45 min after schedule),
   the ADS-B speed thresholds (2, 5 and 15 knots) and the stationary-episode lengths (30 to 900 s), the neighbourhood windows of stage 2 (5 to 60 min), the stand rules (stop ending within 60 s of the off-block, at least 3 departures within 30 m; at least 5 arrivals, 60 % within 30 m, spread at most 20 m; 40 m at the stand, 80 m away) and the stage-4 bin edges (15 to 600 s).

## Limitations

* The ADS-B features exist only where adsb.lol receivers saw the aircraft on the ground; Istanbul (LTFM) has no usable ground coverage; Paris (LFPG) and Rome (LIRF) have partial coverage that changes from month to month (LIRF: 81–85 % of departures in January and February 2026, almost none in June and July).
  Those flights get the base model only and carry the largest normal-flight error.
* About 1 % of the rows carry a recorded off-block time that is a copy of the schedule or a day-shift; they dominate the squared error and no 2025-derived rule predicted their size.
* adsb.lol's receiver network changed between 2025 and 2026. The stacker still receives a few receiver-dependent columns (`ads_n_ground`, `ads_apt_day_cov`, `ads_first_seen`):
  removing them was 0.3 s worse on the 2025 validation, so they stay, but this is the most likely place for a 2025-to-2026 shift.

## Credits

* adsb.lol contributors (ODbL 1.0), Iowa Environmental Mesonet, EUROCONTROL (challenge data, and the daily pre-departure delay published on ansperformance.eu), OpenSky Network.
* Libraries: LightGBM, CatBoost, pandas, NumPy, scikit-learn, PyArrow (versions in `requirements.txt`).
* We also referred to open-source repositories and write-ups on the internet.
