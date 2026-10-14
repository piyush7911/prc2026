# Design decisions

This note explains why the pipeline is built the way it is. The README describes what each stage does and how to run it; here each design choice is
stated with the 2025 evidence that led to it. All numbers are 2025 out-of-fold RMSE in seconds (six two-month folds, see the README) unless stated otherwise.

## 1. What the error is made of

Taxi-out is `takeoff − off-block`. Take-off is known for every scored row; the off-block time is not. Two facts about the 2025 labels shape everything else.

* **The squared error is concentrated.** On the final pipeline's 2025 predictions the 100 worst rows carry 20 % of the squared error, the 1,000 worst 32 %, the 10,000 worst 50 %.
  147 rows (records shifted by a day or otherwise huge, about half of them at Rome) carry 14 %. Normal flights (81 % of the squared error) behave like a regression problem; the tail does not.
* **Many recorded off-block times are copies of a plan.** 0.1 % of labels equal the schedule to the second, 1.3 % equal LOBT, 1.8 % equal the Network Manager (NM) actual off-block AOBT_3.
  These rows are part of the scored set and cannot be removed from the target.

Consequences: (a) the model has to learn the normal case well and hedge the tail, not fit it; (b) any gain that comes from a few dozen rows is noise for the next year;
(c) every validation number is reported both on all rows and with the five corrupt day-early records removed, and the decision rule protects against few-row gains (section 4).

## 2. Predict the residual to the best off-block clock

Every learner predicts `y − b`, where `b = takeoff − clock` for the best plausible clock (NM AOBT_3, else LOBT, else EOBT_1, else 1,000 s; Rome rows without NM initial time use
the schedule). The NM clock alone already explains most of the variance. Learning the residual makes the trees spend their capacity on how far a recorded off-block departs from the
clock (queue, stand, de-icing, defects), and the same target works across airports whose taxi levels differ by minutes.

All rows are kept in training, including copied and day-shifted labels. Dropping, down-weighting or capping them made normal flights better and the full
RMSE worse (README, "What did not work"): the scored set contains such rows, so the model must learn when the clock is not to be trusted.

## 3. Lanes by available evidence, not by airport

The error of a row depends mostly on which evidence exists for it. Rows are split by two facts known for 2026 as well:

| Lane | Definition | Final rows | Extra evidence used |
|---|---|---|---|
| covered | NM clock present and the aircraft's own ground track received by adsb.lol | 454,526 | ADS-B stacker (stage 1), runway neighbours (stage 2), stand readout (stage 4) |
| uncovered | NM clock present, no usable ground track | 204,189 | neighbours (stage 2), Istanbul runway queue (stage 2b) |
| unm_nr | no NM clock, outside Rome | 11,393 | ADS-B corrector where the track reaches take-off (stage 1b) |
| unm_rome | no NM clock, Rome | 682 | daily posterior (stage 3) |

The ADS-B stacker is the single largest step (−13.7 s over the four final months): the ground track shows when the aircraft actually started to move, which no planning
field contains. It is applied only where that track exists.

**The assigned stand makes the ground track a precise clock.** "End of the last stop" confuses the stand with taxiway holds, de-icing pads and runway queues. Every
departure record names its stand, so the stand's position can be learned: on 2025, a departure whose ADS-B stop ended within 60 s of its recorded off-block was parked
at its stand, and the median of such parking positions locates the stand (749 stands, median scatter 3 m; positions learned from odd and from even months agree to
1.3 m, so the label only selects clean observations and cannot leak into a feature). The time the aircraft leaves that position is then read for every departure. Where
it exists (about 14 % of departures, the same share in 2025 and 2026) its median error is 39 s, against 172 s for the end of the last stop and 128 s for the NM clock;
the stacker learns when to trust it. This new information reduced the four-month 2025 error by about 1 s with every month better. The label rule is what makes a position a stand: the median of all parking stops at a stand, without it, would add 60,000 departures, but on 2025 labels their times lie 200 to 380 s after the recorded off-block (0 to 10 % within 60 s), because the stop most often seen is the hold after pushback, not the stand (at EGLL these positions lie a median 96 m from the label-selected ones). At EGLL, LEMD and LIRF even the label-selected positions align poorly (5 to 19 % within 60 s): the aircraft is rarely received at the stand there. Arriving aircraft give the stands without
any label: an arrival stops at its stand, and its parking position agrees with the departure-located stand to 2.6 m (median). At EDDF, EDDM, EHAM, LEBL and LSZH
these positions add 180 stands and 15,600 departures with a stand time; at EGLL, LEMD and LIRF they are not used, because the departing aircraft is rarely
received there before pushback and the stand times would be wrong. A tree model blurs a precise clock: on the rows that have this reading, the direct readout (minus the leave time plus an airport offset) has a smaller median error than the full pipeline at EDDF, EDDM, LEBL and LSZH, but huge errors where the last sample at the stand is not the pushback. The last model step (stage 4) therefore moves the prediction towards the readout with a weight that depends only on how well the two agree (about 0.6 within 60 s, 0.42 within 120 s, almost nothing beyond); the weights are the same within 0.03 whichever fold is left out, and every month improves (−0.08 s over the four months). The lanes were first defined by ADS-B coverage alone, and that coverage drifted between 2025 and 2026; defining
"matched" by the NM clock and training stages on all matched rows removed that dependence.

## 4. One gate for every component

Every candidate component is tested the same way before it is kept:

1. It is evaluated on the 2025 out-of-fold predictions **on top of everything accepted before it**, through the complete pipeline including the daily layer and the bounds.
2. Its weight is the pooled twelve-month optimum on a fixed grid, and is checked by choosing it on five folds and scoring the sixth.
3. It is kept only if each of January, February, June and July 2025 improves, the four-month total improves, and the 100 most-improved rows carry less than all of the gain.
   For a change that re-trains the whole pipeline (a new ADS-B extraction), the other rows move by re-training noise alone (about 20 s rms on covered rows), so
   the 100 most-improved rows are counted among the rows whose inputs changed (`prc.validate.compare`); see the README, "Where every setting comes from".
4. Before a 2026 file is built, the size of the change it makes on 2026 is compared with its size on 2025 (ratio at most 1.5).

The months mirror the final test months. Rule 3 exists because of section 1: under squared error a component can "win" a month through two or three Rome rows. The decisions are
re-derived by `scripts/run_final.sh` and stored in `docs/decisions/`.

The gate is strict on purpose. It rejected a neural network partner (February worse), a partner trained without day-early labels (January and February worse), a deeper CatBoost on
the usual target, a multiplicative-target member, monthly level corrections and every Rome-specific rule. An audit of the configuration of 12 October 2026 (each component switched off, weights chosen on five folds and scored on the sixth) reproduced
every decision kept until then; the components admitted afterwards (the mixed-target LightGBM member, the assigned-stand timing, stage 4 and the
arrival-located stands) were each checked fold by fold when they were admitted.

## 5. Base: three learners, decorrelators, and a different target where it helps

The base is two LightGBM models and one CatBoost model on about 240 non-ADS-B features, blended 0.45 / 0.55 (pooled optimum 0.55; 0.50 to 0.60 when each fold is left out).

* **Decorrelators.** Four LightGBM models whose residual is anchored on other clocks (LOBT, none, EOBT_1, IOBT) make different mistakes when the NM clock itself is wrong.
  They enter only where an NM clock exists (weight 0.05 each, −0.32 s): their anchors have no Rome schedule fallback and were harmful on Rome rows.
* **Mixed-target members.** A CatBoost model with a different target: for departures at most 1 h late it predicts the taxi time directly (capped at 2 h); for later ones it predicts
  `y / d`, the fraction of the delay to take-off spent after off-block, weighted by `d²` because the squared error of a long-delay row scales with `d²`. It is a different model
  from the residual learners and is blended 50/50 into the base **only for departures at most 1 h late** (−0.69 s; the same weight in all six nested folds). Used on all rows it gained more (−2.5 s), but the 100 most-improved rows carried 116 % of that gain, so that variant was rejected. Both scopes were fixed
  before the results were seen. It is trained on a GPU (depth 10, 3,000 iterations); the final prediction averages eight seeds.
  The same target with LightGBM heads adds a second member on those departures (weight 0.2, −0.16 s, gain in the uncovered lane). A deeper copy of the existing
  CatBoost member on the usual residual target did not pass: diversity comes from the target, not from size.

## 6. Collective correction (stages 2 and 2b)

Errors cluster in time on the same runway: when the clock of one departure is late, its neighbours' are too. Stage 2 predicts a flight's remaining error from the ADS-B
corrections of covered neighbours on the same runway (±5 to 30 min) and at the airport (±15 to 60 min), with the flight itself excluded. Istanbul has no ground coverage and a
mostly default NM off-block time, so the gap between that clock and take-off is the runway queue; stage 2b learns it from the neighbours' take-off delays and replaces stage 2 there.

## 7. Daily reconciliation with a published total

EUROCONTROL publishes, per airport and day, the total pre-departure delay `DLY_ALL_PRE_2`. On the 2025 labels, 60 × that value equals the sum over the day's departures of
`max(0, d − y)` with `d = takeoff − schedule` (correlation 0.9996, median ratio 1.0001). The day total therefore constrains our predictions jointly. Stage 3:

* treats each row with a delay as a prior `y ~ N(prediction, v)`, with `v` the 2025 error variance of its lane and delay band, and shares the day's discrepancy between rows
  in proportion to `v` (a Kalman update, shrunk by 0.75);
* corrects the bias that the convexity of `max(0, ·)` creates in the predicted day sum;
* for Rome departures more than 1 h late without NM clock, uses the two recording modes of section 1 (off-block at the schedule, or after a hold of about 1,020 s, the 2025 median): an exact
  posterior over the day's combinations with a heavy-tailed (Student-t) likelihood, so a row moves only when the day total is decisive.

Its parameters are fitted on 2025 and validated by refitting without each held-out fold (−8.7 s over the four months, every month better). It is the only step that uses
information about 2026 beyond the features of a row, and that information is a published aggregate, not a label.

## 8. Bounds from how the data set was built

The last step applies limits that held without exception on 2025:

* the organisers attached an NM flight only when `|off-block − LOBT| ≤ 3,606 s` (no exception in 2,062,577 matched 2025 departures), so a prediction is kept inside
  that window (widened by the ±12 s take-off dither of the final file);
* LEMD and LEBL taxi-out never exceeded 3,604 s;
* a departure more than 2 h late is not predicted above its delay to take-off.

The bounds have no fitted parameter: they encode facts that held without exception on 2025, and they change only predictions that violate them.

## 9. What was deliberately left out

* **Rome rules.** Rome's corrupt records can be identified (classifier AUC 0.99) but the size of their error cannot be predicted from the data. Every rule tried gained on 2025 through
  fewer than 100 rows. Rome is handled by the base, the bounds and the daily posterior only.
* **Monthly level calibration** to EUROCONTROL's monthly mean taxi-out: its offset to the competition labels changes with the season (winter de-icing), and no version improved all
  four months.

## 10. Known weaknesses

* LTFM has no usable ground coverage, and Paris and Rome only partial coverage that changes from month to month; their uncovered flights carry the largest
  normal-flight error.
* adsb.lol's receiver network changed between 2025 and 2026; a few receiver-dependent columns remain in the stacker (removing them was worse on 2025).
* The tail (section 1) dominates any single month's RMSE; month-to-month differences of a few seconds are not evidence of a better model.
* A model trained to predict the final pipeline's remaining 2025 error from all features makes held-out months worse in every lane: the present features are used up.
  Further gains need new information about the off-block moment (above all, ground surveillance at the uncovered airports), not more modelling of the same columns.
