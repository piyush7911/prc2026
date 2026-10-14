# Figures of the paper

All seven figures are drawn by code, so they can be regenerated and every number in them is traceable.

```
PYTHONPATH=src python docs/paper/figure_data.py     # collects the numbers into docs/paper/figure_data.json
python docs/paper/make_figures.py                   # draws docs/paper/figures/fig1..fig7 as vector PDF (the figures of docs/paper.pdf)
```

`figure_data.py` needs the pipeline outputs (`outputs/final`), the organisers' 2025 files and one adsb.lol day (set `PRC_DATA` if `data/` is elsewhere).
Its output, `figure_data.json`, is small and committed, so `make_figures.py` runs without the large inputs.

| File | Content | Source of the numbers |
|---|---|---|
| `fig1_pipeline` | inputs, features, stage 0, corrections, bounds, gate | the pipeline as implemented (`src/prc/models/pipeline.py`) |
| `fig2_clocks` | clocks of one real departure (Zurich, June 2025): schedule and planned clocks, NM actual off-block, recorded off-block, ADS-B end of last stop, take-off; `d`, `y`, `b` | organisers' record and adsb.lol track of that departure (anonymised: no flight number, callsign or aircraft) |
| `fig3_lanes` | lane decision tree, 2026 rows per lane, 2025 RMSE per lane and month | `P.lanes` on the 2026 file; `validation_2025.json` |
| `fig4_adsb` | (a) extraction chain; (b) the same departure's ground speed with the extracted events and holds | adsb.lol positions of that day; `prc.adsb.features` output |
| `fig5_daily` | (a) published daily total against the label-implied sum, 3,650 airport-days of 2025; (b) gain of the daily layer per month | EUROCONTROL file and 2025 labels; `daily.json` gate deltas |
| `fig6_gate` | (a) six two-month folds; (b) the gate | protocol as implemented |
| `fig7_error_concentration` | (a) cumulative share of squared error by worst rows; (b) gain of each kept component (on top of those accepted before it) | final 2025 out-of-fold predictions; README component tables |

Style: Helvetica 6 to 9 pt at the printed width (6.4 in), Okabe–Ito colour-blind-safe palette with one colour per pipeline part
(base blue, ADS-B green, neighbours orange, daily layer reddish purple, bounds grey), vector PDF with embedded fonts (the JOAS HTML build converts PDF figures to PNG itself).
