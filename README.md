# GTA VI Twitch CCU forecast

Forecasts the hourly Twitch viewer counts of Grand Theft Auto VI's
release day (2026-11-19, UTC) for hours 2-24, as five quantiles per
hour, given everything recorded up to the end of hour 1.

![competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fadliferdian74-lgtm/gta6-twitch-ccu-forecast-01/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fadliferdian74-lgtm/gta6-twitch-ccu-forecast-01/reward.svg)

## How it works

`main.py` (Python 3, standard library only, deterministic, no network)
fits its model at runtime from the case file's own pool of past big
launches:

- Each comparable with a complete release day contributes its day
  shape as log ratios `ln(y_h / y_1)` for hours 2-24.
- Comparables are weighted by a Gaussian kernel on three launch-start
  features: log hour-1 viewers, the pre-day peak relative to hour 1,
  and the last pre-day hour relative to hour 1. A rolling-midnight
  console launch (already out in earlier time zones, so its pre-day
  ramp is high) is thereby matched with its own family rather than
  with front-loaded global unlocks.
- Each hour's quantiles are the weighted empirical quantiles of the
  pool's log ratios at that hour, scaled by the case's hour-1 print —
  non-decreasing in the level by construction.

The design was selected by leave-one-out backtests over the
comparables pool and validated end-to-end on the four practice cases,
where it scores a mean log-pinball loss of about 0.223 against the
house persistence baseline's 0.387.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one
`<id>.out` forecast per case to the output directory.
