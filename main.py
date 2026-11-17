#!/usr/bin/env python3
"""Forecast a big game launch's first-day Twitch viewers, hours 2-24.

Model (fitted at runtime from the case's own comparables pool):

  * Every past launch in the case file with a complete release day
    contributes its day shape as log ratios r_h = ln(y_h / y_1) for
    h = 2..24, where y_1 is that launch's hour-1 viewers.
  * Launches are weighted by a Gaussian kernel on three features that
    describe how a launch day starts: the log of hour-1 viewers, the log
    ratio of the pre-day peak (max viewers in the 24 h before the day)
    to hour 1, and the log ratio of the last pre-day hour to hour 1.
    The pre-day peak axis is deliberately soft (wide bandwidth): a big
    pre-day audience says a first regional wave or an early-access crowd
    was already there, but which of those it was matters more than the
    peak's exact size, and leave-one-out backtests punish a tight fit
    on it. The hour-1 size and the last pre-day hour carry the signal.
  * Each hour's five quantiles are the weighted empirical quantiles of
    the pool's log ratios at that hour, pooled with the two neighbouring
    hours (a launch day's shape is smooth, and pooling triples the
    effective sample behind every quantile), scaled back by the case's
    own hour-1 print.
  * The fan is then calibrated against the pool itself: leave-one-out
    forecasts inside the pool measure, per day-part (early, middle,
    late), how often each nominal level actually covered and how far
    the median ran from the outcome. The levels are nudged halfway
    toward the levels that would have covered correctly, and each
    day-part is shifted by its measured median bias. On the pool this
    corrects a small systematic underforecast that grows over the day.

The design was chosen by leave-one-out backtests over the comparables
pool and by end-to-end runs on the four practice cases, where it scores
well under the house persistence baseline. It is deterministic: no
randomness, no network, standard library only.
"""

import json
import math
import os
from datetime import datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Gaussian kernel bandwidths per feature, picked on leave-one-out
# backtests and frozen: (feature name, bandwidth in log units).
KERNEL = (("logy1", 1.0), ("premax", 1.8), ("prelast", 0.5))

# Day-parts for the calibration step: (first hour, last hour).
BLOCKS = ((2, 8), (9, 16), (17, 24))
# Coverage levels at which the calibration measures the pool's fan.
CAL_GRID = (0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.90, 0.95, 0.97)
CAL_LEVEL_SHRINK = 0.5    # move nominal levels halfway to calibrated ones
CAL_SHIFT_CLAMP = 0.3     # cap on a day-part's median-bias shift, log units
CAL_MIN_POOL = 8          # below this the pool cannot calibrate itself

# Normal quantiles at LEVELS, only for the no-pool fallback below.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hourly_viewers(points):
    """Map whole-hour start datetimes to viewers for a point list."""
    out = {}
    for p in points:
        v = p.get("viewers")
        if v is None:
            continue
        try:
            out[parse_ts(p["from"])] = float(v)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def pre_features(hour_map, day_start, y1):
    """(log pre-day peak / y1, log last pre-day hour / y1).

    The peak is over the 24 hourly slots before day_start; the last
    pre-day hour is the latest slot before day_start (searched back up
    to 48 h). Missing data falls back to y1 itself, i.e. a log ratio
    of 0, which is the pool's neutral point.
    """
    peak = None
    for k in range(24, 0, -1):
        v = hour_map.get(day_start - timedelta(hours=k))
        if v is not None and v > 0 and (peak is None or v > peak):
            peak = v
    last = None
    for k in range(1, 49):
        v = hour_map.get(day_start - timedelta(hours=k))
        if v is not None and v > 0:
            last = v
            break
    if peak is None:
        peak = y1
    if last is None:
        last = y1
    return math.log(peak / y1), math.log(last / y1)


def launch_shape(comp):
    """One comparable's day shape, or None if its day is unusable.

    Returns (ratios, features): ratios[h - 2] = ln(y_h / y_1) for
    h = 2..24 and features is the kernel feature dict.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hour_map = hourly_viewers(comp.get("hourly") or [])
    day = []
    for h in range(24):
        v = hour_map.get(day_start + timedelta(hours=h))
        if v is None or v <= 0:
            return None
        day.append(v)
    y1 = day[0]
    premax, prelast = pre_features(hour_map, day_start, y1)
    feats = {"logy1": math.log(y1), "premax": premax, "prelast": prelast}
    ratios = [math.log(day[h] / y1) for h in range(1, 24)]
    return ratios, feats


def kernel_weight(target_feats, comp_feats):
    d2 = 0.0
    for name, bw in KERNEL:
        d = (target_feats[name] - comp_feats[name]) / bw
        d2 += d * d
    return math.exp(-0.5 * d2)


def weighted_quantile(pairs, tau):
    """Smallest value whose cumulative weight reaches tau of the total."""
    return weighted_quantile_sorted(sorted(pairs), tau)


def weighted_quantile_sorted(ordered, tau):
    """weighted_quantile for pairs already sorted by value."""
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, w in ordered:
        acc += w
        if acc >= tau * total:
            return value
    return ordered[-1][0]


def hour_pairs(pool, target_feats, hour):
    """(log ratio, weight) pairs for one hour, neighbours pooled in.

    pool holds (ratios, feats) entries; the weight is the kernel weight
    of the entry against the target, shared by the hour and its two
    neighbours.
    """
    pairs = []
    for ratios, cfeats in pool:
        w = kernel_weight(target_feats, cfeats)
        for hh in range(max(2, hour - 1), min(24, hour + 1) + 1):
            pairs.append((ratios[hh - 2], w))
    return pairs


def block_index(hour):
    for bi, (first, last) in enumerate(BLOCKS):
        if first <= hour <= last:
            return bi
    return len(BLOCKS) - 1


def fit_calibration(pool):
    """Per-day-part level adjustments and median shifts for the fan.

    Runs leave-one-out forecasts inside the pool: every launch is
    forecast from the others, and per day-part we record how often each
    grid level covered the outcome and how far the median missed. The
    returned adjustments map each nominal level to the level whose
    measured coverage matches it (shrunk halfway back to nominal), and
    each day-part to its median bias, clamped.
    """
    n_blocks = len(BLOCKS)
    cover = [[[] for _ in CAL_GRID] for _ in range(n_blocks)]
    resid = [[] for _ in range(n_blocks)]
    for j, (ratios_j, feats_j) in enumerate(pool):
        rest = pool[:j] + pool[j + 1:]
        for hi, hour in enumerate(range(2, 25)):
            bi = block_index(hour)
            pairs = sorted(hour_pairs(rest, feats_j, hour))
            r = ratios_j[hi]
            for gi, g in enumerate(CAL_GRID):
                cover[bi][gi].append(
                    1.0 if r <= weighted_quantile_sorted(pairs, g) else 0.0)
            resid[bi].append(r - weighted_quantile_sorted(pairs, 0.5))
    adj = []
    for bi in range(n_blocks):
        curve = []
        for gi, g in enumerate(CAL_GRID):
            vals = cover[bi][gi]
            curve.append((sum(vals) / len(vals) if vals else g, g))
        row = []
        for tau in LEVELS:
            gstar = None
            for (c0, g0), (c1, g1) in zip(curve, curve[1:]):
                if (c0 - tau) * (c1 - tau) <= 0 and c1 != c0:
                    gstar = g0 + (tau - c0) * (g1 - g0) / (c1 - c0)
                    break
            if gstar is None:
                gstar = min(curve, key=lambda cg: abs(cg[0] - tau))[1]
            a = tau + CAL_LEVEL_SHRINK * (gstar - tau)
            row.append(min(0.98, max(0.02, a)))
        row.sort()
        adj.append(row)
    shift = []
    for bi in range(n_blocks):
        rs = sorted(resid[bi])
        med = rs[len(rs) // 2] if rs else 0.0
        shift.append(max(-CAL_SHIFT_CLAMP, min(CAL_SHIFT_CLAMP, med)))
    return adj, shift


def target_features(case):
    """The case's own kernel features: hour-1 print and pre-day ramp."""
    day_start = parse_ts(case["day_start"])
    hour_map = hourly_viewers(case.get("history_hourly") or [])
    known = (case.get("known") or [{}])[0]
    y1 = known.get("viewers")
    if y1 is None or y1 <= 0:
        # Fall back to the history's own hour-1 slot before giving up.
        y1 = hour_map.get(day_start)
    if y1 is None or y1 <= 0:
        y1 = 1.0
    y1 = float(y1)
    premax, prelast = pre_features(hour_map, day_start, y1)
    feats = {"logy1": math.log(y1), "premax": premax, "prelast": prelast}
    return y1, feats


def baseline_row(y1, hour):
    """House-style persistence row, used only if the pool is empty."""
    spread = 0.15 * math.sqrt(hour - 1)
    return [y1 * math.exp(z * spread) for z in Z]


def forecast(case):
    """Return the 23 output rows for one case."""
    y1, tfeats = target_features(case)
    pool = []
    for comp in case.get("comparables") or []:
        shape = launch_shape(comp)
        if shape is not None:
            pool.append(shape)
    if len(pool) >= CAL_MIN_POOL:
        adj, shift = fit_calibration(pool)
    else:
        adj = [list(LEVELS) for _ in BLOCKS]
        shift = [0.0] * len(BLOCKS)
    rows = []
    for hour in range(2, 25):
        if pool:
            bi = block_index(hour)
            pairs = hour_pairs(pool, tfeats, hour)
            qs = [y1 * math.exp(weighted_quantile(pairs, adj[bi][li]) + shift[bi])
                  for li in range(len(LEVELS))]
        else:
            qs = baseline_row(y1, hour)
        # Weighted quantiles are non-decreasing in the level by
        # construction; enforce it anyway and clamp to (0, 1e9].
        row = []
        prev = 0.0
        for q in qs:
            q = min(max(q, prev), 1e9)
            if q <= 0.0:
                q = prev if prev > 0.0 else 1.0
            row.append(round(q, 4))
            prev = q
        rows.append({"hour": hour, "q": row})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        case_id = case.get("id") or name[:-3]
        payload = {"hours": forecast(case)}
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
