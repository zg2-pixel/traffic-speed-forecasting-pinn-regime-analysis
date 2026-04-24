"""
event_labeling.py

Detect sustained low-speed/high-occupancy congestion events in
per-station traffic time series, and split each event into
onset / congested / recovery phases. Also derives physics-loss
thresholds and per-phase weights from the training split.

Inputs:
    data/processed/traffic_clean.csv  (or a --data_path override)

Outputs:
    data/processed/
        all_splits_with_regime_labels.csv        (main: feeds v6 and physics models)
        event_labels/
            physics_event_calibration.json       (thresholds + weights for physics loss)
            detected_congestion_events.csv
            station_regime_summary.csv
            slope_distributions.png
            regime_delta_speed_boxplot.png
            speed_occ_scatter_by_regime.png
            three_day_event_overview_station_<id>_<split>.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Config
DEFAULT_DATA_PATHS = [
    "data/processed/traffic_clean.csv",
    "/mnt/data/traffic_clean.csv",
]

DEFAULT_MAIN_OUTPUT_DIR = "data/processed"
DEFAULT_AUX_OUTPUT_SUBDIR = "event_labels"

REQUIRED_COLS = ["timestamp", "station", "avg_speed", "total_flow", "avg_occupancy"]


# Helpers
def resolve_data_path(user_path):
    candidates = [user_path] if user_path else []
    candidates.extend(DEFAULT_DATA_PATHS)
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find traffic_clean.csv. Pass --data_path explicitly.")


def safe_quantile(series, q, default):
    series = pd.Series(series).dropna()
    if len(series) == 0:
        return float(default)
    return float(series.quantile(q))


def make_dir(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


# Data loading and splits
def load_data(data_path):
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    df = df.dropna(subset=REQUIRED_COLS).copy()
    return df


def add_time_split(df, train_days, val_days):
    dates = sorted(df["timestamp"].dt.date.unique())
    if len(dates) < train_days + val_days:
        raise ValueError(
            f"Only {len(dates)} unique dates, need at least "
            f"{train_days + val_days} for train+val.")
    train_end = pd.Timestamp(dates[train_days - 1])
    val_end = pd.Timestamp(dates[train_days + val_days - 1])

    out = df.copy()
    d = out["timestamp"].dt.date
    out["split"] = np.where(
        d <= train_end.date(), "train",
        np.where(d <= val_end.date(), "val", "test"),
    )
    return out, train_end, val_end


# Feature engineering
def compute_station_features(sdf, slope_lookback):
    sdf = sdf.sort_values("timestamp").reset_index(drop=True).copy()
    speed = sdf["avg_speed"]
    occ = sdf["avg_occupancy"]
    flow = sdf["total_flow"]

    sdf["speed_diff_1"] = speed.diff()
    sdf["occ_diff_1"] = occ.diff()
    sdf["flow_diff_1"] = flow.diff()

    sdf["speed_slope_lb"] = (speed - speed.shift(slope_lookback)) / float(slope_lookback)
    sdf["occ_slope_lb"] = (occ - occ.shift(slope_lookback)) / float(slope_lookback)
    sdf["flow_slope_lb"] = (flow - flow.shift(slope_lookback)) / float(slope_lookback)

    sdf["future_speed_diff_1"] = speed.shift(-1) - speed
    sdf["future_speed_diff_2"] = speed.shift(-2) - speed
    sdf["speed_roll3"] = speed.rolling(3, min_periods=1).mean()
    return sdf


def build_feature_table(df, slope_lookback):
    parts = []
    for sid, sdf in df.groupby("station", sort=True):
        part = compute_station_features(sdf, slope_lookback=slope_lookback)
        part["station"] = sid
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["station", "timestamp"]).reset_index(drop=True)


# Threshold derivation (from training split)
def derive_thresholds(train_df):
    t = {}
    t["low_speed_threshold"] = safe_quantile(train_df["avg_speed"], 0.20, 45.0)
    t["high_occ_threshold"] = safe_quantile(train_df["avg_occupancy"], 0.80, 0.15)
    t["exit_speed_threshold"] = safe_quantile(
        train_df["avg_speed"], 0.35, t["low_speed_threshold"] + 5.0)
    pos_diffs = train_df["future_speed_diff_1"].dropna()
    t["strong_rise_delta"] = safe_quantile(pos_diffs[pos_diffs > 0], 0.80, 3.0)
    t["strong_drop_delta"] = safe_quantile(
        (-pos_diffs[pos_diffs < 0]), 0.80, 3.0)
    return t


# Event detection helpers
def rolling_all_true(mask, window):
    if window <= 1:
        return mask.astype(bool)
    ser = pd.Series(mask.astype(int))
    return (ser.rolling(window=window, min_periods=window).sum().to_numpy() >= window)


def contiguous_segments(mask):
    segs = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def _find_core_platform_bounds(speed_segment, rel_margin=0.08):
    # Find the low-speed platform around the event minimum instead of
    # a single point. Platform = contiguous points whose speed stays
    # within tol of the local minimum.
    if len(speed_segment) == 0:
        return 0, 0

    min_idx = int(np.argmin(speed_segment))
    min_speed = float(speed_segment[min_idx])
    seg_max = float(np.max(speed_segment))
    seg_min = float(np.min(speed_segment))
    span = max(seg_max - seg_min, 1.0)
    tol = max(1.0, rel_margin * span)
    threshold = min_speed + tol

    left = min_idx
    while left - 1 >= 0 and speed_segment[left - 1] <= threshold:
        left -= 1
    right = min_idx
    while right + 1 < len(speed_segment) and speed_segment[right + 1] <= threshold:
        right += 1
    return left, right


def _force_three_phase_split(speed_segment):
    # Split an event into onset -> core -> recovery. Core is the
    # platform around the minimum. Onset/recovery are forced to exist
    # whenever event length permits (>=3 steps).
    m = len(speed_segment)
    if m <= 0:
        return 0, 0
    if m == 1:
        return 0, 0
    if m == 2:
        return 1, 1

    raw_start, raw_end = _find_core_platform_bounds(speed_segment)
    core_start = max(1, raw_start)
    core_end = min(m - 2, raw_end)

    if core_start > core_end:
        min_idx = int(np.argmin(speed_segment))
        core_start = min(max(1, min_idx), m - 2)
        core_end = core_start
    return core_start, core_end


# Event detection: per station
def detect_station_congestion_events(sdf, thresholds, core_window_steps,
                                     max_transition_extension, occ_relax_factor):
    sdf = sdf.sort_values("timestamp").reset_index(drop=True).copy()
    n = len(sdf)

    speed = sdf["avg_speed"].to_numpy(dtype=float)
    occ = sdf["avg_occupancy"].to_numpy(dtype=float)

    low_speed = speed <= thresholds["low_speed_threshold"]
    exit_zone = speed <= thresholds["exit_speed_threshold"]
    high_occ = occ >= thresholds["high_occ_threshold"]
    occ_relaxed = occ >= (thresholds["high_occ_threshold"] * occ_relax_factor)

    core_seed = low_speed & high_occ
    core_hit = rolling_all_true(core_seed, core_window_steps)
    core_hit = np.roll(core_hit, -(core_window_steps - 1))
    if core_window_steps > 1:
        core_hit[-(core_window_steps - 1):] = False

    sdf["regime"] = "other"
    sdf["physics_event"] = "other"
    sdf["is_congestion"] = 0
    sdf["congestion_event_id"] = -1
    sdf["event_phase"] = "other"

    events = []
    prev_end = -1
    event_id = 0

    for seed_start, seed_end in contiguous_segments(core_hit):
        core_start = seed_start
        core_end = min(n - 1, seed_end + core_window_steps - 1)

        # Expand core while low_speed + relaxed occupancy hold
        while core_start > 0 and low_speed[core_start - 1] and occ_relaxed[core_start - 1]:
            core_start -= 1
        while core_end + 1 < n and low_speed[core_end + 1] and occ_relaxed[core_end + 1]:
            core_end += 1

        # Extend left as long as we stay in low-speed or exit-zone band
        start = core_start
        while start > prev_end + 1 and (
                low_speed[start - 1]
                or (exit_zone[start - 1] and occ_relaxed[start - 1])):
            start -= 1

        # Extend further left if a strong drop into congestion is visible
        left_extra = 0
        while start > prev_end + 1 and left_extra < max_transition_extension:
            cur = start - 1
            drop = speed[cur] - speed[cur + 1]
            if drop >= thresholds["strong_drop_delta"] or (
                    exit_zone[cur] and occ_relaxed[cur]):
                start = cur
                left_extra += 1
            else:
                break

        # Extend right symmetrically
        end = core_end
        while end + 1 < n and (
                low_speed[end + 1]
                or (exit_zone[end + 1] and occ_relaxed[end + 1])):
            end += 1

        right_extra = 0
        while end + 1 < n and right_extra < max_transition_extension:
            nxt = end + 1
            rise = speed[nxt] - speed[nxt - 1]
            if rise >= thresholds["strong_rise_delta"] or (
                    exit_zone[nxt] and occ_relaxed[nxt]):
                end = nxt
                right_extra += 1
                if not low_speed[nxt] and rise < thresholds["strong_rise_delta"]:
                    break
            else:
                break

        if start <= prev_end:
            start = prev_end + 1
        if end < core_end or start > core_start:
            continue

        # Guarantee enough length for onset + core + recovery
        while (end - start + 1) < 3:
            extended = False
            if start > prev_end + 1:
                start -= 1
                extended = True
            if (end - start + 1) >= 3:
                break
            if end + 1 < n:
                end += 1
                extended = True
            if not extended:
                break

        # Phase labels must cover [start, end] contiguously:
        # onset -> congested -> recovery.
        idx = np.arange(start, end + 1)
        event_speed = speed[start:end + 1]
        phase = np.full(len(idx), "congested", dtype=object)

        core_local_s, core_local_e = _force_three_phase_split(event_speed)
        phase[:core_local_s] = "onset"
        phase[core_local_e + 1:] = "recovery"

        # Safety: if any hole sneaks in, rebuild from scratch
        if np.any(pd.isna(phase)) or np.any(phase == "other"):
            phase = np.full(len(idx), "congested", dtype=object)
            phase[:core_local_s] = "onset"
            phase[core_local_e + 1:] = "recovery"

        core_start = start + core_local_s
        core_end = start + core_local_e

        sdf.loc[idx, "regime"] = phase
        sdf.loc[idx, "event_phase"] = phase
        sdf.loc[idx, "physics_event"] = "congestion"
        sdf.loc[idx, "is_congestion"] = 1
        sdf.loc[idx, "congestion_event_id"] = event_id

        events.append({
            "station": int(sdf.loc[start, "station"]),
            "event_id": int(event_id),
            "start_idx": int(start),
            "core_start_idx": int(core_start),
            "core_end_idx": int(core_end),
            "onset_steps": int(core_local_s),
            "core_steps": int(core_local_e - core_local_s + 1),
            "recovery_steps": int(len(idx) - core_local_e - 1),
            "end_idx": int(end),
            "start_time": str(pd.Timestamp(sdf.loc[start, "timestamp"])),
            "core_start_time": str(pd.Timestamp(sdf.loc[core_start, "timestamp"])),
            "core_end_time": str(pd.Timestamp(sdf.loc[core_end, "timestamp"])),
            "end_time": str(pd.Timestamp(sdf.loc[end, "timestamp"])),
            "duration_steps": int(end - start + 1),
            "core_duration_steps": int(core_end - core_start + 1),
            "min_speed": float(np.min(speed[start:end + 1])),
            "mean_speed": float(np.mean(speed[start:end + 1])),
            "max_occ": float(np.max(occ[start:end + 1])),
        })
        prev_end = end
        event_id += 1

    return sdf, events


def detect_congestion_events(df, thresholds, core_window_steps,
                             max_transition_extension, occ_relax_factor):
    parts = []
    all_events = []
    next_global_id = 0

    for sid, sdf in df.groupby("station", sort=True):
        labeled, events = detect_station_congestion_events(
            sdf=sdf,
            thresholds=thresholds,
            core_window_steps=core_window_steps,
            max_transition_extension=max_transition_extension,
            occ_relax_factor=occ_relax_factor,
        )
        local_ids = sorted([eid for eid in labeled["congestion_event_id"].unique() if eid >= 0])
        id_map = {old: next_global_id + i for i, old in enumerate(local_ids)}
        if id_map:
            labeled.loc[labeled["congestion_event_id"] >= 0, "congestion_event_id"] = labeled.loc[
                labeled["congestion_event_id"] >= 0, "congestion_event_id"
            ].map(id_map)
        for ev in events:
            ev = ev.copy()
            ev["event_id"] = int(id_map[ev["event_id"]])
            all_events.append(ev)
        next_global_id += len(local_ids)
        parts.append(labeled)

    out = pd.concat(parts, ignore_index=True).sort_values(
        ["station", "timestamp"]).reset_index(drop=True)
    event_df = pd.DataFrame(all_events)
    return out, event_df


# Calibration stats (for physics loss)
def compute_regime_smoothness_limits(df):
    out = {}
    for regime in ["other", "onset", "congested", "recovery"]:
        sub = df[df["regime"] == regime]
        delta = sub["future_speed_diff_1"].abs().dropna()
        out[f"{regime}_delta_v_q90"] = safe_quantile(delta, 0.90, 3.0)
        out[f"{regime}_delta_v_q95"] = safe_quantile(delta, 0.95, 5.0)
    return out


def compute_transition_weights(df):
    counts = df["regime"].value_counts()
    base = max(int(counts.get("other", 1)), 1)
    weights = {}
    for regime in ["other", "onset", "congested", "recovery"]:
        cnt = max(int(counts.get(regime, 1)), 1)
        raw = np.sqrt(base / cnt)
        weights[regime] = float(max(1.0, round(raw, 3)))
    return weights


def compute_station_summary(df):
    rows = []
    for sid, sdf in df.groupby("station", sort=True):
        row = {
            "station": sid,
            "n_rows": int(len(sdf)),
            "speed_mean": float(sdf["avg_speed"].mean()),
            "speed_q20": safe_quantile(sdf["avg_speed"], 0.20, np.nan),
            "speed_q80": safe_quantile(sdf["avg_speed"], 0.80, np.nan),
            "occ_mean": float(sdf["avg_occupancy"].mean()),
            "occ_q80": safe_quantile(sdf["avg_occupancy"], 0.80, np.nan),
            "flow_mean": float(sdf["total_flow"].mean()),
        }
        shares = sdf["regime"].value_counts(normalize=True)
        for regime in ["other", "onset", "congested", "recovery"]:
            row[f"share_{regime}"] = float(shares.get(regime, 0.0))
        row["share_congestion_event"] = float((sdf["is_congestion"] == 1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# Diagnostic plots
def plot_slope_distributions(train_df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(train_df["speed_slope_lb"].dropna(), bins=50)
    axes[0].set_title("Speed slope")
    axes[1].hist(train_df["occ_slope_lb"].dropna(), bins=50)
    axes[1].set_title("Occupancy slope")
    axes[2].hist(train_df["future_speed_diff_1"].dropna(), bins=50)
    axes[2].set_title("Next-step speed change")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "slope_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_regime_delta_boxplot(train_df, out_dir):
    order = ["other", "onset", "congested", "recovery"]
    data, labels = [], []
    for regime in order:
        vals = train_df.loc[train_df["regime"] == regime, "future_speed_diff_1"].abs().dropna()
        if len(vals) > 0:
            data.append(vals.values)
            labels.append(regime)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("|Δ speed| by event phase")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "regime_delta_speed_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_speed_occ_by_regime(train_df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    max_pts = 3000
    for regime in ["other", "onset", "congested", "recovery"]:
        sub = train_df[train_df["regime"] == regime]
        if len(sub) == 0:
            continue
        if len(sub) > max_pts:
            sub = sub.sample(max_pts, random_state=42)
        ax.scatter(sub["avg_occupancy"], sub["avg_speed"], s=8, alpha=0.35, label=regime)
    ax.set_xlabel("Occupancy")
    ax.set_ylabel("Speed (mph)")
    ax.set_title("Speed–Occupancy scatter by event phase")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speed_occ_scatter_by_regime.png", dpi=150, bbox_inches="tight")
    plt.close()


def merge_consecutive_segments(timestamps, labels, base_freq):
    segments = []
    if len(timestamps) == 0:
        return segments
    start_ts = pd.Timestamp(timestamps.iloc[0])
    prev_ts = pd.Timestamp(timestamps.iloc[0])
    prev_label = str(labels.iloc[0])
    for idx in range(1, len(timestamps)):
        cur_ts = pd.Timestamp(timestamps.iloc[idx])
        cur_label = str(labels.iloc[idx])
        gap = cur_ts - prev_ts
        if cur_label != prev_label or gap > base_freq:
            segments.append((start_ts, prev_ts + base_freq, prev_label))
            start_ts = cur_ts
            prev_label = cur_label
        prev_ts = cur_ts
    segments.append((start_ts, prev_ts + base_freq, prev_label))
    return segments


def build_plot_phase_segments(window, base_freq, max_bridge_gap):
    event_rows = window[window["congestion_event_id"] >= 0].copy().sort_values(
        "timestamp").reset_index(drop=True)
    if len(event_rows) == 0:
        return []

    half_step = base_freq / 2
    raw = []
    ts_list = event_rows["timestamp"].tolist()
    ev_ids = event_rows["congestion_event_id"].tolist()
    phases = event_rows["event_phase"].tolist()

    for i in range(len(event_rows)):
        ts = pd.Timestamp(ts_list[i])
        eid = int(ev_ids[i])
        ph = str(phases[i])
        if ph not in {"onset", "congested", "recovery"}:
            continue

        left = ts - half_step
        right = ts + half_step

        if i > 0:
            prev_ts = pd.Timestamp(ts_list[i - 1])
            prev_id = int(ev_ids[i - 1])
            gap = ts - prev_ts
            if prev_id == eid and gap <= max_bridge_gap:
                left = prev_ts + gap / 2

        if i + 1 < len(event_rows):
            next_ts = pd.Timestamp(ts_list[i + 1])
            next_id = int(ev_ids[i + 1])
            gap = next_ts - ts
            if next_id == eid and gap <= max_bridge_gap:
                right = ts + gap / 2

        raw.append((left, right, ph))

    if not raw:
        return []

    merged = []
    cur_s, cur_e, cur_p = raw[0]
    for s, e, p in raw[1:]:
        if p == cur_p and s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e, cur_p))
            cur_s, cur_e, cur_p = s, e, p
    merged.append((cur_s, cur_e, cur_p))
    return merged


def plot_three_day_event_overview(labeled_df, out_dir, station_id=None,
                                  start_date=None, split="test",
                                  plot_gap_bridge_minutes=45):
    src = labeled_df[labeled_df["split"] == split].copy()
    if len(src) == 0:
        raise ValueError(f"No rows for split='{split}'.")
    if station_id is None:
        station_id = int(src["station"].value_counts().index[0])
    src = src[src["station"] == station_id].sort_values("timestamp").reset_index(drop=True)
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
    else:
        start_ts = pd.Timestamp(src["timestamp"].dt.floor("D").iloc[0])
    end_ts = start_ts + pd.Timedelta(days=3)
    window = src[(src["timestamp"] >= start_ts) & (src["timestamp"] < end_ts)].copy()
    if len(window) == 0:
        raise ValueError("Empty plotting window.")

    dt_diffs = window["timestamp"].sort_values().diff().dropna()
    base_freq = dt_diffs.mode().iloc[0] if len(dt_diffs) > 0 else pd.Timedelta(minutes=5)
    phase_colors = {"onset": "#FFD54F", "congested": "#EF5350", "recovery": "#81C784"}

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    segments = build_plot_phase_segments(
        window=window,
        base_freq=base_freq,
        max_bridge_gap=pd.Timedelta(minutes=plot_gap_bridge_minutes),
    )
    for ax in axes:
        seen = set()
        for s, e, p in segments:
            if p in phase_colors:
                label = p if p not in seen else None
                ax.axvspan(s, e, color=phase_colors[p], alpha=0.22, label=label)
                seen.add(p)
        ax.grid(True, alpha=0.3)
    axes[0].plot(window["timestamp"], window["avg_speed"], linewidth=1.2, label="avg_speed")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Speed (mph)")
    axes[0].set_title(f"3-day event overview | split={split} | station={station_id}")
    axes[1].plot(window["timestamp"], window["avg_occupancy"], linewidth=1.0, label="avg_occupancy")
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Occupancy")
    axes[1].set_xlabel("Timestamp")
    plt.tight_layout()
    out_path = out_dir / f"three_day_event_overview_station_{station_id}_{split}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# IO
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Detect congestion events and split each into onset/congested/recovery phases.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to traffic_clean.csv (defaults to data/processed/traffic_clean.csv)")
    parser.add_argument("--main_out_dir", type=str, default=DEFAULT_MAIN_OUTPUT_DIR,
                        help="Where all_splits_with_regime_labels.csv is written.")
    parser.add_argument("--aux_subdir", type=str, default=DEFAULT_AUX_OUTPUT_SUBDIR,
                        help="Subdir (under main_out_dir) for auxiliary artifacts.")
    parser.add_argument("--train_days", type=int, default=22)
    parser.add_argument("--val_days", type=int, default=3)
    parser.add_argument("--slope_lookback", type=int, default=3)
    parser.add_argument("--core_window_steps", type=int, default=4)
    parser.add_argument("--max_transition_extension", type=int, default=6)
    parser.add_argument("--occ_relax_factor", type=float, default=0.85)
    parser.add_argument("--plot_station", type=int, default=None)
    parser.add_argument("--plot_split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--plot_start_date", type=str, default=None)
    parser.add_argument("--plot_gap_bridge_minutes", type=int, default=45)
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_path)
    main_out = make_dir(args.main_out_dir)
    aux_out = make_dir(main_out / args.aux_subdir)

    print(f"Data:    {data_path}")
    print(f"Main:    {main_out}/all_splits_with_regime_labels.csv")
    print(f"Aux:     {aux_out}/")

    df = load_data(data_path)
    df, train_end, val_end = add_time_split(df, args.train_days, args.val_days)
    feat_df = build_feature_table(df, slope_lookback=args.slope_lookback)

    train_df = feat_df[feat_df["split"] == "train"].copy().reset_index(drop=True)
    thresholds = derive_thresholds(train_df)

    labeled_df, event_df = detect_congestion_events(
        df=feat_df,
        thresholds=thresholds,
        core_window_steps=args.core_window_steps,
        max_transition_extension=args.max_transition_extension,
        occ_relax_factor=args.occ_relax_factor,
    )

    train_labeled = labeled_df[labeled_df["split"] == "train"].copy()
    smoothness = compute_regime_smoothness_limits(train_labeled)
    weights = compute_transition_weights(train_labeled)

    # Main product
    labeled_df.to_csv(main_out / "all_splits_with_regime_labels.csv", index=False)

    # Aux products
    compute_station_summary(train_labeled).to_csv(
        aux_out / "station_regime_summary.csv", index=False)
    event_df.to_csv(aux_out / "detected_congestion_events.csv", index=False)
    calibration = {
        "metadata": {
            "data_path": data_path,
            "train_end_date": str(train_end.date()),
            "val_end_date": str(val_end.date()),
            "event_definition": (
                "Find a sustained low-speed/high-occupancy core, expand "
                "left/right through contiguous low-speed points, and label "
                "the transitions before/after the core as onset/recovery "
                "until speed returns to non-low-speed."
            ),
        },
        "thresholds_for_physics": thresholds,
        "event_detection_params": {
            "core_window_steps": int(args.core_window_steps),
            "max_transition_extension": int(args.max_transition_extension),
            "occ_relax_factor": float(args.occ_relax_factor),
        },
        "regime_smoothness": smoothness,
        "weights": weights,
        "notes": {"recommended_event_labels_for_physics": ["congestion", "other"]},
    }
    save_json(calibration, aux_out / "physics_event_calibration.json")

    plot_slope_distributions(train_labeled, aux_out)
    plot_regime_delta_boxplot(train_labeled, aux_out)
    plot_speed_occ_by_regime(train_labeled, aux_out)
    plot_three_day_event_overview(
        labeled_df=labeled_df,
        out_dir=aux_out,
        station_id=args.plot_station,
        start_date=args.plot_start_date,
        split=args.plot_split,
        plot_gap_bridge_minutes=args.plot_gap_bridge_minutes,
    )

    # Summary
    n_events = len(event_df)
    total_rows = len(labeled_df)
    phase_share = labeled_df["regime"].value_counts(normalize=True).to_dict()
    print(f"\nDetected {n_events} congestion events across {labeled_df['station'].nunique()} stations.")
    print(f"Phase share (all splits): "
          + ", ".join(f"{k}={v*100:.1f}%" for k, v in phase_share.items()))
    print(f"Done. Rows labeled: {total_rows:,}")


if __name__ == "__main__":
    main()
