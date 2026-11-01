#!/usr/bin/env python3
"""Speed and distance from pitch-plane player positions (PRD step 5, on ground truth).

Reads SoccerNetGS Labels-GameState.json files and builds one trajectory per track ID
from the foot-point positions in meters. Each trajectory is cleaned (spikes removed,
short gaps filled, long gaps split) and smoothed with a Savitzky-Golay filter before
speed and distance are computed from consecutive positions.

The noisy positions (bbox_pitch_raw) are compared with the dataset's smoothed
positions (bbox_pitch), which serve as the reference. bbox_pitch is assumed to be the
temporally smoothed version (dataset v1.3 changelog); it is a reference, not truth.
Distance is not counted across gaps longer than --max-gap frames, so a player who
leaves the frame has a lower total than the real distance run.
"""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LABEL_FILE = "Labels-GameState.json"
FIELDS = {"raw": "bbox_pitch_raw", "reference": "bbox_pitch"}
KMH = 3.6


@dataclass(frozen=True)
class Settings:
    window: int = 13  # smoothing window in frames (about 0.5 s at 25 fps); must be odd
    polyorder: int = 2  # polynomial order of the smoothing filter
    max_gap: int = 10  # fill gaps up to this many missing frames; split longer ones
    spike_window: int = 5  # frames in the running median used to find spikes
    spike_threshold: float = 1.0  # meters from the running median that counts as a spike


def load_tracks(data, field, categories):
    """Return {track_id: (frames, xy)} using the foot point (bottom middle) of each box.

    frames is a sorted array of frame numbers; xy is an N x 2 array of meters.
    Annotations without a pitch position or a track ID are skipped.
    """
    frame_of = {}  # image_id -> frame number
    for order, image in enumerate(data["images"]):
        stem = Path(image["file_name"]).stem
        frame_of[image["image_id"]] = int(stem) if stem.isdigit() else order + 1
    rows = defaultdict(list)  # track_id -> [(frame, x, y), ...]
    for annotation in data["annotations"]:
        if annotation.get("supercategory") != "object" or annotation.get("category_id") not in categories:
            continue
        track = annotation.get("track_id")
        position = annotation.get(field)
        frame = frame_of.get(annotation.get("image_id"))
        if track is None or frame is None or not position:
            continue
        x, y = position.get("x_bottom_middle"), position.get("y_bottom_middle")
        if x is None or y is None:
            continue
        rows[track].append((frame, float(x), float(y)))
    tracks = {}
    for track, points in rows.items():
        array = np.array(sorted(points))
        _, first = np.unique(array[:, 0], return_index=True)  # one position per frame
        array = array[first]
        tracks[track] = (array[:, 0].astype(int), array[:, 1:])
    return tracks


def drop_spikes(frames, xy, settings):
    """Remove points far from the running median of their neighbors."""
    if len(frames) < settings.spike_window:
        return frames, xy
    median = np.column_stack([median_filter(xy[:, c], size=settings.spike_window, mode="nearest")
                              for c in (0, 1)])
    keep = np.linalg.norm(xy - median, axis=1) <= settings.spike_threshold
    return frames[keep], xy[keep]


def clean_track(frames, xy, settings):
    """Spike removal, gap handling, and smoothing.

    Returns a list of segments (frames, smoothed xy, observed flags). A segment is a
    run of consecutive frames; gaps of up to max_gap frames are filled by linear
    interpolation, longer gaps start a new segment, and segments shorter than the
    smoothing window are dropped.
    """
    frames, xy = drop_spikes(frames, xy, settings)
    if len(frames) == 0:
        return []
    segments = []
    breaks = np.flatnonzero(np.diff(frames) - 1 > settings.max_gap) + 1
    for seg_frames, seg_xy in zip(np.split(frames, breaks), np.split(xy, breaks)):
        full = np.arange(seg_frames[0], seg_frames[-1] + 1)
        if len(full) < settings.window:
            continue
        filled = np.column_stack([np.interp(full, seg_frames, seg_xy[:, c]) for c in (0, 1)])
        smooth = savgol_filter(filled, settings.window, settings.polyorder, axis=0)
        segments.append((full, smooth, np.isin(full, seg_frames)))
    return segments


def track_motion(segments, fps):
    """Table with columns frame, x, y, observed, speed (m/s), step (m); None if empty.

    The first row of each segment has no step, so its speed and step are NaN.
    """
    parts = []
    for frames, xy, observed in segments:
        step = np.concatenate([[np.nan], np.linalg.norm(np.diff(xy, axis=0), axis=1)])
        parts.append(np.column_stack([frames, xy, observed, step * fps, step]))
    return np.vstack(parts) if parts else None


def unsmoothed_steps(frames, xy):
    """Meters moved between consecutive observed frames, with no cleaning at all."""
    consecutive = np.diff(frames) == 1
    return np.linalg.norm(np.diff(xy, axis=0), axis=1)[consecutive]


def compare(raw_tracks, reference_tracks, settings, fps):
    """Motion tables for both sets of tracks, plus arrays pooled for the report."""
    tables = {"raw": {}, "reference": {}}
    pooled = defaultdict(list)
    for name, tracks in (("raw", raw_tracks), ("reference", reference_tracks)):
        for track, (frames, xy) in tracks.items():
            table = track_motion(clean_track(frames, xy, settings), fps)
            if table is not None:
                tables[name][track] = table
                pooled[f"{name}_smoothed_speed"].append(table[:, 4][~np.isnan(table[:, 4])])
            steps = unsmoothed_steps(frames, xy)
            if len(steps):
                pooled[f"{name}_unsmoothed_speed"].append(steps * fps)
                pooled[f"{name}_unsmoothed_distance"].append([steps.sum()])
    for track in tables["raw"].keys() & tables["reference"].keys():
        a, b = tables["raw"][track], tables["reference"][track]
        pooled["distance_raw"].append([np.nansum(a[:, 5])])
        pooled["distance_reference"].append([np.nansum(b[:, 5])])
        _, ia, ib = np.intersect1d(a[:, 0], b[:, 0], return_indices=True)
        if len(ia):
            pooled["position_sq_error"].append(np.sum((a[ia, 1:3] - b[ib, 1:3]) ** 2, axis=1))
            speed_a, speed_b = a[ia, 4], b[ib, 4]
            ok = ~np.isnan(speed_a) & ~np.isnan(speed_b)
            pooled["speed_raw"].append(speed_a[ok])
            pooled["speed_reference"].append(speed_b[ok])
    return tables, {key: np.concatenate([np.asarray(v, dtype=float) for v in values])
                    for key, values in pooled.items()}


def merge(parts):
    """Concatenate per-sequence pooled arrays key by key."""
    merged = defaultdict(list)
    for part in parts:
        for key, values in part.items():
            merged[key].append(values)
    return {key: np.concatenate(values) for key, values in merged.items()}


def agreement(estimate, reference):
    """MAE, MSE, and R2 of an estimate against a reference (NaN if fewer than 2 points)."""
    if len(estimate) < 2:
        return float("nan"), float("nan"), float("nan")
    return (mean_absolute_error(reference, estimate), mean_squared_error(reference, estimate),
            r2_score(reference, estimate))


def speed_row(label, speeds, distances, limit):
    if not len(speeds):
        return f"{label:<28} no data"
    return (f"{label:<28} {np.mean(distances):>9.1f} {100 * np.mean(speeds > limit):>12.2f}% "
            f"{np.percentile(speeds, 95):>9.2f} {speeds.max():>9.2f}")


def print_report(pooled, settings, limit):
    print(f"\nmotion summary (smoothing window {settings.window} frames, order {settings.polyorder})")
    print(f"{'variant':<28} {'mean dist/track (m)':>9} {'steps > ' + format(limit, 'g') + ' m/s':>13} "
          f"{'p95 m/s':>9} {'max m/s':>9}")
    empty = np.array([])
    for label, name, kind in (("raw, unsmoothed", "raw", "unsmoothed"),
                              ("reference, unsmoothed", "reference", "unsmoothed"),
                              ("raw, smoothed", "raw", "smoothed"),
                              ("reference, smoothed", "reference", "smoothed")):
        speeds = pooled.get(f"{name}_{kind}_speed", empty)
        if kind == "unsmoothed":
            distances = pooled.get(f"{name}_unsmoothed_distance", empty)
        else:
            distances = pooled.get("distance_raw" if name == "raw" else "distance_reference", empty)
        print(speed_row(label, speeds, distances if len(distances) else [float("nan")], limit))
    if "position_sq_error" in pooled:
        print(f"\nsmoothed raw vs smoothed reference ({len(pooled['distance_raw'])} tracks)")
        print(f"  position RMSE: {np.sqrt(pooled['position_sq_error'].mean()):.3f} m")
        for label, est, ref in (("distance per track (m)", pooled["distance_raw"], pooled["distance_reference"]),
                                ("speed per frame (m/s)", pooled["speed_raw"], pooled["speed_reference"])):
            mae, mse, r2 = agreement(est, ref)
            print(f"  {label}: MAE {mae:.3f}  MSE {mse:.3f}  R2 {r2:.3f}")


def write_table(path_writer, sequence, tables, fps):
    for track, table in sorted(tables.items()):
        distance = np.nancumsum(table[:, 5])
        for row, cumulative in zip(table, distance):
            speed = "" if np.isnan(row[4]) else f"{row[4]:.3f}"
            kmh = "" if np.isnan(row[4]) else f"{row[4] * KMH:.3f}"
            path_writer.writerow([sequence, track, int(row[0]), f"{row[1]:.3f}", f"{row[2]:.3f}",
                                  int(row[3]), speed, kmh, f"{cumulative:.3f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/SoccerNetGS/gamestate-2024"),
                        help="folder holding the train, valid, and test split folders")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--max-sequences", type=int, help="only use the first N sequences")
    parser.add_argument("--categories", default="1,2",
                        help="category IDs to track: 1 player, 2 goalkeeper, 3 referee")
    parser.add_argument("--window", type=int, default=Settings.window)
    parser.add_argument("--polyorder", type=int, default=Settings.polyorder)
    parser.add_argument("--max-gap", type=int, default=Settings.max_gap)
    parser.add_argument("--spike-window", type=int, default=Settings.spike_window)
    parser.add_argument("--spike-threshold", type=float, default=Settings.spike_threshold)
    parser.add_argument("--speed-limit", type=float, default=12.0,
                        help="m/s above which a step is counted as physically implausible")
    parser.add_argument("--sweep", help="comma-separated odd window sizes to compare, e.g. 5,9,13,19,25")
    parser.add_argument("--table", type=Path, help="write the smoothed trajectories to this CSV")
    parser.add_argument("--table-source", choices=tuple(FIELDS), default="reference",
                        help="which positions the CSV holds (raw = bbox_pitch_raw)")
    args = parser.parse_args()

    split_dir = args.data / args.split
    if not split_dir.is_dir():
        parser.error(f"missing split folder: {split_dir}")
    try:
        categories = {int(c) for c in args.categories.split(",")}
    except ValueError:
        parser.error("categories must be comma-separated integers")
    settings = Settings(args.window, args.polyorder, args.max_gap, args.spike_window, args.spike_threshold)

    def check(settings):
        if settings.window % 2 == 0 or settings.window <= settings.polyorder:
            parser.error("window must be odd and larger than polyorder")
        if settings.max_gap < 0 or settings.spike_window % 2 == 0 or settings.spike_threshold <= 0:
            parser.error("max-gap must be >= 0, spike-window odd, spike-threshold positive")

    check(settings)
    sweep_windows = []
    if args.sweep:
        try:
            sweep_windows = [int(w) for w in args.sweep.split(",")]
        except ValueError:
            parser.error("sweep must be comma-separated integers")
        for window in sweep_windows:
            check(Settings(window, settings.polyorder, settings.max_gap,
                           settings.spike_window, settings.spike_threshold))

    sequence_dirs = sorted(p for p in split_dir.iterdir() if (p / LABEL_FILE).is_file())
    if args.max_sequences:
        sequence_dirs = sequence_dirs[:args.max_sequences]
    if not sequence_dirs:
        parser.error(f"no sequence folders with {LABEL_FILE} under {split_dir}")

    loaded = []  # (sequence name, fps, raw tracks, reference tracks)
    for sequence_dir in sequence_dirs:
        try:
            data = json.loads((sequence_dir / LABEL_FILE).read_text())
            fps = float(data["info"].get("frame_rate", 25))
            loaded.append((sequence_dir.name, fps,
                           load_tracks(data, FIELDS["raw"], categories),
                           load_tracks(data, FIELDS["reference"], categories)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"WARNING: skipping {sequence_dir.name}: cannot read labels ({exc})")
    if not loaded:
        raise SystemExit("no readable sequences")

    writer, handle = None, None
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        handle = args.table.open("w", newline="")
        writer = csv.writer(handle)
        writer.writerow(["sequence", "track_id", "frame", "x_m", "y_m", "observed",
                         "speed_mps", "speed_kmh", "cumulative_distance_m"])
    parts = []
    for name, fps, raw_tracks, reference_tracks in loaded:
        tables, pooled = compare(raw_tracks, reference_tracks, settings, fps)
        parts.append(pooled)
        if writer:
            write_table(writer, name, tables[args.table_source], fps)
        if "distance_raw" in pooled:
            share = 100 * np.mean(pooled["raw_unsmoothed_speed"] > args.speed_limit)
            mae = np.mean(np.abs(pooled["distance_raw"] - pooled["distance_reference"]))
            print(f"{name}: tracks={len(pooled['distance_raw'])} raw steps over limit={share:.1f}% "
                  f"distance MAE (smoothed raw vs reference)={mae:.2f} m")
        else:
            print(f"{name}: no tracks long enough to analyze")
    if handle:
        handle.close()
        print(f"trajectories: {args.table} ({args.table_source} positions)")
    print_report(merge(parts), settings, args.speed_limit)

    if sweep_windows:
        print(f"\nwindow sweep (order {settings.polyorder}); smaller windows follow sharp moves, larger ones remove more jitter")
        print(f"{'window':>6} {'raw steps > limit':>18} {'distance MAE (m)':>17} {'speed MAE (m/s)':>16} {'position RMSE (m)':>18}")
        for window in sweep_windows:
            trial = Settings(window, settings.polyorder, settings.max_gap,
                             settings.spike_window, settings.spike_threshold)
            trial_pooled = merge([compare(raw, ref, trial, fps)[1] for _, fps, raw, ref in loaded])
            if "position_sq_error" not in trial_pooled:
                print(f"{window:>6} no data")
                continue
            over = 100 * np.mean(trial_pooled["raw_smoothed_speed"] > args.speed_limit)
            print(f"{window:>6} {over:>17.2f}% "
                  f"{agreement(trial_pooled['distance_raw'], trial_pooled['distance_reference'])[0]:>17.3f} "
                  f"{agreement(trial_pooled['speed_raw'], trial_pooled['speed_reference'])[0]:>16.3f} "
                  f"{np.sqrt(trial_pooled['position_sq_error'].mean()):>18.3f}")


if __name__ == "__main__":
    main()