#!/usr/bin/env python3
"""Check that annotated pitch-line crossings agree with the pitch template.

For sampled frames, fits an image-to-pitch homography from player foot points (image
box bottom-middle paired with the bbox_pitch position in meters), maps each annotated
line crossing through it, and compares the result with the template position. The
comparison is repeated with the template's axes flipped, so the best-matching
convention shows whether the template's orientation is right. Run this before
trusting any keypoint labels.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_template import find_landmarks, pitch_lines_px, pitch_xy

LABEL_FILE = "Labels-GameState.json"
CONVENTIONS = {  # name -> (x sign, y sign) applied to the template
    "as assumed": (1, 1),
    "x flipped": (-1, 1),
    "y flipped": (1, -1),
    "both flipped": (-1, -1),
}
RANSAC_METERS = 1.5  # a foot point farther than this from the fitted mapping is an outlier
MIN_INLIERS = 6
MIN_SPREAD_METERS = 3.0  # foot points must spread at least this much in x and y
WITHIN_METERS = 3.0


def frame_foot_points(annotations):
    """Foot points for one frame: image pixels and pitch meters, as two N x 2 arrays."""
    image_points, pitch_points = [], []
    for annotation in annotations:
        position = annotation.get("bbox_pitch")
        box = annotation.get("bbox_image")
        if not position or not box or position.get("x_bottom_middle") is None:
            continue
        image_points.append((box["x"] + box["w"] / 2, box["y"] + box["h"]))
        pitch_points.append((position["x_bottom_middle"], position["y_bottom_middle"]))
    return np.array(image_points, dtype=np.float32), np.array(pitch_points, dtype=np.float32)


def check_sequence(sequence_dir, sample_every, min_players, categories, errors, per_landmark):
    """Add landmark errors (meters) for one sequence to `errors` and `per_landmark`.

    Returns how many frames were used.
    """
    data = json.loads((sequence_dir / LABEL_FILE).read_text())
    objects_by_image = defaultdict(list)
    pitch_by_image = {}
    for annotation in data["annotations"]:
        if annotation.get("supercategory") == "object" and annotation.get("category_id") in categories:
            objects_by_image[annotation["image_id"]].append(annotation)
        elif annotation.get("supercategory") == "pitch":
            pitch_by_image[annotation["image_id"]] = annotation
    usable = [image for image in data["images"]
              if image.get("has_labeled_pitch") and image["image_id"] in pitch_by_image]
    used = 0
    for image in usable[::sample_every]:
        image_points, pitch_points = frame_foot_points(objects_by_image[image["image_id"]])
        if len(image_points) < min_players:
            continue
        if np.ptp(pitch_points[:, 0]) < MIN_SPREAD_METERS or np.ptp(pitch_points[:, 1]) < MIN_SPREAD_METERS:
            continue
        homography, mask = cv2.findHomography(image_points, pitch_points, cv2.RANSAC, RANSAC_METERS)
        if homography is None or int(mask.sum()) < MIN_INLIERS:
            continue
        lines = pitch_lines_px(pitch_by_image[image["image_id"]], image["width"], image["height"])
        landmarks = find_landmarks(lines)
        if len(landmarks) < 2:
            continue
        names = list(landmarks)
        pixels = np.array([landmarks[name] for name in names], dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pixels, homography).reshape(-1, 2)
        for convention, (sign_x, sign_y) in CONVENTIONS.items():
            for name, point in zip(names, mapped):
                x, y = pitch_xy(name)
                error = float(np.hypot(point[0] - sign_x * x, point[1] - sign_y * y))
                errors[convention].append(error)
                per_landmark[convention][name].append(error)
        used += 1
    return used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/SoccerNetGS/gamestate-2024"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--max-sequences", type=int, help="only use the first N sequences")
    parser.add_argument("--sample-every", type=int, default=25, help="use every Nth frame with pitch lines")
    parser.add_argument("--min-players", type=int, default=8,
                        help="frames need at least this many foot points to fit a mapping")
    parser.add_argument("--categories", default="1,2,3",
                        help="object categories used as foot points: 1 player, 2 goalkeeper, 3 referee")
    args = parser.parse_args()
    split_dir = args.data / args.split
    if not split_dir.is_dir():
        parser.error(f"missing split folder: {split_dir}")
    if min(args.sample_every, args.min_players) <= 0 or (args.max_sequences is not None and args.max_sequences <= 0):
        parser.error("sample-every, min-players, and max-sequences must be positive")
    try:
        categories = {int(c) for c in args.categories.split(",")}
    except ValueError:
        parser.error("categories must be comma-separated integers")
    sequence_dirs = sorted(p for p in split_dir.iterdir() if (p / LABEL_FILE).is_file())
    if args.max_sequences:
        sequence_dirs = sequence_dirs[:args.max_sequences]
    if not sequence_dirs:
        parser.error(f"no sequence folders with {LABEL_FILE} under {split_dir}")

    errors = defaultdict(list)
    per_landmark = {name: defaultdict(list) for name in CONVENTIONS}
    frames = 0
    for sequence_dir in sequence_dirs:
        try:
            frames += check_sequence(sequence_dir, args.sample_every, args.min_players, categories,
                                     errors, per_landmark)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"WARNING: skipping {sequence_dir.name}: cannot read labels ({exc})")
    if not errors:
        raise SystemExit("no frame had enough foot points and visible line crossings to compare")
    print(f"{frames} frames from {len(sequence_dirs)} sequences, {len(errors['as assumed'])} landmark comparisons")
    print(f"{'template convention':<22} {'median (m)':>10} {'90th pct (m)':>13} {'within ' + format(WITHIN_METERS, 'g') + ' m':>10}")
    medians = {}
    for convention in CONVENTIONS:
        values = np.array(errors[convention])
        medians[convention] = float(np.median(values))
        print(f"{convention:<22} {medians[convention]:>10.2f} {np.percentile(values, 90):>13.2f} "
              f"{100 * np.mean(values <= WITHIN_METERS):>9.1f}%")
    best = min(medians, key=medians.get)
    runner_up = sorted(medians.values())[1]
    print(f"\nbest match: {best} (median error {medians[best]:.2f} m; next best {runner_up:.2f} m)")
    if best != "as assumed":
        print("The template orientation is probably wrong: flip the matching axis in pitch_template.py.")
    elif runner_up < 2 * medians[best]:
        print("WARNING: the best convention is not clearly better than the others; do not trust the labels yet.")
    print(f"\nper-landmark median error under '{best}' (large values suggest a wrong template position):")
    rows = sorted(((float(np.median(v)), len(v), name) for name, v in per_landmark[best].items()), reverse=True)
    for median, count, name in rows:
        print(f"  {name:<36} {median:>6.2f} m  (n={count})")


if __name__ == "__main__":
    main()