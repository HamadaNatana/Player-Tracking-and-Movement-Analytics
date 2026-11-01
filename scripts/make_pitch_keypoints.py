#!/usr/bin/env python3
"""Build a YOLO-pose pitch-landmark dataset from SoccerNetGS pitch-line annotations.

Each frame becomes one "pitch" object covering the whole image, with one keypoint per
landmark in pitch_template.LANDMARK_NAMES (a crossing of two annotated lines). A
landmark that is not visible gets visibility 0. Run check_pitch_labels.py first to
confirm the template orientation.
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from pitch_template import LANDMARK_NAMES, find_landmarks, pitch_lines_px, pitch_xy

LABEL_FILE = "Labels-GameState.json"
SPLITS = {"train": "train", "valid": "val"}


def keypoint_row(landmarks, width, height):
    """YOLO-pose label line: class, full-image box, then x y visibility per landmark."""
    values = ["0", "0.5", "0.5", "1.0", "1.0"]
    for name in LANDMARK_NAMES:
        if name in landmarks:
            x = min(1.0, max(0.0, landmarks[name][0] / width))
            y = min(1.0, max(0.0, landmarks[name][1] / height))
            values += [f"{x:.6f}", f"{y:.6f}", "2"]
        else:
            values += ["0", "0", "0"]
    return " ".join(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/SoccerNetGS/gamestate-2024"))
    parser.add_argument("--output", type=Path, default=Path("data/pitchpose"))
    parser.add_argument("--sample-every", type=int, default=5,
                        help="keep every Nth frame with pitch lines (neighboring frames are near-duplicates)")
    parser.add_argument("--min-keypoints", type=int, default=4,
                        help="skip frames with fewer visible landmarks (a homography needs at least 4)")
    parser.add_argument("--max-sequences", type=int, help="only use the first N sequences per split")
    parser.add_argument("--image-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()
    if min(args.sample_every, args.min_keypoints) <= 0 or (args.max_sequences is not None and args.max_sequences <= 0):
        parser.error("sample-every, min-keypoints, and max-sequences must be positive")
    for source_split in SPLITS:
        if not (args.data / source_split).is_dir():
            parser.error(f"missing split folder: {args.data / source_split}")
    stage = args.output.parent / ".pitchpose-building"
    if stage.exists():
        shutil.rmtree(stage)

    counts = defaultdict(Counter)
    seen_landmarks = defaultdict(Counter)  # split -> how many frames show each landmark
    try:
        for source_split, split in SPLITS.items():
            images_dir = stage / "images" / split
            labels_dir = stage / "labels" / split
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            sequence_dirs = sorted(p for p in (args.data / source_split).iterdir() if (p / LABEL_FILE).is_file())
            if args.max_sequences:
                sequence_dirs = sequence_dirs[:args.max_sequences]
            for sequence_dir in sequence_dirs:
                try:
                    data = json.loads((sequence_dir / LABEL_FILE).read_text())
                    pitch_by_image = {a["image_id"]: a for a in data["annotations"]
                                      if a.get("supercategory") == "pitch"}
                    usable = [image for image in data["images"]
                              if image.get("has_labeled_pitch") and image["image_id"] in pitch_by_image]
                    frame_dir = sequence_dir / data["info"].get("im_dir", "img1")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(f"WARNING: skipping {sequence_dir.name}: cannot read labels ({exc})")
                    counts[split]["skipped_sequences"] += 1
                    continue
                for image in usable[::args.sample_every]:
                    source = frame_dir / image["file_name"]
                    if not source.is_file():
                        counts[split]["missing_images"] += 1
                        continue
                    lines = pitch_lines_px(pitch_by_image[image["image_id"]], image["width"], image["height"])
                    landmarks = find_landmarks(lines)
                    if len(landmarks) < args.min_keypoints:
                        counts[split]["too_few_keypoints"] += 1
                        continue
                    stem = f"{sequence_dir.name}_{Path(image['file_name']).stem}"
                    destination = images_dir / f"{stem}{source.suffix.lower()}"
                    if args.image_mode == "copy":
                        shutil.copy2(source, destination)
                    else:
                        final_parent = args.output / "images" / split
                        destination.symlink_to(os.path.relpath(source.resolve(), final_parent.resolve()))
                    (labels_dir / f"{stem}.txt").write_text(
                        keypoint_row(landmarks, image["width"], image["height"]) + "\n")
                    counts[split]["images"] += 1
                    counts[split]["keypoints"] += len(landmarks)
                    seen_landmarks[split].update(landmarks.keys())
            if not counts[split]["images"]:
                raise ValueError(f"no usable frames in {source_split}")
        count = len(LANDMARK_NAMES)
        (stage / "dataset.yaml").write_text(
            "train: images/train\nval: images/val\n"
            f"kpt_shape: [{count}, 3]\n"
            f"flip_idx: {list(range(count))}\n"
            "names:\n  0: pitch\n")
        (stage / "landmarks.json").write_text(json.dumps(
            [{"index": i, "name": name, "x_m": pitch_xy(name)[0], "y_m": pitch_xy(name)[1]}
             for i, name in enumerate(LANDMARK_NAMES)], indent=2) + "\n")
        if args.output.exists():
            shutil.rmtree(args.output)
        stage.rename(args.output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    for split in SPLITS.values():
        images = counts[split]["images"]
        print(f"{split}: {dict(counts[split])}, mean visible landmarks per frame {counts[split]['keypoints'] / images:.1f}")
    print("frames showing each landmark (train):")
    for name in LANDMARK_NAMES:
        print(f"  {name:<36} {seen_landmarks['train'][name]:>6}")
    print(f"dataset: {args.output} ({args.image_mode}); flip augmentation must be off when training")


if __name__ == "__main__":
    main()