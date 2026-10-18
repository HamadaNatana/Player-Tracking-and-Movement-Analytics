#!/usr/bin/env python3
"""Check that the converted four-class YOLO dataset is usable."""

import argparse
import math
from pathlib import Path

import yaml

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/yolo/dataset.yaml"))
    args = parser.parse_args()
    dataset = args.data / "dataset.yaml" if args.data.is_dir() else args.data
    if not dataset.is_file():
        parser.error(f"dataset YAML missing: {dataset}")
    try:
        config = yaml.safe_load(dataset.read_text())
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"cannot read dataset YAML: {exc}")
    if not isinstance(config, dict) or config.get("names") != NAMES:
        parser.error(f"dataset YAML must define classes {NAMES}")
    root = dataset.resolve().parent
    errors = []
    totals = [0, 0, 0]
    for split in ("train", "val", "test"):
        image_path = config.get(split)
        if image_path != f"images/{split}":
            errors.append(f"{split}: expected images/{split} in dataset YAML")
            continue
        image_dir = root / image_path
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: missing image or label directory")
            continue
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS]
        labels = list(label_dir.glob("*.txt"))
        objects = 0
        if not images or not labels:
            errors.append(f"{split}: no images or labels")
        if len(images) != len(labels):
            errors.append(f"{split}: image and label counts differ")
        image_stems = {image.stem for image in images}
        for label in labels:
            if label.stem not in image_stems:
                errors.append(f"label has no image: {label}")
        for image in images:
            if not image.is_file():
                errors.append(f"missing image: {image}")
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                errors.append(f"missing label: {label}")
                continue
            for line_number, row in enumerate(label.read_text().splitlines(), 1):
                fields = row.split()
                try:
                    if len(fields) != 5 or fields[0] not in ("0", "1", "2", "3"):
                        raise ValueError("expected class ID 0-3 and four coordinates")
                    x, y, w, h = map(float, fields[1:])
                    if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h))
                            or w <= 0 or h <= 0):
                        raise ValueError("coordinates must be normalized; width and height positive")
                except ValueError as exc:
                    errors.append(f"{label}:{line_number}: {exc}")
                    continue
                objects += 1
        print(f"{split}: images={len(images)}, labels={len(labels)}, objects={objects}")
        if not objects:
            errors.append(f"{split}: no valid objects")
        totals[0] += len(images)
        totals[1] += len(labels)
        totals[2] += objects
    for error in errors[:20]:
        print(f"ERROR: {error}")
    if len(errors) > 20:
        print(f"ERROR: {len(errors) - 20} more errors")
    if errors or totals[2] == 0:
        raise SystemExit(f"validation failed ({len(errors)} errors; {totals[2]} objects)")
    print(f"validation passed: images={totals[0]}, labels={totals[1]}, objects={totals[2]}")


if __name__ == "__main__":
    main()
