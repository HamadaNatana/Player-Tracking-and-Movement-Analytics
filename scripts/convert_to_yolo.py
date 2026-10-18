#!/usr/bin/env python3
"""Convert local SoccerNetGS game-state annotations to YOLO detection data."""

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": "train", "valid": "val", "test": "test"}
NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
CLASS_IDS = {1: 0, 2: 1, 3: 2, 4: 3}


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def yolo_row(annotation, width, height):
    box = annotation["bbox_image"]
    x, y = number(box["x_center"]), number(box["y_center"])
    w, h = number(box["w"]), number(box["h"])
    if w <= 0 or h <= 0:
        raise ValueError("box width and height must be positive")
    left, right = max(0.0, x - w / 2), min(width, x + w / 2)
    top, bottom = max(0.0, y - h / 2), min(height, y + h / 2)
    if left >= right or top >= bottom:
        raise ValueError("box is outside image")
    cls = CLASS_IDS[annotation["category_id"]]
    return f"{cls} {(left + right) / (2 * width):.8f} {(top + bottom) / (2 * height):.8f} {(right - left) / width:.8f} {(bottom - top) / height:.8f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-mode", choices=("symlink", "copy"), default="symlink",
                        help="symlink raw images (default) or copy them for portability")
    args = parser.parse_args()
    raw = ROOT / "data/SoccerNetGS/gamestate-2024"
    output = ROOT / "data/yolo"
    stage = output.parent / ".yolo-building"
    for source_split in SPLITS:
        if not (raw / source_split).is_dir():
            parser.error(f"missing raw split: {raw / source_split}")
    if stage.exists():
        shutil.rmtree(stage)

    counts = defaultdict(Counter)
    warned = 0

    def warn(message):
        nonlocal warned
        warned += 1
        if warned <= 20:
            print(f"WARNING: {message}")

    try:
        for source_split, split in SPLITS.items():
            images_dir = stage / "images" / split
            labels_dir = stage / "labels" / split
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            seen = set()
            for sequence in sorted((raw / source_split).iterdir()):
                if not sequence.is_dir():
                    continue
                annotation_file = sequence / "Labels-GameState.json"
                try:
                    data = json.loads(annotation_file.read_text())
                    images, annotations = data["images"], data["annotations"]
                    if not isinstance(images, list) or not isinstance(annotations, list):
                        raise ValueError("images and annotations must be lists")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    warn(f"skipping missing or invalid annotations {annotation_file}: {exc}")
                    counts[split]["skipped_sequences"] += 1
                    continue
                by_image = defaultdict(list)
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        counts[split]["skipped_annotations"] += 1
                        warn(f"invalid annotation in {annotation_file}")
                        continue
                    if annotation.get("supercategory") != "object":
                        continue
                    category = annotation.get("category_id")
                    if category == 7:  # SoccerNetGS "other" objects are outside this model.
                        continue
                    if category not in CLASS_IDS:
                        counts[split]["skipped_annotations"] += 1
                        warn(f"unknown object category {category} in {annotation_file}")
                        continue
                    if not isinstance(annotation.get("image_id"), (str, int)):
                        counts[split]["skipped_annotations"] += 1
                        warn(f"annotation missing or invalid image_id in {annotation_file}")
                        continue
                    by_image[annotation["image_id"]].append(annotation)
                for image in images:
                    try:
                        image_id = image["image_id"]
                        filename = image["file_name"]
                        width, height = number(image["width"]), number(image["height"])
                        if (not isinstance(filename, str) or Path(filename).name != filename
                                or width <= 0 or height <= 0):
                            raise ValueError("invalid image metadata")
                        source = sequence / "img1" / filename
                        if not source.is_file():
                            raise FileNotFoundError(source)
                    except (KeyError, TypeError, ValueError, OSError) as exc:
                        counts[split]["skipped_images"] += 1
                        warn(f"skipping image in {annotation_file}: {exc}")
                        continue
                    stem = f"{sequence.name}_{image_id}_{Path(filename).stem}"
                    if stem.casefold() in seen:
                        raise ValueError(f"duplicate output filename in {split}: {stem}")
                    seen.add(stem.casefold())
                    rows = []
                    for annotation in by_image.get(image_id, []):
                        try:
                            rows.append(yolo_row(annotation, width, height))
                        except (KeyError, TypeError, ValueError) as exc:
                            counts[split]["skipped_annotations"] += 1
                            warn(f"skipping annotation {annotation.get('id')} in {annotation_file}: {exc}")
                    destination = images_dir / f"{stem}{Path(filename).suffix.lower()}"
                    if args.image_mode == "copy":
                        shutil.copy2(source, destination)
                    else:
                        final_parent = output / "images" / split
                        destination.symlink_to(os.path.relpath(source, final_parent))
                    (labels_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                    counts[split]["images"] += 1
                    counts[split]["objects"] += len(rows)
            if not counts[split]["images"]:
                raise ValueError(f"no usable images in {source_split}")
        (stage / "dataset.yaml").write_text(
            "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
            + "".join(f"  {key}: {name}\n" for key, name in NAMES.items()))
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    for split in SPLITS.values():
        print(f"{split}: {dict(counts[split])}")
    if warned > 20:
        print(f"WARNING: {warned - 20} additional warnings omitted")
    print(f"dataset: {output} ({args.image_mode})")


if __name__ == "__main__":
    main()
