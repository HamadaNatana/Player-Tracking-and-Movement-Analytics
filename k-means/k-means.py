"""Fit two-team jersey-color centers with K-Means, one fit per video sequence.

Reads player boxes from the converted YOLO dataset, takes a torso color for each
player, and fits k=2 K-Means separately for every sequence (each match has its
own kits). Saves the centers so later pipeline steps can label any player.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLAYER_CLASS = "0"  # dataset.yaml: 0 player; goalkeepers (1) and referees (2) are excluded
TORSO_X = (0.2, 0.8)  # fraction of box width kept for the torso crop
TORSO_Y = (0.15, 0.55)  # fraction of box height kept (below head, above shorts)
GRASS_HUE = (35, 85)  # OpenCV hue range treated as grass
GRASS_MIN_SATURATION = 50
MIN_PIXELS = 20  # fewer non-grass pixels than this: use the whole crop
MIN_ROWS = 40  # fewer player colors than this in a sequence: skip it
SCAN_KS = range(1, 7)
PREVIEW_SIZE = (32, 48)  # width, height of each preview crop
PREVIEW_PER_TEAM = 12


def torso_crop(image, box):
    """Return the torso region of a pixel box (x1, y1, x2, y2), or None if too small."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left = max(0, int(round(x1 + TORSO_X[0] * width)))
    right = min(image.shape[1], int(round(x1 + TORSO_X[1] * width)))
    top = max(0, int(round(y1 + TORSO_Y[0] * height)))
    bottom = min(image.shape[0], int(round(y1 + TORSO_Y[1] * height)))
    if right - left < 4 or bottom - top < 4:
        return None
    return image[top:bottom, left:right]


def crop_color(crop, grass_filter=True):
    """Median Lab color of a crop (OpenCV 8-bit Lab), ignoring grass-colored pixels."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    if grass_filter:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        grass = ((hsv[:, 0] >= GRASS_HUE[0]) & (hsv[:, 0] <= GRASS_HUE[1])
                 & (hsv[:, 1] >= GRASS_MIN_SATURATION))
        if (~grass).sum() >= MIN_PIXELS:
            lab = lab[~grass]
    return np.median(lab, axis=0)


def read_player_boxes(label_file, width, height, min_box_height):
    """Pixel boxes for player rows (class 0) in a YOLO label file."""
    boxes = []
    if not label_file.is_file():
        return boxes
    for row in label_file.read_text().splitlines():
        fields = row.split()
        if len(fields) != 5 or fields[0] != PLAYER_CLASS:
            continue
        x, y, w, h = (float(v) for v in fields[1:])
        if h * height < min_box_height:
            continue
        boxes.append((int((x - w / 2) * width), int((y - h / 2) * height),
                      int((x + w / 2) * width), int((y + h / 2) * height)))
    return boxes


def collect_colors(data_dir, split, sample_every, min_box_height, grass_filter, preview_sequences):
    """Return {sequence: (colors, preview_crops)} from every Nth frame of each sequence."""
    image_dir = data_dir / "images" / split
    label_dir = data_dir / "labels" / split
    by_sequence = defaultdict(list)
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in EXTENSIONS:
            # convert_to_yolo.py names files <sequence>_<image_id>_<frame>
            by_sequence[path.name.split("_", 1)[0]].append(path)
    result = {}
    unreadable = 0
    for sequence, paths in by_sequence.items():
        colors, previews = [], []
        for path in paths[::sample_every]:
            image = cv2.imread(str(path))
            if image is None:
                unreadable += 1
                continue
            height, width = image.shape[:2]
            for box in read_player_boxes(label_dir / f"{path.stem}.txt", width, height, min_box_height):
                crop = torso_crop(image, box)
                if crop is None:
                    continue
                colors.append(crop_color(crop, grass_filter))
                if sequence in preview_sequences:
                    previews.append(cv2.resize(crop, PREVIEW_SIZE))
        result[sequence] = (np.array(colors), previews)
    if unreadable:
        print(f"WARNING: {unreadable} images could not be read (broken symlinks?)")
    return result


def fit_teams(colors, trim):
    """Fit k=2 K-Means, drop the farthest `trim` fraction of rows, refit.

    Returns None if the sequence has too few or too uniform colors. Team 0 is the
    darker center (lower L) so labels stay the same from run to run.
    """
    if len(colors) < MIN_ROWS or len(np.unique(colors, axis=0)) < 2:
        return None
    model = KMeans(n_clusters=2, n_init=10, random_state=0).fit(colors)
    kept = colors
    if trim > 0:
        distance = np.linalg.norm(colors - model.cluster_centers_[model.labels_], axis=1)
        mask = distance <= np.quantile(distance, 1 - trim)
        if mask.sum() >= MIN_ROWS:
            kept = colors[mask]
            model = KMeans(n_clusters=2, init=model.cluster_centers_, n_init=1).fit(kept)
    if len(set(model.labels_)) < 2:
        return None
    order = np.argsort(model.cluster_centers_[:, 0])
    labels = np.argsort(order)[model.labels_]
    return {
        "centers": model.cluster_centers_[order],
        "kept": kept,
        "labels": labels,
        "silhouette": float(silhouette_score(kept, labels)),
    }


def scan_k(kept):
    """Normalized inertia and silhouette for each k in SCAN_KS (None where undefined)."""
    inertia, silhouette = {}, {}
    for k in SCAN_KS:
        if len(kept) <= k:
            continue
        model = KMeans(n_clusters=k, n_init=5, random_state=0).fit(kept)
        inertia[k] = model.inertia_
        if k >= 2:
            silhouette[k] = float(silhouette_score(kept, model.labels_))
    base = inertia.get(1)
    if not base:
        return {}, {}
    return {k: v / base for k, v in inertia.items()}, silhouette


def write_preview(path, previews, centers, colors):
    """Save a strip of torso crops per team so the split can be checked by eye."""
    if not previews:
        return
    # Rows in `previews` line up with `colors`; label each by its nearest center.
    labels = np.linalg.norm(colors[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
    width, height = PREVIEW_SIZE
    canvas = np.zeros((2 * height, PREVIEW_PER_TEAM * width, 3), dtype=np.uint8)
    for team in (0, 1):
        members = [previews[i] for i in np.flatnonzero(labels == team)]
        step = max(1, len(members) // PREVIEW_PER_TEAM)
        for slot, crop in enumerate(members[::step][:PREVIEW_PER_TEAM]):
            canvas[team * height:(team + 1) * height, slot * width:(slot + 1) * width] = crop
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


class TeamAssigner:
    """Label a player's jersey color as team 0 or 1 using a saved per-sequence fit."""

    def __init__(self, sequences):
        self.centers = {name: np.array(info["centers"]) for name, info in sequences.items()}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["sequences"])

    def predict_color(self, color, sequence):
        return int(np.linalg.norm(self.centers[sequence] - np.asarray(color), axis=1).argmin())

    def predict_crop(self, image, box, sequence, grass_filter=True):
        crop = torso_crop(image, box)
        if crop is None:
            return None
        return self.predict_color(crop_color(crop, grass_filter), sequence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/yolo"),
                        help="YOLO dataset directory (or its dataset.yaml)")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-every", type=int, default=25,
                        help="use every Nth frame of each sequence (25 is about 1 per second)")
    parser.add_argument("--min-box-height", type=int, default=60,
                        help="skip player boxes shorter than this many pixels")
    parser.add_argument("--trim", type=float, default=0.05,
                        help="fraction of farthest-from-center rows dropped before the final fit")
    parser.add_argument("--no-grass-filter", action="store_true",
                        help="keep grass-colored pixels (use if a team wears green)")
    parser.add_argument("--output", type=Path, default=Path("models/team-centers.json"))
    parser.add_argument("--scan-k", action="store_true",
                        help="print inertia and silhouette for k=1..6 to confirm k=2")
    parser.add_argument("--preview-dir", type=Path,
                        help="write a crop strip per team for a few sequences")
    parser.add_argument("--preview-count", type=int, default=5)
    args = parser.parse_args()
    data_dir = args.data.parent if args.data.is_file() else args.data
    if not (data_dir / "images" / args.split).is_dir() or not (data_dir / "labels" / args.split).is_dir():
        parser.error(f"missing images or labels for split {args.split} under {data_dir}")
    if min(args.sample_every, args.min_box_height, args.preview_count) <= 0:
        parser.error("sample-every, min-box-height, and preview-count must be positive")
    if not 0 <= args.trim < 0.5:
        parser.error("trim must be at least 0 and below 0.5")

    sequences_on_disk = sorted({p.name.split("_", 1)[0] for p in (data_dir / "images" / args.split).iterdir()
                                if p.suffix.lower() in EXTENSIONS})
    preview_sequences = set(sequences_on_disk[:args.preview_count]) if args.preview_dir else set()
    collected = collect_colors(data_dir, args.split, args.sample_every, args.min_box_height,
                               not args.no_grass_filter, preview_sequences)

    fitted, skipped = {}, []
    inertia_by_k, silhouette_by_k = defaultdict(list), defaultdict(list)
    for sequence, (colors, previews) in sorted(collected.items()):
        fit = fit_teams(colors, args.trim)
        if fit is None:
            skipped.append(sequence)
            print(f"WARNING: skipping {sequence}: {len(colors)} player colors (need {MIN_ROWS}+ with variation)")
            continue
        sizes = np.bincount(fit["labels"], minlength=2).tolist()
        fitted[sequence] = {
            "centers": fit["centers"].round(2).tolist(),
            "rows": len(colors),
            "trimmed": len(colors) - len(fit["kept"]),
            "cluster_sizes": sizes,
            "silhouette": round(fit["silhouette"], 4),
        }
        note = ""
        if fit["silhouette"] < 0.5:
            note += "  LOW SILHOUETTE"
        if min(sizes) < 0.25 * sum(sizes):
            note += "  UNBALANCED"
        print(f"{sequence}: rows={len(colors)} trimmed={fitted[sequence]['trimmed']} "
              f"sizes={sizes} silhouette={fit['silhouette']:.3f}{note}")
        if args.scan_k:
            inertia, silhouette = scan_k(fit["kept"])
            for k, value in inertia.items():
                inertia_by_k[k].append(value)
            for k, value in silhouette.items():
                silhouette_by_k[k].append(value)
        if args.preview_dir and sequence in preview_sequences:
            write_preview(args.preview_dir / f"{sequence}.png", previews, fit["centers"], colors)

    if not fitted:
        raise SystemExit("no sequence had enough player colors to fit")
    scores = [info["silhouette"] for info in fitted.values()]
    print(f"fitted {len(fitted)} sequences, skipped {len(skipped)}; mean silhouette {np.mean(scores):.3f}, "
          f"{sum(s < 0.5 for s in scores)} below 0.5")
    if args.scan_k:
        print("k  mean inertia (relative to k=1)  mean silhouette")
        for k in SCAN_KS:
            if k in inertia_by_k:
                sil = f"{np.mean(silhouette_by_k[k]):.3f}" if k in silhouette_by_k else "-"
                print(f"{k}  {np.mean(inertia_by_k[k]):.3f}                          {sil}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "color_space": "OpenCV 8-bit Lab (L, a, b); team 0 is the darker center",
        "source": {"data": str(data_dir), "split": args.split, "sample_every": args.sample_every,
                   "min_box_height": args.min_box_height, "trim": args.trim,
                   "grass_filter": not args.no_grass_filter},
        "sequences": fitted,
    }, indent=2) + "\n")
    print(f"centers: {args.output}")


if __name__ == "__main__":
    main()