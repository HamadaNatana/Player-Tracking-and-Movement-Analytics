
"""Fit two-team jersey-color centers with K-Means, one fit per SoccerNetGS sequence.

Reads Labels-GameState.json and the img1 frames of each sequence, takes a torso
color for every labeled player, and fits k=2 K-Means per sequence (each match has
its own kits). The ground-truth team side ("left"/"right") is used only to score
the result, never to fit it. Saves the centers so later steps can label any player.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

LABEL_FILE = "Labels-GameState.json"
PLAYER_CATEGORY = 1  # categories: 1 player, 2 goalkeeper, 3 referee, 4 ball, 7 other
SIDES = {"left": 0, "right": 1}
TORSO_X = (0.2, 0.8)  # fraction of box width kept for the torso crop
TORSO_Y = (0.15, 0.55)  # fraction of box height kept (below head, above shorts)
GRASS_HUE = (35, 85)  # OpenCV hue range treated as grass
GRASS_MIN_SATURATION = 50
MIN_PIXELS = 20  # fewer non-grass pixels than this: use the whole crop
MIN_ROWS = 40  # fewer player colors than this in a sequence: skip it
ACCURACY_TARGET = 0.95  # PRD team-assignment target
SILHOUETTE_TARGET = 0.5  # PRD clustering target
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


def load_sequence(sequence_dir, sample_every, min_box_height, grass_filter, keep_previews):
    """Player colors from every Nth labeled frame of one sequence.

    Returns colors (N x 3), sides (0 left, 1 right, -1 unknown), track IDs, preview
    crops (aligned with the rows when keep_previews), and the unreadable-frame count.
    """
    data = json.loads((sequence_dir / LABEL_FILE).read_text())
    frame_dir = sequence_dir / data["info"].get("im_dir", "img1")
    players_by_image = defaultdict(list)  # image_id -> that frame's player annotations
    for annotation in data["annotations"]:
        if (annotation.get("supercategory") == "object"
                and annotation.get("category_id") == PLAYER_CATEGORY):
            players_by_image[annotation["image_id"]].append(annotation)
    colors, sides, tracks, previews = [], [], [], []
    unreadable = 0
    labeled = [image for image in data["images"] if image.get("is_labeled")]
    for image_info in labeled[::sample_every]:
        frame = cv2.imread(str(frame_dir / image_info["file_name"]))
        if frame is None:
            unreadable += 1
            continue
        frame_height, frame_width = frame.shape[:2]
        for annotation in players_by_image.get(image_info["image_id"], []):
            box = annotation["bbox_image"]
            x1, y1 = max(0, int(box["x"])), max(0, int(box["y"]))
            x2 = min(frame_width, int(box["x"] + box["w"]))
            y2 = min(frame_height, int(box["y"] + box["h"]))
            if y2 - y1 < min_box_height or x2 <= x1:
                continue
            crop = torso_crop(frame, (x1, y1, x2, y2))
            if crop is None:
                continue
            colors.append(crop_color(crop, grass_filter))
            side = (annotation.get("attributes") or {}).get("team")
            sides.append(SIDES.get(side, -1))
            tracks.append(annotation.get("track_id", -1))
            if keep_previews:
                previews.append(cv2.resize(crop, PREVIEW_SIZE))
    return (np.array(colors), np.array(sides, dtype=int), np.array(tracks, dtype=int),
            previews, unreadable)


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


def nearest_team(colors, centers):
    """Team 0 or 1 for each color: whichever center is closer."""
    return np.linalg.norm(colors[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)


def score_teams(predicted, sides, tracks):
    """Compare predicted teams with the ground-truth sides.

    Cluster numbers are arbitrary, so both pairings are tried and the better one
    counts. Also scores a per-track majority vote, which is what the pipeline will
    do once a tracker supplies IDs. Returns None if the sequence has no side labels.
    """
    known = sides >= 0
    if not known.any():
        return None
    agreement = float(np.mean(predicted[known] == sides[known]))
    flipped = agreement < 0.5
    aligned = 1 - predicted if flipped else predicted
    votes = defaultdict(Counter)  # track ID -> how often each team was predicted
    truth = {}
    for track, team, side in zip(tracks[known], aligned[known], sides[known]):
        votes[int(track)][int(team)] += 1
        truth[int(track)] = int(side)
    correct = sum(votes[track].most_common(1)[0][0] == truth[track] for track in votes)
    return {
        "accuracy": max(agreement, 1 - agreement),
        "track_vote_accuracy": correct / len(votes),
        "tracks": len(votes),
        "dark_team_side": "right" if flipped else "left",
    }


def scan_k(kept):
    """Normalized inertia and silhouette for each k in SCAN_KS."""
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
    return {k: value / base for k, value in inertia.items()}, silhouette


def write_preview(path, previews, centers, colors):
    """Save a strip of torso crops per team so the split can be checked by eye."""
    if not previews:
        return
    labels = nearest_team(colors, centers)
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
    parser.add_argument("--data", type=Path, default=Path("data/SoccerNetGS/gamestate-2024"),
                        help="folder holding the train, valid, and test split folders")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--max-sequences", type=int,
                        help="only use the first N sequences (quick trial run)")
    parser.add_argument("--sample-every", type=int, default=25,
                        help="use every Nth labeled frame (25 is about 1 per second)")
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
    split_dir = args.data / args.split
    if not split_dir.is_dir():
        parser.error(f"missing split folder: {split_dir}")
    if min(args.sample_every, args.min_box_height, args.preview_count) <= 0:
        parser.error("sample-every, min-box-height, and preview-count must be positive")
    if args.max_sequences is not None and args.max_sequences <= 0:
        parser.error("max-sequences must be positive")
    if not 0 <= args.trim < 0.5:
        parser.error("trim must be at least 0 and below 0.5")
    sequence_dirs = sorted(p for p in split_dir.iterdir() if (p / LABEL_FILE).is_file())
    if not sequence_dirs:
        parser.error(f"no sequence folders with {LABEL_FILE} under {split_dir}")
    if args.max_sequences:
        sequence_dirs = sequence_dirs[:args.max_sequences]

    fitted, skipped = {}, []
    inertia_by_k, silhouette_by_k = defaultdict(list), defaultdict(list)
    for index, sequence_dir in enumerate(sequence_dirs):
        sequence = sequence_dir.name
        want_preview = bool(args.preview_dir) and index < args.preview_count
        try:
            colors, sides, tracks, previews, unreadable = load_sequence(
                sequence_dir, args.sample_every, args.min_box_height,
                not args.no_grass_filter, want_preview)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append(sequence)
            print(f"WARNING: skipping {sequence}: cannot read labels ({exc})")
            continue
        if unreadable:
            print(f"WARNING: {sequence}: {unreadable} frames could not be read")
        fit = fit_teams(colors, args.trim)
        if fit is None:
            skipped.append(sequence)
            print(f"WARNING: skipping {sequence}: {len(colors)} player colors (need {MIN_ROWS}+ with variation)")
            continue
        predicted = nearest_team(colors, fit["centers"])
        score = score_teams(predicted, sides, tracks)
        sizes = np.bincount(fit["labels"], minlength=2).tolist()
        info = {
            "centers": fit["centers"].round(2).tolist(),
            "rows": len(colors),
            "trimmed": len(colors) - len(fit["kept"]),
            "cluster_sizes": sizes,
            "silhouette": round(fit["silhouette"], 4),
        }
        line = (f"{sequence}: rows={len(colors)} trimmed={info['trimmed']} sizes={sizes} "
                f"silhouette={fit['silhouette']:.3f}")
        if score:
            info.update({key: round(value, 4) if isinstance(value, float) else value
                         for key, value in score.items()})
            line += f" accuracy={score['accuracy']:.3f} track_vote={score['track_vote_accuracy']:.3f}"
            if score["accuracy"] < ACCURACY_TARGET:
                line += "  BELOW ACCURACY TARGET"
        if fit["silhouette"] < SILHOUETTE_TARGET:
            line += "  LOW SILHOUETTE"
        if min(sizes) < 0.25 * sum(sizes):
            line += "  UNBALANCED"
        print(line)
        fitted[sequence] = info
        if args.scan_k:
            inertia, silhouette = scan_k(fit["kept"])
            for k, value in inertia.items():
                inertia_by_k[k].append(value)
            for k, value in silhouette.items():
                silhouette_by_k[k].append(value)
        if want_preview:
            write_preview(args.preview_dir / f"{sequence}.png", previews, fit["centers"], colors)

    if not fitted:
        raise SystemExit("no sequence had enough player colors to fit")
    silhouettes = [info["silhouette"] for info in fitted.values()]
    print(f"fitted {len(fitted)} sequences, skipped {len(skipped)}; mean silhouette "
          f"{np.mean(silhouettes):.3f}, {sum(s < SILHOUETTE_TARGET for s in silhouettes)} below {SILHOUETTE_TARGET}")
    scored = [info for info in fitted.values() if "accuracy" in info]
    if scored:
        accuracies = [info["accuracy"] for info in scored]
        votes = [info["track_vote_accuracy"] for info in scored]
        print(f"team accuracy: mean {np.mean(accuracies):.3f}, worst {min(accuracies):.3f}, "
              f"{sum(a < ACCURACY_TARGET for a in accuracies)} of {len(scored)} below {ACCURACY_TARGET}; "
              f"track-vote mean {np.mean(votes):.3f}")
    if args.scan_k:
        print("k  mean inertia (relative to k=1)  mean silhouette")
        for k in SCAN_KS:
            if k in inertia_by_k:
                sil = f"{np.mean(silhouette_by_k[k]):.3f}" if k in silhouette_by_k else "-"
                print(f"{k}  {np.mean(inertia_by_k[k]):.3f}                          {sil}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "color_space": "OpenCV 8-bit Lab (L, a, b); team 0 is the darker center",
        "source": {"data": str(args.data), "split": args.split, "sample_every": args.sample_every,
                   "min_box_height": args.min_box_height, "trim": args.trim,
                   "grass_filter": not args.no_grass_filter},
        "sequences": fitted,
    }, indent=2) + "\n")
    print(f"centers: {args.output}")


if __name__ == "__main__":
    main()