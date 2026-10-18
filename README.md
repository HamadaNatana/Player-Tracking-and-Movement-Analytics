# Soccer Player Tracking and Movement Analytics

This project is building a recorded-video computer-vision pipeline for soccer player
detection, tracking, team assignment, pitch mapping, speed, and distance analytics.
The project follows the requirements in [the PRD](docs/Soccer%20Player%20Tracking%20PRD.md).

## Progress so far

- Selected SoccerNetGS game-state data as the labeled source dataset.
- Kept the raw licensed dataset locally under `data/SoccerNetGS/`.
- Converted SoccerNetGS annotations into YOLO detection format.
- Generated train, validation, and test splits under `data/yolo/`.
- Validated 123,000 images and 2,043,056 labeled objects.
- Trained a YOLOv8n detector for:
  - `player`
  - `goalkeeper`
  - `referee`
  - `ball`
- Saved the best available checkpoint at
  `models/soccer-detect-best.pt`.

The shared checkpoint is the best model available so far. If training is still continuing,
it may be replaced by a later checkpoint after the planned 50 epochs.

## Repository layout

```text
docs/                       Requirements and project documentation
scripts/convert_to_yolo.py  Convert SoccerNetGS annotations to YOLO
scripts/validate_yolo.py    Check the generated YOLO dataset
scripts/train.py            Train or resume the YOLO detector
models/soccer-detect-best.pt Current shared trained checkpoint
data/SoccerNetGS/           Local raw labeled data; not committed
data/SoccerNet_raw_HQ/      Local recorded-video demo footage; not committed
data/yolo/                  Local generated YOLO dataset; not committed
runs/                       Local training outputs; not committed
```

Large datasets, generated labels/images, training runs, sample videos, and the
pretrained `yolov8n.pt` file remain local and are excluded from GitHub.

## Setup

From the project root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The raw SoccerNetGS dataset must be available under:

```text
data/SoccerNetGS/gamestate-2024/
```

Respect the SoccerNetGS license and redistribution terms.

## Data conversion

The source annotations are not in YOLO format. Convert them from the project root:

```sh
python scripts/convert_to_yolo.py --image-mode symlink
```

This creates:

```text
data/yolo/
  dataset.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
```

Symlinks avoid duplicating the raw images. Use `--image-mode copy` only when a portable
standalone dataset is required.

## Dataset readiness check

```sh
python scripts/validate_yolo.py --data data/yolo/dataset.yaml
```

The current generated dataset passed this check with:

- 42,750 train images
- 43,500 validation images
- 36,750 test images
- 2,043,056 total labeled objects

## Training

Start a new training run:

```sh
python scripts/train.py \
  --data data/yolo/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device cpu \
  --name soccer-detect
```

Resume an existing partial run to a total of 50 epochs:

```sh
python scripts/train.py \
  --data data/yolo/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device cpu \
  --name soccer-detect \
  --resume
```

Use `--device mps` on a supported Apple Silicon system or a CUDA device such as `0`
where available. The best checkpoint is written to:

```text
runs/detect/<run-name>/weights/best.pt
```

When a new checkpoint is ready to share, copy it to:

```sh
cp runs/detect/soccer-detect/weights/best.pt models/soccer-detect-best.pt
```

## Current limitations and next steps

YOLO detects the four object classes above; it does not identify Team 1 versus Team 2.
Team assignment is a later jersey-color clustering step. Tracking, pitch perspective
transformation, speed/distance calculations, and output video visualization are also
later PRD phases.

The raw `data/SoccerNet_raw_HQ/` videos can be used later as demo inputs. They are not
training or evaluation data for this detector because they do not include the labeled
player IDs, pitch calibration, speed ground truth, or team labels required for the full
analytics workflow.
