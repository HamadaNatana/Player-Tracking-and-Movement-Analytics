#!/usr/bin/env python3
"""Train one SoccerNetGS YOLO detection model."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/yolo/dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.pt",
                        help="official pretrained YOLO detection weights, or a local .pt file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu, mps, or a CUDA device such as 0")
    parser.add_argument("--name", default="soccer-detect")
    parser.add_argument("--resume", action="store_true",
                        help="resume the existing run from its last.pt checkpoint")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"dataset YAML missing: {args.data}")
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("epochs, imgsz, and batch must be positive")
    if not args.model.endswith(".pt"):
        parser.error("model must be a pretrained detection .pt checkpoint")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be one directory name")
    run = Path("runs/detect") / args.name
    checkpoint = run / "weights/last.pt"
    if run.exists() and not args.resume:
        parser.error(f"run already exists: {run}; choose another --name")
    if args.resume and not checkpoint.is_file():
        parser.error(f"resume checkpoint missing: {checkpoint}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"ultralytics is required: {exc}") from exc
    model = YOLO(str(checkpoint) if args.resume else args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(run.parent.resolve()),
        name=run.name,
        exist_ok=True,
        workers=0,
        plots=False,
        resume=args.resume,
    )
    best_checkpoint = run / "weights/best.pt"
    if not best_checkpoint.is_file() or best_checkpoint.stat().st_size == 0:
        raise SystemExit(f"training did not create best.pt: {best_checkpoint.resolve()}")
    print(f"checkpoint: {best_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
