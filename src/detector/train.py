"""Train one YOLOv8s detector on one GroupKFold fold."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.detector.dataset import SEED, build_fold_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/detector/yolov8s.yaml"))
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    data_yaml = build_fold_dataset(args.fold)
    model = YOLO(config["model"])
    train_config = {
        "data": str(data_yaml),
        "epochs": 2 if args.smoke else config["epochs"],
        "imgsz": 640 if args.smoke else config["imgsz"],
        "batch": -1 if args.smoke else (args.batch if args.batch is not None else config["batch"]),
        "device": config["device"],
        "workers": config["workers"],
        "patience": config["patience"],
        "project": str(Path(config["project"]).resolve()),
        "name": f"fold{args.fold}",
        "seed": SEED,
        "deterministic": True,
        "pretrained": True,
        "exist_ok": True,
        "verbose": True,
    }
    model.train(**train_config)


if __name__ == "__main__":
    main()
