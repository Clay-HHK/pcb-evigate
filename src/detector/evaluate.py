"""Evaluate one trained detector on its held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    fold = args.fold
    model = YOLO(f"cache/detector_runs/fold{fold}/weights/best.pt")
    metrics = model.val(
        data=f"cache/detector_dataset/fold{fold}/data.yaml",
        split="test",
        conf=0.001,
        iou=0.7,
        imgsz=640,
        batch=16,
        device=0,
        project=str(Path("cache/detector_eval").resolve()),
        name=f"fold{fold}",
        exist_ok=True,
        verbose=True,
    )
    output = Path("experiments/stage2")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"metrics_fold{fold}.json").write_text(
        json.dumps(
            {"fold": fold, "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
