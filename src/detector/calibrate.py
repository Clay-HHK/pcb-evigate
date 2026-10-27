"""Fit one temperature on matched validation detections."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from src.detector.dataset import build_fold_dataset, read_manifest, rows_for_role
from src.detector.predict import predict_records


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def read_gt(annotation_path: Path) -> list[tuple[list[float], int]]:
    result = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        x1, y1, x2, y2, class_id = line.split()
        result.append(([float(x1), float(y1), float(x2), float(y2)], int(class_id) - 1))
    return result


def calibration_samples(records: list[dict], rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    logits, targets = [], []
    for record, row in zip(records, rows):
        ground_truth = read_gt(Path(row["anno_path"]))
        for box, class_id, score in zip(record["boxes"], record["classes"], record["scores"]):
            best_iou = 0.0
            best_class = -1
            for gt_box, gt_class in ground_truth:
                current_iou = box_iou(box, gt_box)
                if current_iou > best_iou:
                    best_iou, best_class = current_iou, gt_class
            target = float(best_iou >= 0.5 and class_id == best_class)
            probability = min(max(float(score), 1e-6), 1.0 - 1e-6)
            logits.append(math.log(probability / (1.0 - probability)))
            targets.append(target)
    return logits, targets


def fit_temperature(logits: list[float], targets: list[float]) -> float:
    if not logits:
        return 1.0
    values = torch.tensor(logits, dtype=torch.float64)
    labels = torch.tensor(targets, dtype=torch.float64)
    log_temperature = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(values / torch.exp(log_temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    build_fold_dataset(args.fold)
    weights = args.weights or Path(f"cache/detector_runs/fold{args.fold}/weights/best.pt")
    model = YOLO(str(weights))
    rows = rows_for_role(args.fold, "val")
    records = predict_records(model, rows)
    logits, targets = calibration_samples(records, rows)
    temperature = fit_temperature(logits, targets)
    output = Path("cache/detector_predictions")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"temperature_fold{args.fold}.json"
    path.write_text(json.dumps({"fold": args.fold, "temperature": temperature, "n_samples": len(logits)}, indent=2) + "\n", encoding="utf-8")
    print(f"{path}: T={temperature:.6f}, n={len(logits)}")


if __name__ == "__main__":
    main()
