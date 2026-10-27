"""Write low-confidence and validation-selected detector predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from src.detector.dataset import build_fold_dataset, rows_for_role


LOW_CONFIDENCE = 0.001


def predict_records(model: YOLO, rows: list[dict[str, str]], conf: float = LOW_CONFIDENCE) -> list[dict]:
    records = []
    for row in rows:
        result = model.predict(row["test_path"], conf=conf, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        scores = result.boxes.conf.cpu().numpy().tolist()
        records.append({"pair_id": row["pair_id"], "boxes": boxes, "classes": classes, "scores": scores})
    return records


def read_ground_truth(annotation_path: Path) -> list[tuple[list[float], int]]:
    ground_truth = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        x1, y1, x2, y2, class_id = line.split()
        ground_truth.append(([float(x1), float(y1), float(x2), float(y2)], int(class_id) - 1))
    return ground_truth


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def select_working_confidence(records: list[dict], rows: list[dict[str, str]]) -> tuple[float, float]:
    detections = []
    total_ground_truth = 0
    for record, row in zip(records, rows):
        ground_truth = read_ground_truth(Path(row["anno_path"]))
        total_ground_truth += len(ground_truth)
        for box, class_id, score in zip(record["boxes"], record["classes"], record["scores"]):
            detections.append((float(score), row["pair_id"], box, int(class_id), ground_truth))
    detections.sort(key=lambda item: -item[0])
    matched: dict[str, set[int]] = {}
    true_positives = 0
    false_positives = 0
    best_f1 = -1.0
    best_threshold = 1.0
    for index, (score, pair_id, box, class_id, ground_truth) in enumerate(detections):
        used = matched.setdefault(pair_id, set())
        candidates = [
            (gt_index, box_iou(box, gt_box))
            for gt_index, (gt_box, gt_class) in enumerate(ground_truth)
            if gt_index not in used and gt_class == class_id
        ]
        gt_index, iou = max(candidates, key=lambda item: item[1]) if candidates else (-1, 0.0)
        if iou >= 0.5:
            used.add(gt_index)
            true_positives += 1
        else:
            false_positives += 1
        next_score = detections[index + 1][0] if index + 1 < len(detections) else -1.0
        if next_score == score:
            continue
        false_negatives = total_ground_truth - true_positives
        denominator = 2 * true_positives + false_positives + false_negatives
        f1 = 2 * true_positives / denominator if denominator else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = score
    return best_threshold, best_f1


def filter_records(records: list[dict], threshold: float) -> list[dict]:
    filtered = []
    for record in records:
        keep = [index for index, score in enumerate(record["scores"]) if float(score) >= threshold]
        filtered.append(
            {
                "pair_id": record["pair_id"],
                "boxes": [record["boxes"][index] for index in keep],
                "classes": [record["classes"][index] for index in keep],
                "scores": [record["scores"][index] for index in keep],
            }
        )
    return filtered


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--conf", type=float, default=LOW_CONFIDENCE)
    args = parser.parse_args()
    build_fold_dataset(args.fold)
    weights = args.weights or Path(f"cache/detector_runs/fold{args.fold}/weights/best.pt")
    model = YOLO(str(weights))
    validation_rows = rows_for_role(args.fold, "val")
    validation_records = predict_records(model, validation_rows, args.conf)
    threshold, validation_f1 = select_working_confidence(validation_records, validation_rows)
    target_records = validation_records if args.split == "val" else predict_records(
        model, rows_for_role(args.fold, args.split), args.conf
    )
    output = Path("cache/detector_predictions")
    output.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "test" else f"_{args.split}"
    low_confidence_path = output / f"fold{args.fold}{suffix}_lowconf.jsonl"
    working_path = output / f"fold{args.fold}{suffix}.jsonl"
    threshold_path = output / f"working_point_fold{args.fold}.json"
    write_jsonl(low_confidence_path, target_records)
    write_jsonl(working_path, filter_records(target_records, threshold))
    threshold_path.write_text(
        json.dumps(
            {
                "fold": args.fold,
                "selection_split": "val",
                "iou": 0.5,
                "low_confidence": args.conf,
                "working_confidence": threshold,
                "validation_f1": validation_f1,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"低阈值预测: {low_confidence_path}, 图像数 {len(target_records)}")
    print(f"工作点预测: {working_path}, conf={threshold:.6f}, val F1={validation_f1:.6f}")


if __name__ == "__main__":
    main()
