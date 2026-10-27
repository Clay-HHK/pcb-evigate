"""Decompose detector output into five mutually exclusive error records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.detector.calibrate import box_iou, read_gt
from src.detector.dataset import rows_for_role


ERROR_NAMES = ["TP", "错分类", "FP", "漏检", "纯定位错误"]


def decompose_record(record: dict, row: dict[str, str]) -> list[dict]:
    ground_truth = [{"box": box, "class_id": class_id, "matched": False} for box, class_id in read_gt(Path(row["anno_path"]))]
    detections = sorted(
        zip(record["boxes"], record["classes"], record["scores"]), key=lambda item: (-float(item[2]), item[0])
    )
    output = []
    for box, class_id, score in detections:
        ious = [box_iou(box, gt["box"]) if not gt["matched"] else -1.0 for gt in ground_truth]
        best_index = max(range(len(ground_truth)), key=lambda index: ious[index]) if ground_truth else -1
        best_iou = ious[best_index] if best_index >= 0 else 0.0
        if best_iou >= 0.5:
            gt = ground_truth[best_index]
            gt["matched"] = True
            error_type = "TP" if class_id == gt["class_id"] else "错分类"
            gt_box, gt_class = gt["box"], gt["class_id"]
        else:
            same_class = [
                (index, box_iou(box, gt["box"]))
                for index, gt in enumerate(ground_truth)
                if not gt["matched"] and class_id == gt["class_id"]
            ]
            loc_index, loc_iou = max(same_class, key=lambda item: item[1]) if same_class else (-1, 0.0)
            if loc_iou > 0.0:
                ground_truth[loc_index]["matched"] = True
                error_type, gt_box, gt_class, best_iou = "纯定位错误", ground_truth[loc_index]["box"], class_id, loc_iou
            else:
                error_type, gt_box, gt_class = "FP", [], -1
        output.append(
            {
                "pair_id": row["pair_id"],
                "error_type": error_type,
                "pred_box": json.dumps([round(float(value), 4) for value in box], ensure_ascii=False),
                "gt_box": json.dumps([round(float(value), 4) for value in gt_box], ensure_ascii=False),
                "pred_class": int(class_id),
                "gt_class": int(gt_class),
                "score": round(float(score), 6),
                "iou": round(float(best_iou), 6),
            }
        )
    for gt in ground_truth:
        if not gt["matched"]:
            output.append(
                {
                    "pair_id": row["pair_id"],
                    "error_type": "漏检",
                    "pred_box": "[]",
                    "gt_box": json.dumps([round(float(value), 4) for value in gt["box"]], ensure_ascii=False),
                    "pred_class": -1,
                    "gt_class": int(gt["class_id"]),
                    "score": 0.0,
                    "iou": 0.0,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    prediction_path = args.predictions or Path(f"cache/detector_predictions/fold{args.fold}.jsonl")
    predictions = {record["pair_id"]: record for record in map(json.loads, prediction_path.read_text(encoding="utf-8").splitlines())}
    rows = rows_for_role(args.fold, "test")
    records = [item for row in rows for item in decompose_record(predictions[row["pair_id"]], row)]
    output = Path("experiments/stage2")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"error_decomposition_fold{args.fold}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    counts = {name: sum(item["error_type"] == name for item in records) for name in ERROR_NAMES}
    correctable = counts["错分类"] + counts["FP"] + counts["漏检"]
    markdown = output / f"error_decomposition_fold{args.fold}.md"
    markdown.write_text(
        "\n".join(
            [
                f"# fold_{args.fold} 检测错误分解",
                "",
                "| 类型 | 数量 |",
                "| --- | ---: |",
                *[f"| {name} | {counts[name]} |" for name in ERROR_NAMES],
                "",
                f"可纠错空间绝对数量：**{correctable}**（错分类、FP、漏检之和）。",
                "",
                "该数量是后续复检系统在本折上可能改善检测效用的直接上限，TP 不计入可纠错空间。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False))
    print(f"可纠错空间绝对数量: {correctable}")


if __name__ == "__main__":
    main()
